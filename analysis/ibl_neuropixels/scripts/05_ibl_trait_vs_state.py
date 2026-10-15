"""In vivo firing statistics of motor-cortex units (IBL Brain-Wide Map, DANDI:000409).

The in vivo analogue of 'type vs state' is 'stable unit trait vs fluctuating state'.
For good single units in MOp / MOs (other isocortex as reference) we compute firing
statistics in independent time blocks and ask:
  1. Reliability: how much between-unit variance is stable across interleaved blocks,
     across the two halves of the recording (slow drift), and across behavioural
     context (task vs passive/spontaneous epochs)? (ICC(1), Spearman-Brown-free.)
  2. How much of the stable between-unit variance do coarse 'type' proxies explain
     (layer x waveform class), and how much do session/animal (shared state) explain?
Inputs: data/ibl_motor_units.parquet, data/raw/<session>.npz (from ibl_extract_units.py)
Outputs: results/aim2_*.csv

Usage:  uv run python analysis/ibl_neuropixels/scripts/05_ibl_trait_vs_state.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from concurrent.futures import ProcessPoolExecutor
RESULTS = paths.results(__file__)
DATA = paths.data(__file__)

IBL = DATA
units = pd.read_parquet(IBL / "ibl_motor_units.parquet")
units = units[(units["good"] == 1) & units["spike_width_class"].isin(["wide", "narrow"])].copy()
units["cls"] = units["spike_width_class"].map({"wide": "RS", "narrow": "FS"})
units["layer"] = units["layer"].astype(str)
N_BLOCKS = 20          # equal-duration blocks across the whole recording


def block_stats(st, t0, t1):
    s = st[(st >= t0) & (st < t1)]
    d = {"rate": len(s) / (t1 - t0)}
    if len(s) >= 10:
        isi = np.diff(s)
        a, b = isi[:-1], isi[1:]
        d["lv"] = 3 * np.mean(((a - b) / (a + b)) ** 2)
        d["burst"] = np.mean(isi < 0.010)
    else:
        d["lv"] = d["burst"] = np.nan
    return d


def per_session(sid):
    z = np.load(IBL / "raw" / f"{sid}.npz", allow_pickle=True)
    u = units[units["session_id"] == sid]
    idx = {int(k): i for i, k in enumerate(z["unit_index"])}
    off, st_all = z["offsets"], z["spike_times"]
    rows = []
    for _, r in u.iterrows():
        if int(r["unit_index"]) not in idx:
            continue
        i = idx[int(r["unit_index"])]
        st = st_all[off[i]:off[i + 1]]
        if len(st) < 200:
            continue
        t0, t1 = st_all.min(), st_all.max()     # recording span from all cached units
        edges = np.linspace(t0, t1, N_BLOCKS + 1)
        for b in range(N_BLOCKS):
            d = block_stats(st, edges[b], edges[b + 1])
            d.update(uid=f"{sid}_{r['probe_name']}_{r['unit_index']}", block=b)
            rows.append(d)
    return rows


def icc1(df, col):
    """One-way random-effects ICC across 2 'raters' (columns a,b) per unit."""
    x = df[[f"{col}_a", f"{col}_b"]].dropna().values
    n, k = x.shape
    if n < 10:
        return np.nan, n
    gm = x.mean()
    msb = k * np.sum((x.mean(1) - gm) ** 2) / (n - 1)
    msw = np.sum((x - x.mean(1, keepdims=True)) ** 2) / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw), n


if __name__ == "__main__":
    sessions = units["session_id"].unique()
    with ProcessPoolExecutor(16) as ex:
        rows = [r for rs in ex.map(per_session, sessions) for r in rs]
    B = pd.DataFrame(rows)
    B["log_rate"] = np.log10(B["rate"] + 0.05)
    units["uid"] = units["session_id"] + "_" + units["probe_name"] + "_" + units["unit_index"].astype(str)
    meta = units.set_index("uid")[["session_id", "subject", "lab", "region_group", "layer", "cls", "area"]]
    B = B.join(meta, on="uid")
    B.to_parquet(RESULTS / "aim2_unit_blocks.parquet")

    stats_cols = ["log_rate", "lv", "burst"]
    splits = {"interleaved (odd vs even blocks)": (B["block"] % 2 == 0, B["block"] % 2 == 1),
              "first vs second half": (B["block"] < N_BLOCKS // 2, B["block"] >= N_BLOCKS // 2)}
    rel = []
    for name, (ma, mb) in splits.items():
        A = B[ma].groupby("uid")[stats_cols].mean().add_suffix("_a")
        Bb = B[mb].groupby("uid")[stats_cols].mean().add_suffix("_b")
        J = A.join(Bb).join(meta)
        for (rg, cl), g in list(J.groupby(["region_group", "cls"])) + [((rg, "all"), g) for rg, g in J.groupby("region_group")]:
            for c in stats_cols:
                v, n = icc1(g, c)
                rel.append({"split": name, "region": rg, "cls": cl, "stat": c, "icc": v, "n_units": n})
    # behavioural context: task vs passive/spontaneous epoch, from the unit table
    ctx = units.set_index("uid")
    ctx = ctx.assign(log_rate_a=np.log10(ctx["taskepoch_rate_hz"] + 0.05), log_rate_b=np.log10(ctx["spont_rate_hz"] + 0.05),
                     lv_a=ctx["taskepoch_lv"], lv_b=ctx["spont_lv"],
                     burst_a=ctx["taskepoch_frac_isi_lt10ms"], burst_b=ctx["spont_frac_isi_lt10ms"])
    ctx = ctx[(ctx["taskepoch_dur_s"] > 300) & (ctx["spont_dur_s"] > 120)]
    for (rg, cl), g in ctx.groupby(["region_group", "cls"]):
        for c in stats_cols:
            v, n = icc1(g, c)
            rel.append({"split": "task vs passive/spontaneous", "region": rg, "cls": cl, "stat": c, "icc": v, "n_units": n})
    rel = pd.DataFrame(rel)
    rel.to_csv(RESULTS / "aim2_reliability.csv", index=False)

    # variance partition of the unit-mean statistics (stable part) in MOp/MOs
    U = B.groupby("uid")[stats_cols].mean().join(meta)
    U.to_csv(RESULTS / "aim2_unit_means.csv")
    part = []
    for rg in ["MOp", "MOs", "other_isocortex"]:
        g = U[U["region_group"] == rg].dropna(subset=stats_cols)
        g = g[g["layer"].isin(["2/3", "4", "5", "6a", "6b", "1"])]
        for c in stats_cols:
            y = (g[c] - g[c].mean()) / g[c].std()
            d = g.assign(y=y)
            m_type = smf.ols("y ~ C(layer) * C(cls)", d).fit()
            m_full = smf.ols("y ~ C(layer) * C(cls) + C(session_id)", d).fit()
            m_sess = smf.ols("y ~ C(session_id)", d).fit()
            part.append({"region": rg, "stat": c, "n_units": len(d), "n_sessions": d["session_id"].nunique(),
                         "n_subjects": d["subject"].nunique(),
                         "adjR2_layer_x_class": m_type.rsquared_adj,
                         "adjR2_session_only": m_sess.rsquared_adj,
                         "adjR2_layer_x_class_plus_session": m_full.rsquared_adj})
    pd.DataFrame(part).to_csv(RESULTS / "aim2_variance_partition.csv", index=False)

    # descriptive distributions by region / layer / class
    desc = U.groupby(["region_group", "layer", "cls"])[stats_cols].agg(["median", "count"])
    desc.to_csv(RESULTS / "aim2_descriptive.csv")
    print(rel.pivot_table(index=["region", "cls", "stat"], columns="split", values="icc").round(3).to_string())
    print(pd.DataFrame(part).round(3).to_string())
