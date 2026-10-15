"""SFARI enrichment among within-type excitability genes: M1 (discovery) + V1 (test).

Gene-level within-type screens (q < 0.05) against ephys features, then SFARI Gene
enrichment of the hit genes. M1 is the DISCOVERY cohort (characterisation only);
mouse V1 L5 ET (n=436) is the test cohort. V1 is a single transcriptomic
subclass, so its estimate is inherently within-type and cannot reflect a
between-type pooling effect.

Exclusions: (a) channel/receptor/synaptic genes; (b) genes whose only hit is the
membrane time constant (a cell-size axis). The primary condition applies both.
If fewer than 20 V1 genes survive the exclusions the enrichment test is reported
as uninformative.

The primary test is the EXPRESSION-MATCHED null, not Fisher exact. SFARI genes
are long and broadly expressed, so they are over-represented among any set of
well-measured genes; an unmatched test is biased upward by construction. The
null matches on joint mean-expression x detection deciles; Fisher exact is also
reported for comparison. Gene list: SFARI Gene (ref/sfari_genes.csv).

Usage:
  uv run python analysis/cross_dataset/scripts/38_asd_m1_v1.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)

import re
from importlib import import_module

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

_m3 = import_module("03_gene_level_rheobase")
_m4 = import_module("04_ephys_wide_screen")
_m28 = import_module("28_v1_confirmatory")
from plasticity_common import PRIMARY_EPHYS, load_ephys, load_meta  # noqa: E402

ROOT = paths.REPO
V1D = ROOT / "ReplicationData" / "MouseV1_Sorensen2026"
OUT = paths.results(__file__)
SFARI = ROOT / "ref" / "sfari_genes.csv"
N_MATCH = 10000
N_BINS = 10
RNG = np.random.default_rng(0)
CH = "|".join(_m3.CHANNEL_PATTERNS)

# V1 ephys features screened
V1_FLAT = ["input_resistance", "tau", "v_baseline", "sag_nearest_minus_100",
           "rheobase_i", "fi_linear_fit_slope", "ap_1_threshold_v_short_square",
           "ap_1_width_short_square",
           "ap_1_upstroke_downstroke_ratio_short_square",
           "ap_1_peak_v_short_square", "short_square_current"]
V1_STEP = {"adapt": "mean", "isi_cv": "mean", "latency": "mean",
           "avg_rate": "max"}
TAU_FEATURES = {"Membrane time constant (ms)", "tau"}


def load_sfari():
    s = pd.read_csv(SFARI)
    s = s[s["gene-score"].notna()]
    allg = set(s["gene-symbol"].astype(str).str.upper())
    tier1 = set(s.loc[s["gene-score"] == 1, "gene-symbol"].astype(str).str.upper())
    return allg, tier1


def matched_null(hits, background, meanx, detect, target, n_draw=N_MATCH):
    """Draw sets the same size as `hits`, matched to it on expression decile.

    Returns (observed overlap, null mean, null sd, one-sided p, z).
    """
    bg = np.array(sorted(background))
    mx = np.array([meanx[g] for g in bg])
    dt = np.array([detect[g] for g in bg])
    # joint decile bin on mean expression x detection
    bm = np.clip(np.digitize(mx, np.quantile(mx, np.linspace(0, 1, N_BINS + 1)[1:-1])), 0, N_BINS - 1)
    bd = np.clip(np.digitize(dt, np.quantile(dt, np.linspace(0, 1, N_BINS + 1)[1:-1])), 0, N_BINS - 1)
    key = bm * N_BINS + bd
    pools = {k: bg[key == k] for k in np.unique(key)}

    hit_arr = np.array(sorted(set(hits) & set(bg)))
    if len(hit_arr) == 0:
        return 0, np.nan, np.nan, np.nan, np.nan
    hidx = {g: i for i, g in enumerate(bg)}
    hkeys = np.array([key[hidx[g]] for g in hit_arr])

    obs = int(sum(1 for g in hit_arr if g.upper() in target))
    nulls = np.empty(n_draw)
    for i in range(n_draw):
        c = 0
        for k in hkeys:
            pool = pools[k]
            c += 1 if RNG.choice(pool).upper() in target else 0
        nulls[i] = c
    mu, sd = float(nulls.mean()), float(nulls.std(ddof=1))
    p = float((nulls >= obs).mean())
    z = (obs - mu) / sd if sd > 0 else np.nan
    return obs, mu, sd, p, z


def enrich(hits, background, meanx, detect, target, label, rows, cohort, tier):
    bg = set(background)
    hits = set(hits) & bg
    if len(hits) < 5:
        print(f"  {label:44s} only {len(hits)} hits -- uninformative")
        return
    obs, mu, sd, p_m, z = matched_null(hits, bg, meanx, detect, target)
    tgt_bg = sum(1 for g in bg if g.upper() in target)
    exp_f = len(hits) * tgt_bg / len(bg)
    a, b = obs, len(hits) - obs
    c, d = tgt_bg - obs, len(bg) - len(hits) - (tgt_bg - obs)
    orr, p_f = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
    print(f"  {label:44s} {len(hits):4d} hits  obs {obs:3d}  "
          f"exp(unmatched) {exp_f:5.1f}  OR {orr:5.2f}  Fisher p {p_f:8.5f}   "
          f"| matched null {mu:5.1f}+-{sd:4.1f}  z {z:+5.2f}  p {p_m:.4f}")
    rows.append({"cohort": cohort, "tier": tier, "condition": label,
                 "n_hits": len(hits), "n_bg": len(bg), "obs": obs,
                 "exp_unmatched": exp_f, "OR": orr, "fisher_p": p_f,
                 "matched_mean": mu, "matched_sd": sd, "matched_z": z,
                 "matched_p": p_m})


def screen_cohort(expr, ephys_df, features, cov, label):
    """Genome-wide within-type screen; returns hit table (gene, feature, rho, q)."""
    hits = []
    for feat in features:
        y = ephys_df[feat]
        ok = y.notna() & cov.notna().all(axis=1)
        if ok.sum() < 60:
            continue
        r = _m3.screen(expr.loc[ok], y.loc[ok], cov.loc[ok])
        sig = r[r["q"] < 0.05]
        for _, row in sig.iterrows():
            hits.append({"gene": row.gene, "feature": feat, "rho": row.rho,
                         "q": row.q})
        print(f"    {label} {feat:46s} n={int(ok.sum()):4d}  hits={len(sig)}")
    return pd.DataFrame(hits)


def apply_exclusions(H, expr_cols):
    """(a) drop channels/receptors/synaptic; (b) drop genes whose ONLY hit is tau."""
    keep_a = H[~H["gene"].str.match(CH, na=False)]
    per_gene = keep_a.groupby("gene")["feature"].apply(set)
    only_tau = {g for g, fs in per_gene.items() if fs <= TAU_FEATURES}
    keep_b = keep_a[~keep_a["gene"].isin(only_tau)]
    return keep_a, keep_b


def main() -> None:
    allg, tier1 = load_sfari()
    print(f"[sfari] {len(allg)} scored genes, {len(tier1)} at score 1")
    rows = []

    # ================= mouse M1 -- DISCOVERY, characterisation only =========
    print("\n" + "=" * 108)
    print("MOUSE M1 (DISCOVERY COHORT -- characterisation, not evidence)")
    print("=" * 108)
    meta = load_meta(); ephys = load_ephys()
    lc, qc = _m3.load_full_logcpm(meta.index, use_introns=True)
    expr, _ = _m3.filter_genes(lc, drop_channels=False)
    d0 = meta.loc[expr.index].join(qc, how="left")
    glut = d0.index[d0["RNA family"].isin(["IT", "ET", "CT", "NP"])]
    expr_m = expr.loc[glut]; d = d0.loc[glut]
    s = d["RNA type"].astype(str)
    k = s.value_counts()[lambda x: x >= 8].index
    s = s.where(s.isin(k), "other")
    cov = pd.concat([d[_m3.COVARS],
                     pd.get_dummies(s, prefix="t", drop_first=True).astype(float)],
                    axis=1)
    feats = [f for f in PRIMARY_EPHYS + _m4.EXTRA_EPHYS if f in ephys.columns]
    print(f"  {len(glut)} glutamatergic cells, {expr_m.shape[1]} genes, "
          f"{len(feats)} features")
    Hm = screen_cohort(expr_m, ephys.reindex(glut), feats, cov, "M1")
    Hm.to_csv(OUT / "AK_asd_m1_hits.csv", index=False)
    bg_m = list(expr_m.columns)
    mean_m = expr_m.mean().to_dict(); det_m = (expr_m > 0).mean().to_dict()
    a_m, b_m = apply_exclusions(Hm, bg_m)
    print(f"\n  M1 hit genes: all {Hm.gene.nunique()}, "
          f"after channel filter {a_m.gene.nunique()}, "
          f"after +tau filter {b_m.gene.nunique()}")
    for tier, tg in [("all(1-3)", allg), ("score1", tier1)]:
        print(f"\n  --- SFARI tier {tier} ---")
        enrich(Hm.gene.unique(), bg_m, mean_m, det_m, tg,
               "M1 all hits", rows, "mouse M1 (discovery)", tier)
        enrich(a_m.gene.unique(), bg_m, mean_m, det_m, tg,
               "M1 minus channels/receptors/synaptic", rows,
               "mouse M1 (discovery)", tier)
        enrich(b_m.gene.unique(), bg_m, mean_m, det_m, tg,
               "M1 minus channels AND tau-only", rows,
               "mouse M1 (discovery)", tier)

    # ================= mouse V1 -- THE TEST =================================
    print("\n" + "=" * 108)
    print("MOUSE V1 L5 ET -- TEST COHORT")
    print("=" * 108)
    lcv, cnt, q2 = _m28.load_v1()
    lcv.index = lcv.index.astype(str); q2.index = q2.index.astype(str)
    exprv, audit = _m3.filter_genes(lcv, drop_channels=False)
    ph = pd.read_csv(V1D / "exc_mMET_ephys_features.csv")
    ph["specimen_id"] = ph["specimen_id"].astype(str); ph = ph.set_index("specimen_id")
    mt0 = pd.read_csv(V1D / "inferred_met_types.csv").rename(
        columns={"Unnamed: 0": "specimen_id"})
    mt0["specimen_id"] = mt0["specimen_id"].astype(str); mt0 = mt0.set_index("specimen_id")
    dv = q2.join(mt0[["inferred_met_type"]], how="inner")
    dv = dv[dv["inferred_met_type"].astype(str).str.startswith("L5 ET")]
    iv = exprv.index.intersection(dv.index)
    dv = dv.loc[iv]; exprv = exprv.loc[iv]

    E = pd.DataFrame(index=iv)
    for f in V1_FLAT:
        if f in ph.columns:
            E[f] = ph[f].reindex(iv)
    for stem, how in V1_STEP.items():
        cols = [c for c in ph.columns
                if re.match(rf"^{stem}_\d+_long_square$", c)]
        if cols:
            sub = ph[cols].reindex(iv)
            E[f"{stem}_{how}"] = sub.max(axis=1) if how == "max" else sub.mean(axis=1)
    mt = dv["inferred_met_type"].astype(str)
    kk = mt.value_counts()[lambda x: x >= 8].index
    mt = mt.where(mt.isin(kk), "other")
    covv = pd.concat([dv[["log_libsize", "n_detected", "top50_share"]],
                      pd.get_dummies(mt, prefix="m", drop_first=True).astype(float)],
                     axis=1)
    print(f"  {len(iv)} L5 ET cells, {exprv.shape[1]} genes, {E.shape[1]} features")
    print(f"  detection floor at n={len(iv)}: "
          f"{np.tanh(2.8016/np.sqrt(len(iv)-3)):.3f}")
    Hv = screen_cohort(exprv, E, list(E.columns), covv, "V1")
    Hv.to_csv(OUT / "AL_asd_v1_hits.csv", index=False)
    bg_v = list(exprv.columns)
    mean_v = exprv.mean().to_dict(); det_v = (exprv > 0).mean().to_dict()
    a_v, b_v = apply_exclusions(Hv, bg_v) if len(Hv) else (Hv, Hv)
    nb = b_v.gene.nunique() if len(b_v) else 0
    print(f"\n  V1 hit genes: all {Hv.gene.nunique() if len(Hv) else 0}, "
          f"after channel filter {a_v.gene.nunique() if len(a_v) else 0}, "
          f"after +tau filter {nb}")
    if nb < 20:
        print(f"\n  >>> FUTILITY RULE: only {nb} genes survive "
              f"exclusions; enrichment test is uninformative.")
    for tier, tg in [("all(1-3)", allg), ("score1", tier1)]:
        print(f"\n  --- SFARI tier {tier} ---")
        if len(Hv):
            enrich(Hv.gene.unique(), bg_v, mean_v, det_v, tg,
                   "V1 all hits", rows, "mouse V1 (TEST)", tier)
            enrich(a_v.gene.unique(), bg_v, mean_v, det_v, tg,
                   "V1 minus channels/receptors/synaptic", rows,
                   "mouse V1 (TEST)", tier)
            enrich(b_v.gene.unique(), bg_v, mean_v, det_v, tg,
                   "V1 minus channels AND tau-only  <-- PRIMARY", rows,
                   "mouse V1 (TEST)", tier)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "AM_asd_enrichment.csv", index=False)
    print(f"\n[write] {OUT/'AM_asd_enrichment.csv'}")

    # overlap of hit genes between cohorts, as a descriptive anchor
    if len(Hv) and len(Hm):
        ov = set(b_m.gene) & set(b_v.gene)
        print(f"\n  genes hit in BOTH cohorts after exclusions: {len(ov)}")
        if ov:
            sf = sorted(g for g in ov if g.upper() in allg)
            print(f"    {sorted(ov)[:40]}")
            print(f"    of which SFARI: {sf}")


if __name__ == "__main__":
    main()
