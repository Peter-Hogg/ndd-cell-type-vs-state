"""Is the activity/adaptation link WITHIN a cell type, or only ACROSS types?

`36_adaptation_three_cohorts.py` compares mouse M1 pooled pyramidal cells with
mouse V1 L5 ET and macaque L5 ET, but the cohorts are not the same population:
M1 pooled spans five pyramidal layers, V1 is L5 ET alone. Here the M1 effect is
decomposed by population. Two possibilities with opposite meanings:

  (a) M1 effect holds WITHIN L5     -> a V1 null contradicts it
  (b) M1 effect exists only POOLED  -> it is a between-type (pooling) effect,
                                       and V1 L5 ET does not test it

Also re-tests V1 with single well-populated current steps (adapt_4, adapt_6,
adapt_2) instead of the mean over 11 steps, in case averaging across steps with
very different sample sizes dilutes the signal.

Cochran's Q across layers tests whether the per-layer effects are even
consistent with one another.

Usage:
  uv run python analysis/cross_dataset/scripts/37_adaptation_population_control.py
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

_m3 = import_module("03_gene_level_rheobase")
_m20 = import_module("20_layer_meta_analysis")
_m22 = import_module("22_repairs_excitatory")
_m23 = import_module("23_functional_categories_by_layer")
_m28 = import_module("28_v1_confirmatory")
_m31 = import_module("31_ieg_independent_of_channels")
_m36 = import_module("36_adaptation_three_cohorts")
from plasticity_common import load_ephys, load_meta  # noqa: E402

ROOT = paths.REPO
V1D = ROOT / "ReplicationData" / "MouseV1_Sorensen2026"
OUT = paths.results(__file__)
N_PERM = 3000
RNG = np.random.default_rng(0)
IEG = _m23.CATEGORIES["IEG_TF"]
MEAS = "ISI average adaptation index"      # M1 adaptation measure
MIN_N = 30


def one(expr_hk, expr_raw, y, cov, dums, grp, label):
    n = len(y)
    C = _m22.build_design(cov, dums, n)
    v, ng = _m31.index_score_filtered(expr_hk, expr_raw, IEG)
    if v is None:
        return None
    gv = _m22.rank_resid(v, C); gv -= gv.mean()
    gv = gv / gv.std() if gv.std() > 0 else gv
    yr = _m22.rank_resid(np.asarray(y, float), C)
    yc = (yr - yr.mean()) / yr.std()
    obs = float((gv * yc).sum() / (n - 1))
    perms = np.empty(N_PERM)
    for i in range(N_PERM):
        yp = yr.copy()
        for g in np.unique(grp):
            m = grp == g
            yp[m] = RNG.permutation(yp[m])
        yp = (yp - yp.mean()) / yp.std()
        perms[i] = float((gv * yp).sum() / (n - 1))
    p = float(2 * min((perms <= obs).mean(), (perms >= obs).mean()))
    floor = float(np.tanh(2.8016 / np.sqrt(max(n - 3, 4))))
    return {"population": label, "n": n, "n_genes": ng, "rho": obs,
            "p_perm": p, "floor80": floor}


def main() -> None:
    rows = []
    meta = load_meta(); ephys = load_ephys()
    lc, qc = _m3.load_full_logcpm(meta.index, use_introns=True)
    hkn, _ = _m36.hk_normalise(lc, _m36.HK_MOUSE)
    d0 = meta.loc[lc.index].join(qc, how="left")
    rt = d0["RNA type"].astype(str)
    lay = pd.Series("other", index=d0.index)
    for nm, pat in _m20.LAYER_PATTERNS.items():
        lay[rt.str.match(pat, na=False)] = nm
    d0["layer"] = lay
    yall = ephys[MEAS]

    pops = {"M1 pyramidal pooled (5 layers)": d0.index[lay != "other"],
            "M1 L5 exc (IT+ET)": d0.index[rt.str.match(r"^L5 (IT|ET)_\d+$", na=False)],
            "M1 L5 ET only": d0.index[rt.str.match(r"^L5 ET_\d+$", na=False)]}
    for nm in sorted(set(lay[lay != "other"])):
        pops[f"M1 {nm} only"] = d0.index[lay == nm]

    print(f"[M1] measure = {MEAS}")
    for label, pidx in pops.items():
        pidx = lc.index.intersection(pidx)
        y = yall.reindex(pidx)
        keep = pidx[y.notna().values]
        if len(keep) < MIN_N:
            print(f"  skip {label}: n={len(keep)} < {MIN_N}")
            continue
        d = d0.loc[keep]
        s = d["RNA type"]
        k = s.value_counts()[lambda x: x >= 8].index
        s = s.where(s.isin(k), "other")
        r = one(hkn.loc[keep], lc.loc[keep], y.loc[keep], d[_m3.COVARS],
                pd.get_dummies(s, prefix="t", drop_first=True).astype(float),
                s.to_numpy(), label)
        if r:
            rows.append(r)

    # ---- V1, single best-populated step rather than the mean ---------------
    lcv, cnt, q2 = _m28.load_v1()
    lcv.index = lcv.index.astype(str); q2.index = q2.index.astype(str)
    hkv, _ = _m36.hk_normalise(lcv, _m36.HK_MOUSE)
    ph = pd.read_csv(V1D / "exc_mMET_ephys_features.csv")
    ph["specimen_id"] = ph["specimen_id"].astype(str); ph = ph.set_index("specimen_id")
    mt0 = pd.read_csv(V1D / "inferred_met_types.csv").rename(
        columns={"Unnamed: 0": "specimen_id"})
    mt0["specimen_id"] = mt0["specimen_id"].astype(str); mt0 = mt0.set_index("specimen_id")
    dv = q2.join(mt0[["inferred_met_type"]], how="inner")
    dv = dv[dv["inferred_met_type"].astype(str).str.startswith("L5 ET")]
    iv = lcv.index.intersection(dv.index)
    for step in [4, 6, 2]:
        col = f"adapt_{step}_long_square"
        y = ph[col].reindex(iv)
        keep = iv[y.notna().values]
        if len(keep) < MIN_N:
            continue
        d = dv.loc[keep]
        mt = d["inferred_met_type"].astype(str)
        kk = mt.value_counts()[lambda x: x >= 8].index
        mt = mt.where(mt.isin(kk), "other")
        r = one(hkv.loc[keep], lcv.loc[keep], y.loc[keep],
                d[["log_libsize", "n_detected"]],
                pd.get_dummies(mt, prefix="m", drop_first=True).astype(float),
                mt.to_numpy(), f"V1 L5 ET  {col}")
        if r:
            rows.append(r)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "AJ_adaptation_population_control.csv", index=False)

    print("\n" + "=" * 92)
    print("ACTIVITY INDEX vs ADAPTATION — pooled versus within-type")
    print("=" * 92)
    print(f"  {'population':34s} {'n':>4s} {'rho':>8s} {'p':>8s} {'floor80':>8s}")
    for _, r in R.iterrows():
        star = " *" if r.p_perm < 0.05 else "  "
        print(f"  {r.population:34s} {int(r.n):4d} {r.rho:+8.3f} "
              f"{r.p_perm:8.4f} {r.floor80:8.3f}{star}")

    # ---- Cochran Q across the M1 single-layer strata -----------------------
    lay_rows = R[R.population.str.match(r"^M1 (L\d|L\d/\d)")
                 & ~R.population.str.contains("pooled")]
    if len(lay_rows) >= 3:
        z = np.arctanh(lay_rows["rho"].to_numpy())
        w = lay_rows["n"].to_numpy() - 3.0
        zbar = float((w * z).sum() / w.sum())
        Q = float((w * (z - zbar) ** 2).sum())
        df = len(z) - 1
        pq = float(stats.chi2.sf(Q, df))
        I2 = max(0.0, 100 * (Q - df) / Q) if Q > 0 else 0.0
        print(f"\n  across {len(z)} M1 strata: pooled Fisher-z rho = "
              f"{np.tanh(zbar):+.3f}")
        print(f"  Cochran Q = {Q:.2f} (df={df}) p = {pq:.4f}, I2 = {I2:.0f}%")
        print("  a within-type effect should show a pooled rho close to the "
              "pooled-population value;")
        print("  if it collapses toward zero, the pooled effect was BETWEEN types.")

    print(f"\n[write] {OUT/'AJ_adaptation_population_control.csv'}")


if __name__ == "__main__":
    main()
