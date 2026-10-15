"""EXPLORATORY — IEG index vs gene-set statistic, and which genes carry the set effect.

Re-uses the V1 cohort tested in `analysis/v1_patchseq/scripts/28_v1_confirmatory.py`
with different estimators; all results here are exploratory.

Three questions:

1. INDEX vs SET. The set test A2 averages 35 per-gene correlations. An
   "IEG index" instead averages expression into one score per cell and takes a
   single correlation. These differ: averaging expression can cancel signal when
   genes point different ways, while averaging correlations preserves consistent
   direction. Both are computed here, in both cohorts.

2. WHICH GENES. Per-gene rho vs rheobase for all 35 IEG_TF genes in mouse M1 L5
   (n=126) and mouse V1 L5 ET (n=436).

3. IS IT A GROUP EFFECT? Two diagnostics:
   * correlate the 35 per-gene rho values BETWEEN cohorts. If the gene-level
     pattern reproduces, the programme is coherent; if not, the set mean is just
     an average of noise that happens to lean negative.
   * leave-one-out on the V1 set statistic: if dropping any single gene collapses
     it, the effect is concentrated rather than distributed.

Usage:
  uv run python analysis/cross_dataset/scripts/30_index_vs_set_exploratory.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)

from importlib import import_module

import numpy as np
import pandas as pd
from scipy import stats

_m3 = import_module("03_gene_level_rheobase")
_m20 = import_module("20_layer_meta_analysis")
_m22 = import_module("22_repairs_excitatory")
_m23 = import_module("23_functional_categories_by_layer")
_m28 = import_module("28_v1_confirmatory")
from plasticity_common import load_ephys, load_meta  # noqa: E402

ROOT = paths.REPO
V1 = ROOT / "ReplicationData" / "MouseV1_Sorensen2026"
OUT = paths.results(__file__)
N_PERM = 5000
RNG = np.random.default_rng(0)
CAT = _m23.CATEGORIES["IEG_TF"]


def analyse(expr, y, C, genes, n, label):
    """Return per-gene rho, the set mean, and the index rho."""
    cols = [g for g in genes if g in expr.columns]
    E = expr[cols]
    E = E.loc[:, (E > 0).mean(axis=0) >= 0.25]
    yr = _m22.rank_resid(y, C)
    yc = (yr - yr.mean()) / yr.std()

    per = {}
    Er = []
    for g in E.columns:
        gv = _m22.rank_resid(E[g].to_numpy(), C)
        gv = gv - gv.mean()
        s = gv.std()
        gv = gv / s if s > 0 else gv
        Er.append(gv)
        per[g] = float((gv * yc).sum() / (n - 1))
    Er = np.column_stack(Er)
    set_mean = float(np.mean(list(per.values())))

    # index: mean z-scored expression -> one correlation
    Z = (E - E.mean()) / E.std(ddof=1)
    idx_score = _m22.rank_resid(Z.mean(axis=1).to_numpy(), C)
    a = idx_score - idx_score.mean()
    r_index = float((a * yc).sum() / np.sqrt((a ** 2).sum() * (yc ** 2).sum())
                    * np.sqrt((yc ** 2).sum()) / np.sqrt((yc ** 2).sum()))
    aa = a / a.std()
    r_index = float((aa * yc).sum() / (n - 1))
    dof = n - 2 - (C.shape[1] - 1)
    t = r_index * np.sqrt(dof / max(1e-12, 1 - r_index ** 2))
    p_index = float(2 * stats.t.sf(abs(t), dof))

    # permutation null for both statistics, same draws
    grp = getattr(analyse, "_grp", None)
    nulls_set, nulls_idx = [], []
    for _ in range(N_PERM):
        yp = yr.copy()
        for gg in np.unique(grp):
            m = grp == gg
            yp[m] = RNG.permutation(yp[m])
        yp = (yp - yp.mean()) / yp.std()
        nulls_set.append(float(np.mean((Er * yp[:, None]).sum(0) / (n - 1))))
        nulls_idx.append(float((aa * yp).sum() / (n - 1)))
    ns, ni = np.asarray(nulls_set), np.asarray(nulls_idx)
    return {
        "label": label, "n": n, "n_genes": E.shape[1], "per_gene": per,
        "set_mean": set_mean, "set_p": float((ns <= set_mean).mean()),
        "set_z": (set_mean - ns.mean()) / ns.std(ddof=1),
        "index_rho": r_index, "index_p_param": p_index,
        "index_p_perm": float((ni <= r_index).mean()),
        "index_z": (r_index - ni.mean()) / ni.std(ddof=1),
        "Er": Er, "yc": yc, "cols": list(E.columns),
    }


def main() -> None:
    print("*" * 76)
    print("EXPLORATORY — V1 re-used with different estimators")
    print("*" * 76)

    # ---------------- mouse M1 ----------------
    meta = load_meta(); ephys = load_ephys()
    log_cpm, qc = _m3.load_full_logcpm(meta.index, use_introns=True)
    expr, _ = _m3.filter_genes(log_cpm, drop_channels=False)
    d0 = meta.loc[expr.index].join(qc, how="left")
    rt = d0["RNA type"].astype(str)
    l5 = d0.index[rt.str.match(r"^L5 (IT|ET)_\d+$", na=False)]
    idx = expr.index.intersection(l5)
    idx = idx[ephys["Rheobase (pA)"].reindex(idx).notna().values]
    d = d0.loc[idx]; n1 = len(idx)
    s = d["RNA type"]; k = s.value_counts()[lambda x: x >= 8].index
    s = s.where(s.isin(k), "other")
    C1 = _m22.build_design(d[_m3.COVARS],
                           pd.get_dummies(s, prefix="t", drop_first=True).astype(float), n1)
    analyse._grp = s.to_numpy()
    m1 = analyse(expr.loc[idx], ephys["Rheobase (pA)"].reindex(idx).to_numpy(),
                 C1, CAT, n1, f"mouse M1 L5 exc (n={n1})")

    # ---------------- mouse V1 ----------------
    lc, counts, q2 = _m28.load_v1()
    lc.index = lc.index.astype(str); q2.index = q2.index.astype(str)
    phys = pd.read_csv(V1 / "exc_mMET_ephys_features.csv")
    phys["specimen_id"] = phys["specimen_id"].astype(str)
    phys = phys.set_index("specimen_id")
    met = pd.read_csv(V1 / "inferred_met_types.csv").rename(
        columns={"Unnamed: 0": "specimen_id"})
    met["specimen_id"] = met["specimen_id"].astype(str); met = met.set_index("specimen_id")
    dv = q2.join(met[["inferred_met_type"]], how="inner").join(phys[["rheobase_i"]],
                                                              how="inner")
    dv = dv[dv["rheobase_i"].notna()]
    dv = dv[dv["inferred_met_type"].astype(str).str.startswith("L5 ET")]
    iv = lc.index.intersection(dv.index); dv = dv.loc[iv]; n2 = len(iv)
    mt = dv["inferred_met_type"].astype(str)
    kk = mt.value_counts()[lambda x: x >= 8].index
    mt = mt.where(mt.isin(kk), "other")
    C2 = _m22.build_design(dv[["log_libsize", "n_detected", "top50_share"]],
                           pd.get_dummies(mt, prefix="m", drop_first=True).astype(float), n2)
    analyse._grp = mt.to_numpy()
    v1 = analyse(lc.loc[iv], dv["rheobase_i"].to_numpy(), C2, CAT, n2,
                 f"mouse V1 L5 ET (n={n2})")

    # ---------------- 1. index vs set ----------------
    print("\n" + "=" * 76)
    print("1. INDEX (one score per cell) vs SET (mean of per-gene rho)")
    print("=" * 76)
    for r in (m1, v1):
        print(f"\n  {r['label']}   {r['n_genes']} genes")
        print(f"    SET   mean rho {r['set_mean']:+.4f}   perm p {r['set_p']:.4f}"
              f"   z {r['set_z']:+.2f}")
        print(f"    INDEX     rho {r['index_rho']:+.4f}   perm p {r['index_p_perm']:.4f}"
              f"   z {r['index_z']:+.2f}   (parametric p {r['index_p_param']:.4g})")

    # ---------------- 2/3. which genes, and is it a group? ----------------
    common = [g for g in m1["cols"] if g in v1["cols"]]
    tab = pd.DataFrame({"M1_rho": [m1["per_gene"][g] for g in common],
                        "V1_rho": [v1["per_gene"][g] for g in common]}, index=common)
    tab["same_sign"] = np.sign(tab.M1_rho) == np.sign(tab.V1_rho)
    tab = tab.sort_values("V1_rho")
    tab.to_csv(OUT / "AC_pergene_IEGTF_both_cohorts.csv")

    print("\n" + "=" * 76)
    print("2. PER-GENE rho vs rheobase, both cohorts")
    print("=" * 76)
    print(tab.to_string(float_format=lambda v: f"{v:+.3f}"))

    print("\n" + "=" * 76)
    print("3. IS IT A GROUP EFFECT?")
    print("=" * 76)
    rr, pp = stats.spearmanr(tab.M1_rho, tab.V1_rho)
    print(f"  cross-cohort correlation of the {len(tab)} per-gene rho values:")
    print(f"    Spearman rho = {rr:+.3f}   p = {pp:.4g}")
    print(f"  genes with the same sign in both: {int(tab.same_sign.sum())}/{len(tab)}"
          f"  (binomial p = {stats.binomtest(int(tab.same_sign.sum()), len(tab), 0.5).pvalue:.3g})")
    neg = int((tab.V1_rho < 0).sum())
    print(f"  genes negative in V1: {neg}/{len(tab)}"
          f"  (binomial p = {stats.binomtest(neg, len(tab), 0.5).pvalue:.3g})")

    print("\n  leave-one-out on the V1 set mean (is it concentrated?):")
    per = np.array([v1["per_gene"][g] for g in v1["cols"]])
    full = per.mean()
    loo = np.array([np.delete(per, i).mean() for i in range(len(per))])
    worst = int(np.argmax(loo))
    print(f"    full set mean {full:+.4f}")
    print(f"    LOO range     {loo.min():+.4f} to {loo.max():+.4f}")
    print(f"    dropping {v1['cols'][worst]:8s} weakens it most -> {loo[worst]:+.4f}")
    print(f"    still negative after dropping ANY single gene: {bool((loo<0).all())}")


if __name__ == "__main__":
    main()
