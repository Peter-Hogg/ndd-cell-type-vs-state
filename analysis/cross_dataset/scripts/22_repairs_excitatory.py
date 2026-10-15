"""Two robustness checks on excitatory set-level and variance-coupling tests.

REPAIR A — permutation null for the set-level test.
`19_setlevel_crossclass.py` compares a gene set's mean rho against random
expression-matched sets. That null is anti-conservative: the real set is
CO-EXPRESSED, so its mean rho has larger sampling variance than a set of
unrelated genes. This null permutes the PHENOTYPE (rheobase) instead, which
leaves the gene-gene correlation structure untouched. Permutation is done WITHIN
transcriptomic type so the within-type design is preserved.

REPAIR B — cell-quality control on the variance coupling.
`analysis/m1_patchseq/scripts/17_bidirectional_test.py` compares rheobase
variance in cells far from vs near their type's mean activity state. A sick or
dying cell could show BOTH aberrant IEG expression and aberrant rheobase,
producing the same coupling with no plasticity involved. Tested here by
(i) asking whether |state| tracks quality metrics at all, and (ii) redoing the
tercile test after residualising both variables on quality.

Excitatory cells only. GABAergic populations are reported for REPAIR A purely as
a specificity control -- no claim is made about interneuron plasticity.

Usage:
  uv run python analysis/cross_dataset/scripts/22_repairs_excitatory.py
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
from plasticity_common import load_ephys, load_meta  # noqa: E402

OUT = paths.results(__file__)
ENDPOINT = "Rheobase (pA)"
N_PERM = 5000
RNG = np.random.default_rng(0)


def _clean(v: np.ndarray) -> np.ndarray:
    """Rank-transform, filling non-finite values with the median rank."""
    v = np.asarray(v, dtype=float)
    bad = ~np.isfinite(v)
    if bad.all():
        return np.zeros_like(v)
    if bad.any():
        v = v.copy()
        v[bad] = np.nanmedian(v[~bad])
    return stats.rankdata(v)


def rank_resid(v: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Residualise ranks on the design. pinv rather than lstsq: with collinear
    type dummies in small strata the design is rank-deficient and lstsq's SVD can
    fail outright (it did, on the Sst control arm)."""
    r = _clean(v)
    return r - C @ (np.linalg.pinv(C) @ r)


def build_design(cov: pd.DataFrame, dums: pd.DataFrame, n: int) -> np.ndarray:
    """Intercept + ranked covariates + type dummies, with constant and duplicate
    columns dropped so the design is at least usable by pinv."""
    cols = [np.ones(n)]
    for c in cov.columns:
        cols.append(_clean(cov[c].to_numpy()))
    for c in dums.columns:
        cols.append(dums[c].to_numpy(dtype=float))
    M = np.column_stack(cols)
    keep = [0]
    for j in range(1, M.shape[1]):
        if M[:, j].std() < 1e-12:
            continue
        if any(abs(np.corrcoef(M[:, j], M[:, k])[0, 1]) > 0.999 for k in keep[1:]):
            continue
        keep.append(j)
    return M[:, keep]


def main() -> None:
    meta = load_meta()
    ephys = load_ephys()
    y_all = ephys[ENDPOINT]
    log_cpm, qc = _m3.load_full_logcpm(meta.index, use_introns=True)
    expr_all, _ = _m3.filter_genes(log_cpm, drop_channels=False)
    design = meta.loc[expr_all.index].join(qc, how="left")

    rt = design["RNA type"].astype(str)
    lay = pd.Series("other", index=design.index)
    for name, pat in _m20.LAYER_PATTERNS.items():
        lay[rt.str.match(pat, na=False)] = name
    design["layer"] = lay

    # atlas-nominated L5 PT state genes (outcome-blind stage 1)
    atlas = (pd.read_csv(OUT / "Q_l5pt_targeted_state_axis.csv")
             .rename(columns={"Unnamed: 0": "gene"}).set_index("gene"))
    col = "L5 PT CTX"
    panel = {g for pl in import_module("11_atlas_extract_panels").PANELS.values() for g in pl}
    nominated = [g for g in atlas[col].dropna().sort_values(ascending=False).index
                 if g not in panel][:80]

    pops = {
        "L5_excitatory": design.index[design["layer"] == "L5 ET"].union(
            design.index[design["layer"] == "L5 IT"]),
        "pyramidal_all": design.index[design["layer"] != "other"],
        "Sst_control": design.index[design["RNA family"] == "Sst"],
        "Pvalb_control": design.index[design["RNA family"] == "Pvalb"],
    }

    print("=" * 84)
    print("REPAIR A — permutation null (phenotype shuffled within type)")
    print("=" * 84)
    rowsA = []
    for pop, idx in pops.items():
        idx = expr_all.index.intersection(idx)
        y = y_all.reindex(idx)
        keep = y.notna()
        idx = idx[keep.values]
        if len(idx) < 40:
            continue
        genes = [g for g in nominated if g in expr_all.columns]
        E = expr_all.loc[idx, genes]
        det = (E > 0).mean(axis=0) >= 0.25
        E = E.loc[:, det]
        d = design.loc[idx]
        n = len(idx)

        cov = d[_m3.COVARS].copy()
        srt = d["RNA type"]
        k = srt.value_counts()[lambda s: s >= 8].index
        srt = srt.where(srt.isin(k), "other")
        dums = pd.get_dummies(srt, prefix="t", drop_first=True).astype(float)
        C = build_design(cov, dums, n)

        Er = np.column_stack([rank_resid(E[g].to_numpy(), C) for g in E.columns])
        Er -= Er.mean(0)
        Er /= np.where(Er.std(0) > 0, Er.std(0), np.nan)
        yr = rank_resid(y.loc[idx].to_numpy(), C)

        def set_mean(yv):
            yv = yv - yv.mean()
            s = yv.std()
            if s == 0:
                return np.nan
            return float(np.nanmean((Er * (yv / s)[:, None]).sum(0) / (n - 1)))

        obs = set_mean(yr)
        # permute within transcriptomic type
        groups = srt.to_numpy()
        null = np.empty(N_PERM)
        for b in range(N_PERM):
            yp = yr.copy()
            for g in np.unique(groups):
                mask = groups == g
                yp[mask] = RNG.permutation(yp[mask])
            null[b] = set_mean(yp)
        p_lo = float((null <= obs).mean())
        z = (obs - null.mean()) / null.std(ddof=1)
        tag = "  <- specificity control" if "control" in pop else ""
        print(f"  {pop:16s} n={n:4d}  {E.shape[1]:3d} genes  "
              f"mean rho {obs:+.4f}  perm null {null.mean():+.4f}±{null.std(ddof=1):.4f}"
              f"  z={z:+.2f}  p={p_lo:.4f}{tag}")
        rowsA.append({"population": pop, "n": n, "n_genes": E.shape[1],
                      "mean_rho": obs, "perm_mean": float(null.mean()),
                      "perm_sd": float(null.std(ddof=1)), "z": z, "p_lower": p_lo})
    pd.DataFrame(rowsA).to_csv(OUT / "U_repairA_permutation.csv", index=False)
    print("\n  Compare with the expression-matched null in 19: if z shrinks a lot,")
    print("  the expression-matched significance reflects ignored co-expression.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 84)
    print("REPAIR B — is the variance coupling explained by cell quality?")
    print("=" * 84)
    cells = pd.read_csv(paths.M1 / "results" / "cells_l5.csv", index_col=0)
    cells = cells[cells[ENDPOINT].notna()].copy()
    grp = "RNA type"
    QCV = ["n_detected", "libsize", "Hold Time (min)"]

    def wresid(df, col):
        return df[col] - df.groupby(grp, observed=True)[col].transform("mean")

    cells["rheo_r"] = wresid(cells, ENDPOINT)
    cells["state_r"] = wresid(cells, "plasticity_score")
    cells["abs_state"] = cells["state_r"].abs()

    print("  (i) does |state deviation| track quality metrics?")
    for q in QCV:
        if q in cells.columns:
            s = cells[["abs_state", q]].dropna()
            r, p = stats.spearmanr(s["abs_state"], s[q])
            flag = "  <- CONFOUNDED" if p < 0.05 else ""
            print(f"      |state| vs {q:18s} rho = {r:+.3f}  p = {p:.3g}{flag}")

    def tercile_test(df, label):
        q1, q2 = df["abs_state"].quantile([1/3, 2/3])
        lo = df.loc[df["abs_state"] <= q1, "rheo_r"]
        hi = df.loc[df["abs_state"] >= q2, "rheo_r"]
        if len(lo) < 8 or len(hi) < 8:
            return
        W, p = stats.levene(lo, hi, center="median")
        vr = hi.var(ddof=1) / lo.var(ddof=1)
        print(f"      {label:34s} SD {lo.std(ddof=1):5.1f} -> {hi.std(ddof=1):5.1f} pA"
              f"   ratio {vr:4.2f}   Levene p = {p:.4g}")

    print("\n  (ii) tercile test, before and after removing quality")
    tercile_test(cells, "raw (as in script 17)")

    have = [q for q in QCV if q in cells.columns and cells[q].notna().mean() > 0.8]
    sub = cells.dropna(subset=have + ["rheo_r", "state_r"]).copy()
    Cq = np.column_stack([np.ones(len(sub))] +
                         [_clean(sub[q].to_numpy()) for q in have])
    P = np.linalg.pinv(Cq)
    for c in ("rheo_r", "state_r"):
        v = sub[c].to_numpy(dtype=float)
        sub[c] = v - Cq @ (P @ v)
    sub["abs_state"] = sub["state_r"].abs()
    tercile_test(sub, f"after removing {len(have)} quality covars")
    print(f"      quality covariates used: {have}   n = {len(sub)}")
    print("\n  If the ratio collapses toward 1, the coupling was cell health.")
    print("  If it survives, bidirectional plasticity remains viable.")


if __name__ == "__main__":
    main()
