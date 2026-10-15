"""Set-level test across cell classes, with GABAergic cells as control.

The aggregate is tested instead of single genes, and every Patch-seq cell class
is used rather than only the L5 excitatory population (n=126):

  * Set, not gene. If the atlas-nominated state genes are systematically
    negatively correlated with rheobase in Patch-seq, that is a coherent
    aggregate signal even when no individual gene clears FDR.
  * Every class. Mouse M1 Patch-seq has ~290 Pvalb, ~270 Sst and ~150 Vip
    cells against 126 L5 excitatory; the GABAergic arms serve as a negative
    control if the effect is excitatory-specific, and as a better-powered
    test if it is not.

Design is a crossed one -- gene sets nominated from atlas excitatory (L5 PT, the
ET homolog) and atlas GABAergic clusters, each tested in both Patch-seq
excitatory and GABAergic populations. Specificity shows up as a diagonal.

The null is expression-matched: random gene sets of the same size drawn from the
same mean-expression deciles, so the comparison controls for the fact that
highly-expressed genes have less noisy correlations.

Usage:
  uv run python analysis/cross_dataset/scripts/19_setlevel_crossclass.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)

from importlib import import_module

import h5py
import numpy as np
import pandas as pd
from scipy import stats

_m3 = import_module("03_gene_level_rheobase")
PANELS = import_module("11_atlas_extract_panels").PANELS
from plasticity_common import load_ephys, load_meta  # noqa: E402

ROOT = paths.REPO
H5 = ROOT / "Gene_Expression" / "Data" / "expression_matrix.hdf5"
ATLAS_META = ROOT / "Gene_Expression" / "Data" / "metadata.csv"
OUT = paths.results(__file__)

STATE_PANEL = "jun_ap1"
TOP_K = 80
N_NULL = 500
MIN_CLUSTER_N = 20
MIN_DETECT = 0.15
ENDPOINT = "Rheobase (pA)"

# atlas donor populations for stage 1
ATLAS_SETS = {
    "exc_L5PT": ["L5 PT CTX"],
    "gaba_PvalbSstVip": ["Pvalb", "Sst", "Vip"],
}
# Patch-seq test populations for stage 2, by RNA family
PSEQ_POPS = {
    "pseq_L5_exc": ("l5", None),
    "pseq_Pvalb": ("family", "Pvalb"),
    "pseq_Sst": ("family", "Sst"),
    "pseq_Vip": ("family", "Vip"),
    "pseq_GABA_all": ("family", ("Pvalb", "Sst", "Vip", "Lamp5")),
}


# ---------------------------------------------------------------- stage 1

def atlas_state_genes() -> dict[str, pd.Series]:
    meta = pd.read_csv(ATLAS_META, low_memory=False).set_index("sample_name")
    out = {}
    with h5py.File(H5, "r") as f:
        genes = np.array([g.decode() for g in f["gene_names"][:]])
        cells = np.array([c.decode() for c in f["sample_names"][:]])
        libsize = np.asarray(f["data/total_exon_counts"][:], dtype=np.float64)
        grp = f["data/exon"]
        p = np.asarray(grp["p"][:], dtype=np.int64)
        n_all = int(grp["dims"][0])
        gi = {g: i for i, g in enumerate(genes)}
        md = meta.reindex(cells)

        def raw(j):
            lo, hi = p[j], p[j + 1]
            v = np.zeros(n_all)
            if hi > lo:
                v[np.asarray(grp["i"][lo:hi], dtype=np.int64)] = \
                    np.asarray(grp["x"][lo:hi], dtype=np.float64)
            return v

        for name, subclasses in ATLAS_SETS.items():
            sub = md["subclass_label"].astype(str)
            sel = np.zeros(len(md), dtype=bool)
            for s in subclasses:
                sel |= sub.str.startswith(s).to_numpy()
            clu = md["cluster_label"].to_numpy()
            vc = pd.Series(clu[sel]).value_counts()
            big = set(vc[vc >= MIN_CLUSTER_N].index)
            sel &= np.array([c in big for c in clu])
            idx = np.flatnonzero(sel)
            cats = pd.Series(clu[idx]).astype("category")
            codes = cats.cat.codes.to_numpy()
            ng = len(cats.cat.categories)
            cnt = np.bincount(codes, minlength=ng).astype(float)
            lib = libsize[idx]
            print(f"[stage1 {name}] {len(idx)} atlas cells, {ng} clusters")

            def wc(v):
                m = np.bincount(codes, weights=v, minlength=ng) / cnt
                return v - m[codes]

            cols = [g for g in PANELS[STATE_PANEL] if g in gi]
            Z = np.column_stack([np.log1p(raw(gi[g])[idx] / lib * 1e6) for g in cols])
            sd = Z.std(0, ddof=1)
            Z = (Z - Z.mean(0)) / np.where(sd > 0, sd, np.nan)
            st = wc(np.nanmean(Z, axis=1))
            sc = st - st.mean()
            sn = np.sqrt((sc ** 2).sum())

            rho = np.full(len(genes), np.nan)
            for j in range(len(genes)):
                lo, hi = p[j], p[j + 1]
                if hi - lo < 5:
                    continue
                v = raw(j)[idx]
                if (v > 0).mean() < MIN_DETECT:
                    continue
                r = wc(np.log1p(v / lib * 1e6)); r -= r.mean()
                den = np.sqrt((r ** 2).sum()) * sn
                if den > 0:
                    rho[j] = float((r * sc).sum() / den)
            s = pd.Series(rho, index=genes).dropna()
            panel_genes = {g for pl in PANELS.values() for g in pl}
            s = s.drop(index=[g for g in panel_genes if g in s.index], errors="ignore")
            out[name] = s.nlargest(TOP_K)
            print(f"    nominated {len(out[name])} genes, "
                  f"top r = {out[name].iloc[0]:+.3f}")
    return out


# ---------------------------------------------------------------- stage 2

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sets = atlas_state_genes()

    meta = load_meta()
    ephys = load_ephys()
    log_cpm, qc = _m3.load_full_logcpm(meta.index, use_introns=True)
    expr_all, _ = _m3.filter_genes(log_cpm, drop_channels=False)
    design_all = meta.loc[expr_all.index].join(qc, how="left")
    print(f"\n[stage2] Patch-seq matrix {expr_all.shape[0]} cells x "
          f"{expr_all.shape[1]} genes")

    l5_ids = set(meta.index[meta["RNA type"].str.match(r"^L5 (IT|ET)_\d+$", na=False)])
    rows = []
    for pop, (kind, val) in PSEQ_POPS.items():
        if kind == "l5":
            idx = expr_all.index.intersection(list(l5_ids))
        else:
            fams = (val,) if isinstance(val, str) else val
            idx = design_all.index[design_all["RNA family"].isin(fams)]
        idx = expr_all.index.intersection(idx)
        y = ephys[ENDPOINT].reindex(idx)
        if y.notna().sum() < 40:
            print(f"[skip] {pop}: n={y.notna().sum()}")
            continue
        expr = expr_all.loc[idx]
        expr = expr.loc[:, (expr > 0).mean(axis=0) >= 0.25]
        d = design_all.loc[idx]
        cov = d[_m3.COVARS].copy()
        rt = d["RNA type"]
        keep = rt.value_counts()[lambda s: s >= 8].index
        rt = rt.where(rt.isin(keep), "other")
        if rt.nunique() > 1:
            cov = pd.concat([cov, pd.get_dummies(rt, prefix="t",
                                                 drop_first=True).astype(float)], axis=1)
        res = _m3.screen(expr, y, covars=cov).set_index("gene")
        n_eff = int(res["n"].iloc[0])
        mean_expr = expr.mean()
        bins = pd.qcut(mean_expr, 10, labels=False, duplicates="drop")
        rng = np.random.default_rng(0)
        by_bin = {b: np.array(bins.index[bins == b]) for b in bins.dropna().unique()}

        print(f"\n[{pop}] n={n_eff}, {expr.shape[1]} genes, "
              f"floor |rho|>={np.tanh(2.8016/np.sqrt(n_eff-3)):.3f}")
        for sname, s in sets.items():
            present = [g for g in s.index if g in res.index]
            if len(present) < 15:
                continue
            obs = float(res.loc[present, "rho"].mean())
            n_neg = int((res.loc[present, "rho"] < 0).sum())
            # expression-matched null on the SET MEAN
            null = []
            for _ in range(N_NULL):
                pick = []
                for g in present:
                    b = bins.get(g)
                    cand = by_bin.get(b)
                    if cand is not None and len(cand):
                        pick.append(cand[rng.integers(len(cand))])
                if len(pick) >= 15:
                    null.append(res.loc[pick, "rho"].mean())
            null = np.asarray(null)
            z = (obs - null.mean()) / null.std(ddof=1)
            p_emp = float((null <= obs).mean())
            p_sign = float(stats.binomtest(n_neg, len(present), 0.5).pvalue)
            rows.append({"set": sname, "population": pop, "n_cells": n_eff,
                         "n_genes": len(present), "mean_rho": obs,
                         "null_mean": float(null.mean()),
                         "null_sd": float(null.std(ddof=1)), "z": z,
                         "p_emp_lower": p_emp,
                         "frac_negative": n_neg / len(present),
                         "p_sign": p_sign})
            print(f"    {sname:20s} mean rho {obs:+.4f} "
                  f"(null {null.mean():+.4f}±{null.std(ddof=1):.4f})  z={z:+.2f}"
                  f"  p={p_emp:.3f}   {100*n_neg/len(present):.0f}% negative"
                  f" (sign p={p_sign:.3g})")

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "R_setlevel_crossclass.csv", index=False)

    print("\n" + "=" * 96)
    print("CROSSED DESIGN — set-level mean rho vs rheobase (z vs expression-matched null)")
    print("=" * 96)
    if len(R):
        piv = R.pivot(index="set", columns="population", values="z")
        print(piv.to_string(float_format=lambda v: f"{v:+.2f}"))
        print("\n  Specificity would appear as a diagonal: excitatory-derived sets")
        print("  strong in excitatory cells, GABA-derived in GABA cells. A flat")
        print("  pattern means the axis is generic; nothing anywhere means no")
        print("  set-level signal at all.")
    print(f"\n[write] {OUT/'R_setlevel_crossclass.csv'}")


if __name__ == "__main__":
    main()
