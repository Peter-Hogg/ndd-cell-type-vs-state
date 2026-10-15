"""Tests of layer specificity, one per dataset.

TEST 1 (Patch-seq) — FORMAL heterogeneity of the IEG->rheobase coupling.
"Each layer's top ephys feature differs" is NOT evidence of layer specificity:
with 6 layers x 29 features the argmax differs by chance almost surely. This
test fixes the feature (rheobase) and asks whether the category-level effect
differs across layers more than sampling error allows -- Cochran's Q on the
per-layer effects, as in the single-gene layer meta-analysis
(`analysis/m1_patchseq/scripts/20_layer_meta_analysis.py`). Input is
`analysis/m1_patchseq/results/V_functional_categories_by_layer.csv`.

TEST 2 (RNA-seq atlas) — is the activity PROGRAMME itself layer-specific?
Independent of any electrophysiology. Compute the within-cluster state axis
separately inside each glutamatergic subclass, then correlate the genome-wide
loading vectors between layers. If the programme is shared, loadings correlate
near 1; if layer-specific, they diverge. 48,796 cells, so this is well powered
where the Patch-seq test is not.

Usage:
  uv run python analysis/cross_dataset/scripts/25_layer_specificity_test.py
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

_m23 = import_module("23_functional_categories_by_layer")
PANELS = import_module("11_atlas_extract_panels").PANELS

ROOT = paths.REPO
H5 = ROOT / "Gene_Expression" / "Data" / "expression_matrix.hdf5"
ATLAS_META = ROOT / "Gene_Expression" / "Data" / "metadata.csv"
OUT = paths.results(__file__)

STATE_PANEL = "jun_ap1"
MIN_CLUSTER_N = 20
MIN_DETECT = 0.15
SUBCLASSES = ["L2/3 IT CTX", "L4/5 IT CTX", "L5 IT CTX", "L5 PT CTX",
              "L6 IT CTX", "L6 CT CTX"]


def cochran_q(z_vals, n_vals):
    ok = [(z, n) for z, n in zip(z_vals, n_vals)
          if np.isfinite(z) and n and n > 4]
    if len(ok) < 2:
        return {}
    # convert category-level z back to an effect on the Fisher scale using the
    # observed mean rho, weighted by stratum size
    return ok


def main() -> None:
    # ---------------- TEST 1 ----------------------------------------------
    print("=" * 88)
    print("TEST 1 — formal heterogeneity of the IEG_TF -> rheobase coupling")
    print("=" * 88)
    v = pd.read_csv(paths.M1 / "results" / "V_functional_categories_by_layer.csv")
    v = v[(v["category"] == "IEG_TF") & (~v["population"].str.contains("control"))]
    v = v[v["population"].isin(["L2/3 IT", "L4/5 IT", "L5 IT", "L5 ET",
                                "L6 CT", "L6 IT"])]
    print(v[["population", "n", "n_genes", "mean_rho", "z", "p_two_sided"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Fisher-z on the category mean rho, inverse-variance weights ~ (n-3)
    z = np.arctanh(v["mean_rho"].clip(-0.999, 0.999).to_numpy())
    w = (v["n"].to_numpy() - 3).astype(float)
    zbar = (w * z).sum() / w.sum()
    Q = float((w * (z - zbar) ** 2).sum())
    df = len(z) - 1
    pQ = float(stats.chi2.sf(Q, df))
    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    print(f"\n  pooled (FE) rho = {np.tanh(zbar):+.4f}")
    print(f"  Cochran Q = {Q:.2f} (df {df}), p = {pQ:.4g}, I² = {I2:.0f}%")
    if pQ < 0.05:
        print("  -> LAYER-SPECIFIC: the coupling genuinely differs across layers.")
    else:
        print("  -> NOT formally heterogeneous. The apparent layer differences are")
        print("     within sampling error; do not claim layer specificity from this.")
    print(f"\n  (compare: single-gene Q for Jun in 20_layer_meta_analysis.py)")
    print("  NB the category mean rho is an average over ~34 correlated genes, so")
    print("  the (n-3) weights understate its variance; this Q is approximate and")
    print("  anti-conservative. Treat p just below 0.05 with caution.")

    # ---------------- TEST 2 ----------------------------------------------
    print("\n" + "=" * 88)
    print("TEST 2 — is the activity programme layer-specific in the ATLAS?")
    print("=" * 88)
    meta = pd.read_csv(ATLAS_META, low_memory=False).set_index("sample_name")
    loadings = {}
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
            v_ = np.zeros(n_all)
            if hi > lo:
                v_[np.asarray(grp["i"][lo:hi], dtype=np.int64)] = \
                    np.asarray(grp["x"][lo:hi], dtype=np.float64)
            return v_

        for sc in SUBCLASSES:
            sel = np.array(md["subclass_label"].astype(str) == sc, dtype=bool)
            clu = md["cluster_label"].to_numpy()
            vc = pd.Series(clu[sel]).value_counts()
            big = set(vc[vc >= MIN_CLUSTER_N].index)
            sel &= np.array([c in big for c in clu])
            idx = np.flatnonzero(sel)
            if len(idx) < 150:
                print(f"  [skip] {sc}: {len(idx)} cells")
                continue
            cats = pd.Series(clu[idx]).astype("category")
            codes = cats.cat.codes.to_numpy()
            ng = len(cats.cat.categories)
            cnt = np.bincount(codes, minlength=ng).astype(float)
            lib = libsize[idx]

            def wc(vv):
                mm = np.bincount(codes, weights=vv, minlength=ng) / cnt
                return vv - mm[codes]

            cols = [g for g in PANELS[STATE_PANEL] if g in gi]
            Zm = np.column_stack([np.log1p(raw(gi[g])[idx] / lib * 1e6) for g in cols])
            sd = Zm.std(0, ddof=1)
            Zm = (Zm - Zm.mean(0)) / np.where(sd > 0, sd, np.nan)
            st = wc(np.nanmean(Zm, axis=1))
            sc_ = st - st.mean()
            sn = np.sqrt((sc_ ** 2).sum())

            rho = np.full(len(genes), np.nan)
            for j in range(len(genes)):
                lo, hi = p[j], p[j + 1]
                if hi - lo < 5:
                    continue
                vv = raw(j)[idx]
                if (vv > 0).mean() < MIN_DETECT:
                    continue
                r = wc(np.log1p(vv / lib * 1e6)); r -= r.mean()
                den = np.sqrt((r ** 2).sum()) * sn
                if den > 0:
                    rho[j] = float((r * sc_).sum() / den)
            loadings[sc] = pd.Series(rho, index=genes)
            print(f"  {sc:14s} {len(idx):6d} cells, {ng:3d} clusters, "
                  f"{int(np.isfinite(rho).sum()):6d} genes")

    L = pd.DataFrame(loadings)
    L.to_csv(OUT / "X_atlas_state_loadings_by_layer.csv")
    common = L.dropna()
    print(f"\n  genes testable in all {L.shape[1]} subclasses: {len(common)}")
    print("\n  cross-layer correlation of genome-wide state-axis loadings:")
    corr = common.corr(method="spearman")
    print(corr.to_string(float_format=lambda x: f"{x:+.3f}"))
    off = corr.to_numpy()[np.triu_indices_from(corr, k=1)]
    print(f"\n  mean off-diagonal rho = {off.mean():+.3f}  "
          f"(range {off.min():+.3f} to {off.max():+.3f})")
    print("\n  Interpretation: ~1.0 means one shared programme across layers;")
    print("  moderate values mean the activity programme itself is layer-specific,")
    print("  which would independently support layer-specific coupling.")

    # which panel genes are most layer-divergent?
    pg = [g for g in PANELS[STATE_PANEL] if g in common.index]
    if pg:
        print("\n  state-panel genes, loading per layer:")
        print(common.loc[pg].to_string(float_format=lambda x: f"{x:+.3f}"))
    sd_across = common.std(axis=1)
    print("\n  20 genes with the MOST layer-divergent loading:")
    top = sd_across.nlargest(20)
    print(common.loc[top.index].assign(sd=top).to_string(
        float_format=lambda x: f"{x:+.3f}"))
    print(f"\n[write] {OUT/'X_atlas_state_loadings_by_layer.csv'}")


if __name__ == "__main__":
    main()
