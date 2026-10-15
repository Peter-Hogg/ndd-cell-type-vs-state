"""Cluster-targeted stage 1: nominate candidates from the L5 PT (=ET) subclass.

The global state axis in `analysis/allen_atlas/scripts/15_state_axis_function.py`
averages over all glutamatergic clusters, while the Patch-seq tests are in L5 ET.
Tasic's "L5 PT CTX" (pyramidal-tract) is the atlas homolog of that population,
so candidates are selected inside it.

Still outcome-blind: the atlas has no electrophysiology, so nothing here can leak
rheobase information, and FDR in stage 2 is taken over the nominated set
(see `16_two_stage_candidate_test.py`).

Produces, per target subclass, the genes whose within-cluster residual expression
tracks the within-cluster residual state axis. Reports:
  * L5 PT specific candidates and how they rank globally
  * genes shared with, or unique to, L5 PT vs L5 IT vs L2/3 IT
  * for each candidate, the mouse M1 Patch-seq rheobase result under
    candidate-set FDR

Usage:
  uv run python analysis/cross_dataset/scripts/18_l5pt_targeted_candidates.py
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
from statsmodels.stats.multitest import multipletests

PANELS = import_module("11_atlas_extract_panels").PANELS

ROOT = paths.REPO
H5 = ROOT / "Gene_Expression" / "Data" / "expression_matrix.hdf5"
META = ROOT / "Gene_Expression" / "Data" / "metadata.csv"
OUT = paths.results(__file__)

TARGETS = ["L5 PT CTX", "L5 IT CTX", "L2/3 IT CTX"]
STATE_PANEL = "jun_ap1"
MIN_CLUSTER_N = 20
MIN_DETECT = 0.15
TOP_K = 60


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META, low_memory=False).set_index("sample_name")

    with h5py.File(H5, "r") as f:
        genes = np.array([g.decode() for g in f["gene_names"][:]])
        cells = np.array([c.decode() for c in f["sample_names"][:]])
        libsize = np.asarray(f["data/total_exon_counts"][:], dtype=np.float64)
        grp = f["data/exon"]
        p = np.asarray(grp["p"][:], dtype=np.int64)
        n_all = int(grp["dims"][0])
        gi = {g: i for i, g in enumerate(genes)}
        md = meta.reindex(cells)

        def raw(j: int) -> np.ndarray:
            lo, hi = p[j], p[j + 1]
            v = np.zeros(n_all, dtype=np.float64)
            if hi > lo:
                v[np.asarray(grp["i"][lo:hi], dtype=np.int64)] = \
                    np.asarray(grp["x"][lo:hi], dtype=np.float64)
            return v

        per_target = {}
        for target in TARGETS:
            sel = np.array(md["subclass_label"].astype(str) == target, dtype=bool)
            clu = md["cluster_label"].to_numpy()
            vc = pd.Series(clu[sel]).value_counts()
            big = set(vc[vc >= MIN_CLUSTER_N].index)
            sel &= np.array([c in big for c in clu])
            idx = np.flatnonzero(sel)
            if len(idx) < 100:
                print(f"[skip] {target}: only {len(idx)} cells")
                continue
            cats = pd.Series(clu[idx]).astype("category")
            codes = cats.cat.codes.to_numpy()
            ng = len(cats.cat.categories)
            cnt = np.bincount(codes, minlength=ng).astype(float)
            lib = libsize[idx]
            print(f"[{target}] {len(idx)} cells, {ng} clusters "
                  f"(>= {MIN_CLUSTER_N} cells)")

            def wc(v: np.ndarray) -> np.ndarray:
                m = np.bincount(codes, weights=v, minlength=ng) / cnt
                return v - m[codes]

            def logcpm(j: int) -> np.ndarray:
                return np.log1p(raw(j)[idx] / lib * 1e6)

            cols = [g for g in PANELS[STATE_PANEL] if g in gi]
            Z = np.column_stack([logcpm(gi[g]) for g in cols])
            sd = Z.std(0, ddof=1)
            Z = (Z - Z.mean(0)) / np.where(sd > 0, sd, np.nan)
            state = wc(np.nanmean(Z, axis=1))
            sc = state - state.mean()
            sn = np.sqrt((sc ** 2).sum())

            rho = np.full(len(genes), np.nan)
            for j in range(len(genes)):
                lo, hi = p[j], p[j + 1]
                if hi - lo < 5:
                    continue
                v = raw(j)[idx]
                if (v > 0).mean() < MIN_DETECT:
                    continue
                r = wc(np.log1p(v / lib * 1e6))
                r -= r.mean()
                den = np.sqrt((r ** 2).sum()) * sn
                if den > 0:
                    rho[j] = float((r * sc).sum() / den)
            df = pd.DataFrame({"gene": genes, "r": rho}).dropna()
            df["rank_pct"] = 100 * df["r"].rank(pct=True)
            per_target[target] = df.set_index("gene")
            print(f"    {len(df)} genes tested, state sd = {state.std(ddof=1):.3f}")

    combined = pd.DataFrame({t: d["r"] for t, d in per_target.items()})
    combined.to_csv(OUT / "Q_l5pt_targeted_state_axis.csv")

    panel_genes = {g for pl in PANELS.values() for g in pl}
    pt = "L5 PT CTX"
    print("\n" + "=" * 94)
    print(f"TOP {TOP_K} STATE-AXIS GENES IN {pt} (the atlas ET homolog)")
    print("=" * 94)
    cand = (per_target[pt].drop(index=[g for g in panel_genes
                                       if g in per_target[pt].index],
                                errors="ignore")
            .nlargest(TOP_K, "r"))
    other = [t for t in per_target if t != pt]
    show = cand.copy()
    for t in other:
        show[t] = per_target[t]["r"].reindex(show.index)
    show["PT_minus_IT"] = show["r"] - show.get("L5 IT CTX", np.nan)
    print(show.head(30)[["r", "rank_pct"] + other + ["PT_minus_IT"]]
          .to_string(float_format=lambda v: f"{v:+.3f}"))

    print(f"\n--- most PT-SPECIFIC (largest {pt} minus L5 IT) ---")
    spec = show.dropna(subset=["PT_minus_IT"]).nlargest(15, "PT_minus_IT")
    print(spec[["r"] + other + ["PT_minus_IT"]].to_string(
        float_format=lambda v: f"{v:+.3f}"))

    # ---- stage 2: test the nominated set in mouse M1 Patch-seq -------------
    m1p = paths.M1 / "results" / "E_gene_rheobase_adjusted.csv"
    if m1p.exists():
        m1 = (pd.read_csv(m1p).dropna(subset=["p"])
              .drop_duplicates("gene").set_index("gene"))
        nominated = [g for g in cand.index if g in m1.index]
        sub = m1.loc[nominated, ["rho", "p"]].copy()
        sub["q_candidate_set"] = multipletests(sub["p"], method="fdr_bh")[1]
        sub["atlas_r_PT"] = cand["r"].reindex(sub.index)
        sub = sub.sort_values("p")
        sub.to_csv(OUT / "Q_l5pt_candidates_stage2.csv")
        print("\n" + "=" * 94)
        print(f"STAGE 2 — those {len(nominated)} candidates vs rheobase "
              f"(mouse M1 L5, FDR over the nominated set)")
        print("=" * 94)
        print(sub.head(15).to_string(float_format=lambda v: f"{v:.4g}"))
        hits = sub[sub["q_candidate_set"] < 0.05]
        print(f"\n  q<0.05: {len(hits)} of {len(sub)}")
        if len(hits):
            print(hits.to_string(float_format=lambda v: f"{v:.4g}"))
        print("\n  NOTE: exploratory. Genes surfacing here are candidates for")
        print("  testing in the V1 cohort.")
        print(f"\n[write] {OUT/'Q_l5pt_candidates_stage2.csv'}")
    print(f"[write] {OUT/'Q_l5pt_targeted_state_axis.csv'}")


if __name__ == "__main__":
    main()
