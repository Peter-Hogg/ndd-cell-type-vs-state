"""Two-stage design: select candidates outcome-blind, then test in Patch-seq.

Stage 1 uses the RNA-seq atlas -- which contains no electrophysiology -- to
nominate candidate genes; stage 2 tests only those against rheobase in the
mouse M1 Patch-seq cohort. Because the selection never sees rheobase, the
multiple-testing correction in stage 2 is taken over the candidate set rather
than the genome.

Three correction framings are reported side by side so the effect of the
denominator is explicit:

  genome-wide          all tested genes
  curated IEG panel    the curated plasticity gene list (outcome-blind)
  atlas state axis     top-K genes on the atlas within-cluster state axis
                       (outcome-blind, from a different dataset with no ephys)

Caveat printed in the output: Jun's Patch-seq association was already known, so
for Jun specifically the atlas-preselected test is not blind.

Usage:  uv run python analysis/cross_dataset/scripts/16_two_stage_candidate_test.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)


import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from plasticity_common import CORE_IEG, EXPANDED_IEG  # noqa: E402

RESULTS = paths.results(__file__)
TOP_K = 150
FOCUS = ["Jun", "Junb", "Jund", "Gadd45g", "Egr1", "Fos", "Npas4", "Arc",
         "Nptx1", "Nptx2", "Sipa1l1", "Sipa1l2", "Homer1", "Bdnf", "Kcnip1"]


def bh(p: pd.Series) -> pd.Series:
    ok = p.notna()
    out = pd.Series(np.nan, index=p.index)
    if ok.any():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def main() -> None:
    m1 = pd.read_csv(paths.M1 / "results" / "E_gene_rheobase_adjusted.csv")
    m1 = m1.dropna(subset=["p"]).drop_duplicates("gene").set_index("gene")
    print(f"[stage 2] mouse M1 L5 excitatory, covariate-adjusted: "
          f"{len(m1)} genes tested, n = {int(m1['n'].iloc[0])} cells")

    atlas = pd.read_csv(paths.ATLAS / "results" / "O_state_axis_gene_correlations.csv")
    atlas = atlas.dropna(subset=["r_within"]).drop_duplicates("gene")
    print(f"[stage 1] atlas within-cluster state axis: {len(atlas)} genes ranked "
          f"(no electrophysiology involved)")

    panel = [g for g in EXPANDED_IEG if g in m1.index]
    top = atlas.nlargest(TOP_K, "r_within")["gene"].tolist()
    top_in = [g for g in top if g in m1.index]

    frames = {
        "genome_wide": m1.index.tolist(),
        "curated_ieg_panel": panel,
        f"atlas_state_top{TOP_K}": top_in,
    }
    print(f"\ncandidate-set sizes: "
          + ", ".join(f"{k}={len(v)}" for k, v in frames.items()))

    q = pd.DataFrame(index=m1.index)
    for name, genes in frames.items():
        s = pd.Series(np.nan, index=m1.index)
        s.loc[genes] = bh(m1.loc[genes, "p"]).values
        q[name] = s

    tbl = m1[["rho", "p", "n"]].join(q)
    tbl.to_csv(RESULTS / "P_two_stage_candidate_q.csv")

    print("\n" + "=" * 96)
    print("SAME p-VALUES, THREE DENOMINATORS  (mouse M1 L5, rheobase, adjusted)")
    print("=" * 96)
    hdr = (f"  {'gene':10s} {'rho':>7s} {'p':>10s}"
           f" {'q genome':>10s} {'q IEG panel':>12s} {'q atlas top':>12s}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for g in FOCUS:
        if g not in tbl.index:
            print(f"  {g:10s}  not tested (filtered out)")
            continue
        r = tbl.loc[g]
        def f(v):
            return "     -    " if not np.isfinite(v) else f"{v:10.4f}"
        star = ""
        if np.isfinite(r["curated_ieg_panel"]) and r["curated_ieg_panel"] < 0.05:
            star += " PANEL-SIG"
        if np.isfinite(r[f"atlas_state_top{TOP_K}"]) and r[f"atlas_state_top{TOP_K}"] < 0.05:
            star += " ATLAS-SIG"
        print(f"  {g:10s} {r['rho']:+7.3f} {r['p']:10.4g}"
              f" {f(r['genome_wide'])} {f(r['curated_ieg_panel']):>12s}"
              f" {f(r[f'atlas_state_top{TOP_K}']):>12s}{star}")

    for name in ("curated_ieg_panel", f"atlas_state_top{TOP_K}"):
        sig = tbl[tbl[name] < 0.05].sort_values("rho")
        print(f"\n--- ALL hits at q<0.05 under '{name}' "
              f"({len(frames[name])} candidates) ---")
        if len(sig):
            print(sig[["rho", "p", name]].to_string(
                float_format=lambda v: f"{v:.4g}"))
        else:
            print("  none")

    print("\n--- where the focus genes sit on the ATLAS state axis (stage 1) ---")
    a = atlas.set_index("gene")
    for g in FOCUS:
        if g in a.index:
            r = a.loc[g]
            print(f"  {g:10s} atlas r_within {r['r_within']:+.3f}"
                  f"   rank pct {r['rank_pct']:5.1f}"
                  f"   {'IN top-K' if g in top else ''}")

    print("\n" + "!" * 96)
    print("NOTE: Jun's Patch-seq association was already known, so for Jun the")
    print("atlas-preselected framing is NOT blind. The curated IEG panel is taken")
    print("from the literature, independent of any rheobase test. Genes newly")
    print("surfacing here are candidates for the V1 cohort, not established hits.")
    print("!" * 96)
    print(f"\n[write] {RESULTS/'P_two_stage_candidate_q.csv'}")


if __name__ == "__main__":
    main()
