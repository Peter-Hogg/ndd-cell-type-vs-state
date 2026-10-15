"""Patch-seq cell-type census, and a genome-wide layer-pooled candidate screen.

Part 1  the full cell-type breakdown: every RNA family and type, how many have a
        rheobase measurement, and the |rho| each population can resolve.

Part 2  the layer-pooled meta-analysis from
        `analysis/m1_patchseq/scripts/20_layer_meta_analysis.py`, run
        genome-wide rather than on six focus genes. Per-layer partial Spearman,
        then fixed- and random-effects pooling with Cochran Q, so a gene is only
        credited with a pooled effect if its effect is actually homogeneous
        across layers. Reported under two denominators: genome-wide, and
        restricted to the atlas-nominated state genes (outcome-blind, so the
        candidate set is the denominator -- see `16_two_stage_candidate_test.py`).

Usage:
  uv run python analysis/cross_dataset/scripts/21_breakdown_and_genomewide_pool.py
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
from statsmodels.stats.multitest import multipletests

_m3 = import_module("03_gene_level_rheobase")
_m20 = import_module("20_layer_meta_analysis")
from plasticity_common import load_ephys, load_meta  # noqa: E402

OUT = paths.results(__file__)
ENDPOINT = "Rheobase (pA)"
MIN_STRATUM_N = 30
MIN_STRATA = 4


def floor_rho(n):
    return float(np.tanh(2.8016 / np.sqrt(n - 3))) if n and n > 4 else np.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    ephys = load_ephys()
    y_all = ephys[ENDPOINT]

    # ---------------- Part 1: census -----------------------------------
    m = meta.copy()
    m["has_rheo"] = m.index.isin(y_all.dropna().index)
    print("=" * 84)
    print("PART 1 — PATCH-SEQ CELL-TYPE CENSUS (mouse M1, Scala et al.)")
    print("=" * 84)
    fam = (m.groupby("RNA family", observed=True)
           .agg(cells=("has_rheo", "size"), with_rheobase=("has_rheo", "sum"))
           .sort_values("with_rheobase", ascending=False))
    fam["detects_rho"] = fam["with_rheobase"].map(floor_rho)
    fam["class"] = ["GABAergic" if f in ("Pvalb", "Sst", "Vip", "Lamp5", "Sncg")
                    else "glutamatergic" if f in ("IT", "ET", "CT", "NP")
                    else "other" for f in fam.index]
    print("\n-- by RNA family --")
    print(fam.to_string(float_format=lambda v: f"{v:.3f}"))

    rt = m["RNA type"].astype(str)
    lay = pd.Series("other", index=m.index)
    for name, pat in _m20.LAYER_PATTERNS.items():
        lay[rt.str.match(pat, na=False)] = name
    m["layer"] = lay
    pyr = (m[m["layer"] != "other"].groupby("layer", observed=True)
           .agg(cells=("has_rheo", "size"), with_rheobase=("has_rheo", "sum"))
           .sort_values("with_rheobase", ascending=False))
    pyr["detects_rho"] = pyr["with_rheobase"].map(floor_rho)
    print("\n-- pyramidal, by layer --")
    print(pyr.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n  pyramidal total with rheobase: {int(pyr['with_rheobase'].sum())}"
          f"  -> pooled floor {floor_rho(int(pyr['with_rheobase'].sum())):.3f}")
    gaba = int(fam.loc[fam['class'] == 'GABAergic', 'with_rheobase'].sum())
    print(f"  GABAergic total with rheobase: {gaba}"
          f"  -> pooled floor {floor_rho(gaba):.3f}")
    print("\n  NB: the best-powered SINGLE populations here are GABAergic")
    print("      (Pvalb, Sst), not pyramidal. L5 ET is among the smallest.")
    fam.to_csv(OUT / "T_census_family.csv")
    pyr.to_csv(OUT / "T_census_layer.csv")

    # ---------------- Part 2: genome-wide layer-pooled screen ----------
    log_cpm, qc = _m3.load_full_logcpm(meta.index, use_introns=True)
    expr_all, _ = _m3.filter_genes(log_cpm, drop_channels=False)
    design_all = meta.loc[expr_all.index].join(qc, how="left")
    design_all["layer"] = lay.reindex(design_all.index)

    print("\n" + "=" * 84)
    print("PART 2 — GENOME-WIDE, POOLED ACROSS LAYERS (homogeneity-aware)")
    print("=" * 84)
    per = {}
    for layer_name in _m20.LAYER_PATTERNS:
        idx = design_all.index[design_all["layer"] == layer_name]
        idx = expr_all.index.intersection(idx)
        y = y_all.reindex(idx)
        if y.notna().sum() < MIN_STRATUM_N:
            print(f"  [skip] {layer_name}: n={y.notna().sum()}")
            continue
        expr = expr_all.loc[idx]
        expr = expr.loc[:, (expr > 0).mean(axis=0) >= 0.25]
        d = design_all.loc[idx]
        cov = d[_m3.COVARS].copy()
        srt = d["RNA type"]
        k = srt.value_counts()[lambda s: s >= 8].index
        srt = srt.where(srt.isin(k), "other")
        if srt.nunique() > 1:
            cov = pd.concat([cov, pd.get_dummies(srt, prefix="t",
                                                 drop_first=True).astype(float)], axis=1)
        res = _m3.screen(expr, y, covars=cov).set_index("gene")
        per[layer_name] = res[["rho", "n"]]
        print(f"  {layer_name:10s} n={int(res['n'].iloc[0]):4d}  "
              f"{len(res):5d} genes  floor {floor_rho(int(res['n'].iloc[0])):.3f}")

    genes = sorted(set().union(*[set(v.index) for v in per.values()]))
    Z = pd.DataFrame({k: v["rho"].reindex(genes) for k, v in per.items()})
    N = pd.DataFrame({k: v["n"].reindex(genes) for k, v in per.items()})
    ok = Z.notna() & (N > 4)
    zz = np.arctanh(Z.clip(-0.999999, 0.999999)).where(ok)
    w = (N - 3).where(ok)
    kk = ok.sum(axis=1)

    sw = w.sum(axis=1)
    z_fe = (w * zz).sum(axis=1) / sw
    se_fe = np.sqrt(1 / sw)
    Q = (w * (zz.sub(z_fe, axis=0)) ** 2).sum(axis=1)
    df = kk - 1
    C = sw - (w ** 2).sum(axis=1) / sw
    tau2 = ((Q - df) / C).clip(lower=0)
    w_re = (1 / (1 / w + tau2.values[:, None])).where(ok)
    z_re = (w_re * zz).sum(axis=1) / w_re.sum(axis=1)
    se_re = np.sqrt(1 / w_re.sum(axis=1))

    R = pd.DataFrame({
        "k_strata": kk, "n_total": N.where(ok).sum(axis=1).astype("Int64"),
        "rho_FE": np.tanh(z_fe), "p_FE": 2 * stats.norm.sf(np.abs(z_fe / se_fe)),
        "rho_RE": np.tanh(z_re), "p_RE": 2 * stats.norm.sf(np.abs(z_re / se_re)),
        "Q": Q, "p_Q": stats.chi2.sf(Q, df),
        "I2": ((Q - df) / Q).clip(lower=0) * 100,
    })
    R = R[R["k_strata"] >= MIN_STRATA].dropna(subset=["p_RE"])
    R["q_RE_genomewide"] = multipletests(R["p_RE"], method="fdr_bh")[1]

    atlas_p = paths.ATLAS / "results" / "O_state_axis_gene_correlations.csv"
    if atlas_p.exists():
        atlas = pd.read_csv(atlas_p).dropna(subset=["r_within"]).drop_duplicates("gene")
        cand = set(atlas.nlargest(300, "r_within")["gene"])
        inc = R.index.isin(cand)
        R["in_atlas_top300"] = inc
        R["q_RE_atlas_set"] = np.nan
        if inc.sum() > 5:
            R.loc[inc, "q_RE_atlas_set"] = multipletests(
                R.loc[inc, "p_RE"], method="fdr_bh")[1]

    R.sort_values("p_RE").to_csv(OUT / "T_genomewide_layerpooled.csv")
    print(f"\n  {len(R)} genes with >= {MIN_STRATA} usable strata, "
          f"pooled n up to {int(R['n_total'].max())}")

    print("\n-- top 20 by pooled RE p (I² = heterogeneity; low is poolable) --")
    top = R.sort_values("p_RE").head(20)
    print(top[["k_strata", "n_total", "rho_RE", "p_RE", "I2",
               "q_RE_genomewide"]].to_string(float_format=lambda v: f"{v:.4g}"))

    print("\n-- HOMOGENEOUS hits (I² < 25%) ranked by pooled p --")
    hom = R[(R["I2"] < 25)].sort_values("p_RE").head(20)
    print(hom[["n_total", "rho_RE", "p_RE", "I2", "q_RE_genomewide"]
              + (["q_RE_atlas_set"] if "q_RE_atlas_set" in R.columns else [])]
          .to_string(float_format=lambda v: f"{v:.4g}"))

    print(f"\n  genome-wide q<0.05: {int((R['q_RE_genomewide'] < 0.05).sum())}")
    if "q_RE_atlas_set" in R.columns:
        s = R[R["q_RE_atlas_set"] < 0.05].sort_values("p_RE")
        print(f"  atlas-candidate-set q<0.05: {len(s)}")
        if len(s):
            print(s[["n_total", "rho_RE", "p_RE", "I2", "q_RE_atlas_set"]]
                  .to_string(float_format=lambda v: f"{v:.4g}"))
    print(f"\n[write] {OUT/'T_genomewide_layerpooled.csv'}")


if __name__ == "__main__":
    main()
