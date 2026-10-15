"""Seven functional gene categories, tested as SETS, layer by layer.

The 128-gene panel is organised into seven functional groups: seven
category-level tests instead of 128 gene-level ones, each a coherent biological
hypothesis, and a multiple-testing burden small enough for n=49-84 per layer.

Method as in `22_repairs_excitatory.py`: partial Spearman within transcriptomic
type with technical covariates, then a **phenotype-permutation null** (rheobase
shuffled within type) so the co-expression structure of each category is
preserved. An expression-matched null is not used because it is biased.

Hypothesis for L2/3: learning drives spine turnover in L2/3, and the
structural/cytoskeletal category is the one that should track that. L2/3 IT is
also the largest pyramidal population here (n=84).

Excitatory only; Pvalb is carried as a specificity control.

Usage:  uv run python analysis/m1_patchseq/scripts/23_functional_categories_by_layer.py
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
_m20 = import_module("20_layer_meta_analysis")
_m22 = import_module("22_repairs_excitatory")
from plasticity_common import load_ephys, load_meta  # noqa: E402

OUT = paths.results(__file__)
ENDPOINT = "Rheobase (pA)"
N_PERM = 3000
RNG = np.random.default_rng(0)

RAW_CATEGORIES: dict[str, str] = {
    "IEG_TF": """FOS FOSB FOSL1 FOSL2 JUN JUNB JUND EGR1 EGR2 EGR3 EGR4 NPAS4
        KLF2 KLF4 KLF6 KLF9 KLF10 KLF11 ATF3 ATF4 ATF5 ATF7 CREM CREB1 CREB3
        CREB5 SRF ELK1 ELK3 ELK4 NR4A1 NR4A2 NR4A3 BTG1 BTG2 IER2 IER3 IER5""",
    "synaptic_plasticity": """ARC HOMER1 HOMER1A BDNF NPTX2 NPTX1 NPTXR SYT4
        CAMK2G CAMK1G CAMK4 MAPK8IP2 MAPK1IP1L RGS2 RGS4 RGS14 DUSP1 DUSP4
        DUSP5 DUSP6 PPP1R1B PPP1R15A PLK2 PLK3 PRKCZ PRKCI RAPGEF4 RAPGEF5
        RAPGEF3""",
    "epigenetic": """GADD45A GADD45B GADD45G DNMT3A DNMT3B MECP2 HDAC4 HDAC5
        HDAC9""",
    "ca_camp_adaptors": """RCAN1 RCAN2 RCAN3 CALN1 CALM1 CALM2 CALM3 CAMKK1
        CAMKK2 AKAP5 AKAP6 AKAP11 AKAP12 AKAP13""",
    "stress_injury_ARG": """HSPB1 HSPA1A HSPA1B DDIT3 CHAC1 TRIB1 TRIB2 TRIB3
        SGK1 SGK3 SERTAD1 SERTAD2 SERTAD3""",
    "structural_late": """BAIAP2 BAIAP3 SIPA1L1 SIPA1L2 SYNPO SYNPO2 SYNGAP1
        DLG4 DLG3 SHANK1 SHANK2 SHANK3""",
    "mapk_rho_cytoskeletal": """RHOA RHOB RHOC RHOU RND1 RND2 RND3 MAPK8 MAPK10
        MAPT RASGRF1 RASGRF2 RAP1GAP RAP1GDS1 PAK1 PAK3""",
}


def to_mouse(sym: str) -> str:
    """Human-cased symbol -> mouse casing, annotations stripped."""
    s = re.sub(r"\(.*?\)", "", sym).strip()
    return s[0] + s[1:].lower() if s else s


CATEGORIES = {k: sorted({to_mouse(g) for g in v.split()})
              for k, v in RAW_CATEGORIES.items()}


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

    print("category membership found in the Patch-seq matrix:")
    for k, v in CATEGORIES.items():
        found = [g for g in v if g in expr_all.columns]
        print(f"  {k:24s} {len(found):3d}/{len(v):3d}")

    pops = {}
    for lname in ["L2/3 IT", "L4/5 IT", "L5 IT", "L5 ET", "L6 CT", "L6 IT"]:
        pops[lname] = design.index[design["layer"] == lname]
    pops["L5 exc (IT+ET)"] = design.index[design["layer"].isin(["L5 IT", "L5 ET"])]
    pops["Pvalb (control)"] = design.index[design["RNA family"] == "Pvalb"]

    rows = []
    for pop, idx in pops.items():
        idx = expr_all.index.intersection(idx)
        y = y_all.reindex(idx)
        idx = idx[y.notna().values]
        if len(idx) < 30:
            continue
        d = design.loc[idx]
        n = len(idx)
        cov = d[_m3.COVARS].copy()
        srt = d["RNA type"]
        k = srt.value_counts()[lambda s: s >= 8].index
        srt = srt.where(srt.isin(k), "other")
        dums = pd.get_dummies(srt, prefix="t", drop_first=True).astype(float)
        C = _m22.build_design(cov, dums, n)
        yr = _m22.rank_resid(y.loc[idx].to_numpy(), C)
        groups = srt.to_numpy()

        # pre-generate the permutation set once per population so every
        # category is tested against the identical null draws
        perms = []
        for _ in range(N_PERM):
            yp = yr.copy()
            for g in np.unique(groups):
                m = groups == g
                yp[m] = RNG.permutation(yp[m])
            perms.append(yp - yp.mean())
        perms = np.column_stack(perms)
        perms /= np.where(perms.std(0) > 0, perms.std(0), np.nan)
        yc = (yr - yr.mean()) / yr.std()

        print(f"\n{'='*88}\n{pop}   n={n}   floor |rho|>={np.tanh(2.8016/np.sqrt(n-3)):.3f}\n{'='*88}")
        print(f"  {'category':24s} {'g':>3s} {'mean rho':>9s} {'null':>16s} "
              f"{'z':>7s} {'p':>7s}")
        for cat, genes in CATEGORIES.items():
            cols = [g for g in genes if g in expr_all.columns]
            E = expr_all.loc[idx, cols]
            E = E.loc[:, (E > 0).mean(axis=0) >= 0.25]
            if E.shape[1] < 5:
                continue
            Er = np.column_stack([_m22.rank_resid(E[g].to_numpy(), C)
                                  for g in E.columns])
            Er -= Er.mean(0)
            sd = Er.std(0)
            Er /= np.where(sd > 0, sd, np.nan)
            obs = float(np.nanmean((Er * yc[:, None]).sum(0) / (n - 1)))
            nulls = np.nanmean((Er.T @ perms) / (n - 1), axis=0)
            z = (obs - nulls.mean()) / nulls.std(ddof=1)
            p_two = float(2 * min((nulls <= obs).mean(), (nulls >= obs).mean()))
            rows.append({"population": pop, "n": n, "category": cat,
                         "n_genes": E.shape[1], "mean_rho": obs,
                         "null_mean": float(nulls.mean()),
                         "null_sd": float(nulls.std(ddof=1)), "z": z,
                         "p_two_sided": p_two})
            print(f"  {cat:24s} {E.shape[1]:3d} {obs:+9.4f} "
                  f"{nulls.mean():+8.4f}±{nulls.std(ddof=1):.4f} {z:+7.2f} "
                  f"{p_two:7.4f}")

    R = pd.DataFrame(rows)
    exc = R[~R["population"].str.contains("control")].copy()
    exc["q_bh"] = multipletests(exc["p_two_sided"], method="fdr_bh")[1]
    R = R.merge(exc[["population", "category", "q_bh"]],
                on=["population", "category"], how="left")
    R.to_csv(OUT / "V_functional_categories_by_layer.csv", index=False)

    print("\n" + "=" * 88)
    print("z MATRIX  (negative = category tracks LOWER rheobase / more excitable)")
    print("=" * 88)
    piv = R.pivot(index="category", columns="population", values="z")
    order = [c for c in ["L2/3 IT", "L4/5 IT", "L5 IT", "L5 ET", "L5 exc (IT+ET)",
                         "L6 CT", "L6 IT", "Pvalb (control)"] if c in piv.columns]
    print(piv[order].to_string(float_format=lambda v: f"{v:+.2f}"))

    print(f"\nBH-corrected across the {len(exc)} excitatory tests "
          f"(controls excluded):")
    sig = R[(R["q_bh"] < 0.10)].sort_values("p_two_sided")
    if len(sig):
        print(sig[["population", "category", "n", "n_genes", "mean_rho",
                   "z", "p_two_sided", "q_bh"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    else:
        print("  nothing at q<0.10")
    print(f"\n[write] {OUT/'V_functional_categories_by_layer.csv'}")


if __name__ == "__main__":
    main()
