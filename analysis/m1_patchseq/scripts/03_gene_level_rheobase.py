"""Genome-wide gene-vs-rheobase screen in L5 excitatory cells, plus the
co-expression module around the hits.

Averaging a 20-128 gene panel dilutes a signal carried by a subset of genes
when much of the panel is uncorrelated or anti-correlated with rheobase. So
screen per gene, then ask what co-expresses with the hits.

Three deliberate choices, all reversible via flags:

* Ion channel genes are excluded from the primary screen (--keep-channels to
  disable). Not because the relationship is circular -- Kv/Nav expression
  driving rheobase is a mechanism, not a tautology -- but because it is
  uninformative. They come back as a covariate in the sensitivity analysis.
* ERCC spike-ins and Gm*/Rik model genes are dropped (85 ERCC and 18,194 Gm*
  entries in the 42,466-gene index).
* Sequencing depth, genes detected and ex vivo hold time are partialled out,
  because the composite score tracks genes detected.

Everything is FDR-corrected genome-wide. Effect sizes for genes selected *by*
this screen are upward-biased (winner's curse) -- they are hypothesis-generating
and need replication in an independent dataset.

Usage:  uv run python analysis/m1_patchseq/scripts/03_gene_level_rheobase.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)

import argparse
import re

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from plasticity_common import (
    CORE_IEG,
    EXPANDED_IEG,
    EXON,
    INTRON,
    l5_excitatory_ids,
    load_ephys,
    load_meta,
)

RESULTS = paths.results(__file__)
TARGET = "Rheobase (pA)"
MIN_DETECT_FRAC = 0.25   # gene must be detected in >= 25% of the L5 cells
COVARS = ["log_libsize", "n_detected", "Hold Time (min)"]

# Voltage-, ligand- and calcium-gated channels plus their auxiliary subunits.
# Prefix-matched, so this is intentionally broad.
CHANNEL_PATTERNS = (
    r"^Kcn", r"^Scn", r"^Cacn", r"^Hcn", r"^Clcn", r"^Trp[acmvp]", r"^Ano",
    r"^Asic", r"^Nalcn", r"^Ryr", r"^Itpr", r"^P2rx", r"^Gabr", r"^Grin",
    r"^Gria", r"^Grik", r"^Grm", r"^Chrn", r"^Htr3", r"^Glr[ab]", r"^Cnga",
    r"^Cngb", r"^Kcnip", r"^Dpp6", r"^Dpp10", r"^Fgf1[1-4]$", r"^Sh3kbp1$",
)

# Rapid vs delayed primary response genes (Tyssowski et al. 2018, Neuron).
# The paper's claim is that the delayed effector programme, not the rapid
# transcription-factor programme, is what tracks intrinsic excitability.
RAPID_PRG = [
    "Fos", "Fosb", "Jun", "Junb", "Jund", "Egr1", "Egr2", "Egr3", "Egr4",
    "Npas4", "Arc", "Ier2", "Btg2", "Atf3", "Nr4a1",
]
DELAYED_PRG = [
    "Nptx1", "Nptx2", "Nptxr", "Homer1", "Bdnf", "Syt4", "Rgs4", "Gadd45b",
    "Gadd45g", "Sgk1", "Crem", "Nr4a2", "Nr4a3", "Pcsk1", "Vgf", "Scg2",
]


def load_full_logcpm(cells: pd.Index, use_introns: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full genes x cells matrix as cells x genes log-CPM, restricted to ``cells``.

    Library size and detected-gene counts are computed over the *whole* gene
    index before any filtering, so normalisation is not affected by the
    downstream gene selection.
    """
    print("[expr] reading exon counts ...")
    exon = pd.read_csv(EXON, index_col=0)
    exon.index = exon.index.astype(str).str.strip()
    if use_introns:
        print("[expr] reading intron counts ...")
        intron = pd.read_csv(INTRON, index_col=0)
        intron.index = intron.index.astype(str).str.strip()
        counts = exon.add(intron.reindex_like(exon).fillna(0), fill_value=0)
        del intron
    else:
        counts = exon
    del exon

    counts.columns = counts.columns.astype(str).str.strip()
    libsize = counts.sum(axis=0)
    detected = (counts > 0).sum(axis=0)
    ercc = counts.loc[counts.index.str.startswith("ERCC-")].sum(axis=0)

    qc = pd.DataFrame({
        "libsize": libsize,
        "n_detected": detected,
        "ercc_frac": ercc / libsize,
    })
    qc["log_libsize"] = np.log10(qc["libsize"].replace(0, np.nan))

    keep_cells = [c for c in counts.columns if c in set(cells)]
    counts = counts[keep_cells]
    cpm = counts.div(libsize[keep_cells], axis=1) * 1e6
    log_cpm = np.log1p(cpm).T.astype(np.float32)
    print(f"[expr] matrix: {log_cpm.shape[0]} cells x {log_cpm.shape[1]} genes")
    return log_cpm, qc


def filter_genes(log_cpm: pd.DataFrame, drop_channels: bool) -> tuple[pd.DataFrame, dict]:
    g = log_cpm.columns
    audit = {"start": len(g)}

    is_ercc = g.str.startswith("ERCC-")
    is_model = g.str.match(r"^(Gm\d|.*Rik$|LOC\d|A[A-Z]\d{6}|BC\d{6})")
    detected_frac = (log_cpm > 0).mean(axis=0)
    too_sparse = detected_frac < MIN_DETECT_FRAC

    drop = is_ercc | is_model | too_sparse.to_numpy()
    audit.update({
        "ercc": int(is_ercc.sum()),
        "model_genes": int(is_model.sum()),
        f"detected_lt_{int(MIN_DETECT_FRAC*100)}pct": int(too_sparse.sum()),
    })

    if drop_channels:
        is_chan = g.str.match("|".join(CHANNEL_PATTERNS))
        audit["ion_channels"] = int((is_chan & ~drop).sum())
        drop = drop | is_chan

    out = log_cpm.loc[:, ~drop]
    audit["retained"] = out.shape[1]
    return out, audit


def _rank(a: np.ndarray) -> np.ndarray:
    return stats.rankdata(a, axis=0)


def screen(expr: pd.DataFrame, y: pd.Series, covars: pd.DataFrame | None) -> pd.DataFrame:
    """Vectorised Spearman of every gene against y, optionally covariate-adjusted.

    Rank-transform, residualise on the covariate ranks, then Pearson on the
    residuals -- identical to a partial Spearman, but done as one matrix
    operation over ~20k genes rather than a Python loop.
    """
    ok = y.notna()
    if covars is not None:
        ok &= covars.notna().all(axis=1)
    E = expr.loc[ok].to_numpy(dtype=np.float64)
    yv = y.loc[ok].to_numpy(dtype=np.float64)
    n = len(yv)

    Er = _rank(E)
    yr = stats.rankdata(yv)

    if covars is not None:
        C = np.column_stack([np.ones(n)] + [stats.rankdata(covars.loc[ok, c]) for c in covars])
        # Residualise both sides on the covariate design in one solve.
        beta_E, *_ = np.linalg.lstsq(C, Er, rcond=None)
        Er = Er - C @ beta_E
        beta_y, *_ = np.linalg.lstsq(C, yr, rcond=None)
        yr = yr - C @ beta_y
        dof = n - 2 - covars.shape[1]
    else:
        dof = n - 2

    Ec = Er - Er.mean(axis=0)
    yc = yr - yr.mean()
    denom = np.sqrt((Ec**2).sum(axis=0) * (yc**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (Ec * yc[:, None]).sum(axis=0) / denom
    rho = np.clip(np.nan_to_num(rho), -0.999999, 0.999999)
    t = rho * np.sqrt(dof / (1 - rho**2))
    p = 2 * stats.t.sf(np.abs(t), dof)

    out = pd.DataFrame({"gene": expr.columns, "rho": rho, "p": p, "n": n})
    out["q"] = multipletests(out["p"], method="fdr_bh")[1]
    return out.sort_values("rho").reset_index(drop=True)


def annotate_programme(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["programme"] = "other"
    df.loc[df["gene"].isin(EXPANDED_IEG), "programme"] = "panel (other)"
    df.loc[df["gene"].isin(DELAYED_PRG), "programme"] = "delayed PRG"
    df.loc[df["gene"].isin(RAPID_PRG), "programme"] = "rapid PRG"
    return df


def main(drop_channels: bool) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("Genome-wide gene vs rheobase screen, L5 excitatory")
    print("=" * 74)

    meta = load_meta()
    l5 = l5_excitatory_ids(meta)
    ephys = load_ephys()

    log_cpm, qc = load_full_logcpm(l5)
    expr, audit = filter_genes(log_cpm, drop_channels)
    print("[filter]", " | ".join(f"{k}={v}" for k, v in audit.items()))

    design = meta.loc[expr.index].join(qc, how="left")
    y = ephys[TARGET].reindex(expr.index)
    cov = design[COVARS]

    raw = annotate_programme(screen(expr, y, covars=None))
    adj = annotate_programme(screen(expr, y, covars=cov))

    raw.to_csv(RESULTS / "E_gene_rheobase_raw.csv", index=False)
    adj.to_csv(RESULTS / "E_gene_rheobase_adjusted.csv", index=False)
    print(f"[write] {RESULTS/'E_gene_rheobase_raw.csv'}")
    print(f"[write] {RESULTS/'E_gene_rheobase_adjusted.csv'}")

    print(f"\ngenes tested: {len(raw)}  |  n cells: {int(raw['n'].iloc[0])}")
    print(f"FDR q<0.05  raw: {int((raw['q']<0.05).sum())}   "
          f"depth+holdtime-adjusted: {int((adj['q']<0.05).sum())}")

    print("\n--- 25 strongest negative correlations with rheobase (raw) ---")
    cols = ["gene", "rho", "p", "q", "programme"]
    print(raw.head(25)[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print("\n--- flagged genes, raw vs adjusted ---")
    flagged = ["Gadd45g", "Egr1", "Nptx2", "Nptx1", "Arc", "Fos", "Npas4",
               "Jun", "Junb", "Homer1", "Bdnf", "Syt4", "Sgk1", "Rgs4"]
    m = (raw.set_index("gene").reindex(flagged)[["rho", "p", "q"]]
         .join(adj.set_index("gene").reindex(flagged)[["rho", "p", "q"]],
               lsuffix="_raw", rsuffix="_adj"))
    m["rank_raw"] = [int((raw["rho"] < raw.set_index("gene")["rho"].get(g, np.nan)).sum()) + 1
                     if g in set(raw["gene"]) else -1 for g in flagged]
    print(m.to_string(float_format=lambda v: f"{v:.4g}"))

    print("\n--- rapid vs delayed primary response genes ---")
    for label in ("rapid PRG", "delayed PRG"):
        for tbl, tag in ((raw, "raw"), (adj, "adj")):
            s = tbl[tbl["programme"] == label]
            if len(s):
                print(f"  {label:12s} [{tag}] n={len(s):2d}  "
                      f"mean rho={s['rho'].mean():+.3f}  "
                      f"median rho={s['rho'].median():+.3f}  "
                      f"min={s['rho'].min():+.3f}  q<0.05: {int((s['q']<0.05).sum())}")
    rp = raw[raw["programme"] == "rapid PRG"]["rho"]
    dp = raw[raw["programme"] == "delayed PRG"]["rho"]
    if len(rp) > 2 and len(dp) > 2:
        u, pu = stats.mannwhitneyu(rp, dp)
        print(f"  rapid vs delayed rho distributions: Mann-Whitney p = {pu:.4f}")

    # ---- co-expression module around the hits ---------------------------
    print("\n--- co-expression module around the rheobase-correlated hits ---")
    hits = adj[adj["q"] < 0.05]
    seed_source = "adjusted q<0.05"
    if len(hits) < 10:
        hits = raw.reindex(raw["rho"].sort_values().index).head(150)
        seed_source = "150 most negative (raw), adjusted screen too sparse"
    seeds = [g for g in hits["gene"] if g in expr.columns]
    print(f"  seeds: {len(seeds)} genes ({seed_source})")

    seed_expr = expr[seeds]
    seed_score = ((seed_expr - seed_expr.mean()) / seed_expr.std(ddof=1)).mean(axis=1)
    module = screen(expr, seed_score, covars=cov)
    module = annotate_programme(module).sort_values("rho", ascending=False)
    module.to_csv(RESULTS / "F_module_coexpression.csv", index=False)
    print(f"[write] {RESULTS/'F_module_coexpression.csv'}")
    print(f"  genes co-expressed with the seed module at q<0.05: "
          f"{int((module['q']<0.05).sum())}")
    print("\n  top 30 module members (excluding the seeds themselves):")
    non_seed = module[~module["gene"].isin(seeds)]
    print(non_seed.head(30)[["gene", "rho", "q", "programme"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    seed_df = pd.DataFrame({"gene": seeds, "seed": True})
    seed_df.to_csv(RESULTS / "F_module_seeds.csv", index=False)
    print(f"[write] {RESULTS/'F_module_seeds.csv'}")

    # ---- sensitivity: do channels explain the module? -------------------
    if drop_channels:
        chan = log_cpm.loc[:, log_cpm.columns.str.match("|".join(CHANNEL_PATTERNS))]
        chan = chan.loc[:, (chan > 0).mean(axis=0) >= MIN_DETECT_FRAC]
        if chan.shape[1] >= 3:
            from numpy.linalg import svd
            Z = ((chan - chan.mean()) / chan.std(ddof=1)).fillna(0).to_numpy()
            pcs = svd(Z - Z.mean(axis=0), full_matrices=False)[0][:, :3]
            cov2 = cov.copy()
            for i in range(3):
                cov2[f"chanPC{i+1}"] = pcs[:, i]
            s_score = pd.Series(seed_score, index=expr.index)
            r0, p0 = stats.spearmanr(*pd.concat([s_score, y], axis=1).dropna().to_numpy().T)
            adj2 = screen(pd.DataFrame({"module": s_score}), y, covars=cov2)
            print(f"\n[sensitivity] module score vs rheobase")
            print(f"  raw                                    rho = {r0:+.3f}, p = {p0:.3g}")
            print(f"  + depth/holdtime                       "
                  f"rho = {float(screen(pd.DataFrame({'m': s_score}), y, cov)['rho'].iloc[0]):+.3f}")
            print(f"  + depth/holdtime + 3 ion-channel PCs   "
                  f"rho = {float(adj2['rho'].iloc[0]):+.3f}, "
                  f"p = {float(adj2['p'].iloc[0]):.3g}")
            print(f"  ({chan.shape[1]} channel genes summarised as 3 PCs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-channels", action="store_true",
                    help="do not exclude ion channel genes from the screen")
    a = ap.parse_args()
    main(drop_channels=not a.keep_channels)
