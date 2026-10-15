"""Shared loading / scoring helpers for the Patch-seq plasticity analysis.

Deliberately dependency-light and deterministic: every function takes explicit
inputs and returns plain pandas objects, so results can be re-derived top to
bottom without any hidden kernel state.

Dataset: Scala et al. 2021 mouse MOp Patch-seq (GEO GSE163764 / DANDI:000008).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from paths import REPO

# ---------------------------------------------------------------- paths

META = str(REPO / "PatchSeqExpression/m1_patchseq_meta_data.csv")
EXON = str(REPO / "PatchSeqExpression/GSE163764_m1_patchseq_exon_counts.csv")
INTRON = str(REPO / "PatchSeqExpression/GSE163764_m1_patchseq_intron_counts.csv")
EPHYS = str(REPO / "m1_patchseq_ephys_features.csv")

# ---------------------------------------------------------------- gene sets

# 20-gene core IEG set (as in Gene_Expression/Figure1.ipynb).
CORE_IEG = [
    "Fos", "Fosb", "Junb", "Egr1", "Egr2", "Npas4", "Arc", "Nr4a1", "Nr4a2",
    "Bdnf", "Nptx2", "Syt4", "Camk2g", "Rgs4", "Dusp1", "Atf3", "Crem",
    "Gadd45b", "Gadd45g", "Prkcz",
]

# Expanded activity/plasticity programme (mouse gene symbols).
EXPANDED_IEG = [
    "Fos", "Fosb", "Fosl1", "Fosl2", "Jun", "Junb", "Jund",
    "Egr1", "Egr2", "Egr3", "Egr4", "Npas4",
    "Klf2", "Klf4", "Klf6", "Klf9", "Klf10", "Klf11",
    "Atf3", "Atf4", "Atf5", "Atf7", "Crem", "Creb1", "Creb3", "Creb5",
    "Srf", "Elk1", "Elk3", "Elk4", "Nr4a1", "Nr4a2", "Nr4a3",
    "Btg1", "Btg2", "Ier2", "Ier3", "Ier5",
    "Arc", "Homer1", "Bdnf", "Nptx2", "Nptx1", "Nptxr", "Syt4",
    "Camk2g", "Camk1g", "Camk4", "Mapk8ip2", "Mapk1ip1l",
    "Rgs2", "Rgs4", "Rgs14", "Dusp1", "Dusp4", "Dusp5", "Dusp6",
    "Ppp1r1b", "Ppp1r15a", "Plk2", "Plk3", "Prkcz", "Prkci",
    "Rapgef4", "Rapgef5", "Rapgef3",
    "Gadd45a", "Gadd45b", "Gadd45g", "Dnmt3a", "Dnmt3b", "Mecp2",
    "Hdac4", "Hdac5", "Hdac9",
    "Rcan1", "Rcan2", "Rcan3", "Caln1", "Calm1", "Calm2", "Calm3",
    "Camkk1", "Camkk2", "Akap5", "Akap6", "Akap11", "Akap12", "Akap13",
    "Hspb1", "Hspa1a", "Hspa1b", "Ddit3", "Chac1",
    "Trib1", "Trib2", "Trib3", "Sgk1", "Sgk3",
    "Sertad1", "Sertad2", "Sertad3",
    "Baiap2", "Baiap3", "Sipa1l1", "Sipa1l2", "Synpo", "Synpo2", "Syngap1",
    "Dlg4", "Dlg3", "Shank1", "Shank2", "Shank3",
    "Rhoa", "Rhob", "Rhoc", "Rhou", "Rnd1", "Rnd2", "Rnd3",
    "Mapk8", "Mapk10", "Mapt", "Rasgrf1",
]

# Ephys features carried through the primary analysis. The same panel is used
# for the cross-population confound analysis and the within-type test.
PRIMARY_EPHYS = [
    "Input resistance (MOhm)",
    "AP threshold (mV)",
    "AP width (ms)",
    "Resting membrane potential (mV)",
    "ISI adaptation index",
    "Spike frequency adaptation",
    "Rheobase (pA)",
]

# ---------------------------------------------------------------- metadata


def load_meta(path: str = META, apply_qc: bool = True) -> pd.DataFrame:
    """Load Patch-seq metadata, indexed by cell id.

    With ``apply_qc`` (default) drops the 97 cells the authors label
    ``RNA family == "low quality"`` and the 7 cells carrying an entry in
    ``Exclusion reasons`` (transcriptome/physiology mismatches and one cell
    outside M1). Both exclusions come from the original authors, not from us.
    """
    meta = pd.read_csv(path, sep=None, engine="python")
    meta["Cell"] = meta["Cell"].astype(str).str.strip()
    for col in ("RNA family", "RNA type"):
        meta[col] = meta[col].astype(str).str.strip()
    meta = meta.set_index("Cell")

    if apply_qc:
        n0 = len(meta)
        meta = meta[meta["RNA family"] != "low quality"]
        n_lq = n0 - len(meta)
        n1 = len(meta)
        meta = meta[meta["Exclusion reasons"].isna()]
        print(
            f"[meta] {n0} cells -> dropped {n_lq} low-quality, "
            f"{n1 - len(meta)} flagged -> {len(meta)} retained"
        )
    return meta


def l5_excitatory_ids(meta: pd.DataFrame) -> pd.Index:
    """Cell ids for L5 excitatory transcriptomic types (``L5 IT_*``/``L5 ET_*``).

    This is the primary analysis population and the level at which the
    type-vs-state question is actually posed.
    """
    mask = meta["RNA type"].str.match(r"^L5 (IT|ET)_\d+$", na=False)
    return meta.index[mask]


# ---------------------------------------------------------------- expression


def _sum_counts_chunked(
    path: str, genes: list[str], chunksize: int = 4000
) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """Stream a genes x cells counts CSV once.

    Returns ``(libsize, gene_counts, ercc_counts, n_detected)`` where
    ``gene_counts`` is a genes x cells frame restricted to ``genes``. Streaming
    keeps peak memory low and, more importantly, means library size is computed
    over the *whole* index rather than whatever subset happens to be in memory.
    """
    wanted = set(genes)
    libsize = None
    ercc = None
    detected = None
    collected: list[pd.DataFrame] = []

    for chunk in pd.read_csv(path, index_col=0, chunksize=chunksize):
        chunk.index = chunk.index.astype(str).str.strip()
        chunk = chunk.astype(np.float64)

        libsize = chunk.sum(axis=0) if libsize is None else libsize + chunk.sum(axis=0)
        det = (chunk > 0).sum(axis=0)
        detected = det if detected is None else detected + det

        is_ercc = chunk.index.str.startswith("ERCC-")
        e = chunk.loc[is_ercc].sum(axis=0)
        ercc = e if ercc is None else ercc + e

        hit = chunk.index.intersection(wanted)
        if len(hit):
            collected.append(chunk.loc[hit])

    gene_counts = (
        pd.concat(collected) if collected else pd.DataFrame(columns=libsize.index)
    )
    return libsize, gene_counts, ercc, detected


def load_expression(
    genes: list[str],
    use_introns: bool = True,
    exon_path: str = EXON,
    intron_path: str = INTRON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(log_cpm, qc)`` for the requested genes.

    ``log_cpm`` is cells x genes, ``log1p`` of CPM. ``qc`` carries per-cell
    library size, detected-gene count and ERCC spike-in fraction, which the
    technical-confound analysis needs.

    With ``use_introns`` (default) exon and intron counts are summed, which is
    what Scala et al. use for this dataset; intronic reads are a large fraction
    of Patch-seq libraries and dropping them discards real signal. Set False for
    the exon-only sensitivity check.
    """
    libsize, gcounts, ercc, detected = _sum_counts_chunked(exon_path, genes)
    print(f"[expr] exon: {len(gcounts)} of {len(genes)} requested genes found")

    if use_introns:
        li, gi, ei, di = _sum_counts_chunked(intron_path, genes)
        libsize = libsize.add(li, fill_value=0)
        ercc = ercc.add(ei, fill_value=0)
        detected = detected.add(di, fill_value=0)
        gcounts = gcounts.add(
            gi.reindex(index=gcounts.index, columns=gcounts.columns, fill_value=0),
            fill_value=0,
        )
        print("[expr] intron counts summed into exon counts")

    cpm = gcounts.div(libsize, axis=1) * 1e6
    log_cpm = np.log1p(cpm).T  # -> cells x genes
    log_cpm.index = log_cpm.index.astype(str).str.strip()

    qc = pd.DataFrame(
        {
            "libsize": libsize,
            "n_detected": detected,
            "ercc_frac": (ercc / libsize).replace([np.inf, -np.inf], np.nan),
        }
    )
    qc.index = qc.index.astype(str).str.strip()
    qc["log_libsize"] = np.log10(qc["libsize"].replace(0, np.nan))
    return log_cpm, qc


def plasticity_score(
    log_cpm: pd.DataFrame, population: pd.Index | None = None
) -> pd.Series:
    """Mean z-scored log-CPM across the IEG panel = per-cell plasticity score.

    ``population`` sets the cells over which each gene's mean/SD are taken. The
    z-scoring reference is a real analysis choice, not a detail: standardising
    over a mixed excitatory+inhibitory population bakes broad-class expression
    differences into the score, so within-L5 tests must standardise within L5.
    Pass the L5 index explicitly to do that.
    """
    ref = log_cpm if population is None else log_cpm.loc[log_cpm.index.intersection(population)]
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0, ddof=1)

    keep = sd > 0
    if (~keep).any():
        print(f"[score] dropping {int((~keep).sum())} zero-variance gene(s): "
              f"{list(sd.index[~keep])}")
    z = (log_cpm.loc[:, keep] - mu[keep]) / sd[keep]
    return z.mean(axis=1).rename("plasticity_score")


# ---------------------------------------------------------------- ephys


def load_ephys(path: str = EPHYS) -> pd.DataFrame:
    ephys = pd.read_csv(path)
    ephys["cell id"] = ephys["cell id"].astype(str).str.strip()
    return ephys.set_index("cell id")
