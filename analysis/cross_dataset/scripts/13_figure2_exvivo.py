"""Figure 2: IEG expression versus ex vivo time in mouse M1 and macaque Patch-seq.

  A  mouse M1 -- IEG composite score vs ex vivo hold time (Spearman)
  B  macaque (Dembrow et al. 2024) -- per-gene culture-vs-acute shift in median
     log-CPM for activity and housekeeping genes, as percentiles of the
     whole-transcriptome shift distribution
  C  library-complexity control -- genes detected per cell, acute vs cultured,
     and genome-wide counts of genes up / down in culture
  D  NPAS4 per cell, acute vs cultured

Panel C checks for a CPM normalisation artifact: if culture merely reduced
library complexity, every non-housekeeping gene would shift down together.

Usage:  uv run python analysis/cross_dataset/scripts/13_figure2_exvivo.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from figure_style import (  # noqa: E402
    BLUE, GREEN, INK, INK_2, INK_MUTED, NEUTRAL, RED,
    apply_style, panel_label, save,
)

RESULTS = paths.results(__file__)
DEM = paths.REPO / "ReplicationData" / "Dembrow2024"

IEGS = ["NPAS4", "BDNF", "FOS", "FOSB", "GADD45G", "ARC", "EGR1", "JUNB", "JUN"]
CONTROLS = ["ACTB", "GAPDH", "SLC17A7"]
HIGHLIGHT = ["NPAS4", "BDNF", "FOS", "FOSB", "GADD45G", "JUN"]


def load_macaque():
    """L5 ET cells with acute/culture labels, log1p CPM expression."""
    ft = pd.read_feather(DEM / "macaque_L5_patchseq.feather")
    phys = pd.read_csv(DEM / "L5_physiology_121223.csv", low_memory=False)
    cells = pd.read_csv(DEM / "cell_list_L5.csv", low_memory=False)
    K = "container"
    ft[K] = ft["patched_cell_container_label"].astype(str).str.strip()
    phys[K] = phys["sample_id"].astype(str).str.strip()
    cells[K] = cells["patched_cell_container"].astype(str).str.strip()

    meta = (phys[[K, "slice_type"]]
            .merge(cells[[K, "subclass_Corr_label_M1", "Genes.Detected", "QC"]],
                   on=K, how="inner"))
    meta = meta[meta["QC"] == True].drop_duplicates(subset=K).set_index(K)
    meta = meta[meta["subclass_Corr_label_M1"].astype(str).str.contains("L5 ET")]

    expr = ft.set_index(K).drop(columns=["sample_id",
                                         "patched_cell_container_label",
                                         "total_reads_label"])
    idx = meta.index.intersection(expr.index)
    return meta.loc[idx], np.log1p(expr.loc[idx])


def main() -> None:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # ---------------- A: mouse hold time ----------------
    m1 = pd.read_csv(paths.M1 / "results" / "cells_l5.csv", index_col=0)
    x, y = m1["Hold Time (min)"], m1["plasticity_score"]
    ok = x.notna() & y.notna()
    rho, p = stats.spearmanr(x[ok], y[ok])
    ax_a.scatter(x[ok], y[ok], s=17, facecolor=BLUE, edgecolor="white",
                 linewidth=0.5, alpha=0.9, zorder=3)
    b, a = np.polyfit(x[ok], y[ok], 1)
    xs = np.linspace(x[ok].min(), x[ok].max(), 40)
    ax_a.plot(xs, a + b * xs, color=INK_2, lw=1.8, zorder=4)
    ax_a.set_xlabel("Ex vivo hold time (min)")
    ax_a.set_ylabel("IEG composite score")
    ax_a.set_title("Mouse M1: signal decays out of the animal\n"
                   f"ρ = {rho:+.2f}, p = {p:.1g}, n = {int(ok.sum())}",
                   loc="left", color=INK, fontsize=9, linespacing=1.5)
    panel_label(ax_a, "A")

    meta, expr = load_macaque()
    acute = meta.index[meta["slice_type"] == "NHP_acute"]
    cult = meta.index[meta["slice_type"] == "NHP_culture"]

    # ---------------- B: shift per gene vs the all-gene reference ----------
    # A histogram of 39,670 genes is dominated by a spike at zero and forces the
    # gene labels on top of each other, so show the genes of interest directly
    # against the whole-transcriptome spread instead.
    shift = expr.loc[cult].median() - expr.loc[acute].median()
    shift = shift.replace([np.inf, -np.inf], np.nan).dropna()
    lo5, lo25, hi75, hi95 = np.percentile(shift, [5, 25, 75, 95])

    rows = [(g, shift[g], 100.0 * (shift < shift[g]).mean(), "ieg")
            for g in IEGS if g in shift.index]
    rows += [(g, shift[g], 100.0 * (shift < shift[g]).mean(), "ctl")
             for g in CONTROLS if g in shift.index]
    rows.sort(key=lambda r: r[1])
    ypos = np.arange(len(rows))

    ax_b.axvspan(lo5, hi95, color="#eeeeec", zorder=0)
    ax_b.axvspan(lo25, hi75, color="#dddddb", zorder=1)
    ax_b.axvline(0, color=INK_MUTED, lw=0.8, zorder=2)
    for y, (g, v, pct, kind) in zip(ypos, rows):
        col = RED if kind == "ieg" else GREEN
        ax_b.plot([0, v], [y, y], color=INK_MUTED, lw=1.0, zorder=3)
        ax_b.scatter(v, y, s=40, facecolor=col, edgecolor="white",
                     linewidth=0.7, zorder=4)
        ax_b.text(v - 0.22 if v < 0 else v + 0.22, y, f"{pct:.1f}%",
                  ha="right" if v < 0 else "left", va="center",
                  fontsize=6.2, color=col, fontweight="bold")
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax_b.set_xlim(-5.0, 2.2)
    ax_b.set_xlabel("Δ median log-CPM, cultured − acute")
    ax_b.set_title(f"Macaque: activity genes fall far below the\n"
                   f"whole-transcriptome spread ({len(shift):,} genes)",
                   loc="left", color=INK, fontsize=9, linespacing=1.5)
    ax_b.legend(handles=[
        Line2D([], [], marker="o", ls="", markerfacecolor=RED,
               markeredgecolor="white", markersize=6, label="activity gene"),
        Line2D([], [], marker="o", ls="", markerfacecolor=GREEN,
               markeredgecolor="white", markersize=6, label="housekeeping"),
        Line2D([], [], color="#dddddb", lw=6, label="all genes, IQR / 5–95%"),
    ], loc="lower right", fontsize=6, handletextpad=0.5, borderpad=0.25)
    ax_b.grid(axis="y", visible=False)
    panel_label(ax_b, "B")

    # ---------------- C: artifact control ----------------
    qa = meta.loc[acute, "Genes.Detected"].dropna()
    qc = meta.loc[cult, "Genes.Detected"].dropna()
    pq = stats.mannwhitneyu(qa, qc)[1]
    bp = ax_c.boxplot([qa, qc], positions=[0, 1], widths=0.55,
                      patch_artist=True, showfliers=False,
                      medianprops=dict(color=INK, lw=1.4))
    for patch, col in zip(bp["boxes"], [BLUE, GREEN]):
        patch.set_facecolor(col); patch.set_alpha(0.55)
        patch.set_edgecolor(INK_2); patch.set_linewidth(0.8)
    for el in ("whiskers", "caps"):
        for ln in bp[el]:
            ln.set_color(INK_2); ln.set_linewidth(0.8)
    rng = np.random.default_rng(0)
    for i, v in enumerate([qa, qc]):
        ax_c.scatter(rng.normal(i, 0.07, len(v)), v, s=7, color=INK_2,
                     alpha=0.45, linewidths=0, zorder=3)
    n_up = int((shift > 0).sum()); n_dn = int((shift < 0).sum())
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels([f"acute\n(n={len(qa)})", f"cultured\n(n={len(qc)})"],
                         fontsize=7.5)
    ax_c.set_ylabel("Genes detected")
    ax_c.set_title("Not a normalisation artifact: cultured\n"
                   f"libraries are richer (p = {pq:.3f})",
                   loc="left", color=INK, fontsize=9, linespacing=1.5)
    ax_c.text(0.5, 0.04,
              f"genome-wide: {n_up:,} genes up in culture\nvs {n_dn:,} down",
              transform=ax_c.transAxes, ha="center", va="bottom", fontsize=6.5,
              color=INK_2, linespacing=1.4)
    ax_c.grid(axis="x", visible=False)
    panel_label(ax_c, "C")

    # ---------------- D: NPAS4 ----------------
    gene = "NPAS4"
    va, vc = expr.loc[acute, gene], expr.loc[cult, gene]
    pu = stats.mannwhitneyu(va, vc)[1]
    bp = ax_d.boxplot([va, vc], positions=[0, 1], widths=0.55,
                      patch_artist=True, showfliers=False,
                      medianprops=dict(color=INK, lw=1.4))
    for patch, col in zip(bp["boxes"], [BLUE, GREEN]):
        patch.set_facecolor(col); patch.set_alpha(0.55)
        patch.set_edgecolor(INK_2); patch.set_linewidth(0.8)
    for el in ("whiskers", "caps"):
        for ln in bp[el]:
            ln.set_color(INK_2); ln.set_linewidth(0.8)
    for i, v in enumerate([va, vc]):
        ax_d.scatter(rng.normal(i, 0.07, len(v)), v, s=9, color=INK_2,
                     alpha=0.5, linewidths=0, zorder=3)
    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels([f"acute\n(n={len(va)})", f"cultured\n(n={len(vc)})"],
                         fontsize=7.5)
    ax_d.set_ylabel(f"{gene}  log-CPM")
    frac0 = 100.0 * (vc == 0).mean()
    ax_d.set_title(f"{gene} is abolished by culture\n"
                   f"p = {pu:.1g}; {frac0:.0f}% of cultured cells at zero",
                   loc="left", color=INK, fontsize=9, linespacing=1.5)
    ax_d.grid(axis="x", visible=False)
    panel_label(ax_d, "D")

    fig.tight_layout(w_pad=2.4, h_pad=2.8)
    save(fig, "Fig2_exvivo_degradation")

    print(f"\nmouse M1 hold time: rho={rho:+.3f} p={p:.3g} n={int(ok.sum())}")
    print(f"macaque L5 ET: acute {len(acute)}, cultured {len(cult)}")
    print(f"genes up in culture {n_up:,} / down {n_dn:,} / total {len(shift):,}")
    for g in IEGS + CONTROLS:
        if g in shift.index:
            print(f"  {g:9s} shift {shift[g]:+.3f}  "
                  f"pct {100.0*(shift<shift[g]).mean():5.1f}")


if __name__ == "__main__":
    main()
