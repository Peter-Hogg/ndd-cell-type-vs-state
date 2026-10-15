"""Build the two Patch-seq figures from the tables written by 01.

Fig3  the within-type test: activity state vs intrinsic ephys inside a
      transcriptomic type, and the effect size that n could resolve
Fig4  the confound: pooled correlation vs between-class structure, plus the
      depth and ex vivo hold-time confounds on the score itself

Reads only results/*.csv; rerun 01 first if the inputs changed.

Usage:  uv run python analysis/m1_patchseq/scripts/02_figures_ephys_confound.py
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

from figure_style import (
    BLUE,
    FAMILY_ORDER,
    GREEN,
    INK,
    INK_2,
    INK_MUTED,
    NEUTRAL,
    RED,
    RHO_CMAP,
    apply_style,
    panel_label,
    save,
)

RESULTS = paths.results(__file__)

SHORT = {
    "Input resistance (MOhm)": "Input resistance",
    "AP threshold (mV)": "AP threshold",
    "AP width (ms)": "AP width",
    "Resting membrane potential (mV)": "Resting Vm",
    "ISI adaptation index": "ISI adaptation",
    "Spike frequency adaptation": "Spike freq. adapt.",
    "Rheobase (pA)": "Rheobase",
}


def _rho_stats(x, y) -> str:
    """Spearman rho / p / n as a one-line string, for use in a panel title.

    Stats live in the title rather than inside the axes: an in-plot annotation
    has to dodge the point cloud, and in a multi-panel figure it always
    eventually collides with something.
    """
    ok = x.notna() & y.notna()
    rho, p = stats.spearmanr(x[ok], y[ok])
    p_txt = "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"
    return f"ρ = {rho:+.2f}, {p_txt}, n = {int(ok.sum())}"


def eta_sq_label(df: pd.DataFrame, value: str, group: str) -> str:
    """One-way eta^2 + Kruskal p as a title line (mirrors 01's eta_squared)."""
    sub = df[[value, group]].dropna()
    groups = [g[value].to_numpy() for _, g in sub.groupby(group, observed=True)
              if len(g) > 1]
    grand = sub[value].mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_t = ((sub[value] - grand) ** 2).sum()
    _, p = stats.kruskal(*groups)
    p_txt = "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"
    return f"η² = {ss_b / ss_t:.2f} by RNA family, {p_txt}, n = {len(sub)}"


def _subtitle(ax, main: str, stats_line: str) -> None:
    ax.set_title(f"{main}\n{stats_line}", loc="left", color=INK, fontsize=9,
                 linespacing=1.5)


def _fit_line(ax, x, y, color, lw=2.0, ls="-", zorder=3, label=None):
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return
    b, a = np.polyfit(x[ok], y[ok], 1)
    xs = np.linspace(x[ok].min(), x[ok].max(), 50)
    ax.plot(xs, a + b * xs, color=color, lw=lw, ls=ls, zorder=zorder, label=label)


# ------------------------------------------------------------------ figure 3


def figure3(cells: pd.DataFrame, a_tbl: pd.DataFrame, b_tbl: pd.DataFrame,
            d_partial: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # --- A: focal feature, within L5 -------------------------------------
    # Marker shape carries identity alongside hue, so the two groups stay
    # separable in grayscale, in print and under any CVD.
    for fam, color, marker in (("IT", BLUE, "o"), ("ET", GREEN, "^")):
        sub = cells[cells["RNA family"] == fam]
        ax_a.scatter(sub["plasticity_score"], sub["Rheobase (pA)"], s=24,
                     marker=marker, facecolor=color, edgecolor="white",
                     linewidth=0.6, alpha=0.9, zorder=3,
                     label=f"L5 {fam} (n={len(sub)})")
    _fit_line(ax_a, cells["plasticity_score"], cells["Rheobase (pA)"], INK_2, lw=1.8)
    ax_a.set_xlabel("Plasticity score (IEG, z within L5)")
    ax_a.set_ylabel("Rheobase (pA)")
    _subtitle(ax_a, "Strongest within-L5 association",
              _rho_stats(cells["plasticity_score"], cells["Rheobase (pA)"]))
    ax_a.legend(loc="lower left", handletextpad=0.4, borderpad=0.2)
    panel_label(ax_a, "A")

    # --- B: all features vs the detection floor --------------------------
    core = (a_tbl[a_tbl["population"].str.contains("core")]
            .set_index("feature").reindex(SHORT.keys()).reset_index())
    ypos = np.arange(len(core))[::-1]
    mdr = float(core["min_detectable_rho"].max())

    ax_b.axvspan(-mdr, mdr, color=GRID_BAND, zorder=0)
    ax_b.axvline(0, color=INK_MUTED, lw=0.8, zorder=1)
    for y, (_, r) in zip(ypos, core.iterrows()):
        sig = bool(r["p_fdr"] < 0.05)
        ax_b.plot([0, r["rho"]], [y, y], color=INK_MUTED, lw=1.2, zorder=2)
        ax_b.scatter(r["rho"], y, s=44, zorder=3,
                     facecolor=RED if sig else BLUE,
                     edgecolor="white", linewidth=0.7)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels([SHORT[f] for f in core["feature"]])
    ax_b.set_xlim(-0.45, 0.45)
    ax_b.set_xlabel("Spearman ρ vs plasticity score")
    _subtitle(ax_b, f"Within L5 (n={int(core['n'].max())}): nothing survives FDR",
              f"shaded band: |ρ| < {mdr:.2f}, below the detection floor at 80% power")
    ax_b.grid(axis="y", visible=False)
    panel_label(ax_b, "B")

    # --- C: within individual transcriptomic type ------------------------
    piv = b_tbl.pivot_table(index="feature", columns="RNA type", values="rho")
    piv = piv.reindex(SHORT.keys())
    ns = b_tbl.groupby("RNA type")["n"].max().to_dict()
    im = ax_c.imshow(piv.to_numpy(), cmap=RHO_CMAP, vmin=-0.5, vmax=0.5,
                     aspect="auto")
    ax_c.set_xticks(range(len(piv.columns)))
    ax_c.set_xticklabels([f"{c}\n(n={ns.get(c, '?')})" for c in piv.columns],
                         fontsize=7)
    ax_c.set_yticks(range(len(piv.index)))
    ax_c.set_yticklabels([SHORT[f] for f in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if pd.notna(v):
                ax_c.text(j, i, f"{v:+.2f}", ha="center", va="center",
                          fontsize=6.5, color=INK)
    ax_c.grid(visible=False)
    ax_c.set_title("Within a single type: 0/14 significant", loc="left", color=INK)
    cb = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.03)
    cb.set_label("ρ", fontsize=7, rotation=0, labelpad=6)
    cb.outline.set_visible(False)
    panel_label(ax_c, "C")

    # --- D: raw vs confound-adjusted -------------------------------------
    d = d_partial.set_index("feature").reindex(SHORT.keys()).reset_index()
    ypos = np.arange(len(d))[::-1]
    for y, (_, r) in zip(ypos, d.iterrows()):
        ax_d.plot([r["rho_raw"], r["rho_partial"]], [y, y],
                  color=INK_MUTED, lw=1.0, zorder=2)
        ax_d.scatter(r["rho_raw"], y, s=34, facecolor="white",
                     edgecolor=INK_2, linewidth=1.2, zorder=3)
        sig = bool(r["p_partial_fdr"] < 0.05)
        ax_d.scatter(r["rho_partial"], y, s=44,
                     facecolor=RED if sig else BLUE,
                     edgecolor="white", linewidth=0.7, zorder=4)
        if sig:
            # Above the marker: to its right sits the raw circle, to its left
            # the axis tick label.
            ax_d.text(r["rho_partial"], y + 0.30,
                      f"FDR = {r['p_partial_fdr']:.3f}", fontsize=6.5,
                      color=RED, ha="center", va="bottom", fontweight="bold")
    ax_d.axvline(0, color=INK_MUTED, lw=0.8, zorder=1)
    ax_d.set_yticks(ypos)
    ax_d.set_yticklabels([SHORT[f] for f in d["feature"]])
    ax_d.set_xlim(-0.45, 0.45)
    ax_d.set_xlabel("Spearman ρ vs plasticity score")
    _subtitle(ax_d, "Removing technical confounds",
              "depth, spike-ins and hold time partialled out")
    # Legend below the axes: inside, it lands on the feature rows.
    ax_d.legend(handles=[
        Line2D([], [], marker="o", ls="", markerfacecolor="white",
               markeredgecolor=INK_2, markersize=6, label="raw"),
        Line2D([], [], marker="o", ls="", markerfacecolor=BLUE,
               markeredgecolor="white", markersize=7, label="adjusted (n.s.)"),
        Line2D([], [], marker="o", ls="", markerfacecolor=RED,
               markeredgecolor="white", markersize=7, label="adjusted (FDR<0.05)"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
        columnspacing=1.1, handletextpad=0.35)
    ax_d.grid(axis="y", visible=False)
    panel_label(ax_d, "D")

    fig.tight_layout(w_pad=2.4, h_pad=2.6)
    save(fig, "Fig3_within_type_null")


# ------------------------------------------------------------------ figure 4


def figure4(cells_all: pd.DataFrame, c_mixed: pd.DataFrame,
            c_fam: pd.DataFrame, d_tech: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.2, 7.1))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 1.0, 0.95],
                          hspace=0.62, wspace=0.85,
                          top=0.94, bottom=0.065, left=0.115, right=0.93)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :2])
    ax_c = fig.add_subplot(gs[1, 2:])
    ax_d1 = fig.add_subplot(gs[2, :2])
    ax_d2 = fig.add_subplot(gs[2, 2:])

    # --- A: pooled vs within-family (the Simpson's-paradox panel) --------
    fams = [f for f in FAMILY_ORDER if f in set(c_fam["RNA family"])]
    piv = c_fam.pivot_table(index="feature", columns="RNA family", values="rho")
    piv = piv.reindex(index=list(SHORT.keys()), columns=fams)
    pooled = c_mixed.set_index("feature")["rho"].reindex(SHORT.keys())

    # Pooled column, a spacer, then the within-family columns. The blank column
    # is the 2px surface gap: it stops the pooled value reading as one of them.
    mat = np.column_stack([pooled.to_numpy(), np.full(len(pooled), np.nan),
                           piv.to_numpy()])
    labels = ["Pooled\n(all cells)", ""] + fams

    im = ax_a.imshow(mat, cmap=RHO_CMAP, vmin=-0.5, vmax=0.5, aspect="auto")
    ax_a.set_xticks(range(len(labels)))
    ax_a.set_xticklabels(labels, fontsize=7)
    ax_a.set_yticks(range(len(pooled)))
    ax_a.set_yticklabels([SHORT[f] for f in pooled.index])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax_a.text(j, i, f"{v:+.2f}", ha="center", va="center",
                          fontsize=6.5,
                          fontweight="bold" if j == 0 else "normal", color=INK)
    ax_a.grid(visible=False)
    ax_a.set_title("Pooled correlations are between-class structure, "
                   "not state→physiology", loc="left", color=INK)
    cb = fig.colorbar(im, ax=ax_a, fraction=0.022, pad=0.015)
    cb.set_label("ρ", fontsize=7, rotation=0, labelpad=6)
    cb.outline.set_visible(False)
    panel_label(ax_a, "A", dx=-0.075, dy=1.14)

    # --- B: score differs by broad class ---------------------------------
    sub = cells_all[cells_all["RNA family"].isin(fams)]
    data = [sub.loc[sub["RNA family"] == f, "plasticity_score_allref"].dropna()
            for f in fams]
    bp = ax_b.boxplot(data, positions=range(len(fams)), widths=0.6,
                      patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.4))
    for patch in bp["boxes"]:
        patch.set_facecolor(NEUTRAL)
        patch.set_edgecolor(INK_2)
        patch.set_linewidth(0.8)
    for el in ("whiskers", "caps"):
        for line in bp[el]:
            line.set_color(INK_2)
            line.set_linewidth(0.8)
    for i, f in enumerate(fams):
        v = sub.loc[sub["RNA family"] == f, "plasticity_score_allref"].dropna()
        ax_b.scatter(np.random.default_rng(0).normal(i, 0.075, len(v)), v,
                     s=3.5, color=INK_2, alpha=0.30, zorder=1, linewidths=0)
    ax_b.set_xticks(range(len(fams)))
    ax_b.set_xticklabels(fams, fontsize=7)
    ax_b.set_ylabel("Plasticity score\n(z across all cells)")
    e2 = eta_sq_label(sub, "plasticity_score_allref", "RNA family")
    _subtitle(ax_b, "Score is largely a class property", e2)
    ax_b.grid(axis="x", visible=False)
    panel_label(ax_b, "B", dx=-0.24, dy=1.14)

    # --- C: small multiples, pooled slope vs within-family slopes --------
    # Seven families in one scatter would break the all-pairs CVD floor, so
    # facet with a single hue instead and carry identity by position + label.
    ax_c.axis("off")
    inner = ax_c.get_subplotspec().subgridspec(2, 4, hspace=0.55, wspace=0.30)
    feat = "Input resistance (MOhm)"
    xall = cells_all["plasticity_score_allref"]
    yall = cells_all[feat]
    ok = xall.notna() & yall.notna()
    b_pool, a_pool = np.polyfit(xall[ok], yall[ok], 1)

    for k, f in enumerate(fams):
        axk = fig.add_subplot(inner[k // 4, k % 4])
        s = cells_all[cells_all["RNA family"] == f]
        axk.scatter(s["plasticity_score_allref"], s[feat], s=4.5,
                    color=BLUE, alpha=0.55, linewidths=0, zorder=2)
        xs = np.linspace(xall[ok].min(), xall[ok].max(), 20)
        axk.plot(xs, a_pool + b_pool * xs, color=INK_MUTED, lw=1.2, ls="--",
                 zorder=3)
        _fit_line(axk, s["plasticity_score_allref"], s[feat], RED, lw=1.6, zorder=4)
        r = c_fam[(c_fam["RNA family"] == f) & (c_fam["feature"] == feat)]["rho"]
        axk.set_title(f"{f}  ρ={float(r.iloc[0]):+.2f}" if len(r) else f,
                      fontsize=6.5, loc="left", color=INK, pad=2)
        axk.set_xticklabels([])
        axk.set_yticklabels([])
        axk.tick_params(length=0)
        for sp in axk.spines.values():
            sp.set_linewidth(0.6)
    lg = fig.add_subplot(inner[1, 3])
    lg.axis("off")
    lg.legend(handles=[
        Line2D([], [], color=INK_MUTED, ls="--", lw=1.2, label="pooled fit"),
        Line2D([], [], color=RED, lw=1.6, label="within-family fit"),
    ], loc="center", fontsize=6.2)
    ax_c.set_title(f"{SHORT[feat]} vs score, by family\n"
                   "pooled slope is steeper than every within-family slope",
                   loc="left", color=INK, pad=20, fontsize=9, linespacing=1.5)
    panel_label(ax_c, "C", dx=-0.11, dy=1.24)

    # --- D/E: what actually drives the score -----------------------------
    l5 = cells_all[cells_all["RNA type"].str.match(r"^L5 (IT|ET)_\d+$", na=False)]
    for axk, col, xlabel, letter, title in (
        (ax_d1, "n_detected", "Genes detected", "D",
         "Score tracks library complexity"),
        (ax_d2, "Hold Time (min)", "Ex vivo hold time (min)", "E",
         "Score tracks time in the rig"),
    ):
        axk.scatter(l5[col], l5["plasticity_score"], s=16, facecolor=BLUE,
                    edgecolor="white", linewidth=0.5, alpha=0.9, zorder=3)
        _fit_line(axk, l5[col], l5["plasticity_score"], INK_2, lw=1.6)
        axk.set_xlabel(xlabel)
        axk.set_ylabel("Plasticity score")
        _subtitle(axk, title, _rho_stats(l5[col], l5["plasticity_score"]))
        panel_label(axk, letter, dx=-0.24, dy=1.16)

    save(fig, "Fig4_confounds")


# ------------------------------------------------------------------ entry

GRID_BAND = "#f0f0ee"


def main() -> None:
    apply_style()
    cells_l5 = pd.read_csv(RESULTS / "cells_l5.csv", index_col=0)
    cells_all = pd.read_csv(RESULTS / "cells_all.csv", index_col=0)
    a_tbl = pd.read_csv(RESULTS / "A_within_L5.csv")
    b_tbl = pd.read_csv(RESULTS / "B_within_type.csv")
    c_mixed = pd.read_csv(RESULTS / "C_mixed_population.csv")
    c_fam = pd.read_csv(RESULTS / "C_within_family.csv")
    d_tech = pd.read_csv(RESULTS / "D_technical.csv")
    d_partial = pd.read_csv(RESULTS / "D_partial.csv")

    figure3(cells_l5, a_tbl, b_tbl, d_partial)
    figure4(cells_all, c_mixed, c_fam, d_tech)


if __name__ == "__main__":
    main()
