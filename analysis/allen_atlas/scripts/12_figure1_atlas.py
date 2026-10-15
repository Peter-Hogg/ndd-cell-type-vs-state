"""Plot activity-gene panel scores across glutamatergic cells of the Allen SmartSeq atlas.

Uses the Allen SmartSeq atlas (Tasic et al. 2018), glutamatergic cells only, with
panel scores from 11_atlas_extract_panels.py. Panels:
  A  eta^2 of each panel score on subclass vs. region
  B  pairwise Spearman rho between panel scores
  C  subclass mean scores per panel
  D  t-SNE maps coloured by each panel score

Usage:  uv run python analysis/allen_atlas/scripts/12_figure1_atlas.py
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
    BLUE, GREEN, INK, INK_2, INK_MUTED, NEUTRAL, RHO_CMAP,
    apply_style, panel_label, save,
)

RESULTS = paths.results(__file__)
SCORES = {
    "score_canonical_ieg": "Canonical IEG",
    "score_jun_ap1": "Jun / AP-1",
    "score_delayed_effector": "Delayed effector",
}
SUBCLASS_ORDER = ["L2/3 IT CTX", "L4/5 IT CTX", "L5 IT CTX", "L5 PT CTX",
                  "L6 IT CTX", "L6 CT CTX", "L5/6 NP CTX", "L6b CTX"]


def eta2(d: pd.DataFrame, val: str, grp: str) -> float:
    s = d[[val, grp]].dropna()
    gm = s[val].mean()
    ssb = sum(len(x) * (x[val].mean() - gm) ** 2
              for _, x in s.groupby(grp, observed=True) if len(x) > 1)
    return float(ssb / ((s[val] - gm) ** 2).sum())


def main() -> None:
    apply_style()
    df = pd.read_parquet(RESULTS / "M_atlas_panel_scores.parquet")
    g = df[df["class_label"].astype(str).str.contains("Glutamatergic", na=False)].copy()
    keys = list(SCORES)

    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.92, 0.92],
                          hspace=0.58, wspace=0.55,
                          top=0.93, bottom=0.045, left=0.115, right=0.965)

    # ---- A: eta^2, cell type vs region -------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    sub_e = [eta2(g, k, "subclass_label") for k in keys]
    reg_e = [eta2(g, k, "region_label") for k in keys]
    x = np.arange(len(keys))
    w = 0.36
    ax.bar(x - w / 2, sub_e, w, color=BLUE, edgecolor="white", linewidth=0.8,
           label="subclass (cell type)", zorder=3)
    ax.bar(x + w / 2, reg_e, w, color=GREEN, edgecolor="white", linewidth=0.8,
           label="region", zorder=3)
    for xi, v in zip(x - w / 2, sub_e):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    for xi, v in zip(x + w / 2, reg_e):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([SCORES[k] for k in keys], fontsize=7.5)
    ax.set_ylabel("η²  (variance explained)")
    ax.set_ylim(0, max(sub_e) * 1.25)
    ax.set_title("Delayed-effector genes are largely cell-type identity;\n"
                 "Jun/AP-1 is the least type-determined",
                 loc="left", color=INK, fontsize=9, linespacing=1.5)
    ax.legend(loc="upper left", bbox_to_anchor=(0.30, 1.0))
    ax.grid(axis="x", visible=False)
    panel_label(ax, "A", dx=-0.13)

    # ---- B: pairwise separability ------------------------------------------
    ax_b = fig.add_subplot(gs[0, 2])
    pairs = [(0, 1), (0, 2), (1, 2)]
    rhos = [stats.spearmanr(g[keys[i]], g[keys[j]])[0] for i, j in pairs]
    ypos = np.arange(len(pairs))[::-1]
    ax_b.axvspan(0.9, 1.0, color="#f0e4e4", zorder=0)
    for y, r in zip(ypos, rhos):
        ax_b.plot([0, r], [y, y], color=INK_MUTED, lw=1.2, zorder=2)
        ax_b.scatter(r, y, s=46, facecolor=BLUE, edgecolor="white",
                     linewidth=0.7, zorder=3)
        ax_b.text(r + 0.04, y, f"{r:+.2f}", va="center", fontsize=7, color=INK)
    SHORT_LBL = {0: "IEG", 1: "Jun", 2: "Delayed"}
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels([f"{SHORT_LBL[i]} vs {SHORT_LBL[j]}" for i, j in pairs],
                         fontsize=6.8)
    ax_b.set_xlim(0, 1.0)
    ax_b.set_ylim(-0.6, len(pairs) - 0.4)
    ax_b.set_xlabel("Spearman ρ")
    ax_b.set_title("Separable, not\nredundant", loc="left", color=INK,
                   fontsize=9, linespacing=1.5)
    # Band caption inside the band, so it cannot collide with the axis furniture.
    ax_b.text(0.95, np.mean(ypos), "ρ > 0.9 = same axis", rotation=90,
              ha="center", va="center", fontsize=6, color="#a05a5a")
    ax_b.grid(axis="y", visible=False)
    panel_label(ax_b, "B", dx=-0.50)

    # ---- C: subclass means, per panel -------------------------------------
    ax_c = fig.add_subplot(gs[1, :])
    t = (g[g["subclass_label"].isin(SUBCLASS_ORDER)]
         .groupby("subclass_label", observed=True)[keys].mean()
         .reindex(SUBCLASS_ORDER))
    m = t.to_numpy().T
    vmax = np.nanmax(np.abs(m))
    im = ax_c.imshow(m, cmap=RHO_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    ax_c.set_xticks(range(len(SUBCLASS_ORDER)))
    ax_c.set_xticklabels(SUBCLASS_ORDER, fontsize=7, rotation=30, ha="right")
    ax_c.set_yticks(range(len(keys)))
    ax_c.set_yticklabels([SCORES[k] for k in keys], fontsize=7.5)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax_c.text(j, i, f"{m[i, j]:+.2f}", ha="center", va="center",
                      fontsize=6.5, color=INK)
    ax_c.grid(visible=False)
    rho_ord = stats.spearmanr(t.iloc[:, 0], t.iloc[:, 1])[0]
    ax_c.set_title("The programmes rank cell types differently "
                   f"(orderings ρ = {rho_ord:+.2f} for IEG vs Jun/AP-1)",
                   loc="left", color=INK, fontsize=9)
    cb = fig.colorbar(im, ax=ax_c, fraction=0.020, pad=0.012)
    cb.set_label("mean score", fontsize=7)
    cb.outline.set_visible(False)
    panel_label(ax_c, "C", dx=-0.098, dy=1.12)

    # ---- D: t-SNE small multiples -----------------------------------------
    have_tsne = {"Lim1", "Lim2"}.issubset(g.columns) and g["Lim1"].notna().any()
    for k, key in enumerate(keys):
        axk = fig.add_subplot(gs[2, k])
        if have_tsne:
            s = g[["Lim1", "Lim2", key]].dropna()
            v = s[key].to_numpy()
            lo, hi = np.nanpercentile(v, [2, 98])
            sc = axk.scatter(s["Lim1"], s["Lim2"], c=v, cmap="magma",
                             vmin=lo, vmax=hi, s=0.7, linewidths=0,
                             rasterized=True)
            axk.set_xticks([]); axk.set_yticks([])
            for sp in axk.spines.values():
                sp.set_linewidth(0.6)
            cb = fig.colorbar(sc, ax=axk, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=6)
            cb.outline.set_visible(False)
        else:
            axk.text(0.5, 0.5, "no embedding", ha="center", va="center",
                     fontsize=7, color=INK_2)
            axk.set_xticks([]); axk.set_yticks([])
        axk.set_title(SCORES[key], loc="left", color=INK, fontsize=8)
        axk.grid(visible=False)
        if k == 0:
            panel_label(axk, "D", dx=-0.22, dy=1.20)

    save(fig, "Fig1_activity_programmes_atlas")

    # ---- summary numbers -------------------------------------------------
    print(f"\nglutamatergic cells: {len(g):,}")
    for k in keys:
        print(f"  {SCORES[k]:18s} eta2 subclass {eta2(g,k,'subclass_label'):.3f}"
              f"   eta2 region {eta2(g,k,'region_label'):.3f}")
    for (i, j), r in zip(pairs, rhos):
        print(f"  rho {SCORES[keys[i]]:18s} vs {SCORES[keys[j]]:18s} = {r:+.3f}")


if __name__ == "__main__":
    main()
