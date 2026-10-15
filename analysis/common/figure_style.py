"""Shared figure style: palette, rcParams, panel labels and multi-format save.

Palette slots are taken from a validated categorical order (blue, green) and a
blue/neutral-gray/red diverging ramp for signed correlations. Both were checked
with the data-viz validator against a white surface under all-pairs CVD rules:
lightness band, chroma floor, CVD separation, normal-vision floor and contrast
all pass. Do not substitute hues here without re-validating.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Categorical slots, fixed order — never cycled.
BLUE = "#2a78d6"
GREEN = "#008300"
RED = "#e34948"

# Recessive ink.
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e6e6e3"
NEUTRAL = "#b9b8b2"      # single-hue fills where position carries identity
MIDPOINT = "#f2f2f0"     # neutral gray midpoint for the diverging ramp

# Diverging ramp for signed quantities (rho): two hues, neutral gray midpoint.
# Never a rainbow, never a hue at the midpoint.
RHO_CMAP = LinearSegmentedColormap.from_list("rho_div", [BLUE, MIDPOINT, RED], N=256)

FAMILY_ORDER = ["IT", "ET", "CT", "Pvalb", "Sst", "Vip", "Lamp5"]


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_2,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "lines.linewidth": 2.0,
        "svg.fonttype": "none",   # keep text editable in Illustrator
        "pdf.fonttype": 42,
    })


def panel_label(ax, letter: str, dx: float = -0.20, dy: float = 1.06) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left", color=INK)


def save(fig, stem: str, outdir=None) -> None:
    import pathlib
    import paths
    d = pathlib.Path(outdir) if outdir else paths.main_figures()
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        p = d / f"{stem}.{ext}"
        fig.savefig(p)
        print(f"[fig] {p}")
    plt.close(fig)
