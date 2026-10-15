"""Shared matplotlib style (editable-text SVG + 300-dpi PNG)."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import paths

mpl.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#898781", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.titlesize": 9, "axes.titleweight": "bold", "legend.frameon": False})
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "violet": "#4a3aa7", "grey": "#898781", "light": "#d9d8d4", "ink": "#0b0b0b"}
CLASS_C = {"Glutamatergic": C["blue"], "GABAergic": C["orange"]}
SET_LABEL = {"IEG_rapid": "Rapid IEGs", "IEG_delayed": "Delayed IEGs",
             "IEG_proposal": "Proposal IEG set", "stress": "Cell-stress genes"}


def save(fig, name):
    FIGURES = paths.main_figures()
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def label(ax, s):
    ax.text(-0.18, 1.06, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def null_panel(ax, d, xcol_obs, lo, hi, med, title, xlabel):
    """Observed gene-set statistic (dot) vs expression-matched null (bar = 95% interval)."""
    sets = list(SET_LABEL)
    for j, cl in enumerate(["Glutamatergic", "GABAergic"]):
        for i, s in enumerate(sets):
            r = d[(d["class"] == cl) & (d["gene_set"] == s)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = i + (0.18 if j == 0 else -0.18)
            ax.plot([r[lo], r[hi]], [y, y], color=C["light"], lw=5, solid_capstyle="round", zorder=1)
            ax.plot(r[med], y, "|", color=C["grey"], ms=7, zorder=2)
            ax.plot(r[xcol_obs], y, "o", color=CLASS_C[cl], ms=5, mec="white", mew=0.8, zorder=3,
                    label=cl if i == 0 else None)
    ax.set_yticks(range(len(sets)), [SET_LABEL[s] for s in sets])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
