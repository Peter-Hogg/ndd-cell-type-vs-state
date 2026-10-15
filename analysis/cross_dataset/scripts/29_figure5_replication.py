"""Figure 5: M1 -> V1 replication of the IEG_TF gene-set effect on rheobase, set vs single gene.

  A  discovery -> replication for the IEG_TF category, both against phenotype-
     permutation nulls: mouse M1 L5 (n=126) and mouse V1 L5 ET (n=436)
  B  the V1 permutation null distribution with the observed value marked
  C  Jun's rho vs rheobase across M1 populations and in V1 (with 95% CI),
     a winner's-curse comparison
  D  the five V1 tests (A1-A3, B1-B2) against alpha = 0.025 per hypothesis

Panel B recomputes the V1 null only to draw it; the test statistics and p-values
in panels C-D are read from `analysis/v1_patchseq/scripts/28_v1_confirmatory.py`
output (`analysis/v1_patchseq/results/AA_v1_confirmatory.csv`).

Usage:
  uv run python analysis/cross_dataset/scripts/29_figure5_replication.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)

from importlib import import_module

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

_m22 = import_module("22_repairs_excitatory")
_m23 = import_module("23_functional_categories_by_layer")
_m28 = import_module("28_v1_confirmatory")
from figure_style import (  # noqa: E402
    BLUE, GREEN, INK, INK_2, INK_MUTED, NEUTRAL, RED,
    apply_style, panel_label, save,
)

ROOT = paths.REPO
V1 = ROOT / "ReplicationData" / "MouseV1_Sorensen2026"
OUT = paths.results(__file__)
N_PERM = 5000
RNG = np.random.default_rng(0)


def v1_null_and_obs():
    """Recompute the V1 IEG_TF set statistic and its permutation null, to draw."""
    log_cpm, counts, qc = _m28.load_v1()
    log_cpm.index = log_cpm.index.astype(str); qc.index = qc.index.astype(str)
    phys = pd.read_csv(V1 / "exc_mMET_ephys_features.csv")
    phys["specimen_id"] = phys["specimen_id"].astype(str)
    phys = phys.set_index("specimen_id")
    met = pd.read_csv(V1 / "inferred_met_types.csv").rename(
        columns={"Unnamed: 0": "specimen_id"})
    met["specimen_id"] = met["specimen_id"].astype(str)
    met = met.set_index("specimen_id")
    d = qc.join(met[["inferred_met_type"]], how="inner").join(
        phys[["rheobase_i"]], how="inner")
    d = d[d["rheobase_i"].notna()]
    d = d[d["inferred_met_type"].astype(str).str.startswith("L5 ET")]
    idx = log_cpm.index.intersection(d.index); d = d.loc[idx]; n = len(idx)

    mt = d["inferred_met_type"].astype(str)
    keep = mt.value_counts()[lambda s: s >= 8].index
    mt = mt.where(mt.isin(keep), "other")
    dums = pd.get_dummies(mt, prefix="m", drop_first=True).astype(float)
    C = _m22.build_design(d[["log_libsize", "n_detected", "top50_share"]], dums, n)
    y = _m22.rank_resid(d["rheobase_i"].to_numpy(), C)
    yc = (y - y.mean()) / y.std()
    grp = mt.to_numpy()

    cols = [g for g in _m23.CATEGORIES["IEG_TF"] if g in log_cpm.columns]
    E = log_cpm.loc[idx, cols]
    E = E.loc[:, (E > 0).mean(axis=0) >= 0.25]
    Er = np.column_stack([_m22.rank_resid(E[g].to_numpy(), C) for g in E.columns])
    Er -= Er.mean(0); sd = Er.std(0)
    Er /= np.where(sd > 0, sd, np.nan)
    obs = float(np.nanmean((Er * yc[:, None]).sum(0) / (n - 1)))
    P = []
    for _ in range(N_PERM):
        yp = y.copy()
        for g in np.unique(grp):
            m = grp == g
            yp[m] = RNG.permutation(yp[m])
        P.append(yp - yp.mean())
    P = np.column_stack(P); P /= np.where(P.std(0) > 0, P.std(0), np.nan)
    nulls = np.nanmean((Er.T @ P) / (n - 1), axis=0)
    return obs, nulls, n, E.shape[1]


def main() -> None:
    apply_style()
    res = pd.read_csv(paths.V1 / "results" / "AA_v1_confirmatory.csv")
    obs_v1, nulls_v1, n_v1, ngene_v1 = v1_null_and_obs()

    # mouse M1 discovery value for the same category
    m1 = pd.read_csv(paths.M1 / "results" / "V_functional_categories_by_layer.csv")
    m1 = m1[(m1.category == "IEG_TF") & (m1.population == "L5 exc (IT+ET)")].iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # ---- A: discovery -> replication -------------------------------------
    lab = [f"M1 L5\ndiscovery\nn={int(m1.n)}",
           f"V1 L5 ET\nreplication\nn={n_v1}"]
    vals = [m1.mean_rho, obs_v1]
    nl = [(m1.null_mean, m1.null_sd), (nulls_v1.mean(), nulls_v1.std(ddof=1))]
    x = np.arange(2)
    for i, (v, (nm, ns)) in enumerate(zip(vals, nl)):
        ax_a.add_patch(plt.Rectangle((i - 0.28, nm - 1.96 * ns), 0.56, 2 * 1.96 * ns,
                                     color=NEUTRAL, alpha=0.45, zorder=1))
        ax_a.plot([i - 0.28, i + 0.28], [nm, nm], color=INK_MUTED, lw=1.0, zorder=2)
        ax_a.scatter([i], [v], s=90, facecolor=BLUE, edgecolor="white",
                     linewidth=1.0, zorder=4)
        ax_a.text(i + 0.16, v, f"{v:+.3f}", fontsize=7.5, va="center", color=INK)
    ax_a.axhline(0, color=INK_MUTED, lw=0.8, zorder=1)
    ax_a.set_xticks(x); ax_a.set_xticklabels(lab, fontsize=7.5)
    ax_a.set_xlim(-0.6, 1.7)
    ax_a.set_ylabel("IEG_TF set mean ρ vs rheobase")
    ax_a.set_title("The activity-gene programme replicates\n"
                   "grey band = permutation null 95%",
                   loc="left", color=INK, fontsize=9, linespacing=1.5)
    ax_a.grid(axis="x", visible=False)
    panel_label(ax_a, "A")

    # ---- B: the actual V1 null -------------------------------------------
    ax_b.hist(nulls_v1, bins=60, color=NEUTRAL, edgecolor="none", zorder=2)
    ax_b.axvline(obs_v1, color=RED, lw=2.0, zorder=4)
    p = float((nulls_v1 <= obs_v1).mean())
    ax_b.annotate(f"observed\n{obs_v1:+.4f}\np = {p:.4f}",
                  xy=(obs_v1, ax_b.get_ylim()[1] * 0.55),
                  xytext=(obs_v1 + 0.012, ax_b.get_ylim()[1] * 0.78),
                  fontsize=7, color=RED, fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax_b.set_xlabel("set mean ρ under the null (rheobase shuffled within type)")
    ax_b.set_ylabel("permutations")
    ax_b.set_title(f"V1 set test\n{ngene_v1} genes, {n_v1} L5 ET cells, "
                   f"{N_PERM:,} permutations",
                   loc="left", color=INK, fontsize=9, linespacing=1.5)
    panel_label(ax_b, "B")

    # ---- C: Jun's shrinkage ---------------------------------------------
    jun = res[res.test == "A1_Jun"].iloc[0]
    pts = [("mouse M1 L5 ET_1\n(type, n=32)", -0.539, None, None),
           ("mouse M1 L5 ET\n(class, n=49)", -0.386, None, None),
           ("mouse M1 L5 exc\n(n=126)", -0.371, None, None),
           ("mouse M1 glut.\n(n=410)", -0.077, None, None),
           ("mouse V1 L5 ET\n(n=436)", jun.rho, jun.ci_lo, jun.ci_hi)]
    ypos = np.arange(len(pts))[::-1]
    for yy, (lb, r, lo, hi) in zip(ypos, pts):
        col = RED if lo is not None else BLUE
        if lo is not None:
            ax_c.plot([lo, hi], [yy, yy], color=col, lw=1.8, zorder=3)
        ax_c.scatter([r], [yy], s=60, facecolor=col, edgecolor="white",
                     linewidth=0.8, zorder=4)
        ax_c.text(r, yy - 0.34, f"{r:+.3f}", fontsize=6.8, ha="center",
                  va="top", color=col)
    ax_c.axvline(0, color=INK_MUTED, lw=0.8)
    ax_c.set_yticks(ypos)
    ax_c.set_yticklabels([p[0] for p in pts], fontsize=6.8)
    ax_c.set_xlabel("Jun ρ vs rheobase")
    ax_c.set_xlim(-0.66, 0.14)
    ax_c.set_ylim(-0.75, len(pts) - 0.35)
    ax_c.set_title("No single gene replicates: Jun shrinks 4×\n"
                   "red = independent cohort, with 95% CI",
                   loc="left", color=INK, fontsize=9, linespacing=1.5)
    ax_c.grid(axis="y", visible=False)
    panel_label(ax_c, "C")

    # ---- D: the five V1 tests --------------------------------------------
    order = ["A2_IEG_TF_set", "A1_Jun", "A3_module_sign", "B1_Jarid2", "B2_Gphn"]
    nice = {"A2_IEG_TF_set": "A2  IEG_TF category", "A1_Jun": "A1  Jun",
            "A3_module_sign": "A3  150-gene module", "B1_Jarid2": "B1  Jarid2",
            "B2_Gphn": "B2  Gphn"}
    rows = []
    for t in order:
        r = res[res.test == t]
        if not len(r):
            continue
        r = r.iloc[0]
        eff = r.rho if np.isfinite(r.get("rho", np.nan)) else r.mean_rho
        pv = r.p if np.isfinite(r.get("p", np.nan)) else r.p_one_sided_neg
        rows.append((nice[t], eff, pv, t.startswith("A2")))
    ypos = np.arange(len(rows))[::-1]
    for yy, (lb, eff, pv, ok) in zip(ypos, rows):
        col = GREEN if pv < 0.025 and ok else RED if pv >= 0.025 else BLUE
        ax_d.plot([0, eff], [yy, yy], color=INK_MUTED, lw=1.1, zorder=2)
        ax_d.scatter([eff], [yy], s=64, facecolor=col, edgecolor="white",
                     linewidth=0.8, zorder=3)
        verdict = "replicated" if (pv < 0.025 and ok) else "not replicated"
        ax_d.text(0.055, yy, f"p={pv:.3g}  {verdict}", fontsize=6.6,
                  va="center", color=col if verdict == "replicated" else INK_2)
    ax_d.axvline(0, color=INK_MUTED, lw=0.8)
    ax_d.set_yticks(ypos)
    ax_d.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax_d.set_xlim(-0.15, 0.30)
    ax_d.set_xlabel("effect (ρ, or set mean ρ)")
    ax_d.set_title("All five V1 tests\nα = 0.025 per hypothesis",
                   loc="left", color=INK, fontsize=9, linespacing=1.5)
    ax_d.grid(axis="y", visible=False)
    panel_label(ax_d, "D")

    fig.tight_layout(w_pad=2.6, h_pad=2.8)
    save(fig, "Fig5_programme_replicates")
    print(f"\nV1: obs {obs_v1:+.4f}, null {nulls_v1.mean():+.4f}±"
          f"{nulls_v1.std(ddof=1):.4f}, p={p:.4f}, n={n_v1}, genes={ngene_v1}")
    print(f"M1: obs {m1.mean_rho:+.4f}, null {m1.null_mean:+.4f}±{m1.null_sd:.4f}")


if __name__ == "__main__":
    main()
