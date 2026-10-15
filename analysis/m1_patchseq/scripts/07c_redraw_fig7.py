"""Redraw figure 7 from saved results of 07_per_cluster_slopes.py (no re-permutation)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)
"""Per-cluster IEG -> ephys slopes in M1 Patch-seq.

For every transcriptomic cluster (RNA type) with >= MIN_N cells with ephys, fit
  feature_z ~ IEG_rapid_z + log(n_genes)      (within that cluster only, HC3 SE)
for the 12 key features. Then ask:
  1. Do slopes differ between clusters? Cochran's Q / I^2 per feature from the
     per-cluster betas, and a pooled interaction F-test (IEG x cluster).
  2. Are there more nominal hits (p < 0.05) than chance? Permutation null:
     shuffle the IEG score within each cluster (5,000x), recount hits.
  3. Random-effects pooled slope per feature (DerSimonian-Laird).
Outputs: results/cluster_slopes.csv, results/cluster_slope_heterogeneity.csv,
         results/cluster_slope_permutation.csv, figures/fig7_per_cluster_slopes.{svg,png}
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import scanpy as sc
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from common import EPHYS_KEY  # noqa: E402
RESULTS = paths.results(__file__)
DATA = paths.data(__file__)
from plotstyle import C as COL, save, label

MIN_N = 15
N_PERM = 5000
rng = np.random.default_rng(0)
LOGF = {"Input resistance (MOhm)", "Rheobase (pA)", "Membrane time constant (ms)", "AP width (ms)",
        "Latency (ms)", "Max number of APs"}
SHORT = {"Input resistance (MOhm)": "Input res.", "Resting membrane potential (mV)": "Resting Vm",
         "Rheobase (pA)": "Rheobase", "AP threshold (mV)": "AP thresh.", "AP width (ms)": "AP width",
         "Afterhyperpolarization (mV)": "AHP", "Membrane time constant (ms)": "Membrane τ",
         "Sag ratio": "Sag ratio", "ISI adaptation index": "ISI adapt.", "Max number of APs": "Max APs",
         "Burstiness": "Burstiness", "Latency (ms)": "Latency"}

o = sc.read_h5ad(DATA / "patchseq_m1.h5ad", backed="r").obs.join(
    pd.read_csv(RESULTS / "aim1_scores.csv", index_col=0)[["IEG_rapid"]]).copy()
o = o[o["has_ephys"]].copy()
o["ttype"] = o["ttype"].astype(str)
o["lg"] = np.log(o["n_genes"])
for f in EPHYS_KEY:
    y = o["ephys: " + f]
    o[f] = np.log(y.where(y > 0)) if f in LOGF else y
counts = o["ttype"].value_counts()
clusters = counts[counts >= MIN_N].index.tolist()
o = o[o["ttype"].isin(clusters)].copy()
cls_of = o.groupby("ttype")["class"].first()
# order clusters: glutamatergic first, then by family, then name
clusters = sorted(clusters, key=lambda t: (cls_of[t] != "Glutamatergic", t))
print(f"{len(clusters)} clusters with >= {MIN_N} cells; {len(o)} cells")


def zs(v):
    return (v - v.mean()) / v.std()


def cluster_fits(df, score_col):
    """beta, se, p of score within each cluster for each feature."""
    out = []
    for t, g in df.groupby("ttype"):
        for f in EPHYS_KEY:
            d = g[[f, score_col, "lg"]].dropna()
            if len(d) < MIN_N or d[f].std() == 0:
                continue
            X = sm.add_constant(np.c_[zs(d[score_col]), d["lg"] - d["lg"].mean()])
            r = sm.OLS(zs(d[f]).values, X).fit(cov_type="HC3")
            out.append((t, f, len(d), r.params[1], r.bse[1], r.pvalues[1]))
    return pd.DataFrame(out, columns=["ttype", "feature", "n", "beta", "se", "p"])



obs_fit = pd.read_csv(RESULTS / "cluster_slopes.csv")
het = pd.read_csv(RESULTS / "cluster_slope_heterogeneity.csv")
perm = pd.read_csv(RESULTS / "cluster_slope_permutation.csv")
null_hits = np.load(RESULTS / "cluster_slope_null_hits.npy")
obs_hits = int(perm["observed_hits_p05"].iloc[0])
# ---- figure
fig = plt.figure(figsize=(7.4, 7.6), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1.9, 1], width_ratios=[1.6, 1])
ax = fig.add_subplot(gs[0, :])
label(ax, "A")
B = obs_fit.pivot(index="ttype", columns="feature", values="beta").reindex(index=clusters, columns=EPHYS_KEY)
P = obs_fit.pivot(index="ttype", columns="feature", values="p").reindex(index=clusters, columns=EPHYS_KEY)
Q = obs_fit.pivot(index="ttype", columns="feature", values="q_all").reindex(index=clusters, columns=EPHYS_KEY)
cmap = LinearSegmentedColormap.from_list("div", [COL["blue"], "#eeeeec", COL["orange"]])
im = ax.imshow(B.values.T, cmap=cmap, norm=TwoSlopeNorm(0, -0.8, 0.8), aspect="auto")
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        if np.isnan(P.iloc[i, j]):
            continue
        if Q.iloc[i, j] < 0.05:
            ax.text(i, j, "★", ha="center", va="center", fontsize=7, color=COL["ink"])
        elif P.iloc[i, j] < 0.05:
            ax.text(i, j, "•", ha="center", va="center", fontsize=8, color=COL["ink"])
xl = [f"{t} ({counts[t]})" for t in clusters]
ax.set_xticks(range(len(clusters)), xl, rotation=60, ha="right", fontsize=5.8)
ax.set_yticks(range(len(EPHYS_KEY)), [SHORT[f] for f in EPHYS_KEY], fontsize=6.5)
n_glut = sum(cls_of[t] == "Glutamatergic" for t in clusters)
ax.axvline(n_glut - 0.5, color=COL["ink"], lw=1)
ax.text((n_glut - 1) / 2, -0.9, "glutamatergic", ha="center", fontsize=7, color=COL["ink"])
ax.text(n_glut + (len(clusters) - n_glut - 1) / 2, -0.9, "GABAergic", ha="center", fontsize=7, color=COL["ink"])
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
cb.set_label("within-cluster β (SD per SD IEG)", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
ax.set_title(f"IEG → ephys slope fitted separately in each cluster ({len(clusters)} clusters with ≥{MIN_N} cells)",
             loc="left", pad=30)
ax.text(1.0, 1.01, "• p<0.05   ★ q<0.05 across all tests", transform=ax.transAxes, ha="right", va="bottom",
        fontsize=6, color=COL["ink"])

ax = fig.add_subplot(gs[1, 0])
label(ax, "B")
hs = het.set_index("feature").loc[EPHYS_KEY]
y = np.arange(len(EPHYS_KEY))
ax.hlines(y, hs["ci_lo"], hs["ci_hi"], color=COL["violet"], lw=1.3)
ax.plot(hs["beta_random"], y, "o", color=COL["violet"], ms=4)
for i, (f, r) in enumerate(hs.iterrows()):
    ax.text(0.62, i, f"I²={100*r['I2']:.0f}%  p_het" + ("<0.01" if r['p_interaction'] < 0.01 else "=" + f"{r['p_interaction']:.2f}"), va="center", fontsize=5.8,
            color=COL["ink"])
ax.axvline(0, color=COL["grey"], lw=0.8)
ax.set_yticks(y, [SHORT[f] for f in EPHYS_KEY], fontsize=6.5)
ax.invert_yaxis()
ax.set_xlim(-0.4, 1.15)
ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
ax.set_xlabel("Pooled within-cluster β (random effects, 95% CI)")
ax.set_title("Pooled slope + heterogeneity", loc="left")

ax = fig.add_subplot(gs[1, 1])
label(ax, "C")
bins = np.arange(null_hits.min() - 0.5, max(null_hits.max(), obs_hits) + 1.5)
ax.hist(null_hits, bins=bins, color=COL["light"], edgecolor="white")
ax.axvline(obs_hits, color=COL["orange"], lw=2)
r = perm.iloc[0]
ax.text(obs_hits, ax.get_ylim()[1] * 0.92, f" observed {obs_hits}\n perm p = {r['p_perm_hits']:.3f}\n strongest single\n test: perm p = {r['p_perm_min_p']:.3f}",
        color=COL["orange"], fontsize=6.5, va="top")
ax.set_xlabel(f"Nominal hits (p<0.05) of {int(r['n_tests'])} tests")
ax.set_ylabel("Shuffles")
ax.set_title("Hits vs within-cluster shuffle", loc="left")
save(fig, "fig7_per_cluster_slopes")

