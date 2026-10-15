"""Figure 6: IEG index vs intrinsic properties across pyramidal-cell populations
(L2/3 IT, L4/5 IT, L5 IT, L5 ET, L6 IT, L6 CT), one identical model per group.

Model: feature_z ~ rapid-IEG index_z [+ C(ttype)] + log(n_genes), SE clustered by mouse;
BH-FDR across the 12 features within each group; '+stress' adds the cell-stress score.
Output: results/cross_layer_assoc.csv, figures/fig6_cross_layer_pyramidal.{svg,png}
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import scanpy as sc
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from common import EPHYS_KEY  # noqa: E402
RESULTS = paths.results(__file__)
DATA = paths.data(__file__)
from plotstyle import C as COL, save, label

GROUPS = [("L2/3 IT", r"L2/3 IT_"), ("L4/5 IT", r"L4/5 IT_"), ("L5 IT", r"L5 IT_"),
          ("L5 ET", r"L5 ET_"), ("L6 IT", r"L6 IT_"), ("L6 CT", r"L6 CT ")]
LOGF = {"Input resistance (MOhm)", "Rheobase (pA)", "Membrane time constant (ms)", "AP width (ms)",
        "Latency (ms)", "Max number of APs"}
SHORT = {"Input resistance (MOhm)": "Input resistance", "Resting membrane potential (mV)": "Resting Vm",
         "Rheobase (pA)": "Rheobase", "AP threshold (mV)": "AP threshold", "AP width (ms)": "AP width",
         "Afterhyperpolarization (mV)": "AHP", "Membrane time constant (ms)": "Membrane τ",
         "Sag ratio": "Sag ratio", "ISI adaptation index": "ISI adaptation", "Max number of APs": "Max APs",
         "Burstiness": "Burstiness", "Latency (ms)": "Latency"}

o = sc.read_h5ad(DATA / "patchseq_m1.h5ad", backed="r").obs.join(
    pd.read_csv(RESULTS / "aim1_scores.csv", index_col=0)[["IEG_rapid", "stress"]]).copy()
o["ttype"] = o["ttype"].astype(str)
o["lg"] = np.log(o["n_genes"])
o = o[o["has_ephys"]]

rows = []
for g, pat in GROUPS:
    base = o[o["ttype"].str.match(pat)]
    for f in EPHYS_KEY:
        d = base.copy()
        y = d["ephys: " + f]
        d["y"] = np.log(y.where(y > 0)) if f in LOGF else y
        d = d.dropna(subset=["y", "IEG_rapid"])
        d = d[d["ttype"].map(d["ttype"].value_counts()) >= 2]
        for c, src in [("y", "y"), ("s", "IEG_rapid"), ("st", "stress")]:
            d[c] = (d[src] - d[src].mean()) / d[src].std()
        T = " + C(ttype)" if d["ttype"].nunique() > 1 else ""
        for model, extra in [("base", ""), ("+stress", " + st")]:
            r = smf.ols("y ~ s" + T + " + lg" + extra, d).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(d["mouse"])[0]})
            rows.append({"group": g, "feature": f, "model": model, "n": int(r.nobs), "n_mice": d["mouse"].nunique(),
                         "beta": r.params["s"], "ci_lo": r.conf_int().loc["s", 0],
                         "ci_hi": r.conf_int().loc["s", 1], "p": r.pvalues["s"]})
res = pd.DataFrame(rows)
res["q_fdr"] = res.groupby(["group", "model"])["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
res["q_global"] = np.nan
m = res["model"] == "base"
res.loc[m, "q_global"] = multipletests(res.loc[m, "p"], method="fdr_bh")[1]
res.to_csv(RESULTS / "cross_layer_assoc.csv", index=False)

# ------------------------------------------------------------------ figure
gnames = [g for g, _ in GROUPS]
fig = plt.figure(figsize=(7.4, 6.4), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])

# A: IEG index level by group
ax = fig.add_subplot(gs[0, 0])
label(ax, "A")
rng = np.random.default_rng(0)
for i, (g, pat) in enumerate(GROUPS):
    v = o.loc[o["ttype"].str.match(pat), "IEG_rapid"].dropna()
    ax.scatter(v, i + rng.uniform(-0.22, 0.22, len(v)), s=5, color=COL["blue"], alpha=0.35, lw=0)
    q1, med, q3 = v.quantile([.25, .5, .75])
    ax.plot([q1, q3], [i, i], color=COL["ink"], lw=2.2, solid_capstyle="butt")
    ax.plot(med, i, "|", color=COL["ink"], ms=11, mew=2)
ax.set_yticks(range(len(gnames)), [f"{g}\n(n={int((o['ttype'].str.match(p)).sum())})" for g, p in GROUPS], fontsize=6.5)
ax.invert_yaxis()
ax.set_xlabel("Rapid-IEG index")
ax.set_title("IEG level by population", loc="left")

# B: heatmap of within-group betas
ax = fig.add_subplot(gs[0, 1:])
label(ax, "B")
b = res[m].pivot(index="group", columns="feature", values="beta").loc[gnames, EPHYS_KEY]
p = res[m].pivot(index="group", columns="feature", values="p").loc[gnames, EPHYS_KEY]
q = res[m].pivot(index="group", columns="feature", values="q_fdr").loc[gnames, EPHYS_KEY]
cmap = LinearSegmentedColormap.from_list("div", [COL["blue"], "#eeeeec", COL["orange"]])
im = ax.imshow(b.values, cmap=cmap, norm=TwoSlopeNorm(0, -0.5, 0.5), aspect="auto")
for i in range(b.shape[0]):
    for j in range(b.shape[1]):
        mark = "★" if q.iloc[i, j] < 0.05 else ("•" if p.iloc[i, j] < 0.05 else "")
        ax.text(j, i, f"{b.iloc[i, j]:+.2f}{mark}", ha="center", va="center", fontsize=5.6,
                color=COL["ink"])
ax.set_xticks(range(len(EPHYS_KEY)), [SHORT[f] for f in EPHYS_KEY], rotation=45, ha="right", fontsize=6.5)
ax.set_yticks(range(len(gnames)), gnames, fontsize=7)
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
cb.set_label("β (SD per SD IEG index)", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
ax.set_title("Within-population effects", loc="left", pad=14)
ax.text(1.0, 1.02, f"• p<0.05   ★ q<0.05 within row   ·   global FDR (72 tests): min q = {res.loc[m, 'q_global'].min():.2f}",
        transform=ax.transAxes, fontsize=6, color=COL["ink"], ha="right", va="bottom")

# C-E: forest for the three features that were nominal somewhere
for k, f in enumerate(["Rheobase (pA)", "Resting membrane potential (mV)", "Membrane time constant (ms)"]):
    ax = fig.add_subplot(gs[1, k])
    label(ax, "CDE"[k])
    for j, (model, col) in enumerate([("base", COL["violet"]), ("+stress", COL["aqua"])]):
        s = res[(res["feature"] == f) & (res["model"] == model)].set_index("group").loc[gnames]
        y = np.arange(len(gnames)) + (j - 0.5) * 0.28
        ax.hlines(y, s["ci_lo"], s["ci_hi"], color=col, lw=1.2)
        ax.plot(s["beta"], y, "o", color=col, ms=3.8,
                label={"base": "IEG index", "+stress": "+ stress-gene adj."}[model])
    ax.axvline(0, color=COL["grey"], lw=0.8)
    ax.set_yticks(range(len(gnames)), gnames if k == 0 else [""] * len(gnames), fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.8, 0.8)
    ax.set_xlabel("β (95% CI)")
    ax.set_title(SHORT[f], loc="left")
    if k == 1:
        ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
save(fig, "fig6_cross_layer_pyramidal")

pd.set_option("display.width", 200)
print(res[m].sort_values("p").head(10)[["group", "feature", "n", "beta", "p", "q_fdr", "q_global"]].round(3).to_string(index=False))
print("global min q:", res.loc[m, "q_global"].min().round(3))
print(o.groupby(o["ttype"].str.extract(r"^(L2/3 IT|L4/5 IT|L5 IT|L5 ET|L6 IT|L6 CT)")[0])["IEG_rapid"].median().round(2))
