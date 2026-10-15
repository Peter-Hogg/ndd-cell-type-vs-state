"""Figure 5: L5 pyramidal focus — IEG index vs intrinsic properties (from 06 / 06b)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import statsmodels.formula.api as smf
RESULTS = paths.results(__file__)
DATA = paths.data(__file__)
from plotstyle import C, save, label

res = pd.read_csv(RESULTS / "l5_ieg_index_assoc.csv")
rob = pd.read_csv(RESULTS / "l5_it_robustness.csv")
FAM_C = {"all L5": C["ink"], "IT only": C["blue"], "ET only": C["orange"]}

fig = plt.figure(figsize=(7.4, 6.6), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])

# A: forest, core set, rapid index, all 29 features, three subsets
ax = fig.add_subplot(gs[:, 0])
label(ax, "A")
d = res[(res["cell_set"] == "core") & (res["index"] == "IEG_rapid")]
order = d[d["subset"] == "all L5"].sort_values("beta")["feature"].tolist()
for k, sub in enumerate(["all L5", "IT only", "ET only"]):
    s = d[d["subset"] == sub].set_index("feature").reindex(order)
    y = np.arange(len(order)) + (k - 1) * 0.25
    ax.hlines(y, s["ci_lo"], s["ci_hi"], color=FAM_C[sub], lw=0.8, alpha=0.8)
    hit = s["q_fdr"] < 0.05
    ax.plot(s["beta"][~hit], y[~hit.values], "o", color=FAM_C[sub], ms=2.8,
            label=f"{sub} (n={int(s['n'].max())})")
    ax.plot(s["beta"][hit], y[hit.values], "*", color=FAM_C[sub], ms=7)
ax.axvline(0, color=C["grey"], lw=0.8)
ax.set_yticks(range(len(order)), order, fontsize=5.5)
ax.set_xlabel("β (SD feature per SD rapid-IEG index)\nwithin t-type, 95% CI")
ax.set_title("L5 pyramidal: 29 features", loc="left")
ax.plot([], [], "*", color=C["grey"], ms=7, label="q<0.05 within family\n(none survive global FDR)")
ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.4, -0.11), ncol=2)

# B, C: L5 IT scatter (within-type residuals)
a = sc.read_h5ad(DATA / "patchseq_m1.h5ad", backed="r")
obs = a.obs.join(pd.read_csv(RESULTS / "aim1_scores.csv", index_col=0)[["IEG_rapid"]]).copy()
obs["ttype"] = obs["ttype"].astype(str)
for k, (feat, ylab) in enumerate([("Rheobase (pA)", "log rheobase"), ("Sag ratio", "sag ratio")]):
    ax = fig.add_subplot(gs[0, 1 + k])
    label(ax, "BC"[k])
    for sname, mask, col, mk in [("L5 IT (discovery)", obs["ttype"].str.match(r"L5 IT_"), C["blue"], "o"),
                                 ("L4/5 IT (replication)", obs["ttype"].str.match(r"L4/5 IT_"), C["aqua"], "s")]:
        q = obs[mask & obs["has_ephys"]].copy()
        v = q["ephys: " + feat]
        q["y"] = np.log(v.where(v > 0)) if feat == "Rheobase (pA)" else v
        q = q.dropna(subset=["y"])
        ry = q["y"] - q.groupby("ttype")["y"].transform("mean")
        rx = q["IEG_rapid"] - q.groupby("ttype")["IEG_rapid"].transform("mean")
        ax.scatter(rx, ry, s=9, color=col, marker=mk, alpha=0.6, lw=0, label=sname)
        b = np.polyfit(rx, ry, 1)
        xs = np.linspace(rx.min(), rx.max(), 2)
        ax.plot(xs, np.polyval(b, xs), color=col, lw=1.5)
        r = rob[(rob["set"] == sname) & (rob["feature"] == feat) & (rob["model"] == "base")].iloc[0]
        ax.text(0.98, 0.97 - 0.07 * (sname.startswith("L4")), f"{sname.split(' (')[0]}: β={r['beta']:+.2f}, p={r['p']:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6, color=col)
    ax.set_xlabel("Rapid-IEG index\n(within-type residual)")
    ax.set_ylabel(f"{ylab} (within-type residual)")
    ax.set_title(["L5 IT: lower rheobase", "L5 IT: more sag"][k], loc="left")
    if k == 0:
        ax.legend(fontsize=5.5, loc="lower left")

# D: robustness / replication forest
ax = fig.add_subplot(gs[1, 1])
label(ax, "D")
rows = [("L5 IT (discovery)", "base", "L5 IT"), ("L5 IT (discovery)", "all covariates", "L5 IT + covariates"),
        ("L5 IT_1 only", "base", "L5 IT_1 only"), ("L4/5 IT (replication)", "base", "L4/5 IT (held out)"),
        ("L5 ET", "base", "L5 ET")]
for j, (feat, col) in enumerate([("Rheobase (pA)", C["violet"]), ("Sag ratio", C["yellow"])]):
    for i, (sname, model, lab) in enumerate(rows):
        r = rob[(rob["set"] == sname) & (rob["feature"] == feat) & (rob["model"] == model)].iloc[0]
        y = i + (j - 0.5) * 0.3
        ax.hlines(y, r["ci_lo"], r["ci_hi"], color=col, lw=1.2)
        ax.plot(r["beta"], y, "o", color=col, ms=4, label=feat.split(" (")[0] if i == 0 else None)
ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
ax.axvline(0, color=C["grey"], lw=0.8)
ax.set_yticks(range(len(rows)), [r[2] for r in rows], fontsize=6.5)
ax.invert_yaxis()
ax.set_xlabel("β (95% CI)")
ax.set_title("Inconsistent across groups", loc="left")

# E: p-value distribution over every test run
ax = fig.add_subplot(gs[1, 2])
label(ax, "E")
ax.hist(res["p"], bins=20, range=(0, 1), color=C["light"], edgecolor="white")
ax.axhline(len(res) / 20, color=C["ink"], ls="--", lw=0.8)
ax.text(0.5, len(res) / 20 * 1.05, "uniform (no signal)", ha="center", fontsize=6)
ax.set_xlabel("p-value (all 516 tests)")
ax.set_ylabel("Count")
ax.set_title(f"Global FDR: min q = {res['q_global'].min():.2f}", loc="left")
save(fig, "fig5_l5_ieg_index")
print("written")
