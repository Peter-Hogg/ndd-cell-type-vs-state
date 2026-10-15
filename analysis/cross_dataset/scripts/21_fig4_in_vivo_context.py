"""Figure 4: in vivo trait vs state (IBL) and the Patch-seq-derived prediction.
Also writes results/aim2_patchseq_tercile_prediction.csv (high- vs low-IEG cells, within type)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import scanpy as sc
RESULTS = paths.results(__file__)
FIGURES = paths.figures(__file__)
DATA = paths.data(__file__)

from plotstyle import C, save, label

rel = pd.read_csv(paths.IBL / "results" / "aim2_reliability.csv")
part = pd.read_csv(paths.IBL / "results" / "aim2_variance_partition.csv")
units = pd.read_parquet(paths.IBL / "data" / "ibl_motor_units.parquet")
units = units[(units["good"] == 1) & (units["spike_width_class"] == "wide") & (units["region_group"] == "MOp")
              & (units["taskepoch_dur_s"] > 300) & (units["spont_dur_s"] > 120)]

# ---- Patch-seq prediction: high vs low rapid-IEG tercile within t-type (glutamatergic)
a = sc.read_h5ad(paths.M1 / "data" / "patchseq_m1.h5ad", backed="r")
obs = a.obs.join(pd.read_csv(paths.M1 / "results" / "aim1_scores.csv", index_col=0)[["IEG_rapid"]])
g = obs[(obs["class"] == "Glutamatergic") & obs["has_ephys"]].copy()
g = g[g["ttype"].map(g["ttype"].value_counts()) >= 6]
g["ieg_rank"] = g.groupby("ttype")["IEG_rapid"].rank(pct=True)
g["tercile"] = np.select([g["ieg_rank"] <= 1 / 3, g["ieg_rank"] > 2 / 3], ["low", "high"], "mid")
pred = []
for f in ["Burstiness", "ISI adaptation index", "Max number of APs", "Rheobase (pA)", "Input resistance (MOhm)"]:
    col = "ephys: " + f
    v = g[col] - g.groupby("ttype")[col].transform("median")      # within-type centred
    hi, lo = v[g["tercile"] == "high"].dropna(), v[g["tercile"] == "low"].dropna()
    ks = stats.ks_2samp(hi, lo)
    pred.append({"feature": f, "n_high": len(hi), "n_low": len(lo), "median_high": hi.median(),
                 "median_low": lo.median(), "ks_D": ks.statistic, "ks_p": ks.pvalue,
                 "mw_p": stats.mannwhitneyu(hi, lo).pvalue})
pred = pd.DataFrame(pred)
pred.to_csv(RESULTS / "aim2_patchseq_tercile_prediction.csv", index=False)

fig, axs = plt.subplots(2, 3, figsize=(7.4, 5.4), constrained_layout=True)
# A: reliability across splits
ax = axs[0, 0]
label(ax, "A")
splits = ["interleaved (odd vs even blocks)", "first vs second half", "task vs passive/spontaneous"]
slab = ["Interleaved\nblocks", "Session\nhalves", "Task vs\npassive"]
stat_c = {"log_rate": C["blue"], "lv": C["aqua"], "burst": C["orange"]}
stat_l = {"log_rate": "log rate", "lv": "LV (irregularity)", "burst": "burst fraction"}
for k, st in enumerate(stat_c):
    for rg, mk in [("MOp", "o"), ("MOs", "s")]:
        d = rel[(rel["region"] == rg) & (rel["cls"] == "RS") & (rel["stat"] == st)].set_index("split").loc[splits]
        x = np.arange(3) + (k - 1) * 0.22 + (0.05 if rg == "MOs" else -0.05)
        ax.plot(x, d["icc"], mk, color=stat_c[st], ms=4.5, mfc=stat_c[st] if rg == "MOp" else "white",
                label=f"{stat_l[st]}" if rg == "MOp" else None)
ax.plot([], [], "o", color=C["grey"], label="MOp (filled)"); ax.plot([], [], "s", color=C["grey"], mfc="white", label="MOs (open)")
ax.set_xticks(range(3), slab, fontsize=6.5)
ax.set_ylim(0, 1.02)
ax.set_ylabel("ICC across split (RS units)")
ax.set_title("Stability by timescale", loc="left")
ax.legend(fontsize=6, loc="lower left")

# B, C: task vs spontaneous scatter (MOp RS)
for k, (xa, xb, lab, lg) in enumerate([("taskepoch_rate_hz", "spont_rate_hz", "Firing rate (Hz)", True),
                                       ("taskepoch_frac_isi_lt10ms", "spont_frac_isi_lt10ms", "Burst fraction (ISI<10 ms)", False)]):
    ax = axs[0, 1 + k]
    label(ax, "BC"[k])
    x, y = units[xb], units[xa]
    if lg:
        x, y = x + 0.05, y + 0.05
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.scatter(x, y, s=4, color=C["blue"], alpha=0.4, lw=0)
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, color=C["grey"], lw=0.8, ls="--")
    r = rel[(rel["region"] == "MOp") & (rel["cls"] == "RS") & (rel["split"] == splits[2]) &
            (rel["stat"] == ("log_rate" if lg else "burst"))].iloc[0]
    ax.text(0.04, 0.96, f"MOp RS, n={int(r['n_units'])}\nICC = {r['icc']:.2f}", transform=ax.transAxes, va="top", fontsize=7)
    ax.set_xlabel(f"{lab.split(' (')[0]}, passive")
    ax.set_ylabel(f"{lab.split(' (')[0]}, task")
    ax.set_title(["Rate re-sorts with context", "Bursts preserved"][k], loc="left")

# D: variance partition
ax = axs[1, 0]
label(ax, "D")
rows = [(rg, st) for rg in ["MOp", "MOs"] for st in ["log_rate", "burst"]]
for i, (rg, st) in enumerate(rows):
    p = part[(part["region"] == rg) & (part["stat"] == st)].iloc[0]
    ceil = rel[(rel["region"] == rg) & (rel["cls"] == "all") & (rel["stat"] == st) & (rel["split"] == splits[1])]["icc"].iloc[0]
    t = max(p["adjR2_layer_x_class"], 0)
    s = max(p["adjR2_layer_x_class_plus_session"] - t, 0)
    ax.barh(i, t, color=C["violet"], height=0.6, label="layer × RS/FS" if i == 0 else None)
    ax.barh(i, s, left=t, color=C["yellow"], height=0.6, label="+ session" if i == 0 else None)
    ax.plot(ceil, i, "|", color=C["ink"], ms=12, mew=1.5, label="split-half ICC (stable ceiling)" if i == 0 else None)
ax.set_yticks(range(len(rows)), [f"{rg} {stat_l[st]}" for rg, st in rows])
ax.set_xlim(0, 1)
ax.set_xlabel("Fraction of between-unit variance")
ax.set_title("Coarse type explains little", loc="left")
ax.invert_yaxis()
ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, handlelength=1)

# E: Patch-seq prediction
ax = axs[1, 1]
label(ax, "E")
f = "ephys: Burstiness"
for terc, col in [("low", C["aqua"]), ("high", C["violet"])]:
    v = (g[f] - g.groupby("ttype")[f].transform("median"))[g["tercile"] == terc].dropna().sort_values()
    ax.step(v, np.arange(1, len(v) + 1) / len(v), color=col, lw=1.5, label=f"{terc} rapid-IEG tercile (n={len(v)})")
r = pred.set_index("feature").loc["Burstiness"]
ax.text(0.97, 0.05, f"KS p = {r['ks_p']:.2f}", transform=ax.transAxes, ha="right", fontsize=7)
ax.set_xlabel("In vitro burstiness (within-t-type centred)")
ax.set_ylabel("Cumulative fraction")
ax.set_title("Patch-seq predicts no shift", loc="left")
ax.legend(fontsize=6, loc="upper left")

ax = axs[1, 2]
label(ax, "F")
ax.barh(range(len(pred)), pred["ks_D"], color=C["light"], height=0.6)
for i, r in pred.iterrows():
    ax.text(r["ks_D"] + 0.005, i, f"p={r['ks_p']:.2f}", va="center", fontsize=6.5)
ax.set_yticks(range(len(pred)), [p.replace(" (", "\n(") for p in pred["feature"]], fontsize=6.5)
ax.invert_yaxis()
ax.set_xlim(0, 0.35)
ax.set_xlabel("KS distance\n(high vs low IEG tercile)")
ax.set_title("No IEG phenotype", loc="left")
save(fig, "fig4_in_vivo_trait_vs_state")
print(pred.round(3).to_string())
