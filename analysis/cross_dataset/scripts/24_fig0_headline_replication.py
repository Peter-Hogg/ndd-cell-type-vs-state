"""Figure 0: the M1 -> V1 activity-response gene-set effect on rheobase, drawn from the
m1_v1_validation tables (read-only; no values recomputed).
Output: figures/fig0_replicated_set_effect.{svg,png}"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import VALIDATION_TABLES as T  # noqa: E402
from plotstyle import C, save, label

coh = {"m1": ("M1 · 126 cells / 54 mice", C["blue"]), "v1": ("V1 · 436 cells / 163 mice", C["orange"])}
spec = pd.read_csv(T / "normalization_and_covariate_specifications.csv")
spec = spec[spec["covariates"] == "primary"]
pg = pd.read_csv(T / "m1_per_gene_effects.csv").merge(pd.read_csv(T / "v1_per_gene_effects.csv"), on="gene",
                                                       suffixes=("_m1", "_v1"))
fam = pd.read_csv(T / "channel_receptor_set_results.csv")
scaf = pd.read_csv(T / "scaffolding_panel_results.csv")
prim = pd.read_csv(T / "primary_cohort_results.csv").set_index("cohort")
pool = pd.read_csv(T / "pooling_benchmark_summary.csv").set_index("model")

fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.8), constrained_layout=True)

# A: primary estimate across normalisations
ax = axs[0, 0]
label(ax, "A")
norms = [("logCPM", "log-CPM (primary)"), ("housekeeping_ratio", "housekeeping ratio"), ("within_cell_rank", "within-cell rank")]
for j, (c, (lab, col)) in enumerate(coh.items()):
    for i, (n, nl) in enumerate(norms):
        r = spec[(spec["cohort"] == c) & (spec["normalization"] == n)].iloc[0]
        y = i + (j - 0.5) * 0.3
        ax.hlines(y, r["ci_lo"], r["ci_hi"], color=col, lw=1.6 if i == 0 else 1.0)
        ax.plot(r["effect"], y, "o", color=col, ms=6 if i == 0 else 4, mfc=col if i == 0 else "white",
                label=lab if i == 0 else None)
ax.axvline(0, color=C["grey"], lw=0.8)
ax.set_yticks(range(3), [n[1] for n in norms])
ax.invert_yaxis()
ax.set_xlabel("Mean gene-wise partial rank ρ with rheobase\n(33 activity-response TFs; animal-bootstrap 95% CI)")
ax.set_title("Set effect replicates M1 → V1", loc="left")
ax.legend(fontsize=6.5, loc="lower right")

# B: per-gene effects in both cohorts
ax = axs[0, 1]
label(ax, "B")
ax.axhline(0, color=C["grey"], lw=0.8); ax.axvline(0, color=C["grey"], lw=0.8)
ax.scatter(pg["effect_m1"], pg["effect_v1"], s=16, color=C["violet"], alpha=0.75, lw=0)
for g in ["Jun", "Junb", "Fos", "Egr1", "Npas4", "Atf4", "Nr4a1"]:
    r = pg[pg["gene"] == g]
    if len(r):
        ax.annotate(g, (r["effect_m1"].iloc[0], r["effect_v1"].iloc[0]), fontsize=6, xytext=(3, 2),
                    textcoords="offset points", fontstyle="italic")
ax.plot(prim.loc["m1", "effect"], prim.loc["v1", "effect"], "D", color=C["ink"], ms=6, label="set mean")
ax.set_xlabel("M1 per-gene partial ρ")
ax.set_ylabel("V1 per-gene partial ρ")
ax.set_title("Distributed, not one gene", loc="left")
ax.legend(fontsize=6.5, loc="upper left")
n_both_neg = int(((pg["effect_m1"] < 0) & (pg["effect_v1"] < 0)).sum())
ax.text(0.03, 0.03, f"{n_both_neg}/{len(pg)} genes negative in both", transform=ax.transAxes, ha="left",
        fontsize=6.5, color=C["ink"])

# C: specificity — other fixed gene families
ax = axs[1, 0]
label(ax, "C")
rows = [("activity-response TFs", None)]
fam_l = {"ion_channel_or_auxiliary": "ion channels/aux.", "neurotransmitter_receptor": "NT receptors"}
scaf_l = {"ca_camp_adaptors": "Ca/cAMP adaptors", "structural_late": "structural (late)",
          "synaptic_plasticity": "synaptic plasticity", "combined_scaffolding_camk": "scaffold+CaMK union"}
labels = ["activity-response TFs"] + list(fam_l.values()) + list(scaf_l.values())
for j, (c, (lab, col)) in enumerate(coh.items()):
    vals = [(prim.loc[c, "effect"], prim.loc[c, "ci_lo"], prim.loc[c, "ci_hi"])]
    for k in fam_l:
        r = fam[(fam["cohort"] == c) & (fam["family"] == k)].iloc[0]
        vals.append((r["effect"], r["ci_lo"], r["ci_hi"]))
    for k in scaf_l:
        r = scaf[(scaf["cohort"] == c) & (scaf["panel"] == k)].iloc[0]
        vals.append((r["effect"], r["ci_lo"], r["ci_hi"]))
    for i, (e, lo, hi) in enumerate(vals):
        y = i + (j - 0.5) * 0.3
        ax.hlines(y, lo, hi, color=col, lw=1.2)
        ax.plot(e, y, "o", color=col, ms=4)
ax.axvline(0, color=C["grey"], lw=0.8)
ax.axhspan(-0.5, 0.5, color=C["light"], alpha=0.4, lw=0)
ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
ax.invert_yaxis()
ax.set_xlabel("Mean partial rank ρ with rheobase (95% CI)")
ax.set_title("Other gene families: no matching effect", loc="left")

# D: pooling benchmark
ax = axs[1, 1]
label(ax, "D")
order = [("naive", "naive pooled"), ("family", "+ cell family"), ("fine_type", "+ fine type"),
         ("type_technical", "+ type & technical")]
x = np.arange(len(order))
ax.bar(x, [pool.loc[m, "hits_q05"] for m, _ in order], color=[C["orange"]] + [C["blue"]] * 3, width=0.6)
for i, (m, _) in enumerate(order):
    ax.text(i, pool.loc[m, "hits_q05"] + 0.12, f"{int(pool.loc[m, 'hits_q05'])}/28\n|ρ̃| {pool.loc[m, 'median_abs_rho']:.3f}",
            ha="center", fontsize=6)
ax.set_xticks(x, [o[1] for o in order], fontsize=6.5, rotation=15)
ax.set_ylabel("Gene–ephys associations at q<0.05\n(|ρ̃| = median |ρ|)")
ax.set_ylim(0, 7.8)
ax.set_title("Naive pooling inflates hits 6×", loc="left")
save(fig, "fig0_replicated_set_effect")
print("written", n_both_neg, len(pg))
