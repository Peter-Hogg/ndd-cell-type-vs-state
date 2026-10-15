"""Figures 1-3: transcriptomic type vs IEG state (Patch-seq + Allen control) and ephys.
SVG with editable text + 300-dpi PNG."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from common import EPHYS_KEY  # noqa: E402
RESULTS = paths.results(__file__)
FIGURES = paths.figures(__file__)
DATA = paths.data(__file__)
from plotstyle import C, CLASS_C, SET_LABEL, save, label, null_panel


# ======================================================================= Figure 1
scores = pd.read_csv(paths.M1 / "results" / "aim1_scores.csv", index_col=0)
part = pd.read_csv(paths.M1 / "results" / "aim1_score_variance_partition.csv")
gnull = pd.read_csv(paths.M1 / "results" / "aim1_geneset_variance_null.csv").rename(columns={"scope": "class"})
coh = pd.read_csv(paths.M1 / "results" / "aim1_ieg_coherence.csv")
cls = pd.read_csv(paths.M1 / "results" / "aim1_classification.csv")

fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
# A: variance partition of the rapid-IEG score
ax = axs[0, 0]
label(ax, "A")
rel = coh.set_index(["class", "gene_set"])["split_half_reliability"]
for i, cl in enumerate(["Glutamatergic", "GABAergic"]):
    r = part[(part["scope"] == cl) & (part["score"] == "IEG_rapid")].iloc[0]
    t = max(r["adjR2_ttype"], 0)
    mo = max(r["adjR2_mouse_within_type"], 0) * (1 - t)
    rest = 1 - t - mo
    rel_ = rel[(cl, "IEG_rapid")]
    coherent = rest * rel_ - 0  # reliable (signal) part of the residual, upper bound
    noise = rest - coherent
    segs = [(t, C["violet"], "t-type"), (mo, C["yellow"], "Mouse (within type)"),
            (coherent, C["aqua"], "Within-type, reliable"), (noise, C["light"], "Unreliable (gene sampling)")]
    left = 0
    for w, col, lab in segs:
        ax.barh(i, w - 0.004, left=left, color=col, height=0.55, label=lab if i == 0 else None)
        if w > 0.06:
            ax.text(left + w / 2, i, f"{100*w:.0f}%", ha="center", va="center", fontsize=7,
                    color="white" if col in (C["violet"],) else C["ink"])
        left += w
ax.set_yticks([0, 1], ["Glutamatergic\n(n=410)", "GABAergic\n(n=817)"])
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("Fraction of rapid-IEG score variance")
ax.set_title("IEG state is mostly within-type", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=7)

# B: type-specificity of IEG genes vs matched null
ax = axs[0, 1]
label(ax, "B")
d = gnull[(gnull["stat"] == "r2_ttype") & gnull["class"].isin(["Glutamatergic", "GABAergic"])]
null_panel(ax, d, "observed_median", "null_q025", "null_q975", "null_median",
           "IEGs are not less type-specific", "Median per-gene adj. R² explained by t-type")
ax.legend(loc="lower right", fontsize=7)

# C: within-type coherence
ax = axs[1, 0]
label(ax, "C")
null_panel(ax, coh.assign(lo=np.nan), "mean_pairwise_r_within_type", "null_median", "null_q975",
           "null_median", "Rapid IEGs co-vary within type", "Mean within-type pairwise r")
ax.text(0.98, 0.02, "bar: null median–97.5th pct.\n(expression-matched genes)", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=6.5, color=C["grey"])
ax.legend(loc="upper right", fontsize=7)

# D: t-type decoding with/without IEGs
ax = axs[1, 1]
label(ax, "D")
conds = [("acc_hvg", "All HVGs"), ("acc_hvg_minus_IEG", "HVGs − IEGs"),
         ("acc_hvg_minus_random", "HVGs − random"), ("acc_IEG_only", "IEGs only"),
         ("acc_random_matched_mean", "Matched random set")]
w = 0.38
for j, cl in enumerate(["Glutamatergic", "GABAergic"]):
    r = cls[cls["scope"] == cl].iloc[0]
    x = np.arange(len(conds)) + (j - 0.5) * w
    ax.bar(x, [r[c] for c, _ in conds], width=w - 0.04, color=CLASS_C[cl], label=cl)
    ax.hlines(r["majority_class_rate"], x[0] - w / 2, x[-1] + w / 2, color=CLASS_C[cl], ls=":", lw=1)
ax.set_xticks(range(len(conds)), [l for _, l in conds], rotation=30, ha="right")
ax.set_ylabel("t-type kNN accuracy (5-fold CV)")
ax.set_title("IEG removal leaves t-types intact", loc="left")
ax.legend(fontsize=7, loc="upper right")
ax.text(2.2, cls["majority_class_rate"].max() + 0.012, "dotted: majority-class rate", fontsize=6.5,
        color=C["grey"], ha="center")
save(fig, "fig1_patchseq_type_vs_state")

# ======================================================================= Figure 2
cvn = pd.read_csv(paths.ATLAS / "results" / "aim1b_ieg_cells_vs_nuclei.csv")
tsn = pd.read_csv(paths.ATLAS / "results" / "aim1b_type_specificity_null.csv")
ht = pd.read_csv(paths.M1 / "results" / "aim1_hold_time.csv")
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9), constrained_layout=True)
ax = axs[0]
label(ax, "A")
d = cvn[(cvn["counts"] == "exon+intron") & (cvn["class"] == "Glutamatergic")].pivot(
    index="gene", columns="prep", values="mean_logcpm")
d = d.sort_values("cells")
for i, (g, r) in enumerate(d.iterrows()):
    ax.plot([r["nuclei"], r["cells"]], [i, i], color=C["light"], lw=2, zorder=1)
    ax.plot(r["nuclei"], i, "o", color=C["aqua"], ms=5, label="Nuclei" if i == 0 else None, zorder=2)
    ax.plot(r["cells"], i, "o", color=C["violet"], ms=5, label="Dissociated cells" if i == 0 else None, zorder=2)
ax.set_yticks(range(len(d)), d.index, fontstyle="italic")
ax.set_xlabel("Mean log(CPM+1), glutamatergic")
ax.set_title("Dissociation induces IEGs", loc="left")
ax.legend(fontsize=7, loc="lower right")

ax = axs[1]
label(ax, "B")
d = tsn[(tsn["labels"] == "subclass_label") & (tsn["gene_set"] == "IEG_rapid")]
for j, prep in enumerate(["cells", "nuclei"]):
    for i, cl in enumerate(["Glutamatergic", "GABAergic"]):
        r = d[(d["prep"] == prep) & (d["class"] == cl)].iloc[0]
        x = j + (i - 0.5) * 0.3
        ax.plot([x, x], [r["null_q025"], r["null_q975"]], color=C["light"], lw=6, solid_capstyle="round")
        ax.plot(x, r["null_median"], "_", color=C["grey"], ms=8)
        ax.plot(x, r["observed_median_r2"], "o", color=CLASS_C[cl], ms=6, mec="white",
                label=cl if j == 0 else None)
ax.set_xticks([0, 1], ["Allen MOp\ncells", "Allen MOp\nnuclei"])
ax.set_xlim(-0.6, 1.6)
ax.set_ylabel("Median adj. R² by subclass\n(rapid IEGs vs matched null)")
ax.set_title("Alignment needs dissociation", loc="left")
ax.legend(fontsize=7)

ax = axs[2]
label(ax, "C")
s = scores.dropna(subset=["hold_time"]).copy()
s["resid"] = s["IEG_rapid"] - s.groupby("ttype")["IEG_rapid"].transform("mean")
for cl in ["Glutamatergic", "GABAergic"]:
    q = s[s["class"] == cl]
    ax.scatter(q["hold_time"] + np.random.default_rng(0).uniform(-0.8, 0.8, len(q)), q["resid"], s=5,
               color=CLASS_C[cl], alpha=0.35, lw=0, label=cl)
bins = pd.cut(s["hold_time"], [0, 10, 20, 30, 40, 50, 90])
mb = s.groupby(bins, observed=True).agg(x=("hold_time", "median"), y=("resid", "mean"),
                                        se=("resid", lambda v: v.std() / np.sqrt(len(v))))
ax.errorbar(mb["x"], mb["y"], yerr=1.96 * mb["se"], color=C["ink"], lw=1.5, marker="o", ms=4, capsize=0)
r = ht[ht["score"] == "IEG_rapid"].iloc[0]
ax.text(0.98, 0.98, f"β = {r['beta_per_10min']:+.03f} per 10 min\n95% CI [{r['ci_lo']:+.03f}, {r['ci_hi']:+.03f}]",
        transform=ax.transAxes, ha="right", va="top", fontsize=7)
ax.set_xlabel("Slice hold time (min)")
ax.set_ylabel("Rapid-IEG score\n(within-t-type residual)")
ax.set_title("Within type: weak, n.s. decline", loc="left")
save(fig, "fig2_ex_vivo_induction_controls")

# ======================================================================= Figure 3
assoc = pd.read_csv(paths.M1 / "results" / "aim1_ephys_assoc.csv")
mv = pd.read_csv(paths.M1 / "results" / "aim1_ephys_multivariate.csv")
struct = pd.read_csv(paths.M1 / "results" / "aim3_cca_structure.csv")
emb = pd.read_csv(paths.M1 / "data" / "aim3_embedding.csv")
fig = plt.figure(figsize=(7.4, 6.4), constrained_layout=True)
gs = fig.add_gridspec(2, 3)
ax = fig.add_subplot(gs[0, 0])
label(ax, "A")
feats = EPHYS_KEY
for j, cl in enumerate(["Glutamatergic", "GABAergic"]):
    d = assoc[(assoc["scope"] == cl) & (assoc["score"] == "IEG_rapid")].set_index("feature").loc[feats]
    y = np.arange(len(feats)) + (j - 0.5) * 0.35
    ax.hlines(y, d["ci_lo"], d["ci_hi"], color=CLASS_C[cl], lw=1.2)
    ax.plot(d["beta_sd_per_sd"], y, "o", color=CLASS_C[cl], ms=3.5, label=cl)
ax.axvline(0, color=C["grey"], lw=0.8)
ax.set_yticks(range(len(feats)), [f.replace(" (", "\n(") if len(f) > 22 else f for f in feats], fontsize=6.5)
ax.invert_yaxis()
ax.set_xlim(-0.35, 0.35)
ax.set_xlabel("β (SD feature / SD rapid-IEG score)\nwithin t-type, 95% CI")
ax.set_title("IEG state vs ephys", loc="left")

ax = fig.add_subplot(gs[0, 1])
label(ax, "B")
for cl in ["Glutamatergic", "GABAergic"]:
    d = assoc[(assoc["scope"] == cl) & (assoc["score"] == "IEG_rapid")]
    ax.scatter(d["partialR2_ttype"], d["partialR2_score"].clip(lower=1e-4), s=18, color=CLASS_C[cl],
               edgecolor="white", lw=0.5, label=cl)
ax.plot([1e-4, 1], [1e-4, 1], color=C["grey"], lw=0.8, ls="--")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.05, 1); ax.set_ylim(1e-4, 1)
ax.set_xlabel("Partial R², t-type")
ax.set_ylabel("Partial R², rapid-IEG score")
ax.set_title("Type ≫ state", loc="left")
ax.text(0.06, 0.5, "y = x", color=C["grey"], fontsize=6.5)
ax.legend(fontsize=6.5, loc="lower right")

ax = fig.add_subplot(gs[0, 2])
label(ax, "C")
labs = ["IEG_rapid", "IEG_delayed", "IEG_proposal", "stress"]
for j, cl in enumerate(["Glutamatergic", "GABAergic"]):
    d = mv[(mv["scope"] == cl)].set_index("score").loc[labs]
    x = np.arange(len(labs)) + (j - 0.5) * 0.35
    ax.bar(x, d["cv_R2"], width=0.3, color=CLASS_C[cl], label=cl)
    ax.plot(x, d["null_q95"], "_", color=C["ink"], ms=9)
ax.axhline(0, color=C["grey"], lw=0.8)
ax.set_xticks(range(len(labs)), ["Rapid", "Delayed", "Proposal", "Stress"], fontsize=6.5)
ax.set_ylabel("Held-out R² (ephys → score)")
ax.set_title("Ephys ↛ IEG state", loc="left")
ax.legend(fontsize=6.5, loc="upper left")
acc = mv[mv["score"] == "ttype_decoding_acc"].set_index("scope")
ax.text(0.02, 0.30, "same features decode t-type:\n" + "\n".join(
    f"{k[:4]}. {acc.loc[k,'cv_R2']:.2f} (chance {acc.loc[k,'null_q95']:.2f})" for k in acc.index),
    transform=ax.transAxes, ha="left", va="bottom", fontsize=6)
ax.text(0.02, 0.62, "tick: 95th pct. of\nwithin-type\npermutation null", transform=ax.transAxes, fontsize=6,
        color=C["grey"])

FAM_C = {"IT": C["blue"], "ET": C["orange"], "CT": C["aqua"], "NP": C["yellow"],
         "Pvalb": C["blue"], "Sst": C["orange"], "Vip": C["aqua"], "Lamp5": C["yellow"], "Sncg": C["magenta"]}
for k, cl in enumerate(["Glutamatergic", "GABAergic"]):
    sub = gs[1, k].subgridspec(2, 1) if False else None
    ax = fig.add_subplot(gs[1, k])
    label(ax, "DE"[k])
    e = emb[emb["class"] == cl]
    for fam, q in e.groupby("family"):
        ax.scatter(q["umap1"], q["umap2"], s=4, color=FAM_C.get(fam, C["grey"]), lw=0, label=f"{fam} ({len(q)})")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{cl[:4]}. joint CCA space", loc="left")
    ax.legend(fontsize=6, markerscale=2.5, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3,
              handletextpad=0.1, columnspacing=0.6)
    # inset colored by IEG score
    ins = ax.inset_axes([0.62, -0.02, 0.4, 0.4])
    v = e["IEG_rapid"]
    ins.scatter(e["umap1"], e["umap2"], s=1.5, c=v, cmap="Blues", vmin=v.quantile(.05), vmax=v.quantile(.95), lw=0)
    ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_visible(True); sp.set_color(C["light"])
    ins.set_title("rapid-IEG score", fontsize=6, pad=1)

ax = fig.add_subplot(gs[1, 2])
label(ax, "F")
for j, cl in enumerate(["Glutamatergic", "GABAergic"]):
    d = struct[struct["class"] == cl]
    ax.plot(d["cc"] + (j - 0.5) * 0.15, d["adjR2_ttype"], "o-", color=CLASS_C[cl], ms=4, lw=1.2,
            label=f"{cl[:4]}. — t-type")
    ax.plot(d["cc"] + (j - 0.5) * 0.15, d["r2_IEG_rapid"], "s--", color=CLASS_C[cl], ms=4, lw=1.2,
            mfc="white", label=f"{cl[:4]}. — rapid IEG")
ax.set_xticks(range(1, 6))
ax.set_xlabel("Canonical dimension")
ax.set_ylabel("Variance explained (R²)")
ax.set_ylim(-0.02, 1)
ax.set_title("Joint axes ≈ t-type", loc="left")
ax.legend(fontsize=6, loc="upper right")
save(fig, "fig3_ephys_and_joint_embedding")
print("figures written")
