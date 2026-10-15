"""Pool all glutamatergic cells for power, but estimate WITHIN type.

IT/ET/CT span L2/3 to L6 and differ in both rheobase and IEG expression, so
pooling without type in the model manufactures correlations out of type
identity (Fig4 in 02_figures_ephys_confound.py). Adding RNA type as a covariate
gives the common within-type slope while using every cell -- a fixed-effects /
within estimator.

Power: n=126 (L5 only) resolves |rho| > 0.247; n~414 (all glutamatergic)
resolves |rho| > 0.137 at 80% power.

Reports, for each ephys feature:
  * naive pooled estimate (no type in model)  -- shown only to expose the bias
  * within-type estimate (type dummies)        -- the actual estimate
  * L5-excitatory subgroup

Usage:  uv run python analysis/m1_patchseq/scripts/05_glutamatergic_within_type.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)

from importlib import import_module

import numpy as np
import pandas as pd
from scipy import stats

from plasticity_common import (
    PRIMARY_EPHYS, l5_excitatory_ids, load_ephys, load_meta,
)

_m3 = import_module("03_gene_level_rheobase")
_m4 = import_module("04_ephys_wide_screen")
CHANNEL_PATTERNS, COVARS, MIN_DETECT_FRAC = (
    _m3.CHANNEL_PATTERNS, _m3.COVARS, _m3.MIN_DETECT_FRAC
)
load_full_logcpm, screen = _m3.load_full_logcpm, _m3.screen
EXTRA_EPHYS = _m4.EXTRA_EPHYS

RESULTS = paths.results(__file__)
GLUT_FAMILIES = ["IT", "ET", "CT", "NP"]
FOCUS = ["Jun", "Junb", "Gadd45g", "Egr1", "Fos", "Npas4", "Arc",
         "Nptx1", "Nptx2", "Homer1", "Bdnf"]


def min_detectable_rho(n: int, alpha=0.05, power=0.80) -> float:
    if n < 5:
        return np.nan
    return float(np.tanh((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power))
                         / np.sqrt(n - 3)))


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    glut = meta.index[meta["RNA family"].isin(GLUT_FAMILIES)]
    l5 = set(l5_excitatory_ids(meta))
    ephys = load_ephys()

    print(f"[pop] glutamatergic cells: {len(glut)}")
    print(meta.loc[glut, "RNA family"].value_counts().to_string())
    print(f"\n[power] n={len(glut)} resolves |rho| > "
          f"{min_detectable_rho(len(glut)):.3f}   "
          f"(vs {min_detectable_rho(126):.3f} at n=126)")

    log_cpm, qc = load_full_logcpm(glut)
    g = log_cpm.columns
    drop = (g.str.startswith("ERCC-")
            | g.str.match(r"^(Gm\d|.*Rik$|LOC\d|A[A-Z]\d{6}|BC\d{6})")
            | ((log_cpm > 0).mean(axis=0) < MIN_DETECT_FRAC).to_numpy())
    expr = log_cpm.loc[:, ~drop]
    is_channel = pd.Series(
        expr.columns.str.match("|".join(CHANNEL_PATTERNS)), index=expr.columns)
    print(f"[genes] {expr.shape[1]} tested "
          f"({int(is_channel.sum())} ion channel / subunit)")

    design = meta.loc[expr.index].join(qc, how="left")
    tech = design[COVARS].copy()

    # Type dummies. Drop rare types so the design stays well conditioned.
    rt = design["RNA type"]
    keep_types = rt.value_counts()[lambda s: s >= 8].index
    rt = rt.where(rt.isin(keep_types), "other")
    dummies = pd.get_dummies(rt, prefix="t", drop_first=True).astype(float)
    within = pd.concat([tech, dummies], axis=1)
    print(f"[model] within-type design: {len(COVARS)} technical + "
          f"{dummies.shape[1]} type dummies ({rt.nunique()} types)\n")

    is_l5 = pd.Series([i in l5 for i in expr.index], index=expr.index)
    features = [f for f in PRIMARY_EPHYS + EXTRA_EPHYS if f in ephys.columns]

    rows, focus_rows, hit_frames = [], [], []
    for feat in features:
        y = ephys[feat].reindex(expr.index)
        if y.notna().sum() < 60:
            continue

        naive = screen(expr, y, covars=tech)
        wth = screen(expr, y, covars=within)

        hits = wth[wth["q"] < 0.05].copy()
        hits["is_channel"] = hits["gene"].map(is_channel).fillna(False)
        n_chan = int(hits["is_channel"].sum())

        top_n = naive.reindex(naive["rho"].abs().sort_values(ascending=False).index).iloc[0]
        top_w = wth.reindex(wth["rho"].abs().sort_values(ascending=False).index).iloc[0]

        rows.append({
            "feature": feat,
            "n": int(wth["n"].iloc[0]),
            "naive_hits_q05": int((naive["q"] < 0.05).sum()),
            "within_hits_q05": len(hits),
            "within_hits_channel": n_chan,
            "within_hits_nonchannel": len(hits) - n_chan,
            "naive_best_gene": top_n["gene"], "naive_best_rho": top_n["rho"],
            "within_best_gene": top_w["gene"], "within_best_rho": top_w["rho"],
            "within_min_q": float(wth["q"].min()),
        })
        if len(hits):
            h = hits.copy()
            h["feature"] = feat
            hit_frames.append(h)

        # focus genes, all three estimates
        for gene in FOCUS:
            if gene not in expr.columns:
                continue
            rn = naive.set_index("gene").loc[gene]
            rw = wth.set_index("gene").loc[gene]
            sub = pd.concat([expr.loc[is_l5, gene], y[is_l5]], axis=1).dropna()
            r_l5 = stats.spearmanr(*sub.to_numpy().T)[0] if len(sub) > 8 else np.nan
            focus_rows.append({
                "feature": feat, "gene": gene,
                "rho_naive": rn["rho"], "q_naive": rn["q"],
                "rho_within": rw["rho"], "p_within": rw["p"], "q_within": rw["q"],
                "rho_L5only": r_l5, "n": int(rw["n"]),
            })

    S = pd.DataFrame(rows).sort_values("within_hits_q05", ascending=False)
    S.to_csv(RESULTS / "I_glut_within_type_summary.csv", index=False)
    print("=" * 104)
    print("Glutamatergic pooled, WITHIN-type estimate vs naive pooling")
    print("=" * 104)
    print(S[["feature", "n", "naive_hits_q05", "within_hits_q05",
             "within_hits_channel", "within_hits_nonchannel",
             "within_best_gene", "within_best_rho", "within_min_q"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\n[write] {RESULTS/'I_glut_within_type_summary.csv'}")

    print(f"\nTotal within-type hits (q<0.05): {int(S['within_hits_q05'].sum())}"
          f"  | channel {int(S['within_hits_channel'].sum())}"
          f"  | non-channel {int(S['within_hits_nonchannel'].sum())}")
    print(f"Naive pooling would have reported: "
          f"{int(S['naive_hits_q05'].sum())} hits "
          f"({int(S['naive_hits_q05'].sum()) - int(S['within_hits_q05'].sum()):+d} "
          f"vs within-type)")
    print(f"Features with any within-type hit: "
          f"{int((S['within_hits_q05']>0).sum())} of {len(S)}")

    if hit_frames:
        H = pd.concat(hit_frames, ignore_index=True)
        H.to_csv(RESULTS / "I_glut_within_type_hits.csv", index=False)
        print(f"[write] {RESULTS/'I_glut_within_type_hits.csv'}")
        print("\n--- channel hits, within-type (the positive control) ---")
        ch = H[H["is_channel"]].reindex(
            H[H["is_channel"]]["rho"].abs().sort_values(ascending=False).index)
        print(ch.head(20)[["feature", "gene", "rho", "q"]].to_string(index=False,
              float_format=lambda v: f"{v:.3g}") if len(ch) else "  none")

    F = pd.DataFrame(focus_rows)
    F.to_csv(RESULTS / "I_glut_focus_genes.csv", index=False)
    print(f"\n[write] {RESULTS/'I_glut_focus_genes.csv'}")
    print("\n--- focus genes vs Rheobase: naive vs within-type vs L5-only ---")
    rb = F[F["feature"] == "Rheobase (pA)"].set_index("gene").reindex(FOCUS)
    print(rb[["rho_naive", "q_naive", "rho_within", "p_within", "q_within",
              "rho_L5only", "n"]]
          .to_string(float_format=lambda v: f"{v:.4g}"))

    print("\n--- focus genes: best feature by within-type q ---")
    best = (F.sort_values("q_within").groupby("gene", as_index=False).first()
            .sort_values("q_within"))
    print(best[["gene", "feature", "rho_within", "p_within", "q_within"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
