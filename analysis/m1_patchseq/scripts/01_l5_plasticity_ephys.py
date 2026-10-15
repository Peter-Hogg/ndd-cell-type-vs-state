"""Does activity state predict intrinsic ephys *within* type?

Writes every number it reports to results/, and never reuses a variable across
analyses with a different cell population.

Four analyses:
  A  within-L5-excitatory score vs ephys (primary test)
  B  within transcriptomic type (strictest level, smallest n)
  C  across the full mixed population (the confound demonstration)
  D  technical confounds: depth, ERCC spike-ins, ex vivo hold time

Usage:  uv run python analysis/m1_patchseq/scripts/01_l5_plasticity_ephys.py [--exon-only]
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)

import argparse

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from plasticity_common import (
    CORE_IEG,
    EXPANDED_IEG,
    PRIMARY_EPHYS,
    l5_excitatory_ids,
    load_ephys,
    load_expression,
    load_meta,
    plasticity_score,
)

RESULTS = paths.results(__file__)
ALPHA = 0.05
POWER = 0.80

# ---------------------------------------------------------------- statistics


def min_detectable_rho(n: int, alpha: float = ALPHA, power: float = POWER) -> float:
    """Smallest |rho| detectable at the given n, via the Fisher-z approximation.

    Reported alongside every null result: "no correlation" is only meaningful
    next to the effect size the sample could actually have resolved.
    """
    if n < 5:
        return np.nan
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return float(np.tanh((z_a + z_b) / np.sqrt(n - 3)))


def spearman_table(
    df: pd.DataFrame, score_col: str, features: list[str], label: str
) -> pd.DataFrame:
    """Spearman score-vs-feature across ``features``, with BH-FDR over the panel."""
    rows = []
    for feat in features:
        sub = df[[score_col, feat]].dropna()
        if len(sub) < 5:
            rows.append({"feature": feat, "rho": np.nan, "p": np.nan, "n": len(sub)})
            continue
        rho, p = stats.spearmanr(sub[score_col], sub[feat])
        rows.append({"feature": feat, "rho": rho, "p": p, "n": len(sub)})

    out = pd.DataFrame(rows)
    ok = out["p"].notna()
    out["p_fdr"] = np.nan
    if ok.any():
        out.loc[ok, "p_fdr"] = multipletests(out.loc[ok, "p"], method="fdr_bh")[1]
    out["min_detectable_rho"] = out["n"].map(min_detectable_rho)
    out["population"] = label
    return out.sort_values("rho", ascending=False).reset_index(drop=True)


def _rank(s: pd.Series) -> np.ndarray:
    return stats.rankdata(s.to_numpy())


def partial_spearman(
    df: pd.DataFrame, x: str, y: str, covars: list[str]
) -> tuple[float, float, int]:
    """Spearman correlation of x and y after linearly removing ``covars``.

    Rank-transform everything, then correlate OLS residuals. Equivalent to
    pingouin's ``partial_corr(method="spearman")`` but with no extra dependency.
    """
    sub = df[[x, y] + covars].dropna()
    n = len(sub)
    if n < len(covars) + 5:
        return np.nan, np.nan, n

    C = np.column_stack([np.ones(n)] + [_rank(sub[c]) for c in covars])

    def resid(col: str) -> np.ndarray:
        v = _rank(sub[col])
        beta, *_ = np.linalg.lstsq(C, v, rcond=None)
        return v - C @ beta

    rx, ry = resid(x), resid(y)
    # Pearson on rank residuals; dof reduced by the covariates removed.
    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = n - 2 - len(covars)
    if dof <= 0 or not np.isfinite(r) or abs(r) >= 1:
        return r, np.nan, n
    t = r * np.sqrt(dof / (1 - r**2))
    return r, float(2 * stats.t.sf(abs(t), dof)), n


def eta_squared(df: pd.DataFrame, value: str, group: str) -> tuple[float, float, int]:
    """One-way eta^2 (variance in ``value`` explained by ``group``) + Kruskal p."""
    sub = df[[value, group]].dropna()
    groups = [g[value].to_numpy() for _, g in sub.groupby(group, observed=True) if len(g) > 1]
    if len(groups) < 2:
        return np.nan, np.nan, len(sub)
    grand = sub[value].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((sub[value] - grand) ** 2).sum()
    _, p = stats.kruskal(*groups)
    return float(ss_between / ss_total), float(p), len(sub)


# ---------------------------------------------------------------- main


def main(use_introns: bool) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = "" if use_introns else "_exononly"

    print("=" * 74)
    print("Patch-seq plasticity state vs intrinsic electrophysiology")
    print(f"counts: {'exon + intron' if use_introns else 'exon only'}")
    print("=" * 74)

    meta = load_meta()
    l5 = l5_excitatory_ids(meta)
    print(f"[pop] L5 excitatory transcriptomic types: {len(l5)} cells")
    print(meta.loc[l5, "RNA type"].value_counts().to_string())

    genes = sorted(set(CORE_IEG) | set(EXPANDED_IEG))
    log_cpm, qc = load_expression(genes, use_introns=use_introns)

    missing = [g for g in CORE_IEG if g not in log_cpm.columns]
    if missing:
        print(f"[score] core IEG genes absent from the index: {missing}")

    core_present = [g for g in CORE_IEG if g in log_cpm.columns]
    exp_present = [g for g in EXPANDED_IEG if g in log_cpm.columns]
    print(f"[score] core panel {len(core_present)}/{len(CORE_IEG)} genes, "
          f"expanded panel {len(exp_present)}/{len(EXPANDED_IEG)} genes")

    ephys = load_ephys()

    # Two scoring references. The L5-standardised score is the primary one; the
    # all-cell version exists only to reproduce and explain analysis C.
    score_l5ref = plasticity_score(log_cpm[core_present], population=l5)
    score_allref = plasticity_score(log_cpm[core_present], population=None)
    score_exp_l5ref = plasticity_score(log_cpm[exp_present], population=l5)

    base = meta.join(qc, how="left")
    base = base.join(ephys, how="inner")
    base["plasticity_score"] = score_l5ref.reindex(base.index)
    base["plasticity_score_allref"] = score_allref.reindex(base.index)
    base["plasticity_score_expanded"] = score_exp_l5ref.reindex(base.index)
    base.index.name = "cell id"
    print(f"[merge] QC-passing cells with ephys + expression: {len(base)}")

    l5_df = base.loc[base.index.intersection(l5)].copy()
    print(f"[merge] L5 excitatory analysis set: {len(l5_df)}")

    tables: dict[str, pd.DataFrame] = {}

    # ---- A. within L5 excitatory (the hypothesis test) -------------------
    print("\n" + "-" * 74)
    print("A. Within L5 excitatory (primary test)")
    print("-" * 74)
    a_core = spearman_table(l5_df, "plasticity_score", PRIMARY_EPHYS, "L5 exc (core IEG)")
    a_exp = spearman_table(
        l5_df, "plasticity_score_expanded", PRIMARY_EPHYS, "L5 exc (expanded IEG)"
    )
    print(a_core.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  minimum |rho| detectable at n={len(l5_df)}, "
          f"{int(POWER*100)}% power: {min_detectable_rho(len(l5_df)):.3f}")
    print("\n  same test, expanded IEG panel:")
    print(a_exp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    tables["A_within_L5"] = pd.concat([a_core, a_exp], ignore_index=True)

    # ---- B. within transcriptomic type ----------------------------------
    print("\n" + "-" * 74)
    print("B. Within individual transcriptomic type")
    print("-" * 74)
    per_type = []
    for rna_type, grp in l5_df.groupby("RNA type", observed=True):
        if len(grp) < 10:
            continue
        t = spearman_table(grp, "plasticity_score", PRIMARY_EPHYS, rna_type)
        t["RNA type"] = rna_type
        per_type.append(t)
    if per_type:
        b = pd.concat(per_type, ignore_index=True)
        print(b[["RNA type", "feature", "rho", "p", "p_fdr", "n", "min_detectable_rho"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        tables["B_within_type"] = b
        n_sig = int((b["p_fdr"] < ALPHA).sum())
        print(f"\n  {n_sig} of {len(b)} within-type tests significant after FDR")
    else:
        print("  no transcriptomic type reaches n >= 10")

    # ---- C. the confound -------------------------------------------------
    print("\n" + "-" * 74)
    print("C. Across the full mixed population (confound demonstration)")
    print("-" * 74)
    c_all = spearman_table(
        base, "plasticity_score_allref", PRIMARY_EPHYS, "all QC cells (mixed classes)"
    )
    print(c_all.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    tables["C_mixed_population"] = c_all

    fam = base[base["RNA family"].isin(["IT", "ET", "CT", "Pvalb", "Sst", "Vip", "Lamp5"])]
    print("\n  variance in plasticity score explained by broad RNA family:")
    for col in ["plasticity_score_allref", "plasticity_score"]:
        e2, p, n = eta_squared(fam, col, "RNA family")
        print(f"    {col:26s} eta^2 = {e2:.3f}  (Kruskal p = {p:.3g}, n = {n})")

    print("\n  same correlations computed within each RNA family:")
    within_fam = []
    for family, grp in fam.groupby("RNA family", observed=True):
        if len(grp) < 20:
            continue
        t = spearman_table(grp, "plasticity_score_allref", PRIMARY_EPHYS, family)
        t["RNA family"] = family
        within_fam.append(t)
    if within_fam:
        wf = pd.concat(within_fam, ignore_index=True)
        pivot = wf.pivot_table(index="feature", columns="RNA family", values="rho")
        pooled = c_all.set_index("feature")["rho"]
        pivot.insert(0, "MIXED (pooled)", pooled)
        print(pivot.to_string(float_format=lambda v: f"{v:+.3f}"))
        tables["C_within_family"] = wf
        print("\n  Pooled column vs the within-family columns: if the pooled effect")
        print("  is larger than every within-family effect, it is between-class")
        print("  structure, not a state-to-physiology relationship.")

    # ---- D. technical confounds -----------------------------------------
    print("\n" + "-" * 74)
    print("D. Technical and ex vivo confounds (within L5)")
    print("-" * 74)
    tech = ["log_libsize", "n_detected", "ercc_frac", "Hold Time (min)"]
    tech_rows = []
    for t_col in tech:
        sub = l5_df[["plasticity_score", t_col]].dropna()
        if len(sub) < 5:
            continue
        rho, p = stats.spearmanr(sub["plasticity_score"], sub[t_col])
        tech_rows.append({"covariate": t_col, "rho_vs_score": rho, "p": p, "n": len(sub)})
    tech_df = pd.DataFrame(tech_rows)
    print(tech_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    tables["D_technical"] = tech_df

    print("\n  score vs ephys, partialling out depth, spike-ins and hold time:")
    covars = [c for c in tech if l5_df[c].notna().sum() > len(l5_df) * 0.8]
    part_rows = []
    for feat in PRIMARY_EPHYS:
        r_raw, p_raw = stats.spearmanr(
            *l5_df[["plasticity_score", feat]].dropna().to_numpy().T
        )
        r_adj, p_adj, n_adj = partial_spearman(l5_df, "plasticity_score", feat, covars)
        part_rows.append(
            {"feature": feat, "rho_raw": r_raw, "p_raw": p_raw,
             "rho_partial": r_adj, "p_partial": p_adj, "n": n_adj}
        )
    part = pd.DataFrame(part_rows)
    ok = part["p_partial"].notna()
    part["p_partial_fdr"] = np.nan
    if ok.any():
        part.loc[ok, "p_partial_fdr"] = multipletests(
            part.loc[ok, "p_partial"], method="fdr_bh"
        )[1]
    print(part.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  covariates removed: {covars}")
    tables["D_partial"] = part

    # ---- persist ---------------------------------------------------------
    for name, tbl in tables.items():
        path = RESULTS / f"{name}{suffix}.csv"
        tbl.to_csv(path, index=False)
        print(f"[write] {path}")

    keep = [
        "RNA family", "RNA type", "RNA type confidence", "Inferred layer",
        "Hold Time (min)", "libsize", "n_detected", "ercc_frac",
        "plasticity_score", "plasticity_score_expanded", "plasticity_score_allref",
    ] + PRIMARY_EPHYS
    cells_path = RESULTS / f"cells_l5{suffix}.csv"
    l5_df[[c for c in keep if c in l5_df.columns]].to_csv(cells_path)
    print(f"[write] {cells_path}")

    all_path = RESULTS / f"cells_all{suffix}.csv"
    base[[c for c in keep if c in base.columns]].to_csv(all_path)
    print(f"[write] {all_path}")

    # ---- headline --------------------------------------------------------
    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    best = a_core.loc[a_core["p_fdr"].idxmin()] if a_core["p_fdr"].notna().any() else None
    if best is not None:
        print(f"Within L5 excitatory (n={len(l5_df)}), strongest association:")
        print(f"  {best['feature']}: rho = {best['rho']:+.3f}, "
              f"p = {best['p']:.3g}, FDR = {best['p_fdr']:.3g}")
        print(f"  survives FDR at alpha={ALPHA}: {bool(best['p_fdr'] < ALPHA)}")
        print(f"  smallest |rho| this n could resolve: "
              f"{min_detectable_rho(len(l5_df)):.3f}")
    strongest_mixed = c_all.reindex(c_all["rho"].abs().sort_values(ascending=False).index).iloc[0]
    print(f"\nAcross the mixed population (n={int(strongest_mixed['n'])}), strongest:")
    print(f"  {strongest_mixed['feature']}: rho = {strongest_mixed['rho']:+.3f}, "
          f"p = {strongest_mixed['p']:.3g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exon-only",
        action="store_true",
        help="use exon counts alone (sensitivity check; default sums exon+intron)",
    )
    args = ap.parse_args()
    main(use_introns=not args.exon_only)
