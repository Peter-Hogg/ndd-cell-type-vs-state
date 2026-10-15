"""Acute vs cultured preparation diagnostic for macaque L5 ET cells (Dembrow et al. 2024).

07_dembrow_confirmatory.py pools cultured and acute L5 ET cells with preparation
as a covariate. Three parts:

  1. IEG (and control-gene) expression by preparation, Mann-Whitney U and AUC --
     involves no outcome variable.
  2. Endpoint (rheobase / threshold current) by preparation.
  3. JUN vs rheobase partial Spearman within each preparation, reported for both
     arms with their detection floors (|rho| at 80% power), plus a Fisher-z test
     of the arm difference.

Usage:  uv run python analysis/macaque_patchseq/scripts/08_dembrow_prep_diagnostic.py
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

_m7 = import_module("07_dembrow_confirmatory")
load, covariates, partial_spearman, fisher_ci, ENDPOINT = (
    _m7.load, _m7.covariates, _m7.partial_spearman, _m7.fisher_ci, _m7.ENDPOINT
)

RESULTS = paths.results(__file__)
IEGS = ["JUN", "JUNB", "JUND", "FOS", "FOSB", "EGR1", "NPAS4", "ARC",
        "GADD45G", "NPTX1", "NPTX2", "HOMER1", "BDNF"]
CONTROLS = ["KCNIP1", "ACTB", "GAPDH", "SLC17A7", "FEZF2"]


def floor_rho(n):
    return float(np.tanh(2.8016 / np.sqrt(n - 3))) if n > 4 else np.nan


def main() -> None:
    print("*" * 78)
    print("PREPARATION DIAGNOSTIC — acute vs cultured L5 ET")
    print("*" * 78)

    meta, expr, _ = load()
    et = meta.index[meta["is_ET"]]
    m, e = meta.loc[et], expr.loc[et]
    acute = m.index[m["slice_type"] == "NHP_acute"]
    cult = m.index[m["slice_type"] == "NHP_culture"]

    print(f"\nL5 ET: {len(m)} cells -> acute {len(acute)}, cultured {len(cult)}")
    print(f"  DIV in cultured arm: "
          f"{m.loc[cult,'DIV'].describe()[['count','min','50%','max']].to_dict()}")
    print(f"  detection floor: acute |rho| >= {floor_rho(len(acute)):.3f}   "
          f"cultured |rho| >= {floor_rho(len(cult)):.3f}")

    # ---- 1. does culture perturb IEG expression? (no outcome variable) ----
    print("\n" + "=" * 78)
    print("1. IEG EXPRESSION by preparation (no outcome variable involved)")
    print("=" * 78)
    print(f"  {'gene':9s} {'acute med':>10s} {'cult med':>10s} {'delta':>8s} "
          f"{'MWU p':>9s} {'AUC':>6s}")
    rows = []
    for gene in IEGS + CONTROLS:
        if gene not in e.columns:
            continue
        a, c = e.loc[acute, gene], e.loc[cult, gene]
        u, p = stats.mannwhitneyu(a, c)
        auc = u / (len(a) * len(c))
        tag = "IEG" if gene in IEGS else "control"
        rows.append({"gene": gene, "class": tag, "acute_median": a.median(),
                     "cult_median": c.median(), "delta": c.median() - a.median(),
                     "mwu_p": p, "auc": auc})
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {gene:9s} {a.median():10.3f} {c.median():10.3f} "
              f"{c.median()-a.median():+8.3f} {p:9.3g} {auc:6.3f} {star}")
    R1 = pd.DataFrame(rows)
    n_ieg_sig = int(((R1["class"] == "IEG") & (R1["mwu_p"] < 0.05)).sum())
    n_ctl_sig = int(((R1["class"] == "control") & (R1["mwu_p"] < 0.05)).sum())
    print(f"\n  IEGs differing at p<0.05: {n_ieg_sig}/{int((R1['class']=='IEG').sum())}"
          f"   controls: {n_ctl_sig}/{int((R1['class']=='control').sum())}")

    # ---- 2. endpoint by preparation ----
    print("\n" + "=" * 78)
    print("2. ENDPOINT (rheobase) by preparation")
    print("=" * 78)
    ya, yc = m.loc[acute, ENDPOINT], m.loc[cult, ENDPOINT]
    u, p = stats.mannwhitneyu(ya.dropna(), yc.dropna())
    print(f"  acute    median {ya.median():7.1f} pA   IQR "
          f"[{ya.quantile(.25):.0f}, {ya.quantile(.75):.0f}]  n={ya.notna().sum()}")
    print(f"  cultured median {yc.median():7.1f} pA   IQR "
          f"[{yc.quantile(.25):.0f}, {yc.quantile(.75):.0f}]  n={yc.notna().sum()}")
    print(f"  Mann-Whitney p = {p:.4g}   AUC = {u/(len(ya)*len(yc)):.3f}")

    # ---- 3. symmetric per-arm JUN association ----
    print("\n" + "=" * 78)
    print("3. JUN vs rheobase, BOTH arms")
    print("=" * 78)
    C = covariates(meta)
    keep = [c for c in C.columns if c not in ("prep",)]  # constant within arm
    rows3 = []
    for label, idx in (("acute", acute), ("cultured", cult)):
        r, p, n, ncov = partial_spearman(e.loc[idx, "JUN"],
                                        m.loc[idx, ENDPOINT], C.loc[idx, keep])
        lo, hi = fisher_ci(r, n)
        rows3.append({"arm": label, "rho": r, "p": p, "n": n,
                      "ci_lo": lo, "ci_hi": hi, "floor": floor_rho(n)})
        print(f"  {label:9s} rho = {r:+.3f}  p = {p:.4g}  n = {n}  "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]   (floor {floor_rho(n):.3f})")
    R3 = pd.DataFrame(rows3)
    if R3["rho"].notna().all() and (R3["n"] > 4).all():
        dz = np.arctanh(R3["rho"].iloc[0]) - np.arctanh(R3["rho"].iloc[1])
        se = np.sqrt(1 / (R3["n"].iloc[0] - 3) + 1 / (R3["n"].iloc[1] - 3))
        pc = float(2 * stats.norm.sf(abs(dz / se)))
        print(f"\n  arms differ? Fisher-z diff = {dz:+.3f}, SE = {se:.3f}, p = {pc:.4g}")

    print("\n" + "-" * 78)
    print("NOTES")
    print("-" * 78)
    print("  Per-arm detection floors are ~0.45-0.50 (|rho| at 80% power).")
    print("  Part 1 tests whether culture perturbs the exposure variable (IEG")
    print("  expression) independently of the outcome.")

    R1.to_csv(RESULTS / "L_prep_ieg_expression.csv", index=False)
    R3.to_csv(RESULTS / "L_prep_jun_per_arm.csv", index=False)
    print(f"\n[write] {RESULTS/'L_prep_ieg_expression.csv'}")
    print(f"[write] {RESULTS/'L_prep_jun_per_arm.csv'}")


if __name__ == "__main__":
    main()
