"""Bidirectional plasticity: test magnitude, not direction.

Hypothesis: LTP is bidirectional. If some cells are potentiated and others
depressed, a *signed* correlation between an activity score and rheobase cancels
to zero even when plasticity is widespread. Absent a time-locked learning event
that biases the whole population one way, the expected signature is a **flat but
widened spread**, not a shift.

That has two consequences:

1. Excess within-cluster variance of IEG expression in the atlas
   (`14_within_cluster_spread.py`, versus an expression-matched null) is the
   predicted signature.
2. The Patch-seq test is on **magnitudes**: do cells far from their own
   type's mean activity state also sit far from their type's mean rheobase, in
   EITHER direction? A signed Spearman cannot see this; three tests here can.

  T1  Spearman(|state residual|, |rheobase residual|)
  T2  heteroscedasticity -- Breusch-Pagan of squared rheobase residual on state
  T3  rheobase variance in the top vs bottom tercile of |state deviation|
      (Levene + variance ratio)

Usage:  uv run python analysis/m1_patchseq/scripts/17_bidirectional_test.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)


import numpy as np
import pandas as pd
from scipy import stats

RESULTS = paths.results(__file__)
ENDPOINT = "Rheobase (pA)"
SCORES = ["plasticity_score", "plasticity_score_expanded"]


def within_resid(df: pd.DataFrame, col: str, grp: str) -> pd.Series:
    """Deviation from the cell's own transcriptomic-type mean."""
    return df[col] - df.groupby(grp, observed=True)[col].transform("mean")


def main() -> None:
    d = pd.read_csv(RESULTS / "cells_l5.csv", index_col=0)
    d = d[d[ENDPOINT].notna()].copy()
    grp = "RNA type"
    print(f"mouse M1 L5 excitatory: n = {len(d)}, "
          f"{d[grp].nunique()} transcriptomic types")

    d["rheo_resid"] = within_resid(d, ENDPOINT, grp)
    print(f"rheobase residual sd = {d['rheo_resid'].std(ddof=1):.1f} pA\n")

    for score in SCORES:
        if score not in d.columns:
            continue
        s = d[[score, "rheo_resid", grp]].dropna().copy()
        s["state_resid"] = within_resid(s, score, grp)
        s["abs_state"] = s["state_resid"].abs()
        s["abs_rheo"] = s["rheo_resid"].abs()
        n = len(s)

        print("=" * 74)
        print(f"{score}   (n = {n})")
        print("=" * 74)

        # --- reference: the signed test that cancels under bidirectionality --
        r_signed, p_signed = stats.spearmanr(s["state_resid"], s["rheo_resid"])
        print(f"  [ref] signed   rho(state, rheobase)        = {r_signed:+.3f}"
              f"  p = {p_signed:.3g}")

        # --- T1: magnitude coupling -----------------------------------------
        r_abs, p_abs = stats.spearmanr(s["abs_state"], s["abs_rheo"])
        print(f"  T1    magnitude rho(|state|, |rheobase|)    = {r_abs:+.3f}"
              f"  p = {p_abs:.3g}")

        # --- T2: heteroscedasticity -----------------------------------------
        # Breusch-Pagan: regress squared rheobase residual on the state score.
        try:
            import statsmodels.api as sm
            X = sm.add_constant(s[["state_resid"]].to_numpy())
            y = (s["rheo_resid"] ** 2).to_numpy()
            fit = sm.OLS(y, X).fit()
            lm = fit.rsquared * n
            p_bp = float(stats.chi2.sf(lm, 1))
            print(f"  T2    Breusch-Pagan  LM = {lm:.2f}  p = {p_bp:.3g}"
                  f"   (slope {fit.params[1]:+.1f})")
        except Exception as e:
            print(f"  T2    unavailable: {e}")

        # --- T3: variance by |state| tercile --------------------------------
        q1, q2 = s["abs_state"].quantile([1 / 3, 2 / 3])
        lo = s.loc[s["abs_state"] <= q1, "rheo_resid"]
        hi = s.loc[s["abs_state"] >= q2, "rheo_resid"]
        if len(lo) > 5 and len(hi) > 5:
            W, p_lev = stats.levene(lo, hi, center="median")
            vr = hi.var(ddof=1) / lo.var(ddof=1)
            # exact F test on the variance ratio
            p_f = 2 * min(stats.f.sf(vr, len(hi) - 1, len(lo) - 1),
                          stats.f.cdf(vr, len(hi) - 1, len(lo) - 1))
            print(f"  T3    rheobase SD: low|state| {lo.std(ddof=1):6.1f} pA "
                  f"(n={len(lo)})   high|state| {hi.std(ddof=1):6.1f} pA "
                  f"(n={len(hi)})")
            print(f"        variance ratio = {vr:.2f}   Levene p = {p_lev:.3g}"
                  f"   F-test p = {p_f:.3g}")
            print(f"        -> bidirectional plasticity predicts ratio > 1")
        print()

    # ---- the functional-sign point: Arc vs Bdnf are opposite effectors -----
    print("=" * 74)
    print("Do opposite-sign effectors cancel inside the composite?")
    print("=" * 74)
    print("  Arc drives AMPAR endocytosis (depressive/homeostatic);")
    print("  Bdnf is potentiating. Averaging them mixes functional signs.")
    gg = pd.read_csv(RESULTS / "E_gene_rheobase_adjusted.csv").set_index("gene")
    for g in ["Arc", "Bdnf", "Nptx2", "Homer1", "Plk2", "Jun"]:
        if g in gg.index:
            r = gg.loc[g]
            print(f"    {g:8s} rho = {r['rho']:+.3f}  p = {r['p']:.3g}")
    print("\n  If depressive and potentiating effectors carried opposite-signed")
    print("  rheobase correlations, that would support cancellation. Signs above")
    print("  being uniformly weak-negative argues instead for a common weak")
    print("  effect or none.")


if __name__ == "__main__":
    main()
