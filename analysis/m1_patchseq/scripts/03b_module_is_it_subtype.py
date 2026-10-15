"""Is the rheobase-correlated gene module a plasticity signal, or IT-vs-ET
subtype identity in disguise?

Pooling across broad RNA families can manufacture a score-to-physiology
correlation out of class identity (01, analysis C). L5 IT and L5 ET cells
differ substantially in rheobase, so the same confound can operate *inside* the
L5 excitatory population. The module hits include IT/ET markers (Neurod6,
Unc5d, Cdh10, Sorcs3).

Runs under both exon-only and exon+intron quantification.

Usage:  uv run python analysis/m1_patchseq/scripts/03b_module_is_it_subtype.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)


import numpy as np
import pandas as pd
from scipy import stats

from plasticity_common import l5_excitatory_ids, load_ephys, load_meta
from importlib import import_module

_m3 = import_module("03_gene_level_rheobase")
COVARS, TARGET, filter_genes, load_full_logcpm, screen = (
    _m3.COVARS, _m3.TARGET, _m3.filter_genes, _m3.load_full_logcpm, _m3.screen
)

RESULTS = paths.results(__file__)
FLAGGED = ["Jun", "Junb", "Gadd45g", "Nptx2", "Nptx1", "Egr1", "Fos", "Npas4",
           "Arc", "Homer1"]


def partial_spearman(x, y, covar_df):
    """Spearman of x vs y after residualising both on covar_df (ranks)."""
    df = pd.concat([pd.Series(x, name="x"), pd.Series(y, name="y"), covar_df],
                   axis=1).dropna()
    n = len(df)
    C = np.column_stack([np.ones(n)] +
                        [stats.rankdata(df[c]) for c in covar_df.columns])

    def resid(v):
        r = stats.rankdata(v)
        b, *_ = np.linalg.lstsq(C, r, rcond=None)
        return r - C @ b

    rx, ry = resid(df["x"]), resid(df["y"])
    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = n - 2 - covar_df.shape[1]
    t = r * np.sqrt(dof / (1 - r**2))
    return r, float(2 * stats.t.sf(abs(t), dof)), n


def main() -> None:
    meta = load_meta()
    l5 = l5_excitatory_ids(meta)
    ephys = load_ephys()

    for use_introns, tag in ((True, "exon+intron"), (False, "exon only")):
        print("=" * 74)
        print(f"quantification: {tag}")
        print("=" * 74)

        log_cpm, qc = load_full_logcpm(l5, use_introns=use_introns)
        expr, _ = filter_genes(log_cpm, drop_channels=True)
        design = meta.loc[expr.index].join(qc, how="left")
        y = ephys[TARGET].reindex(expr.index)
        fam = design["RNA family"]  # IT or ET within this population
        cov = design[COVARS]

        print(f"\n  cells: {len(expr)}  (IT {int((fam=='IT').sum())}, "
              f"ET {int((fam=='ET').sum())})")
        print(f"  rheobase by family: "
              f"IT median {y[fam=='IT'].median():.0f} pA, "
              f"ET median {y[fam=='ET'].median():.0f} pA, "
              f"Mann-Whitney p = "
              f"{stats.mannwhitneyu(y[fam=='IT'].dropna(), y[fam=='ET'].dropna())[1]:.2e}")

        # --- individual flagged genes -----------------------------------
        print(f"\n  {'gene':9s} {'rho':>7s} {'+cov':>7s} {'+cov+fam':>9s} "
              f"{'IT only':>8s} {'ET only':>8s}")
        rows = []
        for g in FLAGGED:
            if g not in expr.columns:
                print(f"  {g:9s}  (filtered out)")
                continue
            v = expr[g]
            r_raw, p_raw = stats.spearmanr(*pd.concat([v, y], axis=1).dropna().to_numpy().T)
            r_cov, p_cov, _ = partial_spearman(v, y, cov)
            cov_fam = cov.copy()
            cov_fam["is_ET"] = (fam == "ET").astype(float)
            r_fam, p_fam, _ = partial_spearman(v, y, cov_fam)
            sub_r = {}
            for f in ("IT", "ET"):
                mask = fam == f
                d = pd.concat([v[mask], y[mask]], axis=1).dropna()
                sub_r[f] = stats.spearmanr(*d.to_numpy().T)[0] if len(d) > 8 else np.nan
            print(f"  {g:9s} {r_raw:+7.3f} {r_cov:+7.3f} {r_fam:+9.3f} "
                  f"{sub_r['IT']:+8.3f} {sub_r['ET']:+8.3f}"
                  + ("  *" if p_fam < 0.05 else ""))
            rows.append({"gene": g, "quant": tag, "rho_raw": r_raw,
                         "rho_cov": r_cov, "p_cov": p_cov,
                         "rho_cov_fam": r_fam, "p_cov_fam": p_fam,
                         "rho_IT": sub_r["IT"], "rho_ET": sub_r["ET"]})

        # --- the module score -------------------------------------------
        seeds = pd.read_csv(RESULTS / "F_module_seeds.csv")["gene"].tolist()
        seeds = [g for g in seeds if g in expr.columns]
        se = expr[seeds]
        mod = ((se - se.mean()) / se.std(ddof=1)).mean(axis=1)

        r0, p0 = stats.spearmanr(*pd.concat([mod, y], axis=1).dropna().to_numpy().T)
        r1, p1, _ = partial_spearman(mod, y, cov)
        cov_fam = cov.copy()
        cov_fam["is_ET"] = (fam == "ET").astype(float)
        r2, p2, _ = partial_spearman(mod, y, cov_fam)

        print(f"\n  module score ({len(seeds)} seeds) vs rheobase:")
        print(f"    raw                       rho = {r0:+.3f}  p = {p0:.3g}")
        print(f"    + depth/holdtime          rho = {r1:+.3f}  p = {p1:.3g}")
        print(f"    + depth/holdtime + IT/ET  rho = {r2:+.3f}  p = {p2:.3g}")
        for f in ("IT", "ET"):
            mask = fam == f
            d = pd.concat([mod[mask], y[mask]], axis=1).dropna()
            if len(d) > 8:
                rr, pp = stats.spearmanr(*d.to_numpy().T)
                print(f"    within {f} only (n={len(d)})     rho = {rr:+.3f}  p = {pp:.3g}")

        # module score vs IT/ET identity itself
        auc = stats.mannwhitneyu(mod[fam == "IT"], mod[fam == "ET"])
        n1, n2 = int((fam == "IT").sum()), int((fam == "ET").sum())
        print(f"    module score separates IT from ET: AUC = "
              f"{auc[0]/(n1*n2):.3f}, p = {auc[1]:.3g}")

        pd.DataFrame(rows).to_csv(
            RESULTS / f"G_subtype_check_{'ei' if use_introns else 'exon'}.csv",
            index=False)
        print(f"\n[write] {RESULTS}/G_subtype_check_"
              f"{'ei' if use_introns else 'exon'}.csv\n")


if __name__ == "__main__":
    main()
