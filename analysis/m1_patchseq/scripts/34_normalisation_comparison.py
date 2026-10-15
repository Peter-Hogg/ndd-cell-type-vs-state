"""Does housekeeping normalisation fix the composition confound?

The IEG index can track library composition (top-50 transcript share), which
confounds its rheobase association. One remedy is to normalise every gene to a
housekeeping reference first, qPCR-style.

Concern: if housekeeping genes ARE the top-50 genes, a gene/housekeeping ratio
would move twice when composition shifts and amplify the confound. If the top-50
are mostly mitochondrial or non-coding transcripts rather than classic
housekeeping genes, the concern does not apply. This is checked directly.

Four normalisations of the same IEG_TF gene set, compared on two criteria:

  CONFOUND   |rho(index, top50_share)|   smaller is better
  SIGNAL     rho(index, rheobase) with composition already controlled

  1 cpm            log1p(CPM), mean over genes           (baseline)
  2 housekeeping   log CPM(gene) - log geomean(HK genes)
  3 matched_ref    log CPM(gene) - log CPM(expression-matched non-HK genes)
  4 within_rank    per-cell rank of each gene / n_genes   (scale-invariant)

Also reports whether the housekeeping genes are themselves stable, since an
unstable reference is worse than none.

Usage:  uv run python analysis/m1_patchseq/scripts/34_normalisation_comparison.py
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

_m3 = import_module("03_gene_level_rheobase")
_m22 = import_module("22_repairs_excitatory")
_m23 = import_module("23_functional_categories_by_layer")
_m31 = import_module("31_ieg_independent_of_channels")
from plasticity_common import load_ephys, load_meta  # noqa: E402

OUT = paths.results(__file__)
RHEO = "Rheobase (pA)"
IEG = _m23.CATEGORIES["IEG_TF"]

HK = ["Gapdh", "Actb", "Actg1", "Tuba1a", "Tuba1b", "Tubb5", "Ubb", "Ubc",
      "Eef1a1", "Eef2", "Rplp0", "Rps18", "Rpl13a", "Ppia", "Pgk1", "Ldha",
      "Aldoa", "Eno1", "Atp5b", "Hprt", "Tbp", "B2m", "Sdha", "Ywhaz"]


def main() -> None:
    meta = load_meta(); ephys = load_ephys()
    log_cpm, qc = _m3.load_full_logcpm(meta.index, use_introns=True)
    cpm = np.expm1(log_cpm.to_numpy(dtype=np.float64))
    tot = cpm.sum(1, keepdims=True); tot[tot == 0] = np.nan

    d0 = meta.loc[log_cpm.index].join(qc, how="left")
    srt = np.sort(cpm, axis=1)[:, ::-1]
    d0["top50_share"] = srt[:, :50].sum(1) / tot[:, 0]

    # ---- what ARE the top-50 genes? -----------------------------------------
    med = pd.Series(np.nanmedian(cpm, axis=0), index=log_cpm.columns)
    top50 = med.nlargest(50)
    hk_present = [g for g in HK if g in log_cpm.columns]
    overlap = [g for g in top50.index if g in hk_present]
    print("=" * 84)
    print("IS THE TOP-50 ACTUALLY HOUSEKEEPING GENES?")
    print("=" * 84)
    print(f"  top-50 by median CPM, first 20:\n    {list(top50.index[:20])}")
    print(f"\n  classic housekeeping genes among the top-50: {len(overlap)}/50"
          f"   {overlap}")
    print(f"  -> {'HK genes dominate the top-50' if len(overlap) >= 8 else 'top-50 are mostly other transcripts'}")

    # ---- are the HK genes stable? ------------------------------------------
    print("\n" + "=" * 84)
    print("ARE THE HOUSEKEEPING GENES STABLE? (an unstable reference is worse than none)")
    print("=" * 84)
    rows = []
    for g in hk_present:
        v = log_cpm[g].to_numpy()
        r_comp = stats.spearmanr(v, d0["top50_share"])[0]
        rows.append({"gene": g, "median_cpm": med[g], "sd_log": float(v.std(ddof=1)),
                     "r_with_composition": r_comp})
    H = pd.DataFrame(rows).sort_values("r_with_composition")
    print(H.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  mean |r with composition| across HK genes: "
          f"{H.r_with_composition.abs().mean():.3f}")

    # ---- population --------------------------------------------------------
    rt = d0["RNA type"].astype(str)
    idx = log_cpm.index[rt.str.match(r"^L5 (IT|ET)_\d+$", na=False)]
    idx = idx[ephys[RHEO].reindex(idx).notna().values]
    d = d0.loc[idx]; n = len(idx)
    y = ephys[RHEO].reindex(idx).to_numpy()
    s = d["RNA type"]; k = s.value_counts()[lambda x: x >= 8].index
    s = s.where(s.isin(k), "other")
    dums = pd.get_dummies(s, prefix="t", drop_first=True).astype(float)
    print(f"\n[pop] L5 exc n={n}")

    E = log_cpm.loc[idx]
    ieg_cols = [g for g in IEG if g in E.columns and (E[g] > 0).mean() >= 0.25]
    hk_cols = [g for g in hk_present if (E[g] > 0).mean() >= 0.5]

    # 1 cpm
    v_cpm = E[ieg_cols].apply(lambda c: (c - c.mean()) / c.std(ddof=1)).mean(axis=1)
    # 2 housekeeping: log ratio to geometric mean of HK (mean of logs)
    hk_ref = E[hk_cols].mean(axis=1)
    v_hk = (E[ieg_cols].sub(hk_ref, axis=0)).apply(
        lambda c: (c - c.mean()) / c.std(ddof=1)).mean(axis=1)
    # 3 matched reference (non-HK, expression matched)
    pool = [g for g in E.columns if g not in set(ieg_cols) | set(hk_cols)
            and (E[g] > 0).mean() >= 0.25]
    pm = E[pool].mean()
    ref = []
    for g in ieg_cols:
        cand = (pm - E[g].mean()).abs().sort_values()
        for c in cand.index:
            if c not in ref:
                ref.append(c); break
    mref = E[ref].mean(axis=1)
    v_ref = (E[ieg_cols].sub(mref, axis=0)).apply(
        lambda c: (c - c.mean()) / c.std(ddof=1)).mean(axis=1)
    # 4 within-cell rank
    Rk = E.rank(axis=1, pct=True)
    v_rank = Rk[ieg_cols].apply(lambda c: (c - c.mean()) / c.std(ddof=1)).mean(axis=1)

    variants = {"1 cpm (current)": v_cpm, "2 housekeeping ratio": v_hk,
                "3 matched reference": v_ref, "4 within-cell rank": v_rank}

    print("\n" + "=" * 84)
    print("NORMALISATION COMPARISON — IEG index, L5 exc")
    print("=" * 84)
    print(f"  {'normalisation':24s} {'|r| w/ composition':>19s} "
          f"{'rho vs rheo':>12s} {'p':>9s} {'+comp ctrl':>11s} {'p':>9s}")
    res = []
    base_cols = _m3.COVARS
    C0 = _m22.build_design(d[base_cols], dums, n)
    C1 = _m22.build_design(d[base_cols + ["top50_share"]], dums, n)
    for lab, v in variants.items():
        vv = v.to_numpy()
        rc = stats.spearmanr(vv, d["top50_share"])[0]
        r0, p0 = _m31.partial(vv, y, C0, n)
        r1, p1 = _m31.partial(vv, y, C1, n)
        print(f"  {lab:24s} {abs(rc):19.3f} {r0:+12.3f} {p0:9.4g} "
              f"{r1:+11.3f} {p1:9.4g}")
        res.append({"normalisation": lab, "abs_r_composition": abs(rc),
                    "rho_rheobase": r0, "p": p0,
                    "rho_rheobase_compctrl": r1, "p_compctrl": p1})

    R = pd.DataFrame(res)
    R.to_csv(OUT / "AG_normalisation_comparison.csv", index=False)
    print("\n  Goal: small |r with composition| AND a surviving rho under")
    print("  composition control. A normalisation that only shrinks the raw rho")
    print("  has removed signal, not confound.")
    print(f"\n[write] {OUT/'AG_normalisation_comparison.csv'}")


if __name__ == "__main__":
    main()
