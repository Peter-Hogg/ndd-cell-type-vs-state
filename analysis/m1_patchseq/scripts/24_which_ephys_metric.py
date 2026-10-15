"""Which ephys feature best tracks activity state? Not just rheobase.

Rheobase is a composite (input resistance x threshold x channel complement),
and the intrinsic-plasticity literature points at more specific readouts:

  afterhyperpolarization   the classic learning-related intrinsic change
                           (AHP reduction, SK-channel modulation)
  sag ratio/area, rebound  Ih / HCN, bidirectionally regulated by LTP
  spike-freq / ISI adapt.  SK and M-current accommodation
  input resistance         general, but strongly type-determined
  membrane time constant   largely a cell-size axis

Instrument: the category-level test with a phenotype-permutation null (from 23).
The IEG transcription-factor category is the probe; `synaptic_plasticity` and
`stress_injury_ARG` are carried for comparison.

Multiplicity: one category x all features is a clean 29-test family per
population; BH applied within population.

Usage:  uv run python analysis/m1_patchseq/scripts/24_which_ephys_metric.py
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
from statsmodels.stats.multitest import multipletests

_m3 = import_module("03_gene_level_rheobase")
_m20 = import_module("20_layer_meta_analysis")
_m22 = import_module("22_repairs_excitatory")
_m23 = import_module("23_functional_categories_by_layer")
from plasticity_common import load_ephys, load_meta  # noqa: E402

OUT = paths.results(__file__)
N_PERM = 3000
RNG = np.random.default_rng(0)
PROBES = ["IEG_TF", "synaptic_plasticity", "stress_injury_ARG"]

# mechanistic annotation, for interpretation only -- not used in the statistics
MECHANISM = {
    "Afterhyperpolarization (mV)": "AHP / SK  ** classic intrinsic plasticity",
    "Afterdepolarization (mV)": "ADP / Ca-dependent",
    "Sag ratio": "Ih / HCN  ** LTP-regulated",
    "Sag area (mV*s)": "Ih / HCN  ** LTP-regulated",
    "Sag time (s)": "Ih / HCN",
    "Rebound (mV)": "Ih / HCN",
    "Rebound number of APs": "Ih / HCN",
    "Spike frequency adaptation": "SK / M-current accommodation",
    "ISI adaptation index": "SK / M-current accommodation",
    "ISI average adaptation index": "SK / M-current accommodation",
    "Input resistance (MOhm)": "leak conductance (type-confounded)",
    "Membrane time constant (ms)": "Rm x Cm -> CELL SIZE, not plasticity",
    "Rheobase (pA)": "composite excitability (used so far)",
    "AP threshold (mV)": "Nav availability",
    "AP width (ms)": "Kv / repolarisation",
    "Upstroke-to-downstroke ratio": "Nav / Kv kinetics",
    "Max number of APs": "output gain",
    "Burstiness": "output pattern",
}


def main() -> None:
    meta = load_meta()
    ephys = load_ephys()
    log_cpm, qc = _m3.load_full_logcpm(meta.index, use_introns=True)
    expr_all, _ = _m3.filter_genes(log_cpm, drop_channels=False)
    design = meta.loc[expr_all.index].join(qc, how="left")

    rt = design["RNA type"].astype(str)
    lay = pd.Series("other", index=design.index)
    for name, pat in _m20.LAYER_PATTERNS.items():
        lay[rt.str.match(pat, na=False)] = name
    design["layer"] = lay

    features = [c for c in ephys.columns if ephys[c].notna().sum() > 300]
    print(f"testing {len(features)} ephys features x {len(PROBES)} categories")

    # Layers are kept SEPARATE. IEG_TF can differ in sign between L2/3 and
    # L5 IT (script 23), so pooling them averages opposite-signed effects
    # -- the confound shown in Fig4. `pyramidal_POOLED` is retained only to
    # show what that pooling does, and must not be read as a result.
    pops = {
        "L2/3 IT": design.index[design["layer"] == "L2/3 IT"],
        "L4/5 IT": design.index[design["layer"] == "L4/5 IT"],
        "L5 IT": design.index[design["layer"] == "L5 IT"],
        "L5 ET": design.index[design["layer"] == "L5 ET"],
        "L6 CT": design.index[design["layer"] == "L6 CT"],
        "L6 IT": design.index[design["layer"] == "L6 IT"],
        "pyramidal_POOLED (invalid)": design.index[design["layer"] != "other"],
        "Pvalb (control)": design.index[design["RNA family"] == "Pvalb"],
    }

    rows = []
    for pop, pidx in pops.items():
        pidx = expr_all.index.intersection(pidx)
        if len(pidx) < 30:
            continue
        print(f"\n{'='*94}\n{pop}  (n up to {len(pidx)})\n{'='*94}")

        cat_cache = {}
        for feat in features:
            y = ephys[feat].reindex(pidx)
            idx = pidx[y.notna().values]
            if len(idx) < 40:
                continue
            d = design.loc[idx]
            n = len(idx)
            cov = d[_m3.COVARS].copy()
            srt = d["RNA type"]
            k = srt.value_counts()[lambda s: s >= 8].index
            srt = srt.where(srt.isin(k), "other")
            dums = pd.get_dummies(srt, prefix="t", drop_first=True).astype(float)
            C = _m22.build_design(cov, dums, n)
            yr = _m22.rank_resid(y.loc[idx].to_numpy(), C)
            yc = (yr - yr.mean()) / yr.std()
            groups = srt.to_numpy()

            perms = []
            for _ in range(N_PERM):
                yp = yr.copy()
                for g in np.unique(groups):
                    m = groups == g
                    yp[m] = RNG.permutation(yp[m])
                perms.append(yp - yp.mean())
            P = np.column_stack(perms)
            P /= np.where(P.std(0) > 0, P.std(0), np.nan)

            for cat in PROBES:
                key = (tuple(idx), cat)
                if key not in cat_cache:
                    cols = [g for g in _m23.CATEGORIES[cat] if g in expr_all.columns]
                    E = expr_all.loc[idx, cols]
                    E = E.loc[:, (E > 0).mean(axis=0) >= 0.25]
                    if E.shape[1] < 5:
                        cat_cache[key] = None
                    else:
                        Er = np.column_stack([_m22.rank_resid(E[g].to_numpy(), C)
                                              for g in E.columns])
                        Er -= Er.mean(0)
                        sd = Er.std(0)
                        Er /= np.where(sd > 0, sd, np.nan)
                        cat_cache[key] = Er
                Er = cat_cache[key]
                if Er is None:
                    continue
                obs = float(np.nanmean((Er * yc[:, None]).sum(0) / (n - 1)))
                nulls = np.nanmean((Er.T @ P) / (n - 1), axis=0)
                z = (obs - nulls.mean()) / nulls.std(ddof=1)
                p = float(2 * min((nulls <= obs).mean(), (nulls >= obs).mean()))
                rows.append({"population": pop, "feature": feat, "category": cat,
                             "n": n, "n_genes": Er.shape[1], "mean_rho": obs,
                             "z": z, "p": p,
                             "mechanism": MECHANISM.get(feat, "")})

        sub = pd.DataFrame([r for r in rows if r["population"] == pop])
        if sub.empty or "category" not in sub.columns:
            # every feature fell below the 40-cell minimum for this population
            print("  (no feature reached the minimum n; nothing to report)")
            continue
        for cat in PROBES:
            s = sub[sub["category"] == cat].copy()
            if not len(s):
                continue
            s["q_bh"] = multipletests(s["p"], method="fdr_bh")[1]
            s = s.reindex(s["z"].abs().sort_values(ascending=False).index)
            print(f"\n  -- probe: {cat} --")
            print(f"  {'feature':32s} {'n':>4s} {'mean rho':>9s} {'z':>7s} "
                  f"{'p':>7s} {'q':>7s}  mechanism")
            for _, r in s.head(8).iterrows():
                star = " *" if r.q_bh < 0.10 else ""
                print(f"  {r.feature[:32]:32s} {int(r.n):4d} {r.mean_rho:+9.4f} "
                      f"{r.z:+7.2f} {r.p:7.4f} {r.q_bh:7.4f}{star}  {r.mechanism}")

    R = pd.DataFrame(rows)
    q = []
    for (pop, cat), g in R.groupby(["population", "category"]):
        g = g.copy()
        g["q_bh"] = multipletests(g["p"], method="fdr_bh")[1]
        q.append(g)
    R = pd.concat(q, ignore_index=True)
    R.to_csv(OUT / "W_which_ephys_metric.csv", index=False)

    print("\n" + "=" * 94)
    print("EVERYTHING AT q<0.10, EXCITATORY POPULATIONS ONLY")
    print("=" * 94)
    sig = R[(R["q_bh"] < 0.10) & (~R["population"].str.contains("control"))]
    sig = sig.reindex(sig["z"].abs().sort_values(ascending=False).index)
    if len(sig):
        print(sig[["population", "category", "feature", "n", "mean_rho", "z",
                   "p", "q_bh", "mechanism"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    else:
        print("  nothing")
    ctl = R[(R["q_bh"] < 0.10) & (R["population"].str.contains("control"))]
    print(f"\n  Pvalb control hits at q<0.10: {len(ctl)}"
          + (f" ({', '.join(ctl['feature'].unique()[:5])})" if len(ctl) else ""))
    print(f"\n[write] {OUT/'W_which_ephys_metric.csv'}")


if __name__ == "__main__":
    main()
