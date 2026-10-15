"""What else does the IEG index predict in mouse M1?

The index (one composite score per cell over the 38-gene IEG_TF set, no ion
channels) is run against every ephys feature.

Two populations:
  L5 exc (n=126)
  pyramidal pooled (409) type in the model, better powered

Two models per feature:
  baseline        depth, genes detected, hold time, top50_share, type dummies
  + channel PCs   5 PCs of ion-channel expression, to test channel independence

BH-corrected across the feature family within each population. Features
mechanistically coupled to rheobase are flagged, so a hit there is not counted as
an independent discovery.

Usage:  uv run python analysis/m1_patchseq/scripts/32_index_all_features_M1.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)

import re
from importlib import import_module

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

_m3 = import_module("03_gene_level_rheobase")
_m20 = import_module("20_layer_meta_analysis")
_m22 = import_module("22_repairs_excitatory")
_m23 = import_module("23_functional_categories_by_layer")
_m31 = import_module("31_ieg_independent_of_channels")
from plasticity_common import load_ephys, load_meta  # noqa: E402

OUT = paths.results(__file__)
N_PERM = 3000
RNG = np.random.default_rng(0)
IEG = _m23.CATEGORIES["IEG_TF"]

# features that are arithmetically or biophysically tied to rheobase -- a hit here
# is not independent evidence
RHEO_LINKED = {"Rheobase (pA)", "Input resistance (MOhm)", "AP threshold (mV)",
               "Resting membrane potential (mV)", "Membrane time constant (ms)"}


def main() -> None:
    meta = load_meta(); ephys = load_ephys()
    log_cpm, qc = _m3.load_full_logcpm(meta.index, use_introns=True)
    expr, _ = _m3.filter_genes(log_cpm, drop_channels=False)
    d0 = meta.loc[expr.index].join(qc, how="left")
    rt = d0["RNA type"].astype(str)
    lay = pd.Series("other", index=d0.index)
    for nm, pat in _m20.LAYER_PATTERNS.items():
        lay[rt.str.match(pat, na=False)] = nm
    d0["layer"] = lay

    # top50 composition share, from CPM recovered out of log1p
    cpm = np.expm1(log_cpm.to_numpy(dtype=np.float64))
    tot = cpm.sum(1, keepdims=True); tot[tot == 0] = np.nan
    d0["top50_share"] = np.sort(cpm, axis=1)[:, ::-1][:, :50].sum(1) / tot[:, 0]

    pops = {
        "L5 exc (n=126)": d0.index[rt.str.match(r"^L5 (IT|ET)_\d+$", na=False)],
        "pyramidal pooled": d0.index[d0["layer"] != "other"],
    }
    feats = [c for c in ephys.columns if ephys[c].notna().sum() > 300]
    rows = []

    for pop, pidx in pops.items():
        pidx = expr.index.intersection(pidx)
        print(f"\n{'='*92}\n{pop}   {len(pidx)} cells   {len(feats)} features\n{'='*92}")
        for feat in feats:
            y = ephys[feat].reindex(pidx)
            idx = pidx[y.notna().values]
            if len(idx) < 40:
                continue
            d = d0.loc[idx]; n = len(idx)
            base = d[_m3.COVARS + ["top50_share"]]
            s = d["RNA type"]
            k = s.value_counts()[lambda x: x >= 8].index
            s = s.where(s.isin(k), "other")
            dums = pd.get_dummies(s, prefix="t", drop_first=True).astype(float)
            E = expr.loc[idx]

            ieg, n_ieg = _m31.index_score(E, IEG)
            pcs, n_chan, _ = _m31.chan_pcs(E)
            cov2 = base.copy()
            for i in range(pcs.shape[1]):
                cov2[f"chanPC{i+1}"] = pcs[:, i]

            C0 = _m22.build_design(base, dums, n)
            C1 = _m22.build_design(cov2, dums, n)
            r0, p0 = _m31.partial(ieg, y.loc[idx].to_numpy(), C0, n)
            r1, p1 = _m31.partial(ieg, y.loc[idx].to_numpy(), C1, n)

            # permutation null within type on the baseline model
            gi_ = _m22.rank_resid(ieg, C0); gi_ -= gi_.mean()
            gi_ /= gi_.std() if gi_.std() > 0 else 1
            yr = _m22.rank_resid(y.loc[idx].to_numpy(), C0)
            grp = s.to_numpy(); nulls = np.empty(N_PERM)
            for b in range(N_PERM):
                yp = yr.copy()
                for g in np.unique(grp):
                    m = grp == g
                    yp[m] = RNG.permutation(yp[m])
                yp = (yp - yp.mean()) / yp.std()
                nulls[b] = float((gi_ * yp).sum() / (n - 1))
            p_perm = float(2 * min((nulls <= r0).mean(), (nulls >= r0).mean()))

            rows.append({"population": pop, "feature": feat, "n": n,
                         "rho": r0, "p_param": p0, "p_perm": p_perm,
                         "rho_chanadj": r1, "p_chanadj": p1,
                         "rheobase_linked": feat in RHEO_LINKED})

        sub = pd.DataFrame([r for r in rows if r["population"] == pop])
        if sub.empty:
            continue
        sub["q_bh"] = multipletests(sub["p_perm"], method="fdr_bh")[1]
        sub = sub.reindex(sub["rho"].abs().sort_values(ascending=False).index)
        print(f"  {'feature':32s} {'n':>4s} {'rho':>7s} {'p_perm':>8s} {'q':>7s} "
              f"{'+chanPC':>8s}  link")
        for _, r in sub.head(12).iterrows():
            star = " *" if r.q_bh < 0.10 else "  "
            lk = "rheo-linked" if r.rheobase_linked else ""
            print(f"  {r.feature[:32]:32s} {int(r.n):4d} {r.rho:+7.3f} "
                  f"{r.p_perm:8.4f} {r.q_bh:7.4f}{star} {r.rho_chanadj:+8.3f}  {lk}")
        for k_, v in sub[["feature", "q_bh"]].values[:0]:
            pass

    R = pd.DataFrame(rows)
    q = []
    for pop, g in R.groupby("population"):
        g = g.copy(); g["q_bh"] = multipletests(g["p_perm"], method="fdr_bh")[1]
        q.append(g)
    R = pd.concat(q, ignore_index=True)
    R.to_csv(OUT / "AE_index_all_features_M1.csv", index=False)

    print("\n" + "=" * 92)
    print("EVERYTHING AT q<0.10")
    print("=" * 92)
    sig = R[R.q_bh < 0.10].reindex(
        R[R.q_bh < 0.10]["rho"].abs().sort_values(ascending=False).index)
    if len(sig):
        print(sig[["population", "feature", "n", "rho", "p_perm", "q_bh",
                   "rho_chanadj", "p_chanadj", "rheobase_linked"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        ind = sig[~sig.rheobase_linked]
        print(f"\n  of these, NOT arithmetically tied to rheobase: {len(ind)}")
        if len(ind):
            print(ind[["population", "feature", "rho", "q_bh"]]
                  .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    else:
        print("  nothing")
    print(f"\n[write] {OUT/'AE_index_all_features_M1.csv'}")


if __name__ == "__main__":
    main()
