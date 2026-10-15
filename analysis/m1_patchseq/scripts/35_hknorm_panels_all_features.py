"""With housekeeping normalisation, does any panel predict any ephys feature?

Housekeeping normalisation reduces the composition confound on the activity
index (`34_normalisation_comparison.py`). Composition can *mask* real signal as
well as create spurious signal, so every panel is re-tested under HK
normalisation.

Index construction, per cell:
    HK-normalised gene = log1p(CPM_gene) - mean(log1p(CPM) over housekeeping genes)
    index = mean of z-scored HK-normalised values across the panel

Panels: the two activity families and the two scaffolding families, each tested
against all 29 features.

Populations: L5 excitatory (n=126) and all pyramidal with type in the model
(n=409, better powered).

Phenotype-permutation null within transcriptomic type; BH across the 29-feature
family within each panel x population.

Usage:  uv run python analysis/m1_patchseq/scripts/35_hknorm_panels_all_features.py
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
_m31 = import_module("31_ieg_independent_of_channels")
PANELS = import_module("11_atlas_extract_panels").PANELS
from plasticity_common import load_ephys, load_meta  # noqa: E402

OUT = paths.results(__file__)
N_PERM = 3000
RNG = np.random.default_rng(0)

HK = ["Gapdh", "Actb", "Actg1", "Tuba1a", "Tuba1b", "Tubb5", "Ubb", "Ubc",
      "Eef1a1", "Eef2", "Rplp0", "Rps18", "Rpl13a", "Ppia", "Pgk1", "Ldha",
      "Aldoa", "Eno1", "Atp5b", "Hprt", "Tbp", "B2m", "Sdha", "Ywhaz"]

TEST_PANELS = {
    "IEG_TF (activity)": _m23.CATEGORIES["IEG_TF"],
    "canonical_ieg (activity)": PANELS["canonical_ieg"],
    "jun_ap1 (activity)": PANELS["jun_ap1"],
    "delayed_effector (SCAFFOLD)": PANELS["delayed_effector"],
    "structural_late (SCAFFOLD)": _m23.CATEGORIES["structural_late"],
    "synaptic_plasticity (mixed)": _m23.CATEGORIES["synaptic_plasticity"],
}


def main() -> None:
    meta = load_meta(); ephys = load_ephys()
    log_cpm, qc = _m3.load_full_logcpm(meta.index, use_introns=True)
    d0 = meta.loc[log_cpm.index].join(qc, how="left")
    rt = d0["RNA type"].astype(str)
    lay = pd.Series("other", index=d0.index)
    for nm, pat in _m20.LAYER_PATTERNS.items():
        lay[rt.str.match(pat, na=False)] = nm
    d0["layer"] = lay

    hk_cols = [g for g in HK if g in log_cpm.columns
               and (log_cpm[g] > 0).mean() >= 0.5]
    print(f"[hk] {len(hk_cols)}/{len(HK)} housekeeping genes usable")
    hk_ref = log_cpm[hk_cols].mean(axis=1)
    hknorm = log_cpm.sub(hk_ref, axis=0)      # log ratio to HK reference

    pops = {
        "L5 exc": d0.index[rt.str.match(r"^L5 (IT|ET)_\d+$", na=False)],
        "pyramidal pooled": d0.index[d0["layer"] != "other"],
    }
    feats = [c for c in ephys.columns if ephys[c].notna().sum() > 300]
    rows = []

    for pop, pidx in pops.items():
        pidx = hknorm.index.intersection(pidx)
        print(f"\n{'='*94}\n{pop}   {len(pidx)} cells\n{'='*94}")
        for feat in feats:
            y = ephys[feat].reindex(pidx)
            idx = pidx[y.notna().values]
            if len(idx) < 40:
                continue
            d = d0.loc[idx]; n = len(idx)
            s = d["RNA type"]
            k = s.value_counts()[lambda x: x >= 8].index
            s = s.where(s.isin(k), "other")
            dums = pd.get_dummies(s, prefix="t", drop_first=True).astype(float)
            C = _m22.build_design(d[_m3.COVARS], dums, n)
            yr = _m22.rank_resid(y.loc[idx].to_numpy(), C)
            grp = s.to_numpy()
            perms = []
            for _ in range(N_PERM):
                yp = yr.copy()
                for g in np.unique(grp):
                    m = grp == g
                    yp[m] = RNG.permutation(yp[m])
                yp = (yp - yp.mean()) / yp.std()
                perms.append(yp)
            P = np.column_stack(perms)

            for pname, genes in TEST_PANELS.items():
                v, ng = _m31.index_score(hknorm.loc[idx], genes)
                if ng < 5:
                    continue
                gv = _m22.rank_resid(v, C); gv -= gv.mean()
                gv = gv / gv.std() if gv.std() > 0 else gv
                yc = (yr - yr.mean()) / yr.std()
                obs = float((gv * yc).sum() / (n - 1))
                nulls = (gv @ P) / (n - 1)
                p = float(2 * min((nulls <= obs).mean(), (nulls >= obs).mean()))
                rows.append({"population": pop, "panel": pname, "feature": feat,
                             "n": n, "n_genes": ng, "rho": obs, "p_perm": p})

    R = pd.DataFrame(rows)
    out = []
    for (pop, pname), g in R.groupby(["population", "panel"]):
        g = g.copy()
        g["q_bh"] = multipletests(g["p_perm"], method="fdr_bh")[1]
        out.append(g)
    R = pd.concat(out, ignore_index=True)
    R.to_csv(OUT / "AH_hknorm_panels_all_features.csv", index=False)

    for pop in pops:
        print(f"\n{'='*94}\n{pop} — best feature per panel (HK-normalised)\n{'='*94}")
        sub = R[R.population == pop]
        print(f"  {'panel':30s} {'g':>3s} {'best feature':28s} {'rho':>7s} "
              f"{'p':>8s} {'q':>7s}")
        for pname in TEST_PANELS:
            s = sub[sub.panel == pname]
            if s.empty:
                continue
            b = s.loc[s["rho"].abs().idxmax()]
            star = " *" if b.q_bh < 0.10 else ""
            print(f"  {pname:30s} {int(b.n_genes):3d} {b.feature[:28]:28s} "
                  f"{b.rho:+7.3f} {b.p_perm:8.4f} {b.q_bh:7.4f}{star}")

    print("\n" + "=" * 94)
    print("EVERYTHING AT q<0.10")
    print("=" * 94)
    sig = R[R.q_bh < 0.10].reindex(
        R[R.q_bh < 0.10]["rho"].abs().sort_values(ascending=False).index)
    if len(sig):
        print(sig[["population", "panel", "feature", "n", "rho", "p_perm", "q_bh"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    else:
        print("  NOTHING at q<0.10 under housekeeping normalisation")
    print(f"\n[write] {OUT/'AH_hknorm_panels_all_features.csv'}")


if __name__ == "__main__":
    main()
