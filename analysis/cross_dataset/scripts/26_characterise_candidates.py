"""Characterise Jarid2 and Gphn: activity-state predictor, or type/size marker?

A gene is only useful here if its variation reflects activity state rather than
cell identity or cell size. Ion channels are excluded by design: Kcnip1 is
carried only as a calibration anchor for what a biophysical/type gene looks like,
and Jun as the anchor for a known activity gene.

Six criteria, each with an anchor to calibrate against:

  ATLAS (48,796 glutamatergic cells — well powered)
   1 eta² by subclass / by leaf cluster    low  = not type-determined
   2 within-cluster spread vs an
     expression-matched null (z)           high = real biological variation
   3 within-cluster correlation with the
     Jun/AP-1 state axis                   high = co-regulated with activity

  PATCH-SEQ (n=410 glutamatergic, within-type)
   4 rho vs rheobase, covariate-adjusted   the effect of interest
   5 rho vs membrane time constant AND
     rho vs rheobase partialling tau       does it survive the CELL-SIZE axis?
   6 per-layer estimates + Cochran Q       homogeneous, or one-layer artifact?

Criterion 5 is the decisive one. Membrane time constant is a cell-size axis
(tau = Rm x Cm); a candidate whose rheobase association vanishes once tau is
partialled out is a size proxy, not a state predictor.

Usage:
  uv run python analysis/cross_dataset/scripts/26_characterise_candidates.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)

from importlib import import_module

import h5py
import numpy as np
import pandas as pd
from scipy import stats

_m3 = import_module("03_gene_level_rheobase")
_m20 = import_module("20_layer_meta_analysis")
_m22 = import_module("22_repairs_excitatory")
PANELS = import_module("11_atlas_extract_panels").PANELS
from plasticity_common import load_ephys, load_meta  # noqa: E402

ROOT = paths.REPO
H5 = ROOT / "Gene_Expression" / "Data" / "expression_matrix.hdf5"
ATLAS_META = ROOT / "Gene_Expression" / "Data" / "metadata.csv"
OUT = paths.results(__file__)

CANDIDATES = ["Jarid2", "Gphn"]
ANCHORS = ["Jun", "Kcnip1"]           # activity anchor / channel anchor
GENES = CANDIDATES + ANCHORS
TAU = "Membrane time constant (ms)"
RHEO = "Rheobase (pA)"
N_NULL = 400
MIN_CLUSTER_N = 20
RNG = np.random.default_rng(0)


def atlas_part() -> pd.DataFrame:
    meta = pd.read_csv(ATLAS_META, low_memory=False).set_index("sample_name")
    rows = {}
    with h5py.File(H5, "r") as f:
        genes = np.array([g.decode() for g in f["gene_names"][:]])
        cells = np.array([c.decode() for c in f["sample_names"][:]])
        libsize = np.asarray(f["data/total_exon_counts"][:], dtype=np.float64)
        grp = f["data/exon"]; p = np.asarray(grp["p"][:], dtype=np.int64)
        n_all = int(grp["dims"][0]); gi = {g: i for i, g in enumerate(genes)}
        md = meta.reindex(cells)
        sel = np.array(md["class_label"].astype(str)
                       .str.contains("Glutamatergic", na=False), dtype=bool)
        clu = md["cluster_label"].to_numpy()
        vc = pd.Series(clu[sel]).value_counts()
        big = set(vc[vc >= MIN_CLUSTER_N].index)
        sel &= np.array([c in big for c in clu])
        idx = np.flatnonzero(sel)
        cats = pd.Series(clu[idx]).astype("category")
        codes = cats.cat.codes.to_numpy(); ng = len(cats.cat.categories)
        cnt = np.bincount(codes, minlength=ng).astype(float)
        subc = md["subclass_label"].to_numpy()[idx]
        lib = libsize[idx]
        print(f"[atlas] {len(idx)} glutamatergic cells, {ng} clusters")

        def logcpm(j):
            lo, hi = p[j], p[j + 1]
            v = np.zeros(n_all)
            if hi > lo:
                v[np.asarray(grp["i"][lo:hi], dtype=np.int64)] = \
                    np.asarray(grp["x"][lo:hi], dtype=np.float64)
            return np.log1p(v[idx] / lib * 1e6)

        def wc(v):
            m = np.bincount(codes, weights=v, minlength=ng) / cnt
            return v - m[codes]

        def eta2(v, g):
            gm = v.mean(); ssb = 0.0
            for u in pd.unique(g):
                m = g == u
                if m.sum() > 1:
                    ssb += m.sum() * (v[m].mean() - gm) ** 2
            return float(ssb / ((v - gm) ** 2).sum())

        # state axis
        cols = [g for g in PANELS["jun_ap1"] if g in gi]
        Z = np.column_stack([logcpm(gi[g]) for g in cols])
        sd = Z.std(0, ddof=1)
        Z = (Z - Z.mean(0)) / np.where(sd > 0, sd, np.nan)
        state = wc(np.nanmean(Z, axis=1)); state -= state.mean()
        sn = np.sqrt((state ** 2).sum())

        # expression-matched null pool: mean-expression deciles
        means = {}
        step = max(1, len(genes) // 3000)
        pool = list(range(0, len(genes), step))
        for j in pool:
            lo, hi = p[j], p[j + 1]
            means[j] = (hi - lo)          # detection count as a cheap proxy
        pool_arr = np.array(sorted(means, key=lambda j: means[j]))

        for gene in GENES:
            if gene not in gi:
                continue
            j = gi[gene]
            v = logcpm(j)
            r = wc(v); r -= r.mean()
            den = np.sqrt((r ** 2).sum()) * sn
            r_state = float((r * state).sum() / den) if den > 0 else np.nan
            sd_within = float(r.std(ddof=1))
            # matched null: nearest-neighbour genes by detection count
            pos = int(np.searchsorted(np.array([means[k] for k in pool_arr]),
                                      p[j + 1] - p[j]))
            lo_i, hi_i = max(0, pos - 150), min(len(pool_arr), pos + 150)
            cand = [k for k in pool_arr[lo_i:hi_i] if k != j]
            nulls = []
            for k in RNG.choice(cand, size=min(N_NULL, len(cand)), replace=False):
                rk = wc(logcpm(int(k)))
                nulls.append(rk.std(ddof=1))
            nulls = np.asarray(nulls)
            rows[gene] = {
                "atlas_eta2_subclass": eta2(v, subc),
                "atlas_eta2_cluster": eta2(v, cats.to_numpy()),
                "atlas_sd_within": sd_within,
                "atlas_null_sd": float(nulls.mean()),
                "atlas_spread_z": float((sd_within - nulls.mean()) / nulls.std(ddof=1)),
                "atlas_r_state_axis": r_state,
            }
            print(f"  {gene:8s} done")
    return pd.DataFrame(rows).T


def patchseq_part() -> pd.DataFrame:
    meta = load_meta(); ephys = load_ephys()
    log_cpm, qc = _m3.load_full_logcpm(meta.index, use_introns=True)
    expr, _ = _m3.filter_genes(log_cpm, drop_channels=False)
    design = meta.loc[expr.index].join(qc, how="left")
    rt = design["RNA type"].astype(str)
    lay = pd.Series("other", index=design.index)
    for nm, pat in _m20.LAYER_PATTERNS.items():
        lay[rt.str.match(pat, na=False)] = nm
    design["layer"] = lay
    idx = expr.index[(design["layer"] != "other")]
    idx = idx[ephys[RHEO].reindex(idx).notna().values]
    d = design.loc[idx]; n = len(idx)
    print(f"\n[patch-seq] {n} pyramidal cells with rheobase")

    cov = d[_m3.COVARS].copy()
    srt = d["RNA type"]
    k = srt.value_counts()[lambda s: s >= 8].index
    srt = srt.where(srt.isin(k), "other")
    dums = pd.get_dummies(srt, prefix="t", drop_first=True).astype(float)
    C = _m22.build_design(cov, dums, n)

    y = _m22.rank_resid(ephys[RHEO].reindex(idx).to_numpy(), C)
    tau = _m22.rank_resid(ephys[TAU].reindex(idx).to_numpy(), C)
    ieg = [g for g in PANELS["canonical_ieg"] if g in expr.columns]
    Zi = expr.loc[idx, ieg]
    Zi = (Zi - Zi.mean()) / Zi.std(ddof=1)
    iegs = _m22.rank_resid(Zi.mean(axis=1).to_numpy(), C)

    def pear(a, b):
        a = a - a.mean(); b = b - b.mean()
        den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        if den == 0:
            return np.nan, np.nan
        r = float((a * b).sum() / den)
        dof = n - 2 - (C.shape[1] - 1)
        t = r * np.sqrt(dof / max(1e-12, 1 - r ** 2))
        return r, float(2 * stats.t.sf(abs(t), dof))

    rows = {}
    for gene in GENES:
        if gene not in expr.columns:
            continue
        g = _m22.rank_resid(expr.loc[idx, gene].to_numpy(), C)
        r_rheo, p_rheo = pear(g, y)
        r_tau, _ = pear(g, tau)
        r_ieg, p_ieg = pear(g, iegs)
        # rheobase partialling out tau
        b = (g * tau).sum() / (tau ** 2).sum()
        g_t = g - b * tau
        b2 = (y * tau).sum() / (tau ** 2).sum()
        y_t = y - b2 * tau
        r_adj, p_adj = pear(g_t, y_t)
        # per-layer + Cochran Q
        zs, ns = [], []
        for lname in _m20.LAYER_PATTERNS:
            li = d.index[d["layer"] == lname]
            li = idx.intersection(li)
            if len(li) < 30:
                continue
            sub = pd.DataFrame({"g": expr.loc[li, gene],
                                "y": ephys[RHEO].reindex(li)}).dropna()
            rr = stats.spearmanr(sub["g"], sub["y"])[0]
            zs.append(np.arctanh(np.clip(rr, -0.999, 0.999))); ns.append(len(sub))
        w = np.array(ns, dtype=float) - 3
        zbar = (w * np.array(zs)).sum() / w.sum()
        Q = float((w * (np.array(zs) - zbar) ** 2).sum())
        dfq = len(zs) - 1
        rows[gene] = {
            "pseq_rho_rheobase": r_rheo, "pseq_p_rheobase": p_rheo,
            "pseq_rho_tau": r_tau,
            "pseq_rho_rheo_given_tau": r_adj, "pseq_p_rheo_given_tau": p_adj,
            "pseq_rho_vs_IEGscore": r_ieg, "pseq_p_vs_IEGscore": p_ieg,
            "pseq_Q_p": float(stats.chi2.sf(Q, dfq)),
            "pseq_I2": max(0.0, (Q - dfq) / Q) * 100 if Q > 0 else 0.0,
        }
    return pd.DataFrame(rows).T


def main() -> None:
    A = atlas_part()
    P = patchseq_part()
    T = A.join(P, how="outer").reindex(GENES)
    T.to_csv(OUT / "Y_candidate_characterisation.csv")

    lbl = {"Jarid2": "candidate", "Gphn": "candidate",
           "Jun": "ANCHOR: activity gene", "Kcnip1": "ANCHOR: ion channel"}
    print("\n" + "=" * 92)
    print("IS IT A STATE PREDICTOR?")
    print("=" * 92)
    for g in GENES:
        if g not in T.index:
            continue
        r = T.loc[g]
        print(f"\n{g}  [{lbl.get(g,'')}]")
        print(f"  ATLAS  eta² subclass {r.atlas_eta2_subclass:.3f} | "
              f"cluster {r.atlas_eta2_cluster:.3f}   (low = not type-determined)")
        print(f"         within-cluster spread z vs matched null "
              f"{r.atlas_spread_z:+.1f}   (high = real variation)")
        print(f"         r with Jun/AP-1 state axis {r.atlas_r_state_axis:+.3f}"
              f"   (high = co-regulated with activity)")
        print(f"  PSEQ   rho vs rheobase {r.pseq_rho_rheobase:+.3f} "
              f"(p={r.pseq_p_rheobase:.3g})")
        print(f"         rho vs tau (CELL SIZE) {r.pseq_rho_tau:+.3f}")
        print(f"         rho vs rheobase | tau  {r.pseq_rho_rheo_given_tau:+.3f} "
              f"(p={r.pseq_p_rheo_given_tau:.3g})   <-- decisive")
        print(f"         rho vs IEG composite   {r.pseq_rho_vs_IEGscore:+.3f} "
              f"(p={r.pseq_p_vs_IEGscore:.3g})")
        print(f"         layer homogeneity Q p={r.pseq_Q_p:.3g}, I²={r.pseq_I2:.0f}%")
    print(f"\n[write] {OUT/'Y_candidate_characterisation.csv'}")


if __name__ == "__main__":
    main()
