"""SFARI ASD risk genes in the Allen SmartSeq atlas: type-specificity and within-cluster spread.

Uses Allen SmartSeq atlas (Tasic et al. 2018) glutamatergic cells (leaf clusters
>= 20 cells) and scored SFARI genes (ref/sfari_genes.csv) expressed in >= 15% of
cells. Four descriptions, each against a background matched on mean expression x
detection fraction:

  1 eta^2 by subclass and by leaf cluster   -- variance explained by cell-type
                                               identity
  2 within-cluster spread vs matched null   -- within-cluster SD and SD fraction
  3 subclass expression profile             -- SFARI minus background mean
                                               expression per pyramidal subclass
  4 rank among the most type-variable genes -- AUC of SFARI membership against
                                               cluster eta^2

Usage:
  uv run python analysis/allen_atlas/scripts/39_asd_atlas_patterns.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)


import h5py
import numpy as np
import pandas as pd
from scipy import stats


ROOT = paths.REPO
H5 = ROOT / "Gene_Expression" / "Data" / "expression_matrix.hdf5"
META = ROOT / "Gene_Expression" / "Data" / "metadata.csv"
SFARI = ROOT / "ref" / "sfari_genes.csv"
OUT = paths.results(__file__)
MIN_CLUSTER_N = 20
MIN_DETECT = 0.15
N_BG = 6000
N_NULL = 400
N_BINS = 10
RNG = np.random.default_rng(0)


def main() -> None:
    s = pd.read_csv(SFARI)
    s = s[s["gene-score"].notna()]
    sf_all = set(s["gene-symbol"].astype(str).str.upper())
    sf_1 = set(s.loc[s["gene-score"] == 1, "gene-symbol"].astype(str).str.upper())
    print(f"[sfari] {len(sf_all)} scored, {len(sf_1)} score-1")

    meta = pd.read_csv(META, low_memory=False).set_index("sample_name")
    with h5py.File(H5, "r") as f:
        genes = np.array([g.decode() for g in f["gene_names"][:]])
        cells = np.array([c.decode() for c in f["sample_names"][:]])
        lib = np.asarray(f["data/total_exon_counts"][:], dtype=np.float64)
        grp = f["data/exon"]
        p = np.asarray(grp["p"][:], dtype=np.int64)
        nall = int(grp["dims"][0])
        gi = {g: i for i, g in enumerate(genes)}
        md = meta.reindex(cells)
        sel = np.array(md["class_label"].astype(str)
                       .str.contains("Glutamatergic", na=False), dtype=bool)
        clu = md["cluster_label"].to_numpy()
        vc = pd.Series(clu[sel]).value_counts()
        big = set(vc[vc >= MIN_CLUSTER_N].index)
        sel &= np.array([c in big for c in clu])
        idx = np.flatnonzero(sel)
        cats = pd.Series(clu[idx]).astype("category")
        codes = cats.cat.codes.to_numpy()
        ng = len(cats.cat.categories)
        cnt = np.bincount(codes, minlength=ng).astype(float)
        subc = pd.Series(md["subclass_label"].to_numpy()[idx]).astype(str)
        L = lib[idx]
        print(f"[atlas] {len(idx)} glutamatergic cells, {ng} clusters, "
              f"{subc.nunique()} subclasses")

        def lc(j):
            lo, hi = p[j], p[j + 1]
            v = np.zeros(nall)
            if hi > lo:
                v[np.asarray(grp["i"][lo:hi], dtype=np.int64)] = \
                    np.asarray(grp["x"][lo:hi], dtype=np.float64)
            return np.log1p(v[idx] / L * 1e6)

        # SFARI genes present in the atlas + a random background pool
        sf_present = [g for g in genes if g.upper() in sf_all]
        pool = [g for g in genes if g.upper() not in sf_all]
        bg = list(RNG.choice(pool, size=min(N_BG, len(pool)), replace=False))
        cache = {}
        for g in sf_present + bg:
            if g in gi:
                cache[g] = lc(gi[g])
    detect = {g: float((v > 0).mean()) for g, v in cache.items()}
    meanx = {g: float(v.mean()) for g, v in cache.items()}
    sf_use = [g for g in sf_present if g in cache and detect[g] >= MIN_DETECT]
    bg_use = [g for g in bg if g in cache and detect[g] >= MIN_DETECT]
    print(f"[genes] SFARI expressed >= {MIN_DETECT:.0%}: {len(sf_use)}; "
          f"background {len(bg_use)}")

    sub_codes = subc.astype("category").cat.codes.to_numpy()
    n_sub = sub_codes.max() + 1

    def eta2(v, g, k):
        gm = v.mean()
        c = np.bincount(g, minlength=k).astype(float)
        m = np.bincount(g, weights=v, minlength=k) / np.where(c > 0, c, 1)
        ssb = float((c * (m - gm) ** 2).sum())
        return ssb / float(((v - gm) ** 2).sum())

    def within_sd(v):
        m = np.bincount(codes, weights=v, minlength=ng) / cnt
        return float((v - m[codes]).std(ddof=1))

    rec = []
    for g in sf_use + bg_use:
        v = cache[g]
        rec.append({"gene": g, "is_sfari": g in set(sf_use),
                    "is_sfari1": g.upper() in sf_1,
                    "mean": meanx[g], "detect": detect[g],
                    "eta2_subclass": eta2(v, sub_codes, n_sub),
                    "eta2_cluster": eta2(v, codes, ng),
                    "within_sd": within_sd(v), "total_sd": float(v.std(ddof=1))})
    D = pd.DataFrame(rec)
    D["within_frac"] = D["within_sd"] / D["total_sd"]
    D.to_csv(OUT / "AN_asd_atlas_gene_stats.csv", index=False)

    # expression-matched comparison: bin background on mean x detection
    bgD = D[~D.is_sfari]
    bm = np.clip(np.digitize(bgD["mean"], np.quantile(bgD["mean"], np.linspace(0, 1, N_BINS + 1)[1:-1])), 0, N_BINS - 1)
    bd = np.clip(np.digitize(bgD["detect"], np.quantile(bgD["detect"], np.linspace(0, 1, N_BINS + 1)[1:-1])), 0, N_BINS - 1)
    bgD = bgD.assign(key=bm * N_BINS + bd)
    pools = {k: v for k, v in bgD.groupby("key")}
    sfD = D[D.is_sfari].copy()
    qm = np.quantile(bgD["mean"], np.linspace(0, 1, N_BINS + 1)[1:-1])
    qd = np.quantile(bgD["detect"], np.linspace(0, 1, N_BINS + 1)[1:-1])
    sfD["key"] = (np.clip(np.digitize(sfD["mean"], qm), 0, N_BINS - 1) * N_BINS
                  + np.clip(np.digitize(sfD["detect"], qd), 0, N_BINS - 1))
    sfD = sfD[sfD["key"].isin(pools)]

    print("\n" + "=" * 92)
    print("SFARI GENES vs EXPRESSION-MATCHED BACKGROUND — Allen atlas, glutamatergic")
    print("=" * 92)
    print(f"  {'statistic':22s} {'SFARI':>9s} {'matched null':>22s} {'z':>7s} {'p':>8s}")
    rows = []
    for stat in ["eta2_subclass", "eta2_cluster", "within_frac", "within_sd"]:
        obs = float(sfD[stat].mean())
        nulls = np.empty(N_NULL)
        for i in range(N_NULL):
            vals = [pools[k][stat].sample(1, random_state=int(RNG.integers(1e9))).iloc[0]
                    for k in sfD["key"]]
            nulls[i] = float(np.mean(vals))
        mu, sd = nulls.mean(), nulls.std(ddof=1)
        z = (obs - mu) / sd if sd > 0 else np.nan
        pv = float(2 * min((nulls <= obs).mean(), (nulls >= obs).mean()))
        print(f"  {stat:22s} {obs:9.4f} {mu:12.4f} +- {sd:6.4f} {z:+7.2f} {pv:8.4f}")
        rows.append({"statistic": stat, "sfari": obs, "null_mean": mu,
                     "null_sd": sd, "z": z, "p": pv, "n_sfari": len(sfD)})
    pd.DataFrame(rows).to_csv(OUT / "AO_asd_atlas_summary.csv", index=False)

    # AUC of SFARI membership against cluster eta2, matched set as comparator
    y = np.r_[np.ones(len(sfD)), np.zeros(len(bgD))]
    x = np.r_[sfD["eta2_cluster"].to_numpy(), bgD["eta2_cluster"].to_numpy()]
    u = stats.mannwhitneyu(x[y == 1], x[y == 0], alternative="two-sided")
    auc = u.statistic / (len(sfD) * len(bgD))
    print(f"\n  AUC(SFARI vs background) on cluster eta2 = {auc:.3f}  "
          f"p = {u.pvalue:.3g}")
    print("  0.5 = ASD genes are no more cell-type-specific than matched genes")

    # which subclass expresses SFARI genes most, relative to background
    print("\n" + "=" * 92)
    print("SUBCLASS PROFILE — mean z-scored expression, SFARI minus matched background")
    print("=" * 92)
    prof = []
    for sc in sorted(subc.unique()):
        m = (subc == sc).to_numpy()
        if m.sum() < 100:
            continue
        def sc_mean(gs):
            return float(np.mean([cache[g][m].mean() - cache[g].mean()
                                  for g in gs]))
        prof.append({"subclass": sc, "n_cells": int(m.sum()),
                     "sfari": sc_mean(list(sfD.gene)),
                     "background": sc_mean(list(bgD.gene))})
    P = pd.DataFrame(prof)
    P["delta"] = P["sfari"] - P["background"]
    P = P.sort_values("delta", ascending=False)
    P.to_csv(OUT / "AP_asd_subclass_profile.csv", index=False)
    print(P.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n[write] {OUT/'AO_asd_atlas_summary.csv'}, "
          f"{OUT/'AP_asd_subclass_profile.csv'}")


if __name__ == "__main__":
    main()
