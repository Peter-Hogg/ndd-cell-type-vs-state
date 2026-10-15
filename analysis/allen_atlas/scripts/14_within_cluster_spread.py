"""Within-cluster spread of activity-gene panel scores in the Allen SmartSeq atlas.

Three checks on whether activity-panel variance is a population axis beyond
cell-type identity:

  1. MAGNITUDE   most of the variance survives conditioning on cluster identity,
                 even at the finest (leaf t-type) level
  2. NOT NOISE   the within-cluster spread exceeds an EXPRESSION-MATCHED NULL:
                 random genes matched on mean-expression decile, which carry the
                 same Poisson/dropout noise
  3. ORTHOGONAL  the within-cluster residual is not explained by sequencing
                 depth, and the three programmes' residuals are only weakly
                 correlated with each other -- separate axes, not one

Run on the Allen SmartSeq atlas (Tasic et al. 2018) glutamatergic cells.

Usage:  uv run python analysis/allen_atlas/scripts/14_within_cluster_spread.py
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

from importlib import import_module  # noqa: E402

# Module name starts with a digit, so it cannot be imported with `import`.
# Reusing the panel definitions rather than re-declaring keeps the two scripts
# from drifting apart.
PANELS = import_module("11_atlas_extract_panels").PANELS

ROOT = paths.REPO
H5 = ROOT / "Gene_Expression" / "Data" / "expression_matrix.hdf5"
META = ROOT / "Gene_Expression" / "Data" / "metadata.csv"
OUT = paths.results(__file__)

N_CONTROL = 4000        # random genes forming the matching pool
N_NULL = 300            # expression-matched null composites per panel
MIN_CLUSTER_N = 30      # clusters smaller than this are unstable
RNG = np.random.default_rng(0)


def load(wanted: list[str], n_control: int):
    with h5py.File(H5, "r") as f:
        genes = [g.decode() for g in f["gene_names"][:]]
        cells = np.array([c.decode() for c in f["sample_names"][:]])
        libsize = np.asarray(f["data/total_exon_counts"][:], dtype=np.float64)
        gi = {g: i for i, g in enumerate(genes)}
        found = [g for g in wanted if g in gi]

        pool = [g for g in genes if g not in set(found)]
        ctrl = list(RNG.choice(pool, size=min(n_control, len(pool)), replace=False))
        cols = found + ctrl

        grp = f["data/exon"]
        n_cells = int(grp["dims"][0])
        p = np.asarray(grp["p"][:], dtype=np.int64)
        X = np.zeros((n_cells, len(cols)), dtype=np.float32)
        for k, g in enumerate(cols):
            j = gi[g]
            lo, hi = p[j], p[j + 1]
            if hi > lo:
                X[np.asarray(grp["i"][lo:hi], dtype=np.int64), k] = \
                    np.asarray(grp["x"][lo:hi], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_cpm = np.log1p(X / libsize[:, None] * 1e6)
    return pd.DataFrame(log_cpm, index=cells, columns=cols), found, ctrl, libsize


def eta2(vals: np.ndarray, groups: np.ndarray) -> float:
    ok = np.isfinite(vals)
    v, g = vals[ok], groups[ok]
    gm = v.mean()
    ssb = 0.0
    for u in pd.unique(g):
        m = g == u
        if m.sum() > 1:
            ssb += m.sum() * (v[m].mean() - gm) ** 2
    return float(ssb / ((v - gm) ** 2).sum())


def within_resid(vals: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Score minus its cluster mean -- the part type identity cannot explain."""
    s = pd.Series(vals)
    return (s - s.groupby(pd.Series(groups).values).transform("mean")).to_numpy()


def composite(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    sub = df[cols]
    z = (sub - sub.mean()) / sub.std(ddof=1).replace(0, np.nan)
    return z.mean(axis=1).to_numpy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panels = PANELS
    wanted = sorted({g for p in panels.values() for g in p})
    df, found, ctrl, libsize = load(wanted, N_CONTROL)
    print(f"[load] {df.shape[0]} cells | {len(found)} panel genes | {len(ctrl)} controls")

    meta = pd.read_csv(META, low_memory=False).set_index("sample_name")
    meta = meta[["class_label", "subclass_label", "cluster_label"]]
    d = df.join(meta, how="inner")
    d = d[d["class_label"].astype(str).str.contains("Glutamatergic", na=False)]
    libs = pd.Series(libsize, index=df.index).reindex(d.index)

    vc = d["cluster_label"].value_counts()
    keep = vc[vc >= MIN_CLUSTER_N].index
    d = d[d["cluster_label"].isin(keep)]
    libs = libs.reindex(d.index)
    print(f"[pop] {len(d)} glutamatergic cells in {len(keep)} clusters "
          f"(>= {MIN_CLUSTER_N} cells each)\n")

    clus = d["cluster_label"].to_numpy()
    subc = d["subclass_label"].to_numpy()
    cls = d["class_label"].to_numpy()
    mean_expr = d[found + ctrl].mean()

    rows, resid_store = [], {}
    for name, panel in panels.items():
        cols = [g for g in panel if g in d.columns]
        score = composite(d, cols)

        e_cls = eta2(score, cls)
        e_sub = eta2(score, subc)
        e_clu = eta2(score, clus)
        r = within_resid(score, clus)
        resid_store[name] = r
        sd_within = float(np.nanstd(r, ddof=1))
        sd_total = float(np.nanstd(score, ddof=1))

        # --- expression-matched null -------------------------------------
        # For each panel gene, draw a control gene from the same mean-expression
        # decile, build a composite of identical size, and take its within-
        # cluster SD. Repeating gives the null a same-sized, same-expression
        # composite would show.
        bins = pd.qcut(mean_expr, 10, labels=False, duplicates="drop")
        ctrl_by_bin = {b: [g for g in ctrl if bins.get(g) == b] for b in set(bins.dropna())}
        null_sd = []
        for _ in range(N_NULL):
            pick = []
            for g in cols:
                b = bins.get(g)
                cand = ctrl_by_bin.get(b, [])
                if cand:
                    pick.append(cand[RNG.integers(len(cand))])
            if len(pick) < 3:
                continue
            ns = composite(d, pick)
            null_sd.append(np.nanstd(within_resid(ns, clus), ddof=1))
        null_sd = np.asarray(null_sd, dtype=float)
        z = (sd_within - null_sd.mean()) / null_sd.std(ddof=1) if len(null_sd) > 2 else np.nan
        pct = 100.0 * (null_sd < sd_within).mean() if len(null_sd) else np.nan

        # --- orthogonality to depth --------------------------------------
        ok = np.isfinite(r) & libs.notna().to_numpy()
        rho_depth, p_depth = stats.spearmanr(r[ok], np.log10(libs.to_numpy()[ok]))

        rows.append({
            "panel": name, "n_genes": len(cols),
            "eta2_class": e_cls, "eta2_subclass": e_sub, "eta2_cluster": e_clu,
            "frac_within_cluster": 1 - e_clu,
            "sd_total": sd_total, "sd_within_cluster": sd_within,
            "null_sd_mean": float(null_sd.mean()) if len(null_sd) else np.nan,
            "null_sd_sd": float(null_sd.std(ddof=1)) if len(null_sd) > 2 else np.nan,
            "z_vs_null": z, "pct_above_null": pct,
            "rho_vs_log_depth": rho_depth, "p_vs_depth": p_depth,
        })

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "N_within_cluster_spread.csv", index=False)

    print("=" * 88)
    print("1. HOW MUCH VARIANCE SURVIVES CONDITIONING ON CELL TYPE?")
    print("=" * 88)
    print(f"  {'panel':18s} {'eta2 class':>11s} {'eta2 subclass':>14s} "
          f"{'eta2 cluster':>13s} {'% WITHIN cluster':>17s}")
    for _, x in R.iterrows():
        print(f"  {x.panel:18s} {x.eta2_class:11.3f} {x.eta2_subclass:14.3f} "
              f"{x.eta2_cluster:13.3f} {100*x.frac_within_cluster:16.1f}%")

    print("\n" + "=" * 88)
    print("2. IS THE WITHIN-CLUSTER SPREAD BIGGER THAN EXPRESSION-MATCHED NOISE?")
    print("=" * 88)
    print(f"  {'panel':18s} {'SD within':>10s} {'null SD':>16s} {'z':>8s} {'pct':>7s}")
    for _, x in R.iterrows():
        print(f"  {x.panel:18s} {x.sd_within_cluster:10.4f} "
              f"{x.null_sd_mean:9.4f}±{x.null_sd_sd:.4f} {x.z_vs_null:8.1f} "
              f"{x.pct_above_null:6.1f}%")
    print("\n  z >> 0 means the spread is biological, not Poisson/dropout noise.")

    print("\n" + "=" * 88)
    print("3. ORTHOGONALITY")
    print("=" * 88)
    print("  within-cluster residual vs sequencing depth:")
    for _, x in R.iterrows():
        flag = "" if abs(x.rho_vs_log_depth) < 0.1 else "  <- depth-contaminated"
        print(f"    {x.panel:18s} rho = {x.rho_vs_log_depth:+.3f} "
              f"(p = {x.p_vs_depth:.2g}){flag}")
    print("\n  between programmes, AFTER removing cluster identity:")
    names = list(resid_store)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = resid_store[names[i]], resid_store[names[j]]
            ok = np.isfinite(a) & np.isfinite(b)
            rr, _ = stats.spearmanr(a[ok], b[ok])
            print(f"    {names[i]:18s} vs {names[j]:18s} rho = {rr:+.3f}")
    print(f"\n[write] {OUT/'N_within_cluster_spread.csv'}")


if __name__ == "__main__":
    main()
