"""Within-cluster spread of an LTP gene index in the Allen SmartSeq atlas.

Four tests in Allen SmartSeq atlas (Tasic et al. 2018) glutamatergic cells,
leaf clusters >= 20 cells:

  1 eta² by subclass / by cluster       low  = not cell-type identity
  2 within-cluster spread vs an
    expression-matched null             high = biological variation
  3 orthogonality to COMPOSITION        Spearman r with the per-cell top-50
                                        gene share (library-skew proxy)
  4 orthogonality to the IEG axis       Spearman r with the canonical IEG +
                                        Jun/AP-1 index, within cluster

The index is built composition-robustly, as a log-ratio against an
expression-matched reference set rather than against housekeeping genes:

    index = mean log-CPM(LTP genes) - mean log-CPM(matched reference genes)

Housekeeping normalisation is not used: housekeeping genes are the top-50
genes, so when composition shifts toward them an LTP/housekeeping ratio moves
twice, amplifying the confound rather than cancelling it.

Because the ionotropic receptors (Gria*, Grin*) are both core LTP machinery and
ion channels, the set is tested with and without them.

Usage:
  uv run python analysis/allen_atlas/scripts/33_ltp_index_spread.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)

import re

import h5py
import numpy as np
import pandas as pd
from scipy import stats

from importlib import import_module  # noqa: E402
PANELS = import_module("11_atlas_extract_panels").PANELS

ROOT = paths.REPO
H5 = ROOT / "Gene_Expression" / "Data" / "expression_matrix.hdf5"
META = ROOT / "Gene_Expression" / "Data" / "metadata.csv"
OUT = paths.results(__file__)
MIN_CLUSTER_N = 20
MIN_DETECT = 0.15
N_NULL = 400
N_CONTROL = 4000
RNG = np.random.default_rng(0)

# LTP mechanism genes -- excludes immediate-early genes, since IEGs are
# transient. Grouped by function.
LTP = {
    "ampa_trafficking": ["Gria1", "Gria2", "Gria3", "Grip1", "Pick1", "Nsf",
                         "Cacng2", "Cacng3", "Cacng8", "Shisa9", "Shisa6",
                         "Prrt1", "Nptx1", "Nptx2", "Nptxr"],
    "nmda": ["Grin1", "Grin2a", "Grin2b", "Grin2d"],
    "kinases_phosphatases": ["Camk2a", "Camk2b", "Camk2g", "Camk4", "Prkca",
                             "Prkcb", "Prkcg", "Ppp3ca", "Ppp3cb", "Ppp1ca",
                             "Ppp1r9a", "Ppp1r9b", "Nrgn", "Calm1", "Calm2"],
    "spine_structural": ["Dlg4", "Shank1", "Shank2", "Shank3", "Syngap1",
                         "Cfl1", "Limk1", "Rac1", "Cdc42", "Pak1", "Pak3",
                         "Dbn1", "Cttn", "Arpc2", "Arpc3"],
    "late_ltp_translation": ["Mtor", "Rps6kb1", "Eif4e", "Eif4ebp1", "Eif4ebp2",
                             "Eef2k", "Crtc1", "Sik1", "Sik2", "Creb1"],
    "bdnf_adhesion": ["Bdnf", "Ntrk2", "Nlgn1", "Nrxn1", "Lrrtm1", "Lrrtm2",
                      "Cdh2", "Nos1"],
}
RECEPTORS = r"^(Gria|Grin|Cacng)"


def main() -> None:
    ltp_all = sorted({g for v in LTP.values() for g in v})
    ltp_norec = [g for g in ltp_all if not re.match(RECEPTORS, g)]
    ieg = PANELS["canonical_ieg"] + PANELS["jun_ap1"]
    wanted = sorted(set(ltp_all) | set(ieg))

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
        subc = md["subclass_label"].to_numpy()[idx]
        L = lib[idx]
        print(f"[atlas] {len(idx)} glutamatergic cells, {ng} clusters")

        def lc(j):
            lo, hi = p[j], p[j + 1]
            v = np.zeros(nall)
            if hi > lo:
                v[np.asarray(grp["i"][lo:hi], dtype=np.int64)] = \
                    np.asarray(grp["x"][lo:hi], dtype=np.float64)
            return np.log1p(v[idx] / L * 1e6)

        def wc(v):
            m = np.bincount(codes, weights=v, minlength=ng) / cnt
            return v - m[codes]

        # control pool for expression matching + composition proxy
        pool = [g for g in genes if g not in set(wanted)]
        ctrl = list(RNG.choice(pool, size=min(N_CONTROL, len(pool)), replace=False))
        cache = {}
        for g in wanted + ctrl:
            if g in gi:
                cache[g] = lc(gi[g])
        detect = {g: float((v > 0).mean()) for g, v in cache.items()}
        meanx = {g: float(v.mean()) for g, v in cache.items()}

        # top-50 composition share per cell, computed over the control pool as a
        # proxy for library skew (full-matrix version is prohibitive here)
        Cm = np.column_stack([np.expm1(cache[g]) for g in ctrl if g in cache])
        top50 = np.sort(Cm, axis=1)[:, ::-1][:, :50].sum(1) / Cm.sum(1)

    def eta2(v, g):
        gm = v.mean(); ssb = 0.0
        for u in pd.unique(g):
            m = g == u
            if m.sum() > 1:
                ssb += m.sum() * (v[m].mean() - gm) ** 2
        return float(ssb / ((v - gm) ** 2).sum())

    def matched_reference(target_genes, exclude=()):
        """Nearest control gene by mean expression, excluding the targets.

        The exclusion matters: without it, calling this on control genes returns
        those same genes, the log-ratio is identically zero, and the null has zero
        variance.
        """
        bad = set(target_genes) | set(exclude)
        cg = [g for g in ctrl if g in cache and detect[g] >= MIN_DETECT
              and g not in bad]
        cm = np.array([meanx[g] for g in cg])
        ref = []
        for g in target_genes:
            d = np.abs(cm - meanx[g])
            for k in np.argsort(d):
                if cg[k] not in ref:
                    ref.append(cg[k]); break
        return ref

    def build(target_genes, label):
        present = [g for g in target_genes if g in cache and detect[g] >= MIN_DETECT]
        if len(present) < 8:
            print(f"  {label}: only {len(present)} genes present, skipping")
            return None
        Z = np.column_stack([cache[g] for g in present])
        raw = Z.mean(1)
        ref = matched_reference(present)
        R = np.column_stack([cache[g] for g in ref])
        ratio = raw - R.mean(1)             # log-ratio to matched reference
        return {"label": label, "n_genes": len(present), "genes": present,
                "raw": raw, "ratio": ratio, "ref": ref}

    ieg_raw = build(ieg, "IEG reference axis")
    sets = [("LTP all", ltp_all), ("LTP no receptors", ltp_norec)] + \
           [(f"  sub: {k}", v) for k, v in LTP.items()]

    rows = []
    print("\n" + "=" * 96)
    print("LTP INDEX — within-cluster spread, type-orthogonality, composition-independence")
    print("=" * 96)
    hdr = (f"  {'index':22s} {'g':>3s} {'eta2_sub':>9s} {'eta2_clu':>9s} "
           f"{'sd_within':>10s} {'null_sd':>9s} {'z':>7s} {'r_top50':>8s} {'r_IEG':>7s}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for label, gs in sets:
        b = build(gs, label)
        if b is None:
            continue
        v = b["ratio"]                       # composition-robust form
        r = wc(v)
        sd_w = float(r.std(ddof=1))
        # expression-matched null: random sets of the same size, same deciles
        nulls = []
        avail = [g for g in ctrl if g in cache and detect[g] >= MIN_DETECT]
        for _ in range(N_NULL):
            alt = list(RNG.choice(avail, size=len(b["genes"]), replace=False))
            Za = np.column_stack([cache[g] for g in alt])
            refa = matched_reference(alt)          # excludes alt by construction
            Ra = np.column_stack([cache[g] for g in refa])
            nulls.append(float(wc(Za.mean(1) - Ra.mean(1)).std(ddof=1)))
        nulls = np.asarray(nulls)
        z = (sd_w - nulls.mean()) / nulls.std(ddof=1)
        r_top = stats.spearmanr(wc(v), top50)[0]
        r_ieg = (stats.spearmanr(wc(v), wc(ieg_raw["ratio"]))[0]
                 if ieg_raw else np.nan)
        print(f"  {label:22s} {b['n_genes']:3d} {eta2(v, subc):9.3f} "
              f"{eta2(v, cats.to_numpy()):9.3f} {sd_w:10.4f} "
              f"{nulls.mean():9.4f} {z:+7.1f} {r_top:+8.3f} {r_ieg:+7.3f}")
        rows.append({"index": label, "n_genes": b["n_genes"],
                     "eta2_subclass": eta2(v, subc),
                     "eta2_cluster": eta2(v, cats.to_numpy()),
                     "sd_within": sd_w, "null_sd": float(nulls.mean()),
                     "spread_z": z, "r_top50_share": r_top, "r_IEG_axis": r_ieg})

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "AF_ltp_index_spread.csv", index=False)
    print("\n  CRITERIA:       spread z > ~3 (real variation), eta2_cluster low")
    print("                  |r_top50| small (not composition), |r_IEG| < ~0.5")
    print("                  (distinct from the IEG axis)")
    print(f"\n[write] {OUT/'AF_ltp_index_spread.csv'}")


if __name__ == "__main__":
    main()
