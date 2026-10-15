"""Gene-set enrichment along the within-cluster activity-state axis (Allen SmartSeq atlas).

The state axis is the jun_ap1 panel composite (panels from
11_atlas_extract_panels.py) residualised within leaf cluster, in glutamatergic
cells of the Allen SmartSeq atlas (Tasic et al. 2018), clusters >= 30 cells.
Every gene detected in >= 10% of cells is residualised the same way and
correlated (Pearson) with the axis; curated gene sets are then compared with the
non-panel background by two-sided Mann-Whitney U (median delta, AUC).

Curated gene sets:
  ltp_core, ampa_traffic, structural   plasticity machinery
  stress_hsp, hypoxia, apoptosis       dissociation / slice-injury response
  translation, cell_cycle              generic transcriptional load
  adhesion, axon_guidance              developmental/laminar gradient

Gene sets are listed explicitly in GENE_SETS rather than pulled from an ontology.

Usage:  uv run python analysis/allen_atlas/scripts/15_state_axis_function.py
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

PANELS = import_module("11_atlas_extract_panels").PANELS

ROOT = paths.REPO
H5 = ROOT / "Gene_Expression" / "Data" / "expression_matrix.hdf5"
META = ROOT / "Gene_Expression" / "Data" / "metadata.csv"
OUT = paths.results(__file__)
MIN_CLUSTER_N = 30
MIN_DETECT = 0.10
STATE_PANEL = "jun_ap1"          # the axis with the lowest type-dependence
SECOND_PANEL = "canonical_ieg"

GENE_SETS: dict[str, list[str]] = {
    "ltp_core": ["Camk2a", "Camk2b", "Camk2g", "Camk4", "Grin1", "Grin2a",
                 "Grin2b", "Gria1", "Gria2", "Gria3", "Dlg4", "Shank1", "Shank2",
                 "Shank3", "Syngap1", "Ppp3ca", "Ppp3cb", "Prkcg", "Prkca",
                 "Ppp1r9a", "Ppp1r9b", "Nsf", "Rab3a", "Ntrk2", "Grip1"],
    "ampa_traffic": ["Gria1", "Gria2", "Nsf", "Grip1", "Pick1", "Stx3", "Stx4a",
                     "Rab8a", "Rab11a", "Cacng2", "Cacng3", "Cacng8", "Nptx1",
                     "Nptx2", "Nptxr", "Shisa6", "Shisa9", "Prrt1"],
    "structural": ["Cfl1", "Limk1", "Rac1", "Cdc42", "Rhoa", "Rock2", "Wasf1",
                   "Arpc2", "Arpc3", "Pak1", "Pak3", "Ephb2", "Drebrin", "Dbn1",
                   "Cttn", "Fmn2", "Ktn1"],
    "stress_hsp": ["Hspa1a", "Hspa1b", "Hspa5", "Hspa8", "Hspb1", "Hsp90aa1",
                   "Hsp90ab1", "Dnajb1", "Dnajb5", "Dnajb9", "Hsph1", "Bag3",
                   "Ahsa1", "Chordc1"],
    "hypoxia": ["Hif1a", "Vegfa", "Slc2a1", "Ldha", "Pgk1", "Bnip3", "Ddit4",
                "Egln3", "Ankrd37", "Adm", "P4ha1", "Aldoa"],
    "apoptosis": ["Casp3", "Casp8", "Casp9", "Bax", "Bad", "Bcl2", "Bcl2l11",
                  "Trp53", "Cdkn1a", "Gadd45a", "Ddit3", "Atf4"],
    "translation": ["Eif4e", "Eif4ebp1", "Eif4ebp2", "Rps6", "Rps6kb1", "Mtor",
                    "Rptor", "Eef1a1", "Eef2", "Eef2k", "Rpl13a", "Rps3",
                    "Rps15", "Rpl4", "Rpl7"],
    "cell_cycle": ["Ccnd1", "Ccnd2", "Ccne1", "Cdk1", "Cdk2", "Cdk4", "Cdk6",
                   "Mki67", "Pcna", "Top2a", "Mcm2", "Mcm3"],
    "adhesion": ["Cdh8", "Cdh10", "Cdh12", "Cdh13", "Cdh18", "Cntn5", "Cntnap2",
                 "Cntnap5a", "Pcdh7", "Pcdh9", "Pcdh11x", "Nrxn1", "Nrxn3",
                 "Nlgn1", "Nlgn2", "Lrrtm3", "Lrrtm4", "Sorcs3", "Ptprd",
                 "Ptprt", "Kirrel3"],
    "axon_guidance": ["Unc5c", "Unc5d", "Robo1", "Robo2", "Slit1", "Slit2",
                      "Epha4", "Epha6", "Ephb1", "Sema3a", "Sema5a", "Sema6d",
                      "Dcc", "Ntng1", "Plxna4", "Nrp1", "Nrp2"],
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META, low_memory=False).set_index("sample_name")
    meta = meta[["class_label", "subclass_label", "cluster_label"]]

    with h5py.File(H5, "r") as f:
        genes = np.array([g.decode() for g in f["gene_names"][:]])
        cells = np.array([c.decode() for c in f["sample_names"][:]])
        libsize = np.asarray(f["data/total_exon_counts"][:], dtype=np.float64)
        grp = f["data/exon"]
        p = np.asarray(grp["p"][:], dtype=np.int64)
        n_cells_all = int(grp["dims"][0])

        md = meta.reindex(cells)
        # .to_numpy() can hand back a read-only view, so copy before &=
        keep = np.array(md["class_label"].astype(str)
                        .str.contains("Glutamatergic", na=False), dtype=bool)
        clus_all = md["cluster_label"].to_numpy()
        vc = pd.Series(clus_all[keep]).value_counts()
        big = set(vc[vc >= MIN_CLUSTER_N].index)
        keep &= np.array([c in big for c in clus_all])
        idx = np.flatnonzero(keep)
        clus = pd.Series(clus_all[idx]).astype("category")
        codes = clus.cat.codes.to_numpy()
        n_grp = len(clus.cat.categories)
        lib = libsize[idx]
        n = len(idx)
        print(f"[pop] {n} glutamatergic cells, {n_grp} clusters "
              f"(>= {MIN_CLUSTER_N} cells)")

        gi = {g: i for i, g in enumerate(genes)}
        counts_grp = np.bincount(codes, minlength=n_grp).astype(np.float64)

        def within_center(v: np.ndarray) -> np.ndarray:
            m = np.bincount(codes, weights=v, minlength=n_grp) / counts_grp
            return v - m[codes]

        def gene_vec(g: str) -> np.ndarray:
            j = gi[g]
            lo, hi = p[j], p[j + 1]
            v = np.zeros(n_cells_all, dtype=np.float64)
            if hi > lo:
                v[np.asarray(grp["i"][lo:hi], dtype=np.int64)] = \
                    np.asarray(grp["x"][lo:hi], dtype=np.float64)
            return np.log1p(v[idx] / lib * 1e6)

        # ---- state axis: within-cluster residual of the panel composite ----
        def panel_resid(panel_name: str) -> np.ndarray:
            cols = [g for g in PANELS[panel_name] if g in gi]
            Z = np.column_stack([gene_vec(g) for g in cols])
            Z = (Z - Z.mean(0)) / np.where(Z.std(0, ddof=1) > 0, Z.std(0, ddof=1), np.nan)
            return within_center(np.nanmean(Z, axis=1))

        state = panel_resid(STATE_PANEL)
        state2 = panel_resid(SECOND_PANEL)
        s_c = state - state.mean()
        s_n = np.sqrt((s_c ** 2).sum())
        print(f"[state] axis = {STATE_PANEL} within-cluster residual, sd={state.std(ddof=1):.4f}")

        # ---- genome-wide within-cluster correlation ------------------------
        print("[scan] correlating every gene's within-cluster residual ...")
        rho = np.full(len(genes), np.nan)
        det = np.full(len(genes), np.nan)
        for j in range(len(genes)):
            lo, hi = p[j], p[j + 1]
            if hi - lo < 10:
                continue
            v = np.zeros(n_cells_all, dtype=np.float64)
            v[np.asarray(grp["i"][lo:hi], dtype=np.int64)] = \
                np.asarray(grp["x"][lo:hi], dtype=np.float64)
            v = v[idx]
            d = float((v > 0).mean())
            det[j] = d
            if d < MIN_DETECT:
                continue
            r = within_center(np.log1p(v / lib * 1e6))
            r -= r.mean()
            den = np.sqrt((r ** 2).sum()) * s_n
            if den > 0:
                rho[j] = float((r * s_c).sum() / den)
            if j % 8000 == 0:
                print(f"    {j}/{len(genes)}", flush=True)

    res = pd.DataFrame({"gene": genes, "r_within": rho, "detect_frac": det}).dropna(
        subset=["r_within"])
    res["rank_pct"] = 100.0 * res["r_within"].rank(pct=True)
    res.to_csv(OUT / "O_state_axis_gene_correlations.csv", index=False)
    print(f"\n[scan] {len(res)} genes tested (detected in >= {MIN_DETECT:.0%} of cells)")

    panel_genes = {g for pl in PANELS.values() for g in pl}
    bg = res[~res["gene"].isin(panel_genes)]

    rows = []
    for name, gs in GENE_SETS.items():
        sub = bg[bg["gene"].isin(gs)]
        if len(sub) < 5:
            rows.append({"set": name, "n_found": len(sub), "mean_r": np.nan,
                         "median_r": np.nan, "mwu_p": np.nan, "delta": np.nan})
            continue
        u, pv = stats.mannwhitneyu(sub["r_within"], bg["r_within"],
                                   alternative="two-sided")
        rows.append({
            "set": name, "n_found": len(sub),
            "mean_r": sub["r_within"].mean(), "median_r": sub["r_within"].median(),
            "delta": sub["r_within"].median() - bg["r_within"].median(),
            "auc": u / (len(sub) * len(bg)), "mwu_p": pv,
        })
    E = pd.DataFrame(rows).sort_values("delta", ascending=False)
    E.to_csv(OUT / "O_state_axis_setenrichment.csv", index=False)

    print("\n" + "=" * 92)
    print(f"WHAT CO-VARIES WITH THE {STATE_PANEL} WITHIN-CLUSTER STATE AXIS?")
    print(f"(background median r = {bg['r_within'].median():+.4f}, "
          f"n = {len(bg):,} genes)")
    print("=" * 92)
    print(f"  {'gene set':16s} {'n':>4s} {'median r':>10s} {'Δ vs bg':>9s} "
          f"{'AUC':>6s} {'p':>11s}")
    for _, x in E.iterrows():
        star = "" if not np.isfinite(x.mwu_p) else (
            "***" if x.mwu_p < 1e-4 else "**" if x.mwu_p < 0.01 else
            "*" if x.mwu_p < 0.05 else "")
        print(f"  {x['set']:16s} {int(x.n_found):4d} {x.median_r:+10.4f} "
              f"{x.delta:+9.4f} {x.auc:6.3f} {x.mwu_p:11.2g} {star}")

    print("\n--- top 25 genes on the state axis (excluding panel genes) ---")
    print(bg.nlargest(25, "r_within")[["gene", "r_within", "detect_frac"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    r2, _ = stats.spearmanr(state, state2)
    print(f"\nsanity: {STATE_PANEL} vs {SECOND_PANEL} residual axes rho = {r2:+.3f}")
    print(f"[write] {OUT/'O_state_axis_gene_correlations.csv'}")
    print(f"[write] {OUT/'O_state_axis_setenrichment.csv'}")


if __name__ == "__main__":
    main()
