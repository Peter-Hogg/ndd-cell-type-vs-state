"""Extract activity-gene panels from the Allen SmartSeq atlas (Tasic et al. 2018).

Panels:
  canonical_ieg     the standard activity markers
  jun_ap1           the Jun/AP-1 axis
  delayed_effector  Nptx/Homer/Bdnf/Syt4 delayed effector genes

Panel scores are the mean z-scored log-CPM over all cells (~73k).

Reads only the gene columns needed (data/exon is cells x genes CSC, so column
slicing is cheap) and takes library sizes from data/total_exon_counts rather than
summing 703M nonzeros.

Usage:  uv run python analysis/allen_atlas/scripts/11_atlas_extract_panels.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)


import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

ROOT = paths.REPO
H5 = ROOT / "Gene_Expression" / "Data" / "expression_matrix.hdf5"
META = ROOT / "Gene_Expression" / "Data" / "metadata.csv"
TSNE = ROOT / "Gene_Expression" / "Data" / "tsne.csv"
OUT = paths.results(__file__)

PANELS = {
    "canonical_ieg": ["Fos", "Fosb", "Arc", "Npas4", "Egr1", "Egr2", "Egr3",
                      "Egr4", "Ier2", "Btg2", "Dusp1"],
    "jun_ap1": ["Jun", "Junb", "Jund", "Gadd45g", "Gadd45b", "Atf3", "Fosl2"],
    "delayed_effector": ["Nptx1", "Nptx2", "Nptxr", "Homer1", "Bdnf", "Syt4",
                         "Rgs4", "Crem"],
}
MARKERS = ["Slc17a7", "Gad1", "Gad2", "Fezf2", "Bcl11b", "Snap25"]


def load_columns(h5_path: pathlib.Path, wanted: list[str]):
    """Return (cells x len(found)) dense counts, cell names, library sizes, found."""
    with h5py.File(h5_path, "r") as f:
        genes = [g.decode() for g in f["gene_names"][:]]
        cells = np.array([c.decode() for c in f["sample_names"][:]])
        libsize = np.asarray(f["data/total_exon_counts"][:], dtype=np.float64)

        gi = {g: i for i, g in enumerate(genes)}
        found = [g for g in wanted if g in gi]
        missing = [g for g in wanted if g not in gi]
        if missing:
            print(f"[extract] absent from atlas: {missing}")

        grp = f["data/exon"]
        n_cells, n_genes = (int(x) for x in grp["dims"][:])
        p = np.asarray(grp["p"][:], dtype=np.int64)
        print(f"[extract] matrix {n_cells} cells x {n_genes} genes; "
              f"pulling {len(found)} gene columns")

        X = np.zeros((n_cells, len(found)), dtype=np.float32)
        for k, g in enumerate(found):
            j = gi[g]
            lo, hi = p[j], p[j + 1]
            if hi > lo:
                rows = np.asarray(grp["i"][lo:hi], dtype=np.int64)
                vals = np.asarray(grp["x"][lo:hi], dtype=np.float64)
                X[rows, k] = vals
    return X, cells, libsize, found


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = sorted({g for p in PANELS.values() for g in p} | set(MARKERS))
    X, cells, libsize, found = load_columns(H5, wanted)

    # counts -> CPM -> log1p, matching the Patch-seq pipeline. Never TPM.
    with np.errstate(invalid="ignore", divide="ignore"):
        cpm = X / libsize[:, None] * 1e6
    log_cpm = pd.DataFrame(np.log1p(cpm), index=cells, columns=found)
    print(f"[extract] log-CPM matrix: {log_cpm.shape}")

    meta = pd.read_csv(META, low_memory=False).set_index("sample_name")
    keep = ["class_label", "subclass_label", "region_label", "cluster_label",
            "cortical_layer_label"]
    meta = meta[[c for c in keep if c in meta.columns]]
    ts = pd.read_csv(TSNE).set_index("sample_name")

    df = log_cpm.join(meta, how="inner").join(ts, how="left")
    print(f"[extract] joined with metadata: {df.shape[0]} cells")
    print(f"[extract] class_label:\n{df['class_label'].value_counts().to_string()}")

    # panel scores: mean z-scored log-CPM, standardised over the analysed
    # population (all cells here; within-subclass analyses re-standardise)
    for name, panel in PANELS.items():
        cols = [g for g in panel if g in df.columns]
        sub = df[cols]
        z = (sub - sub.mean()) / sub.std(ddof=1).replace(0, np.nan)
        df[f"score_{name}"] = z.mean(axis=1)
        print(f"[score] {name:18s} {len(cols)}/{len(panel)} genes: {cols}")

    path = OUT / "M_atlas_panel_scores.parquet"
    df.to_parquet(path)
    print(f"\n[write] {path}  ({df.shape[0]} cells x {df.shape[1]} cols)")

    glut = df[df["class_label"].astype(str).str.contains("Glutamatergic", na=False)]
    print(f"\nglutamatergic cells: {len(glut)}")
    print(glut["subclass_label"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
