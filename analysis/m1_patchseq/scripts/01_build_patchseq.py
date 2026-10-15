"""Assemble the M1 Patch-seq dataset (Scala et al. 2021, GSE163764) into one AnnData.

counts  = exon + intron reads (as in Scala et al. 2021)
obs     = curated metadata + intrinsic ephys features
QC      = drop 'low quality' RNA family, explicit exclusion reasons, <1000 genes
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from common import PATCHSEQ_DIR, EPHYS_CSV  # noqa: E402
DATA = paths.data(__file__)

exon = pd.read_csv(PATCHSEQ_DIR / "GSE163764_m1_patchseq_exon_counts.csv", index_col=0)
intron = pd.read_csv(PATCHSEQ_DIR / "GSE163764_m1_patchseq_intron_counts.csv", index_col=0)
intron = intron.reindex(index=exon.index, columns=exon.columns).fillna(0)
counts = (exon + intron).T                        # cells x genes

meta = pd.read_csv(PATCHSEQ_DIR / "m1_patchseq_meta_data.csv", sep="\t").set_index("Cell")
ephys = pd.read_csv(EPHYS_CSV).set_index("cell id")
ephys.columns = ["ephys: " + c for c in ephys.columns]

cells = counts.index.intersection(meta.index)
obs = meta.loc[cells].join(ephys, how="left")
obs.columns = [str(c).replace("/", "_") for c in obs.columns]

X = sp.csr_matrix(counts.loc[cells].values.astype(np.float32))
a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=counts.columns))
a.layers["counts"] = a.X.copy()
a.obs["n_genes"] = np.asarray((a.X > 0).sum(1)).ravel()
a.obs["n_counts"] = np.asarray(a.X.sum(1)).ravel()

# QC
keep = (a.obs["RNA family"] != "low quality") & a.obs["Exclusion reasons"].isna() \
       & (a.obs["n_genes"] >= 1000) & a.obs["RNA type"].notna()
print(f"cells: {a.n_obs} -> {keep.sum()} after QC")
a = a[keep].copy()

fam = a.obs["RNA family"].astype(str)
a.obs["class"] = np.where(fam.isin(["IT", "ET", "CT", "NP"]), "Glutamatergic", "GABAergic")
a.obs["family"] = fam
a.obs["ttype"] = a.obs["RNA type"].astype(str)
a.obs["batch"] = a.obs["Sequencing batch"].astype(str)
a.obs["mouse"] = a.obs["Mouse"].astype(str)
a.obs["hold_time"] = a.obs["Hold Time (min)"].astype(float)
a.obs["has_ephys"] = a.obs["ephys: Rheobase (pA)"].notna()
# string-typed object columns for h5ad
for c in a.obs.columns:
    if a.obs[c].dtype == object:
        a.obs[c] = a.obs[c].astype(str)

# log-CPM
lib = np.asarray(a.layers["counts"].sum(1)).ravel()
a.X = sp.csr_matrix(np.log1p(a.layers["counts"].multiply(1e6 / lib[:, None]).toarray()).astype(np.float32))
DATA.mkdir(exist_ok=True)
a.write_h5ad(DATA / "patchseq_m1.h5ad", compression="gzip")
print(a.obs["family"].value_counts().to_string())
print("with ephys:", a.obs["has_ephys"].sum(), "| mice:", a.obs["mouse"].nunique(),
      "| t-types:", a.obs["ttype"].nunique())
