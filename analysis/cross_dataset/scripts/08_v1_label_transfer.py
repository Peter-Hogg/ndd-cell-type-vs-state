"""Label transfer of M1 Pvalb t-types to V1 Pvalb cells (outcome-blind; no V1 ephys is read here).

Transfer M1 Pvalb t-type labels to V1 (Gouwens et al. 2020) Pvalb cells with a 15-NN classifier on
30 PCs of z-scored log-CPM over the 2,000 most variable non-activity genes among M1 Pvalb cells.
Also computes the rapid-IEG score for every V1 cell (same procedure as M1).
Outputs: data/v1_interneurons.h5ad, results/v1_pvalb_label_transfer.csv
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from pyarrow import csv as pacsv
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from common import RAW, GENE_SETS, V1_GOUWENS  # noqa: E402
RESULTS = paths.results(__file__)
DATA = paths.data(__file__)

V1 = V1_GOUWENS
out_h5 = DATA / "v1_interneurons.h5ad"
if out_h5.exists():
    v1 = sc.read_h5ad(out_h5)
else:
    t = pacsv.read_csv(V1 / "20200513_Mouse_PatchSeq_Release_count.v2" / "20200513_Mouse_PatchSeq_Release_count.v2.csv").to_pandas()
    t = t.set_index(t.columns[0])
    meta = pd.read_csv(V1 / "metadata_20200711.csv").set_index("transcriptomics_sample_id")
    cells = t.columns.intersection(meta.index)
    X = sp.csr_matrix(t[cells].values.T.astype(np.float32))
    v1 = ad.AnnData(X=X, obs=meta.loc[cells].copy(), var=pd.DataFrame(index=pd.Index(t.index.astype(str).values)))
    v1.obs.index.name = None
    v1.obs = v1.obs.astype({c: str for c in v1.obs.columns if v1.obs[c].dtype == object})
    v1.layers["counts"] = v1.X.copy()
    lib = np.asarray(v1.X.sum(1)).ravel()
    v1.X = sp.csr_matrix(np.log1p(v1.layers["counts"].multiply(1e6 / lib[:, None]).toarray()).astype(np.float32))
    v1.obs["n_genes"] = np.asarray((v1.layers["counts"] > 0).sum(1)).ravel()
    v1 = v1[v1.obs["n_genes"] >= 1000].copy()
    sc.tl.score_genes(v1, [g for g in GENE_SETS["IEG_rapid"] if g in v1.var_names], score_name="IEG_rapid",
                      ctrl_size=100, n_bins=25, random_state=0)
    v1.write_h5ad(out_h5, compression="gzip")
print("V1 cells:", v1.n_obs, "| genes:", v1.n_vars)

m1 = sc.read_h5ad(paths.M1 / "data" / "patchseq_m1.h5ad")
m1 = m1[m1.obs["ttype"].astype(str).str.startswith("Pvalb")].copy()
m1.obs["ttype"] = m1.obs["ttype"].astype(str)
vc = m1.obs["ttype"].value_counts()
m1 = m1[m1.obs["ttype"].isin(vc[vc >= 15].index)].copy()
pv = v1[v1.obs["T-type Label"].astype(str).str.startswith("Pvalb")].copy()
print("M1 Pvalb training cells:", m1.n_obs, dict(m1.obs["ttype"].value_counts()))
print("V1 Pvalb cells:", pv.n_obs)

activity = set(sum(GENE_SETS.values(), []))
shared = m1.var_names.intersection(pv.var_names)
m1s = m1[:, shared].copy()
hv = sc.pp.highly_variable_genes(m1s, n_top_genes=2500, flavor="seurat", inplace=False)
genes = [g for g in shared[hv["highly_variable"].values] if g not in activity][:2000]


def z(M):
    M = np.asarray(M.toarray() if sp.issparse(M) else M)
    return (M - M.mean(0)) / (M.std(0) + 1e-6)


Zm, Zv = z(m1[:, genes].X), z(pv[:, genes].X)
pca = PCA(30, random_state=0).fit(Zm)
knn = KNeighborsClassifier(15).fit(pca.transform(Zm), np.asarray(m1.obs["ttype"].values, dtype=object))
# self-consistency on M1 (5-fold) as a sanity check of the classifier
from sklearn.model_selection import cross_val_score
cv_acc = cross_val_score(KNeighborsClassifier(15), pca.transform(Zm), np.asarray(m1.obs["ttype"].values, dtype=object), cv=5).mean()
proba = knn.predict_proba(pca.transform(Zv))
lab = knn.classes_[proba.argmax(1)]
res = pd.DataFrame({"v1_ttype": pv.obs["T-type Label"].values, "m1_label": lab, "agreement": proba.max(1),
                    "IEG_rapid": pv.obs["IEG_rapid"].values, "n_genes": pv.obs["n_genes"].values,
                    "donor_id": pv.obs["donor_id"].values, "ephys_session_id": pv.obs["ephys_session_id"].values,
                    "cell_specimen_id": pv.obs["cell_specimen_id"].values}, index=pv.obs_names)
res["group"] = np.where((res["m1_label"] == "Pvalb Il1rapl2") & (res["agreement"] >= 0.6), "primary",
                        np.where((res["m1_label"] == "Pvalb Reln") & (res["agreement"] >= 0.6), "secondary", "other"))
res.to_csv(RESULTS / "v1_pvalb_label_transfer.csv")
print(f"M1 5-fold self-accuracy: {cv_acc:.2f}")
print(pd.crosstab(res["v1_ttype"], res["m1_label"]).to_string())
print(res["group"].value_counts().to_string())
