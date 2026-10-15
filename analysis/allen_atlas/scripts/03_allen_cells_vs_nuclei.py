"""Ex vivo induction control: Allen MOp SMART-seq dissociated cells vs. nuclei.

Compares IEG expression, IEG gene-set scores, and type-specificity of IEG genes
(adjusted R^2 on cluster/subclass labels vs. an expression-matched gene null) between
whole-cell dissociated profiles and single-nucleus profiles from frozen tissue, which
are largely protected from dissociation-induced IEG expression.

Cells carry Allen taxonomy labels (Gene_Expression/Data/metadata.csv). Nuclei are labelled at the
subclass level by kNN transfer from cells on shared HVGs (per-dataset scaled).
Outputs: data/allen_mop_{cells,nuclei}.h5ad, results/aim1b_*.csv

Usage:  uv run python analysis/allen_atlas/scripts/03_allen_cells_vs_nuclei.py
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
from sklearn.neighbors import KNeighborsClassifier
from common import ALLEN_DIR, GENE_SETS  # noqa: E402
RESULTS = paths.results(__file__)
DATA = paths.data(__file__)

rng = np.random.default_rng(0)


def load(kind):
    out = DATA / f"allen_mop_{kind}.h5ad"
    if out.exists():
        return sc.read_h5ad(out)
    d = ALLEN_DIR / f"mouse_MOp_{kind}_gene_expression_matrices_2018-10-04"
    mats = {}
    for part in ["exon", "intron"]:
        t = pacsv.read_csv(d / f"mouse_MOp_{kind}_2018-10-04_{part}-matrix.csv").to_pandas()
        t = t.set_index(t.columns[0])
        mats[part] = t
    genes = pd.read_csv(d / f"mouse_MOp_{kind}_2018-10-04_genes-rows.csv")["gene"].values
    samples = pd.read_csv(d / f"mouse_MOp_{kind}_2018-10-04_samples-columns.csv").set_index("seq_name")
    exon = sp.csr_matrix(mats["exon"].values.T.astype(np.float32))
    intron = sp.csr_matrix(mats["intron"].values.T.astype(np.float32))
    a = ad.AnnData(X=exon + intron, obs=samples.loc[mats["exon"].columns].copy(),
                   var=pd.DataFrame(index=pd.Index(genes).astype(str)))
    a.layers["exon"] = exon
    a.obs = a.obs.astype({c: str for c in a.obs.columns if a.obs[c].dtype == object})
    a.write_h5ad(out, compression="gzip")
    return a


def lognorm(a, layer=None):
    M = a.layers[layer] if layer else a.X
    lib = np.asarray(M.sum(1)).ravel()
    return np.log1p(M.multiply(1e6 / lib[:, None]).toarray()).astype(np.float32)


cells, nuclei = load("cells"), load("nuclei")
cells = cells[cells.obs["sample_type"] == "Cells"].copy()
nuclei = nuclei[nuclei.obs["sample_type"] == "Nuclei"].copy()

meta = pd.read_csv(ALLEN_DIR / "Data" / "metadata.csv", usecols=[
    "sample_name", "class_label", "subclass_label", "cluster_label"]).set_index("sample_name")
cells.obs = cells.obs.join(meta, how="left")
cells = cells[cells.obs["subclass_label"].notna() & (cells.obs["genes_detected_cpm_criterion"].astype(float) > 2000)].copy()
nuclei = nuclei[nuclei.obs["genes_detected_cpm_criterion"].astype(float) > 1500].copy()
print("cells", cells.n_obs, "nuclei", nuclei.n_obs)

for a in (cells, nuclei):
    a.layers["lc"] = lognorm(a)
    a.layers["lc_exon"] = lognorm(a, "exon")
    a.X = a.layers["lc"]

# ---- label transfer to nuclei (subclass) on HVGs shared by both
hv = sc.pp.highly_variable_genes(cells, n_top_genes=2000, flavor="seurat", inplace=False)
hvg = cells.var_names[hv["highly_variable"].values]
ieg_all = sorted(set(sum(GENE_SETS.values(), [])))
hvg = [g for g in hvg if g not in ieg_all]


def z(M):
    return (M - M.mean(0)) / (M.std(0) + 1e-6)


from sklearn.decomposition import PCA
Zc, Zn = z(cells[:, hvg].X), z(nuclei[:, hvg].X)
pca = PCA(30, random_state=0).fit(Zc)
knn = KNeighborsClassifier(15).fit(pca.transform(Zc), cells.obs["subclass_label"].values)
nuclei.obs["subclass_label"] = knn.predict(pca.transform(Zn))
nuclei.obs["label_conf"] = knn.predict_proba(pca.transform(Zn)).max(1)
for a in (cells, nuclei):
    sl = a.obs["subclass_label"].astype(str)
    a.obs["class"] = np.where(sl.str.contains("IT|PT|NP|CT|L6b|Car3"), "Glutamatergic",
                              np.where(sl.isin(["Lamp5", "Sncg", "Vip", "Sst", "Pvalb", "Sst Chodl", "Meis2"]),
                                       "GABAergic", "Non-neuronal"))

# ---- scores (same expression-matched control procedure as Patch-seq)
rows = []
for kind, a in [("cells", cells), ("nuclei", nuclei)]:
    for name, gs in GENE_SETS.items():
        sc.tl.score_genes(a, [g for g in gs if g in a.var_names], score_name=name, ctrl_size=100,
                          n_bins=25, random_state=0)
    for g in ["Fos", "Fosb", "Junb", "Egr1", "Npas4", "Arc", "Nr4a1", "Ier2", "Dusp1", "Hspa1a", "Bdnf", "Nptx2"]:
        for layer in ["lc", "lc_exon"]:
            v = a[:, g].layers[layer].ravel()
            for cl in ["Glutamatergic", "GABAergic"]:
                m = (a.obs["class"] == cl).values.copy()
                rows.append({"prep": kind, "gene": g, "counts": "exon+intron" if layer == "lc" else "exon",
                             "class": cl, "mean_logcpm": v[m].mean(), "frac_detected": (v[m] > 0).mean(),
                             "n": int(m.sum())})
pd.DataFrame(rows).to_csv(RESULTS / "aim1b_ieg_cells_vs_nuclei.csv", index=False)

# ---- type-specificity of IEG genes vs expression-matched genes, in both preps


def frac_var_explained(Y, labels):
    codes, uniq = pd.factorize(np.asarray(labels))
    k, n = len(uniq), len(labels)
    means = np.zeros((k, Y.shape[1]))
    np.add.at(means, codes, Y)
    means /= np.bincount(codes)[:, None]
    ss_w = ((Y - means[codes]) ** 2).sum(0)
    ss_t = ((Y - Y.mean(0)) ** 2).sum(0)
    r2 = 1 - ss_w / np.where(ss_t > 0, ss_t, np.nan)
    return 1 - (1 - r2) * (n - 1) / (n - k)


null_rows, score_rows = [], []
for kind, a, lab in [("cells", cells, "cluster_label"), ("cells", cells, "subclass_label"),
                     ("nuclei", nuclei, "subclass_label")]:
    for cl in ["Glutamatergic", "GABAergic"]:
        m = (a.obs["class"] == cl).values.copy()
        if kind == "nuclei":
            m &= (a.obs["label_conf"] >= 0.8).values
        labels = a.obs[lab].values[m]
        vc = pd.Series(labels).value_counts()
        ok = pd.Series(labels).map(vc).values >= 5
        Y = a.X[m][ok]
        labels = labels[ok]
        expressed = (Y > 0).mean(0) >= 0.10
        g = a.var_names[expressed]
        r2 = pd.Series(frac_var_explained(Y[:, expressed], labels), index=g)
        mu = pd.Series(Y[:, expressed].mean(0), index=g)
        bins = pd.qcut(mu.rank(method="first"), 20, labels=False)
        for name, gs in GENE_SETS.items():
            mem = [x for x in gs if x in g]
            obs_med = r2[mem].median()
            pool = {b: g[(bins == b).values & ~g.isin(mem)] for b in range(20)}
            nulls = np.array([r2[[rng.choice(pool[bins[x]]) for x in mem]].median() for _ in range(2000)])
            null_rows.append({"prep": kind, "labels": lab, "class": cl, "gene_set": name,
                              "n_genes": len(mem), "n_cells": int(ok.sum()), "n_groups": len(np.unique(labels)),
                              "observed_median_r2": obs_med, "null_median": np.median(nulls),
                              "null_q025": np.quantile(nulls, .025), "null_q975": np.quantile(nulls, .975),
                              "p_lower": (np.sum(nulls <= obs_med) + 1) / 2001})
            s = a.obs[name].values[m][ok]
            score_rows.append({"prep": kind, "labels": lab, "class": cl, "score": name,
                               "adjR2_label": frac_var_explained(s[:, None], labels)[0],
                               "score_mean": s.mean(), "score_sd": s.std()})
pd.DataFrame(null_rows).to_csv(RESULTS / "aim1b_type_specificity_null.csv", index=False)
pd.DataFrame(score_rows).to_csv(RESULTS / "aim1b_score_by_label.csv", index=False)

for kind, a in [("cells", cells), ("nuclei", nuclei)]:
    a.obs[["class", "subclass_label"] + list(GENE_SETS) + (["cluster_label"] if kind == "cells" else ["label_conf"])] \
        .to_csv(RESULTS / f"aim1b_scores_{kind}.csv")
print("done")
