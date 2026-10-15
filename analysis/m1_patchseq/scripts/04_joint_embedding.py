"""Joint transcriptome-electrophysiology latent space within Patch-seq.

Cross-validated CCA between transcriptomic PCs and intrinsic ephys features, separately
for glutamatergic and GABAergic cells. For each shared canonical dimension we ask what
organizes it: t-type / family labels, or the IEG ('plasticity') score?

Outputs: results/aim3_cca.csv, results/aim3_cca_structure.csv, data/aim3_embedding.csv
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)
import numpy as np
import pandas as pd
import scanpy as sc
import umap
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from common import GENE_SETS, EPHYS_KEY  # noqa: E402
RESULTS = paths.results(__file__)
DATA = paths.data(__file__)

rng = np.random.default_rng(0)
a = sc.read_h5ad(DATA / "patchseq_m1.h5ad")
scores = pd.read_csv(RESULTS / "aim1_scores.csv", index_col=0)
ieg_all = sorted(set(sum(GENE_SETS.values(), [])))
LOGF = ["Input resistance (MOhm)", "Rheobase (pA)", "Membrane time constant (ms)", "AP width (ms)",
        "Latency (ms)", "Max number of APs"]
N_PC, N_CC = 20, 5


def adj_r2(y, labels):
    codes, uniq = pd.factorize(np.asarray(labels))
    k, n = len(uniq), len(y)
    means = np.bincount(codes, y) / np.bincount(codes)
    r2 = 1 - np.sum((y - means[codes]) ** 2) / np.sum((y - y.mean()) ** 2)
    return 1 - (1 - r2) * (n - 1) / (n - k)


cca_rows, struct_rows, emb = [], [], []
for cl in ["Glutamatergic", "GABAergic"]:
    b = a[(a.obs["class"] == cl).values & a.obs["has_ephys"].values].copy()
    E = b.obs[["ephys: " + f for f in EPHYS_KEY]].copy()
    E.columns = EPHYS_KEY
    for f in LOGF:
        E[f] = np.log(E[f].clip(lower=1e-3))
    ok = E.notna().all(1).values
    b, E = b[ok].copy(), E[ok]
    hv = sc.pp.highly_variable_genes(b, n_top_genes=2000, flavor="seurat", inplace=False)
    for gene_mode in ["HVG", "HVG_minus_IEG"]:
        g = b.var_names[hv["highly_variable"].values]
        if gene_mode == "HVG_minus_IEG":
            g = [x for x in g if x not in ieg_all]
        T = b[:, g].X.toarray()
        T = (T - T.mean(0)) / (T.std(0) + 1e-6)
        T = PCA(N_PC, random_state=0).fit_transform(T)
        F = E.values.astype(float)
        F = (F - F.mean(0)) / F.std(0)

        # held-out canonical correlations
        held = np.zeros((len(T), N_CC, 2))
        for tr, te in KFold(5, shuffle=True, random_state=0).split(T):
            m = CCA(N_CC, max_iter=2000).fit(T[tr], F[tr])
            u, v = m.transform(T[te], F[te])
            held[te, :, 0], held[te, :, 1] = u, v
        cc_test = [np.corrcoef(held[:, k, 0], held[:, k, 1])[0, 1] for k in range(N_CC)]
        # permutation null (shuffle cells in ephys space)
        null = []
        for i in range(100):
            Fp = F[rng.permutation(len(F))]
            h = np.zeros((len(T), 2))
            for tr, te in KFold(5, shuffle=True, random_state=i).split(T):
                m = CCA(1, max_iter=2000).fit(T[tr], Fp[tr])
                u, v = m.transform(T[te], Fp[te])
                h[te, 0], h[te, 1] = u[:, 0], v[:, 0]
            null.append(np.corrcoef(h[:, 0], h[:, 1])[0, 1])
        for k in range(N_CC):
            cca_rows.append({"class": cl, "genes": gene_mode, "n": len(T), "cc": k + 1,
                             "heldout_r": cc_test[k], "null_q95_cc1": np.quantile(null, .95)})

        if gene_mode != "HVG":
            continue
        # full-data CCA for interpretation
        m = CCA(N_CC, max_iter=2000).fit(T, F)
        U, V = m.transform(T, F)
        joint = (U + V) / 2
        obs = b.obs.join(scores[list(GENE_SETS)])
        for k in range(N_CC):
            y = joint[:, k]
            row = {"class": cl, "cc": k + 1,
                   "adjR2_ttype": adj_r2(y, obs["ttype"]),
                   "adjR2_family": adj_r2(y, obs["family"]),
                   "adjR2_mouse": adj_r2(y, obs["mouse"])}
            for s in GENE_SETS:
                row[f"r2_{s}"] = np.corrcoef(y, obs[s])[0, 1] ** 2
                # within-type: residual of joint axis vs residual of score
                yr = y - pd.Series(y, index=obs.index).groupby(obs["ttype"]).transform("mean").values
                sr = obs[s].values - obs.groupby("ttype")[s].transform("mean").values
                row[f"r2_within_type_{s}"] = np.corrcoef(yr, sr)[0, 1] ** 2
            # top ephys loadings
            load = pd.Series(np.corrcoef(np.c_[V[:, k], F].T)[0, 1:], index=EPHYS_KEY)
            row["top_ephys"] = "; ".join(f"{i} ({v:+.2f})" for i, v in load.abs().sort_values(ascending=False)
                                         .head(3).rename(lambda x: x).items())
            struct_rows.append(row)
        em = umap.UMAP(n_neighbors=20, min_dist=0.3, random_state=0).fit_transform(joint)
        emb.append(pd.DataFrame({"cell": obs.index, "class": cl, "umap1": em[:, 0], "umap2": em[:, 1],
                                 "ttype": obs["ttype"].values, "family": obs["family"].values,
                                 **{s: obs[s].values for s in GENE_SETS},
                                 **{f"cc{k+1}": joint[:, k] for k in range(N_CC)}}))

pd.DataFrame(cca_rows).to_csv(RESULTS / "aim3_cca.csv", index=False)
pd.DataFrame(struct_rows).to_csv(RESULTS / "aim3_cca_structure.csv", index=False)
pd.concat(emb).to_csv(DATA / "aim3_embedding.csv", index=False)
print(pd.DataFrame(cca_rows).round(3).to_string())
print(pd.DataFrame(struct_rows).round(3).to_string())
