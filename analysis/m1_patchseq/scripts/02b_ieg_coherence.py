"""Is the within-t-type IEG variance a coherent program (a real state) or technical noise?

1. Within-t-type residual co-expression: mean pairwise correlation among IEG genes vs.
   expression-matched random gene sets (2000 draws).
2. Split-half reliability of the IEG score within t-type (random gene splits,
   Spearman-Brown corrected) -> upper bound on how much within-type score variance
   is signal rather than sampling noise.
Outputs: results/aim1_ieg_coherence.csv, results/aim1_ieg_pairwise_corr.csv
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)
import numpy as np
import pandas as pd
import scanpy as sc
from common import GENE_SETS  # noqa: E402
RESULTS = paths.results(__file__)
DATA = paths.data(__file__)

rng = np.random.default_rng(1)
a = sc.read_h5ad(DATA / "patchseq_m1.h5ad")
X = a.X.toarray()
genes = pd.Index(a.var_names)


def residualize(Y, labels):
    codes = pd.factorize(np.asarray(labels))[0]
    means = np.zeros((codes.max() + 1, Y.shape[1]))
    np.add.at(means, codes, Y)
    means /= np.bincount(codes)[:, None]
    return Y - means[codes]


def mean_offdiag_corr(R):
    C = np.corrcoef(R.T)
    iu = np.triu_indices_from(C, 1)
    return np.nanmean(C[iu])


rows, pairs = [], []
for cl in ["Glutamatergic", "GABAergic"]:
    m = (a.obs["class"] == cl).values
    tt = a.obs["ttype"].values[m]
    keep = pd.Series(tt).map(pd.Series(tt).value_counts()).values >= 3
    Y = X[m][keep]
    tt = tt[keep]
    det = (Y > 0).mean(0)
    ok = det >= 0.10
    R = residualize(Y[:, ok], tt)
    g_ok = genes[ok]
    mu = pd.Series(Y[:, ok].mean(0), index=g_ok)
    bins = pd.qcut(mu.rank(method="first"), 20, labels=False)
    col = {g: i for i, g in enumerate(g_ok)}
    for name, gs in GENE_SETS.items():
        mem = [g for g in gs if g in col]
        obs_c = mean_offdiag_corr(R[:, [col[g] for g in mem]])
        pool = {b: np.where((bins == b).values & ~g_ok.isin(mem))[0] for b in range(20)}
        null = np.array([mean_offdiag_corr(R[:, [rng.choice(pool[bins[g]]) for g in mem]])
                         for _ in range(2000)])
        # split-half reliability of the set score (z-scored residual genes, mean)
        Z = R[:, [col[g] for g in mem]]
        Z = (Z - Z.mean(0)) / Z.std(0)
        rel = []
        for _ in range(500):
            p = rng.permutation(len(mem))
            h1, h2 = Z[:, p[: len(mem) // 2]].mean(1), Z[:, p[len(mem) // 2:]].mean(1)
            r = np.corrcoef(h1, h2)[0, 1]
            rel.append(2 * r / (1 + r))
        rows.append({"class": cl, "gene_set": name, "n_genes": len(mem), "n_cells": len(tt),
                     "mean_pairwise_r_within_type": obs_c, "null_median": np.median(null),
                     "null_q975": np.quantile(null, .975),
                     "p_higher": (np.sum(null >= obs_c) + 1) / 2001,
                     "split_half_reliability": np.median(rel)})
        if name == "IEG_rapid":
            C = pd.DataFrame(np.corrcoef(R[:, [col[g] for g in mem]].T), index=mem, columns=mem)
            C.insert(0, "class", cl)
            pairs.append(C)
pd.DataFrame(rows).to_csv(RESULTS / "aim1_ieg_coherence.csv", index=False)
pd.concat(pairs).to_csv(RESULTS / "aim1_ieg_pairwise_corr.csv")
print(pd.DataFrame(rows).round(3).to_string())
