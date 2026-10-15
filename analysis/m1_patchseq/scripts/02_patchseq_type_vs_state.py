"""Is IEG ('plasticity') expression a cell-type property or a cell-state property
in M1 Patch-seq, and does it explain intrinsic electrophysiology beyond t-type?

Outputs (results/aim1_*):
  scores.csv                 per-cell gene-set scores (expression-matched control sets)
  gene_variance.csv          per-gene fraction of variance explained by t-type / by mouse
  geneset_variance_null.csv  IEG sets vs expression-matched random gene sets
  classification.csv         t-type decodability with / without IEG genes
  hold_time.csv              IEG score vs time from slicing
  ephys_assoc.csv            per-feature IEG association within t-type (+ type R2)
  ephys_multivariate.csv     cross-validated prediction of IEG state from ephys
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)
import json
import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold, GroupKFold
from statsmodels.stats.multitest import multipletests
from common import GENE_SETS, EPHYS_KEY  # noqa: E402
RESULTS = paths.results(__file__)
DATA = paths.data(__file__)

rng = np.random.default_rng(0)
RESULTS.mkdir(exist_ok=True)
a = sc.read_h5ad(DATA / "patchseq_m1.h5ad")
X = a.X.toarray()
genes = np.asarray(a.var_names)
obs = a.obs.copy()
obs["ttype"] = obs["ttype"].astype(str)  # categorical dtype adds empty dummies to C(ttype)

# ---------------------------------------------------------------- 1. scores
for name, gs in GENE_SETS.items():
    sc.tl.score_genes(a, [g for g in gs if g in a.var_names], score_name=name,
                      ctrl_size=100, n_bins=25, random_state=0)
    obs[name] = a.obs[name]
score_cols = list(GENE_SETS)
obs[["ttype", "family", "class", "mouse", "batch", "hold_time", "n_genes"] + score_cols] \
    .to_csv(RESULTS / "aim1_scores.csv")

# ---------------------------------------------------------- 2. per-gene variance
expressed = (X > 0).mean(0) >= 0.10
Xe, ge = X[:, expressed], genes[expressed]


def frac_var_explained(Y, labels, min_n=1):
    """Adjusted R^2 of a one-way ANOVA of every column of Y on `labels`."""
    labels = pd.Series(np.asarray(labels))
    ok = labels.map(labels.value_counts()) >= min_n
    Y, labels = Y[ok.values], labels[ok].values
    codes, uniq = pd.factorize(labels)
    k, n = len(uniq), len(labels)
    means = np.zeros((k, Y.shape[1]))
    np.add.at(means, codes, Y)
    means /= np.bincount(codes)[:, None]
    ss_w = ((Y - means[codes]) ** 2).sum(0)
    ss_t = ((Y - Y.mean(0)) ** 2).sum(0)
    r2 = 1 - ss_w / np.where(ss_t > 0, ss_t, np.nan)
    return 1 - (1 - r2) * (n - 1) / (n - k)


def residualize(Y, labels):
    codes = pd.factorize(np.asarray(labels))[0]
    means = np.zeros((codes.max() + 1, Y.shape[1]))
    np.add.at(means, codes, Y)
    means /= np.bincount(codes)[:, None]
    return Y - means[codes]


rows = []
for scope, m in [("all", np.ones(len(obs), bool)),
                 ("Glutamatergic", (obs["class"] == "Glutamatergic").values),
                 ("GABAergic", (obs["class"] == "GABAergic").values)]:
    Y = Xe[m]
    tt = obs["ttype"].values[m]
    r2_type = frac_var_explained(Y, tt)
    # mouse effect on within-t-type residuals, restricted to mice with >=3 cells
    resid = residualize(Y, tt)
    r2_mouse = frac_var_explained(resid, obs["mouse"].values[m], min_n=3)
    rows.append(pd.DataFrame({"scope": scope, "gene": ge, "mean_logcpm": Y.mean(0),
                              "detect": (Y > 0).mean(0), "var": Y.var(0),
                              "r2_ttype": r2_type, "r2_mouse_within_type": r2_mouse}))
gv = pd.concat(rows)
gv.to_csv(RESULTS / "aim1_gene_variance.csv", index=False)

# expression-matched null for gene-set medians
null_rows = []
for scope, d in gv.groupby("scope"):
    d = d.set_index("gene")
    bins = pd.qcut(d["mean_logcpm"].rank(method="first"), 20, labels=False)
    for name, gs in GENE_SETS.items():
        members = [g for g in gs if g in d.index]
        obs_stats = d.loc[members, ["r2_ttype", "r2_mouse_within_type", "var"]].median()
        pool = {b: d.index[(bins == b) & ~d.index.isin(members)].values for b in range(20)}
        nulls = []
        for _ in range(2000):
            draw = [rng.choice(pool[bins[g]]) for g in members]
            nulls.append(d.loc[draw, ["r2_ttype", "r2_mouse_within_type", "var"]].median())
        nulls = pd.DataFrame(nulls)
        for stat in ["r2_ttype", "r2_mouse_within_type", "var"]:
            null_rows.append({"scope": scope, "gene_set": name, "n_genes": len(members),
                              "stat": stat, "observed_median": obs_stats[stat],
                              "null_median": nulls[stat].median(),
                              "null_q025": nulls[stat].quantile(.025),
                              "null_q975": nulls[stat].quantile(.975),
                              "p_lower": (np.sum(nulls[stat] <= obs_stats[stat]) + 1) / (len(nulls) + 1),
                              "p_higher": (np.sum(nulls[stat] >= obs_stats[stat]) + 1) / (len(nulls) + 1)})
pd.DataFrame(null_rows).to_csv(RESULTS / "aim1_geneset_variance_null.csv", index=False)

# score-level variance partition
part = []
for scope, m in [("all", obs.index), ("Glutamatergic", obs.index[obs["class"] == "Glutamatergic"]),
                 ("GABAergic", obs.index[obs["class"] == "GABAergic"])]:
    d = obs.loc[m]
    for s in score_cols:
        r_t = frac_var_explained(d[[s]].values, d["ttype"].values)[0]
        r_m = frac_var_explained(residualize(d[[s]].values, d["ttype"].values), d["mouse"].values, 3)[0]
        r_b = frac_var_explained(residualize(d[[s]].values, d["ttype"].values), d["batch"].values)[0]
        part.append({"scope": scope, "score": s, "adjR2_ttype": r_t,
                     "adjR2_mouse_within_type": r_m, "adjR2_batch_within_type": r_b})
pd.DataFrame(part).to_csv(RESULTS / "aim1_score_variance_partition.csv", index=False)

# --------------------------------------------- 3. do IEGs carry t-type identity?
hv = sc.pp.highly_variable_genes(a, n_top_genes=3000, flavor="seurat", inplace=False)
hvg = genes[hv["highly_variable"].values]
ieg_all = sorted(set(sum(GENE_SETS.values(), [])) & set(genes))
expr_rank = pd.Series(X.mean(0), index=genes)


def knn_acc(gene_list, mask, n_pcs=30, seed=0):
    idx = np.isin(genes, gene_list)
    Z = X[np.ix_(mask, idx)]
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
    y = obs["ttype"].values[mask]
    keep = pd.Series(y).map(pd.Series(y).value_counts()).values >= 5
    Z, y = Z[keep], y[keep]
    Z = PCA(min(n_pcs, Z.shape[1] - 1), random_state=seed).fit_transform(Z)
    accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Z, y):
        accs.append(KNeighborsClassifier(15).fit(Z[tr], y[tr]).score(Z[te], y[te]))
    return np.mean(accs)


cls = []
for scope, mask in [("Glutamatergic", (obs["class"] == "Glutamatergic").values),
                    ("GABAergic", (obs["class"] == "GABAergic").values)]:
    base = knn_acc(hvg, mask)
    no_ieg = knn_acc([g for g in hvg if g not in ieg_all], mask)
    ieg_only = knn_acc(ieg_all, mask, n_pcs=10)
    # same-size, expression-matched random sets
    eb = pd.qcut(expr_rank.rank(method="first"), 50, labels=False)
    rand = []
    for i in range(50):
        draw = [rng.choice(genes[(eb == eb[g]).values & ~np.isin(genes, ieg_all)]) for g in ieg_all]
        rand.append(knn_acc(draw, mask, n_pcs=10, seed=i))
    # removing an equal number of random HVGs
    rem = [knn_acc(list(np.setdiff1d(hvg, rng.choice(hvg, np.isin(hvg, ieg_all).sum() or 1, replace=False))), mask, seed=i)
           for i in range(10)]
    chance = obs.loc[mask, "ttype"].value_counts(normalize=True).max()
    cls.append({"scope": scope, "n_ttypes": obs.loc[mask, "ttype"].nunique(), "acc_hvg": base,
                "acc_hvg_minus_IEG": no_ieg, "acc_hvg_minus_random": np.mean(rem),
                "n_IEG_in_hvg": int(np.isin(hvg, ieg_all).sum()), "acc_IEG_only": ieg_only,
                "acc_random_matched_mean": np.mean(rand),
                "acc_random_matched_q975": np.quantile(rand, .975),
                "p_IEG_only_vs_random": (np.sum(np.array(rand) >= ieg_only) + 1) / 51,
                "majority_class_rate": chance})
pd.DataFrame(cls).to_csv(RESULTS / "aim1_classification.csv", index=False)

# t-type marker lists: how often are IEGs among the top-50 markers of a t-type?
big = obs["ttype"].value_counts()
sc.tl.rank_genes_groups(a, "ttype", groups=list(big.index[big >= 3]), method="wilcoxon", n_genes=50)
mk = pd.DataFrame(a.uns["rank_genes_groups"]["names"])
ieg_markers = {t: sorted(set(mk[t]) & set(ieg_all)) for t in mk.columns}
ieg_markers = {t: v for t, v in ieg_markers.items() if v}
json.dump(ieg_markers, open(RESULTS / "aim1_ttypes_with_IEG_markers.json", "w"), indent=1)

# ---------------------------------------------------------- 4. hold time
ht = []
for s in score_cols:
    d = obs.dropna(subset=["hold_time"])
    f = smf.ols(f"{s} ~ hold_time + C(ttype) + np.log(n_genes)", d).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["mouse"])[0]})
    ht.append({"score": s, "beta_per_10min": 10 * f.params["hold_time"],
               "ci_lo": 10 * f.conf_int().loc["hold_time", 0], "ci_hi": 10 * f.conf_int().loc["hold_time", 1],
               "p": f.pvalues["hold_time"], "n": int(f.nobs)})
pd.DataFrame(ht).to_csv(RESULTS / "aim1_hold_time.csv", index=False)

# ------------------------------------------------ 5. ephys association within type
feat_cols = ["ephys: " + f for f in EPHYS_KEY]
E = obs[feat_cols].copy()
E.columns = EPHYS_KEY
# robust transforms for skewed positive features
for f in ["Input resistance (MOhm)", "Rheobase (pA)", "Membrane time constant (ms)",
          "AP width (ms)", "Latency (ms)", "Max number of APs"]:
    E[f] = np.log(E[f].clip(lower=1e-3))
assoc = []
for scope, m in [("all", obs.index), ("Glutamatergic", obs.index[obs["class"] == "Glutamatergic"]),
                 ("GABAergic", obs.index[obs["class"] == "GABAergic"])]:
    for f in EPHYS_KEY:
        d = pd.DataFrame({"y": E.loc[m, f], "ttype": obs.loc[m, "ttype"], "mouse": obs.loc[m, "mouse"],
                          "hold": obs.loc[m, "hold_time"], "lg": np.log(obs.loc[m, "n_genes"])})
        for s in score_cols:
            d["s"] = obs.loc[m, s]
            dd = d.dropna()
            dd = dd[dd["ttype"].map(dd["ttype"].value_counts()) >= 3]
            sd_y = dd["y"].std()
            dd = dd.assign(y=(dd["y"] - dd["y"].mean()) / sd_y,
                           s=(dd["s"] - dd["s"].mean()) / dd["s"].std())
            full = smf.ols("y ~ s + C(ttype) + lg", dd).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(dd["mouse"])[0]})
            red_t = smf.ols("y ~ s + lg", dd).fit()
            red_s = smf.ols("y ~ C(ttype) + lg", dd).fit()
            assoc.append({"scope": scope, "feature": f, "score": s, "n": int(full.nobs),
                          "beta_sd_per_sd": full.params["s"], "ci_lo": full.conf_int().loc["s", 0],
                          "ci_hi": full.conf_int().loc["s", 1], "p": full.pvalues["s"],
                          "partialR2_score": (red_s.ssr - full.ssr) / red_s.ssr,
                          "partialR2_ttype": (red_t.ssr - full.ssr) / red_t.ssr})
assoc = pd.DataFrame(assoc)
assoc["q_fdr"] = np.nan
for (sc_, s), g in assoc.groupby(["scope", "score"]):
    assoc.loc[g.index, "q_fdr"] = multipletests(g["p"], method="fdr_bh")[1]
assoc.to_csv(RESULTS / "aim1_ephys_assoc.csv", index=False)

# ------------------------------------- 6. multivariate: ephys -> IEG state (CV)
mv = []
Ez = E.copy()
for scope, m in [("Glutamatergic", obs["class"] == "Glutamatergic"), ("GABAergic", obs["class"] == "GABAergic")]:
    d = pd.concat([Ez[m.values], obs.loc[m, score_cols + ["ttype", "mouse"]]], axis=1).dropna()
    d = d[d["ttype"].map(d["ttype"].value_counts()) >= 3]
    F = residualize(d[EPHYS_KEY].values.astype(float), d["ttype"])
    F = (F - F.mean(0)) / F.std(0)
    groups = pd.factorize(d["mouse"])[0]
    for s in score_cols:
        y = residualize(d[[s]].values, d["ttype"]).ravel()

        def cv_r2(yv):
            pred = np.zeros_like(yv)
            for tr, te in GroupKFold(5).split(F, yv, groups):
                pred[te] = RidgeCV(alphas=np.logspace(-2, 4, 20)).fit(F[tr], yv[tr]).predict(F[te])
            return 1 - np.sum((yv - pred) ** 2) / np.sum((yv - yv.mean()) ** 2)
        r2 = cv_r2(y)
        null = []
        tt = d["ttype"].values
        for _ in range(200):
            yp = y.copy()
            for t in np.unique(tt):
                ix = np.where(tt == t)[0]
                yp[ix] = yp[rng.permutation(ix)]
            null.append(cv_r2(yp))
        mv.append({"scope": scope, "score": s, "n": len(y), "cv_R2": r2,
                   "null_q95": np.quantile(null, .95), "p_perm": (np.sum(np.array(null) >= r2) + 1) / 201})
    # reference: how well does ephys decode t-type in the same cells?
    Fall = d[EPHYS_KEY].values.astype(float)
    Fall = (Fall - Fall.mean(0)) / Fall.std(0)
    yt = d["ttype"].values
    keep = pd.Series(yt).map(pd.Series(yt).value_counts()).values >= 5
    accs = [KNeighborsClassifier(15).fit(Fall[keep][tr], yt[keep][tr]).score(Fall[keep][te], yt[keep][te])
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Fall[keep], yt[keep])]
    mv.append({"scope": scope, "score": "ttype_decoding_acc", "n": int(keep.sum()), "cv_R2": np.mean(accs),
               "null_q95": pd.Series(yt[keep]).value_counts(normalize=True).max(), "p_perm": np.nan})
pd.DataFrame(mv).to_csv(RESULTS / "aim1_ephys_multivariate.csv", index=False)
print("done")
