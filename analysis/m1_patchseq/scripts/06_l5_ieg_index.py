"""Layer-focused (L5 by default; `--layer 23` for L2/3): does an expression-based IEG index predict the intrinsic properties
of layer-5 pyramidal neurons?

Cell sets
  core      : t-types L5 IT_* and L5 ET_*
  extended  : core + L4/5 IT_* + L5/6 NP_*  (sensitivity)
IEG indices : rapid IEGs, the IEG_proposal set, delayed IEGs (expression-matched scores from 02)
Model       : feature_z ~ index_z + C(ttype) + log(n_genes), SE clustered by mouse;
              BH-FDR across all 29 features within each (cell set, index).
Also        : minimum detectable effect (80% power, alpha 0.05 two-sided: 2.8 x SE),
              IT-only / ET-only fits, quintile dose-response, CV multivariate prediction.
Outputs     : results/l5_*.csv  (or results/l23_*.csv with --layer 23)
L2/3        : core = L2/3 IT_* (almost entirely L2/3 IT_3); extended = core + L4/5 IT_*
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
RESULTS = paths.results(__file__)
DATA = paths.data(__file__)

rng = np.random.default_rng(0)
a = sc.read_h5ad(DATA / "patchseq_m1.h5ad", backed="r")
obs = a.obs.join(pd.read_csv(RESULTS / "aim1_scores.csv", index_col=0)[["IEG_rapid", "IEG_proposal", "IEG_delayed"]])
tt = obs["ttype"].astype(str)
_ap = argparse.ArgumentParser()
_ap.add_argument("--layer", default="5", choices=["5", "23"])
LAYER = _ap.parse_args().layer
PREFIX = "l5" if LAYER == "5" else "l23"
if LAYER == "5":
    SETS = {"core": tt.str.match(r"L5 (IT|ET)_"),
            "extended": tt.str.match(r"L5 (IT|ET)_|L4/5 IT_|L5/6 NP_")}
else:
    SETS = {"core": tt.str.match(r"L2/3 IT_"),
            "extended": tt.str.match(r"L2/3 IT_|L4/5 IT_")}
INDICES = ["IEG_rapid", "IEG_proposal", "IEG_delayed"]
FEATS = [c for c in obs.columns if c.startswith("ephys: ")]
# log only strictly-positive skewed features; Sag area / Sag time are zero for cells without a
# detectable sag (66/77 L5 IT cells) and are left on the raw scale (effectively presence/size of sag).
POS_SKEW = ["Input resistance (MOhm)", "Rheobase (pA)", "Membrane time constant (ms)", "AP width (ms)",
            "Latency (ms)", "Latency @ +20pA current (ms)", "Max number of APs",
            "AP Fano factor", "ISI Fano factor"]


def prep(mask):
    d = obs[mask & obs["has_ephys"]].copy()
    d["ttype"] = d["ttype"].astype(str)  # categorical dtype would add empty dummies for unused t-types
    for c in FEATS:
        f = c.removeprefix("ephys: ")
        if f in POS_SKEW:
            d[c] = np.log(d[c].where(d[c] > 0))   # non-positive values -> missing, not clipped
    d["family"] = np.where(d["ttype"].str.contains("ET"), "ET",
                           np.where(d["ttype"].str.contains("NP"), "NP", "IT"))
    d["lg"] = np.log(d["n_genes"])
    return d


def fit(d, feat, idx, by_type=True):
    x = d[[feat, idx, "ttype", "mouse", "lg"]].dropna()
    x = x[x["ttype"].map(x["ttype"].value_counts()) >= 2]
    if x[feat].std() == 0 or len(x) < 15:
        return None
    x = x.assign(y=(x[feat] - x[feat].mean()) / x[feat].std(), s=(x[idx] - x[idx].mean()) / x[idx].std())
    form = "y ~ s + C(ttype) + lg" if by_type and x["ttype"].nunique() > 1 else "y ~ s + lg"
    r = smf.ols(form, x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x["mouse"])[0]})
    return {"n": int(r.nobs), "beta": r.params["s"], "se": r.bse["s"], "ci_lo": r.conf_int().loc["s", 0],
            "ci_hi": r.conf_int().loc["s", 1], "p": r.pvalues["s"], "mde_80": 2.8 * r.bse["s"]}


rows = []
for set_name, mask in SETS.items():
    d = prep(mask)
    for idx in INDICES:
        subsets = [("all L5", d), ("IT only", d[d["family"] == "IT"]), ("ET only", d[d["family"] == "ET"])] if LAYER == "5" else [("all L2/3", d)]
        for sub, dd in subsets:
            for c in FEATS:
                r = fit(dd, c, idx)
                if r:
                    rows.append({"cell_set": set_name, "subset": sub, "index": idx,
                                 "feature": c.removeprefix("ephys: "), **r})
res = pd.DataFrame(rows)
res["q_global"] = multipletests(res["p"], method="fdr_bh")[1]
res["q_fdr"] = res.groupby(["cell_set", "subset", "index"])["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
res.to_csv(RESULTS / f"{PREFIX}_ieg_index_assoc.csv", index=False)

# quintile dose-response (core set, rapid index, within-type centred feature)
d = prep(SETS["core"])
d["q"] = d.groupby("ttype")["IEG_rapid"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)
                                                   if len(s) >= 5 else np.nan)
dose = []
for c in FEATS:
    v = d[c] - d.groupby("ttype")[c].transform("mean")
    v = v / d[c].std()
    for q, g in v.groupby(d["q"]):
        dose.append({"feature": c.removeprefix("ephys: "), "quintile": int(q) + 1, "mean_z": g.mean(),
                     "sem": g.std() / np.sqrt(g.notna().sum()), "n": int(g.notna().sum())})
pd.DataFrame(dose).to_csv(RESULTS / f"{PREFIX}_dose_response.csv", index=False)

# multivariate: all features -> within-type IEG index (CV by mouse, within-type permutation null)
mv = []
for set_name, mask in SETS.items():
    d = prep(mask)
    F = d[FEATS].copy()
    F = F.loc[:, F.notna().mean() > 0.95]
    keep = F.notna().all(1)
    d, F = d[keep], F[keep]
    F = F - F.groupby(d["ttype"]).transform("mean")
    F = ((F - F.mean()) / F.std()).fillna(0).values
    groups = pd.factorize(d["mouse"])[0]
    for idx in INDICES:
        y = (d[idx] - d.groupby("ttype")[idx].transform("mean")).values

        def cv(yv):
            pred = np.zeros_like(yv)
            for tr, te in GroupKFold(5).split(F, yv, groups):
                pred[te] = RidgeCV(alphas=np.logspace(-1, 4, 20)).fit(F[tr], yv[tr]).predict(F[te])
            return 1 - np.sum((yv - pred) ** 2) / np.sum((yv - yv.mean()) ** 2)
        r2 = cv(y)
        null = []
        for _ in range(500):
            yp = y.copy()
            for t in d["ttype"].unique():
                ix = np.where(d["ttype"].values == t)[0]
                yp[ix] = yp[rng.permutation(ix)]
            null.append(cv(yp))
        mv.append({"cell_set": set_name, "index": idx, "n": len(y), "n_features": F.shape[1], "cv_R2": r2,
                   "null_q95": np.quantile(null, .95), "p_perm": (np.sum(np.array(null) >= r2) + 1) / 501})
pd.DataFrame(mv).to_csv(RESULTS / f"{PREFIX}_multivariate.csv", index=False)

pd.set_option("display.width", 220)
s = res[res["subset"].str.startswith("all")]
print(s.groupby(["cell_set", "index"]).agg(n=("n", "max"), min_p=("p", "min"), min_q=("q_fdr", "min"),
                                          n_q05=("q_fdr", lambda q: (q < .05).sum()),
                                          median_mde=("mde_80", "median")).round(3))
print(res[res["p"] < 0.05].sort_values("p")[["cell_set", "subset", "index", "feature", "n", "beta", "p", "q_fdr"]]
      .round(4).to_string(index=False))
print(pd.DataFrame(mv).round(3).to_string(index=False))
