"""Stress-test the L5 IT candidate (rapid-IEG index vs rheobase / sag) from 06.

1. Global multiplicity: BH-FDR across every test 06 ran (all sets x subsets x indices x features).
2. Covariates: soma depth, slice hold time, mouse age, Cre line, recording user, sequencing batch.
3. Within-t-type permutation null for the IT-only effect (5,000 shuffles; mouse structure kept
   by shuffling IEG index across cells within t-type).
4. Single-type check: L5 IT_1 alone (removes any between-subtype leakage).
5. Out-of-sample replication: the same model in L4/5 IT cells, which were NOT in the
   discovery set, and in L5 ET cells.
6. Gene-level: which rapid IEGs carry the effect (leave-one-gene-out index).
Outputs: results/l5_it_robustness.csv, results/l5_it_gene_loo.csv
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)
import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from common import IEG_RAPID  # noqa: E402
RESULTS = paths.results(__file__)
DATA = paths.data(__file__)

rng = np.random.default_rng(0)
allres = pd.read_csv(RESULTS / "l5_ieg_index_assoc.csv")
allres["q_global"] = multipletests(allres["p"], method="fdr_bh")[1]
allres.to_csv(RESULTS / "l5_ieg_index_assoc.csv", index=False)

a = sc.read_h5ad(DATA / "patchseq_m1.h5ad")
obs = a.obs.join(pd.read_csv(RESULTS / "aim1_scores.csv", index_col=0)[["IEG_rapid"]]).copy()
obs["ttype"] = obs["ttype"].astype(str)
obs["lg"] = np.log(obs["n_genes"])
obs["depth"] = pd.to_numeric(obs["Soma depth (µm)"], errors="coerce")
obs["age"] = pd.to_numeric(obs["Mouse age"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
for c in ["Cre", "User", "batch"]:
    obs[c] = obs[c].astype(str)
FEATS = {"Rheobase (pA)": True, "Sag ratio": False, "Sag area (mV*s)": False, "Sag time (s)": False,
         "Input resistance (MOhm)": True}

sets = {"L5 IT (discovery)": obs["ttype"].str.match(r"L5 IT_"),
        "L5 IT_1 only": obs["ttype"] == "L5 IT_1",
        "L4/5 IT (replication)": obs["ttype"].str.match(r"L4/5 IT_"),
        "L5 ET": obs["ttype"].str.match(r"L5 ET_")}


def frame(mask, feat, logt, idx="IEG_rapid"):
    d = obs[mask & obs["has_ephys"]].copy()
    y = d["ephys: " + feat]
    d["y"] = np.log(y.where(y > 0)) if logt else y
    d["s"] = d[idx]
    d = d.dropna(subset=["y", "s"])
    d["y"] = (d["y"] - d["y"].mean()) / d["y"].std()
    d["s"] = (d["s"] - d["s"].mean()) / d["s"].std()
    return d


def ols(d, rhs):
    rhs = rhs if d["ttype"].nunique() > 1 else rhs.replace(" + C(ttype)", "")
    return smf.ols("y ~ s" + rhs, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["mouse"])[0]})


rows = []
covsets = {"base": " + C(ttype) + lg",
           "+depth": " + C(ttype) + lg + depth",
           "+hold time": " + C(ttype) + lg + hold_time",
           "+age": " + C(ttype) + lg + age",
           "+Cre": " + C(ttype) + lg + C(Cre)",
           "+user+batch": " + C(ttype) + lg + C(User) + C(batch)",
           "all covariates": " + C(ttype) + lg + depth + hold_time + age + C(Cre)"}
for sname, mask in sets.items():
    for feat, logt in FEATS.items():
        for cname, rhs in (covsets.items() if sname == "L5 IT (discovery)" else [("base", covsets["base"])]):
            d = frame(mask, feat, logt)
            need = [v for v in ["depth", "hold_time", "age"] if v in rhs]
            d = d.dropna(subset=need)
            r = ols(d, rhs)
            row = {"set": sname, "feature": feat, "model": cname, "n": int(r.nobs), "n_mice": d["mouse"].nunique(),
                   "beta": r.params["s"], "ci_lo": r.conf_int().loc["s", 0], "ci_hi": r.conf_int().loc["s", 1],
                   "p": r.pvalues["s"]}
            if cname == "base" and sname in ("L5 IT (discovery)", "L4/5 IT (replication)"):
                # within-type permutation of the IEG index
                b0, null = r.params["s"], []
                for _ in range(5000):
                    dp = d.copy()
                    dp["s"] = dp.groupby("ttype")["s"].transform(lambda v: rng.permutation(v.values))
                    null.append(smf.ols("y ~ s" + (rhs if d["ttype"].nunique() > 1 else " + lg"), dp).fit().params["s"])
                row["p_perm"] = (np.sum(np.abs(null) >= abs(b0)) + 1) / 5001
            if cname == "base":
                dr = d.copy()
                dr["y"] = dr.groupby("ttype")["y"].rank(pct=True)
                dr["s"] = dr.groupby("ttype")["s"].rank(pct=True)
                rr = ols(dr, " + C(ttype)")
                row.update(beta_rank=rr.params["s"], p_rank=rr.pvalues["s"])
            rows.append(row)
rob = pd.DataFrame(rows)
rob.to_csv(RESULTS / "l5_it_robustness.csv", index=False)

# which genes carry it: leave-one-gene-out rapid index (simple mean z of log-CPM, within L5 IT)
m = sets["L5 IT (discovery)"] & obs["has_ephys"]
genes = [g for g in IEG_RAPID if g in a.var_names]
Xg = pd.DataFrame(a[m.values, genes].X.toarray(), index=obs.index[m], columns=genes)
Z = (Xg - Xg.mean()) / Xg.std().replace(0, np.nan)
Z = Z.dropna(axis=1)
loo = []
for feat in ["Rheobase (pA)", "Sag ratio"]:
    for drop in ["(none)"] + list(Z.columns):
        idx = Z.drop(columns=[drop], errors="ignore").mean(1)
        obs.loc[m, "loo"] = idx
        d = frame(m, feat, FEATS[feat], idx="loo")
        r = ols(d, covsets["base"])
        single = None
        if drop != "(none)":
            obs.loc[m, "single"] = Z[drop]
            ds = frame(m, feat, FEATS[feat], idx="single")
            single = ols(ds, covsets["base"])
        loo.append({"feature": feat, "dropped_gene": drop, "beta_without": r.params["s"], "p_without": r.pvalues["s"],
                    "beta_gene_alone": single.params["s"] if single is not None else np.nan,
                    "p_gene_alone": single.pvalues["s"] if single is not None else np.nan,
                    "detect_frac": (Xg[drop] > 0).mean() if drop in Xg else np.nan})
pd.DataFrame(loo).to_csv(RESULTS / "l5_it_gene_loo.csv", index=False)

pd.set_option("display.width", 220)
print("global q (all 06 tests) min:", allres["q_global"].min().round(3),
      "| tests:", len(allres), "| L5 IT rapid rheobase global q:",
      allres.query("cell_set=='core' and subset=='IT only' and index=='IEG_rapid' and feature=='Rheobase (pA)'")["q_global"].round(3).tolist())
print(rob.round(4).to_string(index=False))
print(pd.DataFrame(loo).round(3).to_string(index=False))
