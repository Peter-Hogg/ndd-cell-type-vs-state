"""Cross-layer check of the nominal L2/3 IT signal (rapid-IEG index vs resting potential and
membrane time constant): permutation null, stress-score and covariate adjustment, and the
same model in L4/5 IT, L5 IT and L5 ET. Output: results/l23_cross_layer_check.csv"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)
import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.formula.api as smf
RESULTS = paths.results(__file__)
DATA = paths.data(__file__)

rng = np.random.default_rng(0)
o = sc.read_h5ad(DATA / "patchseq_m1.h5ad", backed="r").obs.join(
    pd.read_csv(RESULTS / "aim1_scores.csv", index_col=0)[["IEG_rapid", "stress"]]).copy()
o["ttype"] = o["ttype"].astype(str)
o["lg"] = np.log(o["n_genes"])
o["depth"] = pd.to_numeric(o["Soma depth (µm)"], errors="coerce")
o["age"] = pd.to_numeric(o["Mouse age"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
o["Cre"] = o["Cre"].astype(str)
rows = []
for sname, pat in [("L2/3 IT", r"L2/3 IT_"), ("L4/5 IT", r"L4/5 IT_"), ("L5 IT", r"L5 IT_"), ("L5 ET", r"L5 ET_")]:
    for f, logt in [("Resting membrane potential (mV)", False), ("Membrane time constant (ms)", True),
                    ("Rheobase (pA)", True)]:
        d = o[o["ttype"].str.match(pat) & o["has_ephys"]].copy()
        y = d["ephys: " + f]
        d["y"] = np.log(y.where(y > 0)) if logt else y
        d = d.dropna(subset=["y", "IEG_rapid"])
        for c, src in [("y", "y"), ("s", "IEG_rapid"), ("st", "stress")]:
            d[c] = (d[src] - d[src].mean()) / d[src].std()
        T = " + C(ttype)" if d["ttype"].nunique() > 1 else ""
        for mname, rhs in [("base", "lg"), ("+stress", "lg + st"),
                           ("+hold+depth+age+Cre", "lg + hold_time + depth + age + C(Cre)")]:
            dd = d.dropna(subset=[c for c in ["hold_time", "depth", "age"] if c in rhs])
            r = smf.ols("y ~ s" + T + " + " + rhs, dd).fit(cov_type="cluster",
                                                           cov_kwds={"groups": pd.factorize(dd["mouse"])[0]})
            row = dict(set=sname, feature=f, model=mname, n=int(r.nobs), beta=r.params["s"],
                       ci_lo=r.conf_int().loc["s", 0], ci_hi=r.conf_int().loc["s", 1], p=r.pvalues["s"])
            if mname == "base":
                null = [smf.ols("y ~ s" + T + " + lg", dd.assign(
                    s=dd.groupby("ttype")["s"].transform(lambda v: rng.permutation(v.values)))).fit().params["s"]
                    for _ in range(2000)]
                row["p_perm"] = (np.sum(np.abs(null) >= abs(r.params["s"])) + 1) / 2001
                row["beta_stress_alone"] = smf.ols("y ~ st" + T + " + lg", dd).fit().params["st"]
                row["r_ieg_stress"] = np.corrcoef(dd["s"], dd["st"])[0, 1]
            rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(RESULTS / "l23_cross_layer_check.csv", index=False)
print(out.round(3).to_string(index=False))
