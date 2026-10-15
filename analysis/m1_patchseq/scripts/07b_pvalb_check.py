"""Stress-test the Pvalb Il1rapl2 per-cluster hit (IEG vs max APs / AP width) from 07.
Output: results/cluster_hit_robustness.csv"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)
import numpy as np, pandas as pd, scanpy as sc, statsmodels.formula.api as smf
RESULTS = paths.results(__file__)
DATA = paths.data(__file__)
rng = np.random.default_rng(0)
o = sc.read_h5ad(DATA / "patchseq_m1.h5ad", backed="r").obs.join(
    pd.read_csv(RESULTS / "aim1_scores.csv", index_col=0)[["IEG_rapid", "IEG_proposal", "IEG_delayed", "stress"]]).copy()
o["ttype"] = o["ttype"].astype(str); o["lg"] = np.log(o["n_genes"])
o["depth"] = pd.to_numeric(o["Soma depth (µm)"], errors="coerce")
o["age"] = pd.to_numeric(o["Mouse age"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
o["Cre"] = o["Cre"].astype(str); o["batch"] = o["batch"].astype(str)
rows = []
for t, f, logt in [("Pvalb Il1rapl2", "Max number of APs", True), ("Pvalb Il1rapl2", "AP width (ms)", True),
                   ("Vip Sncg", "Membrane time constant (ms)", True)]:
    d = o[(o["ttype"] == t) & o["has_ephys"]].copy()
    y = d["ephys: " + f]; d["y"] = np.log(y.where(y > 0)) if logt else y
    d = d.dropna(subset=["y"])
    z = lambda v: (v - v.mean()) / v.std()
    for idx in ["IEG_rapid", "IEG_proposal", "IEG_delayed"]:
        for mname, rhs in [("base", "lg"), ("+stress", "lg + stress"), ("+hold+depth+age", "lg + hold_time + depth + age"),
                           ("+batch", "lg + C(batch)")]:
            dd = d.dropna(subset=[c for c in ["hold_time", "depth", "age"] if c in rhs]).copy()
            dd["s"] = z(dd[idx]); dd["yz"] = z(dd["y"])
            r = smf.ols(f"yz ~ s + {rhs}", dd).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd["mouse"])[0]})
            row = dict(cluster=t, feature=f, index=idx, model=mname, n=int(r.nobs), mice=dd["mouse"].nunique(),
                       beta=r.params["s"], p=r.pvalues["s"])
            if mname == "base":
                rr = smf.ols("yz ~ s + lg", dd.assign(yz=dd["y"].rank(), s=dd[idx].rank())).fit()
                row["spearman_p"] = rr.pvalues["s"]
                null = [smf.ols("yz ~ s + lg", dd.assign(s=rng.permutation(dd["s"].values))).fit().params["s"] for _ in range(5000)]
                row["p_perm"] = (np.sum(np.abs(null) >= abs(r.params["s"])) + 1) / 5001
                row["r_ieg_genes"] = np.corrcoef(dd[idx], dd["lg"])[0, 1]
            rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(RESULTS / "cluster_hit_robustness.csv", index=False)
pd.set_option("display.width", 220)
print(out.round(4).to_string(index=False))
# context: Pvalb clusters, same features
for f in ["Max number of APs", "AP width (ms)"]:
    for t in sorted(o.loc[o["ttype"].str.startswith("Pvalb"), "ttype"].unique()):
        d = o[(o["ttype"] == t) & o["has_ephys"]].dropna(subset=["ephys: " + f, "IEG_rapid"])
        if len(d) >= 10:
            from scipy.stats import spearmanr
            r, p = spearmanr(d["IEG_rapid"], d["ephys: " + f]); print(f"{f[:14]:14s} {t:16s} n={len(d):3d} rho={r:+.2f} p={p:.3f}")
