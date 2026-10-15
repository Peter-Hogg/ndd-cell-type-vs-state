"""Test whether IEG_rapid predicts excitability in V1 Pvalb Patch-seq cells assigned to M1 Pvalb t-types.

Primary  : V1 cells assigned M1 'Pvalb Il1rapl2' (agreement >= 0.6), outcomes max spikes (+) and AP width (-),
           one-sided, alpha 0.025 each (Bonferroni over 2 outcomes).
Secondary: cells assigned 'Pvalb Reln' (>= 0.6); and all V1 Pvalb cells pooled with VISp t-type fixed effects.
Model    : z(log outcome) ~ z(IEG_rapid) + log(n_genes) + C(protocol) [+ C(v1_ttype) for pooled],
           mouse(donor)-clustered SE when >= 10 donors, else HC3.
Verdict  : replicated / partially replicated / did not replicate = 2 / 1 / 0 primary outcomes pass.
Output   : results/v1_pvalb_test.csv

Usage:  uv run python analysis/v1_patchseq/scripts/10_v1_pvalb_test.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
RESULTS = paths.results(__file__)

lab = pd.read_csv(paths.CROSS / "results" / "v1_pvalb_label_transfer.csv", index_col=0)
eph = pd.read_csv(RESULTS / "v1_pvalb_ephys_features.csv", index_col=0)
spike_col = "max_n_spikes" if "max_n_spikes" in eph.columns else "max_rate_hz"
d = lab.join(eph[[spike_col, "ap_width_ms", "protocol"]], how="left")
d["lg"] = np.log(d["n_genes"])
OUTCOMES = [(spike_col, +1), ("ap_width_ms", -1)]


def fit(g, outcome, sign, fixed_type=False):
    x = g.dropna(subset=[outcome, "IEG_rapid"]).copy()
    x = x[x[outcome] > 0]
    if len(x) < 30:
        return {"n": len(x), "status": "underpowered (<30)"}
    x["y"] = np.log(x[outcome])
    x["y"] = (x["y"] - x["y"].mean()) / x["y"].std()
    x["s"] = (x["IEG_rapid"] - x["IEG_rapid"].mean()) / x["IEG_rapid"].std()
    form = "y ~ s + lg + C(protocol)" + (" + C(v1_ttype)" if fixed_type else "")
    nd = x["donor_id"].nunique()
    if nd >= 10:
        r = smf.ols(form, x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x["donor_id"])[0]})
        se_kind = "donor-clustered"
    else:
        r = smf.ols(form, x).fit(cov_type="HC3")
        se_kind = "HC3"
    b, se = r.params["s"], r.bse["s"]
    t = b / se
    p_one = stats.norm.sf(t) if sign > 0 else stats.norm.cdf(t)
    return {"n": int(r.nobs), "donors": nd, "se_kind": se_kind, "beta": b, "ci_lo": b - 1.96 * se,
            "ci_hi": b + 1.96 * se, "p_one_sided": p_one, "predicted_sign": "+" if sign > 0 else "-",
            "pass_alpha_0.025": bool(p_one < 0.025), "status": "tested"}


rows = []
for test, g, fe in [("primary: Pvalb Il1rapl2 (transferred)", d[d["group"] == "primary"], False),
                    ("secondary: Pvalb Reln (transferred)", d[d["group"] == "secondary"], False),
                    ("secondary: all V1 Pvalb, within VISp t-type", d, True)]:
    for outcome, sign in OUTCOMES:
        rows.append({"test": test, "outcome": outcome, **fit(g, outcome, sign, fe)})
res = pd.DataFrame(rows)
res.to_csv(RESULTS / "v1_pvalb_test.csv", index=False)
prim = res[res["test"].str.startswith("primary")]
n_pass = int(prim["pass_alpha_0.025"].fillna(False).sum())
verdict = {2: "replicated", 1: "partially replicated", 0: "did not replicate"}[n_pass]
pd.set_option("display.width", 220)
print(res.round(4).to_string(index=False))
print("PRIMARY VERDICT:", verdict)
