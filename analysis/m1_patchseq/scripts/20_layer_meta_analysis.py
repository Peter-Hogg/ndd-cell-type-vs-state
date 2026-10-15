"""Pool pyramidal cells across layers WITHOUT assuming a common slope.

Pooling raises n from 126 (L5 excitatory) to ~410 (all glutamatergic), but a
single common-slope estimator assumes the effect is the same in every layer.
If the effect is heterogeneous across classes (`06_per_pyramidal_class.py`),
a common-slope estimate averages unequal or opposite-signed effects.

Random-effects meta-analysis gets the power from pooling while allowing the
effect to differ by layer:

  * per-layer estimate      partial Spearman, within-type + technical covariates
  * fixed effects (FE)      inverse-variance, assumes one common slope
  * random effects (RE)     DerSimonian-Laird, allows between-layer variance tau²
  * Cochran Q, I²           is the assumption of homogeneity tenable at all?

Reported for Jun, Olfm1 and the composite state score. If Q is significant, the
RE estimate and the per-layer pattern are the answer and the FE number should
not be quoted.

Usage:  uv run python analysis/m1_patchseq/scripts/20_layer_meta_analysis.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)

from importlib import import_module

import numpy as np
import pandas as pd
from scipy import stats

_m3 = import_module("03_gene_level_rheobase")
from plasticity_common import load_ephys, load_meta  # noqa: E402

OUT = paths.results(__file__)
ENDPOINT = "Rheobase (pA)"
GENES = ["Jun", "Olfm1", "Junb", "Gadd45g", "Nptx2", "Fos"]
MIN_N = 20

# Layer strata parsed from the Scala "RNA type" labels.
LAYER_PATTERNS = {
    "L2/3 IT": r"^L2/3 IT",
    "L4/5 IT": r"^L4/5 IT",
    "L5 IT": r"^L5 IT",
    "L5 ET": r"^L5 ET",
    "L6 IT": r"^L6 IT",
    "L6 CT": r"^L6 CT",
    "L6b": r"^L6b",
    "L5/6 NP": r"^L5/6 NP",
}


def fisher_z(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def meta(rs: list[float], ns: list[int]) -> dict:
    """Fixed- and random-effects pooling of correlations via Fisher z."""
    ok = [(r, n) for r, n in zip(rs, ns) if np.isfinite(r) and n > 4]
    if len(ok) < 2:
        return {}
    z = np.array([fisher_z(r) for r, _ in ok])
    v = np.array([1.0 / (n - 3) for _, n in ok])
    w = 1 / v
    z_fe = float((w * z).sum() / w.sum())
    se_fe = float(np.sqrt(1 / w.sum()))
    Q = float((w * (z - z_fe) ** 2).sum())
    df = len(ok) - 1
    p_Q = float(stats.chi2.sf(Q, df))
    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    w_re = 1 / (v + tau2)
    z_re = float((w_re * z).sum() / w_re.sum())
    se_re = float(np.sqrt(1 / w_re.sum()))
    return {
        "k": len(ok), "n_total": int(sum(n for _, n in ok)),
        "rho_FE": float(np.tanh(z_fe)),
        "p_FE": float(2 * stats.norm.sf(abs(z_fe / se_fe))),
        "rho_RE": float(np.tanh(z_re)),
        "p_RE": float(2 * stats.norm.sf(abs(z_re / se_re))),
        "ci_RE_lo": float(np.tanh(z_re - 1.96 * se_re)),
        "ci_RE_hi": float(np.tanh(z_re + 1.96 * se_re)),
        "Q": Q, "df": df, "p_Q": p_Q, "I2": I2, "tau2": tau2,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta_tbl = load_meta()
    ephys = load_ephys()
    log_cpm, qc = _m3.load_full_logcpm(meta_tbl.index, use_introns=True)
    expr_all, _ = _m3.filter_genes(log_cpm, drop_channels=False)
    design_all = meta_tbl.loc[expr_all.index].join(qc, how="left")

    rt = design_all["RNA type"].astype(str)
    layer = pd.Series("other", index=design_all.index)
    for name, pat in LAYER_PATTERNS.items():
        layer[rt.str.match(pat, na=False)] = name
    design_all["layer"] = layer
    glut = design_all.index[layer != "other"]
    print(f"[pop] pyramidal cells assigned to a layer: {len(glut)}")
    print(design_all.loc[glut, "layer"].value_counts().to_string())

    y_all = ephys[ENDPOINT]
    per_rows, meta_rows = [], []

    for gene in GENES:
        if gene not in expr_all.columns:
            print(f"\n[skip] {gene}: filtered out of the matrix")
            continue
        rs, ns, labels = [], [], []
        for lay in LAYER_PATTERNS:
            idx = design_all.index[(design_all["layer"] == lay)]
            idx = expr_all.index.intersection(idx)
            y = y_all.reindex(idx)
            if y.notna().sum() < MIN_N:
                continue
            cov = design_all.loc[idx, _m3.COVARS].copy()
            sub_rt = design_all.loc[idx, "RNA type"]
            k = sub_rt.value_counts()[lambda s: s >= 8].index
            sub_rt = sub_rt.where(sub_rt.isin(k), "other")
            if sub_rt.nunique() > 1:
                cov = pd.concat([cov, pd.get_dummies(sub_rt, prefix="t",
                                                     drop_first=True).astype(float)],
                                axis=1)
            res = _m3.screen(expr_all.loc[idx, [gene]], y, covars=cov)
            r = float(res["rho"].iloc[0]); n = int(res["n"].iloc[0])
            p = float(res["p"].iloc[0])
            rs.append(r); ns.append(n); labels.append(lay)
            per_rows.append({"gene": gene, "layer": lay, "rho": r, "p": p, "n": n})
        m = meta(rs, ns)
        if m:
            m["gene"] = gene
            meta_rows.append(m)
            print(f"\n{'='*80}\n{gene}\n{'='*80}")
            print(f"  {'layer':10s} {'rho':>8s} {'n':>5s}")
            for lab, r, n in zip(labels, rs, ns):
                print(f"  {lab:10s} {r:+8.3f} {n:5d}")
            print(f"\n  FE (common slope) rho = {m['rho_FE']:+.3f}  p = {m['p_FE']:.4g}"
                  f"   [n = {m['n_total']}, k = {m['k']}]")
            print(f"  RE (heterogeneous) rho = {m['rho_RE']:+.3f}  p = {m['p_RE']:.4g}"
                  f"   95% CI [{m['ci_RE_lo']:+.3f}, {m['ci_RE_hi']:+.3f}]")
            print(f"  heterogeneity: Q = {m['Q']:.2f} (df {m['df']}), "
                  f"p = {m['p_Q']:.4g}, I² = {m['I2']:.0f}%, tau² = {m['tau2']:.4f}")
            if m["p_Q"] < 0.05:
                print("  -> HETEROGENEOUS: do not quote the FE estimate; the")
                print("     per-layer pattern is the result.")
            else:
                print("  -> homogeneous enough; pooling across layers is defensible"
                      " and buys real power.")

    pd.DataFrame(per_rows).to_csv(OUT / "S_layer_per_stratum.csv", index=False)
    pd.DataFrame(meta_rows).to_csv(OUT / "S_layer_meta.csv", index=False)

    print("\n" + "=" * 80)
    print("SUMMARY — does pooling across layers help?")
    print("=" * 80)
    M = pd.DataFrame(meta_rows)
    if len(M):
        print(M[["gene", "k", "n_total", "rho_FE", "p_FE", "rho_RE", "p_RE",
                 "I2", "p_Q"]].to_string(index=False,
                                         float_format=lambda v: f"{v:.4g}"))
    print(f"\n[write] {OUT/'S_layer_per_stratum.csv'}")
    print(f"[write] {OUT/'S_layer_meta.csv'}")


if __name__ == "__main__":
    main()
