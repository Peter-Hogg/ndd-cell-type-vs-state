"""Stratify by pyramidal (glutamatergic) class: is any effect class-specific?

The pooled within-type estimate in 05 is a *common* slope across all
glutamatergic types. If an effect exists in only one class it would be diluted
there. This stratifies instead: each class gets its own estimate, its own
detection floor, and its own positive control, and heterogeneity across classes
is tested formally with Cochran's Q on Fisher-z transformed correlations.

Within a class, transcriptomic type is still a covariate -- IT alone spans L2/3
to L6, so the class is not internally homogeneous.

Usage:  uv run python analysis/m1_patchseq/scripts/06_per_pyramidal_class.py
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

from plasticity_common import PRIMARY_EPHYS, load_ephys, load_meta

_m3 = import_module("03_gene_level_rheobase")
_m4 = import_module("04_ephys_wide_screen")
CHANNEL_PATTERNS, COVARS, MIN_DETECT_FRAC = (
    _m3.CHANNEL_PATTERNS, _m3.COVARS, _m3.MIN_DETECT_FRAC
)
load_full_logcpm, screen = _m3.load_full_logcpm, _m3.screen

RESULTS = paths.results(__file__)
GLUT_FAMILIES = ["IT", "ET", "CT", "NP"]
MIN_CLASS_N = 25
FOCUS = ["Jun", "Junb", "Gadd45g", "Fos", "Egr1", "Npas4", "Arc",
         "Nptx1", "Nptx2", "Kcnip1"]
PRIMARY_FEAT = "Rheobase (pA)"


def min_detectable_rho(n, alpha=0.05, power=0.80):
    if n is None or n < 6:
        return np.nan
    return float(np.tanh((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power))
                         / np.sqrt(n - 3)))


def cochran_q(rhos, ns):
    """Heterogeneity of correlations across strata (Fisher-z, inverse variance)."""
    ok = [(r, n) for r, n in zip(rhos, ns)
          if np.isfinite(r) and n is not None and n > 4 and abs(r) < 1]
    if len(ok) < 2:
        return np.nan, np.nan, np.nan, np.nan
    z = np.array([np.arctanh(r) for r, _ in ok])
    w = np.array([n - 3 for _, n in ok], dtype=float)
    zbar = np.sum(w * z) / np.sum(w)
    Q = float(np.sum(w * (z - zbar) ** 2))
    df = len(ok) - 1
    p = float(stats.chi2.sf(Q, df))
    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    return Q, df, p, I2


def build_covars(design: pd.DataFrame) -> pd.DataFrame:
    """Technical covariates plus type dummies for types with n>=8 in this stratum."""
    cov = design[COVARS].copy()
    rt = design["RNA type"]
    keep = rt.value_counts()[lambda s: s >= 8].index
    rt = rt.where(rt.isin(keep), "other")
    if rt.nunique() > 1:
        d = pd.get_dummies(rt, prefix="t", drop_first=True).astype(float)
        cov = pd.concat([cov, d], axis=1)
    return cov


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    glut = meta.index[meta["RNA family"].isin(GLUT_FAMILIES)]
    ephys = load_ephys()

    log_cpm, qc = load_full_logcpm(glut)
    g = log_cpm.columns
    drop = (g.str.startswith("ERCC-")
            | g.str.match(r"^(Gm\d|.*Rik$|LOC\d|A[A-Z]\d{6}|BC\d{6})")
            | ((log_cpm > 0).mean(axis=0) < MIN_DETECT_FRAC).to_numpy())
    expr_all = log_cpm.loc[:, ~drop]
    design_all = meta.loc[expr_all.index].join(qc, how="left")

    # ---- define strata: broad class, then finer type ---------------------
    strata: list[tuple[str, str, pd.Index]] = []
    for fam in GLUT_FAMILIES:
        idx = design_all.index[design_all["RNA family"] == fam]
        if len(idx) >= MIN_CLASS_N:
            strata.append((f"{fam} (class)", "class", idx))
        else:
            print(f"[skip] class {fam}: n={len(idx)} < {MIN_CLASS_N}")
    for rt, idx in design_all.groupby("RNA type", observed=True).groups.items():
        if len(idx) >= MIN_CLASS_N:
            strata.append((str(rt), "type", pd.Index(idx)))

    print("\n[strata]")
    for name, level, idx in strata:
        print(f"  {name:22s} {level:5s} n={len(idx):3d}  "
              f"detects |rho| >= {min_detectable_rho(len(idx)):.3f}")

    # ---- per stratum: focus genes vs rheobase + genome-wide screen -------
    focus_rows, screen_rows, hit_frames = [], [], []
    for name, level, idx in strata:
        expr = expr_all.loc[idx]
        # re-filter sparse genes inside the stratum
        expr = expr.loc[:, (expr > 0).mean(axis=0) >= MIN_DETECT_FRAC]
        design = design_all.loc[idx]
        cov = build_covars(design)
        is_channel = pd.Series(
            expr.columns.str.match("|".join(CHANNEL_PATTERNS)), index=expr.columns)

        y = ephys[PRIMARY_FEAT].reindex(idx)
        if y.notna().sum() < 20:
            continue
        res = screen(expr, y, covars=cov)
        res_i = res.set_index("gene")
        n_eff = int(res["n"].iloc[0])

        for gene in FOCUS:
            if gene in res_i.index:
                r = res_i.loc[gene]
                focus_rows.append({
                    "stratum": name, "level": level, "gene": gene,
                    "rho": float(r["rho"]), "p": float(r["p"]),
                    "q_genomewide": float(r["q"]), "n": n_eff,
                    "min_detectable_rho": min_detectable_rho(n_eff),
                })

        hits = res[res["q"] < 0.05].copy()
        hits["is_channel"] = hits["gene"].map(is_channel).fillna(False)
        top = res.reindex(res["rho"].abs().sort_values(ascending=False).index).iloc[0]
        screen_rows.append({
            "stratum": name, "level": level, "n": n_eff,
            "genes_tested": expr.shape[1],
            "min_detectable_rho": min_detectable_rho(n_eff),
            "hits_q05": len(hits),
            "hits_channel": int(hits["is_channel"].sum()),
            "best_gene": top["gene"], "best_rho": float(top["rho"]),
            "min_q": float(res["q"].min()),
        })
        if len(hits):
            h = hits.copy(); h["stratum"] = name
            hit_frames.append(h)

    F = pd.DataFrame(focus_rows)
    S = pd.DataFrame(screen_rows)
    F.to_csv(RESULTS / "J_per_class_focus.csv", index=False)
    S.to_csv(RESULTS / "J_per_class_screen.csv", index=False)

    # ---- report ---------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"Per-class genome-wide screen vs {PRIMARY_FEAT}")
    print("=" * 100)
    print(S.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    print("\n" + "=" * 100)
    print(f"Focus genes vs {PRIMARY_FEAT}, by class  (rho; * = p<0.05 uncorrected)")
    print("=" * 100)
    for level in ("class", "type"):
        sub = F[F["level"] == level]
        if not len(sub):
            continue
        piv = sub.pivot_table(index="gene", columns="stratum", values="rho")
        pv = sub.pivot_table(index="gene", columns="stratum", values="p")
        piv = piv.reindex([g for g in FOCUS if g in piv.index])
        ns = sub.groupby("stratum")["n"].max()
        mdr = sub.groupby("stratum")["min_detectable_rho"].max()
        print(f"\n--- by {level} ---")
        hdr = "gene      " + "".join(f"{c[:14]:>16s}" for c in piv.columns)
        print(hdr)
        print("n         " + "".join(f"{('n='+str(int(ns[c]))):>16s}" for c in piv.columns))
        print("floor     " + "".join(f"{('>='+format(mdr[c],'.2f')):>16s}" for c in piv.columns))
        print("-" * len(hdr))
        for gene in piv.index:
            cells = ""
            for c in piv.columns:
                r, p = piv.loc[gene, c], pv.loc[gene, c]
                s = "n/a" if not np.isfinite(r) else f"{r:+.3f}{'*' if p < 0.05 else ' '}"
                cells += f"{s:>16s}"
            print(f"{gene:10s}{cells}")

    print("\n" + "=" * 100)
    print("Heterogeneity across the three broad classes (Cochran's Q on Fisher-z)")
    print("=" * 100)
    cls = F[F["level"] == "class"]
    het = []
    for gene in FOCUS:
        s = cls[cls["gene"] == gene]
        if len(s) < 2:
            continue
        Q, df, p, I2 = cochran_q(s["rho"].tolist(), s["n"].tolist())
        het.append({"gene": gene, "Q": Q, "df": df, "p_heterogeneity": p,
                    "I2_pct": I2,
                    "rho_range": f"{s['rho'].min():+.3f} .. {s['rho'].max():+.3f}"})
    H = pd.DataFrame(het)
    H.to_csv(RESULTS / "J_per_class_heterogeneity.csv", index=False)
    print(H.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n  p_heterogeneity < 0.05 would indicate a genuinely class-specific")
    print("  effect. Non-significant Q with a wide rho range means the strata are")
    print("  simply too small to distinguish -- not evidence of homogeneity.")

    if hit_frames:
        HH = pd.concat(hit_frames, ignore_index=True)
        HH.to_csv(RESULTS / "J_per_class_hits.csv", index=False)
        print(f"\n--- per-class hits at q<0.05 vs {PRIMARY_FEAT} ---")
        print(HH.reindex(HH["rho"].abs().sort_values(ascending=False).index)
              .head(25)[["stratum", "gene", "rho", "q", "is_channel"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    else:
        print(f"\nNo gene reaches q<0.05 vs {PRIMARY_FEAT} in ANY single class.")

    print(f"\n[write] {RESULTS/'J_per_class_focus.csv'}")
    print(f"[write] {RESULTS/'J_per_class_screen.csv'}")
    print(f"[write] {RESULTS/'J_per_class_heterogeneity.csv'}")


if __name__ == "__main__":
    main()
