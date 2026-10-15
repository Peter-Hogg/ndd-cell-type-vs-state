"""Test IEG and ASD-gene associations with rheobase in mouse V1 L5 ET cells (Sorensen et al. 2026).

Population: L5 ET cells (inferred_met_type starting "L5 ET") with rheobase and a
salmon-quantified transcriptome (10_v1_quantify_l5et.py).

FAMILY A — activity state, alpha 0.05
  A1  Jun vs rheobase                              negative, alpha 0.025
  A2  IEG_TF category, set-level, permutation null negative, alpha 0.025
  A3  150-gene mouse module score, SIGN only       negative, descriptive

FAMILY B — non-state excitability axis (ASD-linked genes), alpha 0.05
  B1  Jarid2 vs rheobase                           positive, alpha 0.025
  B2  Gphn vs rheobase                             positive, alpha 0.025

Gene tests are partial Spearman correlations (rank residuals on the covariates);
set tests average per-gene partial correlations over genes detected in >= 25% of
cells against a within-met-type permutation null (5000 permutations).

Zero-inflation rule, from the Jun detection rate D alone (never rheobase):
D >= 0.50 continuous test as specified; 0.30 <= D < 0.50 Spearman plus hurdle
co-primary; D < 0.30 continuous test uninformative.

Covariates: log library size, genes detected, met-type dummies (types with < 8
cells pooled), top50_share. No hold-time covariate is available in V1.

Usage:
  uv run python analysis/v1_patchseq/scripts/28_v1_confirmatory.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)

import re
from importlib import import_module

import numpy as np
import pandas as pd
from scipy import stats

_m22 = import_module("22_repairs_excitatory")
_m23 = import_module("23_functional_categories_by_layer")

ROOT = paths.REPO
V1 = ROOT / "ReplicationData" / "MouseV1_Sorensen2026"
OUT = paths.results(__file__)
N_PERM = 5000
RNG = np.random.default_rng(0)


def load_v1():
    parts = sorted(V1.glob("quant/*.parquet"))
    print(f"[load] {len(parts)} quantified cells")
    frames = {}
    for p in parts:
        g = pd.read_parquet(p, columns=["gene", "NumReads"])
        frames[p.stem] = g.set_index("gene")["NumReads"]
    M = pd.DataFrame(frames).T           # cells x genes, estimated counts
    M.index.name = "specimen_id"
    lib = M.sum(axis=1)
    cpm = M.div(lib, axis=0) * 1e6
    top50 = np.sort(cpm.to_numpy(), axis=1)[:, ::-1][:, :50].sum(1) / 1e6
    qc = pd.DataFrame({"log_libsize": np.log10(lib.replace(0, np.nan)),
                       "n_detected": (M > 0).sum(axis=1),
                       "top50_share": top50}, index=M.index)
    return np.log1p(cpm), M, qc


def main() -> None:
    log_cpm, counts, qc = load_v1()
    log_cpm.index = log_cpm.index.astype(str)
    qc.index = qc.index.astype(str)
    counts.index = counts.index.astype(str)

    phys = pd.read_csv(V1 / "exc_mMET_ephys_features.csv")
    phys["specimen_id"] = phys["specimen_id"].astype(str)
    phys = phys.set_index("specimen_id")
    met = pd.read_csv(V1 / "inferred_met_types.csv").rename(
        columns={"Unnamed: 0": "specimen_id"})
    met["specimen_id"] = met["specimen_id"].astype(str)
    met = met.set_index("specimen_id")

    d = qc.join(met[["inferred_met_type", "t_type"]], how="inner")
    d = d.join(phys[["rheobase_i"]], how="inner")
    d = d[d["rheobase_i"].notna()]
    d = d[d["inferred_met_type"].astype(str).str.startswith("L5 ET")]
    idx = log_cpm.index.intersection(d.index)
    d = d.loc[idx]
    n = len(idx)
    print(f"[pop] L5 ET cells with rheobase and transcriptome: {n}")
    print(d["inferred_met_type"].value_counts().to_string())
    print(f"[power] floor |rho| >= {np.tanh(2.8016/np.sqrt(n-3)):.3f} at 80%\n")

    # ---- zero-inflation rule (exposure only, never touches rheobase) ------
    jun_counts = counts.loc[idx, "Jun"] if "Jun" in counts.columns else None
    D = float((jun_counts > 0).mean()) if jun_counts is not None else np.nan
    med = float(jun_counts[jun_counts > 0].median()) if jun_counts is not None else np.nan
    print("=" * 78)
    print("ZERO-INFLATION RULE, computed from Jun counts alone")
    print("=" * 78)
    print(f"  Jun detection rate D = {D:.3f}   median counts where detected = {med:.1f}")
    branch = ("D >= 0.50 : primary as specified" if D >= 0.50 else
              "0.30 <= D < 0.50 : Spearman primary + hurdle co-primary" if D >= 0.30
              else "D < 0.30 : continuous test UNINFORMATIVE a priori")
    print(f"  branch -> {branch}\n")

    # ---- design ----------------------------------------------------------
    mt = d["inferred_met_type"].astype(str)
    keep = mt.value_counts()[lambda s: s >= 8].index
    mt = mt.where(mt.isin(keep), "other")
    dums = pd.get_dummies(mt, prefix="m", drop_first=True).astype(float)
    cov = d[["log_libsize", "n_detected", "top50_share"]]
    C = _m22.build_design(cov, dums, n)
    y = _m22.rank_resid(d["rheobase_i"].to_numpy(), C)
    yc = (y - y.mean()) / y.std()
    groups = mt.to_numpy()

    def gene_rho(gene):
        if gene not in log_cpm.columns:
            return None
        g = _m22.rank_resid(log_cpm.loc[idx, gene].to_numpy(), C)
        a = g - g.mean(); b = y - y.mean()
        den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        if den == 0:
            return None
        r = float((a * b).sum() / den)
        dof = n - 2 - (C.shape[1] - 1)
        t = r * np.sqrt(dof / max(1e-12, 1 - r ** 2))
        se = 1 / np.sqrt(n - 3)
        z = np.arctanh(np.clip(r, -0.999, 0.999))
        return {"rho": r, "p": float(2 * stats.t.sf(abs(t), dof)),
                "ci_lo": float(np.tanh(z - 1.96 * se)),
                "ci_hi": float(np.tanh(z + 1.96 * se)), "n": n}

    perms = []
    for _ in range(N_PERM):
        yp = y.copy()
        for gg in np.unique(groups):
            m = groups == gg
            yp[m] = RNG.permutation(yp[m])
        perms.append(yp - yp.mean())
    P = np.column_stack(perms)
    P /= np.where(P.std(0) > 0, P.std(0), np.nan)

    def set_test(genes, label):
        cols = [g for g in genes if g in log_cpm.columns]
        E = log_cpm.loc[idx, cols]
        E = E.loc[:, (E > 0).mean(axis=0) >= 0.25]
        if E.shape[1] < 5:
            return None
        Er = np.column_stack([_m22.rank_resid(E[g].to_numpy(), C) for g in E.columns])
        Er -= Er.mean(0)
        sd = Er.std(0)
        Er /= np.where(sd > 0, sd, np.nan)
        obs = float(np.nanmean((Er * yc[:, None]).sum(0) / (n - 1)))
        nulls = np.nanmean((Er.T @ P) / (n - 1), axis=0)
        z = (obs - nulls.mean()) / nulls.std(ddof=1)
        return {"label": label, "n_genes": E.shape[1], "mean_rho": obs,
                "null_mean": float(nulls.mean()), "null_sd": float(nulls.std(ddof=1)),
                "z": z, "p_one_sided_neg": float((nulls <= obs).mean())}

    rows = []
    print("=" * 78)
    print("FAMILY A — ACTIVITY STATE")
    print("=" * 78)
    a1 = gene_rho("Jun")
    if a1:
        sig = a1["p"] < 0.025 and a1["rho"] < 0
        print(f"  A1  Jun vs rheobase        rho = {a1['rho']:+.3f}  p = {a1['p']:.4g}"
              f"   95% CI [{a1['ci_lo']:+.3f}, {a1['ci_hi']:+.3f}]")
        print(f"      predicted negative, alpha 0.025  ->  "
              f"{'REPLICATED' if sig else 'NOT replicated'}")
        rows.append({"test": "A1_Jun", **a1})

    a2 = set_test(_m23.CATEGORIES["IEG_TF"], "IEG_TF category")
    if a2:
        sig = a2["p_one_sided_neg"] < 0.025 and a2["mean_rho"] < 0
        print(f"\n  A2  IEG_TF set-level ({a2['n_genes']} genes)  mean rho = "
              f"{a2['mean_rho']:+.4f}")
        print(f"      permutation null {a2['null_mean']:+.4f} ± {a2['null_sd']:.4f}"
              f"   z = {a2['z']:+.2f}   p = {a2['p_one_sided_neg']:.4f}")
        print(f"      predicted negative, alpha 0.025  ->  "
              f"{'REPLICATED' if sig else 'NOT replicated'}")
        rows.append({"test": "A2_IEG_TF_set", **a2})

    seeds = pd.read_csv(paths.M1 / "results" / "F_module_seeds.csv")["gene"].astype(str).tolist()
    a3 = set_test(seeds, "module (150 seeds)")
    if a3:
        print(f"\n  A3  module score ({a3['n_genes']} of {len(seeds)} seeds present)"
              f"  mean rho = {a3['mean_rho']:+.4f}   z = {a3['z']:+.2f}"
              f"   p = {a3['p_one_sided_neg']:.4f}")
        print(f"      sign predicted negative  ->  "
              f"{'sign correct' if a3['mean_rho'] < 0 else 'SIGN OPPOSITE'}")
        rows.append({"test": "A3_module_sign", **a3})

    print("\n" + "=" * 78)
    print("FAMILY B — NON-STATE EXCITABILITY AXIS (ASD-linked genes)")
    print("=" * 78)
    for tid, gene in (("B1", "Jarid2"), ("B2", "Gphn")):
        r = gene_rho(gene)
        if r:
            sig = r["p"] < 0.025 and r["rho"] > 0
            print(f"  {tid}  {gene:7s} vs rheobase   rho = {r['rho']:+.3f}"
                  f"  p = {r['p']:.4g}   95% CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")
            print(f"      predicted positive, alpha 0.025  ->  "
                  f"{'REPLICATED' if sig else 'NOT replicated'}")
            rows.append({"test": f"{tid}_{gene}", **r})

    R = pd.DataFrame(rows)
    R.insert(0, "jun_detection_rate_D", D)
    R.to_csv(OUT / "AA_v1_confirmatory.csv", index=False)
    print(f"\n[write] {OUT/'AA_v1_confirmatory.csv'}")


if __name__ == "__main__":
    main()
