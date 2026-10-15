"""Test IEG associations with rheobase in macaque L5 Patch-seq cells (Dembrow et al. 2024).

Population: QC-passed L5 cells with expression and threshold current; L5 ET =
M1-correlation subclass label containing "L5 ET". Statistics are partial Spearman
correlations (ranks residualised on rank-transformed covariates) with Fisher-z CIs.
Covariates: log total reads, genes detected, and cortical area / slice prep /
species indicators where the minority level has >= 5 cells.

  H1   primary   JUN vs rheobase in L5 ET, predicted negative, alpha=0.05
  H1b  contrast  ET vs non-ET Fisher-z difference, alpha=0.05
  S1-S9          secondaries, Bonferroni alpha=0.0056 (S6-S8 predicted NULL,
                 S9 = Kcnip1 positive control)
  H2             mouse 150-gene module score, sign test, alpha=0.05

Endpoint mapping: mouse "Rheobase (pA)" <-> macaque
"threshold_current".

Usage:  uv run python analysis/macaque_patchseq/scripts/07_dembrow_confirmatory.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)


import numpy as np
import pandas as pd
from scipy import stats

D = paths.REPO / "ReplicationData" / "Dembrow2024"
RESULTS = paths.results(__file__)
ENDPOINT = "threshold_current"
ALPHA_PRIMARY = 0.05
N_SECONDARY = 9
ALPHA_SECONDARY = 0.05 / N_SECONDARY

PRIMARY_GENE = "JUN"
SECONDARIES = [
    ("S1", "JUNB", "negative"), ("S2", "GADD45G", "negative"),
    ("S3", "FOS", "negative"), ("S4", "EGR1", "negative"),
    ("S5", "NPAS4", "negative"), ("S6", "ARC", "null"),
    ("S7", "NPTX2", "null"), ("S8", "NPTX1", "null"),
    ("S9", "KCNIP1", "negative (positive control)"),
]


def fisher_ci(r: float, n: int, conf=0.95) -> tuple[float, float]:
    if not np.isfinite(r) or n < 5 or abs(r) >= 1:
        return (np.nan, np.nan)
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    c = stats.norm.ppf(1 - (1 - conf) / 2)
    return (float(np.tanh(z - c * se)), float(np.tanh(z + c * se)))


def partial_spearman(x, y, C: pd.DataFrame | None):
    """Spearman of x vs y after residualising both on rank-transformed covariates."""
    df = pd.concat([pd.Series(np.asarray(x), name="x", index=y.index),
                    y.rename("y")], axis=1)
    if C is not None and C.shape[1]:
        df = pd.concat([df, C], axis=1)
    df = df.dropna()
    n = len(df)
    ncov = 0 if C is None else C.shape[1]
    if n < ncov + 6:
        return np.nan, np.nan, n, ncov

    if ncov:
        M = np.column_stack([np.ones(n)] +
                            [stats.rankdata(df[c]) for c in C.columns])
        keep = [0] + [i for i in range(1, M.shape[1])
                      if np.std(M[:, i]) > 1e-9]
        M = M[:, keep]
        ncov = M.shape[1] - 1

        def resid(col):
            v = stats.rankdata(df[col])
            b, *_ = np.linalg.lstsq(M, v, rcond=None)
            return v - M @ b
        rx, ry = resid("x"), resid("y")
    else:
        rx, ry = stats.rankdata(df["x"]), stats.rankdata(df["y"])

    if np.std(rx) < 1e-9 or np.std(ry) < 1e-9:
        return np.nan, np.nan, n, ncov
    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = n - 2 - ncov
    if dof <= 0 or abs(r) >= 1:
        return r, np.nan, n, ncov
    t = r * np.sqrt(dof / (1 - r**2))
    return r, float(2 * stats.t.sf(abs(t), dof)), n, ncov


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ft = pd.read_feather(D / "macaque_L5_patchseq.feather")
    phys = pd.read_csv(D / "L5_physiology_121223.csv", low_memory=False)
    cells = pd.read_csv(D / "cell_list_L5.csv", low_memory=False)

    K = "container"
    ft[K] = ft["patched_cell_container_label"].astype(str).str.strip()
    phys[K] = phys["sample_id"].astype(str).str.strip()
    cells[K] = cells["patched_cell_container"].astype(str).str.strip()

    meta = (phys[[K, ENDPOINT, "Brain_Region", "Species", "slice_type", "DIV"]]
            .merge(cells[[K, "subclass_Corr_label_M1", "cluster_Corr_label_M1",
                          "Cortical_area", "Genes.Detected", "QC"]],
                   on=K, how="inner"))
    meta = meta[meta[ENDPOINT].notna() & (meta["QC"] == True)]
    meta = meta.drop_duplicates(subset=K).set_index(K)

    expr = ft.set_index(K)
    reads = expr["total_reads_label"]
    expr = expr.drop(columns=["sample_id", "patched_cell_container_label",
                              "total_reads_label"])
    # Feather is CPM (rows sum to 1e6). log1p to match the mouse pipeline.
    expr = np.log1p(expr)

    idx = meta.index.intersection(expr.index)
    meta, expr = meta.loc[idx], expr.loc[idx]
    meta["log_reads"] = np.log10(reads.reindex(idx))
    meta["is_ET"] = meta["subclass_Corr_label_M1"].astype(str).str.contains("L5 ET")
    return meta, expr, cells


def covariates(meta: pd.DataFrame) -> pd.DataFrame:
    """Covariates: depth, genes detected, area, ex vivo prep."""
    C = pd.DataFrame(index=meta.index)
    C["log_reads"] = meta["log_reads"]
    C["genes_detected"] = meta["Genes.Detected"]
    for col, name in (("Cortical_area", "area"), ("slice_type", "prep"),
                      ("Species", "species")):
        vc = meta[col].value_counts()
        if len(vc) > 1 and vc.iloc[1] >= 5:
            C[name] = (meta[col] == vc.index[0]).astype(float)
    return C


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    meta, expr, _ = load()
    et = meta.index[meta["is_ET"]]
    non = meta.index[~meta["is_ET"]]

    print("=" * 78)
    print("REPLICATION — Dembrow et al. 2024 (macaque)")
    print("partial Spearman, ranks residualised on covariates")
    print("=" * 78)
    print(f"paired cells: {len(meta)}   L5 ET: {len(et)}   non-ET: {len(non)}")
    print(f"endpoint: {ENDPOINT} (= mouse Rheobase (pA))")
    print(f"  area:  {meta['Cortical_area'].value_counts().to_dict()}")
    print(f"  prep:  {meta['slice_type'].value_counts().to_dict()}")
    print(f"  L5 ET prep: {meta.loc[et,'slice_type'].value_counts().to_dict()}")

    C = covariates(meta)
    print(f"  covariates: {list(C.columns)}")

    y_all = meta[ENDPOINT]
    rows = []

    def run(label, gene, pred, idx, Cx):
        if gene not in expr.columns:
            print(f"  {label} {gene}: ABSENT from dataset")
            return None
        r, p, n, ncov = partial_spearman(
            expr.loc[idx, gene], y_all.loc[idx], Cx.loc[idx])
        lo, hi = fisher_ci(r, n)
        rows.append({"test": label, "gene": gene, "prediction": pred,
                     "rho": r, "p": p, "n": n, "n_covar": ncov,
                     "ci_lo": lo, "ci_hi": hi})
        return r, p, n, lo, hi

    # ---------------- H1 : primary ----------------
    print("\n" + "-" * 78)
    print("H1 (PRIMARY, alpha=0.05, one test): JUN vs rheobase in L5 ET")
    print("  predicted: NEGATIVE   mouse estimate: -0.386 (class) / -0.539 (type)")
    print("-" * 78)
    out = run("H1", PRIMARY_GENE, "negative", et, C)
    if out:
        r, p, n, lo, hi = out
        print(f"  rho = {r:+.3f}   p = {p:.4g}   n = {n}   95% CI [{lo:+.3f}, {hi:+.3f}]")
        sig = p < ALPHA_PRIMARY
        right = sig and r < 0
        print(f"\n  significant at alpha=0.05 : {sig}")
        print(f"  in predicted direction    : {r < 0}")
        print(f"  >>> H1 {'REPLICATED' if right else 'NOT replicated'}")
        if not right:
            print("      (distinguish 'refuted' from 'inconclusive'")
            print("       using the CI and the S9 positive control below)")

    # ---------------- H1b : class contrast ----------------
    print("\n" + "-" * 78)
    print("H1b (contrast, alpha=0.05): JUN effect ET vs non-ET")
    print("-" * 78)
    r_et, p_et, n_et, _ = partial_spearman(expr.loc[et, PRIMARY_GENE],
                                           y_all.loc[et], C.loc[et])
    r_no, p_no, n_no, _ = partial_spearman(expr.loc[non, PRIMARY_GENE],
                                            y_all.loc[non], C.loc[non])
    print(f"  L5 ET  : rho = {r_et:+.3f}  p = {p_et:.4g}  n = {n_et}")
    print(f"  non-ET : rho = {r_no:+.3f}  p = {p_no:.4g}  n = {n_no}")
    if all(np.isfinite([r_et, r_no])) and n_et > 4 and n_no > 4:
        dz = np.arctanh(r_et) - np.arctanh(r_no)
        se = np.sqrt(1 / (n_et - 3) + 1 / (n_no - 3))
        zst = dz / se
        p_c = float(2 * stats.norm.sf(abs(zst)))
        print(f"  Fisher-z difference = {dz:+.3f}  SE = {se:.3f}  "
              f"z = {zst:+.2f}  p = {p_c:.4g}")
        print(f"  >>> H1b {'SUPPORTED' if (p_c < 0.05 and r_et < r_no) else 'NOT supported'}")
        rows.append({"test": "H1b_contrast", "gene": PRIMARY_GENE,
                     "prediction": "ET more negative than non-ET",
                     "rho": dz, "p": p_c, "n": n_et + n_no, "n_covar": C.shape[1],
                     "ci_lo": np.nan, "ci_hi": np.nan})

    # ---------------- secondaries ----------------
    print("\n" + "-" * 78)
    print(f"SECONDARIES in L5 ET (Bonferroni alpha = {ALPHA_SECONDARY:.4f})")
    print("-" * 78)
    print(f"  {'test':5s} {'gene':9s} {'predicted':28s} {'rho':>7s} {'p':>9s} "
          f"{'95% CI':>18s}  verdict")
    for label, gene, pred in SECONDARIES:
        o = run(label, gene, pred, et, C)
        if not o:
            continue
        r, p, n, lo, hi = o
        if pred == "null":
            v = "null as predicted" if p >= ALPHA_SECONDARY else "NOT null - refutes"
        else:
            v = "significant" if (p < ALPHA_SECONDARY and r < 0) else "n.s."
        print(f"  {label:5s} {gene:9s} {pred:28s} {r:+7.3f} {p:9.4g} "
              f"[{lo:+.2f},{hi:+.2f}]  {v}")

    # ---------------- H2 : module ----------------
    print("\n" + "-" * 78)
    print("H2 (alpha=0.05): mouse 150-gene module score vs rheobase, L5 ET")
    print("  prediction is SIGN ONLY (mouse module was selected on the mouse data)")
    print("-" * 78)
    seeds = pd.read_csv(paths.M1 / "results" / "F_module_seeds.csv")["gene"].astype(str).tolist()
    mapped = {s: s.upper() for s in seeds if s.upper() in expr.columns}
    print(f"  mouse seeds: {len(seeds)}  ->  1:1 ortholog match: {len(mapped)} "
          f"({len(seeds)-len(mapped)} dropped, not substituted)")
    if len(mapped) >= 20:
        sub = expr.loc[:, sorted(set(mapped.values()))]
        z = (sub - sub.mean()) / sub.std(ddof=1)
        mod = z.mean(axis=1)
        r, p, n, _ = partial_spearman(mod.loc[et], y_all.loc[et], C.loc[et])
        lo, hi = fisher_ci(r, n)
        print(f"  rho = {r:+.3f}   p = {p:.4g}   n = {n}   95% CI [{lo:+.3f},{hi:+.3f}]")
        print(f"  >>> H2 {'SUPPORTED' if (p < 0.05 and r < 0) else 'NOT supported'} "
              f"(sign {'negative as predicted' if r < 0 else 'POSITIVE, opposite'})")
        rows.append({"test": "H2_module", "gene": f"module({len(mapped)} genes)",
                     "prediction": "negative", "rho": r, "p": p, "n": n,
                     "n_covar": C.shape[1], "ci_lo": lo, "ci_hi": hi})

    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "K_dembrow_confirmatory.csv", index=False)
    print(f"\n[write] {RESULTS/'K_dembrow_confirmatory.csv'}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(R.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
