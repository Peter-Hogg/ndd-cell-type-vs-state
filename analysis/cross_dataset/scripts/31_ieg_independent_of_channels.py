"""Is the activity-state effect INDEPENDENT of ion-channel expression?

The IEG_TF category (all transcription factors, zero channels) is tested against
rheobase, which is set biophysically by channels. Two readings:

  (i)  activity TFs act THROUGH channel expression -> the effect is a channel
       effect wearing a TF label, and "activity state predicts excitability" is
       mechanistically downstream but not independent
  (ii) the TF signal is independent of measured channel expression -> activity
       state contributes something channels do not explain

Tested by adding principal components of ion-channel expression as covariates and
seeing whether the IEG index survives. Channels are also tested on their own, so
the two contributions can be compared side by side rather than asserted.

Run in mouse M1 L5 exc (n=126) and mouse V1 L5 ET (n=436). EXPLORATORY for V1.

Usage:
  uv run python analysis/cross_dataset/scripts/31_ieg_independent_of_channels.py
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

_m3 = import_module("03_gene_level_rheobase")
_m22 = import_module("22_repairs_excitatory")
_m23 = import_module("23_functional_categories_by_layer")
_m28 = import_module("28_v1_confirmatory")
from plasticity_common import load_ephys, load_meta  # noqa: E402

ROOT = paths.REPO
V1D = ROOT / "ReplicationData" / "MouseV1_Sorensen2026"
OUT = paths.results(__file__)
CH = "|".join(_m3.CHANNEL_PATTERNS)
IEG = _m23.CATEGORIES["IEG_TF"]
N_PC = 5


def index_score(expr, cols):
    cols = [c for c in cols if c in expr.columns]   # some TFs absent per cohort
    E = expr[cols]
    E = E.loc[:, (E > 0).mean(axis=0) >= 0.25]
    Z = (E - E.mean()) / E.std(ddof=1)
    return Z.mean(axis=1).to_numpy(), E.shape[1]


def index_score_filtered(expr_norm, expr_raw, cols):
    """Index over `cols`, with the detection filter taken from RAW expression.

    Critical for housekeeping-normalised input: those values are log-RATIOS and
    are negative for any gene below the reference, so a `(value > 0)` filter
    applied to them silently drops most of the panel rather than dropping
    undetected genes. The filter must come from the raw matrix.
    """
    cols = [c for c in cols if c in expr_norm.columns and c in expr_raw.columns]
    keep = [c for c in cols if (expr_raw[c] > 0).mean() >= 0.25]
    if len(keep) < 5:
        return None, len(keep)
    E = expr_norm[keep]
    Z = (E - E.mean()) / E.std(ddof=1)
    return Z.mean(axis=1).to_numpy(), len(keep)


def chan_pcs(expr, n_pc=N_PC):
    cols = [c for c in expr.columns if re.match(CH, c)]
    E = expr[cols]
    E = E.loc[:, (E > 0).mean(axis=0) >= 0.25]
    Z = np.array(((E - E.mean()) / E.std(ddof=1)).fillna(0.0), dtype=float)
    Z -= Z.mean(0)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    return U[:, :n_pc], E.shape[1], var[:n_pc]


def partial(a, b, C, n):
    ar = _m22.rank_resid(a, C); br = _m22.rank_resid(b, C)
    ar -= ar.mean(); br -= br.mean()
    den = np.sqrt((ar ** 2).sum() * (br ** 2).sum())
    if den == 0:
        return np.nan, np.nan
    r = float((ar * br).sum() / den)
    dof = n - 2 - (C.shape[1] - 1)
    t = r * np.sqrt(dof / max(1e-12, 1 - r ** 2))
    return r, float(2 * stats.t.sf(abs(t), dof))


def run(expr, y, base_cov, dums, n, label):
    C0 = _m22.build_design(base_cov, dums, n)
    pcs, n_chan, var = chan_pcs(expr)
    cov2 = base_cov.copy()
    for i in range(pcs.shape[1]):
        cov2[f"chanPC{i+1}"] = pcs[:, i]
    C1 = _m22.build_design(cov2, dums, n)

    ieg, n_ieg = index_score(expr, IEG)
    chan_idx = pcs[:, 0]

    print(f"\n{'='*78}\n{label}   n={n}\n{'='*78}")
    print(f"  IEG index from {n_ieg} TFs (0 channels); channel PCs from "
          f"{n_chan} channel genes, PC1-{N_PC} explain "
          f"{100*var.sum():.0f}% of channel variance")

    r0, p0 = partial(ieg, y, C0, n)
    r1, p1 = partial(ieg, y, C1, n)
    rc, pc = partial(chan_idx, y, C0, n)
    print(f"\n  IEG index vs rheobase")
    print(f"    baseline                     rho = {r0:+.4f}  p = {p0:.4g}")
    print(f"    + {N_PC} ion-channel PCs           rho = {r1:+.4f}  p = {p1:.4g}"
          f"    <-- decisive")
    change = 100 * (1 - abs(r1) / abs(r0)) if r0 else np.nan
    print(f"    attenuation {change:+.0f}%")
    print(f"\n  for comparison, channel PC1 vs rheobase   rho = {rc:+.4f}  p = {pc:.4g}")
    return {"cohort": label, "n": n, "n_ieg": n_ieg, "n_chan": n_chan,
            "ieg_base": r0, "ieg_base_p": p0,
            "ieg_chanadj": r1, "ieg_chanadj_p": p1,
            "attenuation_pct": change, "chanPC1": rc, "chanPC1_p": pc}


def main() -> None:
    rows = []
    # ---- mouse M1 ----
    meta = load_meta(); ephys = load_ephys()
    log_cpm, qc = _m3.load_full_logcpm(meta.index, use_introns=True)
    expr, _ = _m3.filter_genes(log_cpm, drop_channels=False)  # keep channels!
    d0 = meta.loc[expr.index].join(qc, how="left")
    rt = d0["RNA type"].astype(str)
    l5 = d0.index[rt.str.match(r"^L5 (IT|ET)_\d+$", na=False)]
    idx = expr.index.intersection(l5)
    idx = idx[ephys["Rheobase (pA)"].reindex(idx).notna().values]
    d = d0.loc[idx]; n = len(idx)
    s = d["RNA type"]; k = s.value_counts()[lambda x: x >= 8].index
    s = s.where(s.isin(k), "other")
    rows.append(run(expr.loc[idx], ephys["Rheobase (pA)"].reindex(idx).to_numpy(),
                    d[_m3.COVARS],
                    pd.get_dummies(s, prefix="t", drop_first=True).astype(float),
                    n, "mouse M1 L5 exc"))

    # ---- mouse V1 (EXPLORATORY) ----
    lc, counts, q2 = _m28.load_v1()
    lc.index = lc.index.astype(str); q2.index = q2.index.astype(str)
    phys = pd.read_csv(V1D / "exc_mMET_ephys_features.csv")
    phys["specimen_id"] = phys["specimen_id"].astype(str); phys = phys.set_index("specimen_id")
    met = pd.read_csv(V1D / "inferred_met_types.csv").rename(columns={"Unnamed: 0": "specimen_id"})
    met["specimen_id"] = met["specimen_id"].astype(str); met = met.set_index("specimen_id")
    dv = q2.join(met[["inferred_met_type"]], how="inner").join(phys[["rheobase_i"]], how="inner")
    dv = dv[dv["rheobase_i"].notna()]
    dv = dv[dv["inferred_met_type"].astype(str).str.startswith("L5 ET")]
    iv = lc.index.intersection(dv.index); dv = dv.loc[iv]; n2 = len(iv)
    mt = dv["inferred_met_type"].astype(str)
    kk = mt.value_counts()[lambda x: x >= 8].index
    mt = mt.where(mt.isin(kk), "other")
    rows.append(run(lc.loc[iv], dv["rheobase_i"].to_numpy(),
                    dv[["log_libsize", "n_detected", "top50_share"]],
                    pd.get_dummies(mt, prefix="m", drop_first=True).astype(float),
                    n2, "mouse V1 L5 ET  (EXPLORATORY)"))

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "AD_ieg_vs_channels.csv", index=False)
    print(f"\n[write] {OUT/'AD_ieg_vs_channels.csv'}")
    print("\nIf the IEG index survives channel adjustment in both cohorts, activity")
    print("state contributes something measured channel expression does not explain.")


if __name__ == "__main__":
    main()
