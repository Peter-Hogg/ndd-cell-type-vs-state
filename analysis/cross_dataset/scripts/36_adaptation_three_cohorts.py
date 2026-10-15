"""Does the activity-index / firing-adaptation link appear in all three cohorts?

Tests the housekeeping-normalised IEG_TF index against firing-adaptation measures
(cf. `analysis/m1_patchseq/scripts/35_hknorm_panels_all_features.py` for M1)
in each cohort, as a cross-cohort consistency check.

Cohorts and their adaptation measures:
  mouse M1   ISI average adaptation index, ISI adaptation index,
             spike frequency adaptation, ISI Fano factor, ISI CV
  mouse V1   adapt_long_square (averaged over current steps), isi_cv_long_square
  macaque    SFA (spike-frequency accommodation), CV, gain_CV

SIGN CALIBRATION IS ESSENTIAL and is done, not assumed. Adaptation indices use
inconsistent conventions between labs -- some increase with adaptation, others are
ratios that decrease. Each measure is calibrated against **spike count**, whose
direction is unambiguous: a cell that adapts strongly fires fewer spikes during a
sustained step. Every reported rho is then flipped so that

    positive rho  =  more activity-gene expression, MORE adaptation

Without this step a genuine replication could appear as a failure.

EXPLORATORY in all three cohorts; cross-cohort agreement is the quantity of
interest, not any single p-value. Macaque data: Dembrow et al. 2024.

Usage:
  uv run python analysis/cross_dataset/scripts/36_adaptation_three_cohorts.py
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
_m20 = import_module("20_layer_meta_analysis")
_m22 = import_module("22_repairs_excitatory")
_m23 = import_module("23_functional_categories_by_layer")
_m28 = import_module("28_v1_confirmatory")
_m31 = import_module("31_ieg_independent_of_channels")
from plasticity_common import load_ephys, load_meta  # noqa: E402

ROOT = paths.REPO
V1D = ROOT / "ReplicationData" / "MouseV1_Sorensen2026"
DEM = ROOT / "ReplicationData" / "Dembrow2024"
OUT = paths.results(__file__)
N_PERM = 3000
RNG = np.random.default_rng(0)
IEG = _m23.CATEGORIES["IEG_TF"]

HK_MOUSE = ["Gapdh", "Actb", "Actg1", "Tuba1a", "Tuba1b", "Tubb5", "Ubb", "Ubc",
            "Eef1a1", "Eef2", "Rplp0", "Rps18", "Rpl13a", "Ppia", "Pgk1",
            "Ldha", "Aldoa", "Eno1", "Atp5b", "Ywhaz"]


def parse_ap_list(s):
    """Macaque `APs[][0]` is a stringified list of spike counts, one per current
    step -- not a scalar. Its maximum is the analogue of mouse "Max number of APs",
    which is what the sign calibration needs."""
    out = []
    for v in s:
        try:
            vals = [float(x) for x in str(v).strip("[] ").split(",") if x.strip()]
            out.append(max(vals) if vals else np.nan)
        except ValueError:
            out.append(np.nan)
    return np.array(out, dtype=float)


def hk_normalise(log_cpm, hk_names):
    hk = [g for g in hk_names if g in log_cpm.columns
          and (log_cpm[g] > 0).mean() >= 0.5]
    return log_cpm.sub(log_cpm[hk].mean(axis=1), axis=0), len(hk)


def test_one(expr_hk, expr_raw, y, spikes, C, n, grp, label, measure, panel):
    """Index vs one adaptation measure, with the sign calibrated on spike count."""
    v, ng = _m31.index_score_filtered(expr_hk, expr_raw, panel)
    if v is None:
        return None
    gv = _m22.rank_resid(v, C); gv -= gv.mean()
    gv = gv / gv.std() if gv.std() > 0 else gv
    yr = _m22.rank_resid(np.asarray(y, float), C)
    yc = (yr - yr.mean()) / yr.std()
    obs = float((gv * yc).sum() / (n - 1))

    # sign calibration: does this measure rise or fall with spike count?
    sgn, cal = np.nan, np.nan
    if spikes is not None and np.isfinite(np.asarray(spikes, float)).sum() > n * 0.6:
        sr = _m22.rank_resid(np.asarray(spikes, float), C)
        sr = (sr - sr.mean()) / sr.std()
        cal = float((yc * sr).sum() / (n - 1))
        sgn = -1.0 if cal > 0 else 1.0     # more spikes = LESS adaptation
    obs_h = obs * sgn if np.isfinite(sgn) else np.nan

    perms = []
    for _ in range(N_PERM):
        yp = yr.copy()
        for g in np.unique(grp):
            m = grp == g
            yp[m] = RNG.permutation(yp[m])
        yp = (yp - yp.mean()) / yp.std()
        perms.append(float((gv * yp).sum() / (n - 1)))
    perms = np.asarray(perms)
    p = float(2 * min((perms <= obs).mean(), (perms >= obs).mean()))
    return {"cohort": label, "measure": measure, "n": n, "n_genes": ng,
            "rho_raw": obs, "cal_vs_spikes": cal, "sign_flip": sgn,
            "rho_harmonised": obs_h, "p_perm": p}


def main() -> None:
    rows = []

    # ---------------- mouse M1 ----------------
    meta = load_meta(); ephys = load_ephys()
    lc, qc = _m3.load_full_logcpm(meta.index, use_introns=True)
    hkn, nhk = hk_normalise(lc, HK_MOUSE)
    d0 = meta.loc[lc.index].join(qc, how="left")
    rt = d0["RNA type"].astype(str)
    lay = pd.Series("other", index=d0.index)
    for nm, pat in _m20.LAYER_PATTERNS.items():
        lay[rt.str.match(pat, na=False)] = nm
    idx = lc.index[lay != "other"]
    M1_MEAS = ["ISI average adaptation index", "ISI adaptation index",
               "Spike frequency adaptation", "ISI Fano factor",
               "ISI coefficient of variation"]
    print(f"[M1] {len(idx)} pyramidal cells, {nhk} HK genes")
    for meas in M1_MEAS:
        y = ephys[meas].reindex(idx)
        keep = idx[y.notna().values]
        d = d0.loc[keep]; n = len(keep)
        s = d["RNA type"]; k = s.value_counts()[lambda x: x >= 8].index
        s = s.where(s.isin(k), "other")
        C = _m22.build_design(d[_m3.COVARS],
                             pd.get_dummies(s, prefix="t", drop_first=True).astype(float), n)
        r = test_one(hkn.loc[keep], lc.loc[keep], y.loc[keep],
                     ephys["Max number of APs"].reindex(keep),
                     C, n, s.to_numpy(), "mouse M1 pyramidal", meas, IEG)
        if r: rows.append(r)

    # ---------------- mouse V1 ----------------
    lcv, cnt, q2 = _m28.load_v1()
    lcv.index = lcv.index.astype(str); q2.index = q2.index.astype(str)
    hkv, nhkv = hk_normalise(lcv, HK_MOUSE)
    ph = pd.read_csv(V1D / "exc_mMET_ephys_features.csv")
    ph["specimen_id"] = ph["specimen_id"].astype(str); ph = ph.set_index("specimen_id")
    mt0 = pd.read_csv(V1D / "inferred_met_types.csv").rename(
        columns={"Unnamed: 0": "specimen_id"})
    mt0["specimen_id"] = mt0["specimen_id"].astype(str); mt0 = mt0.set_index("specimen_id")
    adapt_cols = [c for c in ph.columns if re.match(r"^adapt_\d+_long_square$", c)]
    cv_cols = [c for c in ph.columns if re.match(r"^isi_cv_\d+_long_square$", c)]
    rate_cols = [c for c in ph.columns if re.match(r"^avg_rate_\d+_long_square$", c)]
    V1_MEAS = {"adapt (mean over steps)": adapt_cols,
               "isi_cv (mean over steps)": cv_cols}
    print(f"[V1] {nhkv} HK genes; adapt cols {len(adapt_cols)}, "
          f"isi_cv {len(cv_cols)}, rate {len(rate_cols)}")
    dv = q2.join(mt0[["inferred_met_type"]], how="inner")
    dv = dv[dv["inferred_met_type"].astype(str).str.startswith("L5 ET")]
    iv = lcv.index.intersection(dv.index)
    spikes_v1 = ph[rate_cols].mean(axis=1).reindex(iv) if rate_cols else None
    for meas, cols in V1_MEAS.items():
        if not cols:
            continue
        y = ph[cols].mean(axis=1).reindex(iv)
        keep = iv[y.notna().values]
        d = dv.loc[keep]; n = len(keep)
        mt = d["inferred_met_type"].astype(str)
        kk = mt.value_counts()[lambda x: x >= 8].index
        mt = mt.where(mt.isin(kk), "other")
        C = _m22.build_design(d[["log_libsize", "n_detected"]],
                             pd.get_dummies(mt, prefix="m", drop_first=True).astype(float), n)
        r = test_one(hkv.loc[keep], lcv.loc[keep], y.loc[keep],
                     spikes_v1.reindex(keep) if spikes_v1 is not None else None,
                     C, n, mt.to_numpy(), "mouse V1 L5 ET", meas, IEG)
        if r: rows.append(r)

    # ---------------- macaque ----------------
    ftm = pd.read_feather(DEM / "macaque_L5_patchseq.feather")
    phm = pd.read_csv(DEM / "L5_physiology_121223.csv", low_memory=False)
    cel = pd.read_csv(DEM / "cell_list_L5.csv", low_memory=False)
    K = "container"
    ftm[K] = ftm.patched_cell_container_label.astype(str).str.strip()
    phm[K] = phm.sample_id.astype(str).str.strip()
    cel[K] = cel.patched_cell_container.astype(str).str.strip()
    mm = (phm[[K, "SFA", "CV", "gain_CV", "APs[][0]", "Brain_Region"]]
          .merge(cel[[K, "subclass_Corr_label_M1", "Genes.Detected", "QC"]],
                 on=K, how="inner"))
    mm = mm[mm.QC == True].drop_duplicates(K).set_index(K)
    mm = mm[mm.subclass_Corr_label_M1.astype(str).str.contains("L5 ET")]
    em = ftm.set_index(K)
    reads = em.total_reads_label
    em = em.drop(columns=["sample_id", "patched_cell_container_label",
                          "total_reads_label"])
    im = mm.index.intersection(em.index)
    mm = mm.loc[im]; mm["log_reads"] = np.log10(reads.reindex(im))
    lcm = np.log1p(em.loc[im])
    hkm, nhkm = hk_normalise(lcm, [g.upper() for g in HK_MOUSE])
    IEG_U = [g.upper() for g in IEG]
    print(f"[macaque] {len(im)} L5 ET cells, {nhkm} HK genes")
    for meas in ["SFA", "CV", "gain_CV"]:
        if meas not in mm.columns:
            continue
        y = mm[meas]
        keep = im[y.notna().values]
        d = mm.loc[keep]; n = len(keep)
        if n < 25:
            continue
        reg = d["Brain_Region"].astype(str)
        k = reg.value_counts()[lambda x: x >= 5].index
        reg = reg.where(reg.isin(k), "other")
        C = _m22.build_design(d[["log_reads", "Genes.Detected"]],
                             pd.get_dummies(reg, prefix="r", drop_first=True).astype(float), n)
        # temporarily point index_score at the uppercase panel
        r = test_one(hkm.loc[keep], lcm.loc[keep], y.loc[keep],
                     parse_ap_list(d["APs[][0]"]),
                     C, n, reg.to_numpy(), "macaque L5 ET", meas, IEG_U)
        if r: rows.append(r)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "AI_adaptation_three_cohorts.csv", index=False)
    print("\n" + "=" * 100)
    print("ACTIVITY INDEX vs FIRING ADAPTATION — three cohorts, HK-normalised")
    print("  rho_harmonised > 0  =  more activity-gene expression, MORE adaptation")
    print("=" * 100)
    print(f"  {'cohort':22s} {'measure':30s} {'n':>4s} {'rho_raw':>8s} "
          f"{'flip':>5s} {'rho_harm':>9s} {'p':>8s}")
    for _, r in R.iterrows():
        star = " *" if r.p_perm < 0.05 else "  "
        print(f"  {r.cohort:22s} {r.measure[:30]:30s} {int(r.n):4d} "
              f"{r.rho_raw:+8.3f} {r.sign_flip:+5.0f} {r.rho_harmonised:+9.3f} "
              f"{r.p_perm:8.4f}{star}")
    print("\n  sign flip is derived from each measure's correlation with spike count,")
    print("  not assumed. flip = -1 means the raw measure DEcreases with adaptation.")
    ok = R.rho_harmonised.notna()
    if ok.sum() > 1:
        pos = int((R.loc[ok, "rho_harmonised"] > 0).sum())
        print(f"\n  measures with rho_harmonised > 0: {pos}/{int(ok.sum())}")
    print(f"\n[write] {OUT/'AI_adaptation_three_cohorts.csv'}")


if __name__ == "__main__":
    main()
