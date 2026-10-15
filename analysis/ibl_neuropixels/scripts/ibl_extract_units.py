"""Extract per-unit in-vivo firing statistics for motor-cortex (and comparison
isocortex) units from the IBL Brain-Wide Map on DANDI (dandiset 000409).

Nothing is downloaded wholesale: every NWB file is streamed over HTTP range
requests with remfile + h5py, and only the small tables we need are read.

Stages (each is resumable; finished sessions are skipped on re-run):

  1. assets   List all dandiset assets -> data/asset_list.csv.
              One `*_desc-processed_behavior+ecephys.nwb` per session (~1 GB,
              contains Units + trials + passive protocol). The
              `*_desc-raw_ecephys.nwb` files (~95 GB) hold raw/LFP only and
              are never touched.
  2. scan     For every processed file read the units' small columns
              (max_electrode, quality, waveform metrics) and the electrode
              CCF localisation; assign each unit the Allen acronym of its
              max-amplitude electrode -> data/scan/<session>.parquet.
  3. pull     For sessions with target-region units, read spike times of only
              those units (spikes are stored unit-contiguous and units are
              depth-sorted, so cortex units are one contiguous slab), their
              mean waveforms, the trials table, epochs and passive intervals
              -> data/raw/<session>.npz + <session>_trials.parquet.
  4. metrics  Compute per-unit statistics from the cached raw pulls
              -> data/ibl_motor_units.parquet / .csv.

Usage:
    uv run python analysis/ibl_neuropixels/scripts/ibl_extract_units.py all  [--workers 8] [--max-pull-sessions N] [--comparison-scope motor_sessions|all] [--motor-only]
    uv run python analysis/ibl_neuropixels/scripts/ibl_extract_units.py scan|pull|metrics

Region selection: units whose CCF acronym belongs to an isocortical area in
TARGET_PREFIXES (MOp*, MOs* are the primary targets; SSp, SSs, ACA, PL, ILA,
ORB, RSP, VISp(1-6), AUD, FRP, AI, PTLp/VISa/VISrl are comparison areas).
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)

import argparse
import json
import os
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

OUT = paths.data(__file__)
SCAN = OUT / "scan"
RAW = OUT / "raw"
LOG = OUT / "logs"
DANDISET = "000409"
VERSION = "draft"  # same assets as the latest published version

# Allen isocortex areas we keep. Order matters for the regex (longest first).
MOTOR = ("MOp", "MOs")
COMPARISON = ("SSp", "SSs", "ACAd", "ACAv", "PL", "ILA", "ORBl", "ORBm", "ORBvl",
              "RSPagl", "RSPd", "RSPv", "VISp", "VISa", "VISrl", "AUDp", "FRP", "AId", "AIv")
LAYER_RE = r"(1|2/3|2|3|4|5|6a|6b)"
# area prefix must be followed by optional subarea letters for SSp (SSp-bfd etc.),
# then a layer. VISp must not match VISpm/VISpl/VISpor.
TARGET_RE = re.compile(
    r"^(?P<area>MOp|MOs|SSp(?:-[a-z]+)?|SSs|ACAd|ACAv|PL|ILA|ORBl|ORBm|ORBvl|RSPagl|RSPd|RSPv|"
    r"VISp|VISa|VISrl|AUDp|FRP|AId|AIv)" + LAYER_RE + r"$")


def parse_region(acr: str):
    """Return (area, layer) if acr is a target isocortex acronym else (None, None)."""
    m = TARGET_RE.match(acr or "")
    if not m:
        return None, None
    return m.group("area"), m.group(2)


# ----------------------------------------------------------------------------- utils

def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def _open(url):
    import h5py
    import remfile
    rf = remfile.File(url)
    # large chunk cache: spike_times chunks are 1.25M float64 (10 MB) and are
    # revisited when target units form several non-contiguous runs
    return h5py.File(rf, "r", rdcc_nbytes=256 * 1024**2, rdcc_nslots=10007), rf


def _log(msg):
    LOG.mkdir(parents=True, exist_ok=True)
    with open(LOG / "run.log", "a") as fh:
        fh.write(time.strftime("%H:%M:%S ") + msg + "\n")
    print(msg, flush=True)


# ----------------------------------------------------------------------------- stage 1

def stage_assets():
    p = OUT / "asset_list.csv"
    if not p.exists():
        from dandi.dandiapi import DandiAPIClient
        ds = DandiAPIClient().get_dandiset(DANDISET, VERSION)
        rows = [dict(path=a.path, size=a.size, asset_id=a.identifier) for a in ds.get_assets()]
        pd.DataFrame(rows).to_csv(p, index=False)
    df = pd.read_csv(p)
    proc = df[df.path.str.endswith("_desc-processed_behavior+ecephys.nwb")].copy()
    proc["session_id"] = proc.path.str.extract(r"_ses-([0-9a-f-]{36})")[0]
    proc["subject"] = proc.path.str.extract(r"^sub-([^/]+)/")[0]
    return proc.reset_index(drop=True)


def _url(asset_id):
    # Stable S3 URL without a DANDI API round-trip per worker.
    from dandi.dandiapi import DandiAPIClient
    a = DandiAPIClient().get_dandiset(DANDISET, VERSION).get_asset(asset_id)
    return a.get_content_url(follow_redirects=1, strip_query=True)


# ----------------------------------------------------------------------------- stage 2

def scan_one(row):
    out = SCAN / f"{row['session_id']}.parquet"
    if out.exists():
        return row["session_id"], "cached"
    t0 = time.time()
    url = _url(row["asset_id"])
    f, rf = _open(url)
    u = f["units"]
    n = u["id"].shape[0]
    L = f["general/localization/ElectrodesCCFv3"]
    ent = L["localized_entity"][:]
    acr = _dec(L["brain_region"][:])
    lprobe = _dec(L["probe_name"][:])
    acr_map = dict(zip(ent, acr))
    probe_map = dict(zip(ent, lprobe))
    beryl = {}
    if "general/localization/ElectrodesIBLBregma" in f:
        B = f["general/localization/ElectrodesIBLBregma"]
        beryl = dict(zip(B["localized_entity"][:], _dec(B["beryl_location"][:])))
    maxe = u["max_electrode"][:]
    cols = {}
    for k in ["ibl_quality_score", "kilosort2_label", "firing_rate", "spike_count",
              "peak_to_trough_duration_ms", "presence_ratio", "isi_violations_ratio",
              "rp_violation", "sliding_rp_violation", "noise_cutoff",
              "median_spike_amplitude_uV", "missed_spikes_estimate",
              "distance_from_probe_tip_um", "probe_name", "unit_name", "cluster_uuid"]:
        if k in u:
            v = u[k][:]
            cols[k] = _dec(v) if v.dtype.kind in "OS" else v
    d = pd.DataFrame(cols)
    d.insert(0, "unit_index", np.arange(n))
    d.insert(0, "session_id", row["session_id"])
    d["subject"] = _dec([f["general/subject/subject_id"][()]])[0]
    d["lab"] = _dec([f["general/lab"][()]])[0] if "general/lab" in f else ""
    d["max_electrode"] = maxe
    d["acronym"] = [acr_map.get(e, "NA") for e in maxe]
    d["beryl"] = [beryl.get(e, "NA") for e in maxe]
    d["electrode_probe"] = [probe_map.get(e, "NA") for e in maxe]
    d["asset_id"] = row["asset_id"]
    d["n_electrodes_file"] = len(ent)
    d.to_parquet(out, index=False)
    f.close()
    return row["session_id"], f"{n} units {time.time()-t0:.1f}s"


# ----------------------------------------------------------------------------- stage 3

def pull_one(sid, asset_id, unit_idx):
    out = RAW / f"{sid}.npz"
    if out.exists():
        return sid, "cached"
    t0 = time.time()
    unit_idx = np.sort(np.asarray(unit_idx))
    f, rf = _open(_url(asset_id))
    u = f["units"]
    sti = u["spike_times_index"][:].astype(np.int64)
    starts = np.r_[0, sti[:-1]]
    # read spike times in contiguous runs of unit indices
    spikes, offsets = [], [0]
    runs = np.split(unit_idx, np.where(np.diff(unit_idx) != 1)[0] + 1)
    st = u["spike_times"]
    for r in runs:
        a, b = starts[r[0]], sti[r[-1]]
        block = st[a:b]
        for i in r:
            spikes.append(block[starts[i] - a: sti[i] - a])
    for s in spikes:
        offsets.append(offsets[-1] + len(s))
    spk = np.concatenate(spikes) if spikes else np.zeros(0)
    # mean waveforms (units x samples x channels), read in contiguous runs
    wf = []
    W = u["waveform_mean"]
    for r in runs:
        wf.append(W[r[0]:r[-1] + 1])
    wf = np.concatenate(wf, axis=0).astype(np.float32)
    # epochs / passive intervals
    epochs = dict(name=np.array([], str), start=np.zeros(0), stop=np.zeros(0))
    if "intervals/epochs" in f:  # missing in a few files; metrics falls back to trials/spikes
        ep = f["intervals/epochs"]
        epochs = dict(name=_dec(ep["protocol_type"][:]), start=ep["start_time"][:], stop=ep["stop_time"][:])
    pas = {}
    if "processing/passive_protocol/passive_intervals" in f:
        pi = f["processing/passive_protocol/passive_intervals"]
        pas = dict(name=_dec(pi["protocol_name"][:]), start=pi["start_time"][:], stop=pi["stop_time"][:])
    np.savez_compressed(
        out, unit_index=unit_idx, spike_times=spk, offsets=np.asarray(offsets, np.int64),
        waveform_mean=wf, ep_name=epochs["name"], ep_start=epochs["start"], ep_stop=epochs["stop"],
        pas_name=pas.get("name", np.array([], str)), pas_start=pas.get("start", np.zeros(0)),
        pas_stop=pas.get("stop", np.zeros(0)))
    if "intervals/trials" in f:
        t = f["intervals/trials"]
        td = {}
        for k in t:
            if k in ("id",) or k.endswith("_index"):
                continue
            v = t[k][:]
            if v.ndim == 1:
                td[k] = _dec(v) if v.dtype.kind in "OS" else v
        pd.DataFrame(td).to_parquet(RAW / f"{sid}_trials.parquet", index=False)
    f.close()
    return sid, f"{len(unit_idx)} units {len(spk)} spikes {time.time()-t0:.1f}s"


# ----------------------------------------------------------------------------- stage 4

def isi_stats(t):
    """ISI-based statistics for a sorted spike train t (seconds)."""
    res = dict(isi_cv=np.nan, cv2=np.nan, lv=np.nan, frac_isi_lt10ms=np.nan,
               frac_spikes_in_bursts=np.nan, refrac_viol_frac=np.nan, n_isi=0)
    if len(t) < 3:
        return res
    isi = np.diff(t)
    res["n_isi"] = len(isi)
    res["isi_cv"] = isi.std() / isi.mean() if isi.mean() > 0 else np.nan
    a, b = isi[:-1], isi[1:]
    s = a + b
    ok = s > 0
    res["cv2"] = np.mean(2 * np.abs(b[ok] - a[ok]) / s[ok])
    res["lv"] = 3 * np.mean(((a[ok] - b[ok]) / s[ok]) ** 2)
    short = isi < 0.010
    res["frac_isi_lt10ms"] = short.mean()
    inb = np.zeros(len(t), bool)
    inb[:-1] |= short
    inb[1:] |= short
    res["frac_spikes_in_bursts"] = inb.mean()
    res["refrac_viol_frac"] = (isi < 0.0015).mean()
    return res


def window_stats(t, t0, t1, prefix):
    sel = t[(t >= t0) & (t < t1)]
    dur = t1 - t0
    r = {f"{prefix}_dur_s": dur, f"{prefix}_rate_hz": len(sel) / dur if dur > 0 else np.nan}
    s = isi_stats(sel)
    r[f"{prefix}_isi_cv"] = s["isi_cv"]
    r[f"{prefix}_lv"] = s["lv"]
    r[f"{prefix}_cv2"] = s["cv2"]
    r[f"{prefix}_frac_isi_lt10ms"] = s["frac_isi_lt10ms"]
    if dur >= 10:
        c = np.histogram(sel, np.arange(t0, t1 + 1e-9, 1.0))[0]
        r[f"{prefix}_fano_1s"] = c.var() / c.mean() if c.mean() > 0 else np.nan
    return r


def waveform_features(w):
    """w: samples x channels (volts). Trough-to-peak on max peak-to-peak channel.
    Returns ms values at 30 kHz; parabolic sub-sample interpolation not applied."""
    fs = 30000.0
    if w is None or not np.isfinite(w).any():
        return dict(wf_ttp_ms=np.nan, wf_halfwidth_ms=np.nan, wf_pt_ratio=np.nan, wf_amp_uV=np.nan)
    ptp = w.max(0) - w.min(0)
    ch = int(np.nanargmax(ptp))
    x = w[:, ch].astype(float)
    x = x - np.median(x[:10])
    itr = int(np.argmin(x))
    trough = x[itr]
    # positive-dominant spikes (axonal / inverted / near-soma-top waveforms):
    # trough-to-peak is not meaningful for these -> flagged, class 'unclassified'
    positive = bool(x.max() > abs(trough))
    if itr >= len(x) - 2 or trough >= 0:
        return dict(wf_ttp_ms=np.nan, wf_halfwidth_ms=np.nan, wf_pt_ratio=np.nan, wf_amp_uV=ptp[ch] * 1e6)
    ipk = itr + int(np.argmax(x[itr:]))
    ttp = round((ipk - itr) / fs * 1e3, 4)  # rounded so 12 samples == 0.4 ms exactly
    half = trough / 2
    # contiguous run around the trough
    lo = itr
    while lo > 0 and x[lo - 1] <= half:
        lo -= 1
    hi = itr
    while hi < len(x) - 1 and x[hi + 1] <= half:
        hi += 1
    hw = round((hi - lo + 1) / fs * 1e3, 4)
    return dict(wf_ttp_ms=ttp, wf_halfwidth_ms=hw, wf_pt_ratio=abs(x[ipk] / trough),
                wf_amp_uV=ptp[ch] * 1e6, wf_peak_channel=ch, wf_positive_spike=positive)


def metrics_session(sid):
    z = np.load(RAW / f"{sid}.npz", allow_pickle=False)
    spk, off, uidx = z["spike_times"], z["offsets"], z["unit_index"]
    wf = z["waveform_mean"]
    ep = pd.DataFrame(dict(name=z["ep_name"], start=z["ep_start"], stop=z["ep_stop"]))
    pas = pd.DataFrame(dict(name=z["pas_name"], start=z["pas_start"], stop=z["pas_stop"]))
    tp = RAW / f"{sid}_trials.parquet"
    tr = pd.read_parquet(tp) if tp.exists() else None
    epoch_source = "epochs"
    if len(ep) == 0:
        # No epochs table: recording span = 0 .. max(last spike of any pulled unit,
        # last trial stop, last passive stop); task epoch = first..last trial.
        epoch_source = "fallback"
        ends = [spk.max() if len(spk) else 0.0]
        if tr is not None and len(tr):
            ends.append(float(tr.stop_time.max()))
            ep = pd.DataFrame(dict(name=["task"], start=[float(tr.start_time.min())],
                                   stop=[float(tr.stop_time.max())]))
        if len(pas):
            ends.append(float(pas.stop.max()))
        rec_t0, rec_t1 = 0.0, float(max(ends))
    else:
        rec_t0, rec_t1 = float(ep.start.min()), float(ep.stop.max())
    task = ep[ep.name == "task"]
    spont = pas[pas.name == "spontaneousActivity"]
    stim = fb = None
    if tr is not None and "gabor_stimulus_onset_time" in tr:
        stim = tr["gabor_stimulus_onset_time"].to_numpy()
        fb = tr["feedback_time"].to_numpy()
        valid = np.isfinite(stim) & np.isfinite(fb) & (fb > stim)
        stim, fb = stim[valid], fb[valid]
    rows = []
    for k, ui in enumerate(uidx):
        t = np.sort(spk[off[k]:off[k + 1]])
        r = dict(session_id=sid, unit_index=int(ui), epoch_source=epoch_source)
        dur = rec_t1 - rec_t0
        r["n_spikes"] = len(t)
        r["rec_duration_s"] = dur
        r["mean_rate_hz"] = len(t) / dur if dur > 0 else np.nan
        r.update(isi_stats(t))
        if dur > 2:
            c = np.histogram(t, np.arange(rec_t0, rec_t1 + 1e-9, 1.0))[0]
            r["fano_1s"] = c.var() / c.mean() if c.mean() > 0 else np.nan
        if len(task):
            r.update(window_stats(t, float(task.start.iloc[0]), float(task.stop.iloc[0]), "taskepoch"))
        if len(spont):
            r.update(window_stats(t, float(spont.start.iloc[0]), float(spont.stop.iloc[0]), "spont"))
        if stim is not None and len(stim) >= 10:
            # counts in [stimOn, stimOn+0.4)
            cnt = np.searchsorted(t, stim + 0.4) - np.searchsorted(t, stim)
            r["n_trials"] = len(stim)
            r["stim_0_400ms_rate_hz"] = cnt.mean() / 0.4
            r["stim_0_400ms_fano"] = cnt.var() / cnt.mean() if cnt.mean() > 0 else np.nan
            # pre-stimulus baseline: [stimOn-0.4, stimOn) falls in the enforced quiescence period
            cb = np.searchsorted(t, stim) - np.searchsorted(t, stim - 0.4)
            r["prestim_400ms_rate_hz"] = cb.mean() / 0.4
            r["prestim_400ms_fano"] = cb.var() / cb.mean() if cb.mean() > 0 else np.nan
            # task (stimOn -> feedback) vs inter-trial (feedback -> next stimOn)
            ct = np.searchsorted(t, fb) - np.searchsorted(t, stim)
            r["in_trial_rate_hz"] = ct.sum() / (fb - stim).sum()
            iti0, iti1 = fb[:-1], stim[1:]
            okk = iti1 > iti0
            ci = np.searchsorted(t, iti1[okk]) - np.searchsorted(t, iti0[okk])
            r["iti_rate_hz"] = ci.sum() / (iti1[okk] - iti0[okk]).sum() if okk.any() else np.nan
        r.update(waveform_features(wf[k]))
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- driver

def run_pool(fn, jobs, workers, label):
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, *j): j[0] if not isinstance(j[0], dict) else j[0]["session_id"] for j in jobs}
        for fu in as_completed(futs):
            sid = futs[fu]
            done += 1
            try:
                _, msg = fu.result()
                _log(f"[{label} {done}/{len(jobs)} {time.time()-t0:.0f}s] {sid} {msg}")
            except Exception as e:  # noqa: BLE001
                _log(f"[{label} {done}/{len(jobs)}] {sid} ERROR {e!r}")
                with open(LOG / f"{label}_errors.log", "a") as fh:
                    fh.write(f"{sid}\n{traceback.format_exc()}\n")


def load_scan():
    fs = sorted(SCAN.glob("*.parquet"))
    d = pd.concat([pd.read_parquet(p) for p in fs], ignore_index=True)
    ar = d["acronym"].map(parse_region)
    d["area"] = [a for a, _ in ar]
    d["layer"] = [l for _, l in ar]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["scan", "pull", "metrics", "all"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-pull-sessions", type=int, default=None,
                    help="limit pull to N sessions (motor sessions first)")
    ap.add_argument("--motor-only", action="store_true",
                    help="pull only MOp/MOs units (skip comparison areas)")
    ap.add_argument("--comparison-scope", choices=["motor_sessions", "all"], default="motor_sessions",
                    help="comparison-area units are pulled only from sessions that also have motor "
                         "units (default, cheap) or from every session ('all')")
    a = ap.parse_args()
    for p in (SCAN, RAW, LOG):
        p.mkdir(parents=True, exist_ok=True)
    proc = stage_assets()

    if a.stage in ("scan", "all"):
        jobs = [(r,) for r in proc.to_dict("records") if not (SCAN / f"{r['session_id']}.parquet").exists()]
        _log(f"scan: {len(jobs)} of {len(proc)} sessions to scan")
        run_pool(scan_one, jobs, a.workers, "scan")
        d = load_scan()
        d.to_parquet(OUT / "scan_all_units.parquet", index=False)

    if a.stage in ("pull", "all"):
        d = load_scan()
        tgt = d[d.area.notna()]
        if a.motor_only:
            tgt = tgt[tgt.area.isin(MOTOR)]
        motor_sessions = set(tgt[tgt.area.isin(MOTOR)].session_id)
        if a.comparison_scope == "motor_sessions":
            tgt = tgt[tgt.session_id.isin(motor_sessions)]
        g = tgt.groupby("session_id")
        motor_n = tgt[tgt.area.isin(MOTOR)].groupby("session_id").size()
        sessions = sorted(g.groups, key=lambda s: -motor_n.get(s, 0))
        if a.max_pull_sessions:
            sessions = sessions[: a.max_pull_sessions]
        jobs = [(s, g.get_group(s).asset_id.iloc[0], g.get_group(s).unit_index.to_numpy())
                for s in sessions if not (RAW / f"{s}.npz").exists()]
        _log(f"pull: {len(jobs)} sessions to pull ({len(sessions)} selected, "
             f"{(motor_n > 0).sum()} with motor units)")
        run_pool(pull_one, jobs, a.workers, "pull")

    if a.stage in ("metrics", "all"):
        d = load_scan()
        out = []
        sids = [p.stem for p in sorted(RAW.glob("*.npz"))]
        with ProcessPoolExecutor(a.workers) as ex:
            futs = {ex.submit(metrics_session, sid): sid for sid in sids}
            for fu in as_completed(futs):
                try:
                    out.append(fu.result())
                except Exception as e:  # noqa: BLE001
                    _log(f"metrics {futs[fu]} ERROR {e!r}")
        m = pd.concat(out, ignore_index=True)
        keep = ["session_id", "unit_index", "subject", "lab", "probe_name", "unit_name", "cluster_uuid",
                "acronym", "beryl", "area", "layer", "ibl_quality_score", "kilosort2_label",
                "peak_to_trough_duration_ms", "firing_rate", "spike_count", "presence_ratio",
                "isi_violations_ratio", "rp_violation", "sliding_rp_violation", "noise_cutoff",
                "median_spike_amplitude_uV", "missed_spikes_estimate", "distance_from_probe_tip_um",
                "max_electrode"]
        m = m.merge(d[keep], on=["session_id", "unit_index"], how="left")
        m = m.rename(columns={"firing_rate": "ibl_firing_rate", "spike_count": "ibl_spike_count",
                              "peak_to_trough_duration_ms": "ibl_ptt_ms"})
        m["region_group"] = np.where(m.area.isin(MOTOR), m.area, "other_isocortex")
        m["good"] = m.ibl_quality_score >= 0.999
        ttp = m["wf_ttp_ms"]
        pos = m["wf_positive_spike"].fillna(True).astype(bool)
        m["spike_width_class"] = np.select([pos, ttp < 0.4, ttp >= 0.4],
                                           ["unclassified", "narrow", "wide"], "unclassified")
        first = ["session_id", "subject", "lab", "probe_name", "unit_index", "unit_name", "cluster_uuid",
                 "acronym", "area", "layer", "region_group", "beryl", "good", "ibl_quality_score",
                 "kilosort2_label", "spike_width_class", "wf_ttp_ms", "ibl_ptt_ms"]
        m = m[first + [c for c in m.columns if c not in first]]
        m.to_parquet(OUT / "ibl_motor_units.parquet", index=False)
        m.to_csv(OUT / "ibl_motor_units.csv", index=False)
        _log(f"metrics: {len(m)} units from {m.session_id.nunique()} sessions written")


if __name__ == "__main__":
    main()
