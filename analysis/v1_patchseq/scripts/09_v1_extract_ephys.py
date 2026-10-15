"""Download V1 Patch-seq NWBs (DANDI:000020; Gouwens et al. 2020) and extract intrinsic
features per cell. No expression / IEG data is read here (only the ephys_session_id /
cell_specimen_id / donor_id columns of the label-transfer table
cross_dataset/results/v1_pvalb_label_transfer.csv).

Stages (both resumable; run with no args to do both):
  --download   fetch sub-<donor specimen>_ses-<ephys_session_id>_icephys.nwb for every cell into
               ReplicationData/MouseV1_Gouwens2020/nwb/ (skips files whose size equals the DANDI asset size;
               writes to *.part then renames). <= 6 parallel workers.
  --extract    compute features -> results/v1_pvalb_ephys_features.csv
               (+ per-sweep audit table ReplicationData/MouseV1_Gouwens2020/v1_pvalb_longsquare_sweeps.csv)

Asset matching: DANDI paths are sub-<S>/sub-<S>_ses-<ephys_session_id>_icephys.nwb. <S> is neither the
cell_specimen_id nor donor_id; it is the donor's *specimen* id (1:1 with donor_id across all 888 cells).
Cells are matched on ses-<ephys_session_id> (888/888 unique matches).

Implementation: direct h5py reading of the Allen NWB 2.2.5 files (ipfx not used: it pins old
numpy/pynwb and would require changing this project's environment). Spike detection follows the
ipfx defaults:
  * dV/dt from V low-passed by a 4-pole Bessel filter at 10 kHz (zero-phase), V itself unfiltered;
  * putative spike = upward crossing of dV/dt >= 20 mV/ms; peak = max V within 2 ms after crossing
    (ended early at the next crossing); keep if peak > -30 mV and peak - threshold >= 2 mV;
  * threshold refined backwards from the max-upstroke point to where dV/dt < 5 % of max upstroke;
  * the spike must repolarise below its half-amplitude level within 5 ms of the peak;
  * a new crossing is only accepted after dV/dt has gone negative following the previous peak and
    >= 1 ms after the previous threshold (minimum ISI).

Long-square sweeps: CurrentClampSeries whose stimulus name matches
  LS(COARSE|FINE...) (C1LS protocol, e.g. C1LSCOARSE150216, C1LSFINEST150112, C1LSCOARSEMICRO)
  or X1PS_SubThresh / X3LP_Rheo(2) / X4PS_SupraThresh (X-series protocol; Y3LP_Rheo / Y4PS_SupraThresh are a
  repeat run of the same 1-s steps in 4 cells), or subthreshold / subthreshold_171 (1-s negative steps),
AND whose stimulus waveform (after the 10-ms test pulse) is a single constant step of 1.000 +- 0.01 s.
Stimulus window and amplitude are read from the stimulus (DA) series, not from names.
(Not used: X2LP_Search / Y2LP_Search = 0.5-s steps; Rheobase_finder = 0.5-s staircase; short squares,
ramps, noise, chirps.)

Sweep QC (failing sweeps are excluded and counted, never silently): recording must not end
(trailing zero padding / NaN) before stim end + 50 ms; pre-stimulus baseline (last 100 ms before
onset) must lie in [-100, -40] mV.

Features:
  max_n_spikes  = max over QC-passed long-square sweeps of the number of spikes whose threshold time
                  lies within [stim onset, stim offset].
  rheobase_pA   = lowest positive stimulus amplitude among QC-passed long-square sweeps with >= 1 spike in
                  the window (first such sweep chronologically if repeated).
  ap_width_ms   = full width at half amplitude of the FIRST spike on the rheobase sweep, half
                  amplitude = (V_peak - V_threshold)/2 above threshold; crossings linearly interpolated.
Additional QC columns: n_long_square_sweeps (QC-passed), n_ls_failed_qc, max_amp_pA, protocol,
ls_stimuli, allen_max_n_spikes (spike count from the NWB's own processing/spikes module on the
max-count sweep, as a detector cross-check), status, reason.

Usage:  uv run python analysis/v1_patchseq/scripts/09_v1_extract_ephys.py [--download] [--extract]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from common import RAW, V1_GOUWENS  # noqa: E402
RESULTS = paths.results(__file__)

V1 = V1_GOUWENS
NWB_DIR = V1 / "nwb"
ASSETS_JSON = V1 / "asset_list.json"
CELLS_CSV = paths.CROSS / "results" / "v1_pvalb_label_transfer.csv"
OUT_CSV = RESULTS / "v1_pvalb_ephys_features.csv"
SWEEP_CSV = V1 / "v1_pvalb_longsquare_sweeps.csv"

LS_NAME = re.compile(r"LS(COARSE|FINE)|[XY]1PS_SubThresh|[XY]3LP_Rheo|[XY]4PS_SupraThresh|^subthreshold")
DVDT_THR = 20.0      # mV/ms
MIN_PEAK = -30.0     # mV
MIN_HEIGHT = 2.0     # mV, peak - threshold
THR_FRAC = 0.05
MIN_ISI = 1e-3       # s
REPOL_MAX = 5e-3     # s
FILT_HZ = 10e3


def load_cells():
    # only the three ID columns; no other column is read
    return pd.read_csv(CELLS_CSV, index_col=0, usecols=lambda c: c in
                       ("Unnamed: 0", "ephys_session_id", "cell_specimen_id", "donor_id"))


def asset_map():
    if not ASSETS_JSON.exists():
        from dandi.dandiapi import DandiAPIClient
        ds = DandiAPIClient().get_dandiset("000020", "draft")
        json.dump([(a.path, a.size, a.identifier) for a in ds.get_assets()], open(ASSETS_JSON, "w"))
    rows = json.load(open(ASSETS_JSON))
    m = {}
    for p, s, i in rows:
        ses = int(re.search(r"ses-(\d+)", p).group(1))
        m.setdefault(ses, []).append((p, s, i))
    return m


def local_path(asset_path):
    return NWB_DIR / Path(asset_path).name


# ----------------------------------------------------------------------------------- download
def _download(asset):
    import requests
    p, size, aid = asset
    dst = local_path(p)
    if dst.exists() and dst.stat().st_size == size:
        return "skip", p
    tmp = dst.with_suffix(".nwb.part")
    url = f"https://api.dandiarchive.org/api/assets/{aid}/download/"
    for attempt in range(4):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if tmp.stat().st_size != size:
                raise IOError(f"size {tmp.stat().st_size} != {size}")
            os.replace(tmp, dst)
            return "ok", p
        except Exception as e:  # retry
            err = e
            time.sleep(5 * (attempt + 1))
    return f"fail: {err}", p


def download(cells, amap, workers=6):
    NWB_DIR.mkdir(parents=True, exist_ok=True)
    todo = [amap[s][0] for s in cells.ephys_session_id if s in amap]
    print(f"{len(todo)} assets, {sum(a[1] for a in todo) / 1e9:.1f} GB total")
    t0, n, res = time.time(), 0, {}
    with ThreadPoolExecutor(workers) as ex:
        for fut in as_completed([ex.submit(_download, a) for a in todo]):
            st, p = fut.result()
            res[st.split(":")[0]] = res.get(st.split(":")[0], 0) + 1
            n += 1
            if st.startswith("fail"):
                print(st, p, flush=True)
            if n % 50 == 0:
                print(f"  {n}/{len(todo)} {res} {time.time() - t0:.0f}s", flush=True)
    print("download done:", res, f"{time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------------- spikes
def detect_spikes(t, v, start=None, end=None):
    """Return list of dicts (thr_i, peak_i, thr_t, peak_v, thr_v) for spikes; v in mV, t in s."""
    from scipy.signal import bessel, filtfilt
    fs = 1.0 / (t[1] - t[0])
    if FILT_HZ < fs / 2:
        b, a = bessel(4, FILT_HZ / (fs / 2), "low")
        vf = filtfilt(b, a, v)
    else:
        vf = v
    dvdt = np.gradient(vf) * fs / 1e3          # mV/ms
    i0 = 0 if start is None else max(int((start - t[0]) * fs) - int(0.002 * fs), 0)
    i1 = len(v) if end is None else min(int((end - t[0]) * fs) + int(0.005 * fs), len(v))
    above = dvdt[i0:i1] >= DVDT_THR
    cross = np.flatnonzero(above[1:] & ~above[:-1]) + 1 + i0
    win = int(0.002 * fs)
    spikes, last_peak, last_thr_t = [], -1, -np.inf
    for k, c in enumerate(cross):
        if c <= last_peak:
            continue
        if last_peak >= 0 and not np.any(dvdt[last_peak:c] < 0):
            continue
        stop = min(c + win, len(v))
        if k + 1 < len(cross):
            stop = min(stop, cross[k + 1])
        stop = max(stop, c + 2)
        pk = c + int(np.argmax(v[c:stop]))
        up = c + int(np.argmax(dvdt[c:pk + 1])) if pk > c else c
        # refine threshold backwards from max upstroke
        lim = max(last_peak + 1, c - int(0.002 * fs), 0)
        j = up
        while j > lim and dvdt[j] >= THR_FRAC * dvdt[up]:
            j -= 1
        thr_i = j
        if v[pk] <= MIN_PEAK or v[pk] - v[thr_i] < MIN_HEIGHT:
            continue
        # must repolarise: V falls below the half-amplitude level within REPOL_MAX of the peak
        # (rejects onset artefacts / slow plateaus that pass the dV/dt and peak criteria)
        half = v[thr_i] + 0.5 * (v[pk] - v[thr_i])
        if not np.any(v[pk:min(pk + int(REPOL_MAX * fs), len(v))] < half):
            continue
        if t[thr_i] - last_thr_t < MIN_ISI:
            continue
        spikes.append(dict(thr_i=thr_i, peak_i=pk, thr_t=t[thr_i], peak_v=v[pk], thr_v=v[thr_i]))
        last_peak, last_thr_t = pk, t[thr_i]
    return spikes


def half_width(t, v, sp, next_thr_i=None):
    """FWHM at (peak+threshold)/2 for one spike, linear interpolation; ms."""
    lvl = sp["thr_v"] + 0.5 * (sp["peak_v"] - sp["thr_v"])
    pk, th = sp["peak_i"], sp["thr_i"]
    # rising crossing between threshold and peak
    seg = v[th:pk + 1]
    r = np.flatnonzero(seg >= lvl)
    if len(r) == 0 or r[0] == 0:
        return np.nan
    a = th + r[0]
    tr = t[a - 1] + (lvl - v[a - 1]) * (t[a] - t[a - 1]) / (v[a] - v[a - 1])
    stop = len(v) if next_thr_i is None else next_thr_i
    seg = v[pk:stop]
    f = np.flatnonzero(seg < lvl)
    if len(f) == 0:
        return np.nan
    b = pk + f[0]
    tf = t[b - 1] + (lvl - v[b - 1]) * (t[b] - t[b - 1]) / (v[b] - v[b - 1])
    return (tf - tr) * 1e3


# ----------------------------------------------------------------------------------- per cell
def step_epochs(i_pa, fs):
    """Constant nonzero epochs of the stimulus: list of (start_idx, n_samples, amp_pA)."""
    ch = np.flatnonzero(np.abs(np.diff(i_pa)) > 0.5)
    b = np.r_[0, ch + 1, len(i_pa)]
    out = []
    for a, c in zip(b[:-1], b[1:]):
        if c - a > 0.002 * fs:
            amp = float(np.median(i_pa[a:c]))
            if abs(amp) > 0.5:
                out.append((a, c - a, amp))
    return out


def process_cell(ses, path):
    import h5py
    rec = dict(ephys_session_id=ses)
    sweeps = []
    if not Path(path).exists():
        rec.update(status="fail", reason="nwb missing")
        return rec, sweeps
    try:
        h = h5py.File(path, "r")
    except Exception as e:
        rec.update(status="fail", reason=f"nwb unreadable: {e}")
        return rec, sweeps
    with h:
        acq = h["acquisition"]
        allen = h["processing/spikes"] if "processing/spikes" in h else None
        for k in acq:
            g = acq[k]
            if g.attrs.get("neurodata_type") != "CurrentClampSeries":
                continue
            name = g.attrs.get("stimulus_description", "")
            name = name.decode() if isinstance(name, bytes) else str(name)
            if not LS_NAME.search(name):
                continue
            sk = k.replace("_AD", "_DA")
            if sk not in h["stimulus/presentation"]:
                continue
            sweep_no = int(g.attrs.get("sweep_number", -1))
            s = h["stimulus/presentation"][sk]
            fs = float(g["starting_time"].attrs["rate"])
            i_pa = s["data"][:].astype(float) * float(s["data"].attrs["conversion"]) * 1e12
            v = g["data"][:].astype(float) * float(g["data"].attrs["conversion"]) * 1e3
            t = np.arange(len(v)) / fs
            ep = [e for e in step_epochs(i_pa, fs) if e[1] / fs > 0.05]  # drop 10-ms test pulse
            sw = dict(ephys_session_id=ses, sweep=sweep_no, stimulus=name)
            if len(ep) != 1 or abs(ep[0][1] / fs - 1.0) > 0.01:
                sw.update(qc="not a single 1-s step")
                sweeps.append(sw)
                continue
            a, n, amp = ep[0]
            st, en = a / fs, (a + n) / fs
            sw.update(amp_pA=round(amp, 1), stim_start=st, stim_end=en)
            # end of recording: last sample before trailing zeros / NaNs
            good = np.isfinite(v) & (v != 0)
            last = np.flatnonzero(good)
            rec_end = (last[-1] + 1) / fs if len(last) else 0.0
            base = v[max(a - int(0.1 * fs), 0):a]
            bmean = float(np.nanmean(base)) if len(base) else np.nan
            sw.update(rec_end=rec_end, baseline_mV=round(bmean, 2))
            if rec_end < en + 0.05:
                sw.update(qc="recording ends before stim end")
                sweeps.append(sw)
                continue
            if not (-100 <= bmean <= -40):
                sw.update(qc="baseline out of [-100,-40] mV")
                sweeps.append(sw)
                continue
            nn = int(min(len(v), np.flatnonzero(good)[-1] + 1))
            spk = detect_spikes(t[:nn], np.where(np.isfinite(v[:nn]), v[:nn], 0.0), st, en)
            spk = [x for x in spk if st <= x["thr_t"] <= en]
            hw = np.nan
            if spk:
                nxt = spk[1]["thr_i"] if len(spk) > 1 else None
                hw = half_width(t, v, spk[0], nxt)
            n_allen = np.nan
            if allen is not None and f"Sweep_{sweep_no}" in allen:
                n_allen = int(allen[f"Sweep_{sweep_no}/timestamps"].shape[0])
            sw.update(qc="pass", n_spikes=len(spk), first_hw_ms=hw,
                      first_thr_mV=spk[0]["thr_v"] if spk else np.nan,
                      first_peak_mV=spk[0]["peak_v"] if spk else np.nan,
                      allen_n_spikes=n_allen, fs=fs)
            sweeps.append(sw)
    sdf = pd.DataFrame(sweeps)
    names = sorted(set(sdf.stimulus)) if len(sdf) else []
    rec["ls_stimuli"] = ";".join(names)
    rec["protocol"] = ("X-series(2018+)" if any(n[0] in "XY" for n in names) else
                       "C1LS(2015-18)" if names else "")
    ok = sdf[sdf.qc == "pass"] if len(sdf) else sdf
    rec["n_long_square_sweeps"] = len(ok)
    rec["n_ls_failed_qc"] = int((sdf.qc != "pass").sum()) if len(sdf) else 0
    if len(sdf) == 0:
        rec.update(status="fail", reason="no long-square sweeps")
        return rec, sweeps
    if len(ok) == 0:
        rec.update(status="fail", reason="no long-square sweep passed QC: " +
                   ";".join(sorted(set(sdf.qc))))
        return rec, sweeps
    rec["max_amp_pA"] = float(ok.amp_pA.max())
    imax = ok.n_spikes.idxmax()
    rec["max_n_spikes"] = int(ok.n_spikes.max())
    rec["allen_max_n_spikes"] = ok.loc[imax, "allen_n_spikes"]
    spiking = ok[(ok.n_spikes > 0) & (ok.amp_pA > 0)]
    if len(spiking) == 0:
        rec.update(status="partial", reason="no spikes on any QC-passed long square")
        return rec, sweeps
    rmin = spiking.amp_pA.min()
    rsw = spiking[spiking.amp_pA == rmin].sort_values("sweep").iloc[0]
    rec["rheobase_pA"] = float(rmin)
    rec["rheobase_sweep"] = int(rsw.sweep)
    rec["ap_width_ms"] = float(rsw.first_hw_ms)
    if np.isnan(rec["ap_width_ms"]):
        rec.update(status="partial", reason="half-width undefined on rheobase spike")
    else:
        rec.update(status="ok", reason="")
    return rec, sweeps


def extract(cells, amap, workers=6):
    t0 = time.time()
    jobs = {ses: local_path(amap[ses][0][0]) if ses in amap else Path("/nonexistent")
            for ses in cells.ephys_session_id}
    recs, allsw = {}, []
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(process_cell, ses, str(p)): ses for ses, p in jobs.items()}
        for n, fut in enumerate(as_completed(futs), 1):
            ses = futs[fut]
            try:
                r, sw = fut.result()
            except Exception as e:
                r, sw = dict(ephys_session_id=ses, status="fail", reason=f"error: {e!r}"), []
            recs[ses] = r
            allsw += sw
            if n % 100 == 0:
                print(f"  extracted {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    feat = pd.DataFrame([recs[s] for s in cells.ephys_session_id], index=cells.index)
    cols = ["ephys_session_id", "max_n_spikes", "ap_width_ms", "rheobase_pA", "n_long_square_sweeps",
            "status", "reason", "n_ls_failed_qc", "max_amp_pA", "rheobase_sweep", "protocol",
            "allen_max_n_spikes", "ls_stimuli"]
    for c in cols:
        if c not in feat:
            feat[c] = np.nan
    feat = feat[cols]
    feat.to_csv(OUT_CSV)
    pd.DataFrame(allsw).sort_values(["ephys_session_id", "sweep"]).to_csv(SWEEP_CSV, index=False)
    print(f"extraction done in {time.time() - t0:.0f}s -> {OUT_CSV}")
    print(feat.status.value_counts().to_string())
    print(feat.reason[feat.reason.astype(str) != ""].value_counts().to_string())
    for c in ["max_n_spikes", "ap_width_ms", "rheobase_pA", "n_long_square_sweeps"]:
        x = feat[c].dropna()
        print(f"{c}: n={len(x)} median={x.median():.3g} IQR=[{x.quantile(.25):.3g}, {x.quantile(.75):.3g}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", nargs="*", type=int, help="ephys_session_ids (debug)")
    a = ap.parse_args()
    if not (a.download or a.extract):
        a.download = a.extract = True
    cells = load_cells()
    if a.only:
        cells = cells[cells.ephys_session_id.isin(a.only)]
    amap = asset_map()
    miss = [s for s in cells.ephys_session_id if s not in amap]
    multi = [s for s in cells.ephys_session_id if len(amap.get(s, [])) > 1]
    print(f"{len(cells)} cells; {len(miss)} without DANDI asset; {len(multi)} with >1 asset")
    if a.download:
        download(cells, amap, min(a.workers, 8))
    if a.extract and not a.only:
        extract(cells, amap, min(a.workers, 8))
    elif a.extract:
        for ses in cells.ephys_session_id:
            r, sw = process_cell(ses, str(local_path(amap[ses][0][0])))
            print(pd.DataFrame(sw).drop(columns=["stim_start", "rec_end", "fs"], errors="ignore").to_string())
            print(r)
