"""Adapter presenting Allen MIES/Igor NWB v1 files as an AllenSDK NwbDataSet.

`ephys_extractor.cell_extractor_for_nwb(dataset, ramps, short_squares,
long_squares)` expects `dataset.get_sweep(n)` returning a dict with:

    response      membrane voltage in VOLTS   (extractor multiplies by 1e3 -> mV)
    stimulus      command current in AMPERES  (extractor multiplies by 1e12 -> pA)
    sampling_rate Hz
    index_range   (start_idx, end_idx) of the experiment epoch

The DANDI:001455 files are MIES/Igor NWB v1, a different converter from the
Scala NWBs the extractor was originally used with. Differences handled here:

* `stimulus_description` and `sweep_number` are HDF5 *attributes* on each
  acquisition dataset -- no labnotebook parsing needed.
* Clamp modes are mixed within a file: `VoltageClampSeries` test pulses
  (EXTP*) sit alongside `CurrentClampSeries` protocol sweeps. Only current
  clamp is usable, so voltage clamp is filtered out.
* Physical units come from the per-dataset `conversion` attribute, which is
  applied to reach SI. Do not assume a fixed scale factor.
* `session_description` is "PLACEHOLDER" and the top-level `identifier` is a
  content hash -- the Allen specimen ID is `general/session_id`.
"""

from __future__ import annotations

import re

import h5py
import numpy as np

# Allen stimulus-set name prefixes, matched against `stimulus_description`.
LONG_SQUARE = r"^C1LS(COARSE|FINEST)"
SHORT_SQUARE = r"^C1SS(COARSE|FINEST)"
RAMP = r"^C1RP"
TEST_PULSE = r"^EXTP"


class MiesNwbDataSet:
    """Minimal AllenSDK-compatible reader for MIES NWB v1."""

    def __init__(self, path: str):
        self.path = path
        self._f = h5py.File(path, "r")
        self._index = self._build_index()

    # ---------------------------------------------------------------- setup

    def _build_index(self) -> dict[int, dict]:
        idx = {}
        acq = self._f.get("acquisition")
        if acq is None:
            return idx
        for name, node in acq.items():
            if not isinstance(node, h5py.Group) or "data" not in node:
                continue
            a = node.attrs
            ntype = str(a.get("neurodata_type", ""))
            stim = str(a.get("stimulus_description", ""))
            swp = a.get("sweep_number")
            if swp is None:
                m = re.search(r"data_(\d+)_", name)
                swp = int(m.group(1)) if m else None
            if swp is None:
                continue
            idx[int(swp)] = {
                "acq": name,
                "stim_name": stim,
                "neurodata_type": ntype,
                "is_current_clamp": "CurrentClamp" in ntype,
            }
        return idx

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------ accessors

    @property
    def specimen_id(self) -> str | None:
        """Allen specimen ID -- the join key to transcriptomic tables."""
        for p in ("general/session_id", "identifier"):
            try:
                v = self._f[p][()]
                if isinstance(v, np.ndarray):
                    v = v[0]
                s = v.decode() if isinstance(v, bytes) else str(v)
                if p == "general/session_id":
                    return s.strip()
            except Exception:
                continue
        return None

    def subject_field(self, key: str) -> str | None:
        try:
            v = self._f[f"general/subject/{key}"][()]
            if isinstance(v, np.ndarray):
                v = v[0]
            return v.decode() if isinstance(v, bytes) else str(v)
        except Exception:
            return None

    def sweep_table(self) -> list[dict]:
        out = []
        for swp, meta in sorted(self._index.items()):
            out.append({"sweep_number": swp, **meta})
        return out

    def classify_sweeps(self) -> dict[str, list[int]]:
        """Sweep numbers by protocol. Current clamp only."""
        groups: dict[str, list[int]] = {
            "long_squares": [], "short_squares": [], "ramps": [], "test_pulses": [],
            "excluded_voltage_clamp": [], "unknown": [],
        }
        for swp, m in sorted(self._index.items()):
            name = m["stim_name"]
            if not m["is_current_clamp"]:
                groups["excluded_voltage_clamp"].append(swp)
                continue
            if re.match(LONG_SQUARE, name):
                groups["long_squares"].append(swp)
            elif re.match(SHORT_SQUARE, name):
                groups["short_squares"].append(swp)
            elif re.match(RAMP, name):
                groups["ramps"].append(swp)
            elif re.match(TEST_PULSE, name):
                groups["test_pulses"].append(swp)
            else:
                groups["unknown"].append(swp)
        return groups

    # ------------------------------------------------------------- get_sweep

    def get_sweep(self, sweep_number: int) -> dict:
        m = self._index[int(sweep_number)]
        acq = self._f[f"acquisition/{m['acq']}"]
        resp = acq["data"]
        v = np.asarray(resp[()], dtype=np.float64)
        v *= float(resp.attrs.get("conversion", 1.0))   # -> volts

        stim_name = m["acq"].replace("_AD0", "_DA0")
        spath = f"stimulus/presentation/{stim_name}"
        if spath in self._f:
            sd = self._f[spath]["data"]
            i = np.asarray(sd[()], dtype=np.float64)
            i *= float(sd.attrs.get("conversion", 1.0))  # -> amperes
        else:
            i = np.zeros_like(v)

        rate = None
        if "starting_time" in acq:
            rate = acq["starting_time"].attrs.get("rate")
        if rate is None:
            raise ValueError(f"no sampling rate for sweep {sweep_number}")

        n = min(len(v), len(i))
        return {
            "response": v[:n],
            "stimulus": i[:n],
            "sampling_rate": float(rate),
            "index_range": (0, n - 1),
        }

    # --------------------------------------------------------- step timing

    def detect_step(self, sweep_number: int, frac: float = 0.2,
                    skip_s: float = 0.1, min_dur_s: float = 0.05
                    ) -> tuple[float, float] | None:
        """Empirically locate the sustained stimulus step onset/offset, in seconds.

        Two complications this handles:

        * MIES inserts a brief test pulse at the start of every sweep
          (`TP Insert Checkbox: On`, ~10 ms at ~50 pA). A naive first-crossing
          search locks onto that instead of the protocol step, so the leading
          ``skip_s`` seconds are ignored.
        * There can be several excursions per sweep, so rather than taking the
          first and last crossing we take the *longest contiguous* run above
          threshold, which is the sustained step.

        The extractor's LONG_SQUARES_START/END globals are hard-coded to the
        Allen 1.02-2.02 s epoch; measure instead of trusting, and let the caller
        reconcile via ``reset_long_squares_start``.
        """
        d = self.get_sweep(sweep_number)
        i = d["stimulus"]
        dt = 1.0 / d["sampling_rate"]
        k = int(skip_s / dt)
        if len(i) <= k + 10:
            return None

        seg = i[k:]
        base = np.median(seg)
        dev = np.abs(seg - base)
        peak = np.nanmax(dev)
        if not np.isfinite(peak) or peak <= 0:
            return None
        above = dev > frac * peak
        if not above.any():
            return None

        # longest contiguous run of True
        edges = np.diff(np.concatenate(([0], above.view(np.int8), [0])))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        if starts.size == 0:
            return None
        lengths = ends - starts
        j = int(np.argmax(lengths))
        if lengths[j] * dt < min_dur_s:
            return None
        return (float((k + starts[j]) * dt), float((k + ends[j]) * dt))

    def sweep_duration(self, sweep_number: int) -> float:
        d = self.get_sweep(sweep_number)
        return len(d["response"]) / d["sampling_rate"]
