"""Proof of concept: extract rheobase from one DANDI:001455 NWB.

Checks the V1 feature-extraction path on a single 12 MB file before the full
52 GB download. Checks, in order:

  1. specimen ID and subject metadata readable
  2. sweeps classify into long square / short square / ramp, current clamp only
  3. stimulus step timing matches the extractor's hard-coded epoch
  4. cell_extractor_for_nwb runs and yields rheobase

Usage:  uv run python analysis/v1_patchseq/scripts/09_v1_extract_probe.py [path.nwb]
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)

import traceback

import numpy as np


from v1_nwb_adapter import MiesNwbDataSet  # noqa: E402

DEFAULT = "ReplicationData/MouseV1_DANDI001455/probe.nwb"


def main(path: str) -> None:
    print("=" * 76)
    print(f"V1 extraction proof of concept — {pathlib.Path(path).name}")
    print("=" * 76)

    ds = MiesNwbDataSet(path)

    # ---- 1. identity ----------------------------------------------------
    print("\n[1] identity")
    print(f"  specimen ID (join key) : {ds.specimen_id}")
    for k in ("subject_id", "genotype", "sex", "age", "species"):
        print(f"  {k:22s} : {ds.subject_field(k)}")

    # ---- 2. sweep classification ---------------------------------------
    print("\n[2] sweep classification")
    g = ds.classify_sweeps()
    for k, v in g.items():
        print(f"  {k:24s} n={len(v):3d}  {v[:12]}{' ...' if len(v) > 12 else ''}")
    names = {m["stim_name"] for m in ds.sweep_table()}
    print(f"  distinct stimulus names: {sorted(names)}")

    if not g["long_squares"]:
        print("\n  !! no current-clamp long squares — cannot compute rheobase")
        ds.close()
        return

    # ---- 3. step timing vs the extractor's assumed epoch ----------------
    print("\n[3] stimulus step timing (extractor assumes 1.02–2.02 s)")
    steps = []
    for s in g["long_squares"]:
        st = ds.detect_step(s)
        dur = ds.sweep_duration(s)
        if st:
            steps.append(st)
            print(f"  sweep {s:4d}: sweep {dur:.3f} s | step onset {st[0]:.4f} s   "
                  f"offset {st[1]:.4f} s   dur {st[1]-st[0]:.4f} s")
        else:
            print(f"  sweep {s:4d}: sweep {dur:.3f} s | no sustained step detected")
    if steps:
        on = float(np.median([s[0] for s in steps]))
        off = float(np.median([s[1] for s in steps]))
        print(f"  median onset {on:.4f} s, offset {off:.4f} s, duration {off-on:.4f} s")
        ok = abs(on - 1.02) < 0.05 and abs((off - on) - 1.0) < 0.10
        print(f"  matches the 1.02–2.02 s epoch: {ok}")
        if not ok:
            print("  -> extractor globals must be reset via "
                  "ephys_extractor.reset_long_squares_start(onset)")

    # ---- 4. run the extractor ------------------------------------------
    print("\n[4] cell_extractor_for_nwb")
    import ephys_extractor as efex

    if steps:
        on = float(np.median([s[0] for s in steps]))
        if abs(on - efex.LONG_SQUARES_START) > 1e-6:
            print(f"  resetting LONG_SQUARES_START {efex.LONG_SQUARES_START} -> {on:.4f}")
            efex.reset_long_squares_start(on)
        print(f"  epoch now {efex.LONG_SQUARES_START:.4f}–{efex.LONG_SQUARES_END:.4f} s")

    try:
        cell = efex.cell_extractor_for_nwb(
            ds, g["ramps"], g["short_squares"], g["long_squares"])
        cell.process()
        feats = cell.as_dict()
        ls = feats.get("long_squares", {})
        print("\n  ---- extracted features ----")
        print(f"  rheobase_i (pA)          : {ls.get('rheobase_i')}")
        print(f"  input_resistance (MOhm)  : {ls.get('input_resistance')}")
        # NB: the extractor returns tau in SECONDS. The project's
        # m1_patchseq_ephys_features.csv column is "Membrane time constant (ms)".
        # Any cross-cohort join must convert; do not assume these share units.
        print(f"  tau (s)                  : {ls.get('tau')}")
        print(f"  v_baseline (mV)          : {ls.get('v_baseline')}")
        print(f"  sag                      : {ls.get('sag')}")
        print(f"  fi_fit_slope             : {ls.get('fi_fit_slope')}")
        n_spiking = len([s for s in ls.get("sweeps", []) if s.get("avg_rate", 0) > 0])
        print(f"  spiking long squares     : {n_spiking}/{len(ls.get('sweeps', []))}")
        print("\n  >>> SUCCESS — rheobase extractable from MIES NWB")
    except Exception as e:
        print(f"\n  !! extractor FAILED: {type(e).__name__}: {e}")
        traceback.print_exc(limit=6)
        print("\n  >>> the adapter or the timing epoch needs adjustment")

    ds.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
