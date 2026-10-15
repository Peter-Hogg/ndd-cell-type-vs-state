"""Does ANY non-ion-channel gene relate to intrinsic electrophysiology?

Screens every gene against every ephys feature in the L5 excitatory population,
with ion channels left IN so they can serve as the positive control: a screen
that works should rediscover channel genes, and the question is then whether
anything outside them survives.

Covariates: sequencing depth, genes detected, ex vivo hold time, and IT/ET
subtype identity (the subtype confound tested in 03b).

FDR is applied genome-wide within each feature, and the channel enrichment among
hits is tested with Fisher's exact test.

Usage:  uv run python analysis/m1_patchseq/scripts/04_ephys_wide_screen.py
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

from plasticity_common import PRIMARY_EPHYS, l5_excitatory_ids, load_ephys, load_meta

_m3 = import_module("03_gene_level_rheobase")
CHANNEL_PATTERNS, COVARS, MIN_DETECT_FRAC = (
    _m3.CHANNEL_PATTERNS, _m3.COVARS, _m3.MIN_DETECT_FRAC
)
load_full_logcpm, screen = _m3.load_full_logcpm, _m3.screen

RESULTS = paths.results(__file__)

# All 29 features in the Scala feature table, so the answer does not depend on
# which 7 were picked a priori.
EXTRA_EPHYS = [
    "AP Fano factor", "AP amplitude (mV)", "AP amplitude adaptation index",
    "AP amplitude average adaptation index", "AP coefficient of variation",
    "Afterdepolarization (mV)", "Afterhyperpolarization (mV)", "Burstiness",
    "ISI Fano factor", "ISI average adaptation index",
    "ISI coefficient of variation", "Latency (ms)",
    "Latency @ +20pA current (ms)", "Max number of APs",
    "Membrane time constant (ms)", "Rebound (mV)", "Rebound number of APs",
    "Sag area (mV*s)", "Sag ratio", "Sag time (s)",
    "Upstroke-to-downstroke ratio", "Wildness",
]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    l5 = l5_excitatory_ids(meta)
    ephys = load_ephys()

    log_cpm, qc = load_full_logcpm(l5)

    # Keep channels IN; only drop spike-ins, model genes and sparse genes.
    g = log_cpm.columns
    drop = (g.str.startswith("ERCC-")
            | g.str.match(r"^(Gm\d|.*Rik$|LOC\d|A[A-Z]\d{6}|BC\d{6})")
            | ((log_cpm > 0).mean(axis=0) < MIN_DETECT_FRAC).to_numpy())
    expr = log_cpm.loc[:, ~drop]
    is_channel = pd.Series(
        expr.columns.str.match("|".join(CHANNEL_PATTERNS)), index=expr.columns
    )
    print(f"[genes] {expr.shape[1]} tested, of which "
          f"{int(is_channel.sum())} ion channel / auxiliary subunit")

    design = meta.loc[expr.index].join(qc, how="left")
    cov = design[COVARS].copy()
    cov["is_ET"] = (design["RNA family"] == "ET").astype(float)
    print(f"[cells] {len(expr)}  covariates: {list(cov.columns)}\n")

    features = [f for f in PRIMARY_EPHYS + EXTRA_EPHYS if f in ephys.columns]
    summary, all_hits = [], []

    for feat in features:
        y = ephys[feat].reindex(expr.index)
        if y.notna().sum() < 40:
            continue
        res = screen(expr, y, covars=cov)
        res["is_channel"] = res["gene"].map(is_channel).fillna(False)

        hits = res[res["q"] < 0.05]
        n_hit = len(hits)
        n_hit_chan = int(hits["is_channel"].sum())
        n_hit_nonchan = n_hit - n_hit_chan

        # Fisher: are channels over-represented among the hits?
        odds, p_enrich = (np.nan, np.nan)
        if n_hit > 0:
            tab = [[n_hit_chan, n_hit_nonchan],
                   [int(is_channel.sum()) - n_hit_chan,
                    int((~is_channel).sum()) - n_hit_nonchan]]
            odds, p_enrich = stats.fisher_exact(tab)

        top = res.reindex(res["rho"].abs().sort_values(ascending=False).index)
        top20 = top.head(20)

        summary.append({
            "feature": feat,
            "n": int(res["n"].iloc[0]),
            "hits_q05": n_hit,
            "hits_channel": n_hit_chan,
            "hits_nonchannel": n_hit_nonchan,
            "channel_frac_top20": float(top20["is_channel"].mean()),
            "best_rho": float(top["rho"].iloc[0]),
            "best_gene": str(top["gene"].iloc[0]),
            "best_is_channel": bool(top["is_channel"].iloc[0]),
            "min_q": float(res["q"].min()),
            "channel_enrich_odds": odds,
            "channel_enrich_p": p_enrich,
        })
        if n_hit:
            h = hits.copy()
            h["feature"] = feat
            all_hits.append(h)

    s = pd.DataFrame(summary).sort_values("hits_q05", ascending=False)
    s.to_csv(RESULTS / "H_ephys_wide_summary.csv", index=False)
    print("=" * 96)
    print("Genome-wide screen per ephys feature (channels included, "
          "subtype + technical covariates removed)")
    print("=" * 96)
    show = ["feature", "n", "hits_q05", "hits_channel", "hits_nonchannel",
            "best_gene", "best_rho", "best_is_channel", "min_q"]
    print(s[show].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\n[write] {RESULTS/'H_ephys_wide_summary.csv'}")

    tot = int(s["hits_q05"].sum())
    print(f"\nTotal gene x feature associations at q<0.05: {tot}")
    print(f"  ion channel / subunit : {int(s['hits_channel'].sum())}")
    print(f"  everything else       : {int(s['hits_nonchannel'].sum())}")
    n_feat_any = int((s["hits_q05"] > 0).sum())
    print(f"Features with any hit: {n_feat_any} of {len(s)}")

    if all_hits:
        H = pd.concat(all_hits, ignore_index=True)
        H.to_csv(RESULTS / "H_ephys_wide_hits.csv", index=False)
        print(f"[write] {RESULTS/'H_ephys_wide_hits.csv'}")

        print("\n--- non-channel hits (q<0.05), strongest first ---")
        nc = H[~H["is_channel"]].reindex(
            H[~H["is_channel"]]["rho"].abs().sort_values(ascending=False).index)
        if len(nc):
            print(nc.head(40)[["feature", "gene", "rho", "q"]]
                  .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        else:
            print("  none")

        print("\n--- channel hits (q<0.05), strongest first ---")
        ch = H[H["is_channel"]].reindex(
            H[H["is_channel"]]["rho"].abs().sort_values(ascending=False).index)
        if len(ch):
            print(ch.head(25)[["feature", "gene", "rho", "q"]]
                  .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        else:
            print("  none")

        print("\n--- channel enrichment among hits (Fisher) ---")
        e = s[s["hits_q05"] > 0][
            ["feature", "hits_q05", "hits_channel", "channel_enrich_odds",
             "channel_enrich_p"]]
        print(e.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    else:
        print("\nNo gene reaches q<0.05 for ANY ephys feature.")
        print("Reporting the best per feature instead — see the table above.")


if __name__ == "__main__":
    main()
