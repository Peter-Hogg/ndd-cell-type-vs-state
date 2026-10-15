"""Is the Jarid2/Gphn axis a library-composition artifact?

Anti-correlation with the most highly expressed housekeeping transcripts
(Gapdh, Ubb, Atp5b, Tpi1, Mdh1, Hspa8) is the classic signature of a
compositional effect: when housekeeping transcripts take a larger share of a
library, every other gene's CPM is pushed down, manufacturing correlations that
have nothing to do with biology.

Three composition measures are built per cell, then the Jarid2/Gphn rheobase
association is re-tested with them as covariates:

  hk_share      fraction of the library in a curated housekeeping/metabolic set
  top50_share   fraction in that cell's own 50 most abundant genes
  top500_share  fraction in its 500 most abundant genes

If the association collapses, the axis is compositional.

Reference points: Jun (activity anchor) and Kcnip1 (channel anchor) are carried
so the size of any change is interpretable.

Usage:  uv run python analysis/m1_patchseq/scripts/27_composition_artifact_test.py
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

_m3 = import_module("03_gene_level_rheobase")
_m20 = import_module("20_layer_meta_analysis")
_m22 = import_module("22_repairs_excitatory")
from plasticity_common import load_ephys, load_meta  # noqa: E402

OUT = paths.results(__file__)
RHEO = "Rheobase (pA)"
TAU = "Membrane time constant (ms)"
GENES = ["Jarid2", "Gphn", "Jun", "Kcnip1"]

# Curated housekeeping / glycolysis / OXPHOS / chaperone set. Built from the
# genes that appeared as Jarid2's negative atlas correlates plus canonical
# housekeeping, i.e. deliberately biased TOWARDS finding an artifact.
HK = ["Gapdh", "Actb", "Actg1", "Ubb", "Ubc", "Tpi1", "Mdh1", "Mdh2", "Ldha",
      "Pgk1", "Aldoa", "Eno1", "Eno2", "Pkm", "Atp5b", "Atp5a1", "Atp5f1",
      "Cox4i1", "Cox5a", "Cox6a1", "Cox8a", "Ndufa4", "Cycs", "Slc25a4",
      "Slc25a3", "Eef1a1", "Eef2", "Hspa8", "Hsp90aa1", "Hsp90ab1", "Dnaja1",
      "Calm1", "Calm2", "Calm3", "Tuba1a", "Tuba1b", "Tubb5", "Tubb2a",
      "Rplp0", "Rps18", "Rpl13a", "Rps3", "Rpl4", "Prdx2", "Ppia", "Itm2b",
      "Mllt11", "Ndfip1", "Cdc42", "Tspan3"]


def main() -> None:
    meta = load_meta()
    ephys = load_ephys()
    log_cpm, qc = _m3.load_full_logcpm(meta.index, use_introns=True)

    # ---- composition measures, from CPM recovered out of log1p ------------
    cpm = np.expm1(log_cpm.to_numpy(dtype=np.float64))
    tot = cpm.sum(axis=1, keepdims=True)
    tot[tot == 0] = np.nan
    hk_cols = [g for g in HK if g in log_cpm.columns]
    hk_share = cpm[:, [log_cpm.columns.get_loc(g) for g in hk_cols]].sum(1) / tot[:, 0]
    srt = np.sort(cpm, axis=1)[:, ::-1]
    top50 = srt[:, :50].sum(1) / tot[:, 0]
    top500 = srt[:, :500].sum(1) / tot[:, 0]
    comp = pd.DataFrame({"hk_share": hk_share, "top50_share": top50,
                         "top500_share": top500}, index=log_cpm.index)
    print(f"[composition] {len(hk_cols)}/{len(HK)} housekeeping genes found")
    print(comp.describe().loc[["mean", "std", "min", "max"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))

    expr, _ = _m3.filter_genes(log_cpm, drop_channels=False)
    d0 = meta.loc[expr.index].join(qc, how="left").join(comp, how="left")
    rt = d0["RNA type"].astype(str)
    lay = pd.Series("other", index=d0.index)
    for nm, pat in _m20.LAYER_PATTERNS.items():
        lay[rt.str.match(pat, na=False)] = nm
    d0["layer"] = lay
    idx = expr.index[(d0["layer"] != "other")]
    idx = idx[ephys[RHEO].reindex(idx).notna().values]
    d = d0.loc[idx]
    n = len(idx)
    print(f"\n[pop] {n} pyramidal cells with rheobase")

    print("\n=== does the axis itself track composition? ===")
    for g in GENES:
        if g not in expr.columns:
            continue
        row = []
        for c in comp.columns:
            r, p = stats.spearmanr(expr.loc[idx, g], d[c])
            row.append(f"{c} {r:+.3f}(p={p:.2g})")
        print(f"  {g:8s} " + "   ".join(row))

    def design(extra: list[str]):
        cov = d[_m3.COVARS + extra].copy()
        s = d["RNA type"]
        k = s.value_counts()[lambda x: x >= 8].index
        s = s.where(s.isin(k), "other")
        dums = pd.get_dummies(s, prefix="t", drop_first=True).astype(float)
        return _m22.build_design(cov, dums, n)

    models = {
        "baseline (depth, detected, hold time, type)": [],
        "+ hk_share": ["hk_share"],
        "+ top50_share": ["top50_share"],
        "+ all three composition terms": list(comp.columns),
        "+ composition + tau (cell size)": list(comp.columns),
    }

    print("\n" + "=" * 84)
    print("rho(gene, rheobase) UNDER PROGRESSIVELY STRICTER COMPOSITION CONTROL")
    print("=" * 84)
    print(f"  {'model':46s} " + "".join(f"{g:>13s}" for g in GENES))
    for label, extra in models.items():
        C = design(extra)
        y = _m22.rank_resid(ephys[RHEO].reindex(idx).to_numpy(), C)
        if "tau" in label:
            tau = _m22.rank_resid(ephys[TAU].reindex(idx).to_numpy(), C)
            b = (y * tau).sum() / (tau ** 2).sum()
            y = y - b * tau
        cells = []
        for g in GENES:
            if g not in expr.columns:
                cells.append("     n/a     ")
                continue
            gv = _m22.rank_resid(expr.loc[idx, g].to_numpy(), C)
            if "tau" in label:
                bt = (gv * tau).sum() / (tau ** 2).sum()
                gv = gv - bt * tau
            a = gv - gv.mean(); bb = y - y.mean()
            den = np.sqrt((a ** 2).sum() * (bb ** 2).sum())
            r = float((a * bb).sum() / den) if den else np.nan
            dof = n - 2 - (C.shape[1] - 1)
            t = r * np.sqrt(dof / max(1e-12, 1 - r ** 2))
            p = float(2 * stats.t.sf(abs(t), dof))
            star = "***" if p < 1e-4 else "**" if p < 0.01 else "*" if p < 0.05 else "  "
            cells.append(f"{r:+.3f}{star:>4s}  ")
        print(f"  {label:46s} " + "".join(cells))

    print("\n  *** p<1e-4   ** p<0.01   * p<0.05")
    print("\n  If Jarid2/Gphn collapse toward 0 while Kcnip1 holds, the axis is")
    print("  compositional. If they hold, it is not explained by composition.")


if __name__ == "__main__":
    main()
