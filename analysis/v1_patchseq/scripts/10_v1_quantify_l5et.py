"""Quantify mouse V1 L5 ET Patch-seq transcriptomes (Sorensen et al. 2026).

The Sorensen et al. deposit ships processed electrophysiology (rheobase and 255
other features, in ReplicationData/MouseV1_Sorensen2026/) but only raw paired-end
FASTQ for the transcriptomes, on NeMO. Jun is absent from the 108-gene Hrvatin
response-gene list used in that deposit, so transcriptomes are quantified here.

Scope: the 436 cells whose `inferred_met_type` starts with "L5 ET" (28 GB of
FASTQ, versus 103 GB for all 1,528).

Staged per cell so peak disk stays ~100 MB:
    download tar -> extract R1/R2 -> salmon quant -> aggregate to gene
    -> write compact parquet -> DELETE fastq and tar

Resumable: cells with an existing output parquet are skipped.

Quantification: salmon selective alignment against the GENCODE vM25 (mm10) cDNA
transcriptome, k=25 for 51 bp reads. This is cDNA-only, so it corresponds to the
mouse M1 *exon-only* quantification rather than exon+intron.

Usage:
    uv run python analysis/v1_patchseq/scripts/10_v1_quantify_l5et.py --limit 1
    uv run python analysis/v1_patchseq/scripts/10_v1_quantify_l5et.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  (also puts every analysis/*/scripts folder on sys.path)

import argparse
import os
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = paths.REPO
REPL = ROOT / "ReplicationData" / "MouseV1_Sorensen2026"
MANIFEST = REPL / "l5et_fastq_manifest.csv"
QUANT = REPL / "quant"
WORK = REPL / "_work"
SALMON = ROOT / "tools" / "salmon-latest_linux_x86_64" / "bin" / "salmon"
SALMON_LIB = ROOT / "tools" / "salmon-latest_linux_x86_64" / "lib"
INDEX = ROOT / "ref" / "salmon_idx_vM25_k25"
TX2GENE = ROOT / "ref" / "tx2gene_vM25.tsv"


def salmon_env() -> dict:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{SALMON_LIB}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def download(uri: str, dest: Path, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(uri, timeout=600) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            return True
        except Exception as e:
            print(f"      download attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    return False


def extract_pair(tar_path: Path, out_dir: Path) -> tuple[Path, Path] | None:
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".fastq.gz")]
        if len(members) != 2:
            print(f"      expected 2 fastq.gz, found {len(members)}")
            return None
        for m in members:
            m.name = Path(m.name).name          # flatten
            tf.extract(m, out_dir, filter="data")
    r1 = next((p for p in out_dir.glob("*.R1.fastq.gz")), None)
    r2 = next((p for p in out_dir.glob("*.R2.fastq.gz")), None)
    return (r1, r2) if r1 and r2 else None


def quantify(r1: Path, r2: Path, out_dir: Path, threads: int) -> bool:
    cmd = [
        str(SALMON), "quant", "-i", str(INDEX), "-l", "A",
        "-1", str(r1), "-2", str(r2),
        "-p", str(threads), "--validateMappings", "--quiet",
        "-o", str(out_dir),
    ]
    res = subprocess.run(cmd, env=salmon_env(), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"      salmon failed: {res.stderr.strip()[-400:]}")
        return False
    return (out_dir / "quant.sf").exists()


def to_gene_level(quant_sf: Path, t2g: pd.DataFrame) -> pd.DataFrame:
    q = pd.read_csv(quant_sf, sep="\t")
    q = q.merge(t2g, left_on="Name", right_on="tx", how="left")
    unmapped = q["gene"].isna().sum()
    if unmapped:
        q["gene"] = q["gene"].fillna(q["Name"])
    g = (q.groupby("gene", observed=True)[["NumReads", "TPM"]]
         .sum().reset_index())
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only N cells")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--keep-fastq", action="store_true",
                    help="do not delete FASTQ after quantifying (debugging only)")
    args = ap.parse_args()

    for p, what in ((SALMON, "salmon binary"), (INDEX, "salmon index"),
                    (TX2GENE, "tx2gene map"), (MANIFEST, "L5 ET manifest")):
        if not p.exists():
            sys.exit(f"missing {what}: {p}")

    QUANT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    t2g = pd.read_csv(TX2GENE, sep="\t", header=None,
                      names=["tx", "gene", "gene_id"])[["tx", "gene"]]

    man = pd.read_csv(MANIFEST)
    todo = [r for _, r in man.iterrows()
            if not (QUANT / f"{r['specimen_id']}.parquet").exists()]
    if args.limit:
        todo = todo[: args.limit]

    done_already = len(man) - len([r for _, r in man.iterrows()
                                   if not (QUANT / f"{r['specimen_id']}.parquet").exists()])
    print(f"L5 ET cells in manifest : {len(man)}")
    print(f"already quantified      : {done_already}")
    print(f"to process this run     : {len(todo)}")
    print(f"threads                 : {args.threads}\n", flush=True)

    ok = fail = 0
    t_start = time.time()
    for k, r in enumerate(todo, 1):
        sid = str(r["specimen_id"])
        cell_work = WORK / sid
        if cell_work.exists():
            shutil.rmtree(cell_work)
        cell_work.mkdir(parents=True)
        tar_path = cell_work / str(r["file_name"])

        el = time.time() - t_start
        eta = (el / max(k - 1, 1)) * (len(todo) - k + 1) if k > 1 else 0
        print(f"[{k}/{len(todo)}] {sid}  {r['inferred_met_type']}"
              f"   elapsed {el/60:.1f}m  eta {eta/60:.0f}m", flush=True)

        try:
            if not download(str(r["archive_uri"]), tar_path):
                fail += 1
                continue
            pair = extract_pair(tar_path, cell_work)
            if not pair:
                fail += 1
                continue
            tar_path.unlink()                      # tar no longer needed
            sdir = cell_work / "salmon"
            if not quantify(pair[0], pair[1], sdir, args.threads):
                fail += 1
                continue
            g = to_gene_level(sdir / "quant.sf", t2g)
            g["specimen_id"] = sid
            g.to_parquet(QUANT / f"{sid}.parquet", index=False)
            tot = g["NumReads"].sum()
            jun = float(g.loc[g["gene"] == "Jun", "TPM"].sum())
            print(f"      genes={len(g)}  total_reads={tot:,.0f}  Jun TPM={jun:.2f}",
                  flush=True)
            ok += 1
        finally:
            if not args.keep_fastq and cell_work.exists():
                shutil.rmtree(cell_work, ignore_errors=True)

    print(f"\ndone: {ok} ok, {fail} failed, "
          f"{time.time()-t_start:.0f}s total")
    print(f"outputs in {QUANT}")


if __name__ == "__main__":
    main()
