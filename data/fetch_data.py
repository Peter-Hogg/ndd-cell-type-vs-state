"""Download the public input data into the folder layout the analyses expect (repository root).

    uv run python data/fetch_data.py            # core data (~4 GB): M1 Patch-seq, Allen atlas, V1 interneuron counts
    uv run python data/fetch_data.py --list     # show what would be fetched and where

Large or multi-step sources (raw NWB, FASTQ re-quantification, macaque, IBL) are not fetched here;
see the top-level README.md for each.
"""
import argparse
import gzip
import shutil
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE163nnn/GSE163764/suppl/"
BERENS = "https://raw.githubusercontent.com/berenslab/mini-atlas/master/data/"
ALLEN = "https://idk-etl-prod-download-bucket.s3.amazonaws.com/aibs_mouse_ctx-hpf_smart-seq/"
NEMO = ("https://data.nemoarchive.org/other/grant/AIBS_patchseq/transcriptome/scell/SMARTseq/"
        "processed/analysis/20200611/")
GOUWENS_META = ("https://cdn.prod.website-files.com/689cfbd308fa7373b604d290/"
                "68d61b068b5cfcfd77c92050_20200711_patchseq_metadata_mouse.csv")

# (url, destination relative to repo root, post-processing)
FILES = [
    # Mouse M1 Patch-seq, Scala et al. 2021 (expression on GEO; metadata + ephys features from the authors)
    (GEO + "GSE163764_m1_patchseq_exon_counts.csv.gz", "PatchSeqExpression/GSE163764_m1_patchseq_exon_counts.csv", "gunzip"),
    (GEO + "GSE163764_m1_patchseq_intron_counts.csv.gz", "PatchSeqExpression/GSE163764_m1_patchseq_intron_counts.csv", "gunzip"),
    (BERENS + "m1_patchseq_meta_data.csv", "PatchSeqExpression/m1_patchseq_meta_data.csv", None),
    (BERENS + "m1_patchseq_ephys_features.csv", "m1_patchseq_ephys_features.csv", None),
    # Allen mouse cortex + hippocampus SMART-seq taxonomy (2020): labels and full matrix (3.6 GB)
    (ALLEN + "metadata.csv", "Gene_Expression/Data/metadata.csv", None),
    (ALLEN + "expression_matrix.hdf5", "Gene_Expression/Data/expression_matrix.hdf5", None),
    # Mouse V1 GABAergic Patch-seq, Gouwens et al. 2020 (counts on NeMO, metadata from brain-map.org)
    (NEMO + "20200513_Mouse_PatchSeq_Release_count.v2.csv.tar", "ReplicationData/MouseV1_Gouwens2020/count.v2.csv.tar", "untar"),
    (GOUWENS_META, "ReplicationData/MouseV1_Gouwens2020/metadata_20200711.csv", None),
]


def fetch(url, dest, post):
    dest = ROOT / dest
    if dest.exists() or (post == "untar" and (dest.parent / "20200513_Mouse_PatchSeq_Release_count.v2").exists()):
        print(f"exists  {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"fetch   {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    if post == "gunzip":
        with gzip.open(tmp, "rb") as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out, length=1 << 20)
        tmp.unlink()
    elif post == "untar":
        tmp.rename(dest)
        with tarfile.open(dest) as t:
            t.extractall(dest.parent, filter="data")
    else:
        tmp.rename(dest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    for url, dest, post in FILES:
        if args.list:
            print(f"{dest:75s} <- {url}")
        else:
            fetch(url, dest, post)
