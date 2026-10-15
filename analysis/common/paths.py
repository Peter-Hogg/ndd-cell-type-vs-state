"""Repository paths.

Raw public data sit at the repository root (see README). Each dataset folder under analysis/
holds its own scripts/, results/, figures/ and data/ (derived caches). Scripts reuse each
other's functions by module name, so importing this module puts every scripts/ folder on
sys.path.
"""
import sys
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[1]
REPO = ANALYSIS.parent

M1 = ANALYSIS / "m1_patchseq"            # Scala et al. 2021 mouse M1 Patch-seq
ATLAS = ANALYSIS / "allen_atlas"         # Allen mouse cortex SMART-seq atlas, MOp cells/nuclei
V1 = ANALYSIS / "v1_patchseq"            # mouse V1 Patch-seq (Sorensen et al.; Gouwens et al. 2020)
MACAQUE = ANALYSIS / "macaque_patchseq"  # Dembrow et al. macaque Patch-seq
IBL = ANALYSIS / "ibl_neuropixels"       # IBL Brain-Wide Map (DANDI:000409)
CROSS = ANALYSIS / "cross_dataset"       # analyses combining two or more datasets
VALIDATION = CROSS / "m1_v1_validation"  # harmonised, animal-aware M1 -> V1 set test

for _d in [ANALYSIS / "common", *sorted(ANALYSIS.glob("*/scripts")), *sorted(ANALYSIS.glob("*/*/scripts"))]:
    if str(_d) not in sys.path:
        sys.path.append(str(_d))


def _folder(script) -> Path:
    return Path(script).resolve().parents[1]          # <dataset>/scripts/x.py -> <dataset>


def _sub(script, name) -> Path:
    d = _folder(script) / name
    d.mkdir(exist_ok=True)
    return d


def results(script) -> Path:
    return _sub(script, "results")


def figures(script) -> Path:
    return _sub(script, "figures")


def data(script) -> Path:
    return _sub(script, "data")


def main_figures() -> Path:
    """Figures folder of the script being run (used by the shared plotting styles)."""
    return figures(sys.modules["__main__"].__file__)
