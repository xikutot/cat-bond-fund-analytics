from pathlib import Path


def d(dir_path: Path) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

# src/analysis/paths.py
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = d(ROOT_DIR / "data")
RAW_DATA_DIR = d(DATA_DIR / "raw")
PROCESSED_DATA_DIR = d(DATA_DIR / "processed")

NOTEBOOKS = d(ROOT_DIR / "notebooks")
FIGURES_DIR = d(NOTEBOOKS / "figures")
