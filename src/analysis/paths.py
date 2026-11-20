from pathlib import Path


def d(dir_path: Path) -> Path:
    """Return ``dir_path``, creating it once if it does not exist."""
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

ROOT_DIR: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = d(ROOT_DIR / "data")
RAW_DATA_DIR: Path = d(DATA_DIR / "raw")
PROCESSED_DATA_DIR: Path = d(DATA_DIR / "processed")

NOTEBOOKS: Path = d(ROOT_DIR / "notebooks")
FIGURES_DIR: Path = d(NOTEBOOKS / "figures")
