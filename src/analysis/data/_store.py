from pathlib import Path

import pandas as pd


class CsvStore:
    """Read and write CSV files in an existing directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def read(self, name: str, *, skiprows: int = 0) -> pd.DataFrame:
        return pd.read_csv(self.directory / name, dtype=str, skiprows=skiprows)

    def write(
        self,
        frame: pd.DataFrame,
        name: str,
        *,
        index_label: str | None = None,
        float_format: str | None = None,
    ) -> Path:
        path = self.directory / name
        frame.to_csv(path, index_label=index_label, float_format=float_format)
        return path
