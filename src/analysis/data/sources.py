from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import columns as col, utils
from analysis.data import _store as data_store


# Raw source files and row formats
GSW_FILE: str = "feds200628.csv"
GSW_HEADER_ROW: int = 9
MONTH_FORMAT: str = "%Y-%m-%d"
YEAR_PATTERN: str = r"\d{4}"


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceTable:
    data: pd.DataFrame
    metadata: pd.DataFrame

    def metadata_row(self, name: str) -> pd.Series:
        rows = self.metadata.loc[self.metadata.index == name]
        if rows.empty:
            raise ValueError(f"Missing metadata row: {name}")
        return rows.iloc[0]

    def percent_row(self, name: str) -> pd.Series:
        return self.metadata_row(name).map(_parse_percent).reindex(self.data.columns)


@dataclass(frozen=True)
class Sources:
    fund_monthly: SourceTable
    fund_yearly: SourceTable
    asset_classes: SourceTable
    gsw_yields: SourceTable


@dataclass(frozen=True)
class SourceSpec:
    name: str
    file_name: str
    parse: SourceParser
    metadata_rows: tuple[str, ...] = ()
    skip_rows: int = 0

    def load(self, raw: data_store.CsvStore) -> SourceTable:
        return self.parse(raw.read(self.file_name, skiprows=self.skip_rows), self)


SourceParser = Callable[[pd.DataFrame, SourceSpec], SourceTable]
RowKey = Callable[[pd.Series], tuple[pd.Series, pd.Index]]


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------
def _parse_percent(value: object) -> float:
    if value is None or value is pd.NA or value is pd.NaT:
        return np.nan
    if isinstance(value, (float, np.floating)) and bool(np.isnan(value)):
        return np.nan

    text = str(value).strip()
    if not text:
        return np.nan

    number = float(text.removesuffix("%").replace(",", ""))
    return number / 100 if text.endswith("%") else number


def _parse_number(value: object) -> float:
    if value is None or value is pd.NA or value is pd.NaT:
        return np.nan

    text = str(value).strip()
    if not text or text == "NA":
        return np.nan

    return float(text)


# ---------------------------------------------------------------------------
# Row keys
# ---------------------------------------------------------------------------
def _month_key(labels: pd.Series) -> tuple[pd.Series, pd.Index]:
    dates = pd.to_datetime(labels, format=MONTH_FORMAT, errors="coerce")
    valid = dates.notna()
    return valid, pd.DatetimeIndex(dates.loc[valid], name="Month")


def _year_key(labels: pd.Series) -> tuple[pd.Series, pd.Index]:
    valid = labels.str.fullmatch(YEAR_PATTERN, na=False)
    return valid, pd.Index(labels.loc[valid].astype(int), name="Year")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _metadata(
    raw: pd.DataFrame,
    spec: SourceSpec,
    label: str,
    columns: pd.Index,
) -> pd.DataFrame:
    selected = raw.loc[raw[label].isin(spec.metadata_rows)]
    metadata = selected.set_index(label)[columns]

    missing = [name for name in spec.metadata_rows if name not in metadata.index]
    if missing:
        raise ValueError(f"Missing metadata rows in {spec.file_name}: {missing}")

    duplicated = metadata.index[metadata.index.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"Duplicate metadata rows in {spec.file_name}: {duplicated}")

    return metadata


def _parse_table(raw: pd.DataFrame, spec: SourceSpec, key: RowKey) -> SourceTable:
    label = str(raw.columns[0])
    columns = raw.columns[1:]
    metadata = _metadata(raw, spec, label, columns)

    labels = utils.expect(raw[label], pd.Series)
    valid, index = key(labels)
    data = raw.loc[valid, columns].map(_parse_percent)
    data.index = index

    return SourceTable(data.sort_index(), metadata)


def parse_monthly_table(raw: pd.DataFrame, spec: SourceSpec) -> SourceTable:
    return _parse_table(raw, spec, _month_key)


def parse_yearly_table(raw: pd.DataFrame, spec: SourceSpec) -> SourceTable:
    return _parse_table(raw, spec, _year_key)


def parse_gsw_yields(raw: pd.DataFrame, spec: SourceSpec) -> SourceTable:
    utils.expect_columns(raw, [col.GSW_DATE, *col.GSW_ZERO_YIELDS])

    dates = pd.to_datetime(raw[col.GSW_DATE], format=MONTH_FORMAT, errors="coerce")
    data = raw.loc[dates.notna(), list(col.GSW_ZERO_YIELDS)].map(_parse_number)
    data.index = pd.DatetimeIndex(dates.loc[dates.notna()], name="Date")

    return SourceTable(data.sort_index(), pd.DataFrame())


# ---------------------------------------------------------------------------
# Source definitions
# ---------------------------------------------------------------------------
FUND_MONTHLY: SourceSpec = SourceSpec(
    "fund_monthly",
    "monthly.csv",
    parse_monthly_table,
    (col.MANAGEMENT_FEE, col.NORM_MANAGEMENT_FEE),
)
FUND_YEARLY: SourceSpec = SourceSpec(
    "fund_yearly",
    "yearly.csv",
    parse_yearly_table,
)
ASSET_CLASSES: SourceSpec = SourceSpec(
    "asset_classes",
    "asset_classes.csv",
    parse_monthly_table,
)
GSW_YIELDS: SourceSpec = SourceSpec(
    "gsw_yields",
    GSW_FILE,
    parse_gsw_yields,
    skip_rows=GSW_HEADER_ROW,
)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def _reorder_columns(table: SourceTable, columns: pd.Index) -> SourceTable:
    """Apply one column order to a table's data and metadata."""
    return SourceTable(
        table.data.reindex(columns=columns),
        table.metadata.reindex(columns=columns),
    )


def load(directory: Path) -> Sources:
    raw = data_store.CsvStore(directory)
    monthly = FUND_MONTHLY.load(raw)
    # Deterministic fund columns: first observed month, newest first, unobserved last.
    applied = monthly.data.apply(lambda column: column.first_valid_index())
    first_records = pd.DataFrame(applied).iloc[:, 0]
    order = first_records.sort_values(ascending=False, na_position="last").index
    return Sources(
        fund_monthly=_reorder_columns(monthly, order),
        fund_yearly=_reorder_columns(FUND_YEARLY.load(raw), order),
        asset_classes=ASSET_CLASSES.load(raw),
        gsw_yields=GSW_YIELDS.load(raw),
    )
