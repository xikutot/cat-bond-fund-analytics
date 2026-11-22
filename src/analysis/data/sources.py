from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.data.store import CsvStore
from analysis.utils import expect


# ---------------------------------------------------------------------------
# Raw assumptions: metadata row names
# ---------------------------------------------------------------------------
MANAGEMENT_FEE: str = "Management Fee"
NORM_MANAGEMENT_FEE: str = "Norm Management Fee"

# Raw assumptions: column names
CASH: str = "Cash"

# Raw assumptions: row formats
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


@dataclass(frozen=True)
class SourceSpec:
    name: str
    file_name: str
    parse: SourceParser
    metadata_rows: tuple[str, ...] = ()

    def load(self, raw: CsvStore) -> SourceTable:
        return self.parse(raw.read(self.file_name), self)


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

    labels = expect(raw[label], pd.Series)
    valid, index = key(labels)
    data = raw.loc[valid, columns].map(_parse_percent)
    data.index = index

    return SourceTable(data.sort_index(), metadata)


def parse_monthly_table(raw: pd.DataFrame, spec: SourceSpec) -> SourceTable:
    return _parse_table(raw, spec, _month_key)


def parse_yearly_table(raw: pd.DataFrame, spec: SourceSpec) -> SourceTable:
    return _parse_table(raw, spec, _year_key)


# ---------------------------------------------------------------------------
# Source definitions
# ---------------------------------------------------------------------------
FUND_MONTHLY: SourceSpec = SourceSpec(
    "fund_monthly",
    "monthly.csv",
    parse_monthly_table,
    (MANAGEMENT_FEE, NORM_MANAGEMENT_FEE),
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


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_sources(raw: CsvStore) -> Sources:
    return Sources(
        fund_monthly=FUND_MONTHLY.load(raw),
        fund_yearly=FUND_YEARLY.load(raw),
        asset_classes=ASSET_CLASSES.load(raw),
    )
