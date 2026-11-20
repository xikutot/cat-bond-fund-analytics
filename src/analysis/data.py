from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


class CsvStore:
    """Read and write CSV files in an existing directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def read(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.directory / name, dtype=str)

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


@dataclass(frozen=True)
class FundData:
    returns: pd.DataFrame
    actual_fee: pd.Series
    target_fee: pd.Series


@dataclass(frozen=True)
class Panel:
    name: str
    file_name: str
    frame: pd.DataFrame


class PanelBuilder(Protocol):
    def build(
        self,
        data: FundData,
        panels: Mapping[str, pd.DataFrame],
    ) -> Panel: ...


class FeeNormalizer:
    name = "fee_normalized"
    file_name = "cat_bond_returns_fee_normalized.csv"

    def build(
        self,
        data: FundData,
        panels: Mapping[str, pd.DataFrame],
    ) -> Panel:
        actual = data.actual_fee.reindex(data.returns.columns)
        target = data.target_fee.reindex(data.returns.columns)

        missing = actual.notna() ^ target.notna()
        if missing.to_numpy().any():
            raise ValueError(
                f"Incomplete fee data: {list(data.returns.columns[missing.to_numpy()])}"
            )

        adjustment = (actual - target).fillna(0) / 12
        frame = data.returns.map(np.log1p).add(adjustment, axis="columns").map(np.expm1)
        return Panel(self.name, self.file_name, frame)


class CashExcess:
    name = "cash_excess"
    file_name = "cat_bond_returns_cash_excess.csv"

    def build(
        self,
        data: FundData,
        panels: Mapping[str, pd.DataFrame],
    ) -> Panel:
        normalized = panels[FeeNormalizer.name]
        cash = normalized["Cash"].reindex(normalized.index)
        frame = (1 + normalized.drop(columns="Cash")).div(1 + cash, axis="index") - 1
        return Panel(self.name, self.file_name, frame)


PANEL_BUILDERS: tuple[PanelBuilder, ...] = (FeeNormalizer(), CashExcess())


class ReturnProcessor:
    """Load raw fund data and run the configured panel builders."""

    FEE_ROW = "Management Fee"
    TARGET_FEE_ROW = "Norm Management Fee"
    MONTHLY_FILE = "monthly.csv"
    YEARLY_FILE = "yearly.csv"

    def __init__(
        self,
        raw: CsvStore,
        processed: CsvStore,
        builders: tuple[PanelBuilder, ...] = PANEL_BUILDERS,
    ) -> None:
        self.raw = raw
        self.processed = processed
        self.builders = builders

    @staticmethod
    def parse_percent(value: object) -> float:
        if value is None or value is pd.NA or value is pd.NaT:
            return np.nan
        if isinstance(value, (float, np.floating)) and bool(np.isnan(value)):
            return np.nan

        text = str(value).strip()
        if not text:
            return np.nan

        number = float(text.removesuffix("%").replace(",", ""))
        return number / 100 if text.endswith("%") else number

    def parse_monthly(self, raw: pd.DataFrame) -> FundData:
        label = raw.columns[0]
        columns = raw.columns[1:]

        dates = pd.to_datetime(raw[label], format="%Y-%m-%d", errors="coerce")
        is_month = dates.notna()

        returns = raw.loc[is_month, columns].map(self.parse_percent)
        returns.index = pd.DatetimeIndex(dates.loc[is_month], name="Month")
        returns = returns.sort_index()

        def fee(row_name: str) -> pd.Series:
            rows = raw.loc[raw[label].eq(row_name), columns]
            if rows.empty:
                raise ValueError(f"Missing metadata row: {row_name}")
            return rows.iloc[0].map(self.parse_percent).reindex(returns.columns)

        return FundData(returns, fee(self.FEE_ROW), fee(self.TARGET_FEE_ROW))

    def parse_yearly(self, raw: pd.DataFrame) -> pd.DataFrame:
        label = raw.columns[0]
        is_year = raw[label].str.fullmatch(r"\d{4}", na=False)
        yearly = raw.loc[is_year].set_index(label)
        yearly.index = yearly.index.astype(int)
        return yearly.map(self.parse_percent).sort_index()

    def load(self) -> FundData:
        return self.parse_monthly(self.raw.read(self.MONTHLY_FILE))

    def load_yearly(self) -> pd.DataFrame:
        return self.parse_yearly(self.raw.read(self.YEARLY_FILE))

    def run(self, *, save: bool = True) -> dict[str, pd.DataFrame]:
        data = self.load()
        frames: dict[str, pd.DataFrame] = {}

        for builder in self.builders:
            panel = builder.build(data, frames)
            frames[panel.name] = panel.frame

            if save:
                self.processed.write(
                    panel.frame,
                    panel.file_name,
                    index_label="Month",
                    float_format="%.8f",
                )

        return frames
