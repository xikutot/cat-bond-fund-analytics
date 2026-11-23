from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from analysis import columns as col, utils
from analysis.data import _store as data_store, sources as data_sources, treasury


@dataclass(frozen=True)
class Panel:
    name: str
    file_name: str | None
    frame: pd.DataFrame


def excess(returns: pd.DataFrame, cash: pd.Series) -> pd.DataFrame:
    """Geometrically subtract cash from each return series."""
    aligned = cash.reindex(returns.index).add(1)
    return returns.add(1).div(aligned, axis="index").sub(1)


class PanelBuilder(Protocol):
    name: str
    file_name: str | None

    def build(
        self,
        sources: data_sources.Sources,
        panels: Mapping[str, pd.DataFrame],
    ) -> Panel: ...


class TreasuryZeroReturns:
    name: str = col.TREASURY_ZERO_RETURNS
    file_name: str | None = None

    def build(
        self,
        sources: data_sources.Sources,
        panels: Mapping[str, pd.DataFrame],
    ) -> Panel:
        frame = treasury.zero_returns(sources.gsw_yields.data).reindex(
            sources.asset_classes.data.index
        )
        if frame.isna().to_numpy().any():
            raise ValueError("Missing synthetic Treasury returns")
        return Panel(self.name, self.file_name, frame)


class AssetClassReturns:
    name: str = col.ASSET_CLASS_RETURNS
    file_name: str | None = "asset_class_returns.csv"

    def build(
        self,
        sources: data_sources.Sources,
        panels: Mapping[str, pd.DataFrame],
    ) -> Panel:
        return Panel(
            self.name,
            self.file_name,
            sources.asset_classes.data.join(panels[TreasuryZeroReturns.name]),
        )


class FeeNormalizer:
    name: str = col.FEE_NORMALIZED
    file_name: str | None = "cat_bond_returns_fee_normalized.csv"

    def build(
        self,
        sources: data_sources.Sources,
        panels: Mapping[str, pd.DataFrame],
    ) -> Panel:
        monthly = sources.fund_monthly
        actual = monthly.percent_row(col.MANAGEMENT_FEE)
        target = monthly.percent_row(col.NORM_MANAGEMENT_FEE)

        missing = actual.notna() ^ target.notna()
        if missing.to_numpy().any():
            raise ValueError(
                f"Incomplete fee data: {list(monthly.data.columns[missing.to_numpy()])}"
            )

        adjustment = actual.sub(target).fillna(0).div(12)
        frame = (
            monthly.data.map(np.log1p).add(adjustment, axis="columns").map(np.expm1)
        )
        return Panel(self.name, self.file_name, frame)


class CashExcess:
    name: str = col.CASH_EXCESS
    file_name: str | None = "cat_bond_returns_cash_excess.csv"

    def build(
        self,
        sources: data_sources.Sources,
        panels: Mapping[str, pd.DataFrame],
    ) -> Panel:
        normalized = panels[FeeNormalizer.name]
        cash = utils.expect(sources.asset_classes.data[col.CASH], pd.Series).reindex(
            normalized.index
        )
        return Panel(self.name, self.file_name, excess(normalized, cash))


class TotalReturnComparison:
    name: str = col.TOTAL_RETURNS
    file_name: str | None = None

    def build(
        self,
        sources: data_sources.Sources,
        panels: Mapping[str, pd.DataFrame],
    ) -> Panel:
        funds = panels[FeeNormalizer.name]
        assets = panels[AssetClassReturns.name].drop(columns=col.CASH)
        return Panel(self.name, self.file_name, funds.join(assets))


class ExcessReturnComparison:
    name: str = col.EXCESS_RETURNS
    file_name: str | None = None

    def build(
        self,
        sources: data_sources.Sources,
        panels: Mapping[str, pd.DataFrame],
    ) -> Panel:
        funds = panels[CashExcess.name]
        asset_returns = panels[AssetClassReturns.name]
        assets = asset_returns.drop(columns=col.CASH)
        cash = utils.expect(asset_returns[col.CASH], pd.Series)
        return Panel(self.name, self.file_name, funds.join(excess(assets, cash)))


def selected_funds(
    sources: data_sources.Sources, excluded: Sequence[str]
) -> tuple[str, ...]:
    return tuple(
        name for name in sources.fund_monthly.data.columns if name not in excluded
    )


def comparison_frames(
    panels: Mapping[str, pd.DataFrame],
    funds: Sequence[str],
    indexes: Sequence[str],
    factors: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Select the frames used by the exploration chapter."""
    cash_bond = (*panels[CashExcess.name].columns, *indexes)
    return (
        utils.expect_columns(panels[ExcessReturnComparison.name], cash_bond),
        utils.expect_columns(panels[TotalReturnComparison.name], funds),
        utils.expect_columns(panels[ExcessReturnComparison.name], factors),
        cash_bond,
    )


BUILDERS: tuple[PanelBuilder, ...] = (
    TreasuryZeroReturns(),
    AssetClassReturns(),
    FeeNormalizer(),
    CashExcess(),
    TotalReturnComparison(),
    ExcessReturnComparison(),
)


def build(
    sources: data_sources.Sources,
    directory: Path,
    *,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    processed = data_store.CsvStore(directory)
    frames: dict[str, pd.DataFrame] = {}

    for builder in BUILDERS:
        panel = builder.build(sources, frames)
        frames[panel.name] = panel.frame

        if save and panel.file_name is not None:
            processed.write(
                panel.frame,
                panel.file_name,
                index_label="Month",
                float_format="%.8f",
            )

    return frames


def coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Count months with data per column, most complete first."""
    months = frame.notna().sum().rename(col.MONTHS)
    return utils.expect(months.sort_values(ascending=False).to_frame(), pd.DataFrame)
