from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from analysis.data.sources import (
    CASH,
    MANAGEMENT_FEE,
    NORM_MANAGEMENT_FEE,
    Sources,
)
from analysis.data.store import CsvStore
from analysis.utils import expect


def _excess(returns: pd.DataFrame, cash: pd.Series) -> pd.DataFrame:
    aligned = cash.reindex(returns.index).add(1)
    return returns.add(1).div(aligned, axis="index").sub(1)


@dataclass(frozen=True)
class Panel:
    name: str
    file_name: str | None
    frame: pd.DataFrame


class PanelBuilder(Protocol):
    name: str
    file_name: str | None

    def build(
        self,
        sources: Sources,
        panels: Mapping[str, pd.DataFrame],
    ) -> Panel: ...


class AssetClassReturns:
    name: str = "asset_class_returns"
    file_name: str | None = "asset_class_returns.csv"

    def build(
        self,
        sources: Sources,
        panels: Mapping[str, pd.DataFrame],
    ) -> Panel:
        return Panel(self.name, self.file_name, sources.asset_classes.data)


class FeeNormalizer:
    name: str = "fee_normalized"
    file_name: str | None = "cat_bond_returns_fee_normalized.csv"

    def build(
        self,
        sources: Sources,
        panels: Mapping[str, pd.DataFrame],
    ) -> Panel:
        monthly = sources.fund_monthly
        actual = monthly.percent_row(MANAGEMENT_FEE)
        target = monthly.percent_row(NORM_MANAGEMENT_FEE)

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
    name: str = "cash_excess"
    file_name: str | None = "cat_bond_returns_cash_excess.csv"

    def build(
        self,
        sources: Sources,
        panels: Mapping[str, pd.DataFrame],
    ) -> Panel:
        normalized = panels[FeeNormalizer.name]
        cash = expect(sources.asset_classes.data[CASH], pd.Series).reindex(
            normalized.index
        )
        return Panel(self.name, self.file_name, _excess(normalized, cash))


class TotalReturnComparison:
    name: str = "total_returns"
    file_name: str | None = None

    def build(
        self,
        sources: Sources,
        panels: Mapping[str, pd.DataFrame],
    ) -> Panel:
        funds = panels[FeeNormalizer.name]
        assets = panels[AssetClassReturns.name].drop(columns=CASH)
        return Panel(self.name, self.file_name, funds.join(assets))


class ExcessReturnComparison:
    name: str = "excess_returns"
    file_name: str | None = None

    def build(
        self,
        sources: Sources,
        panels: Mapping[str, pd.DataFrame],
    ) -> Panel:
        funds = panels[CashExcess.name]
        asset_returns = panels[AssetClassReturns.name]
        assets = asset_returns.drop(columns=CASH)
        cash = expect(asset_returns[CASH], pd.Series)
        return Panel(self.name, self.file_name, funds.join(_excess(assets, cash)))


BUILDERS: tuple[PanelBuilder, ...] = (
    AssetClassReturns(),
    FeeNormalizer(),
    CashExcess(),
    TotalReturnComparison(),
    ExcessReturnComparison(),
)


def build_panels(
    sources: Sources,
    processed: CsvStore,
    *,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
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
