from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import pandas as pd

from analysis import columns as col, utils
from analysis.data import panels as data_panels


def _exposure_metric(
    exposures: pd.DataFrame,
    factor: str,
    column: str,
) -> pd.Series:
    rows = exposures.loc[exposures[col.FACTOR] == factor]
    return utils.expect(rows.set_index(col.FUND)[column], pd.Series)


def _max_drawdown(returns: pd.Series) -> float:
    values = utils.expect(returns.dropna(), pd.Series)
    growth = (1 + values).cumprod()
    if growth.empty:
        return float("nan")
    return float((growth / growth.cummax() - 1).min() * 100)


def _covariance(returns: pd.Series, proxy: pd.Series) -> float:
    values = utils.expect(returns.dropna(), pd.Series)
    aligned = proxy.reindex(values.index)
    pair = pd.concat([values, aligned], axis=1).dropna()
    if pair.empty:
        return float("nan")
    return float(pair.iloc[:, 0].cov(pair.iloc[:, 1]))


def scorecard(
    *,
    net_returns: pd.DataFrame,
    cash_returns: pd.Series,
    excess_panel: pd.DataFrame,
    funds: Sequence[str],
    cat_exposures: pd.DataFrame,
    control_exposures: pd.DataFrame,
    target: str,
    plenum: str,
    intelligence: str,
    financing_spread: float,
    equity_weight: float,
    cta_weight: float,
    min_months: int,
    min_cat_t: float,
    max_control_t: float,
) -> pd.DataFrame:
    """Turn exposure estimates and net returns into a pass/fail screening table."""
    utils.expect_columns(net_returns, list(funds))
    utils.expect_columns(excess_panel, [col.MSCI_ACWI_INDEX, col.SG_CTA_INDEX])

    net = utils.expect(net_returns.loc[:, list(funds)], pd.DataFrame)
    cash = utils.expect(cash_returns, pd.Series)
    net_excess = data_panels.excess(net, cash)
    equity = utils.expect(excess_panel[col.MSCI_ACWI_INDEX], pd.Series)
    cta = utils.expect(excess_panel[col.SG_CTA_INDEX], pd.Series)
    proxy = utils.expect(equity_weight * equity + cta_weight * cta, pd.Series)

    data = pd.DataFrame(index=pd.Index(list(funds), name=col.FUND))
    data[col.MONTHS] = net_excess.notna().sum()
    data[col.NET_EXCESS] = net_excess.mean() * 1200
    data[col.FINANCED_EXCESS] = data[col.NET_EXCESS] - financing_spread * 100
    data[col.COV_PROXY] = net_excess.apply(_covariance, args=(proxy,)) * 1200
    data[col.MAX_DRAWDOWN] = net_excess.apply(_max_drawdown)

    metrics = {
        col.CAT_BETA: (cat_exposures, target, col.COEFFICIENT),
        col.CAT_T: (cat_exposures, target, col.T_STAT),
        col.PLENUM_BETA: (cat_exposures, plenum, col.COEFFICIENT),
        col.ILS_BETA: (cat_exposures, intelligence, col.COEFFICIENT),
        col.MSCI_T: (control_exposures, col.MSCI_EXCESS, col.T_STAT),
        col.CTA_T: (control_exposures, col.SG_CTA_EXCESS, col.T_STAT),
    }
    for label, (frame, factor, column) in metrics.items():
        data[label] = _exposure_metric(frame, factor, column).reindex(data.index)

    reasons: list[str] = []
    for fund, record in zip(data.index, data.to_dict(orient="records")):
        found: list[str] = []

        if utils.as_float(record[col.MONTHS], label=f"Months for {fund}") < min_months:
            found.append("short history")

        cat_t = utils.as_float(record[col.CAT_T], label=f"Cat t for {fund}")
        if not isfinite(cat_t) or cat_t < min_cat_t:
            found.append("no reliable CAT exposure")

        controls = []
        for column, reason in (
            (col.MSCI_T, "MSCI exposure"),
            (col.CTA_T, "CTA exposure"),
        ):
            value = utils.as_float(record[column], label=f"{column} for {fund}")
            controls.append((reason, value))

        if any(not isfinite(value) for _, value in controls):
            found.append("control exposure unverified")
        for reason, value in controls:
            if isfinite(value) and abs(value) > max_control_t:
                found.append(reason)

        financed = utils.as_float(
            record[col.FINANCED_EXCESS], label=f"Financed excess for {fund}"
        )
        if not isfinite(financed) or financed <= 0:
            found.append("below financing hurdle")

        reasons.append("; ".join(found) if found else "passes")

    data[col.DECISION] = [
        col.REJECT if reason != "passes" else col.PASS for reason in reasons
    ]
    data[col.REASON] = reasons
    return data.reset_index()


def scorecard_view(
    card: pd.DataFrame,
    *,
    decimals: int = 3,
    decision: str | None = None,
) -> pd.DataFrame:
    """Return the scorecard rounded, optionally restricted to one decision."""
    view = card if decision is None else card.loc[card[col.DECISION] == decision]
    return utils.expect(view, pd.DataFrame).round(decimals)
