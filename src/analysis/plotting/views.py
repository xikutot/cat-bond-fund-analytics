from collections.abc import Sequence

import numpy as np
import pandas as pd

from analysis import columns as col, utils


SIGNIFICANCE_LEVELS: tuple[tuple[float, str], ...] = (
    (0.01, "***"),
    (0.05, "**"),
    (0.10, "*"),
)


def window(
    returns: pd.DataFrame,
    columns: Sequence[str] | None,
    start: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    data = returns.sort_index()
    if columns is not None:
        data = data.loc[:, columns]
    if start is not None:
        data = data.loc[pd.Timestamp(start) :]
    return data


def significance_stars(p: float) -> str:
    for threshold, stars in SIGNIFICANCE_LEVELS:
        if p < threshold:
            return stars
    return ""


def correlation_view(
    frame: pd.DataFrame,
    columns: Sequence[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame if columns is None else frame.loc[:, columns]
    fund_columns = list(data.columns)
    index_columns = [
        name for name in frame.columns if name in col.ASSET_CLASS_COLUMNS
    ]
    names = [*fund_columns, *index_columns]
    order = [*fund_columns, "", *index_columns]
    values = pd.DataFrame(
        {name: data.corrwith(frame[name], axis=0) for name in names}
    ).T.reindex(index=order, columns=fund_columns)
    overlap_counts = pd.DataFrame(
        {
            name: data.notna().mul(frame[name].notna(), axis=0).sum()
            for name in names
        }
    ).T.reindex(index=order, columns=fund_columns)
    if values.empty:
        raise ValueError("No correlations available.")
    return values, overlap_counts


def exposure_view(
    exposures: pd.DataFrame,
    *,
    sort_funds: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    utils.expect_columns(
        exposures, (col.FUND, col.FACTOR, col.IMPACT_PER_SD, col.P_VALUE)
    )
    if exposures.empty:
        raise ValueError("No factor exposures available for the heatmap.")

    factors = [str(name) for name in dict.fromkeys(exposures[col.FACTOR])]
    impacts = exposures.pivot(
        index=col.FUND, columns=col.FACTOR, values=col.IMPACT_PER_SD
    ).reindex(columns=factors)
    pvalues = exposures.pivot(
        index=col.FUND, columns=col.FACTOR, values=col.P_VALUE
    ).reindex(columns=factors)

    if sort_funds:
        strength = impacts.abs().max(axis=1)
        order = strength.sort_values(ascending=False, kind="stable").index
        impacts = impacts.loc[order]
        pvalues = pvalues.loc[order]

    raw = impacts.to_numpy(dtype=float)
    limit = max(float(np.nanquantile(np.abs(raw), 0.95)), 0.1)
    labels = impacts.map(lambda value: f"{value:+.1f}") + pvalues.map(
        significance_stars
    )
    labels = labels.where(np.isfinite(impacts), "")
    colors = impacts.abs().gt(0.65 * limit).map(
        lambda bright: "white" if bright else "black"
    )
    return impacts, labels, colors, limit


def anchored_growth(returns: pd.DataFrame, anchor: pd.Timestamp) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    for name in returns.columns:
        values = returns[name].dropna()
        if values.empty or anchor not in values.index:
            series[name] = pd.Series(index=[anchor], dtype=float, name=name)
            continue
        growth = (1 + values).cumprod()
        series[name] = (100 / growth.loc[anchor] * growth).rename(name)
    return pd.DataFrame(series).sort_index()
