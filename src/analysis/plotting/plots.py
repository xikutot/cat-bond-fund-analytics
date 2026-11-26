from collections.abc import Sequence
from textwrap import shorten

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from analysis import utils
from analysis.plotting import draw, output, views


def monthly_heatmap(
    returns: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    start: str | pd.Timestamp | None = None,
    title: str = "Monthly returns",
) -> Figure:
    data = views.window(returns, columns, start).dropna(axis=1, how="all")
    if data.empty:
        raise ValueError("No return data available for the heatmap.")

    fig = draw.heatmap(
        data.T,
        title=title,
        color_label="Monthly return",
        figsize=(14, max(5, len(data.columns) * 0.32)),
        date_columns=True,
        x_rotation=45,
        x_ha="right",
    )
    output.show(fig)
    return fig


def rolling_returns_heatmap(
    returns: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    window: int = 12,
    start: str | pd.Timestamp | None = None,
    title: str | None = None,
) -> Figure:
    data = views.window(returns, columns, start)
    rolling = (
        (1 + data)
        .rolling(window, min_periods=window)
        .apply(np.prod, raw=True)
        .sub(1)
    )
    rolling = rolling.dropna(axis=1, how="all")
    if rolling.empty:
        raise ValueError("No rolling return data available.")

    fig = draw.heatmap(
        rolling.T,
        title=f"Rolling {window}-month returns" if title is None else title,
        color_label="Return",
        figsize=(14, max(5, len(rolling.columns) * 0.32)),
        date_columns=True,
        x_rotation=45,
        x_ha="right",
    )
    output.show(fig)
    return fig


def correlation_heatmap(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    title: str = "Correlation with fund and index returns",
) -> Figure:
    values, bucket_values = views.correlation_view(frame, columns)
    fig = draw.bucketed_matrix(
        values,
        bucket_values=bucket_values,
        title=title,
        color_label="Correlation",
        unit=" mo",
        header="Overlap",
        figsize=(14, max(8, len(values.index) * 0.32)),
        column_labels=[
            shorten(str(name), width=18, placeholder="…") if name else ""
            for name in values.columns
        ],
        x_rotation=45,
        x_ha="right",
        x_labelsize=8,
    )
    output.show(fig)
    return fig


def factor_exposure_heatmap(
    exposures: pd.DataFrame,
    *,
    sort_funds: bool = True,
    title: str = "Factor exposures",
) -> Figure:
    values, labels, colors, limit = views.exposure_view(exposures, sort_funds=sort_funds)
    fig = draw.heatmap(
        values,
        title=title,
        color_label="pp per 1 SD factor move (fixed common-window SD)",
        limit=limit,
        cmap=draw.DIVERGING_CMAP,
        figsize=(15, max(7, len(values) * 0.45)),
        x_rotation=45,
        x_ha="right",
        y_labelsize=8,
        cell_labels=labels,
        cell_colors=colors,
    )
    output.show(fig)
    return fig


def anchored_growth(
    returns: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    anchor: str | pd.Timestamp,
    start: str | pd.Timestamp | None = None,
    title: str = "Growth of 100 at anchor",
) -> Figure:
    data = views.window(returns, columns)
    anchor_date = utils.expect(
        pd.Timestamp(anchor) + pd.offsets.MonthEnd(0), pd.Timestamp
    )
    values = views.anchored_growth(data, anchor_date)
    if start is not None:
        values = values.loc[pd.Timestamp(start) :]
    fig = draw.lines(values, title=title, ylabel="Index (100 = anchor)", logy=True)
    output.show(fig)
    return fig


def drawdowns(
    returns: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    start: str | pd.Timestamp | None = None,
    title: str = "Drawdowns",
) -> Figure:
    data = views.window(returns, columns, start)
    growth = (1 + data).cumprod()
    series = growth.div(growth.cummax()).sub(1)
    fig = draw.lines(
        series,
        title=title,
        ylabel="Drawdown",
        percent=True,
        figsize=(13, 5),
    )
    output.show(fig)
    return fig


def annual_deviations(
    deviations: pd.DataFrame,
    *,
    tol: float,
    title: str = "Monthly-compounded minus reported annual return",
) -> Figure:
    values = deviations.T
    labels = values.map(lambda value: f"{value:.2f}" if abs(value) > tol else "")
    fig = draw.heatmap(
        values,
        title=title,
        color_label="pp",
        limit=tol,
        figsize=(13, 8),
        x_rotation=90,
        column_labels=[str(year) for year in deviations.index],
        cell_labels=labels,
    )
    output.show(fig)
    return fig
