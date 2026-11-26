from collections.abc import Sequence
from textwrap import shorten

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from analysis import utils
from analysis.plotting import buckets, draw, output, views


def _time_heatmap(
    data: pd.DataFrame,
    *,
    title: str,
    color_label: str,
) -> Figure:
    """Draw a dated matrix with twelve sampled date labels."""
    matrix = data.T.rename(columns=lambda date: date.strftime("%Y-%m"))
    fig, ax = plt.subplots(figsize=(14, max(5, len(data.columns) * 0.32)))
    draw.heatmap(matrix, ax, title=title, color_label=color_label, x_tick_count=12)
    return output.show(fig)


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

    return _time_heatmap(data, title=title, color_label="Monthly return")


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
    rolling = utils.expect(rolling.dropna(axis=1, how="all"), pd.DataFrame)
    if rolling.empty:
        raise ValueError("No rolling return data available.")

    return _time_heatmap(
        rolling,
        title=f"Rolling {window}-month returns" if title is None else title,
        color_label="Return",
    )


def correlation_heatmap(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    title: str = "Correlation with fund and index returns",
) -> Figure:
    values, bucket_values = views.correlation_view(frame, columns)
    display = values.rename(
        columns=lambda name: shorten(str(name), width=18, placeholder="…")
        if name
        else ""
    )
    fig = buckets.bucketed_matrix(
        display,
        bucket_values=bucket_values,
        title=title,
        color_label="Correlation",
        unit=" mo",
        header="Overlap",
        figsize=(14, max(8, len(values.index) * 0.32)),
    )
    return output.show(fig)


def factor_exposure_heatmap(
    exposures: pd.DataFrame,
    *,
    sort_funds: bool = True,
    title: str = "Factor exposures",
) -> Figure:
    values, labels = views.exposure_view(exposures, sort_funds=sort_funds)
    fig, ax = plt.subplots(figsize=(15, max(7, len(values) * 0.45)))
    draw.heatmap(
        values,
        ax,
        title=title,
        color_label="pp per 1 SD factor move (fixed common-window SD)",
        cmap="RdBu_r",
    )
    draw.annotate(ax, labels)
    return output.show(fig)


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
        # Slice after the anchoring call above so this window never removes the anchor.
        values = values.loc[pd.Timestamp(start) :]
    fig = draw.lines(values, title=title, ylabel="Index (100 = anchor)", logy=True)
    return output.show(fig)


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
    return output.show(fig)


def annual_deviations(
    deviations: pd.DataFrame,
    *,
    tol: float,
    title: str = "Monthly-compounded minus reported annual return",
) -> Figure:
    values = deviations.T.rename(columns=str)
    labels = values.map(lambda value: f"{value:.2f}" if abs(value) > tol else "")
    fig, ax = plt.subplots(figsize=(13, 8))
    draw.heatmap(values, ax, title=title, color_label="pp", limit=tol)
    draw.annotate(ax, labels)
    return output.show(fig)
