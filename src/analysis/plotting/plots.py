from collections.abc import Sequence
from textwrap import shorten

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import (
    FormatStrFormatter,
    Formatter,
    FuncFormatter,
    PercentFormatter,
)

from analysis.plotting import buckets, draw, output, views


def _return_format() -> Formatter:
    return PercentFormatter(xmax=1, decimals=1)


def _time_heatmap(
    data: pd.DataFrame,
    *,
    title: str,
    color_label: str,
    cmap: str | Colormap = draw.HEATMAP_CMAP,
    norm: Normalize | None = None,
    colorbar_format: Formatter | None = None,
    colorbar_ticks: Sequence[float] | None = None,
) -> Figure:
    """Draw a dated matrix with twelve sampled date labels."""
    matrix = data.T.rename(columns=lambda date: date.strftime("%Y-%m"))
    fig, ax = plt.subplots(figsize=(14, max(5, len(data.columns) * 0.32)))
    draw.heatmap(
        matrix,
        ax,
        title=title,
        color_label=color_label,
        cmap=cmap,
        norm=norm,
        colorbar_format=colorbar_format,
        colorbar_ticks=colorbar_ticks,
        x_tick_count=12,
    )
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

    return _time_heatmap(
        data,
        title=title,
        color_label="Monthly return",
        colorbar_format=_return_format(),
    )


def rolling_returns(
    returns: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    window: int = 12,
    start: str | pd.Timestamp | None = None,
    title: str = "Rolling {window}-month returns",
) -> Figure:
    values = views.rolling_returns(
        returns, window_months=window, columns=columns, start=start
    )

    fig = draw.lines(
        values,
        title=title.format(window=window),
        ylabel=f"Rolling {window}-month return",
        percent=True,
    )
    return output.show(fig)


def rolling_returns_heatmap(
    returns: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    window: int = 12,
    start: str | pd.Timestamp | None = None,
    title: str = "Rolling {window}-month returns",
) -> Figure:
    rolling = views.rolling_returns(
        returns, window_months=window, columns=columns, start=start
    )

    return _time_heatmap(
        rolling,
        title=title.format(window=window),
        color_label="Return",
        colorbar_format=_return_format(),
    )


def correlation_heatmap(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    title: str = "Correlation matrix",
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
    title: str = "Growth rebased to 100 at {anchor}",
) -> Figure:
    values = views.anchored_growth(returns, anchor, columns=columns, start=start)
    fig = draw.lines(
        values,
        title=title.format(anchor=anchor),
        ylabel="Index (100 = anchor)",
        logy=True,
    )
    return output.show(fig)


def drawdowns(
    returns: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    start: str | pd.Timestamp | None = None,
    title: str = "Drawdowns",
) -> Figure:
    values = views.drawdowns(returns, columns=columns, start=start)
    fig = draw.lines(
        values,
        title=title,
        ylabel="Drawdown",
        percent=True,
        figsize=(13, 5),
    )
    return output.show(fig)


def drawdown_heatmap(
    returns: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    start: str | pd.Timestamp | None = None,
    title: str = "Drawdowns",
) -> Figure:
    values = views.drawdowns(returns, columns=columns, start=start).add(1)
    norm = draw.log_norm(
        values,
        vmax=1.0,
        error="Drawdowns must exceed -100% for a log colour scale.",
    )
    return _time_heatmap(
        values,
        title=title,
        color_label="Drawdown",
        cmap="rainbow_r",
        norm=norm,
        colorbar_format=FuncFormatter(lambda value, _: f"{value - 1:.0%}"),
        colorbar_ticks=draw.log_ticks(norm),
    )


def anchored_growth_heatmap(
    returns: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    anchor: str | pd.Timestamp,
    start: str | pd.Timestamp | None = None,
    title: str = "Growth rebased to 100 at {anchor}",
) -> Figure:
    values = views.anchored_growth(returns, anchor, columns=columns, start=start)
    norm = draw.log_norm(
        values,
        vmax=100.0,
        error="Anchored growth must be positive for a log colour scale.",
    )
    return _time_heatmap(
        values,
        title=title.format(anchor=anchor),
        color_label="Index (100 = anchor)",
        cmap="RdYlGn_r",
        norm=norm,
        colorbar_format=FormatStrFormatter("%.0f"),
        colorbar_ticks=draw.log_ticks(norm),
    )


def annual_deviations(
    deviations: pd.DataFrame,
    *,
    tol: float,
    title: str = "Monthly-compounded minus reported annual return",
) -> Figure:
    values = deviations.T.rename(columns=str)
    labels = values.map(lambda value: f"{value:.2f}" if abs(value) > tol else "")
    fig, ax = plt.subplots(figsize=(13, 8))
    draw.heatmap(
        values,
        ax,
        title=title,
        color_label="pp",
        norm=draw.symlog_norm(values.to_numpy(dtype=float), linthresh=tol),
    )
    draw.annotate(ax, labels)
    return output.show(fig)
