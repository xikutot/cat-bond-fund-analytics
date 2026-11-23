from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from analysis import columns as col, utils
from analysis.plotting import figure


SIGNIFICANCE_LEVELS: tuple[tuple[float, str], ...] = (
    (0.01, "***"),
    (0.05, "**"),
    (0.10, "*"),
)


def _stars(p: float) -> str:
    for threshold, stars in SIGNIFICANCE_LEVELS:
        if p < threshold:
            return stars
    return ""


def _heatmap(
    data: pd.DataFrame,
    *,
    title: str,
    color_label: str,
    cmap: str = "RdYlGn",
    x_rotation: int = 0,
    x_ha: str = "center",
    figsize: tuple[float, float] = (14, 8),
    column_labels: Sequence[str] | None = None,
    column_positions: Sequence[int] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[Figure, Axes]:
    values = data.to_numpy(dtype=float)
    limit = max(float(np.nanquantile(np.abs(values), 0.98)), 0.001)
    labels = list(column_labels) if column_labels is not None else list(data.columns)

    fig, ax = figure.matrix(
        values,
        row_labels=[str(name) for name in data.index],
        column_labels=labels,
        column_positions=column_positions,
        title=title,
        color_label=color_label,
        vmin=-limit if vmin is None else vmin,
        vmax=limit if vmax is None else vmax,
        figsize=figsize,
        x_rotation=x_rotation,
        x_ha=x_ha,
        cmap=cmap,
    )
    return fig, ax


def monthly_heatmap(
    returns: pd.DataFrame,
    columns: Sequence[str] | None = None,
    months: int = 36,
    title: str = "Monthly returns",
) -> tuple[Figure, Axes]:
    data = returns.sort_index().tail(months).dropna(axis=1, how="all")
    if columns is not None:
        data = data.loc[:, columns]
    if data.empty:
        raise ValueError("No return data available for the heatmap.")

    values = data.T
    index = utils.expect(data.index, pd.DatetimeIndex)
    ticks = [int(i) for i in np.arange(0, len(index), max(1, len(index) // 10))]
    fig, ax = _heatmap(
        values,
        title=title,
        color_label="Monthly return",
        figsize=(14, max(5, len(data.columns) * 0.32)),
        x_rotation=45,
        x_ha="right",
        column_labels=[utils.expect(index[i], pd.Timestamp).strftime("%Y-%m") for i in ticks],
        column_positions=ticks,
    )
    fig.tight_layout()
    figure.show(fig)
    return fig, ax


def trailing_returns_heatmap(
    returns: pd.DataFrame,
    columns: Sequence[str] | None = None,
    window: int = 12,
    display_months: int = 36,
) -> tuple[Figure, Axes]:
    data = returns.sort_index()
    if columns is not None:
        data = data.loc[:, columns]

    rolling = (
        (1 + data)
        .rolling(window, min_periods=window)
        .apply(np.prod, raw=True)
        .sub(1)
        .tail(display_months)
        .dropna(axis=1, how="all")
    )
    if rolling.empty:
        raise ValueError("No rolling return data available.")

    values = rolling.T
    dates = utils.expect(values.columns, pd.DatetimeIndex)
    labels = [utils.expect(stamp, pd.Timestamp).strftime("%Y-%m") for stamp in dates]
    values = utils.expect(values.set_axis(labels, axis="columns"), pd.DataFrame)
    fig, ax = _heatmap(
        values,
        title=f"Trailing {window}-month returns",
        color_label="Return",
        figsize=(14, max(5, len(values.index) * 0.32)),
        x_rotation=45,
        x_ha="right",
    )
    fig.tight_layout()
    figure.show(fig)
    return fig, ax


def calendar_returns_heatmap(
    returns: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> tuple[Figure, Axes]:
    data = returns.sort_index()
    if columns is not None:
        data = data.loc[:, columns]

    index = utils.expect(data.index, pd.DatetimeIndex)
    years = pd.Index(index.year, name="Year")
    compounded = (1 + data).groupby(years).prod().sub(1)
    complete = data.groupby(years).count().eq(12)
    values = compounded.where(complete).T
    if values.empty:
        raise ValueError("No complete calendar years available.")

    fig, ax = _heatmap(
        values,
        title="Calendar-year returns",
        color_label="Return",
        figsize=(14, max(5, len(values.index) * 0.32)),
        x_rotation=45,
    )
    fig.tight_layout()
    figure.show(fig)
    return fig, ax


def correlation_heatmap(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> tuple[Figure, Axes]:
    data = returns if columns is None else returns.loc[:, columns]
    values = pd.DataFrame(
        {factor: data.corrwith(factors[factor], axis=0) for factor in factors.columns}
    )
    if values.empty:
        raise ValueError("No correlations available.")

    fig, ax = _heatmap(
        values,
        title="Correlation with return factors",
        color_label="Correlation",
        cmap="RdBu_r",
        figsize=(14, max(5, len(values.index) * 0.32)),
        x_rotation=45,
        x_ha="right",
    )
    fig.tight_layout()
    figure.show(fig)
    return fig, ax


def factor_exposure_heatmap(
    exposures: pd.DataFrame,
    *,
    sort_by: str | None = "dominant",
    title: str = "Factor exposures",
) -> tuple[Figure, Axes]:
    utils.expect_columns(
        exposures, (col.FUND, col.FACTOR, col.IMPACT_PER_SD, col.P_VALUE)
    )
    if exposures.empty:
        raise ValueError("No factor exposures available for the heatmap.")

    factors = [str(name) for name in dict.fromkeys(exposures[col.FACTOR])]
    impacts = exposures.pivot(
        index=col.FUND, columns=col.FACTOR, values=col.IMPACT_PER_SD
    )
    pvalues = exposures.pivot(index=col.FUND, columns=col.FACTOR, values=col.P_VALUE)
    impacts = impacts.reindex(columns=factors)
    pvalues = pvalues.reindex(columns=factors)

    if sort_by == "dominant":
        strength = impacts.abs().to_numpy(dtype=float).max(axis=1)
    elif sort_by is not None:
        if sort_by not in impacts.columns:
            raise ValueError(f"Unknown factor to sort by: {sort_by!r}")
        strength = impacts[sort_by].abs().to_numpy(dtype=float)
    else:
        strength = np.zeros(len(impacts.index))

    order = impacts.index[np.argsort(-strength, kind="stable")]
    impacts = impacts.loc[order]
    pvalues = pvalues.loc[order]

    values = impacts.to_numpy(dtype=float)
    limit = max(float(np.nanquantile(np.abs(values), 0.95)), 0.1)
    fig, ax = figure.matrix(
        values,
        row_labels=[str(name) for name in impacts.index],
        column_labels=factors,
        title=title,
        color_label="pp per 1 SD factor move (fixed common-window SD)",
        vmin=-limit,
        vmax=limit,
        figsize=(15, max(7, len(impacts) * 0.45)),
        x_rotation=45,
        x_ha="right",
        cmap="RdBu_r",
    )
    ax.tick_params(axis="y", labelsize=8)

    for row in range(len(impacts.index)):
        for column in range(len(factors)):
            value = utils.as_float(impacts.iat[row, column], label="impact")
            if not np.isfinite(value):
                continue
            stars = _stars(utils.as_float(pvalues.iat[row, column], label="p-value"))
            ax.text(
                column,
                row,
                f"{value:+.1f}{stars}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(value) > 0.65 * limit else "black",
            )

    fig.tight_layout()
    figure.show(fig)
    return fig, ax


def _anchored_growth(returns: pd.DataFrame, anchor: pd.Timestamp) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    for name in returns.columns:
        values = returns[name].dropna()
        if values.empty or anchor not in values.index:
            series[name] = pd.Series(index=[anchor], dtype=float, name=name)
            continue
        growth = (1 + values).cumprod()
        series[name] = (100 / growth.loc[anchor] * growth).rename(name)
    return pd.DataFrame(series).sort_index()


def anchored_growth(
    returns: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    anchor: str | pd.Timestamp,
    title: str = "Growth of 100 at anchor (log₂ scale)",
) -> tuple[Figure, Axes]:
    data = returns.sort_index()
    if columns is not None:
        data = data.loc[:, columns]

    anchor_date = utils.expect(
        pd.Timestamp(anchor) + pd.offsets.MonthEnd(0), pd.Timestamp
    )
    values = _anchored_growth(data, anchor_date)
    fig, ax = figure.lines(values, title=title, ylabel="Index (100 = anchor)", logy=True)

    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size:
        low = int(np.floor(np.log2(finite.min() / 100)))
        high = int(np.ceil(np.log2(finite.max() / 100)))
        powers = list(range(low, high + 1))
        ax.set_yticks([100 * 2**power for power in powers])
        ax.set_yticklabels([str(power) for power in powers])

    figure.show(fig)
    return fig, ax


def drawdowns(
    returns: pd.DataFrame,
    columns: Sequence[str] | None = None,
    months: int | None = None,
) -> tuple[Figure, Axes]:
    data = returns.sort_index()
    if columns is not None:
        data = data.loc[:, columns]
    if months is not None:
        data = data.tail(months)

    growth = (1 + data).cumprod()
    series = growth.div(growth.cummax()).sub(1)
    fig, ax = figure.lines(
        series,
        title="Drawdowns",
        ylabel="Drawdown",
        percent=True,
        figsize=(13, 5),
    )
    figure.show(fig)
    return fig, ax


def annual_deviations(
    deviations: pd.DataFrame,
    tol: float,
) -> tuple[Figure, Axes]:
    values = deviations.T
    fig, ax = _heatmap(
        values,
        title="Monthly-compounded minus reported annual return",
        color_label="pp",
        figsize=(13, 8),
        x_rotation=90,
        column_labels=[str(year) for year in deviations.index],
        vmin=-tol,
        vmax=tol,
    )
    raw = deviations.to_numpy(dtype=float)
    for y in range(len(deviations.columns)):
        for x in range(len(deviations.index)):
            value = raw[x, y]
            if abs(value) > tol:
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    figure.show(fig)
    return fig, ax
