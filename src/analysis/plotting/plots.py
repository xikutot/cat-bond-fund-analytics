from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from analysis.plotting.figure import lines, matrix, show
from analysis.utils import expect


def _common_window(
    returns: pd.DataFrame,
    columns: Sequence[str] | None = None,
    months: int = 36,
) -> pd.DataFrame:
    data = returns.sort_index()

    if columns is not None:
        data = data.loc[:, columns]

    data = data.tail(months)
    data = data.dropna(axis=1)

    if data.empty:
        raise ValueError("No series have complete data for this period.")

    return data


def _cumulative_series(data: pd.DataFrame) -> pd.DataFrame:
    growth = (1 + data).cumprod()
    baseline = pd.DataFrame(
        1.0,
        index=[data.index[0] - pd.offsets.MonthEnd(1)],
        columns=data.columns,
    )
    return pd.concat([baseline, growth])


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

    values = data.to_numpy(dtype=float).T
    limit = max(float(np.nanquantile(np.abs(values), 0.98)), 0.001)
    ticks = [
        int(i)
        for i in np.arange(0, len(data.index), max(1, len(data.index) // 10))
    ]

    fig, ax = matrix(
        values,
        row_labels=list(data.columns),
        column_labels=[data.index[i].strftime("%Y-%m") for i in ticks],
        column_positions=ticks,
        title=title,
        color_label="Monthly return",
        vmin=-limit,
        vmax=limit,
        figsize=(14, max(5, len(data.columns) * 0.32)),
        x_rotation=45,
        x_ha="right",
    )
    fig.tight_layout()
    show(fig)

    return fig, ax


def growth(
    returns: pd.DataFrame,
    columns: Sequence[str] | None = None,
    months: int = 36,
    title: str = "Cumulative growth",
) -> tuple[Figure, Axes]:
    data = _common_window(returns, columns, months)
    series = _cumulative_series(data).mul(100)

    fig, ax = lines(series, title=title, ylabel="Growth of 100")
    show(fig)

    return fig, ax


def drawdowns(
    returns: pd.DataFrame,
    columns: Sequence[str] | None = None,
    months: int = 36,
) -> tuple[Figure, Axes]:
    data = _common_window(returns, columns, months)
    series = _cumulative_series(data)
    series = series.div(series.cummax()).sub(1)

    fig, ax = lines(
        series,
        title="Drawdowns",
        ylabel="Drawdown",
        percent=True,
        figsize=(13, 5),
    )
    show(fig)

    return fig, ax


def rolling_returns(
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

    rolling = expect(rolling, pd.DataFrame)
    fig, ax = lines(
        rolling,
        title=f"Trailing {window}-month compounded returns",
        ylabel="Return",
        percent=True,
    )
    show(fig)

    return fig, ax


def annual_deviations(
    deviations: pd.DataFrame,
    tol: float,
) -> tuple[Figure, Axes]:
    values = deviations.to_numpy(dtype=float)
    fig, ax = matrix(
        values.T,
        row_labels=[name[:32] for name in deviations.columns],
        column_labels=[str(year) for year in deviations.index],
        title="Monthly-compounded minus reported annual return",
        color_label="pp",
        vmin=-tol,
        vmax=tol,
        figsize=(13, 8),
        x_rotation=90,
    )

    for y in range(len(deviations.columns)):
        for x in range(len(deviations.index)):
            value = values[x, y]
            if abs(value) > tol:
                ax.text(
                    x,
                    y,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    fig.tight_layout()
    show(fig)

    return fig, ax
