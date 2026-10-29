import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


def _common_window(returns, columns=None, months=36):
    data = returns.sort_index()

    if columns is not None:
        data = data.loc[:, columns]

    data = data.tail(months)

    excluded = data.columns[data.isna().any()].tolist()
    if excluded:
        print(f"Excluded (incomplete window): {', '.join(excluded)}")

    data = data.dropna(axis=1)

    if data.empty:
        raise ValueError("No series have complete data for this period.")

    return data


def plot_monthly_heatmap(
    returns, months=36, title="Monthly returns"
):
    data = returns.sort_index().tail(months).dropna(axis=1, how="all")

    if data.empty:
        raise ValueError("No return data available for the heatmap.")

    values = data.to_numpy(dtype=float).T
    limit = max(float(np.nanquantile(np.abs(values), 0.98)), 0.001)

    fig, ax = plt.subplots(
        figsize=(14, max(5, len(data.columns) * 0.32))
    )

    image = ax.imshow(
        np.ma.masked_invalid(values),
        aspect="auto",
        interpolation="none",
        cmap="RdYlGn",
        vmin=-limit,
        vmax=limit,
    )

    ax.set_yticks(range(len(data.columns)), labels=data.columns)

    ticks = np.arange(
        0, len(data.index), max(1, len(data.index) // 10)
    )
    ax.set_xticks(
        ticks,
        labels=[data.index[i].strftime("%Y-%m") for i in ticks],
        rotation=45,
        ha="right",
    )

    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Monthly return")
    fig.tight_layout()

    return fig, ax


def plot_growth(
    returns, columns=None, months=36, title="Cumulative growth"
):
    data = _common_window(returns, columns, months)
    growth = (1 + data).cumprod()

    baseline = pd.DataFrame(
        1.0,
        index=[data.index[0] - pd.offsets.MonthEnd(1)],
        columns=data.columns,
    )
    growth = pd.concat([baseline, growth]).mul(100)

    fig, ax = plt.subplots(figsize=(13, 6))
    growth.plot(ax=ax)

    ax.set(title=title, ylabel="Growth of 100", xlabel="")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()

    return fig, ax


def plot_drawdowns(returns, columns=None, months=36):
    data = _common_window(returns, columns, months)
    wealth = (1 + data).cumprod()

    baseline = pd.DataFrame(
        1.0,
        index=[data.index[0] - pd.offsets.MonthEnd(1)],
        columns=data.columns,
    )
    wealth = pd.concat([baseline, wealth])
    drawdowns = wealth.div(wealth.cummax()).sub(1)

    fig, ax = plt.subplots(figsize=(13, 5))
    drawdowns.plot(ax=ax)

    ax.set(title="Drawdowns", ylabel="Drawdown", xlabel="")
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.0%}")
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()

    return fig, ax


def plot_rolling_returns(
    returns, columns=None, window=12, display_months=36
):
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

    fig, ax = plt.subplots(figsize=(13, 6))
    rolling.plot(ax=ax)

    ax.set(
        title=f"Trailing {window}-month compounded returns",
        ylabel="Return",
        xlabel="",
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.0%}")
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()

    return fig, ax
