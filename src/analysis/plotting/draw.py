import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.ticker import FuncFormatter, MaxNLocator

TITLE_SIZE: int = 28


def lines(
    series: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    percent: bool = False,
    logy: bool = False,
    figsize: tuple[float, float] = (15, 7),
) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    series.plot(ax=ax)

    ax.set(ylabel=ylabel, xlabel="")
    ax.set_title(title, fontsize=TITLE_SIZE)
    if logy:
        ax.set_yscale("log")
    if percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="small")
    fig.tight_layout()
    fig.subplots_adjust(right=0.68)

    return fig


def heatmap(
    data: pd.DataFrame,
    ax: Axes,
    *,
    title: str,
    color_label: str,
    limit: float | None = None,
    cmap: str | Colormap = "RdYlGn",
    x_tick_count: int | None = None,
) -> AxesImage:
    """Colour a matrix into ``ax`` and return its image."""
    values = data.to_numpy(dtype=float)
    if limit is None:
        limit = max(float(np.nanquantile(np.abs(values), 0.98)), 0.001)

    columns = [str(name) for name in data.columns]
    image = ax.imshow(
        np.ma.masked_invalid(values),
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
    )
    ax.set_xlim(-0.5, len(columns) - 0.5)
    ax.set_yticks(range(len(data.index)), labels=[str(name) for name in data.index])
    ax.yaxis.tick_right()

    if x_tick_count is None:
        positions = list(range(len(columns)))
    else:
        locator = MaxNLocator(nbins=x_tick_count, integer=True, prune="both")
        positions = [int(v) for v in locator.tick_values(-0.5, len(columns) - 0.5)]
    labels = [columns[position] for position in positions]
    ax.set_xticks(positions, labels=labels, rotation=45, ha="right")
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.figure.colorbar(image, ax=ax, label=color_label, location="left")
    plt.tight_layout()
    return image


def annotate(ax: Axes, labels: pd.DataFrame) -> None:
    """Write cell text on a matrix axes over a translucent light box."""
    for row in range(labels.shape[0]):
        for column in range(labels.shape[1]):
            label = labels.iat[row, column]
            if pd.isna(label) or label == "":
                continue
            ax.text(
                column,
                row,
                str(label),
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                bbox={
                    "facecolor": "white", "edgecolor": "none", "alpha": 0.35, "pad": 0.2
                },
            )
