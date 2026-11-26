from collections.abc import Sequence
from itertools import pairwise
from typing import Literal, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.ticker import FuncFormatter

TITLE_SIZE: int = 28
CELL_SIZE: int = 20
PATCH_SIZES: tuple[int, ...] = (8, 10, 12, 14, 16, 18)
KEY_HEADER_ROWS: int = 2

# Silver borders over RdBu_r keep signed values readable when square size encodes a bucket.
DIVERGING_CMAP: Colormap = colormaps.get_cmap("RdBu_r")
SILVER_BORDER = to_rgb("silver")


class Bucket(NamedTuple):
    label: str
    mask: np.ndarray


def _patch_mask(patch_size: int) -> np.ndarray:
    mask = np.ones((CELL_SIZE, CELL_SIZE))
    offset = (CELL_SIZE - patch_size) // 2
    mask[offset : offset + patch_size, offset : offset + patch_size] = 0
    return mask


class BucketScale:
    """Infer bucket edges, labels, indices, and masks from values."""

    def __init__(self, values: np.ndarray | pd.DataFrame, *, unit: str = "") -> None:
        raw = np.asarray(values, dtype=float)
        finite = np.maximum(raw[np.isfinite(raw)], 1.0)
        if finite.size == 0:
            raise ValueError("No finite values available for buckets.")
        low, high = float(finite.min()), float(finite.max())
        steps = len(PATCH_SIZES) - 1
        interior = np.geomspace(low, high, 2 * steps + 1)[1::2]
        self.edges = np.r_[low, interior, high]
        self.buckets = tuple(
            Bucket(
                f"{np.ceil(left - 1e-9):.0f}–{np.floor(right + 1e-9):.0f}{unit}",
                mask,
            )
            for (left, right), mask in zip(
                pairwise(self.edges), map(_patch_mask, PATCH_SIZES)
            )
        )
        self.indices = self.indices_for(raw)

    def __len__(self) -> int:
        return len(self.buckets)

    def indices_for(self, values: np.ndarray | pd.DataFrame) -> np.ndarray:
        raw = np.atleast_1d(np.asarray(values, dtype=float))
        if self.edges[0] == self.edges[-1]:
            return np.full(raw.shape, len(self) - 1, dtype=int)
        clean = np.where(np.isfinite(raw), np.maximum(raw, 1.0), self.edges[-1])
        return np.searchsorted(self.edges[1:-1], clean, side="right")

    def bucket_for(self, value: float) -> Bucket:
        return self.buckets[int(self.indices_for(np.asarray(value))[0])]


def _bucket_overlay(
    image: AxesImage,
    masked: np.ma.MaskedArray,
    scale: BucketScale,
) -> np.ndarray:
    """Build the silver-bordered overlay for a heatmap."""
    masks = np.stack([bucket.mask for bucket in scale.buckets])[scale.indices]
    block = np.moveaxis(masks, 1, 2).reshape(
        scale.indices.shape[0] * CELL_SIZE, scale.indices.shape[1] * CELL_SIZE
    )
    alpha = image.to_rgba(masked)[..., 3]
    valid = np.repeat(np.repeat(alpha > 0, CELL_SIZE, axis=0), CELL_SIZE, axis=1)
    overlay = np.empty((*block.shape, 4))
    overlay[..., :3] = SILVER_BORDER
    overlay[..., 3] = block * valid
    return overlay


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


def _date_ticks(index: pd.DatetimeIndex) -> tuple[list[int], list[str]]:
    ticks = [int(i) for i in np.arange(0, len(index), max(1, len(index) // 10))]
    formatted = index.strftime("%Y-%m")
    return ticks, [str(formatted[i]) for i in ticks]


def heatmap(
    data: pd.DataFrame,
    *,
    title: str,
    color_label: str,
    limit: float | None = None,
    cmap: str | Colormap = "RdYlGn",
    figsize: tuple[float, float] = (14, 8),
    aspect: Literal["auto", "equal"] = "auto",
    xlim: tuple[float, float] | None = None,
    column_labels: Sequence[str] | None = None,
    date_columns: bool = False,
    x_rotation: int = 0,
    x_ha: str = "center",
    x_labelsize: int | None = None,
    y_labelsize: int | None = None,
    cell_labels: pd.DataFrame | None = None,
    cell_colors: pd.DataFrame | None = None,
) -> Figure:
    values = data.to_numpy(dtype=float)
    if limit is None:
        limit = max(float(np.nanquantile(np.abs(values), 0.98)), 0.001)

    labels = list(column_labels) if column_labels is not None else list(data.columns)
    positions = None
    if date_columns:
        if not isinstance(data.columns, pd.DatetimeIndex):
            raise TypeError("date_columns requires a DatetimeIndex")
        positions, labels = _date_ticks(data.columns)

    masked = np.ma.masked_invalid(values)
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(
        masked,
        aspect=aspect,
        interpolation="none",
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
    )
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_yticks(range(len(data.index)), labels=[str(name) for name in data.index])
    ax.yaxis.tick_right()
    ax.set_xticks(
        range(len(labels)) if positions is None else positions,
        labels=labels,
        rotation=x_rotation,
        ha=x_ha,
    )
    ax.set_title(title, fontsize=TITLE_SIZE)
    fig.colorbar(image, ax=ax, label=color_label, location="left")
    fig.tight_layout()

    if x_labelsize is not None:
        ax.tick_params(axis="x", labelsize=x_labelsize)
    if y_labelsize is not None:
        ax.tick_params(axis="y", labelsize=y_labelsize)

    if cell_labels is not None:
        for row in range(len(data.index)):
            for column in range(len(data.columns)):
                label = cell_labels.iat[row, column]
                if pd.isna(label) or label == "":
                    continue
                color = None if cell_colors is None else str(cell_colors.iat[row, column])
                ax.text(
                    column,
                    row,
                    str(label),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )

    return fig


def _bucket_key(
    fig: Figure,
    matrix: Axes,
    colorbar: Axes,
    scale: BucketScale,
    *,
    header: str,
) -> Axes:
    """Draw the bucket key between the colorbar and matrix."""
    matrix_position = matrix.get_position()
    gap = 0.01
    left = colorbar.get_position().x1 + gap
    right = matrix_position.x0 - gap
    key = fig.add_axes((left, matrix_position.y0, right - left, matrix_position.height))
    key.set_axis_off()
    key_width = (
        (right - left)
        * fig.get_figwidth()
        * len(scale.indices)
        / (matrix_position.height * fig.get_figheight())
    )
    key.set_xlim(0, key_width)
    key.set_ylim(len(scale.indices) - 0.5, -0.5)

    sample_left = key_width - 0.9
    centre = np.asarray(DIVERGING_CMAP(0.5)[:3])
    key.text(
        0,
        KEY_HEADER_ROWS - 1,
        header,
        ha="left",
        va="center",
        fontsize=8,
        color="0.35",
        fontweight="bold",
    )
    for row, bucket in enumerate(scale.buckets, start=KEY_HEADER_ROWS):
        sample = np.empty((*bucket.mask.shape, 4))
        sample[..., :3] = np.where(bucket.mask[..., None] > 0, SILVER_BORDER, centre)
        sample[..., 3] = 1
        key.imshow(
            sample,
            extent=(sample_left, sample_left + 0.8, row + 0.4, row - 0.4),
            aspect="equal",
            interpolation="none",
            zorder=5,
        )
        key.text(
            sample_left - 0.2,
            row,
            bucket.label,
            ha="right",
            va="center",
            fontsize=7,
            color="0.35",
        )
    return key


def bucketed_matrix(
    data: pd.DataFrame,
    *,
    bucket_values: np.ndarray | pd.DataFrame,
    title: str,
    color_label: str,
    unit: str,
    header: str,
    figsize: tuple[float, float] = (14, 8),
    column_labels: Sequence[str] | None = None,
    x_rotation: int = 0,
    x_ha: str = "center",
    x_labelsize: int | None = None,
) -> Figure:
    """Draw a matrix whose cell squares encode bucketed values."""
    scale = BucketScale(bucket_values, unit=unit)
    fig = heatmap(
        data,
        title=title,
        color_label=color_label,
        cmap=DIVERGING_CMAP,
        figsize=figsize,
        aspect="equal",
        xlim=(-0.5, len(data.columns) - 0.5),
        column_labels=column_labels,
        x_rotation=x_rotation,
        x_ha=x_ha,
        x_labelsize=x_labelsize,
    )
    matrix, colorbar = fig.axes
    image = matrix.images[0]
    matrix.imshow(
        _bucket_overlay(image, np.ma.asarray(image.get_array()), scale),
        extent=(-0.5, len(data.columns) - 0.5, len(data.index) - 0.5, -0.5),
        aspect="equal",
        interpolation="none",
    )
    _bucket_key(fig, matrix, colorbar, scale, header=header)
    return fig
