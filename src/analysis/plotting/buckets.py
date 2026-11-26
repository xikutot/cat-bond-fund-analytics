from itertools import pairwise
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from analysis.plotting import draw

CELL_SIZE: int = 20
PATCH_SIZES: tuple[int, ...] = (8, 10, 12, 14, 16, 18)
KEY_HEADER_ROWS: int = 2

# Silver borders keep signed values readable when square size encodes a bucket.
SILVER_BORDER = to_rgb("silver")
SAMPLE_FILL = to_rgb("dimgray")


class Bucket(NamedTuple):
    label: str
    mask: np.ndarray


class BucketScale:
    """Infer bucket edges, labels, indices, and masks from values."""

    def __init__(self, values: np.ndarray | pd.DataFrame, *, unit: str = "") -> None:
        raw = np.asarray(values, dtype=float)
        self.edges = self._bucket_edges(raw, len(PATCH_SIZES))
        self.buckets = tuple(
            Bucket(self._bucket_label(left, right, unit), self._patch_mask(size))
            for (left, right), size in zip(pairwise(self.edges), PATCH_SIZES)
        )
        self.indices = self._indices_for(raw)

    def __len__(self) -> int:
        return len(self.buckets)

    @staticmethod
    def _patch_mask(patch_size: int) -> np.ndarray:
        mask = np.ones((CELL_SIZE, CELL_SIZE))
        offset = (CELL_SIZE - patch_size) // 2
        mask[offset : offset + patch_size, offset : offset + patch_size] = 0
        return mask

    @staticmethod
    def _bucket_edges(values: np.ndarray, count: int) -> np.ndarray:
        """Return ``count + 1`` log-spaced edges covering the finite values."""
        finite = np.maximum(values[np.isfinite(values)], 1.0)
        if finite.size == 0:
            raise ValueError("No finite values available for buckets.")
        low, high = float(finite.min()), float(finite.max())
        interior = np.geomspace(low, high, 2 * count - 1)[1::2]
        return np.r_[low, interior, high]

    @staticmethod
    def _bucket_label(left: float, right: float, unit: str) -> str:
        """Format one bucket's inclusive integer range."""
        return f"{np.ceil(left - 1e-9):.0f}–{np.floor(right + 1e-9):.0f}{unit}"

    def _indices_for(self, values: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Map values to buckets, placing unordered non-finite values last."""
        raw = np.atleast_1d(np.asarray(values, dtype=float))
        high = self.edges[-1]
        comparable = np.nan_to_num(raw, nan=high, posinf=high, neginf=high)
        return np.searchsorted(
            self.edges[1:-1], np.clip(comparable, self.edges[0], high), side="right"
        )


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
    matrix_rows = len(scale.indices)
    gap = 0.01
    left = colorbar.get_position().x1 + gap
    right = matrix_position.x0 - gap
    key = fig.add_axes((left, matrix_position.y0, right - left, matrix_position.height))
    key.set_axis_off()
    gap_width = (right - left) * fig.get_figwidth()
    key_height = matrix_position.height * fig.get_figheight()
    # Key units follow matrix rows so samples match cells and labels fit the gap.
    key_width = gap_width * matrix_rows / key_height
    key.set_xlim(0, key_width)
    key.set_ylim(matrix_rows - 0.5, -0.5)

    sample_size = 1
    row_step = 2
    padding = 0.1
    sample_left = key_width - sample_size - padding
    label_x = sample_left - padding
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
    for index, bucket in enumerate(scale.buckets):
        row = KEY_HEADER_ROWS + index * row_step
        sample = np.empty((*bucket.mask.shape, 4))
        sample[..., :3] = np.where(bucket.mask[..., None] > 0, SILVER_BORDER, SAMPLE_FILL)
        sample[..., 3] = 1
        key.imshow(
            sample,
            extent=(
                sample_left,
                sample_left + sample_size,
                row + sample_size / 2,
                row - sample_size / 2,
            ),
            aspect="equal",
            interpolation="none",
            zorder=5,
        )
        key.text(
            label_x,
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
) -> Figure:
    """Draw a matrix whose cell squares encode bucketed values."""
    scale = BucketScale(bucket_values, unit=unit)
    fig, matrix = plt.subplots(figsize=figsize)
    image = draw.heatmap(
        data,
        matrix,
        title=title,
        color_label=color_label,
        cmap="RdBu_r",
    )
    matrix.set_aspect("equal")
    fig.tight_layout()
    matrix.imshow(
        _bucket_overlay(image, np.ma.asarray(image.get_array()), scale),
        extent=(-0.5, len(data.columns) - 0.5, len(data.index) - 0.5, -0.5),
        aspect="equal",
        interpolation="none",
    )
    colorbar = image.colorbar
    assert colorbar is not None
    _bucket_key(fig, matrix, colorbar.ax, scale, header=header)
    return fig
