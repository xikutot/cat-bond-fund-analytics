import io
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from PIL import Image

from analysis import paths

TITLE_SIZE: int = 28


def lines(
    series: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    percent: bool = False,
    logy: bool = False,
    figsize: tuple[float, float] = (13, 6),
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    series.plot(ax=ax)

    ax.set(ylabel=ylabel, xlabel="")
    ax.set_title(title, fontsize=TITLE_SIZE)
    if logy:
        ax.set_yscale("log")
    if percent:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:.0%}")
        )
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()

    return fig, ax


def matrix(
    values: np.ndarray,
    *,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str,
    color_label: str,
    vmin: float,
    vmax: float,
    figsize: tuple[float, float],
    column_positions: Sequence[int] | None = None,
    x_rotation: int = 0,
    x_ha: str = "center",
    cmap: str = "RdYlGn",
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(
        np.ma.masked_invalid(values),
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    ax.set_xticks(
        range(len(column_labels)) if column_positions is None else column_positions,
        labels=column_labels,
        rotation=x_rotation,
        ha=x_ha,
    )
    ax.set_title(title, fontsize=TITLE_SIZE)
    fig.colorbar(image, ax=ax, label=color_label)

    return fig, ax


def show(fig: Figure) -> None:
    """Display a figure now and close it, so notebook output stays ordered."""
    display(fig)
    plt.close(fig)


def _same_pixels(path: Path, png: bytes) -> bool:
    """Return True if the existing PNG has the same pixels as ``png``."""
    try:
        with Image.open(path) as existing, Image.open(io.BytesIO(png)) as rendered:
            if existing.size != rendered.size:
                return False
            existing_rgba = existing.convert("RGBA")
            rendered_rgba = rendered.convert("RGBA")
            return existing_rgba.tobytes() == rendered_rgba.tobytes()
    except OSError:
        return False


def save(
    fig: Figure,
    path: Path,
    *,
    alt_text: str | None = None,
    dpi: int = 160,
) -> None:
    """Save a figure, rewriting only when its pixels change.

    Prints the Markdown reference for the notebook.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    png = buffer.getvalue()
    if not _same_pixels(path, png):
        path.write_bytes(png)

    description = alt_text or path.stem.replace("_", " ").replace("-", " ")
    description = (
        description.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
    )
    link = path.relative_to(paths.NOTEBOOKS).as_posix()
    print(f"![{description}]({quote(link, safe='._-/')})")
