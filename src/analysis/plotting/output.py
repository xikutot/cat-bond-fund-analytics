import io
from pathlib import Path
from urllib.parse import quote

import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.figure import Figure
from PIL import Image

from analysis import paths


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
