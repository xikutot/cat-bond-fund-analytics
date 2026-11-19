import io
from pathlib import Path
from urllib.parse import quote

from matplotlib.figure import Figure
from PIL import Image

from analysis.paths import FIGURES_DIR


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


def save_figure(
    fig: Figure,
    filename: str,
    *,
    alt_text: str | None = None,
    dpi: int = 160,
) -> None:
    """Save a figure and print its Markdown image reference.

    The file is only rewritten when the rendered pixels change, so PNGs
    re-encoded by a different Pillow/zlib build do not create diff noise.
    """
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("filename must be a filename, not a path.")

    output_path = FIGURES_DIR / filename
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    data = buffer.getvalue()
    if not _same_pixels(output_path, data):
        output_path.write_bytes(data)

    description = (
        alt_text
        or Path(filename).stem.replace("_", " ").replace("-", " ")
    )
    description = (
        description
        .replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )

    markdown_path = quote(filename, safe="._-")
    print(f"![{description}](figures/{markdown_path})")
