from pathlib import Path
from urllib.parse import quote

from matplotlib.figure import Figure

from analysis.paths import FIGURES_DIR


def save_figure(
    fig: Figure,
    filename: str,
    *,
    alt_text: str | None = None,
    dpi: int = 160,
) -> None:
    """Save a figure and print its Markdown image reference."""
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("filename must be a filename, not a path.")

    output_path = FIGURES_DIR / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

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
