import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import matplotlib.pyplot as plt
from IPython.display import DisplayHandle, display
from matplotlib.figure import Figure
from PIL import Image

from analysis import paths

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="figure-save")


def show(fig: Figure) -> Figure:
    """Return a displayed figure after closing it, keeping notebook output ordered."""
    display(fig)
    plt.close(fig)
    return fig


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


def _figure_path(name: str) -> Path:
    """Resolve a bare filename inside the notebook's figures directory."""
    candidate = Path(name)
    if str(candidate) != candidate.name:
        raise ValueError("Pass a figure filename, not a path.")
    return paths.FIGURES_DIR / candidate


def markdown_link(path: Path, *, alt_text: str | None = None) -> str:
    """Return the Markdown reference for a figure path."""
    description = alt_text or path.stem.replace("_", " ").replace("-", " ")
    description = (
        description.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
    )
    link = path.relative_to(paths.NOTEBOOKS).as_posix()
    return f"![{description}]({quote(link, safe='._-/')})"


def _write_if_changed(path: Path, png: bytes) -> None:
    """Atomically replace ``path`` only when its pixels differ from ``png``."""
    if _same_pixels(path, png):
        return

    temporary = path.with_name(f"{path.name}.tmp")
    try:
        temporary.write_bytes(png)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _status_line(link: str, error: BaseException | None) -> str:
    if error is not None:
        return f"❌ {link} — {error}"
    return f"✅ {link}"


def save_async(
    fig: Figure,
    name: str,
    *,
    alt_text: str | None = None,
    dpi: int = 160,
) -> None:
    """Show a Markdown reference now and write the rendered PNG in the background."""
    path = _figure_path(name)
    link = markdown_link(path, alt_text=alt_text)
    status = DisplayHandle()
    status.display(f"⏳ {link}")

    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    except Exception as error:  # noqa: BLE001 - render errors must complete the visible status.
        status.update(_status_line(link, error))
        return

    future = _EXECUTOR.submit(_write_if_changed, path, buffer.getvalue())
    future.add_done_callback(
        lambda done: status.update(_status_line(link, done.exception()))
    )
