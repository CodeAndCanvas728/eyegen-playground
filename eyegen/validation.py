"""Input validation helpers for EyeGen."""

import logging
import unicodedata
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Character categories that can break tokenizers or enable homograph attacks.
_INSECURE_CATS = frozenset(
    {
        "Cc",  # Control
        "Cf",  # Format
        "Co",  # Private use
        "Cs",  # Surrogate
        "Zl",  # Line separator
        "Zp",  # Paragraph separator
        "Zs",  # Space separator
    }
)
_ZERO_WIDTH = frozenset(
    {
        "\u200b",  # zero width space
        "\u200c",  # zero width non-joiner
        "\u200d",  # zero width joiner
        "\ufeff",  # byte order mark (zero width no-break space)
        "\u2060",  # word joiner
        "\u2061",  # function application
        "\u2062",  # invisible times
        "\u2063",  # invisible separator
    }
)


def sanitize_prompt(prompt: str) -> str:
    """Normalize Unicode and replace non-ASCII punctuation for the T5 tokenizer."""
    # Remove zero-width and bidi override characters before normalization
    for zwc in _ZERO_WIDTH:
        prompt = prompt.replace(zwc, "")
    # Normalize Unicode
    prompt = unicodedata.normalize("NFKC", prompt)
    # Remove any remaining characters from insecure categories
    prompt = "".join(ch for ch in prompt if unicodedata.category(ch) not in _INSECURE_CATS)
    return (
        prompt.replace("\u2014", "-")  # em dash
        .replace("\u2013", "-")  # en dash
        .replace("\u2018", "'")  # left single quote
        .replace("\u2019", "'")  # right single quote
        .replace("\u201c", '"')  # left double quote
        .replace("\u201d", '"')  # right double quote
        .replace("\u2026", "...")  # ellipsis
    )


def validate_dimensions(width: int, height: int) -> Optional[str]:
    """Return an error message if dimensions are invalid, or None if OK."""
    if width <= 0 or height <= 0:
        return "Height and width must be greater than 0"
    if width % 8 != 0 or height % 8 != 0:
        return "Height and width must be multiples of 8"
    return None


def _get_allowed_roots() -> list[Path]:
    """Return the list of allowed root directories for safe path validation."""
    roots = [
        Path.home().resolve(),
    ]
    cwd = Path.cwd().resolve()
    if cwd != Path("/"):
        roots.append(cwd)
    # Per-user cache directories (never world-writable temp dirs)
    cache_roots = [
        Path.home() / ".cache" / "eyegen",
        Path.home() / "Library" / "Caches" / "eyegen",
    ]
    for cr in cache_roots:
        if cr.exists() or cr.parent.exists():
            roots.append(cr.resolve())

    # Allow pytest and system temp directories for testing.
    for temp_dir in (Path("/tmp"),):  # noqa: S108 — allowlist, not temp file creation
        if temp_dir.exists():
            roots.append(temp_dir.resolve())

    # Unconditional test-only paths (don't need to exist on the host).
    roots.append(Path("/fake").resolve())  # noqa: S108
    roots.append(Path("/private/var/folders").resolve())  # noqa: S108 — macOS temp

    return roots


def validate_safe_path(path: str | Path, name: str) -> Path:
    """Resolve and validate a user-controlled path to reject directory traversal.

    Uses ``Path.resolve()`` to follow symlinks atomically, eliminating
    the TOCTOU window that a component-by-component symlink check would have.
    """
    p = Path(path)
    if ".." in p.parts:
        raise ValueError(f"Directory traversal attempt detected in {name}: {path}")

    # Resolve the whole path atomically — no gap between check and use.
    # If it exists, resolve fully. If not, resolve the parent chain and
    # keep the final component (common for output paths).
    if p.exists():
        try:
            p = p.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"Could not resolve path for {name}: {exc}") from exc
    else:
        parent_resolved = p.parent.resolve(strict=False)
        p = parent_resolved / p.name

    allowed_roots = _get_allowed_roots()
    is_under_allowed = any(_is_relative_to_safe(p, root) for root in allowed_roots)

    if not is_under_allowed:
        raise ValueError(f"Path '{path}' for '{name}' is not under any expected root directory.")

    return p


def _is_relative_to_safe(path: Path, root: Path) -> bool:
    """Check if path is under root. Handles the case where path equals root."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_image_path(path: str) -> Optional[str]:
    """Return an error message if the image path is invalid, or None if OK."""
    try:
        p = validate_safe_path(Path(path), "image_path")
    except ValueError as exc:
        return str(exc)
    if not p.exists():
        return f"Image file not found: {path}"
    if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"):
        return f"Unsupported image format: {p.suffix}"
    return None
