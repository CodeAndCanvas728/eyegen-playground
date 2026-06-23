"""Input validation helpers for EyeGen."""

import logging
import unicodedata
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def sanitize_prompt(prompt: str) -> str:
    """Normalize Unicode and replace non-ASCII punctuation for the T5 tokenizer."""
    prompt = unicodedata.normalize("NFKC", prompt)
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


def validate_safe_path(path: str | Path, name: str) -> Path:
    """Resolve and validate a user-controlled path to reject directory traversal."""
    p_orig = Path(path)
    if ".." in p_orig.parts:
        raise ValueError(f"Directory traversal attempt detected in {name}: {path}")

    p = p_orig.expanduser().resolve()

    allowed_roots = [
        Path.home().resolve(),
        Path.cwd().resolve(),
        Path("/tmp").resolve(),  # noqa: S108
        Path("/var").resolve(),  # noqa: S108
        Path("/private/var").resolve(),  # noqa: S108
        Path("/var/tmp").resolve(),  # noqa: S108
        Path("/private/tmp").resolve(),  # noqa: S108
        Path("/Volumes").resolve(),  # noqa: S108
    ]

    is_under_allowed = False
    for root in allowed_roots:
        try:
            p.relative_to(root)
            is_under_allowed = True
            break
        except ValueError:
            continue

    if not is_under_allowed:
        raise ValueError(f"Path '{path}' for '{name}' is not under any expected root directory.")

    return p


def validate_image_path(path: str) -> Optional[str]:
    """Return an error message if the image path is invalid, or None if OK."""
    try:
        p = validate_safe_path(path, "image_path")
    except ValueError as exc:
        return str(exc)
    if not p.exists():
        return f"Image file not found: {path}"
    if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"):
        return f"Unsupported image format: {p.suffix}"
    return None
