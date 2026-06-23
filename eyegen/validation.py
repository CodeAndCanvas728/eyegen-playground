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


def is_path_safe(path_str: str, expected_roots: Optional[list[Path]] = None) -> bool:
    """Verify that a path is safe (no traversal and under allowed roots)."""
    if not path_str:
        return True
    path = Path(path_str)
    if ".." in path.parts:
        return False
    try:
        resolved = path.resolve()
    except Exception:
        return False

    import tempfile
    default_roots = [Path.home(), Path.cwd(), Path(tempfile.gettempdir())]
    if expected_roots is None:
        roots = default_roots
    else:
        roots = list(expected_roots) + [Path(tempfile.gettempdir())]

    resolved_roots = []
    for root in roots:
        try:
            resolved_roots.append(root.resolve())
        except (OSError, RuntimeError) as exc:
            log.debug("Could not resolve root %s: %s", root, exc)
            continue

    in_root = False
    for r in resolved_roots:
        if resolved == r or r in resolved.parents:
            in_root = True
            break
    return in_root


def validate_image_path(path: str) -> Optional[str]:
    """Return an error message if the image path is invalid, or None if OK."""
    if not is_path_safe(path, [Path.home(), Path.cwd()]):
        return f"Invalid or unsafe image path: {path}"
    p = Path(path)
    if not p.exists():
        return f"Image file not found: {path}"
    if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"):
        return f"Unsupported image format: {p.suffix}"
    return None
