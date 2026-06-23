"""Input validation helpers for EyeGen."""

import unicodedata
from pathlib import Path
from typing import Optional


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
    if width % 8 != 0 or height % 8 != 0:
        return "Height and width must be multiples of 8"
    return None


def validate_image_path(path: str) -> Optional[str]:
    """Return an error message if the image path is invalid, or None if OK."""
    p = Path(path)
    if not p.exists():
        return f"Image file not found: {path}"
    if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"):
        return f"Unsupported image format: {p.suffix}"
    return None
