"""The ``model_index.json`` a bundle keeps: what it was made from, and how.

Anything that must know what a directory holds without loading the models
reads it: the conversion deciding whether to reuse a bundle, the wrapper
refusing a size, the listing. Whoever finishes making a bundle merges its own
facts into it, so a write never loses what an earlier attempt recorded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def read(directory: Path) -> dict:
    """What *directory* records about itself, when it records anything."""
    try:
        meta = json.loads((directory / "model_index.json").read_text())
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def record(directory: Path, values: dict) -> None:
    """Add *values* to what *directory* records, keeping what is there."""
    meta = read(directory)
    meta.update(values)
    (directory / "model_index.json").write_text(json.dumps(meta, indent=2) + "\n")


def resolution(directory: Path) -> Optional[tuple[int, int]]:
    """The (width, height) a bundle's models were exported at, when recorded."""
    size = read(directory).get("resolution")
    if not isinstance(size, dict):
        return None
    width, height = size.get("width"), size.get("height")
    if not all(isinstance(value, int) and value > 0 for value in (width, height)):
        return None
    return (width, height)
