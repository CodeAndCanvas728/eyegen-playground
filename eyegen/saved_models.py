"""Checking a model that was saved to disk.

A model saved by MFLUX is a directory of safetensors files. Whether that
directory holds a model, and which quantization it is, is read from the
metadata inside those files — the file names do not say, and a directory
with one shard from one model and one from another is not a model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class SavedModelValidation:
    """What a saved model directory turned out to hold."""

    valid: bool
    meta: Optional[dict] = None
    error: Optional[str] = None


def _scan_safetensors_dir(p: Path):
    """Scan *p* for safetensors files and return collected metadata."""
    from safetensors import SafetensorError, safe_open  # type: ignore[import-untyped]

    found_safetensors = False
    read_any = False
    read_failures: list[str] = []
    quant_levels: set = set()
    versions: set = set()

    for subdir in sorted(p.iterdir()):
        if not subdir.is_dir():
            continue
        for sf in sorted(subdir.glob("*.safetensors")):
            found_safetensors = True
            try:
                with safe_open(str(sf), framework="mlx") as f:
                    m = f.metadata() or {}
            except (OSError, ValueError, SafetensorError) as exc:
                log.debug("Could not read safetensors metadata from %s: %s", sf, exc)
                read_failures.append(str(sf))
                continue
            read_any = True
            ql = m.get("quantization_level")
            quant_levels.add(int(ql) if ql and ql != "None" else None)
            versions.add(m.get("mflux_version"))

    return found_safetensors, read_any, read_failures, quant_levels, versions


def validate_saved_model(path: str | Path) -> SavedModelValidation:
    """Check whether *path* contains a valid mflux-saved model."""
    p = Path(str(path)).expanduser()
    if not p.is_dir():
        return SavedModelValidation(valid=False, error=f"Not a directory: {path}")

    try:
        found_safetensors, read_any, read_failures, quant_levels, versions = _scan_safetensors_dir(
            p
        )
    except ImportError:
        log.debug("safetensors not installed, cannot validate model")
        return SavedModelValidation(
            valid=False, error="safetensors not installed, cannot validate model"
        )

    if not found_safetensors:
        return SavedModelValidation(valid=False, error="No .safetensors files found")
    if not read_any:
        return SavedModelValidation(valid=False, error="Could not read safetensors metadata")
    if read_failures:
        return SavedModelValidation(
            valid=False,
            error=f"Could not read safetensors metadata from: {', '.join(read_failures)}",
        )
    if len(quant_levels) > 1:
        return SavedModelValidation(
            valid=False,
            error=f"Inconsistent quantization across shards: {sorted(map(str, quant_levels))}",
        )
    if len(versions) > 1:
        return SavedModelValidation(
            valid=False,
            error=f"Inconsistent mflux_version across shards: {sorted(map(str, versions))}",
        )

    return SavedModelValidation(
        valid=True,
        meta={
            "quantization_level": next(iter(quant_levels)),
            "mflux_version": next(iter(versions)),
        },
    )
