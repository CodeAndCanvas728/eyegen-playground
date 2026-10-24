"""Converting a diffusers snapshot into the bundle CoreML runs.

A diffusers model is not a CoreML model yet, and converting it is the only
way to run it. The conversion happens once: its result is kept under the
CoreML root and found by the next run. Only a conversion that cannot run
stops the caller, and then with the command that runs it by hand.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from eyegen.artifacts import is_coreml_bundle, is_diffusers_snapshot

from .install import get_coreml_model_dir

log = logging.getLogger(__name__)


def converted_if_needed(model_path: Path, compute_unit: str) -> Path:
    """The directory to run, converting a diffusers model first."""
    if not is_diffusers_snapshot(model_path):
        return model_path
    output = _converted_dir(model_path)
    if is_coreml_bundle(output) and _converted_from(output) == str(model_path):
        log.info("Using the CoreML conversion of %s at %s", model_path, output)
        return output
    log.info("Converting %s to CoreML at %s (this takes 15-20 minutes)", model_path, output)
    from .models import convert_to_coreml

    converted = convert_to_coreml(
        str(model_path),
        output,
        compute_unit=compute_unit,
        progress_callback=lambda message: log.info("[coreml-convert] %s", message),
    )
    if not converted or not is_coreml_bundle(output):
        raise RuntimeError(
            f"Could not convert {model_path} to CoreML. "
            f"Run ./generate.py convert-coreml {model_path} to see the error."
        )
    return output


def _converted_dir(snapshot: Path) -> Path:
    """Where the conversion of *snapshot* is kept, or the first free name.

    A converted bundle is named after the model under the CoreML root. A
    snapshot that already lives there cannot be its own output. A directory
    that holds a bundle from somewhere else is left alone, and the next
    suffix is used instead.
    """
    suffix = 1
    while True:
        name = snapshot.name if suffix == 1 else f"{snapshot.name}-{suffix}"
        output = get_coreml_model_dir(name)
        if output == snapshot or snapshot in output.parents:
            suffix += 1
            continue
        if not output.exists():
            return output
        if not output.is_dir():
            suffix += 1
            continue
        if is_coreml_bundle(output):
            if _converted_from(output) == str(snapshot):
                return output
            suffix += 1
            continue
        if (output / "model_index.json").is_file():
            # A directory that claims to be a model but holds no bundle:
            # not this model's, and not safe to write over.
            suffix += 1
            continue
        # An attempt that stopped early: it holds no model, so the name is
        # still this model's rather than shifted to a new one.
        return output


def _converted_from(directory: Path) -> Optional[str]:
    """The model a converted bundle was made from, when it says so."""
    try:
        meta = json.loads((directory / "model_index.json").read_text())
    except (OSError, ValueError):
        return None
    value = meta.get("model_version") if isinstance(meta, dict) else None
    return value if isinstance(value, str) else None
