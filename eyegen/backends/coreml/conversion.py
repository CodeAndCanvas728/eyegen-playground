"""Converting a diffusers snapshot into the bundle CoreML runs.

A diffusers model is not a CoreML model yet, and converting it is the only
way to run it. The conversion happens once: its result is kept under the
CoreML root and found by the next run. Only a conversion that cannot run
stops the caller, and then with the command that runs it by hand.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from eyegen.artifacts import is_coreml_bundle, is_diffusers_snapshot

from . import model_index
from .assets import add_pipeline_assets, record_resolution
from .install import get_coreml_model_dir
from .models import convert_to_coreml
from .offline_runner import BundleError, has_models, pipeline_assets_dir

log = logging.getLogger(__name__)


def converted_if_needed(model_path: Path, compute_unit: str) -> Path:
    """The directory to run, converting a diffusers model first."""
    if not is_diffusers_snapshot(model_path):
        return model_path
    output = _converted_dir(model_path)
    if _converted_from(output) == str(model_path) and has_models(output):
        if _has_pipeline_assets(output):
            record_resolution(output, model_path)
            log.info("Using the CoreML conversion of %s at %s", model_path, output)
            return output
        log.info(
            "Adding the tokenizer and scheduler to the CoreML conversion of %s at %s",
            model_path,
            output,
        )
        if add_pipeline_assets(str(model_path), output):
            return output
        raise RuntimeError(_assets_message(model_path, output))
    log.info("Converting %s to CoreML at %s (this takes 15-20 minutes)", model_path, output)
    converted = convert_to_coreml(
        str(model_path),
        output,
        compute_unit=compute_unit,
        **_settings(output),
        progress_callback=lambda message: log.info("[coreml-convert] %s", message),
    )
    if not converted:
        if has_models(output):
            raise RuntimeError(_assets_message(model_path, output))
        raise RuntimeError(
            f"Could not convert {model_path} to CoreML. "
            f"Run ./generate.py convert-coreml {model_path} to see the error."
        )
    return output


def _assets_message(model_path: Path, output: Path) -> str:
    """What to say when the models are converted but the runner cannot use them."""
    return (
        f"The CoreML models in {output} are converted, but the tokenizer and scheduler "
        f"the runner needs could not be saved. Run ./generate.py convert-coreml "
        f"{model_path} to retry; the models are not converted again."
    )


def _has_pipeline_assets(directory: Path) -> bool:
    """Whether a bundle has the files it needs to run without its source."""
    try:
        pipeline_assets_dir(directory)
    except BundleError:
        return False
    return True


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
    value = model_index.read(directory).get("model_version")
    return value if isinstance(value, str) else None


def _settings(directory: Path) -> dict:
    """The conversion settings a bundle was made with, to finish it the same way."""
    meta = model_index.read(directory)
    keys = ("attention_implementation", "quantize_nbits")
    return {key: meta[key] for key in keys if meta.get(key) is not None}
