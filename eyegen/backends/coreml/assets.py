"""The files a CoreML bundle needs besides its models.

A converted bundle runs without the PyTorch snapshot it came from, but it
still needs the CLIP tokenizer, the scheduler config, and the feature
extractor when there is a safety checker. They are a couple of megabytes and
are written next to the models, which are never touched: the new files are
staged and checked first, so a bundle that already runs keeps running when
fetching them fails.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from . import model_index
from .offline_runner import BundleError, check_assets

log = logging.getLogger(__name__)

_ASSET_DIRS = ("tokenizer", "tokenizer_2", "scheduler", "feature_extractor")
# The unet config is fetched to read the size the models run at, then dropped.
_ASSET_FILES = ("model_index.json", "unet/config.json")
_ASSET_PATTERNS = [f"{name}/*" for name in _ASSET_DIRS] + list(_ASSET_FILES)


def add_pipeline_assets(model_id: str, output_dir: Path) -> bool:
    """Add the files a run needs next to models that are already converted.

    Only the tokenizer, scheduler and feature extractor are written, and only
    once the new ones are complete. False means the models are usable but the
    files could not be fetched, which the next run retries without converting
    anything. Raises ValueError when the bundle records that it was converted
    from another model: that model's files would not describe these models.
    """
    recorded = model_index.read(output_dir).get("model_version")
    if isinstance(recorded, str) and recorded not in (model_id, str(Path(model_id).expanduser())):
        raise ValueError(
            f"{output_dir} was converted from {recorded}, not {model_id}, so its tokenizer "
            "and scheduler would not match the models. Convert into another directory with "
            "--output."
        )
    try:
        meta = _save_pipeline_assets(model_id, output_dir)
    except (BundleError, OSError, ValueError) as exc:
        log.error(
            "The CoreML models in %s are converted, but the files a run needs could not be "
            "saved: %s",
            output_dir,
            exc,
        )
        return False
    model_index.record(output_dir, meta)
    return True


def record_resolution(output: Path, source: Path) -> None:
    """Record the size the source's models run at, when the bundle does not say.

    A bundle converted before the size was recorded is given it from the
    snapshot it came from, so the wrapper can refuse a size it cannot make
    without waiting for the sidecar to load the models.
    """
    if model_index.resolution(output) is not None:
        return
    resolution = _unet_resolution(source)
    if resolution is None:
        return
    try:
        model_index.record(output, {"resolution": resolution})
    except OSError as exc:
        # The record only spares the sidecar a load: without it the runner
        # still refuses a size the loaded models cannot make.
        log.warning("Could not record the size of the models in %s: %s", output, exc)


def _save_pipeline_assets(model_id: str, output_dir: Path) -> dict:
    """Stage the files beside the bundle, check them, then put them in place."""
    staging = Path(tempfile.mkdtemp(dir=output_dir, prefix=".pipeline-assets-"))
    try:
        source = Path(model_id).expanduser()
        index = (
            _copy_pipeline_assets(source, staging)
            if source.is_dir()
            else _download_pipeline_assets(model_id, staging)
        )
        check_assets(staging)
        meta = {"force_zeros_for_empty_prompt": _force_zeros_for_empty_prompt(index)}
        resolution = _unet_resolution(staging)
        if resolution:
            meta["resolution"] = resolution
        shutil.rmtree(staging / "unet", ignore_errors=True)
        _replace_dir(staging, output_dir / "pipeline_assets")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return meta


def _replace_dir(staging: Path, assets: Path) -> None:
    """Put *staging* where *assets* belongs, keeping the old one until it is."""
    old = assets.with_name(f".{assets.name}.old")
    shutil.rmtree(old, ignore_errors=True)  # left by a swap that was interrupted
    if assets.exists():
        assets.rename(old)
    staging.rename(assets)
    shutil.rmtree(old, ignore_errors=True)


def _unet_resolution(assets: Path) -> Optional[dict]:
    """The size the source's unet runs at, when its config says.

    A diffusers unet config gives the latent size in ``sample_size`` and the
    VAE shrinks by eight, so a latent of 64 is a 512-pixel image. The Core ML
    models are exported at that size and cannot be asked for another, so
    recording it lets a run be refused before anything is loaded.
    """
    try:
        config = json.loads((assets / "unet" / "config.json").read_text())
    except (OSError, ValueError):
        return None
    size = config.get("sample_size") if isinstance(config, dict) else None
    values = [size, size] if _is_latent(size) else size
    if not isinstance(values, list) or len(values) != 2:
        return None
    if not all(_is_latent(value) for value in values):
        return None
    return {"height": values[0] * 8, "width": values[1] * 8}


def _is_latent(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _copy_pipeline_assets(source: Path, assets: Path) -> Path:
    for name in _ASSET_DIRS:
        directory = source / name
        if directory.is_dir():
            shutil.copytree(directory, assets / name)
    for name in _ASSET_FILES:
        file = source / name
        if file.is_file():
            target = assets / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
    return assets / "model_index.json"


def _download_pipeline_assets(model_id: str, assets: Path) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model_id,
        allow_patterns=_ASSET_PATTERNS,
        local_dir=str(assets),
    )
    shutil.rmtree(assets / ".cache", ignore_errors=True)
    return assets / "model_index.json"


def _force_zeros_for_empty_prompt(source_index: Path) -> bool:
    if not source_index.is_file():
        return False
    try:
        config = json.loads(source_index.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(config, dict):
        return False
    return bool(config.get("force_zeros_for_empty_prompt", False))
