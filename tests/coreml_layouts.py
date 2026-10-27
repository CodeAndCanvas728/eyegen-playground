"""Directories laid out the way the CoreML conversion's inputs are.

The tests that build a diffusers snapshot, or the bundle a conversion leaves,
differ in what they assert rather than in what they have to create, so the
builders are here.
"""

import json
from pathlib import Path

# What a converted bundle holds: the models the run needs, and the files the
# tokenizer and scheduler are read from.
MODEL_NAMES = ("unet", "text_encoder", "vae_decoder")


def snapshot(directory: Path, *, sample_size: int = 64, merges: bool = True) -> Path:
    """The small files a diffusers model holds next to its weights."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}")
    (directory / "unet").mkdir()
    (directory / "unet" / "config.json").write_text(json.dumps({"sample_size": sample_size}))
    tokenizer = directory / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "vocab.json").write_text("{}")
    if merges:
        (tokenizer / "merges.txt").write_text("")
    scheduler = directory / "scheduler"
    scheduler.mkdir()
    (scheduler / "scheduler_config.json").write_text("{}")
    (directory / "model_index.json").write_text(json.dumps({"force_zeros_for_empty_prompt": True}))
    return directory


def assets(bundle: Path) -> Path:
    """The ``pipeline_assets`` a finished bundle holds."""
    directory = bundle / "pipeline_assets"
    tokenizer = directory / "tokenizer"
    tokenizer.mkdir(parents=True)
    (tokenizer / "vocab.json").write_text("{}")
    (tokenizer / "merges.txt").write_text("")
    scheduler = directory / "scheduler"
    scheduler.mkdir()
    (scheduler / "scheduler_config.json").write_text("{}")
    return directory


def converted(output: Path, model_id: str) -> Path:
    """A bundle as a finished conversion leaves it."""
    output.mkdir(parents=True, exist_ok=True)
    for name in MODEL_NAMES:
        (output / f"{name}.mlpackage").mkdir(exist_ok=True)
    (output / "model_index.json").write_text(json.dumps({"model_version": model_id}))
    assets(output)
    return output
