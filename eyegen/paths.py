"""Where the tool keeps what it downloads.

One root, and everything under it by convention. The root is
configuration: the ``models_dir`` setting, which is read from the
configuration when there is one and from the settings file when there is
not, with ``EYEGEN_MODELS_DIR`` as the override; and the default when
neither names one.

The convention is that a directory holds one thing, and the thing it holds
is named after what it is rather than after a model:

    models_dir/
    ├── sdcli/<model>/     # one directory per model: transformer, encoder, VAE
    ├── coreml/<model>/    # converted bundles
    ├── saved-mflux/       # models saved for a faster load
    ├── bonsai-demo/       # the Bonsai checkout
    └── .coreml-venv/      # the CoreML sidecar

A model of one file and a model of three files cannot share a directory:
their files have the same names — ``ae.safetensors`` is a VAE whatever
model it belongs to — and one would overwrite the other.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from eyegen.validation import validate_model_name

DEFAULT_MODELS_DIR = Path.home() / "models" / "eyegen"

MODELS_DIR_ENV = "EYEGEN_MODELS_DIR"


def models_dir(config=None) -> Path:
    """The root under which the tool keeps what it downloads.

    The order is the setting on the configuration handed in, then the
    environment variable, then the setting in the settings file, then the
    default. The file is read rather than assumed because not every place
    that asks for a directory is given the configuration — a listing, a
    status line — and a setting that works in one of them and not the
    others would be worse than no setting at all.
    """
    configured = _configured(config)
    if configured:
        return configured
    override = os.environ.get(MODELS_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return _configured_from_file() or DEFAULT_MODELS_DIR


def _configured(config) -> Optional[Path]:
    if config is None:
        return None
    value = getattr(config, "models_dir", None)
    return Path(value).expanduser() if value else None


def _configured_from_file() -> Optional[Path]:
    """The root the settings file names, when it names one."""
    from eyegen.config import CONFIG_FILE

    try:
        data = json.loads(CONFIG_FILE.read_text())
    except (OSError, ValueError):
        # No file, or a file that is not there to be asked.
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("models_dir")
    return Path(value).expanduser() if isinstance(value, str) and value else None


def sdcli_model_dir(model: str, config=None) -> Path:
    """The directory holding one model's files for the stable-diffusion.cpp backend.

    The model names the directory, so the name is checked before it becomes
    a path: a name with a path in it would put the files outside the models
    root.
    """
    return models_dir(config) / "sdcli" / validate_model_name(model)


def coreml_models_dir(config=None) -> Path:
    """The directory holding the CoreML bundles."""
    return models_dir(config) / "coreml"


def coreml_venv_dir(config=None) -> Path:
    """The CoreML sidecar's own Python environment."""
    return models_dir(config) / ".coreml-venv"


def bonsai_dir(config=None) -> Path:
    """The Bonsai checkout."""
    return models_dir(config) / "bonsai-demo"


def saved_mflux_dir(config=None) -> Path:
    """The directory holding the models saved for a faster load."""
    return models_dir(config) / "saved-mflux"


#: The directories under the root that the convention owns. A model found
#: in one of them belongs to that backend's own listing; a model anywhere
#: else is found by the walk.
CONVENTIONAL_SUBDIRS = ("sdcli", "coreml", "saved-mflux")


def is_conventional_subdir(directory: Path) -> bool:
    """Whether *directory* is one of the convention's own under the root."""
    return directory.name in CONVENTIONAL_SUBDIRS


#: Where HuggingFace keeps its hub caches, and where OllamaDiffuser keeps
#: its own models — other programs' directories, scanned as their own roots.
HF_HUB_DIRS_ENV = "EYEGEN_HF_HUB_DIRS"
OLLAMA_MODELS_DIR_ENV = "EYEGEN_OLLAMA_MODELS_DIR"


def hf_hub_dirs(config=None) -> list[Path]:
    """The HuggingFace hub caches to scan, most recent first.

    The configured cache, the tool's default, and the package's own default
    are all places a model may have been downloaded to. The environment
    override names exactly one place, for a single run or a test.
    """
    override = os.environ.get(HF_HUB_DIRS_ENV)
    if override:
        return [Path(p).expanduser() for p in override.split(os.pathsep)]

    candidates: list[Path] = []
    if config is not None and getattr(config, "hf_cache_dir", None):
        candidates.append(Path(config.hf_cache_dir).expanduser())
    from eyegen.config import HF_CACHE_DIR  # lazy: config imports this module

    candidates.append(HF_CACHE_DIR)
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    seen: set[Path] = set()
    result: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(candidate)
    return result


def ollama_models_dir() -> Path:
    """The directory OllamaDiffuser keeps its own models in."""
    override = os.environ.get(OLLAMA_MODELS_DIR_ENV)
    if override:
        return Path(override).expanduser()
    try:
        from ollamadiffuser.core.config.settings import settings  # type: ignore[import-untyped]

        return Path(settings.models_dir)
    except (ImportError, AttributeError):
        return Path.home() / ".ollamadiffuser" / "models"
