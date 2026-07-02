"""CoreML installation validation and status."""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .constants import DEFAULT_COREML_MODELS_DIR, DEFAULT_COREML_VENV

log = logging.getLogger(__name__)


@dataclass
class CoreMLInstallStatus:
    installed: bool
    venv_python: Optional[Path]
    models_dir: Path
    model_count: int
    message: str


def get_coreml_models_dir() -> Path:
    override = os.environ.get("COREML_MODELS_DIR")
    return Path(override).expanduser() if override else DEFAULT_COREML_MODELS_DIR


def get_coreml_venv() -> Path:
    override = os.environ.get("COREML_VENV")
    return Path(override).expanduser() if override else DEFAULT_COREML_VENV


def _sidecar_python() -> Optional[Path]:
    candidates = [
        get_coreml_venv() / "bin" / "python",
        get_coreml_venv() / "bin" / "python3",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _sidecar_has_coreml() -> bool:
    py = _sidecar_python()
    if py is None:
        return False
    try:
        r = subprocess.run(  # noqa: S603
            [str(py), "-c", "import python_coreml_stable_diffusion"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return False


def _validate_coreml_install_uncached() -> CoreMLInstallStatus:
    py = _sidecar_python()
    has_coreml = _sidecar_has_coreml() if py else False
    models_dir = get_coreml_models_dir()
    model_count = 0
    if models_dir.is_dir():
        model_count = sum(1 for p in models_dir.iterdir() if p.is_dir())

    if py is None:
        msg = (
            f"CoreML sidecar venv not found at {get_coreml_venv()}. Run: ./scripts/setup-coreml.sh"
        )
        installed = False
    elif not has_coreml:
        msg = (
            f"CoreML sidecar venv exists at {py} but "
            "python_coreml_stable_diffusion is not installed. "
            "Re-run: ./scripts/setup-coreml.sh"
        )
        installed = False
    else:
        msg = f"CoreML ready (venv={py}, models={model_count} in {models_dir})"
        installed = True

    return CoreMLInstallStatus(
        installed=installed,
        venv_python=py,
        models_dir=models_dir,
        model_count=model_count,
        message=msg,
    )


@lru_cache(maxsize=1)
def _validate_coreml_install_cached() -> CoreMLInstallStatus:
    return _validate_coreml_install_uncached()


def validate_coreml_install(force: bool = False) -> CoreMLInstallStatus:
    """Validate CoreML installation status, using a 1-entry cache unless force is True."""
    if force:
        _validate_coreml_install_cached.cache_clear()
    return _validate_coreml_install_cached()


validate_coreml_install.cache_clear = _validate_coreml_install_cached.cache_clear
