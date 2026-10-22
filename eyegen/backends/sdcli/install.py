"""stable-diffusion.cpp CLI installation and model discovery."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .constants import (
    DEFAULT_SDCLI_BIN,
    DEFAULT_SDCLI_MODEL_DIR,
    DEFAULT_SDCLI_TIMEOUT,
    SDCLI_MODELS,
)

log = logging.getLogger(__name__)


@dataclass
class SdCliInstallStatus:
    """Whether the backend can run, and what is missing if it cannot."""

    installed: bool
    binary: Optional[Path]
    model_dir: Path
    missing_files: list[str]
    message: str


def get_sdcli_bin() -> Optional[Path]:
    """Find the sd-cli binary: the environment wins, then the usual place, then PATH."""
    override = os.environ.get("SDCLI_BIN")
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None

    if DEFAULT_SDCLI_BIN.is_file():
        return DEFAULT_SDCLI_BIN

    found = shutil.which("sd-cli")
    return Path(found) if found else None


def get_sdcli_model_dir() -> Path:
    """The directory holding the model files."""
    override = os.environ.get("SDCLI_MODEL_DIR")
    if override:
        return Path(override).expanduser()
    return DEFAULT_SDCLI_MODEL_DIR


def get_sdcli_timeout() -> float:
    """How long a render may take, in seconds."""
    override = os.environ.get("SDCLI_TIMEOUT")
    if override:
        try:
            value = float(override)
        except ValueError:
            log.warning("SDCLI_TIMEOUT is not a number: %s", override)
        else:
            if value > 0:
                return value
    return float(DEFAULT_SDCLI_TIMEOUT)


def missing_model_files(model: str, model_dir: Path) -> list[str]:
    """The files of *model* that are not in *model_dir*."""
    spec = SDCLI_MODELS.get(model)
    if spec is None:
        return []
    wanted = (spec["diffusion_model"], spec["llm"], spec["vae"])
    return [name for name in wanted if not (model_dir / name).is_file()]


def validate_sdcli_install(
    model: str,
    model_dir: Optional[Path] = None,
) -> SdCliInstallStatus:
    """Check that the binary and the model files are in place."""
    binary = get_sdcli_bin()
    directory = model_dir or get_sdcli_model_dir()
    missing = missing_model_files(model, directory)

    if binary is None:
        return SdCliInstallStatus(
            installed=False,
            binary=None,
            model_dir=directory,
            missing_files=missing,
            message=(
                "sd-cli not found. Build it from https://github.com/leejet/stable-diffusion.cpp "
                "with -DGGML_METAL=ON and put the binary at "
                f"{DEFAULT_SDCLI_BIN} (or set SDCLI_BIN)."
            ),
        )

    if missing:
        return SdCliInstallStatus(
            installed=False,
            binary=binary,
            model_dir=directory,
            missing_files=missing,
            message=(
                f"Missing model files in {directory}: {', '.join(missing)}. "
                f"Fetch them with: ./generate.py pull {model}"
            ),
        )

    return SdCliInstallStatus(
        installed=True,
        binary=binary,
        model_dir=directory,
        missing_files=[],
        message=f"sd-cli ready at {binary}",
    )
