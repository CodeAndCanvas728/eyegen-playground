"""Idol sidecar installation validation and status."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from eyegen.paths import idol_dir


@dataclass
class IdolInstallStatus:
    installed: bool
    idol_dir: Path
    venv_python: Path
    sdk_dir: Path
    message: str


def get_idol_dir(config=None) -> Path:
    """The Idol sidecar root: the SDK checkout and its venv live under it."""
    override = os.environ.get("IDOL_DIR")
    if override:
        return Path(override).expanduser()
    return idol_dir(config)


def get_idol_venv_python(config=None) -> Path:
    """The sidecar venv's interpreter — the binary every run is launched with."""
    return get_idol_dir(config) / ".venv" / "bin" / "python"


def get_idol_sdk_dir(config=None) -> Path:
    """The idol-sdk checkout."""
    return get_idol_dir(config) / "idol-sdk"


def validate_idol_install(config=None) -> IdolInstallStatus:
    """Whether the sidecar is ready to run.

    Installed means the venv interpreter exists and the SDK checkout is there;
    the weights themselves download into the HuggingFace cache on first run,
    so they are not part of the check.
    """
    root = get_idol_dir(config)
    venv_python = get_idol_venv_python(config)
    sdk_dir = get_idol_sdk_dir(config)
    has_venv = venv_python.is_file()
    has_sdk = (sdk_dir / "idol" / "sample.py").is_file()

    if has_venv and has_sdk:
        msg = f"Idol ready at {root} (venv: {venv_python})"
    elif not has_sdk:
        msg = f"Idol SDK not found at {sdk_dir}. Run: ./generate.py setup-idol"
    else:
        msg = f"Idol venv not found at {venv_python}. Run: ./generate.py setup-idol"

    return IdolInstallStatus(
        installed=has_venv and has_sdk,
        idol_dir=root,
        venv_python=venv_python,
        sdk_dir=sdk_dir,
        message=msg,
    )
