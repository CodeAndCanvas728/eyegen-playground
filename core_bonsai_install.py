"""Bonsai installation validation and status."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from core_bonsai_constants import DEFAULT_BONSAI_DIR

log = logging.getLogger(__name__)


@dataclass
class BonsaiInstallStatus:
    installed: bool
    bonsai_dir: Path
    has_venv: bool
    has_generate_sh: bool
    has_models: list[str]
    message: str


def get_bonsai_dir() -> Path:
    override = os.environ.get("BONSAI_DIR")
    return Path(override).expanduser() if override else DEFAULT_BONSAI_DIR


def validate_bonsai_install() -> BonsaiInstallStatus:
    bonsai_dir = get_bonsai_dir()
    has_venv = (bonsai_dir / ".venv" / "bin" / "python").is_file()
    has_generate_sh = (bonsai_dir / "scripts" / "generate.sh").is_file()
    has_download_sh = (bonsai_dir / "scripts" / "download_model.sh").is_file()

    models = []
    models_dir = bonsai_dir / "models"
    if models_dir.is_dir():
        for child in sorted(models_dir.iterdir()):
            if child.is_dir() and any(child.glob("*.safetensors")):
                models.append(child.name)

    installed = has_generate_sh and has_download_sh
    if not installed:
        msg = f"Bonsai is not installed at {bonsai_dir}. Run: ./scripts/setup-bonsai.sh"
    elif not has_venv:
        msg = f"Bonsai venv not found at {bonsai_dir}/.venv/. Run: ./scripts/setup-bonsai.sh"
    elif not models:
        msg = (
            "Bonsai is installed but no models are downloaded. "
            "Run: ./generate.py pull-bonsai ternary-mlx"
        )
    else:
        msg = f"Bonsai ready at {bonsai_dir} (models: {', '.join(models)})"

    return BonsaiInstallStatus(
        installed=installed and has_venv,
        bonsai_dir=bonsai_dir,
        has_venv=has_venv,
        has_generate_sh=has_generate_sh,
        has_models=models,
        message=msg,
    )
