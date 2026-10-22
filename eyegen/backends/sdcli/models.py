"""Downloading the files that make up a stable-diffusion.cpp model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from .constants import SDCLI_MODELS
from .install import get_sdcli_model_dir

log = logging.getLogger(__name__)


def list_sdcli_models() -> list[dict]:
    """The models this backend knows, and whether their files are here."""
    model_dir = get_sdcli_model_dir()
    out = []
    for name, spec in SDCLI_MODELS.items():
        wanted = (spec["diffusion_model"], spec["llm"], spec["vae"])
        present = [f for f in wanted if (model_dir / f).is_file()]
        out.append(
            {
                "name": name,
                "model_dir": str(model_dir),
                "files": list(wanted),
                "present": present,
                "installed": len(present) == len(wanted),
                "description": spec.get("description", ""),
            }
        )
    return out


def pull_sdcli_model(
    model: str,
    progress_callback: Optional[Callable[[str], None]] = None,
    model_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Download the files of *model* and return the directory they are in.

    The files come from three repositories: the GGUF transformer, the text
    encoder that conditions it, and the VAE that decodes it. They are
    downloaded one at a time, because together they are several gigabytes
    and a failure halfway through should not repeat the whole download.
    """
    spec = SDCLI_MODELS.get(model)
    if spec is None:
        log.error("Unknown stable-diffusion.cpp model: %s", model)
        if progress_callback:
            progress_callback(f"❌ Unknown model {model}")
        return None

    from huggingface_hub import hf_hub_download

    target = model_dir or get_sdcli_model_dir()
    target.mkdir(parents=True, exist_ok=True)

    def _cb(msg: str) -> None:
        log.info("[sdcli-pull] %s", msg)
        if progress_callback:
            progress_callback(msg)

    for filename, (repo_id, repo_filename) in spec["sources"].items():
        destination = target / filename
        if destination.is_file():
            _cb(f"✅ {filename} is already here")
            continue
        _cb(f"📥 {filename} from {repo_id} ...")
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=repo_filename,
                local_dir=str(target),
            )
        except Exception as exc:  # noqa: BLE001 - the caller reports it
            log.error("Download failed for %s: %s", filename, exc)
            _cb(f"❌ Could not download {filename}: {exc}")
            return None
        # huggingface_hub keeps its own name; the model expects this one.
        downloaded = Path(path)
        if downloaded.name != filename:
            downloaded.replace(destination)
        _cb(f"✅ {filename}")

    return target
