"""Downloading the files that make up a stable-diffusion.cpp model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from eyegen.model_catalogue import load_catalogue
from eyegen.model_specs import ModelSpec

from .constants import SDCLI_MODELS
from .install import get_sdcli_model_dir

log = logging.getLogger(__name__)


def _built_in_specs() -> dict[str, ModelSpec]:
    """The models this backend ships with, as specifications."""
    return {name: ModelSpec.from_dict(name, data) for name, data in SDCLI_MODELS.items()}


def sdcli_specs() -> dict[str, ModelSpec]:
    """Every model this backend can run: the shipped ones and the user's."""
    specs = _built_in_specs()
    for name, spec in load_catalogue().items():
        if spec.backend.value == "sdcli":
            specs[name] = spec
    return specs


def list_sdcli_models() -> list[dict]:
    """The models this backend knows, and whether their files are here."""
    out = []
    for name, spec in sdcli_specs().items():
        model_dir = get_sdcli_model_dir(name)
        wanted = [source.name for source in spec.sources.values()]
        present = [f for f in wanted if (model_dir / f).is_file()]
        out.append(
            {
                "name": name,
                "model_dir": str(model_dir),
                "files": list(wanted),
                "present": present,
                "installed": bool(wanted) and len(present) == len(wanted),
                "description": spec.description or "",
            }
        )
    return out


def pull_sdcli_model(
    model: str,
    progress_callback: Optional[Callable[[str], None]] = None,
    model_dir: Optional[Path] = None,
    spec: Optional[ModelSpec] = None,
) -> Optional[Path]:
    """Download the files of *model* and return the directory they are in.

    The files come from three repositories: the GGUF transformer, the text
    encoder that conditions it, and the VAE that decodes it. They are
    downloaded one at a time, because together they are several gigabytes
    and a failure halfway through should not repeat the whole download.
    """
    if spec is None:
        spec = sdcli_specs().get(model)
    if spec is None:
        log.error("Unknown stable-diffusion.cpp model: %s", model)
        if progress_callback:
            progress_callback(f"❌ Unknown model {model}")
        return None
    if not spec.sources:
        log.error("The model %s names no files to download", model)
        if progress_callback:
            progress_callback(f"❌ The model {model} names no files to download")
        return None

    target = model_dir or get_sdcli_model_dir(model)
    target.mkdir(parents=True, exist_ok=True)
    return _download_sources(spec, target, progress_callback)


def _download_sources(
    spec: ModelSpec,
    target: Path,
    progress_callback: Optional[Callable[[str], None]],
) -> Optional[Path]:
    """Fetch every file of the model, one at a time.

    They are downloaded one at a time, because together they are several
    gigabytes and a failure halfway through should not repeat the whole
    download.
    """
    from huggingface_hub import hf_hub_download

    def _cb(msg: str) -> None:
        log.info("[sdcli-pull] %s", msg)
        if progress_callback:
            progress_callback(msg)

    for source in spec.sources.values():
        destination = target / source.name
        if destination.is_file():
            _cb(f"✅ {source.name} is already here")
            continue
        _cb(f"📥 {source.name} from {source.repo} ...")
        try:
            path = hf_hub_download(
                repo_id=source.repo,
                filename=source.file,
                local_dir=str(target),
            )
        except Exception as exc:  # noqa: BLE001 - the caller reports it
            log.error("Download failed for %s: %s", source.name, exc)
            _cb(f"❌ Could not download {source.name}: {exc}")
            return None
        # huggingface_hub keeps its own name; the model expects this one.
        downloaded = Path(path)
        if downloaded.name != source.name:
            downloaded.replace(destination)
        _cb(f"✅ {source.name}")

    return target
