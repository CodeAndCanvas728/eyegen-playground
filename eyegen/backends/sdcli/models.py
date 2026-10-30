"""Downloading the files that make up a stable-diffusion.cpp model."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

from eyegen.model_catalogue import load_catalogue
from eyegen.model_sources import ModelSource
from eyegen.model_specs import ModelSpec

from .constants import SDCLI_MODELS
from .install import get_sdcli_model_dir, source_is_here
from .pick import VAE_HINTS, pick_by_hints, resolve_llm

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


def list_sdcli_models(config=None) -> list[dict]:
    """The models this backend knows, and whether their files are here."""
    out = []
    for name, spec in sdcli_specs().items():
        model_dir = get_sdcli_model_dir(name, config)
        wanted = [source.name for source in spec.sources.values()]
        present = [
            source.name for source in spec.sources.values() if source_is_here(source, model_dir)
        ]
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
    if spec.is_local:
        # Every file is already on this machine: there is nothing to fetch,
        # and no directory of its own to make for them.
        log.info("The model %s has no files to download", model)
        if progress_callback:
            progress_callback("✅ Every file of the model is already here")
        return target
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
        if source.is_local:
            # The file is already on this machine; it is used where it is.
            _cb(f"✅ {source.name} is already here ({source.path})")
            continue
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
        # huggingface_hub keeps the repository's own folders; the binary
        # expects every file by its own name in the model's directory.
        downloaded = Path(path)
        if downloaded != destination:
            downloaded = downloaded.replace(destination)
        _cb(f"✅ {source.name}")

    return target


def _repo_files(repo: str) -> Optional[list[str]]:
    """The files a HuggingFace repository holds, or None when it cannot be read.

    None says "unknown", not "empty": an unreachable repository must not
    look like a repository that holds no companions. That distinction is
    the whole difference between buying the companions and letting sd-cli
    fail with a file it was never given.
    """
    from huggingface_hub import list_repo_files

    try:
        return list(list_repo_files(repo))
    except Exception:  # noqa: BLE001 - an unreachable repository is an unknown model
        return None


def complete_local_spec(spec: ModelSpec) -> ModelSpec:
    """The companions a lone GGUF folder needs, from the repository it came from.

    A directory holding only the transformer of a stable-diffusion.cpp
    model is not runnable by itself: sd-cli needs the text encoder and the
    VAE as well. When the GGUF sits in a HuggingFace cache layout, the
    repository it came from is known (``models--<org>--<repo>``), and the
    companions are the other weight files of that repository: a VAE and a
    text encoder, picked by what they are called. When the only encoders
    on offer are formats this machine cannot run (an int8 convrot encoder
    has no Metal kernel), the encoder comes from a documented companion
    repository instead. When that repository
    cannot even be listed, the model may be half of one, and the run is
    refused with that said — quietly running a model without its encoder
    hands the failure to sd-cli, where it is a mystery instead of a plan.
    """
    diffusion = spec.sources.get("diffusion_model")
    if diffusion is None or not diffusion.is_local:
        return spec
    if "vae" in spec.sources or "llm" in spec.sources:
        return spec

    from eyegen.artifacts import huggingface_repo_of

    repo = huggingface_repo_of(Path(diffusion.path))
    if not repo:
        return spec

    files = _repo_files(repo)
    if files is None:
        raise RuntimeError(
            f"The model at {diffusion.path} is only the transformer; sd-cli also needs "
            f"its companions, which could not be listed from {repo}.\n"
            "Their download starts from that listing, so it cannot begin while the "
            "repository is unreachable — check the connection and run again."
        )
    vae = pick_by_hints(files, VAE_HINTS)
    llm, llm_repo = resolve_llm(files, repo)
    if not vae and not llm:
        return spec

    new_sources = dict(spec.sources)
    if vae:
        new_sources["vae"] = ModelSource(repo=repo, file=vae)
    if llm:
        new_sources["llm"] = ModelSource(repo=llm_repo, file=llm)
    log.info("Completed local model %s from %s: vae=%s llm=%s", spec.name, repo, vae, llm)
    # rebuild the specification as itself, changed: a hand-rolled copy
    # lists the fields it knows and drops the ones it does not (`alias`).
    return replace(spec, sources=new_sources, repo=repo)
