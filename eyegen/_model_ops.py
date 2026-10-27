"""Model download, listing, caching, and saving operations."""

import logging
from pathlib import Path

from eyegen._mflux import _get_mflux_aliases, _resolve_mflux_class
from eyegen.config import Backend
from eyegen.model_catalogue import find_spec
from eyegen.model_specs import ModelSpec
from eyegen.validation import validate_safe_path

log = logging.getLogger(__name__)


def _apply_hf_cache_safe(hf_cache_dir: str | None) -> None:
    """Validate and apply the HuggingFace cache directory."""
    if hf_cache_dir:
        validate_safe_path(hf_cache_dir, "hf_cache_dir")
    from eyegen.backends import _apply_hf_cache

    _apply_hf_cache({"hf_cache_dir": hf_cache_dir})


def pull_model(
    model_name: str,
    progress_callback=None,
    hf_cache_dir: str | None = None,
) -> bool:
    """Download a GGUF model via ollamadiffuser."""
    _apply_hf_cache_safe(hf_cache_dir)
    apply_ollama_registry()
    from ollamadiffuser.core.models.manager import model_manager  # type: ignore[import-untyped]

    return model_manager.pull_model(model_name, progress_callback=progress_callback)


def pull_any_model(
    model_name: str,
    progress_callback=None,
    hf_cache_dir: str | None = None,
) -> bool:
    """Download a model, whichever backend it belongs to.

    A model the user described in their own catalogue is fetched the way
    they described it. A GGUF model is one file that OllamaDiffuser knows.
    A stable-diffusion.cpp model is three files from three repositories,
    and the one that downloads them is the only one that can.
    """
    from eyegen.backends.sdcli.constants import SDCLI_MODELS
    from eyegen.backends.sdcli.models import pull_sdcli_model

    spec = find_spec(model_name)
    if spec is not None:
        return pull_spec(spec, progress_callback=progress_callback, hf_cache_dir=hf_cache_dir)

    if model_name in SDCLI_MODELS:
        return pull_sdcli_model(model_name, progress_callback=progress_callback) is not None

    return pull_model(
        model_name,
        progress_callback=progress_callback,
        hf_cache_dir=hf_cache_dir,
    )


def apply_ollama_registry() -> None:
    """Point OllamaDiffuser at the model entries we keep for it."""
    from eyegen.ollama_registry import apply_registry

    apply_registry()


def pull_spec(
    spec: ModelSpec,
    progress_callback=None,
    hf_cache_dir: str | None = None,
) -> bool:
    """Fetch a model the way its specification says it is fetched."""
    if spec.backend == Backend.SDCLI:
        from eyegen.backends.sdcli.models import pull_sdcli_model

        return (
            pull_sdcli_model(spec.name, progress_callback=progress_callback, spec=spec) is not None
        )

    if spec.backend == Backend.OLLAMA:
        from eyegen.ollama_registry import register

        register(spec)
        return bool(
            pull_model(
                spec.name,
                progress_callback=progress_callback,
                hf_cache_dir=hf_cache_dir,
            )
        )

    if spec.backend == Backend.COREML:
        raise ValueError(
            f"CoreML models are converted, not fetched. Run "
            f"./generate.py convert-coreml {spec.repo or '<hf-model>'} to make '{spec.name}'."
        )

    raise ValueError(
        f"The {spec.backend.value} backend runs a fixed set of models and cannot "
        f"fetch '{spec.name}' from a repository. Name one of its own models "
        "instead."
    )


def list_ollama_models() -> dict:
    """Return dicts of available and installed ollamadiffuser models."""
    apply_ollama_registry()
    from ollamadiffuser.core.models.manager import model_manager  # type: ignore[import-untyped]

    return {
        "installed": model_manager.list_installed_models(),
        "available": model_manager.list_available_models(),
    }


def list_mflux_models() -> list[dict]:
    """Return a list of available MFLUX model dicts with alias and HF name."""
    try:
        from mflux.models.common.config.model_config import (
            AVAILABLE_MODELS,  # type: ignore[import-untyped]
        )

        return [
            {"alias": alias, "model_name": cfg.model_name}
            for alias, cfg in sorted(AVAILABLE_MODELS.items(), key=lambda x: x[1].priority)
        ]
    except (ImportError, AttributeError) as exc:
        log.warning(
            "Could not load MFLUX models from package, falling back to static list: %s", exc
        )
        aliases = sorted(_get_mflux_aliases())
        return [{"alias": a, "model_name": a} for a in aliases]


def clear_mflux_cache(model_alias: str | None = None, *, force: bool = False) -> list[str]:
    """Delete cached MFLUX / HuggingFace model files and return removed paths."""
    if model_alias is None and not force:
        raise ValueError("Must provide model_alias or set force=True to clear all flux models")

    from huggingface_hub import scan_cache_dir  # type: ignore[import-untyped]

    removed: list[str] = []
    repo_id: str | None = None

    if model_alias:
        try:
            from mflux.models.common.config.model_config import (
                ModelConfig,  # type: ignore[import-untyped]
            )

            mc = ModelConfig.from_name(model_name=model_alias)
            repo_id = mc.model_name
        except (ImportError, AttributeError) as exc:
            log.debug("Could not resolve model alias, using literal: %s", exc)
            repo_id = model_alias

    cache_info = scan_cache_dir()
    revision_hashes = []
    for repo in cache_info.repos:
        if repo_id and repo.repo_id != repo_id:
            continue
        if not repo_id and "flux" not in repo.repo_id.lower():
            continue
        for revision in repo.revisions:
            revision_hashes.append((repo.repo_id, revision.commit_hash))

    if revision_hashes:
        strategy = cache_info.delete_revisions(*[h for _, h in revision_hashes])
        strategy.execute()
        for repo_id_entry, commit_hash in revision_hashes:
            log.info("Removed cached revision %s for %s", commit_hash, repo_id_entry)
            removed.append(f"{repo_id_entry} ({commit_hash[:8]})")

    return removed


def save_mflux_model(
    model_alias: str,
    quantize: int | None,
    output_path: str | Path,
    progress_callback=None,
    hf_cache_dir: str | None = None,
) -> Path:
    """Download an MFLUX model, quantize it, and save to *output_path*."""
    _apply_hf_cache_safe(hf_cache_dir)
    from mflux.models.common.config.model_config import ModelConfig  # type: ignore[import-untyped]

    if quantize is not None and not (1 <= quantize <= 16):
        raise ValueError(f"Invalid quantize value: {quantize}. Must be between 1 and 16.")

    out = Path(validate_safe_path(str(output_path), "output_path"))
    out.mkdir(parents=True, exist_ok=True)

    model_config = ModelConfig.from_name(model_name=model_alias)
    cls = _resolve_mflux_class(model_config)

    q_label = f"{quantize}-bit" if quantize else "full precision"
    if progress_callback:
        progress_callback(f"Downloading & quantizing '{model_alias}' ({q_label})…")
    log.info("Saving MFLUX model '%s' (%s) to %s", model_alias, q_label, out)

    model = cls(model_config=model_config, quantize=quantize)

    if progress_callback:
        progress_callback(f"Saving to {out}…")
    model.save_model(str(out))

    if not any(f.is_file() for f in out.glob("**/*")):
        raise RuntimeError(f"MFLUX model save failed: no files were written to {out}")

    log.info("Model saved: %s", out)
    if progress_callback:
        progress_callback(f"✅ Model saved to {out}")
    return out
