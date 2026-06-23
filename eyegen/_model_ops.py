"""Model download, listing, caching, and saving operations."""

import logging
from pathlib import Path

from eyegen._mflux import _get_mflux_aliases, _resolve_mflux_class
from eyegen.backends import _apply_hf_cache

log = logging.getLogger(__name__)


def pull_model(model_name: str, progress_callback=None, hf_cache_dir: str | None = None) -> bool:
    """Download a GGUF model via ollamadiffuser."""
    _apply_hf_cache({"hf_cache_dir": hf_cache_dir})
    from ollamadiffuser.core.models.manager import model_manager

    return model_manager.pull_model(model_name, progress_callback=progress_callback)


def list_ollama_models() -> dict:
    """Return dicts of available and installed ollamadiffuser models."""
    from ollamadiffuser.core.models.manager import model_manager

    return {
        "installed": model_manager.list_installed_models(),
        "available": model_manager.list_available_models(),
    }


def list_mflux_models() -> list[dict]:
    """Return a list of available MFLUX model dicts with alias and HF name."""
    try:
        from mflux.models.common.config.model_config import AVAILABLE_MODELS

        return [
            {"alias": alias, "model_name": cfg.model_name}
            for alias, cfg in sorted(AVAILABLE_MODELS.items(), key=lambda x: x[1].priority)
        ]
    except (ImportError, AttributeError) as exc:
        log.debug("Could not load MFLUX models from package, falling back to static list: %s", exc)
        aliases = sorted(_get_mflux_aliases())
        return [{"alias": a, "model_name": a} for a in aliases]


def clear_mflux_cache(model_alias: str | None = None) -> list[str]:
    """Delete cached MFLUX / HuggingFace model files and return removed paths."""
    from huggingface_hub import scan_cache_dir

    removed: list[str] = []
    repo_id: str | None = None

    if model_alias:
        try:
            from mflux.models.common.config.model_config import ModelConfig

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
    _apply_hf_cache({"hf_cache_dir": hf_cache_dir})
    from mflux.models.common.config.model_config import ModelConfig

    output_path = Path(output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    model_config = ModelConfig.from_name(model_name=model_alias)
    cls = _resolve_mflux_class(model_config)

    q_label = f"{quantize}-bit" if quantize else "full precision"
    if progress_callback:
        progress_callback(f"Downloading & quantizing '{model_alias}' ({q_label})…")
    log.info("Saving MFLUX model '%s' (%s) to %s", model_alias, q_label, output_path)

    model = cls(model_config=model_config, quantize=quantize)

    if progress_callback:
        progress_callback(f"Saving to {output_path}…")
    model.save_model(str(output_path))

<<<<<<< Updated upstream
    safetensors_files = list(output_path.rglob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"Model saving failed: no safetensors files found in {output_path}")
=======
    # Verify at least one file was written
    if not any(f.is_file() for f in output_path.glob("**/*")):
        raise RuntimeError(f"MFLUX model save failed: no files were written to {output_path}")
>>>>>>> Stashed changes

    log.info("Model saved: %s", output_path)
    if progress_callback:
        progress_callback(f"✅ Model saved to {output_path}")
    return output_path


def validate_saved_model(path: str | Path) -> tuple[bool, dict | None]:
    """Check whether *path* contains a valid mflux-saved model."""
    p = Path(path).expanduser()
    if not p.is_dir():
        return False, None

    meta: dict = {"quantization_level": None, "mflux_version": None}
    found_safetensors = False

    try:
        from safetensors import SafetensorError, safe_open
    except ImportError:
        log.debug("safetensors not installed, cannot validate model")
        return False, None

    for subdir in sorted(p.iterdir()):
        if not subdir.is_dir():
            continue
        for sf in sorted(subdir.glob("*.safetensors")):
            found_safetensors = True
            try:
                with safe_open(str(sf), framework="mlx") as f:
                    m = f.metadata() or {}
                    ql = m.get("quantization_level")
                    meta["quantization_level"] = int(ql) if ql and ql != "None" else None
                    meta["mflux_version"] = m.get("mflux_version")
                return True, meta
            except (OSError, ValueError, SafetensorError) as exc:
                log.debug("Could not read safetensors metadata from %s: %s", sf, exc)
                continue

    if not found_safetensors:
        return False, None

    return True, meta
