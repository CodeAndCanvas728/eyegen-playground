"""Model download, listing, caching, and saving operations."""

import logging
from dataclasses import dataclass
from pathlib import Path

from eyegen._mflux import _get_mflux_aliases, _resolve_mflux_class
from eyegen.validation import validate_safe_path

log = logging.getLogger(__name__)


@dataclass
class SavedModelValidation:
    valid: bool
    meta: dict | None = None
    error: str | None = None


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
    from ollamadiffuser.core.models.manager import model_manager  # type: ignore[import-untyped]

    return model_manager.pull_model(model_name, progress_callback=progress_callback)


def list_ollama_models() -> dict:
    """Return dicts of available and installed ollamadiffuser models."""
    from ollamadiffuser.core.models.manager import model_manager  # type: ignore[import-untyped]

    return {
        "installed": model_manager.list_installed_models(),
        "available": model_manager.list_available_models(),
    }


def list_mflux_models() -> list[dict]:
    """Return a list of available MFLUX model dicts with alias and HF name."""
    try:
        from mflux.models.common.config.model_config import AVAILABLE_MODELS  # type: ignore[import-untyped]

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


def clear_mflux_cache(
    model_alias: str | None = None, *, force: bool = False
) -> list[str]:
    """Delete cached MFLUX / HuggingFace model files and return removed paths."""
    if model_alias is None and not force:
        raise ValueError(
            "Must provide model_alias or set force=True to clear all flux models"
        )

    from huggingface_hub import scan_cache_dir  # type: ignore[import-untyped]

    removed: list[str] = []
    repo_id: str | None = None

    if model_alias:
        try:
            from mflux.models.common.config.model_config import ModelConfig  # type: ignore[import-untyped]

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


def validate_saved_model(path: str | Path) -> SavedModelValidation:
    """Check whether *path* contains a valid mflux-saved model."""
    p = Path(str(path)).expanduser()
    if not p.is_dir():
        return SavedModelValidation(
            valid=False, error=f"Not a directory: {path}"
        )

    meta: dict = {"quantization_level": None, "mflux_version": None}
    found_safetensors = False

    try:
        from safetensors import SafetensorError, safe_open  # type: ignore[import-untyped]
    except ImportError:
        log.debug("safetensors not installed, cannot validate model")
        return SavedModelValidation(
            valid=False, error="safetensors not installed, cannot validate model"
        )

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
                return SavedModelValidation(valid=True, meta=meta)
            except (OSError, ValueError, SafetensorError) as exc:
                log.debug("Could not read safetensors metadata from %s: %s", sf, exc)
                continue

    if not found_safetensors:
        return SavedModelValidation(
            valid=False, error="No .safetensors files found"
        )

    return SavedModelValidation(
        valid=False, error="Could not read safetensors metadata"
    )
