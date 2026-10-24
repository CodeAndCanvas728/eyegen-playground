"""Backend detection and pipeline loaders for EyeGen."""

import logging
import os
from typing import Optional

from eyegen._mflux import _format_mflux_model_error, _resolve_mflux_class
from eyegen._mlx import _get_mlx_supported_models, _patch_mlx_attention
from eyegen.config import DEFAULT_CONFIG, HF_CACHE_DIR, Backend, EyeGenConfig
from eyegen.model_specs import ModelSpec

from .detection import (
    _format_unsupported_error,
    detect_backend,
    resolve_model,
    resolve_spec,
)

__all__ = [
    "detect_backend",
    "get_mflux_pipeline",
    "get_ollama_pipeline",
    "get_pipeline",
    "resolve_model",
    "resolve_spec",
]

log = logging.getLogger(__name__)


def _apply_hf_cache(config: EyeGenConfig):
    """Set HF_HUB_CACHE env var from config if a custom directory is configured."""
    from eyegen.validation import validate_safe_path

    if config.hf_cache_dir:
        p = validate_safe_path(config.hf_cache_dir, "hf_cache_dir")
    elif os.environ.get("HF_HUB_CACHE"):
        p = validate_safe_path(os.environ["HF_HUB_CACHE"], "HF_HUB_CACHE")
    else:
        p = HF_CACHE_DIR
    p.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = str(p)
    log.info("HF_HUB_CACHE set to %s", p)


def get_pipeline(config: EyeGenConfig, use_t5: bool = True):
    """Load and return a diffusionkit DiffusionPipeline (MLX backend)."""
    _apply_hf_cache(config)
    _patch_mlx_attention()
    from diffusionkit.mlx import DiffusionPipeline

    model_version = config.model or DEFAULT_CONFIG.model
    mlx_keys = _get_mlx_supported_models()
    if mlx_keys is not None and model_version not in mlx_keys:
        raise ValueError(_format_unsupported_error(model_version, Backend.MLX))
    return DiffusionPipeline(
        model_version=model_version,
        shift=3.0,
        use_t5=use_t5,
        low_memory_mode=True,
        a16=True,
        w16=True,
    )


def get_ollama_pipeline(
    config: EyeGenConfig,
    name: Optional[str] = None,
    spec: Optional[ModelSpec] = None,
):
    """Load an ollamadiffuser model and return the engine.

    A model that is already on this machine is registered with the package
    first: the package keeps its models in a directory of its own, so the
    file is linked into the directory it expects and the package's own
    settings are told about it.
    """
    _apply_hf_cache(config)
    from eyegen._model_ops import apply_ollama_registry

    apply_ollama_registry()

    model_name = name or config.model or ""
    if spec is not None and spec.backend == Backend.OLLAMA:
        from eyegen.ollama_registry import register_local_gguf

        model_name = register_local_gguf(spec, model_name)

    from ollamadiffuser.core.models.manager import model_manager

    if not model_manager.is_model_installed(model_name):
        raise RuntimeError(
            f"GGUF model '{model_name}' is not installed. "
            f'Pull it first:  ./generate.py pull "{model_name}"'
        )
    model_manager.load_model(model_name)
    engine = model_manager.loaded_model
    if engine is None:
        raise RuntimeError(f"Failed to load ollamadiffuser model: {model_name}")
    return engine


def get_mflux_pipeline(config: EyeGenConfig, quantize: int | None = 4):
    """Load an MFLUX model and return the model instance."""
    _apply_hf_cache(config)
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.utils.exceptions import ModelConfigError

    model_name = config.model or "dev"
    try:
        model_config = ModelConfig.from_name(model_name=model_name)
    except ModelConfigError as exc:
        raise _format_mflux_model_error(model_name, exc) from exc
    cls = _resolve_mflux_class(model_config)

    local_path = config.mflux_model_path
    if local_path:
        from eyegen.validation import validate_safe_path

        p = validate_safe_path(local_path, "mflux_model_path")
        if not p.is_dir():
            raise FileNotFoundError(
                f"MFLUX model path does not exist: {p}\n"
                "Save a model first:  ./generate.py save-model dev --quantize 4"
            )
        log.info("Loading MFLUX model from local path: %s", p)
        return cls(
            model_config=model_config,
            model_path=str(p),
            quantize=None,
        )

    return cls(
        model_config=model_config,
        quantize=quantize,
    )


# Re-export optional backend subpackages.
from eyegen.backends import bonsai, coreml, sdcli  # noqa: E402,F401
