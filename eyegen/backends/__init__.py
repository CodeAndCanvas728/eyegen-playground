"""Backend detection and pipeline loaders for EyeGen."""

import logging
import os
from typing import Optional

from eyegen._mflux import _format_mflux_model_error, _get_mflux_aliases, _resolve_mflux_class
from eyegen._mlx import _get_mlx_supported_models, _patch_mlx_attention
from eyegen.config import DEFAULT_CONFIG, HF_CACHE_DIR, Backend, EyeGenConfig
from eyegen.errors import UnsupportedModelError

log = logging.getLogger(__name__)


def _is_bonsai_model(model: str) -> bool:
    """Return True if *model* looks like a bonsai (PrismML) identifier."""
    m = model.lower()
    return (
        m.startswith("bonsai-")
        or m.startswith("bonsai-image-4b-")
        or m.startswith("prism-ml/bonsai-")
        or "bonsai-image-4b" in m
    )


def _is_coreml_model(model: str, config: Optional[EyeGenConfig] = None) -> bool:
    """Return True if *model* is a CoreML bundle or a known pre-converted HF id."""
    if config and config.coreml_model_path:
        return True
    m = model.lower()
    if m in {
        "sd-1-4",
        "sd-1-5",
        "sd-1-5-palettized",
        "sd-2-base",
        "sd-2-1-base",
        "sd-2-1-base-palettized",
        "sdxl-base",
        "sdxl-ios",
    }:
        return True
    if m.startswith("apple/coreml-stable-diffusion"):
        return True
    return False


def _format_unsupported_error(model: str, attempted_backend: str) -> str:
    """Build a clear 'model not supported' message listing what *is* supported."""
    mlx_keys = _get_mlx_supported_models() or set()
    mflux_aliases = _get_mflux_aliases() - {""}
    parts = [f"Model '{model}' is not supported by the {attempted_backend} backend."]
    if mlx_keys:
        mlx_list = ", ".join(sorted(mlx_keys))
        parts.append(f"MLX (diffusionkit) supports: {mlx_list}.")
    if mflux_aliases:
        aliases_list = ", ".join(sorted(mflux_aliases))
        parts.append(f"MFLUX supports (use --backend mflux): {aliases_list}.")
    parts.append(
        "For OllamaDiffuser (GGUF) models, pull one with `./generate.py pull <name>` first."
    )
    parts.append(
        "For Bonsai (PrismML), use a model name starting with 'bonsai-' "
        "(e.g. bonsai-ternary-mlx) after running ./scripts/setup-bonsai.sh."
    )
    parts.append(
        "For CoreML, use a known alias (sd-1-4, sd-1-5, sd-2-1-base, "
        "sd-2-1-base-palettized) or a full HuggingFace repo id under "
        "'apple/coreml-stable-diffusion-*' after running ./scripts/setup-coreml.sh."
    )
    return " ".join(parts)


def detect_backend(
    model: str,
    override: Backend = Backend.AUTO,
    config: Optional[EyeGenConfig] = None,
) -> Backend:
    """Resolve which backend to use.

    If *override* is ``Backend.AUTO`` (priority order):
      1. ``"gguf"`` in model name → OLLAMA  (substr match first — bonsai/hf "gguf" can't hijack)
      2. Bonsai/PrismML identifier pattern → BONSAI
      3. CoreML bundle / pre-converted HF id / coreml_model_path set → COREML
      4. Known MFLUX alias → MFLUX
      5. diffusionkit MMDIT_CKPT entry → MLX
      6. otherwise → UnsupportedModelError
    """
    if isinstance(override, str):
        override = Backend(override)
    if override != Backend.AUTO:
        return override
    # (1) GGUF check first — substring match so it must run before other patterns
    if "gguf" in model.lower():
        return Backend.OLLAMA
    # (2) Bonsai — model names start with "bonsai-" or "prism-"
    if _is_bonsai_model(model):
        return Backend.BONSAI
    # (3) CoreML — known aliases or coreml_model_path set in config
    if _is_coreml_model(model, config=config):
        return Backend.COREML
    # (4) MFLUX — check live aliases from package, fall back to static set
    if model in _get_mflux_aliases():
        return Backend.MFLUX
    # (5) MLX/diffusionkit — registered MMDIT_CKPT keys
    mlx_keys = _get_mlx_supported_models()
    if mlx_keys is not None and model in mlx_keys:
        return Backend.MLX
    raise UnsupportedModelError(_format_unsupported_error(model, "auto-detected"))


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


def get_ollama_pipeline(config: EyeGenConfig):
    """Load an ollamadiffuser model and return the engine."""
    _apply_hf_cache(config)
    from ollamadiffuser.core.models.manager import model_manager

    model_name = config.model or ""
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
from eyegen.backends import bonsai, coreml  # noqa: E402,F401
