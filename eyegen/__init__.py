"""EyeGen — multi-backend image generation on Apple Silicon."""

from eyegen._model_ops import (
    clear_mflux_cache,
    list_mflux_models,
    list_ollama_models,
    pull_model,
    save_mflux_model,
    validate_saved_model,
)
from eyegen.backends import (
    detect_backend,
    get_mflux_pipeline,
    get_ollama_pipeline,
    get_pipeline,
)
from eyegen.config import (
    CONFIG_DIR,
    CONFIG_FILE,
    DEFAULT_CONFIG,
    HF_CACHE_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    Backend,
    EyeGenConfig,
    load_config,
    save_config,
)
from eyegen.errors import QuantizationError
from eyegen.generation import generate_image
from eyegen.hf import hf_login, hf_logout, hf_status
from eyegen.validation import (
    sanitize_prompt,
    validate_dimensions,
    validate_image_path,
)

__all__ = [
    "Backend",
    "CONFIG_DIR",
    "CONFIG_FILE",
    "DEFAULT_CONFIG",
    "EyeGenConfig",
    "HF_CACHE_DIR",
    "MODELS_DIR",
    "OUTPUT_DIR",
    "PROJECT_ROOT",
    "QuantizationError",
    "clear_mflux_cache",
    "detect_backend",
    "generate_image",
    "get_mflux_pipeline",
    "get_ollama_pipeline",
    "get_pipeline",
    "hf_login",
    "hf_logout",
    "hf_status",
    "list_mflux_models",
    "list_ollama_models",
    "load_config",
    "pull_model",
    "save_config",
    "save_mflux_model",
    "sanitize_prompt",
    "validate_dimensions",
    "validate_image_path",
    "validate_saved_model",
]
