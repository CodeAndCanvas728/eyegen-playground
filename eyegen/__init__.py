"""EyeGen — multi-backend image generation on Apple Silicon."""

from eyegen._model_ops import (
    clear_mflux_cache,
    list_mflux_models,
    list_ollama_models,
    pull_any_model,
    pull_model,
    pull_spec,
    save_mflux_model,
)
from eyegen.backends import (
    bonsai,
    coreml,
    detect_backend,
    get_mflux_pipeline,
    get_ollama_pipeline,
    get_pipeline,
    resolve_model,
    sdcli,
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
from eyegen.errors import ModelBackendConflict, QuantizationError, UnsupportedModelError
from eyegen.generation import generate_image
from eyegen.hf import hf_login, hf_logout, hf_status
from eyegen.model_catalogue import (
    find_spec,
    load_catalogue,
    remove_spec,
    upsert_spec,
)
from eyegen.model_specs import (
    ModelSource,
    ModelSpec,
    ResolvedModel,
    apply_spec,
)
from eyegen.saved_models import SavedModelValidation, validate_saved_model
from eyegen.validation import (
    sanitize_prompt,
    validate_dimensions,
    validate_image_path,
)

__all__ = [
    "Backend",
    "bonsai",
    "coreml",
    "sdcli",
    "CONFIG_DIR",
    "CONFIG_FILE",
    "DEFAULT_CONFIG",
    "EyeGenConfig",
    "HF_CACHE_DIR",
    "MODELS_DIR",
    "ModelBackendConflict",
    "OUTPUT_DIR",
    "PROJECT_ROOT",
    "ModelSource",
    "ModelSpec",
    "QuantizationError",
    "ResolvedModel",
    "SavedModelValidation",
    "UnsupportedModelError",
    "apply_spec",
    "clear_mflux_cache",
    "detect_backend",
    "find_spec",
    "generate_image",
    "get_mflux_pipeline",
    "get_ollama_pipeline",
    "get_pipeline",
    "hf_login",
    "hf_logout",
    "hf_status",
    "list_mflux_models",
    "list_ollama_models",
    "load_catalogue",
    "load_config",
    "pull_any_model",
    "pull_model",
    "pull_spec",
    "remove_spec",
    "resolve_model",
    "save_config",
    "save_mflux_model",
    "sanitize_prompt",
    "upsert_spec",
    "validate_dimensions",
    "validate_image_path",
    "validate_saved_model",
]
