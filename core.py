"""
Shared logic for EyeGen — multi-backend image generation on Apple Silicon.
Used by both the CLI (generate.py) and the GUI (gui.py).

Supports five backends:
  - MLX (diffusionkit) — Apple Silicon native, SD3.5 quantized
  - MFLUX — MLX-native FLUX, FLUX.2, Z-Image, FIBO, Qwen, SeedVR2 models
  - OllamaDiffuser — GGUF-quantized models (FLUX, SDXL, SD1.5, SD3.5, etc.)
  - Bonsai (PrismML) — 1.58-bit ternary + 1-bit binary FLUX.2 Klein 4B
  - CoreML — Stable Diffusion 1.x/2.x via Apple Neural Engine
"""

import json
import logging
import random
import sys
import threading
import unicodedata
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Optional, Any

log = logging.getLogger(__name__)

# Detect whether we're running inside a py2app bundle.
# py2app sets sys.frozen = 'macosx_app' at runtime; the bundle's Resources
# directory is read-only, so user data must go elsewhere.
_BUNDLED = getattr(sys, "frozen", None) == "macosx_app"

# All non-HF-cached model artifacts (saved MFLUX, downloaded Bonsai,
# converted CoreML) live under a single tree at ~/models/eyegen/ regardless
# of dev vs bundled mode. HF downloads themselves use ~/models/.hf-cache/hub/
# by default (configurable via hf_cache_dir). This keeps every model artifact
# the user owns under one parent directory, easy to back up or relocate.
MODELS_DIR = Path.home() / "models" / "eyegen"
HF_CACHE_DIR = Path.home() / "models" / ".hf-cache" / "hub"

if _BUNDLED:
    # User-writable locations when running as a .app bundle
    _APP_SUPPORT = Path.home() / "Library" / "Application Support" / "EyeGen"
    CONFIG_DIR = _APP_SUPPORT
    CONFIG_FILE = _APP_SUPPORT / "config.json"
    OUTPUT_DIR = Path.home() / "Pictures" / "EyeGen"
    PROJECT_ROOT = Path.home()  # not meaningful in bundle context
else:
    # Development / CLI mode: config + outputs live in the project tree
    PROJECT_ROOT = Path(__file__).parent
    CONFIG_DIR = PROJECT_ROOT / "config"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    CONFIG_FILE = CONFIG_DIR / "config.json"

# Ensure directories exist
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Backend constants
# ---------------------------------------------------------------------------

BACKEND_AUTO = "auto"
BACKEND_MLX = "mlx"
BACKEND_OLLAMA = "ollamadiffuser"
BACKEND_MFLUX = "mflux"
BACKEND_BONSAI = "bonsai"
BACKEND_COREML = "coreml"
VALID_BACKENDS = (BACKEND_AUTO, BACKEND_MLX, BACKEND_OLLAMA, BACKEND_MFLUX, BACKEND_BONSAI, BACKEND_COREML)

@dataclass
class EyeGenConfig:
    model: str = "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized"
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    backend: str = BACKEND_AUTO
    mflux_quantize: Optional[int] = 4
    mflux_model_path: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    bonsai_model_path: Optional[str] = None
    coreml_model_path: Optional[str] = None
    coreml_compute_unit: str = "CPU_AND_NE"

    def validate(self) -> list[str]:
        """Validate settings and return a list of error messages (empty if valid)."""
        errors = []
        if self.backend not in VALID_BACKENDS:
            errors.append(f"Invalid backend '{self.backend}'. Must be one of {VALID_BACKENDS}.")
        if self.width % 8 != 0 or self.height % 8 != 0:
            errors.append("Height and width must be multiples of 8.")
        if self.mflux_quantize not in (4, 8, None):
            errors.append("Quantization level must be 4, 8, or None.")
        if self.coreml_compute_unit not in ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"):
            errors.append(
                f"Invalid coreml_compute_unit '{self.coreml_compute_unit}'. "
                "Must be one of: CPU_ONLY, CPU_AND_GPU, CPU_AND_NE, ALL."
            )
        if self.model == "mlx-community/Lance-3B-AWQ-INT4":
            if self.backend not in (BACKEND_AUTO, BACKEND_MLX):
                errors.append("Lance-3B AWQ-INT4 model requires the MLX backend.")
        return errors

    @classmethod
    def from_dict(cls, data: dict) -> "EyeGenConfig":
        """Build config from dict, applying type casting and filtering unknown keys."""
        valid_keys = {f.name for f in fields(cls)}
        kwargs = {}
        for k, v in data.items():
            if k in valid_keys:
                if v == "null" or v == "None":
                    v = None
                kwargs[k] = v
        # Perform typing coercion
        if "height" in kwargs and kwargs["height"] is not None:
            kwargs["height"] = int(kwargs["height"])
        if "width" in kwargs and kwargs["width"] is not None:
            kwargs["width"] = int(kwargs["width"])
        if "num_inference_steps" in kwargs and kwargs["num_inference_steps"] is not None:
            kwargs["num_inference_steps"] = int(kwargs["num_inference_steps"])
        if "guidance_scale" in kwargs and kwargs["guidance_scale"] is not None:
            kwargs["guidance_scale"] = float(kwargs["guidance_scale"])
        if "mflux_quantize" in kwargs and kwargs["mflux_quantize"] is not None:
            if str(kwargs["mflux_quantize"]).strip().lower() in ("null", "none"):
                kwargs["mflux_quantize"] = None
            else:
                kwargs["mflux_quantize"] = int(kwargs["mflux_quantize"])
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return asdict(self)

# Backward-compatibility layer for other modules importing DEFAULT_CONFIG
DEFAULT_CONFIG = EyeGenConfig().to_dict()


# ---------------------------------------------------------------------------
# Quantization error
# ---------------------------------------------------------------------------

class QuantizationError(RuntimeError):
    """Raised when MLX quantized-weight dequantization fails at inference.

    Callers can catch this to retry with ``quantize=None`` (full precision).
    """

    def __init__(self, original: Exception, model_name: str = ""):
        self.original = original
        self.model_name = model_name
        super().__init__(
            f"Quantization error for model '{model_name}': {original}. "
            "Try regenerating with full precision (quantize=None) or clear "
            "the model cache with: ./generate.py clear-cache"
        )


def _is_quantization_error(exc: Exception) -> bool:
    """Return True if *exc* looks like the MLX dequantize uint32 ValueError."""
    msg = str(exc).lower()
    return "dequantize" in msg or ("uint32" in msg and "matrix" in msg)


# ---------------------------------------------------------------------------
# MFLUX model alias cache (lazy-loaded)
# ---------------------------------------------------------------------------

_mflux_aliases: set | None = None


def _get_mflux_aliases() -> set:
    """Return the set of all known MFLUX model aliases and HF model names.

    Lazily loaded from the mflux package on first call.
    Falls back to a static set if mflux is not importable.
    """
    global _mflux_aliases
    if _mflux_aliases is not None:
        return _mflux_aliases

    # Static fallback — covers common models even if mflux isn't installed
    _mflux_aliases = {
        "dev", "schnell", "dev-kontext", "dev-fill", "dev-redux", "dev-depth",
        "dev-controlnet-canny", "schnell-controlnet-canny",
        "dev-controlnet-upscaler", "dev-fill-catvton", "krea-dev",
        "flux2-klein-4b", "flux2-klein-9b", "flux2-klein-base-4b",
        "flux2-klein-base-9b", "flux2-klein-4B", "flux2-klein-9B",
        "klein-4b", "klein-9b", "klein-base-4b", "klein-base-9b",
        "qwen-image", "qwen-image-edit", "qwen", "qwen-edit",
        "fibo", "fibo-lite", "fibo-edit", "fibo-edit-rmbg",
        "z-image", "z-image-turbo", "zimage", "zimage-turbo",
        "seedvr2-3b", "seedvr2-7b", "seedvr2",
    }

    # Try to augment from live package
    try:
        from mflux.models.common.config.model_config import AVAILABLE_MODELS
        for cfg in AVAILABLE_MODELS.values():
            _mflux_aliases.update(cfg.aliases)
            _mflux_aliases.add(cfg.model_name)
    except Exception as exc:
        log.debug("Could not augment MFLUX aliases from live package: %s", exc)

    return _mflux_aliases


def _get_mlx_supported_models() -> set | None:
    """Return the set of model keys known to the MLX/diffusionkit backend.

    Returns ``None`` if diffusionkit is not importable (don't block detection
    in that case — the user will get the standard import error from
    ``get_pipeline``).
    """
    try:
        from diffusionkit.mlx import MMDIT_CKPT
        return set(MMDIT_CKPT.keys())
    except Exception as exc:
        log.debug("Could not introspect diffusionkit MMDIT_CKPT: %s", exc)
        return None


def _format_unsupported_error(model: str, attempted_backend: str) -> str:
    """Build a clear 'model not supported' message listing what *is* supported."""
    mlx_keys = _get_mlx_supported_models() or set()
    mflux_aliases = _get_mflux_aliases() - {""}
    parts = [f"Model '{model}' is not supported by the {attempted_backend} backend."]
    if mlx_keys:
        mlx_list = ", ".join(sorted(mlx_keys))
        parts.append(
            f"MLX (diffusionkit) supports: {mlx_list}."
        )
    if mflux_aliases:
        aliases_list = ", ".join(sorted(mflux_aliases))
        parts.append(
            f"MFLUX supports (use --backend mflux): {aliases_list}."
        )
    parts.append(
        "For OllamaDiffuser (GGUF) models, pull one with "
        "`./generate.py pull <name>` first."
    )
    return " ".join(parts)


def _is_bonsai_model(model: str) -> bool:
    """Return True if *model* looks like a bonsai (PrismML) identifier.

    Accepts: ``bonsai-ternary-mlx``, ``bonsai-binary-mlx``, the full HF
    repo names ``bonsai-image-4B-ternary-mlx`` etc., and the PrismML
    ``prism-ml/bonsai-image-ternary-4B-mlx-2bit`` style identifiers.
    """
    m = model.lower()
    return (
        m.startswith("bonsai-")
        or m.startswith("bonsai-image-4b-")
        or m.startswith("prism-ml/bonsai-")
        or "bonsai-image-4b" in m
    )


def _is_coreml_model(model: str, config: Optional[dict] = None) -> bool:
    """Return True if *model* is a CoreML bundle or a known pre-converted HF id.

    Triggers:
      - ``coreml_model_path`` is set in *config* (any value, including paths
        not yet on disk — the loader will surface the actual error)
      - *model* matches a key in :data:`core_coreml.PRECONVERTED_HF_MODELS`
      - *model* looks like a CoreML HF id (``apple/coreml-stable-diffusion-*``)
    """
    if config and config.get("coreml_model_path"):
        return True
    m = model.lower()
    if m in {"sd-1-4", "sd-1-5", "sd-1-5-palettized",
             "sd-2-base", "sd-2-1-base", "sd-2-1-base-palettized",
             "sdxl-base", "sdxl-ios"}:
        return True
    if m.startswith("apple/coreml-stable-diffusion"):
        return True
    return False


def detect_backend(model: str, override: str = BACKEND_AUTO,
                   config: Optional[dict] = None) -> str:
    """Resolve which backend to use.

    If *override* is ``"auto"``:
      1. ``"gguf"`` in model name (case-insensitive) → ollamadiffuser
      2. bonsai (PrismML) identifier → bonsai
      3. CoreML bundle / pre-converted HF id / coreml_model_path set → coreml
      4. model name matches a known MFLUX alias → mflux
      5. model name is registered in diffusionkit's MMDIT_CKPT → mlx
      6. otherwise → raises ValueError with the list of supported models per backend
    """
    if override != BACKEND_AUTO:
        return override
    if "gguf" in model.lower():
        return BACKEND_OLLAMA
    if _is_bonsai_model(model):
        return BACKEND_BONSAI
    if _is_coreml_model(model, config=config):
        return BACKEND_COREML
    if model in _get_mflux_aliases():
        return BACKEND_MFLUX
    mlx_keys = _get_mlx_supported_models()
    if mlx_keys is not None and model in mlx_keys:
        return BACKEND_MLX
    if mlx_keys is None:
        return BACKEND_MLX
    raise ValueError(_format_unsupported_error(model, "auto-detected"))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load configuration from config.json or return defaults, using EyeGenConfig validation."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            try:
                data = json.load(f)
                cfg = EyeGenConfig.from_dict(data)
                return cfg.to_dict()
            except Exception as e:
                log.warning("Failed to load config.json, resetting to defaults: %s", e)
    return EyeGenConfig().to_dict()


def save_config(config: dict):
    """Save configuration to config.json after running EyeGenConfig validation."""
    cfg = EyeGenConfig.from_dict(config)
    errors = cfg.validate()
    if errors:
        raise ValueError("; ".join(errors))
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# MLX / diffusionkit backend
# ---------------------------------------------------------------------------

_patch_lock = threading.Lock()
_patched_already = False

def _patch_mlx_attention():
    """Compatibility shim: diffusionkit passes `memory_efficient_threshold` to
    scaled_dot_product_attention, but newer MLX dropped that parameter.

    Uses a thread lock to ensure this patch is safely applied exactly once.
    """
    global _patched_already
    if _patched_already:
        return

    with _patch_lock:
        if _patched_already:
            return

        import mlx.core
        import mlx.core.fast

        _REMOVED = {"memory_efficient_threshold"}

        def _make_patched(orig):
            def _patched(*args, **kwargs):
                return orig(*args, **{k: v for k, v in kwargs.items() if k not in _REMOVED})
            return _patched

        if hasattr(mlx.core.fast, "scaled_dot_product_attention"):
            mlx.core.fast.scaled_dot_product_attention = _make_patched(
                mlx.core.fast.scaled_dot_product_attention
            )

        if hasattr(mlx.core, "scaled_dot_product_attention"):
            mlx.core.scaled_dot_product_attention = _make_patched(
                mlx.core.scaled_dot_product_attention
            )

        _patched_already = True


def _apply_hf_cache(config: dict):
    """Set HF_HUB_CACHE env var from config if a custom directory is configured.

    HuggingFace hub reads this env var when downloading or locating cached
    models, so setting it before any pipeline load redirects all downloads
    and cache lookups to the specified directory.

    Priority:
      1. ``config["hf_cache_dir"]`` if set (explicit)
      2. ``$HF_HUB_CACHE`` env var if already set in the shell (respect it)
      3. ``~/models/.hf-cache/hub/`` (project default — all EyeGen HF
         downloads land under one user-owned tree)
    """
    import os
    cache_dir = config.get("hf_cache_dir") if config else None
    if cache_dir:
        p = Path(cache_dir).expanduser()
    elif os.environ.get("HF_HUB_CACHE"):
        p = Path(os.environ["HF_HUB_CACHE"]).expanduser()
    else:
        p = HF_CACHE_DIR
    p.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = str(p)
    log.info("HF_HUB_CACHE set to %s", p)


def get_pipeline(config: dict, use_t5: bool = True):
    """Load and return a diffusionkit DiffusionPipeline (MLX backend).

    Applies the MLX attention patch first.
    """
    _apply_hf_cache(config)
    _patch_mlx_attention()
    from diffusionkit.mlx import DiffusionPipeline

    model_version = config.get("model", DEFAULT_CONFIG["model"])
    mlx_keys = _get_mlx_supported_models()
    if mlx_keys is not None and model_version not in mlx_keys:
        raise ValueError(_format_unsupported_error(model_version, BACKEND_MLX))
    pipeline = DiffusionPipeline(
        model_version=model_version,
        shift=3.0,
        use_t5=use_t5,
        low_memory_mode=True,
        a16=True,
        w16=True,
    )
    return pipeline


def _generate_image_mlx(pipeline, prompt: str, cfg_weight: float, num_steps: int,
                        width: int, height: int, seed: Optional[int] = None,
                        negative_prompt: str = "",
                        image_path: Optional[str] = None,
                        denoise: float = 1.0):
    """Run the MLX/diffusionkit pipeline and return a PIL Image."""
    image, _latent = pipeline.generate_image(
        prompt,
        cfg_weight=cfg_weight,
        num_steps=num_steps,
        latent_size=(height // 8, width // 8),
        seed=seed,
        negative_text=negative_prompt,
        image_path=image_path,
        denoise=denoise,
    )
    return image


# ---------------------------------------------------------------------------
# OllamaDiffuser backend
# ---------------------------------------------------------------------------

def get_ollama_pipeline(config: dict):
    """Load an ollamadiffuser model and return the engine.

    The engine exposes ``generate_image()``.
    """
    _apply_hf_cache(config)
    from ollamadiffuser.core.models.manager import model_manager

    model_name = config.get("model", "")
    if not model_manager.is_model_installed(model_name):
        raise RuntimeError(
            f"GGUF model '{model_name}' is not installed. "
            f"Pull it first:  ./generate.py pull \"{model_name}\""
        )
    model_manager.load_model(model_name)
    engine = model_manager.loaded_model
    if engine is None:
        raise RuntimeError(f"Failed to load ollamadiffuser model: {model_name}")
    return engine


def _generate_image_ollama(engine, prompt: str, cfg_weight: float, num_steps: int,
                           width: int, height: int, seed: Optional[int] = None,
                           negative_prompt: str = "",
                           image_path: Optional[str] = None,
                           denoise: float = 1.0):
    """Run ollamadiffuser engine and return a PIL Image.

    Maps our parameter names to the ollamadiffuser API.
    """
    kwargs = {}

    # img2img: load image and pass as 'image' + 'strength'
    if image_path:
        from PIL import Image as PILImage
        kwargs["image"] = PILImage.open(image_path).convert("RGB")
        kwargs["strength"] = denoise

    # Seed — create a torch Generator with the user's seed
    if seed is not None:
        try:
            import torch
            device = engine.device if hasattr(engine, "device") else "cpu"
            if device == "mps":
                # MPS generator must be created on CPU then used on MPS
                kwargs["generator"] = torch.Generator("cpu").manual_seed(seed)
            else:
                kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)
        except Exception as exc:
            log.debug("Seed not supported by engine, continuing: %s", exc)

    image = engine.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt or "low quality, bad anatomy, worst quality, low resolution",
        num_inference_steps=num_steps,
        guidance_scale=cfg_weight,
        width=width,
        height=height,
        **kwargs,
    )
    return image


def pull_model(model_name: str, progress_callback=None, hf_cache_dir: str | None = None) -> bool:
    """Download a GGUF model via ollamadiffuser.

    *progress_callback*, if provided, is called with ``(message: str)``
    for status updates.
    """
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


# ---------------------------------------------------------------------------
# MFLUX backend
# ---------------------------------------------------------------------------

def _resolve_mflux_class(model_config):
    """Given a resolved ModelConfig, return the appropriate model class.

    Different MFLUX model families use different Python classes.
    """
    name = model_config.model_name

    if "FLUX.2" in name:
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
        return Flux2Klein
    if "Z-Image" in name:
        from mflux.models.z_image.variants.z_image import ZImage
        return ZImage
    if "FIBO" in name or "Fibo" in name:
        from mflux.models.fibo.variants.txt2img.fibo import FIBO
        return FIBO
    if "Qwen" in name:
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
        return QwenImage
    if "SeedVR2" in name:
        from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
        return SeedVR2

    # Default to Flux1 (covers FLUX.1 and custom HF models)
    from mflux.models.flux.variants.txt2img.flux import Flux1
    return Flux1


def get_mflux_pipeline(config: dict, quantize: int | None = 4):
    """Load an MFLUX model and return the model instance.

    The model instance exposes ``generate_image()``.
    *quantize* can be 4 (default), 8, or None.

    If ``config["mflux_model_path"]`` points to a local directory of
    pre-quantized weights (created by :func:`save_mflux_model`), the
    model is loaded from disk and *quantize* is ignored — the stored
    quantization level takes precedence.
    """
    _apply_hf_cache(config)
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.utils.exceptions import ModelConfigError

    model_name = config.get("model", "dev")
    try:
        model_config = ModelConfig.from_name(model_name=model_name)
    except ModelConfigError as exc:
        aliases = ", ".join(sorted(_get_mflux_aliases()))
        raise ValueError(
            f"Model '{model_name}' is not recognized by MFLUX. "
            f"Known aliases: {aliases}. "
            f"Original error: {exc}"
        ) from exc
    cls = _resolve_mflux_class(model_config)

    local_path = config.get("mflux_model_path")
    if local_path:
        p = Path(local_path).expanduser()
        if not p.is_dir():
            raise FileNotFoundError(
                f"MFLUX model path does not exist: {p}\n"
                "Save a model first:  ./generate.py save-model dev --quantize 4"
            )
        log.info("Loading MFLUX model from local path: %s", p)
        return cls(
            model_config=model_config,
            model_path=str(p),
            quantize=None,  # saved model has baked-in quantization
        )

    return cls(
        model_config=model_config,
        quantize=quantize,
    )


def _generate_image_mflux(model, prompt: str, cfg_weight: float, num_steps: int,
                           width: int, height: int, seed: int | None = None,
                           negative_prompt: str = "",
                           image_path: str | None = None,
                           denoise: float = 1.0):
    """Run an MFLUX model and return a PIL Image.

    Maps our parameter names to the MFLUX API.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    kwargs = {
        "seed": seed,
        "prompt": prompt,
        "num_inference_steps": num_steps,
        "width": width,
        "height": height,
        "guidance": cfg_weight,
    }

    # img2img
    if image_path:
        kwargs["image_path"] = image_path
        kwargs["image_strength"] = denoise

    # negative_prompt (supported by Flux1, ZImage, Fibo; ignored by Flux2Klein)
    if negative_prompt:
        try:
            import inspect
            sig = inspect.signature(model.generate_image)
            if "negative_prompt" in sig.parameters:
                kwargs["negative_prompt"] = negative_prompt
        except Exception as exc:
            log.debug("Could not check negative_prompt support: %s", exc)

    try:
        result = model.generate_image(**kwargs)
    except ValueError as exc:
        if _is_quantization_error(exc):
            model_config = getattr(model, "model_config", None)
            model_name = getattr(model_config, "model_name", "unknown") if model_config else "unknown"
            raise QuantizationError(exc, model_name) from exc
        raise

    # Some models return GeneratedImage (with .image attr), others return PIL.Image directly
    if hasattr(result, "image"):
        return result.image
    return result


def list_mflux_models() -> list[dict]:
    """Return a list of available MFLUX model dicts with alias and HF name."""
    try:
        from mflux.models.common.config.model_config import AVAILABLE_MODELS
        return [
            {"alias": alias, "model_name": cfg.model_name}
            for alias, cfg in sorted(AVAILABLE_MODELS.items(), key=lambda x: x[1].priority)
        ]
    except Exception as exc:
        log.debug("Could not load MFLUX models from package, falling back to static list: %s", exc)
        aliases = sorted(_get_mflux_aliases())
        return [{"alias": a, "model_name": a} for a in aliases]


def clear_mflux_cache(model_alias: str | None = None) -> list[str]:
    """Delete cached MFLUX / HuggingFace model files and return removed paths.

    If *model_alias* is given, resolve it to an HF repo-id and delete only
    that model's cache.  Otherwise delete all FLUX-related caches.
    """
    from huggingface_hub import scan_cache_dir

    removed: list[str] = []
    repo_id: str | None = None

    if model_alias:
        try:
            from mflux.models.common.config.model_config import ModelConfig
            mc = ModelConfig.from_name(model_name=model_alias)
            repo_id = mc.model_name
        except Exception as exc:
            log.debug("Could not resolve model alias, using literal: %s", exc)
            repo_id = model_alias  # treat as literal HF repo-id

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
    """Download an MFLUX model, quantize it, and save to *output_path*.

    This is a one-time operation.  The saved directory contains weights
    **and** tokenizers so subsequent loads need no network access.

    *progress_callback*, if provided, is called with ``(message: str)``
    for status updates (downloading, quantizing, saving).

    Returns the resolved output path.
    """
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

    log.info("Model saved: %s", output_path)
    if progress_callback:
        progress_callback(f"✅ Model saved to {output_path}")
    return output_path


def validate_saved_model(path: str | Path) -> tuple[bool, dict | None]:
    """Check whether *path* contains a valid mflux-saved model.

    Returns ``(True, metadata)`` where *metadata* includes at least
    ``quantization_level`` (int or None) and ``mflux_version`` (str or None),
    or ``(False, None)`` if the directory is missing or incomplete.
    """
    p = Path(path).expanduser()
    if not p.is_dir():
        return False, None

    meta: dict = {"quantization_level": None, "mflux_version": None}
    found_safetensors = False

    try:
        from safetensors import safe_open
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
            except Exception as exc:
                log.debug("Could not read safetensors metadata from %s: %s", sf, exc)
                continue

    if not found_safetensors:
        return False, None

    return True, meta


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def sanitize_prompt(prompt: str) -> str:
    """Normalize Unicode and replace non-ASCII punctuation for the T5 tokenizer."""
    prompt = unicodedata.normalize("NFKC", prompt)
    return (
        prompt
        .replace("\u2014", "-")   # em dash
        .replace("\u2013", "-")   # en dash
        .replace("\u2018", "'")   # left single quote
        .replace("\u2019", "'")   # right single quote
        .replace("\u201c", '"')   # left double quote
        .replace("\u201d", '"')   # right double quote
        .replace("\u2026", "...")  # ellipsis
    )


def validate_dimensions(width: int, height: int) -> Optional[str]:
    """Return an error message if dimensions are invalid, or None if OK."""
    if width % 8 != 0 or height % 8 != 0:
        return "Height and width must be multiples of 8"
    return None


def validate_image_path(path: str) -> Optional[str]:
    """Return an error message if the image path is invalid, or None if OK."""
    p = Path(path)
    if not p.exists():
        return f"Image file not found: {path}"
    if p.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'):
        return f"Unsupported image format: {p.suffix}"
    return None


# ---------------------------------------------------------------------------
# Unified generate dispatcher
# ---------------------------------------------------------------------------

def generate_image(pipeline, prompt: str, cfg_weight: float, num_steps: int,
                   width: int, height: int, seed: Optional[int] = None,
                   negative_prompt: str = "",
                   image_path: Optional[str] = None,
                   denoise: float = 1.0,
                   backend: str = BACKEND_MLX):
    """Run the diffusion pipeline and return a PIL Image.

    *backend* should be a resolved backend (``"mlx"``, ``"ollamadiffuser"``,
    ``"mflux"``, or ``"bonsai"``), not ``"auto"`` — call :func:`detect_backend`
    first.
    """
    prompt = sanitize_prompt(prompt)
    negative_prompt = sanitize_prompt(negative_prompt) if negative_prompt else ""

    if backend == BACKEND_OLLAMA:
        return _generate_image_ollama(
            pipeline, prompt, cfg_weight, num_steps,
            width, height, seed, negative_prompt,
            image_path, denoise,
        )

    if backend == BACKEND_MFLUX:
        return _generate_image_mflux(
            pipeline, prompt, cfg_weight, num_steps,
            width, height, seed, negative_prompt,
            image_path, denoise,
        )

    if backend == BACKEND_BONSAI:
        return pipeline.generate_image(
            prompt=prompt, cfg_weight=cfg_weight, num_steps=num_steps,
            width=width, height=height, seed=seed,
            negative_prompt=negative_prompt,
            image_path=image_path, denoise=denoise,
        )

    if backend == BACKEND_COREML:
        return pipeline.generate_image(
            prompt=prompt, cfg_weight=cfg_weight, num_steps=num_steps,
            width=width, height=height, seed=seed,
            negative_prompt=negative_prompt,
            image_path=image_path, denoise=denoise,
        )

    return _generate_image_mlx(
        pipeline, prompt, cfg_weight, num_steps,
        width, height, seed, negative_prompt,
        image_path, denoise,
    )


# ---------------------------------------------------------------------------
# HuggingFace authentication
# ---------------------------------------------------------------------------

def hf_login(token: str) -> dict:
    """Log in to HuggingFace and return the user info dict.

    Saves the token to the local cache so subsequent model downloads
    (both MFLUX and MLX/diffusionkit) can access gated repos.
    Raises on failure.
    """
    from huggingface_hub import login, whoami
    login(token=token)
    return whoami()


def hf_status() -> Optional[dict]:
    """Return HuggingFace user info if logged in, or None."""
    from huggingface_hub import get_token, whoami
    if get_token() is None:
        return None
    try:
        return whoami()
    except Exception:
        return None


def hf_logout():
    """Remove the stored HuggingFace token."""
    from huggingface_hub import logout
    logout()
