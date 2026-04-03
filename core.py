"""
Shared logic for EyeGen — multi-backend image generation on Apple Silicon.
Used by both the CLI (generate.py) and the GUI (gui.py).

Supports three backends:
  - MLX (diffusionkit) — Apple Silicon native, SD3.5 quantized
  - OllamaDiffuser — GGUF-quantized models (FLUX, SDXL, SD1.5, SD3.5, etc.)
  - MFLUX — MLX-native FLUX, FLUX.2, Z-Image, FIBO, Qwen, SeedVR2 models
"""

import json
import sys
from pathlib import Path
from typing import Optional

# Detect whether we're running inside a py2app bundle.
# py2app sets sys.frozen = 'macosx_app' at runtime; the bundle's Resources
# directory is read-only, so user data must go elsewhere.
_BUNDLED = getattr(sys, "frozen", None) == "macosx_app"

if _BUNDLED:
    # User-writable locations when running as a .app bundle
    _APP_SUPPORT = Path.home() / "Library" / "Application Support" / "EyeGen"
    CONFIG_DIR = _APP_SUPPORT
    CONFIG_FILE = _APP_SUPPORT / "config.json"
    OUTPUT_DIR = Path.home() / "Pictures" / "EyeGen"
    PROJECT_ROOT = Path.home()  # not meaningful in bundle context
else:
    # Development / CLI mode: use the project directory
    PROJECT_ROOT = Path(__file__).parent
    CONFIG_DIR = PROJECT_ROOT / "config"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    CONFIG_FILE = CONFIG_DIR / "config.json"

# Ensure directories exist
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Backend constants
# ---------------------------------------------------------------------------

BACKEND_AUTO = "auto"
BACKEND_MLX = "mlx"
BACKEND_OLLAMA = "ollamadiffuser"
BACKEND_MFLUX = "mflux"
VALID_BACKENDS = (BACKEND_AUTO, BACKEND_MLX, BACKEND_OLLAMA, BACKEND_MFLUX)

DEFAULT_CONFIG = {
    "model": "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "backend": BACKEND_AUTO,
    "mflux_quantize": 4,
}


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
    except Exception:
        pass

    return _mflux_aliases


def detect_backend(model: str, override: str = BACKEND_AUTO) -> str:
    """Resolve which backend to use.

    If *override* is ``"auto"``:
      1. ``"gguf"`` in model name (case-insensitive) → ollamadiffuser
      2. model name matches a known MFLUX alias → mflux
      3. otherwise → mlx (diffusionkit)
    """
    if override != BACKEND_AUTO:
        return override
    if "gguf" in model.lower():
        return BACKEND_OLLAMA
    if model in _get_mflux_aliases():
        return BACKEND_MFLUX
    return BACKEND_MLX


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load configuration from config.json or return defaults."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return dict(DEFAULT_CONFIG)


def save_config(config: dict):
    """Save configuration to config.json."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# MLX / diffusionkit backend
# ---------------------------------------------------------------------------

def _patch_mlx_attention():
    """Compatibility shim: diffusionkit passes `memory_efficient_threshold` to
    scaled_dot_product_attention, but newer MLX dropped that parameter.

    We blacklist only the removed kwarg rather than introspecting the signature —
    inspect.signature() on MLX C extensions returns an incomplete result and
    accidentally strips required args like `scale`."""
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


def get_pipeline(config: dict, use_t5: bool = True):
    """Load and return a diffusionkit DiffusionPipeline (MLX backend).

    Applies the MLX attention patch first.
    """
    _patch_mlx_attention()
    from diffusionkit.mlx import DiffusionPipeline

    model_version = config.get("model", DEFAULT_CONFIG["model"])
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
        except Exception:
            pass  # seed not supported — degrade gracefully

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


def pull_model(model_name: str, progress_callback=None) -> bool:
    """Download a GGUF model via ollamadiffuser.

    *progress_callback*, if provided, is called with ``(message: str)``
    for status updates.
    """
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
        from mflux.models.fibo.variants.txt2img.fibo import Fibo
        return Fibo
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
    """
    from mflux.models.common.config.model_config import ModelConfig

    model_name = config.get("model", "dev")
    model_config = ModelConfig.from_name(model_name=model_name)
    cls = _resolve_mflux_class(model_config)

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
        import random
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
        except Exception:
            pass

    result = model.generate_image(**kwargs)

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
    except Exception:
        # Fall back to static list
        aliases = sorted(_get_mflux_aliases())
        return [{"alias": a, "model_name": a} for a in aliases]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def sanitize_prompt(prompt: str) -> str:
    """Replace Unicode punctuation with ASCII equivalents for the T5 tokenizer."""
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
    or ``"mflux"``), not ``"auto"`` — call :func:`detect_backend` first.
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
