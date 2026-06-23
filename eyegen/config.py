"""Configuration and path constants for EyeGen."""

import json
import logging
import sys
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_BUNDLED = getattr(sys, "frozen", None) == "macosx_app"

MODELS_DIR = Path.home() / "models" / "eyegen"
HF_CACHE_DIR = Path.home() / "models" / ".hf-cache" / "hub"

if _BUNDLED:
    _APP_SUPPORT = Path.home() / "Library" / "Application Support" / "EyeGen"
    CONFIG_DIR = _APP_SUPPORT
    CONFIG_FILE = _APP_SUPPORT / "config.json"
    OUTPUT_DIR = Path.home() / "Pictures" / "EyeGen"
    PROJECT_ROOT = Path.home()
else:
    PROJECT_ROOT = Path(__file__).parent.parent
    CONFIG_DIR = PROJECT_ROOT / "config"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    CONFIG_FILE = CONFIG_DIR / "config.json"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class Backend(str, Enum):
    """Supported image-generation backends."""

    AUTO = "auto"
    MLX = "mlx"
    OLLAMA = "ollamadiffuser"
    MFLUX = "mflux"
    BONSAI = "bonsai"
    COREML = "coreml"


BACKEND_AUTO = Backend.AUTO
BACKEND_MLX = Backend.MLX
BACKEND_OLLAMA = Backend.OLLAMA
BACKEND_MFLUX = Backend.MFLUX
BACKEND_BONSAI = Backend.BONSAI
BACKEND_COREML = Backend.COREML


@dataclass
class EyeGenConfig:
    model: str = "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized"
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    backend: Backend = Backend.AUTO
    mflux_quantize: Optional[int] = 4
    mflux_model_path: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    bonsai_model_path: Optional[str] = None
    coreml_model_path: Optional[str] = None
    coreml_compute_unit: str = "CPU_AND_NE"

    def validate(self) -> list[str]:
        """Validate settings and return a list of error messages (empty if valid)."""
        errors = []
        if not isinstance(self.backend, Backend):
            errors.append(f"Invalid backend '{self.backend}'. Must be a Backend enum value.")
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
            if self.backend not in (Backend.AUTO, Backend.MLX):
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

        cls._coerce_int(kwargs, "height")
        cls._coerce_int(kwargs, "width")
        cls._coerce_int(kwargs, "num_inference_steps")
        cls._coerce_float(kwargs, "guidance_scale")
        cls._coerce_optional_int(kwargs, "mflux_quantize")
        cls._coerce_backend(kwargs)
        return cls(**kwargs)

    @staticmethod
    def _coerce_int(kwargs: dict, key: str):
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = int(kwargs[key])

    @staticmethod
    def _coerce_float(kwargs: dict, key: str):
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = float(kwargs[key])

    @staticmethod
    def _coerce_optional_int(kwargs: dict, key: str):
        if key not in kwargs or kwargs[key] is None:
            return
        if str(kwargs[key]).strip().lower() in ("null", "none"):
            kwargs[key] = None
        else:
            kwargs[key] = int(kwargs[key])

    @staticmethod
    def _coerce_backend(kwargs: dict):
        if "backend" in kwargs and kwargs["backend"] is not None:
            kwargs["backend"] = Backend(kwargs["backend"])

    def to_dict(self) -> dict:
        data = asdict(self)
        data["backend"] = self.backend.value
        return data


DEFAULT_CONFIG = EyeGenConfig().to_dict()


def load_config() -> dict:
    """Load configuration from config.json or return defaults."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            try:
                data = json.load(f)
                cfg = EyeGenConfig.from_dict(data)
                return cfg.to_dict()
            except (json.JSONDecodeError, ValueError, TypeError) as e:
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
