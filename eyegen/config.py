"""Configuration and path constants for EyeGen."""

import json
import logging
import sys
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_config_warnings: list[str] = []


def pop_config_warnings() -> list[str]:
    """Return accumulated config-load warnings and clear them."""
    warnings = list(_config_warnings)
    _config_warnings.clear()
    return warnings


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
    subprocess_timeout: int = 300
    download_timeout: int = 1800
    convert_timeout: int = 1800

    def validate(self) -> list[str]:  # noqa: C901
        """Validate settings and return a list of error messages (empty if valid)."""
        errors = []
        if not isinstance(self.backend, Backend):
            errors.append(f"Invalid backend '{self.backend}'. Must be a Backend enum value.")
        if self.width <= 0 or self.height <= 0:
            errors.append("Height and width must be greater than 0.")
        elif self.width % 8 != 0 or self.height % 8 != 0:
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
        if self.subprocess_timeout <= 0:
            errors.append("Subprocess timeout must be greater than 0.")
        if self.download_timeout <= 0:
            errors.append("Download timeout must be greater than 0.")
        if self.convert_timeout <= 0:
            errors.append("Convert timeout must be greater than 0.")

        from eyegen.validation import validate_safe_path

        for path_field, path_val in [
            ("coreml_model_path", self.coreml_model_path),
            ("mflux_model_path", self.mflux_model_path),
            ("bonsai_model_path", self.bonsai_model_path),
            ("hf_cache_dir", self.hf_cache_dir),
        ]:
            if path_val:
                try:
                    validate_safe_path(path_val, path_field)
                except ValueError as exc:
                    errors.append(str(exc))

        return errors

    @classmethod
    def from_dict(cls, data: dict) -> "EyeGenConfig":
        """Build config from dict, applying type casting and rejecting unknown keys."""
        valid_keys = {f.name for f in fields(cls)}
        kwargs = {}
        for k, v in data.items():
            if k in valid_keys:
                if v == "null" or v == "None":
                    v = None
                kwargs[k] = v
            else:
                log.warning("Ignoring unknown config key: %s", k)

        cls._coerce_int(kwargs, "height")
        cls._coerce_int(kwargs, "width")
        cls._coerce_int(kwargs, "num_inference_steps")
        cls._coerce_float(kwargs, "guidance_scale")
        cls._coerce_optional_int(kwargs, "mflux_quantize")
        cls._coerce_int(kwargs, "subprocess_timeout")
        cls._coerce_int(kwargs, "download_timeout")
        cls._coerce_int(kwargs, "convert_timeout")
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

    @classmethod
    def _coerce_backend(cls, kwargs: dict):
        if "backend" not in kwargs or kwargs["backend"] is None:
            return
        val = kwargs["backend"]
        if isinstance(val, Backend):
            return
        if not isinstance(val, str):
            log.warning("Invalid backend type %r, migrating to AUTO", type(val))
            kwargs["backend"] = Backend.AUTO
            return

        val_clean = val.strip().lower()
        mapping = {
            "auto": Backend.AUTO,
            "": Backend.AUTO,
            "mlx": Backend.MLX,
            "ollama": Backend.OLLAMA,
            "ollamadiffuser": Backend.OLLAMA,
            "mflux": Backend.MFLUX,
            "bonsai": Backend.BONSAI,
            "prism": Backend.BONSAI,
            "prismml": Backend.BONSAI,
            "coreml": Backend.COREML,
        }
        if val_clean in mapping:
            kwargs["backend"] = mapping[val_clean]
            return

        try:
            kwargs["backend"] = Backend(val)
        except ValueError:
            log.warning("Unknown backend value %r, migrating to AUTO", val)
            kwargs["backend"] = Backend.AUTO

    def to_dict(self) -> dict:
        data = asdict(self)
        data["backend"] = self.backend.value
        return data


DEFAULT_CONFIG = EyeGenConfig()


def _backup_corrupted_config(exc: Exception) -> None:
    backup_file = CONFIG_FILE.with_suffix(".json.bak")
    msg = f"Corrupted config.json backed up to {backup_file} and reset to defaults."
    _config_warnings.append(msg)
    log.error(
        "Invalid JSON in config.json: %s. Backing up to %s and resetting to defaults.",
        exc,
        backup_file,
        exc_info=True,
    )
    try:
        if CONFIG_FILE.exists():
            CONFIG_FILE.replace(backup_file)
        save_config(EyeGenConfig())
    except Exception as write_err:
        log.error("Failed to write default config: %s", write_err)


def _handle_parse_error(exc: Exception) -> None:
    msg = "Config file had invalid values and was reset to defaults."
    _config_warnings.append(msg)
    log.error(
        "Failed to parse config.json, resetting to defaults: %s",
        exc,
        exc_info=True,
    )
    try:
        save_config(EyeGenConfig())
    except Exception as write_err:
        log.error("Failed to write default config: %s", write_err)


def load_config() -> EyeGenConfig:
    """Load configuration from config.json, migrate/normalize it, or return defaults."""
    if not CONFIG_FILE.exists():
        return EyeGenConfig()

    try:
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        _backup_corrupted_config(e)
        return EyeGenConfig()

    if not isinstance(data, dict):
        log.warning("config.json root is not a dictionary. Resetting to defaults.")
        _handle_parse_error(ValueError("Root is not a dictionary"))
        return EyeGenConfig()

    try:
        cfg = EyeGenConfig.from_dict(data)
        normalized_dict = cfg.to_dict()
        if normalized_dict != data:
            log.warning("Config required migration/normalization; saving config.")
            try:
                save_config(cfg)
            except Exception as save_err:
                log.error("Failed to save migrated config: %s", save_err)
        return cfg
    except (ValueError, TypeError) as e:
        _handle_parse_error(e)
        return EyeGenConfig()


def save_config(config: EyeGenConfig):
    """Save configuration to config.json after running EyeGenConfig validation."""
    errors = config.validate()
    if errors:
        raise ValueError("; ".join(errors))
    with open(CONFIG_FILE, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
