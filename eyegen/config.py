"""Configuration and path constants for EyeGen."""

import logging
import sys
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Optional

from eyegen.paths import DEFAULT_MODELS_DIR

log = logging.getLogger(__name__)


_BUNDLED = getattr(sys, "frozen", None) == "macosx_app"

MODELS_DIR = DEFAULT_MODELS_DIR
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
    SDCLI = "sdcli"


BACKEND_AUTO = Backend.AUTO
BACKEND_MLX = Backend.MLX
BACKEND_OLLAMA = Backend.OLLAMA
BACKEND_MFLUX = Backend.MFLUX
BACKEND_BONSAI = Backend.BONSAI
BACKEND_COREML = Backend.COREML
BACKEND_SDCLI = Backend.SDCLI


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
    models_dir: Optional[str] = None
    bonsai_model_path: Optional[str] = None
    coreml_model_path: Optional[str] = None
    coreml_compute_unit: str = "CPU_AND_NE"
    subprocess_timeout: int = 300
    download_timeout: int = 1800
    convert_timeout: int = 1800
    #: Whether sd-cli places its weights in RAM and streams them to the GPU
    #: per layer instead of keeping them resident. There is no VRAM to save
    #: on Apple Silicon — the memory is unified — so the trade is a slower
    #: generate against a smaller peak footprint; the flag is sd-cli's
    #: ``--offload-to-cpu``.
    sdcli_offload_to_cpu: bool = True
    #: Whether the T5 text encoder is used. T5 is better quality but slower
    #: and not applicable to every backend.
    use_t5: bool = True

    def __post_init__(self):
        self._normalize_model()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ("model", "backend"):
            self._normalize_model()

    def _normalize_model(self):
        model_val = self.__dict__.get("model", None)
        backend_val = self.__dict__.get("backend", None)
        if model_val and backend_val == Backend.COREML and "/" in model_val:
            # A CoreML bundle is a directory on this machine, named without a
            # slash; a repository id reaches it only by way of a conversion.
            # A path that is on this machine is the bundle itself, and is
            # left as the path the user gave.
            from pathlib import Path

            if Path(model_val).expanduser().exists():
                return
            super().__setattr__("model", model_val.replace("/", "-"))

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
        if self.mflux_model_path:
            p = Path(self.mflux_model_path).expanduser()
            if not p.is_dir():
                errors.append(f"mflux_model_path does not exist: {p}")
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

        from eyegen.validation import validate_model_path

        for path_field, path_val in [
            ("coreml_model_path", self.coreml_model_path),
            ("mflux_model_path", self.mflux_model_path),
            ("bonsai_model_path", self.bonsai_model_path),
        ]:
            if path_val:
                try:
                    validate_model_path(path_val, path_field)
                except ValueError as exc:
                    errors.append(str(exc))

        # A cache is kept wherever the user keeps their models, which may be
        # a volume of its own; traversal is still refused.
        if self.hf_cache_dir:
            try:
                validate_model_path(self.hf_cache_dir, "hf_cache_dir")
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
        cls._coerce_bool(kwargs, "sdcli_offload_to_cpu")
        cls._coerce_bool(kwargs, "use_t5")
        cls._coerce_backend(kwargs)
        return cls(**kwargs)

    @staticmethod
    def _coerce_bool(kwargs: dict, key: str):
        if key in kwargs and kwargs[key] is not None and not isinstance(kwargs[key], bool):
            kwargs[key] = str(kwargs[key]).strip().lower() in ("1", "true", "yes", "on")

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

# The file half of the configuration lives with the file: load/save and
# the warnings they raise are re-exported here so callers keep one import.
from eyegen.config_io import load_config, pop_config_warnings, save_config  # noqa: E402,F401
