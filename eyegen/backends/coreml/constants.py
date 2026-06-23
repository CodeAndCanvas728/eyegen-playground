"""CoreML backend constants."""

from pathlib import Path

DEFAULT_COREML_VENV = Path.home() / "models" / "eyegen" / ".coreml-venv"
DEFAULT_COREML_MODELS_DIR = Path.home() / "models" / "eyegen" / "coreml"
PYTHON_REQUIRED = "3.11"

PRECONVERTED_HF_MODELS = {
    "sd-1-4": "apple/coreml-stable-diffusion-v1-4",
    "sd-1-5": "apple/coreml-stable-diffusion-v1-5",
    "sd-1-5-palettized": "apple/coreml-stable-diffusion-v1-5-palettized",
    "sd-2-base": "apple/coreml-stable-diffusion-2-base",
    "sd-2-1-base": "apple/coreml-stable-diffusion-2-1-base",
    "sd-2-1-base-palettized": "apple/coreml-stable-diffusion-2-1-base-palettized",
    "sdxl-base": "apple/coreml-stable-diffusion-xl-base",
    "sdxl-ios": "apple/coreml-stable-diffusion-xl-base-ios",
}

VALID_COMPUTE_UNITS = ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL")
