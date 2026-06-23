"""
CoreML backend — subprocess wrapper for the Apple ``python_coreml_stable_diffusion``
package, which provides Stable Diffusion inference on Apple Silicon via Core ML
(Neural Engine + GPU + CPU).

Why a subprocess wrapper and not an in-process import?
    Apple's ``python_coreml_stable_diffusion`` package pins a specific old
    dependency set: ``diffusers==0.30.2``, ``transformers==4.44.2``,
    ``huggingface-hub==0.24.6``, ``numpy<1.24``, ``diffusionkit==0.4.0``.
    These collide with EyeGen's main 3.14 venv (which uses diffusers 0.38,
    numpy 2.x, etc.). Rather than fragment EyeGen's main venv, we install
    Apple's package in a sidecar Python 3.11 venv at
    ``~/models/eyegen/.coreml-venv/`` and shell out to it.

Supported models:
    Stable Diffusion 1.4, 1.5, 2-base, 2.1-base. SDXL and SD3 have CoreML
    conversions in Apple's package but require more GPU memory and longer
    conversion times. For first-class support, start with the 1.x/2.x family.
    Pre-converted weights on HF (e.g. ``apple/coreml-stable-diffusion-2-1-base-palettized``)
    can be downloaded directly without running the conversion step.

Lifecycle:
    1. User runs ``./scripts/setup-coreml.sh`` (creates the sidecar venv and
       installs ``python_coreml_stable_diffusion``).
    2. User runs ``./generate.py convert-coreml stabilityai/stable-diffusion-2-1-base``
       to convert a model from PyTorch to CoreML (15-20 min on M1 Pro).
       OR skips conversion by using a pre-converted HF model:
       ``./generate.py pull-coreml apple/coreml-stable-diffusion-2-1-base-palettized``.
    3. User can now select "coreml" in the backend dropdown and pick the
       model directory.
"""

from .constants import PRECONVERTED_HF_MODELS, VALID_COMPUTE_UNITS
from .install import (
    CoreMLInstallStatus,
    get_coreml_models_dir,
    get_coreml_venv,
    validate_coreml_install,
)
from .models import (
    convert_to_coreml,
    list_coreml_models,
    pull_preconverted_coreml_model,
)
from .pipeline import CoreMLWrapper, get_coreml_pipeline

__all__ = [
    "CoreMLInstallStatus",
    "CoreMLWrapper",
    "PRECONVERTED_HF_MODELS",
    "VALID_COMPUTE_UNITS",
    "convert_to_coreml",
    "get_coreml_models_dir",
    "get_coreml_pipeline",
    "get_coreml_venv",
    "list_coreml_models",
    "pull_preconverted_coreml_model",
    "validate_coreml_install",
]
