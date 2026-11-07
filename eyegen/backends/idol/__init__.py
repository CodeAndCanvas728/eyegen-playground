"""Idol (speridlabs/idol) image-to-3D backend — one image in, a 3D Gaussian
splat `.ply` out, run through the SDK in a sidecar venv (DD-020).

Why a subprocess rather than a library?
    The SDK needs torch 2.15 + transformers 4.57 in its own Python 3.11
    venv; those conflict with the main venv's stack, exactly like the
    CoreML and Bonsai sidecars. ``python -m idol.sample`` is the SDK's
    documented interface, and on Apple Silicon it runs on MPS via the
    unmerged PR #5 branch that setup-idol.sh checks out.

Lifecycle:
    1. ``./generate.py setup-idol`` — creates ``~/models/eyegen/idol/.venv``
       and clones the SDK (branch ``mps``).
    2. ``./generate.py to3d photo.jpg --mode real`` — the weights
       (~3 GB) and the two gated image encoders download into the
       HuggingFace cache on first run.
"""

from .constants import (
    IDOL_DEFAULT_DEVICE,
    IDOL_DEFAULT_MODE,
    IDOL_DEFAULT_NUM_GAUSSIANS,
    IDOL_DEFAULT_STEPS,
    IDOL_DEFAULT_TIMEOUT,
    IDOL_DEVICES,
    IDOL_GAUSSIAN_CHOICES,
    IDOL_IMAGE_EXTENSIONS,
    IDOL_MAX_GAUSSIANS,
    IDOL_MIN_GAUSSIANS,
    IDOL_MODES,
    IDOL_SDK_REF,
    IDOL_SDK_REPO,
    IDOL_WEIGHTS_REPO,
)
from .install import (
    IdolInstallStatus,
    get_idol_dir,
    get_idol_sdk_dir,
    get_idol_venv_python,
    validate_idol_install,
)
from .runner import IdolResult, IdolRunner, run_to3d

__all__ = [
    "IDOL_DEFAULT_DEVICE",
    "IDOL_DEFAULT_MODE",
    "IDOL_DEFAULT_NUM_GAUSSIANS",
    "IDOL_DEFAULT_STEPS",
    "IDOL_DEFAULT_TIMEOUT",
    "IDOL_DEVICES",
    "IDOL_GAUSSIAN_CHOICES",
    "IDOL_IMAGE_EXTENSIONS",
    "IDOL_MAX_GAUSSIANS",
    "IDOL_MIN_GAUSSIANS",
    "IDOL_MODES",
    "IDOL_SDK_REF",
    "IDOL_SDK_REPO",
    "IDOL_WEIGHTS_REPO",
    "IdolInstallStatus",
    "IdolResult",
    "IdolRunner",
    "get_idol_dir",
    "get_idol_sdk_dir",
    "get_idol_venv_python",
    "run_to3d",
    "validate_idol_install",
]
