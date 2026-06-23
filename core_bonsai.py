"""
Bonsai (PrismML) backend — subprocess wrapper for the PrismML 1.58-bit
ternary + 1-bit binary diffusion transformer models.

This module does NOT import the bonsai kernels directly. Instead, it shells
out to the Bonsai-Image-Demo's own ``scripts/generate.sh``, which runs in
its dedicated Python 3.11 venv (``~/models/eyegen/bonsai-demo/.venv/``) with
the patched mflux-prism + MLX binaries that the 1.58-bit weight format needs.

Why a subprocess wrapper and not an in-process import?
    ``prism-image-studio`` (the public image-studio repo on GitHub) pins a
    patched mflux and a patched MLX in its ``pyproject.toml`` that conflict
    with the upstream ``mflux`` and ``mlx`` EyeGen uses for the existing
    MFLUX + MLX backends. The bonsai-demo's ``setup.sh`` resolves this in
    an isolated venv. Keeping that venv separate means EyeGen's existing
    backends are never affected by the bonsai vendor.

Tradeoff: every call pays the bonsai-demo's cold-start cost (imports +
~4 GiB weight load + first-shape MLX kernel compile), which is ~5-30s
depending on hardware. The bonsai-demo's own docs note this is a CPU/GPU
issue; on Apple Silicon the MLX kernels are precompiled Metal binaries
and cold-start is closer to 5s.

Lifecycle:
    1. User runs ``./scripts/setup-bonsai.sh`` (clones the bonsai-demo into
       ``~/models/eyegen/bonsai-demo/`` and runs its ``setup.sh``).
    2. User runs ``./generate.py pull-bonsai ternary-mlx`` (or the GUI
       equivalent) which invokes ``scripts/download_model.sh``.
    3. User can now select "bonsai" in the backend dropdown. The first
       ``generate`` call pays the cold-start; subsequent calls at the same
       shape benefit from the MLX metallib cache.
"""

from core_bonsai_constants import DEFAULT_VARIANT, SUPPORTED_VARIANTS
from core_bonsai_install import BonsaiInstallStatus, get_bonsai_dir, validate_bonsai_install
from core_bonsai_models import (
    download_bonsai_model,
    list_bonsai_models,
)
from core_bonsai_pipeline import BonsaiWrapper, get_bonsai_pipeline

__all__ = [
    "BonsaiInstallStatus",
    "BonsaiWrapper",
    "DEFAULT_VARIANT",
    "SUPPORTED_VARIANTS",
    "download_bonsai_model",
    "get_bonsai_dir",
    "get_bonsai_pipeline",
    "list_bonsai_models",
    "validate_bonsai_install",
]
