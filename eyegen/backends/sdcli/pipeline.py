"""Entry point for the stable-diffusion.cpp backend.

Building the wrapper that runs a model. The wrapper itself lives in
``runner.py``; what the binary receives and how the render loop works are
in ``render.py``; the file roles and their flags are in ``constants.py``.
"""

from __future__ import annotations

from typing import Optional

from eyegen.config import EyeGenConfig
from eyegen.model_specs import ModelSpec

from .constants import resolve_settings  # noqa: F401 — re-exported for back-compat
from .runner import SdCliWrapper  # noqa: F401 — re-exported for back-compat


def get_sdcli_pipeline(
    config: EyeGenConfig,
    model: Optional[str] = None,
    spec: Optional[ModelSpec] = None,
) -> SdCliWrapper:
    """Build the wrapper for the configured model."""
    name = model or config.model or ""
    return SdCliWrapper(config, name, spec=spec)


def diffusion_path(spec: ModelSpec) -> str:
    """The transformer's own path, for the message that names the model."""
    source = spec.sources.get("diffusion_model")
    return source.path if source is not None else ""
