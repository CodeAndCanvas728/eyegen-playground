"""Internal MLX/diffusionkit compatibility helpers."""

from __future__ import annotations

import importlib.metadata
import logging
import threading

log = logging.getLogger(__name__)

_patch_lock = threading.Lock()
_patched_already = False

_MLX_KNOWN_GOOD = "0.20"


def _make_patched(orig):
    """Wrap *orig* function to strip out the removed keyword arg."""
    _REMOVED = {"memory_efficient_threshold"}

    def _patched(*args, **kwargs):
        return orig(*args, **{k: v for k, v in kwargs.items() if k not in _REMOVED})

    return _patched


def _check_mlx_version() -> str | None:
    """Return the installed MLX version string, or None if MLX is not installed."""
    try:
        return importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        return None


def _patch_mlx_attention():
    """Compatibility shim: diffusionkit passes `memory_efficient_threshold` to
    scaled_dot_product_attention, but newer MLX dropped that parameter.

    Skips patching if the MLX version is newer than the known-good range.
    """
    global _patched_already
    if _patched_already:
        return

    mlx_version = _check_mlx_version()
    if mlx_version and mlx_version not in (_MLX_KNOWN_GOOD,):
        log.warning(
            "MLX version %s may not be compatible with the attention patch "
            "(tested with %s). You may see 'unexpected keyword argument' errors "
            "from diffusionkit.",
            mlx_version,
            _MLX_KNOWN_GOOD,
        )

    with _patch_lock:
        if _patched_already:
            return

        import mlx.core
        import mlx.core.fast

        if hasattr(mlx.core.fast, "scaled_dot_product_attention"):
            mlx.core.fast.scaled_dot_product_attention = _make_patched(
                mlx.core.fast.scaled_dot_product_attention
            )

        if hasattr(mlx.core, "scaled_dot_product_attention"):
            mlx.core.scaled_dot_product_attention = _make_patched(
                mlx.core.scaled_dot_product_attention
            )

        _patched_already = True


def _get_mlx_supported_models() -> set | None:
    """Return the set of model keys known to the MLX/diffusionkit backend."""
    try:
        from diffusionkit.mlx import MMDIT_CKPT

        return set(MMDIT_CKPT.keys())
    except (ImportError, AttributeError) as exc:
        log.debug("Could not introspect diffusionkit MMDIT_CKPT: %s", exc)
        return None
