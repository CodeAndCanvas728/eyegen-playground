"""Internal MLX/diffusionkit compatibility helpers."""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

_patch_lock = threading.Lock()
_patched_already = False


def _make_patched(orig):
    """Wrap *orig* function to strip out the removed keyword arg."""
    _REMOVED = {"memory_efficient_threshold"}

    def _patched(*args, **kwargs):
        return orig(*args, **{k: v for k, v in kwargs.items() if k not in _REMOVED})

    return _patched


def _patch_mlx_attention():
    """Compatibility shim: diffusionkit passes `memory_efficient_threshold` to
    scaled_dot_product_attention, but newer MLX dropped that parameter.
    """
    global _patched_already
    if _patched_already:
        return

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
