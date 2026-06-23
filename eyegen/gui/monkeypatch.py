"""diffusionkit sample_euler monkeypatch for step-level progress."""

import logging
import time

log = logging.getLogger("eyegen")

_sample_euler_patched = False
_sample_euler_state = {"progress_callback": None, "cancel_event": None}


class GenerationCancelled(Exception):
    """Raised when the user cancels a running generation."""


def _patch_sample_euler(progress_callback, cancel_event=None):
    """Replace diffusionkit's sample_euler with a step-reporting wrapper."""
    global _sample_euler_patched
    _sample_euler_state["progress_callback"] = progress_callback
    _sample_euler_state["cancel_event"] = cancel_event
    if _sample_euler_patched:
        return

    import diffusionkit.mlx as dkmlx
    import mlx.core as mx

    def _patched_sample_euler(model, x, sigmas, extra_args=None):
        extra_args = {} if extra_args is None else extra_args
        total = len(sigmas) - 1

        timesteps = model.model.sampler.timestep(sigmas).astype(model.model.activation_dtype)
        model.cache_modulation_params(extra_args.pop("pooled_conditioning"), timesteps)

        iter_time = []
        cancel = _sample_euler_state["cancel_event"]
        cb = _sample_euler_state["progress_callback"]
        for i in range(total):
            if cancel is not None and cancel.is_set():
                model.clear_cache()
                raise GenerationCancelled("Cancelled by user")
            t0 = time.perf_counter()
            denoised = model(x, timesteps[i], sigmas[i], **extra_args)
            d = dkmlx.to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
            mx.eval(x)
            iter_time.append(round(time.perf_counter() - t0, 3))
            cb(i + 1, total)

        model.clear_cache()
        return x, iter_time

    dkmlx.sample_euler = _patched_sample_euler
    _sample_euler_patched = True
