"""Loading a pipeline for a worker thread.

The GUI keeps one pipeline loaded at a time and reuses it while the
model and the settings behind it do not change. This is the half of that
which decides *which* pipeline to load and whether the one it has is the
one it wants.
"""

import hashlib
import json
import logging

from eyegen import (
    Backend,
    EyeGenConfig,
    get_mflux_pipeline,
    get_ollama_pipeline,
    get_pipeline,
)
from eyegen.backends import bonsai, coreml, sdcli
from eyegen.gui.cache import _pipeline_cache, get_cache_generation
from eyegen.gui.monkeypatch import _patch_sample_euler
from eyegen.model_specs import apply_spec

log = logging.getLogger("eyegen")


def _make_cache_key(backend, config: EyeGenConfig, mflux_quantize, use_t5, spec=None):
    """The key the loaded pipeline is kept under.

    The specification is part of the key: the catalogue is read when a model
    is looked up and never cached, so a model the user has just described
    differently is a different pipeline even when the name is the same.
    """
    return (
        backend,
        config.model,
        mflux_quantize,
        use_t5,
        config.mflux_model_path,
        config.bonsai_model_path,
        config.coreml_model_path,
        config.coreml_compute_unit,
        config.sdcli_offload_to_cpu,
        _spec_fingerprint(spec),
    )


def _spec_fingerprint(spec) -> str:
    """A short digest of the specification, or nothing when there is none."""
    if spec is None:
        return ""
    payload = json.dumps(spec.to_dict(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _resolution_for(config: EyeGenConfig):
    """The name of the configured model, and the specification behind it.

    A model the user described in their own catalogue is run the way they
    described it; a model that is already on this machine is run the way
    what is there says; a model that is only a name is run the way its
    backend knows it.
    """
    from eyegen.backends import resolve_spec

    return resolve_spec(config.model)


def load_pipeline_for_worker(worker):
    """The pipeline for the worker's model, and the generation it came with."""
    backend = worker.backend
    resolution = _resolution_for(worker.config)
    name, spec = resolution.name, resolution.spec
    config = apply_spec(worker.config, spec)
    cache_key = _make_cache_key(backend, config, worker.mflux_quantize, worker.use_t5, spec=spec)
    cached_pipeline, cached_key, cached_gen = _pipeline_cache.get_cached()
    if cached_pipeline is not None and cached_key == cache_key:
        log.info("Using cached pipeline (backend=%s)", backend)
        if backend == Backend.MLX:
            _patch_sample_euler(
                lambda step, total: worker.progress.emit(step, total), worker._cancelled
            )
        return cached_pipeline, cached_gen

    if worker._cancelled.is_set():
        return None, None

    worker.status.emit("Loading model…")
    log.info(
        "Loading pipeline (model=%s, backend=%s, t5=%s)",
        config.model or "default",
        backend,
        worker.use_t5,
    )
    if backend == Backend.OLLAMA:
        pipeline = get_ollama_pipeline(config, name=name, spec=spec)
    elif backend == Backend.MFLUX:
        pipeline = get_mflux_pipeline(config, quantize=worker.mflux_quantize)
    elif backend == Backend.BONSAI:
        pipeline = bonsai.get_bonsai_pipeline(config)
    elif backend == Backend.COREML:
        pipeline = coreml.get_coreml_pipeline(config, spec=spec)
    elif backend == Backend.SDCLI:
        pipeline = sdcli.get_sdcli_pipeline(config, name, spec=spec)
    else:
        pipeline = get_pipeline(config, use_t5=worker.use_t5)
    _patch_sample_euler(lambda step, total: worker.progress.emit(step, total), worker._cancelled)

    if not worker._cancelled.is_set():
        _pipeline_cache.set_cached(pipeline, cache_key)
    cached_gen = get_cache_generation()
    return pipeline, cached_gen
