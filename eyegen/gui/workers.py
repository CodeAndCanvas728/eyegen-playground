"""Background worker threads for the EyeGen GUI."""

import logging
import subprocess
import threading
import traceback
from datetime import datetime, timezone

from PySide6.QtCore import QThread, Signal

from eyegen import (
    OUTPUT_DIR,
    Backend,
    generate_image,
    get_mflux_pipeline,
    get_ollama_pipeline,
    get_pipeline,
    pull_model,
    save_mflux_model,
)
from eyegen.backends import bonsai, coreml
from eyegen.gui.cache import _clear_pipeline_cache, _pipeline_cache, _pipeline_cache_lock
from eyegen.gui.monkeypatch import GenerationCancelled, _patch_sample_euler

log = logging.getLogger("eyegen")


def _make_cache_key(backend, config, mflux_quantize, use_t5):
    return (
        backend,
        config.get("model"),
        mflux_quantize,
        use_t5,
        config.get("mflux_model_path"),
        config.get("bonsai_model_path"),
        config.get("coreml_model_path"),
        config.get("coreml_compute_unit"),
    )


def _load_pipeline_for_worker(worker):
    backend = worker.backend
    config = worker.config
    cache_key = _make_cache_key(backend, config, worker.mflux_quantize, worker.use_t5)
    with _pipeline_cache_lock:
        cached_pipeline = _pipeline_cache["pipeline"]
        cached_key = _pipeline_cache["key"]
    if cached_pipeline is not None and cached_key == cache_key:
        log.info("Using cached pipeline (backend=%s)", backend)
        if backend == Backend.MLX:
            _patch_sample_euler(
                lambda step, total: worker.progress.emit(step, total), worker._cancelled
            )
        return cached_pipeline

    if worker._cancelled.is_set():
        return None

    worker.status.emit("Loading model…")
    log.info(
        "Loading pipeline (model=%s, backend=%s, t5=%s)",
        config.get("model", "default"),
        backend,
        worker.use_t5,
    )
    if backend == Backend.OLLAMA:
        pipeline = get_ollama_pipeline(config)
    elif backend == Backend.MFLUX:
        pipeline = get_mflux_pipeline(config, quantize=worker.mflux_quantize)
    elif backend == Backend.BONSAI:
        pipeline = bonsai.get_bonsai_pipeline(config)
    elif backend == Backend.COREML:
        pipeline = coreml.get_coreml_pipeline(config)
    else:
        pipeline = get_pipeline(config, use_t5=worker.use_t5)
        _patch_sample_euler(
            lambda step, total: worker.progress.emit(step, total), worker._cancelled
        )

    with _pipeline_cache_lock:
        if not worker._cancelled.is_set():
            _pipeline_cache["pipeline"] = pipeline
            _pipeline_cache["key"] = cache_key
    return pipeline


class GenerationWorker(QThread):
    finished = Signal(object, str)
    error = Signal(str)
    quantize_failed = Signal(str)
    status = Signal(str)
    progress = Signal(int, int)
    cancelled = Signal(bool)

    def __init__(
        self,
        prompt,
        negative_prompt,
        cfg_weight,
        num_steps,
        width,
        height,
        seed,
        config,
        use_t5,
        image_path=None,
        denoise=1.0,
        backend=Backend.MLX,
        mflux_quantize=4,
    ):
        super().__init__()
        self._cancelled = threading.Event()
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.cfg_weight = cfg_weight
        self.num_steps = num_steps
        self.width = width
        self.height = height
        self.seed = seed
        self.config = config
        self.use_t5 = use_t5
        self.image_path = image_path
        self.denoise = denoise
        self.backend = backend
        self.mflux_quantize = mflux_quantize
        self.pipeline = None

    def cancel(self) -> None:
        self._cancelled.set()
        pipeline = self.pipeline
        if pipeline is not None:
            cancel_fn = getattr(pipeline, "cancel", None)
            if callable(cancel_fn):
                try:
                    cancel_fn()
                except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
                    log.warning("pipeline.cancel() raised: %s", exc)

    def _handle_cancel(self):
        _clear_pipeline_cache()
        self.pipeline = None
        self.cancelled.emit(False)

    def run(self):
        try:
            if self._cancelled.is_set():
                self._handle_cancel()
                return

            pipeline = _load_pipeline_for_worker(self)
            self.pipeline = pipeline

            if self._cancelled.is_set() or pipeline is None:
                self._handle_cancel()
                return

            self.status.emit("Generating…")
            log.info(
                "Generating: steps=%d guidance=%.1f size=%dx%d seed=%s backend=%s",
                self.num_steps,
                self.cfg_weight,
                self.width,
                self.height,
                self.seed,
                self.backend,
            )
            image = generate_image(
                pipeline,
                self.prompt,
                self.cfg_weight,
                self.num_steps,
                self.width,
                self.height,
                self.seed,
                negative_prompt=self.negative_prompt,
                image_path=self.image_path,
                denoise=self.denoise,
                backend=self.backend,
            )

            if self._cancelled.is_set():
                self._handle_cancel()
                return

            self.status.emit("Saving…")
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = OUTPUT_DIR / f"{timestamp}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(out_path)
            log.info("Saved: %s", out_path)

            self.finished.emit(image, str(out_path))
        except GenerationCancelled:
            log.info("Generation cancelled by user")
            self._handle_cancel()
        except (OSError, RuntimeError) as exc:
            if self._cancelled.is_set():
                log.info("Generation cancelled by user: %s", exc)
                self._handle_cancel()
            else:
                full = traceback.format_exc()
                log.error("Generation failed:\n%s", full)
                self.error.emit(full)
        except (ValueError, TypeError):
            full = traceback.format_exc()
            log.error("Generation failed:\n%s", full)
            self.error.emit(full)
        finally:
            self.pipeline = None


class PullWorker(QThread):
    finished = Signal(bool, str)
    status = Signal(str)

    def __init__(self, model_name: str, hf_cache_dir: str | None = None):
        super().__init__()
        self.model_name = model_name
        self.hf_cache_dir = hf_cache_dir

    def run(self):
        try:
            self.status.emit(f"Pulling {self.model_name}…")
            log.info("Pulling model: %s", self.model_name)

            def on_progress(msg):
                if isinstance(msg, str):
                    self.status.emit(msg)

            ok = pull_model(
                self.model_name, progress_callback=on_progress, hf_cache_dir=self.hf_cache_dir
            )
            if ok:
                log.info("Pull complete: %s", self.model_name)
                self.finished.emit(True, f"Model '{self.model_name}' is ready")
            else:
                self.finished.emit(False, f"Failed to pull '{self.model_name}'")
        except (OSError, ValueError, RuntimeError) as exc:
            full = traceback.format_exc()
            log.error("Pull failed:\n%s", full)
            self.finished.emit(False, str(exc))


class SaveModelWorker(QThread):
    finished = Signal(bool, str, str)
    status = Signal(str)

    def __init__(
        self,
        model_alias: str,
        quantize: int | None,
        output_path: str,
        hf_cache_dir: str | None = None,
    ):
        super().__init__()
        self.model_alias = model_alias
        self.quantize = quantize
        self.output_path = output_path
        self.hf_cache_dir = hf_cache_dir

    def run(self):
        try:
            result = save_mflux_model(
                model_alias=self.model_alias,
                quantize=self.quantize,
                output_path=self.output_path,
                progress_callback=lambda msg: self.status.emit(msg),
                hf_cache_dir=self.hf_cache_dir,
            )
            self.finished.emit(True, "Model saved successfully", str(result))
        except (OSError, ValueError, RuntimeError) as exc:
            full = traceback.format_exc()
            log.error("Save model failed:\n%s", full)
            self.finished.emit(False, str(exc), "")
