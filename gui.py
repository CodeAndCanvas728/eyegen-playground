#!/usr/bin/env python3
"""
EyeGen — PySide6 GUI
Native macOS desktop interface for image generation on Apple Silicon.
"""

import io
import json
import logging
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import core_bonsai
import core_coreml
from core import (
    BACKEND_AUTO,
    BACKEND_BONSAI,
    BACKEND_COREML,
    BACKEND_MFLUX,
    BACKEND_MLX,
    BACKEND_OLLAMA,
    CONFIG_DIR,
    DEFAULT_CONFIG,
    MODELS_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    EyeGenConfig,
    clear_mflux_cache,
    detect_backend,
    generate_image,
    get_mflux_pipeline,
    get_ollama_pipeline,
    get_pipeline,
    list_mflux_models,
    load_config,
    pull_model,
    save_mflux_model,
    validate_dimensions,
    validate_image_path,
    validate_saved_model,
)
from gui_hf_login import HFLoginDialog, _cached_hf_status

# ---------------------------------------------------------------------------
# Logging — always write to ~/Library/Logs/EyeGen.log so errors are
# visible even when the GUI status label truncates them.
# ---------------------------------------------------------------------------

LOG_FILE = CONFIG_DIR / "eyegen.log"

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
log = logging.getLogger("eyegen")


# ---------------------------------------------------------------------------
# GUI state persistence
# ---------------------------------------------------------------------------

GUI_STATE_FILE = CONFIG_DIR / "gui_state.json"


def load_gui_state() -> dict:
    if GUI_STATE_FILE.exists():
        try:
            with open(GUI_STATE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_gui_state(state: dict):
    GUI_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GUI_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Monkeypatch: inject step-level progress callback into sample_euler
# ---------------------------------------------------------------------------

_sample_euler_patched = False
# Mutable holder so re-patching can update callbacks without replacing the function
_sample_euler_state = {"progress_callback": None, "cancel_event": None}


def _patch_sample_euler(progress_callback, cancel_event=None):
    """Replace diffusionkit's sample_euler with a step-reporting wrapper.
    On subsequent calls, only the callback/cancel_event references are updated
    without replacing the function again."""
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
            t0 = time.time()
            denoised = model(x, timesteps[i], sigmas[i], **extra_args)
            d = dkmlx.to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
            mx.eval(x)
            iter_time.append(round(time.time() - t0, 3))
            cb(i + 1, total)

        model.clear_cache()
        return x, iter_time

    dkmlx.sample_euler = _patched_sample_euler
    _sample_euler_patched = True


class GenerationCancelled(Exception):
    """Raised when the user cancels a running generation."""


# ---------------------------------------------------------------------------
# Pipeline cache — avoid reloading the model on every generation
# ---------------------------------------------------------------------------

_pipeline_cache: dict = {"pipeline": None, "key": None}
_pipeline_cache_lock = threading.Lock()


def _clear_pipeline_cache():
    with _pipeline_cache_lock:
        _pipeline_cache["pipeline"] = None
        _pipeline_cache["key"] = None


# ---------------------------------------------------------------------------
# Worker thread — runs image generation off the main/UI thread
# ---------------------------------------------------------------------------


class GenerationWorker(QThread):
    finished = Signal(object, str)  # (PIL.Image, output_path)
    error = Signal(str)
    quantize_failed = Signal(str)  # model_name — dequantize error, offer retry
    status = Signal(str)
    progress = Signal(int, int)  # (current_step, total_steps)
    cancelled = Signal()

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
        backend=BACKEND_MLX,
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
        """Request cancellation of the running generation.

        Sets the cooperative cancellation event (checked at step boundaries)
        and, for backends that wrap a subprocess (bonsai/coreml), terminates
        the child so the user does not have to wait minutes for a doomed
        generation to finish. No-op for backends whose pipeline does not
        expose ``cancel()``.
        """
        self._cancelled.set()
        pipeline = self.pipeline
        if pipeline is not None:
            cancel_fn = getattr(pipeline, "cancel", None)
            if callable(cancel_fn):
                try:
                    cancel_fn()
                except Exception as exc:
                    log.warning("pipeline.cancel() raised: %s", exc)

    def run(self):
        try:
            # Check pipeline cache (includes bonsai/coreml config
            # fields so model_path or compute_unit changes invalidate the cache)
            cache_key = (
                self.backend,
                self.config.get("model"),
                self.mflux_quantize,
                self.use_t5,
                self.config.get("mflux_model_path"),
                self.config.get("bonsai_model_path"),
                self.config.get("coreml_model_path"),
                self.config.get("coreml_compute_unit"),
            )
            with _pipeline_cache_lock:
                cached_pipeline = _pipeline_cache["pipeline"]
                cached_key = _pipeline_cache["key"]
            if cached_pipeline is not None and cached_key == cache_key:
                pipeline = cached_pipeline
                log.info("Using cached pipeline (backend=%s)", self.backend)
                if self.backend == BACKEND_MLX:
                    _patch_sample_euler(
                        lambda step, total: self.progress.emit(step, total), self._cancelled
                    )
            else:
                self.status.emit("Loading model…")
                log.info(
                    "Loading pipeline (model=%s, backend=%s, t5=%s)",
                    self.config.get("model", "default"),
                    self.backend,
                    self.use_t5,
                )
                if self.backend == BACKEND_OLLAMA:
                    pipeline = get_ollama_pipeline(self.config)
                elif self.backend == BACKEND_MFLUX:
                    pipeline = get_mflux_pipeline(self.config, quantize=self.mflux_quantize)
                elif self.backend == BACKEND_BONSAI:
                    import core_bonsai

                    pipeline = core_bonsai.get_bonsai_pipeline(self.config)
                elif self.backend == BACKEND_COREML:
                    import core_coreml

                    pipeline = core_coreml.get_coreml_pipeline(self.config)
                else:
                    pipeline = get_pipeline(self.config, use_t5=self.use_t5)
                    _patch_sample_euler(
                        lambda step, total: self.progress.emit(step, total), self._cancelled
                    )
                with _pipeline_cache_lock:
                    _pipeline_cache["pipeline"] = pipeline
                    _pipeline_cache["key"] = cache_key
            self.pipeline = pipeline

            if self._cancelled.is_set():
                self.cancelled.emit()
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
                self.cancelled.emit()
                return

            self.status.emit("Saving…")
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = OUTPUT_DIR / f"{timestamp}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(out_path)
            log.info("Saved: %s", out_path)

            self.finished.emit(image, str(out_path))
        except Exception:
            full = traceback.format_exc()
            log.error("Generation failed:\n%s", full)
            self.error.emit(full)


class PullWorker(QThread):
    """Downloads a GGUF model via ollamadiffuser in a background thread."""

    finished = Signal(bool, str)  # (success, message)
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
        except Exception as exc:
            full = traceback.format_exc()
            log.error("Pull failed:\n%s", full)
            self.finished.emit(False, str(exc))


class SaveModelWorker(QThread):
    """Downloads, quantizes, and saves an MFLUX model in a background thread."""

    finished = Signal(bool, str, str)  # (success, message, saved_path)
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
        except Exception as exc:
            full = traceback.format_exc()
            log.error("Save model failed:\n%s", full)
            self.finished.emit(False, str(exc), "")


class BonsaiSetupWorker(QThread):
    """Runs scripts/setup-bonsai.sh in a subprocess."""

    finished = Signal(bool, str)  # (success, message)

    def __init__(self, script_path: str):
        super().__init__()
        self.script_path = script_path

    def run(self):
        try:
            r = subprocess.run([self.script_path], capture_output=True, text=True)  # noqa: S603
            if r.returncode == 0:
                self.finished.emit(True, "Bonsai installed successfully")
            else:
                msg = (r.stderr or r.stdout or "Unknown error").strip().splitlines()[-5:]
                self.finished.emit(False, f"Setup failed (exit {r.returncode}): " + "\n".join(msg))
        except Exception as exc:
            self.finished.emit(False, f"Setup error: {exc}")


class BonsaiDownloadWorker(QThread):
    """Downloads a bonsai model via core_bonsai.download_bonsai_model()."""

    finished = Signal(bool, str)
    status = Signal(str)

    def __init__(self, variant: str):
        super().__init__()
        self.variant = variant

    def run(self):
        try:
            ok = core_bonsai.download_bonsai_model(
                self.variant,
                progress_callback=lambda m: self.status.emit(m),
            )
            if ok:
                self.finished.emit(True, f"Bonsai model '{self.variant}' is ready")
            else:
                self.finished.emit(False, f"Failed to download bonsai variant '{self.variant}'")
        except Exception as exc:
            self.finished.emit(False, f"Download error: {exc}")


class CoreMLSetupWorker(QThread):
    """Runs scripts/setup-coreml.sh in a subprocess."""

    finished = Signal(bool, str)

    def __init__(self, script_path: str):
        super().__init__()
        self.script_path = script_path

    def run(self):
        try:
            r = subprocess.run([self.script_path], capture_output=True, text=True)  # noqa: S603
            if r.returncode == 0:
                self.finished.emit(True, "CoreML sidecar venv installed")
            else:
                msg = (r.stderr or r.stdout or "Unknown error").strip().splitlines()[-5:]
                self.finished.emit(False, f"Setup failed (exit {r.returncode}): " + "\n".join(msg))
        except Exception as exc:
            self.finished.emit(False, f"Setup error: {exc}")


class CoreMLDownloadWorker(QThread):
    """Downloads a pre-converted CoreML model from Hugging Face."""

    finished = Signal(bool, str)
    status = Signal(str)

    def __init__(self, alias: str):
        super().__init__()
        self.alias = alias

    def run(self):
        try:
            target = core_coreml.pull_preconverted_coreml_model(
                self.alias,
                progress_callback=lambda m: self.status.emit(m),
            )
            if target:
                self.finished.emit(True, f"CoreML model downloaded to {target}")
            else:
                self.finished.emit(False, f"Failed to download CoreML model '{self.alias}'")
        except Exception as exc:
            self.finished.emit(False, f"Download error: {exc}")


# ---------------------------------------------------------------------------
# Helper: PIL Image → QPixmap
# ---------------------------------------------------------------------------


def pil_to_pixmap(pil_image) -> QPixmap:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    qimg = QImage()
    qimg.loadFromData(buf.read())
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# Dimension presets (multiples of 8)
# ---------------------------------------------------------------------------

DIMENSION_PRESETS = [512, 640, 768, 896, 1024]


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EyeGen")
        self.setMinimumSize(950, 650)
        self.resize(1200, 800)

        self.worker: Optional[GenerationWorker] = None
        self.pull_worker: Optional[PullWorker] = None
        self._save_worker: Optional[SaveModelWorker] = None
        self.config = load_config()
        self._gui_state = load_gui_state()
        # Saved steps/guidance before MFLUX auto-set them
        self._pre_mflux_steps: Optional[int] = None
        self._pre_mflux_guidance: Optional[float] = None
        self._restoring_state = False
        self._elapsed_seconds = 0
        self._current_phase = ""
        self._generation_id = 0
        self._autoclear_generation_id = 0
        self._status_clear_timer = QTimer(self)
        self._status_clear_timer.setSingleShot(True)
        self._status_clear_timer.setInterval(5000)
        self._status_clear_timer.timeout.connect(self._clear_status)

        self._build_ui()

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._on_elapsed_tick)

        self._restore_state()

    # ----- UI construction ------------------------------------------------

    def _build_img2img_controls(self) -> QWidget:
        # img2img controls (hidden by default)
        widget = QWidget()
        img2img_layout = QVBoxLayout(widget)
        img2img_layout.setContentsMargins(0, 0, 0, 0)
        img2img_layout.setSpacing(8)

        img2img_layout.addWidget(QLabel("Input Image"))
        image_row = QHBoxLayout()
        self.image_path_input = QLineEdit()
        self.image_path_input.setReadOnly(True)
        self.image_path_input.setPlaceholderText("No image selected…")
        image_row.addWidget(self.image_path_input)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._on_browse_image)
        image_row.addWidget(browse_btn)
        img2img_layout.addLayout(image_row)

        self.image_thumbnail = QLabel()
        self.image_thumbnail.setAlignment(Qt.AlignCenter)
        self.image_thumbnail.setMaximumHeight(80)
        self.image_thumbnail.hide()
        img2img_layout.addWidget(self.image_thumbnail)

        # Known-issue warning for quantized models
        self.img2img_warning = QLabel(
            "⚠ Known issue: 4-bit quantized models may produce output "
            "identical to the input (denoise has no effect)."
        )
        self.img2img_warning.setWordWrap(True)
        self.img2img_warning.setProperty("class", "hint")
        self.img2img_warning.setStyleSheet("color: #cc8800;")
        img2img_layout.addWidget(self.img2img_warning)

        denoise_row = QHBoxLayout()
        denoise_row.addWidget(QLabel("Denoise"))
        self.denoise_spin = QDoubleSpinBox()
        self.denoise_spin.setRange(0.05, 1.0)
        self.denoise_spin.setSingleStep(0.05)
        self.denoise_spin.setValue(0.75)
        denoise_row.addWidget(self.denoise_spin)
        img2img_layout.addLayout(denoise_row)

        self.denoise_slider = QSlider(Qt.Horizontal)
        self.denoise_slider.setRange(5, 100)  # ×100 for int slider
        self.denoise_slider.setValue(75)
        self.denoise_slider.valueChanged.connect(lambda v: self.denoise_spin.setValue(v / 100.0))
        self.denoise_spin.valueChanged.connect(
            lambda v: self.denoise_slider.setValue(int(round(v * 100)))
        )
        img2img_layout.addWidget(self.denoise_slider)

        widget.hide()
        return widget

    def _build_settings_panel(self) -> QGroupBox:
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(8)

        # Steps
        row = QHBoxLayout()
        row.addWidget(QLabel("Steps"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 100)
        self.steps_spin.setValue(self.config.get("num_inference_steps", 30))
        row.addWidget(self.steps_spin)
        settings_layout.addLayout(row)

        self.steps_slider = QSlider(Qt.Horizontal)
        self.steps_slider.setRange(1, 100)
        self.steps_slider.setValue(self.steps_spin.value())
        self.steps_slider.valueChanged.connect(self.steps_spin.setValue)
        self.steps_spin.valueChanged.connect(self.steps_slider.setValue)
        settings_layout.addWidget(self.steps_slider)

        # Guidance scale
        row = QHBoxLayout()
        row.addWidget(QLabel("Guidance"))
        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(1.0, 15.0)
        self.guidance_spin.setSingleStep(0.5)
        self.guidance_spin.setValue(self.config.get("guidance_scale", 7.5))
        row.addWidget(self.guidance_spin)
        settings_layout.addLayout(row)

        self.guidance_slider = QSlider(Qt.Horizontal)
        self.guidance_slider.setRange(10, 150)  # ×10 for int slider
        self.guidance_slider.setValue(int(self.guidance_spin.value() * 10))
        self.guidance_slider.valueChanged.connect(lambda v: self.guidance_spin.setValue(v / 10.0))
        self.guidance_spin.valueChanged.connect(
            lambda v: self.guidance_slider.setValue(int(v * 10))
        )
        settings_layout.addWidget(self.guidance_slider)

        # Width
        row = QHBoxLayout()
        row.addWidget(QLabel("Width"))
        self.width_combo = QComboBox()
        for d in DIMENSION_PRESETS:
            self.width_combo.addItem(str(d), d)
        self.width_combo.setCurrentText(str(self.config.get("width", 1024)))
        row.addWidget(self.width_combo)
        settings_layout.addLayout(row)

        # Height
        row = QHBoxLayout()
        row.addWidget(QLabel("Height"))
        self.height_combo = QComboBox()
        for d in DIMENSION_PRESETS:
            self.height_combo.addItem(str(d), d)
        self.height_combo.setCurrentText(str(self.config.get("height", 1024)))
        row.addWidget(self.height_combo)
        settings_layout.addLayout(row)

        # Seed
        row = QHBoxLayout()
        row.addWidget(QLabel("Seed"))
        self.seed_input = QLineEdit()
        self.seed_input.setPlaceholderText("Random")
        row.addWidget(self.seed_input)
        settings_layout.addLayout(row)

        # T5 toggle
        self.t5_check = QCheckBox("Use T5 encoder (better quality, slower)")
        self.t5_check.setChecked(True)
        settings_layout.addWidget(self.t5_check)

        # Model
        settings_layout.addWidget(QLabel("Model"))
        model_row = QHBoxLayout()
        self.model_input = QLineEdit()
        self.model_input.setText(self.config.get("model", DEFAULT_CONFIG["model"]))
        self.model_input.setToolTip("Hugging Face model ID or OllamaDiffuser model name")
        self.model_input.editingFinished.connect(lambda: self._update_backend_dependent_controls())
        model_row.addWidget(self.model_input)
        self.pull_btn = QPushButton("Pull…")
        self.pull_btn.setFixedWidth(50)
        self.pull_btn.setToolTip("Download this GGUF model via OllamaDiffuser")
        self.pull_btn.clicked.connect(self._on_pull_model)
        model_row.addWidget(self.pull_btn)
        settings_layout.addLayout(model_row)

        # Backend
        row = QHBoxLayout()
        row.addWidget(QLabel("Backend"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Auto", BACKEND_AUTO)
        self.backend_combo.addItem("MLX (diffusionkit)", BACKEND_MLX)
        self.backend_combo.addItem("MFLUX (FLUX/FIBO/Z-Image)", BACKEND_MFLUX)
        self.backend_combo.addItem("OllamaDiffuser (GGUF)", BACKEND_OLLAMA)
        self.backend_combo.addItem("Bonsai (PrismML ternary 1.58-bit)", BACKEND_BONSAI)
        self.backend_combo.addItem("CoreML (Apple Neural Engine)", BACKEND_COREML)
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        row.addWidget(self.backend_combo)
        settings_layout.addLayout(row)

        # MFLUX quantization
        self.quantize_row = QWidget()
        quantize_layout = QHBoxLayout(self.quantize_row)
        quantize_layout.setContentsMargins(0, 0, 0, 0)
        quantize_layout.addWidget(QLabel("Quantize"))
        self.quantize_combo = QComboBox()
        self.quantize_combo.addItem("4-bit (recommended)", 4)
        self.quantize_combo.addItem("8-bit", 8)
        self.quantize_combo.addItem("None (full precision)", 0)
        self.quantize_combo.setToolTip("MFLUX runtime quantization level")
        quantize_layout.addWidget(self.quantize_combo)
        self.quantize_row.hide()
        settings_layout.addWidget(self.quantize_row)

        # MFLUX local model path
        self.model_path_row = QWidget()
        mp_layout = QVBoxLayout(self.model_path_row)
        mp_layout.setContentsMargins(0, 0, 0, 0)
        mp_top = QHBoxLayout()
        mp_top.addWidget(QLabel("Saved model"))
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("None (downloads from HuggingFace)")
        self.model_path_input.setToolTip(
            "Path to a pre-quantized model directory saved with Save Model.\n"
            "Leave blank to download from HuggingFace on each first run."
        )
        self.model_path_input.editingFinished.connect(self._on_model_path_changed)
        mp_top.addWidget(self.model_path_input)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._on_browse_model_path)
        mp_top.addWidget(browse_btn)
        mp_layout.addLayout(mp_top)

        # Status label + Save Model button on second row
        mp_bottom = QHBoxLayout()
        self.model_path_status = QLabel()
        self.model_path_status.setProperty("class", "hint")
        self.model_path_status.setWordWrap(True)
        mp_bottom.addWidget(self.model_path_status, 1)
        self.save_model_btn = QPushButton("Save Model…")
        self.save_model_btn.setFixedWidth(100)
        self.save_model_btn.setToolTip(
            "Download and save a pre-quantized MFLUX model to disk.\n"
            "One-time operation — subsequent loads are instant."
        )
        self.save_model_btn.clicked.connect(self._on_save_model)
        mp_bottom.addWidget(self.save_model_btn)
        mp_layout.addLayout(mp_bottom)
        self.model_path_row.hide()
        settings_layout.addWidget(self.model_path_row)

        # Backend hint label (shown for MFLUX with recommended settings)
        self.backend_hint = QLabel()
        self.backend_hint.setWordWrap(True)
        self.backend_hint.setProperty("class", "hint")
        self.backend_hint.hide()
        settings_layout.addWidget(self.backend_hint)

        # Bonsai setup row (visible only when bonsai is the resolved backend)
        self.bonsai_row = QWidget()
        bonsai_layout = QHBoxLayout(self.bonsai_row)
        bonsai_layout.setContentsMargins(0, 0, 0, 0)
        self.bonsai_status_label = QLabel()
        self.bonsai_status_label.setStyleSheet("color: #888; font-size: 11px;")
        self.bonsai_status_label.setWordWrap(True)
        bonsai_layout.addWidget(self.bonsai_status_label, 1)
        self.bonsai_setup_btn = QPushButton("Setup Bonsai…")
        self.bonsai_setup_btn.setFixedWidth(130)
        self.bonsai_setup_btn.setToolTip(
            "One-time install: clones the Bonsai-Image-Demo repo and runs its\n"
            "setup.sh, which creates a dedicated Python 3.11 venv with the\n"
            "patched mflux + MLX kernels needed for ternary 1.58-bit weights."
        )
        self.bonsai_setup_btn.clicked.connect(self._on_bonsai_setup)
        bonsai_layout.addWidget(self.bonsai_setup_btn)
        self.bonsai_pull_btn = QPushButton("Download Model…")
        self.bonsai_pull_btn.setFixedWidth(140)
        self.bonsai_pull_btn.setToolTip(
            "Download a bonsai model via the bonsai-demo's download script."
        )
        self.bonsai_pull_btn.clicked.connect(self._on_bonsai_pull)
        bonsai_layout.addWidget(self.bonsai_pull_btn)
        self.bonsai_row.hide()
        settings_layout.addWidget(self.bonsai_row)

        # CoreML setup row (visible only when coreml is the resolved backend)
        self.coreml_row = QWidget()
        coreml_layout = QHBoxLayout(self.coreml_row)
        coreml_layout.setContentsMargins(0, 0, 0, 0)
        self.coreml_status_label = QLabel()
        self.coreml_status_label.setStyleSheet("color: #888; font-size: 11px;")
        self.coreml_status_label.setWordWrap(True)
        coreml_layout.addWidget(self.coreml_status_label, 1)
        self.coreml_setup_btn = QPushButton("Setup CoreML…")
        self.coreml_setup_btn.setFixedWidth(130)
        self.coreml_setup_btn.setToolTip(
            "One-time install: creates a sidecar Python 3.11 venv at\n"
            "~/models/eyegen/.coreml-venv/ with Apple's python_coreml_stable_diffusion."
        )
        self.coreml_setup_btn.clicked.connect(self._on_coreml_setup)
        coreml_layout.addWidget(self.coreml_setup_btn)
        self.coreml_pull_btn = QPushButton("Download Model…")
        self.coreml_pull_btn.setFixedWidth(140)
        self.coreml_pull_btn.setToolTip(
            "Download a pre-converted CoreML model from Hugging Face.\n"
            "Or use ./generate.py convert-coreml to convert a PyTorch model from scratch."
        )
        self.coreml_pull_btn.clicked.connect(self._on_coreml_pull)
        coreml_layout.addWidget(self.coreml_pull_btn)
        self.coreml_row.hide()
        settings_layout.addWidget(self.coreml_row)

        # HF Cache Directory (all backends)
        settings_layout.addWidget(QLabel("HF Cache Dir"))
        hf_cache_row = QHBoxLayout()
        self.hf_cache_input = QLineEdit()
        self.hf_cache_input.setPlaceholderText("Default (~/.cache/huggingface/hub)")
        self.hf_cache_input.setToolTip(
            "Directory where HuggingFace caches downloaded model weights.\n"
            "Leave blank to use the default (~/.cache/huggingface/hub)."
        )
        hf_cache_row.addWidget(self.hf_cache_input)
        hf_cache_browse = QPushButton("Browse…")
        hf_cache_browse.setFixedWidth(70)
        hf_cache_browse.clicked.connect(self._on_browse_hf_cache)
        hf_cache_row.addWidget(hf_cache_browse)
        settings_layout.addLayout(hf_cache_row)

        return settings_group

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # --- Left: controls ---
        controls = QWidget()
        controls.setMaximumWidth(420)
        controls.setMinimumWidth(300)
        ctrl_layout = QVBoxLayout(controls)
        ctrl_layout.setContentsMargins(8, 8, 8, 8)
        ctrl_layout.setSpacing(8)

        # Mode tab bar
        self.mode_tabs = QTabBar()
        self.mode_tabs.addTab("Text to Image")
        self.mode_tabs.addTab("Image to Image")
        self.mode_tabs.currentChanged.connect(self._on_mode_changed)
        ctrl_layout.addWidget(self.mode_tabs)

        # img2img controls (hidden by default)
        self.img2img_controls = self._build_img2img_controls()
        ctrl_layout.addWidget(self.img2img_controls)

        # Prompt
        ctrl_layout.addWidget(QLabel("Prompt"))
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Describe the image you want to generate…")
        self.prompt_input.setMaximumHeight(120)
        ctrl_layout.addWidget(self.prompt_input)

        # Negative prompt
        ctrl_layout.addWidget(QLabel("Negative Prompt"))
        self.negative_prompt_input = QTextEdit()
        self.negative_prompt_input.setPlaceholderText("What to avoid (optional)…")
        self.negative_prompt_input.setMaximumHeight(60)
        ctrl_layout.addWidget(self.negative_prompt_input)

        # Generate button
        self.generate_btn = QPushButton("✨  Generate")
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.setToolTip("Generate (⌘↩ / Ctrl+↩)")
        self.generate_btn.clicked.connect(self._on_generate)
        self.generate_shortcut = QShortcut(
            QKeySequence("Ctrl+Return"), self.prompt_input, self._on_generate
        )
        self.generate_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.generate_shortcut_neg = QShortcut(
            QKeySequence("Ctrl+Return"), self.negative_prompt_input, self._on_generate
        )
        self.generate_shortcut_neg.setContext(Qt.WidgetWithChildrenShortcut)
        ctrl_layout.addWidget(self.generate_btn)

        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray;")
        ctrl_layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(8)
        self.progress_bar.hide()
        ctrl_layout.addWidget(self.progress_bar)

        # Settings
        settings_group = self._build_settings_panel()
        ctrl_layout.addWidget(settings_group)

        # HuggingFace login button
        self.hf_btn = QPushButton("🔑  HuggingFace Login")
        self.hf_btn.setToolTip("Log in to download gated models (e.g. FLUX.1-Kontext)")
        self.hf_btn.clicked.connect(self._on_hf_login)
        ctrl_layout.addWidget(self.hf_btn)
        self._refresh_hf_button()

        ctrl_layout.addStretch()

        # Output path label
        self.output_label = QLabel("")
        self.output_label.setWordWrap(True)
        self.output_label.setProperty("class", "hint")
        self.output_label.setStyleSheet("color: gray;")
        ctrl_layout.addWidget(self.output_label)

        splitter.addWidget(controls)

        # --- Right: image preview ---
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel("No image yet")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setProperty("class", "display")
        self.image_label.setStyleSheet(
            "background-color: #1e1e1e; color: #888; border-radius: 8px;"
        )
        preview_layout.addWidget(self.image_label)

        splitter.addWidget(preview)
        splitter.setStretchFactor(0, 0)  # controls don't stretch
        splitter.setStretchFactor(1, 1)  # preview fills space

        self._current_pixmap: Optional[QPixmap] = None

    # ----- Event handling -------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_preview()

    def closeEvent(self, event):
        self._elapsed_timer.stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.terminate()
            self.worker.wait(2000)
        save_gui_state(self._collect_state())
        log.info("GUI state saved")
        super().closeEvent(event)

    def _on_mode_changed(self, index: int):
        is_img2img = index == 1
        self.img2img_controls.setVisible(is_img2img)
        self.width_combo.setEnabled(not is_img2img)
        self.height_combo.setEnabled(not is_img2img)
        self._update_backend_dependent_controls()

    def _on_backend_changed(self, _index: int):
        self._update_backend_dependent_controls()

    def _resolved_backend(self) -> str:
        """Return the concrete backend for the current model + dropdown."""
        override = self.backend_combo.currentData()
        model = self.model_input.text().strip() or DEFAULT_CONFIG["model"]
        config = {"model": model}
        # Only inject coreml_model_path when the user has explicitly chosen CoreML;
        # otherwise arbitrary text in the Model field would short-circuit to coreml
        # in detect_backend (see _is_coreml_model).
        if override == BACKEND_COREML:
            config["coreml_model_path"] = model
        return detect_backend(model, override, config=config)

    def _update_backend_dependent_controls(self):
        """Show/hide/enable controls based on the resolved backend."""
        backend = self._resolved_backend()
        is_mlx = backend == BACKEND_MLX
        is_mflux = backend == BACKEND_MFLUX
        is_ollama = backend == BACKEND_OLLAMA
        is_bonsai = backend == BACKEND_BONSAI
        is_coreml = backend == BACKEND_COREML

        # T5 only applies to MLX
        self.t5_check.setEnabled(is_mlx)
        if is_ollama:
            self.t5_check.setToolTip("T5 is not applicable for GGUF models")
        elif is_mflux:
            self.t5_check.setToolTip("T5 is not applicable for MFLUX models")
        elif is_bonsai:
            self.t5_check.setToolTip("T5 is not applicable for Bonsai (uses Qwen3-4B internally)")
        elif is_coreml:
            self.t5_check.setToolTip("T5 is not applicable for CoreML SD 1.x/2.x models")
        else:
            self.t5_check.setToolTip("")

        # Img2img quantization warning only for MLX
        is_img2img = self.mode_tabs.currentIndex() == 1
        self.img2img_warning.setVisible(is_img2img and is_mlx)

        # Bonsai + CoreML don't support img2img — switch back to txt2img if needed
        if (is_bonsai or is_coreml) and is_img2img:
            self.mode_tabs.setCurrentIndex(0)
            is_img2img = False

        # Quantize dropdown only for MFLUX
        self.quantize_row.setVisible(is_mflux)

        # Model path row only for MFLUX
        self.model_path_row.setVisible(is_mflux)
        if is_mflux:
            self._refresh_model_path_status()
        else:
            # Re-enable Model/Quantize fields when not on MFLUX backend
            self.model_input.setEnabled(True)
            self.model_input.setToolTip("Hugging Face model ID or OllamaDiffuser model name")
            self.quantize_combo.setEnabled(True)
            self.quantize_combo.setToolTip("MFLUX runtime quantization level")

        # Pull button only for OllamaDiffuser
        self.pull_btn.setVisible(is_ollama)

        # Bonsai + CoreML setup rows
        self.bonsai_row.setVisible(is_bonsai)
        if is_bonsai:
            self._refresh_bonsai_status()
        self.coreml_row.setVisible(is_coreml)
        if is_coreml:
            self._refresh_coreml_status()

        # Backend hint label
        if is_mflux:
            hint = "💡 FLUX models: ~4 steps, 3.5–4.0 guidance."
            if not self.model_path_input.text().strip():
                hint += " Save a model locally for faster loading."
            self.backend_hint.setText(hint)
            self.backend_hint.show()
        elif is_mlx and self.model_input.text().strip() == "mlx-community/Lance-3B-AWQ-INT4":
            hint = "💡 Lance-3B: Highly efficient multimodal model. Recommended size: 512x512 or 1024x1024."
            self.backend_hint.setText(hint)
            self.backend_hint.show()
        elif is_bonsai:
            hint = "💡 Bonsai: 4 steps, guidance 1.0, no CFG, no negative prompt, no img2img. Image dimensions must be multiples of 32."
            self.backend_hint.setText(hint)
            self.backend_hint.show()
        elif is_coreml:
            hint = "💡 CoreML: SD 1.x/2.x model. Image dimensions should be 512x512 and multiples of 8. First call pays CoreML compile cost; subsequent calls are fast."
            self.backend_hint.setText(hint)
            self.backend_hint.show()
        else:
            self.backend_hint.hide()

        # Auto-update steps/guidance when switching to/from MFLUX
        if not self._restoring_state:
            if is_mflux:
                if self._pre_mflux_steps is None:
                    self._pre_mflux_steps = self.steps_spin.value()
                    self._pre_mflux_guidance = self.guidance_spin.value()
                    self.steps_spin.setValue(4)
                    self.guidance_spin.setValue(4.0)
            else:
                if self._pre_mflux_steps is not None:
                    self.steps_spin.setValue(self._pre_mflux_steps)
                    self.guidance_spin.setValue(self._pre_mflux_guidance)
                    self._pre_mflux_steps = None
                    self._pre_mflux_guidance = None

    def _on_pull_model(self):
        model = self.model_input.text().strip()
        if not model:
            self.status_label.setText("⚠ Enter a model name first")
            self.status_label.setStyleSheet("color: orange;")
            return
        if self.pull_worker is not None and self.pull_worker.isRunning():
            self.status_label.setText("⚠ A pull is already in progress")
            self.status_label.setStyleSheet("color: orange;")
            return

        self.pull_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.show()

        self.pull_worker = PullWorker(
            model, hf_cache_dir=self.hf_cache_input.text().strip() or None
        )
        self.pull_worker.status.connect(self._on_status)
        self.pull_worker.finished.connect(self._on_pull_finished)
        self.pull_worker.start()

    def _on_pull_finished(self, success: bool, message: str):
        self.pull_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.progress_bar.hide()
        self.progress_bar.reset()
        if success:
            self.status_label.setText(f"✅ {message}")
            self.status_label.setStyleSheet("color: green;")
        else:
            self.status_label.setText(f"❌ {message}")
            self.status_label.setStyleSheet("color: red;")

    # -- Bonsai (PrismML) backend ----------------------------------------

    def _refresh_bonsai_status(self):
        """Show the current bonsai install state in the status label."""
        try:
            import core_bonsai

            status = core_bonsai.validate_bonsai_install()
            self.bonsai_status_label.setText(status.message)
            # Pull enabled once install is ready (whether or not models are present)
            self.bonsai_pull_btn.setEnabled(status.installed)
        except Exception as exc:
            self.bonsai_status_label.setText(f"⚠ {exc}")

    def _on_bonsai_setup(self):
        """Run scripts/setup-bonsai.sh in a subprocess thread."""
        script = PROJECT_ROOT / "scripts" / "setup-bonsai.sh"
        if not script.is_file():
            self.status_label.setText(f"❌ Setup script not found: {script}")
            self.status_label.setStyleSheet("color: red;")
            return
        self.bonsai_setup_btn.setEnabled(False)
        self.bonsai_pull_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.status_label.setText(
            "🌳 Setting up Bonsai (one-time install, may take a few minutes)…"
        )
        self.status_label.setStyleSheet("color: blue;")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()

        self.bonsai_setup_worker = BonsaiSetupWorker(str(script))
        self.bonsai_setup_worker.finished.connect(self._on_bonsai_setup_finished)
        self.bonsai_setup_worker.start()

    def _on_bonsai_setup_finished(self, success: bool, message: str):
        self.bonsai_setup_btn.setEnabled(True)
        self.bonsai_pull_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.progress_bar.hide()
        self.progress_bar.reset()
        if success:
            self.status_label.setText(f"✅ {message}")
            self.status_label.setStyleSheet("color: green;")
        else:
            self.status_label.setText(f"❌ {message}")
            self.status_label.setStyleSheet("color: red;")
        self._refresh_bonsai_status()

    def _on_bonsai_pull(self):
        """Download a bonsai model in a thread."""
        import core_bonsai

        status = core_bonsai.validate_bonsai_install()
        if not status.installed:
            self.status_label.setText("⚠ Bonsai not installed. Click 'Setup Bonsai…' first.")
            self.status_label.setStyleSheet("color: orange;")
            return
        self.bonsai_setup_btn.setEnabled(False)
        self.bonsai_pull_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.status_label.setText("📥 Downloading bonsai model…")
        self.status_label.setStyleSheet("color: blue;")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()

        self.bonsai_pull_worker = BonsaiDownloadWorker(variant=core_bonsai.DEFAULT_VARIANT)
        self.bonsai_pull_worker.status.connect(self._on_status)
        self.bonsai_pull_worker.finished.connect(self._on_bonsai_pull_finished)
        self.bonsai_pull_worker.start()

    def _on_bonsai_pull_finished(self, success: bool, message: str):
        self.bonsai_setup_btn.setEnabled(True)
        self.bonsai_pull_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.progress_bar.hide()
        self.progress_bar.reset()
        if success:
            self.status_label.setText(f"✅ {message}")
            self.status_label.setStyleSheet("color: green;")
        else:
            self.status_label.setText(f"❌ {message}")
            self.status_label.setStyleSheet("color: red;")
        self._refresh_bonsai_status()

    # -- CoreML (Apple Neural Engine) backend ---------------------------

    def _refresh_coreml_status(self):
        """Show the current CoreML install state in the status label."""
        try:
            import core_coreml

            status = core_coreml.validate_coreml_install()
            self.coreml_status_label.setText(status.message)
            self.coreml_pull_btn.setEnabled(status.installed)
        except Exception as exc:
            self.coreml_status_label.setText(f"⚠ {exc}")

    def _on_coreml_setup(self):
        """Run scripts/setup-coreml.sh in a subprocess thread."""
        script = PROJECT_ROOT / "scripts" / "setup-coreml.sh"
        if not script.is_file():
            self.status_label.setText(f"❌ Setup script not found: {script}")
            self.status_label.setStyleSheet("color: red;")
            return
        self.coreml_setup_btn.setEnabled(False)
        self.coreml_pull_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.status_label.setText(
            "🍎 Setting up CoreML (one-time install, may take a few minutes)…"
        )
        self.status_label.setStyleSheet("color: blue;")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()

        self.coreml_setup_worker = CoreMLSetupWorker(str(script))
        self.coreml_setup_worker.finished.connect(self._on_coreml_setup_finished)
        self.coreml_setup_worker.start()

    def _on_coreml_setup_finished(self, success: bool, message: str):
        self.coreml_setup_btn.setEnabled(True)
        self.coreml_pull_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.progress_bar.hide()
        self.progress_bar.reset()
        if success:
            self.status_label.setText(f"✅ {message}")
            self.status_label.setStyleSheet("color: green;")
        else:
            self.status_label.setText(f"❌ {message}")
            self.status_label.setStyleSheet("color: red;")
        self._refresh_coreml_status()

    def _on_coreml_pull(self):
        """Open a small dialog to pick an alias, then download."""
        import core_coreml

        status = core_coreml.validate_coreml_install()
        if not status.installed:
            self.status_label.setText("⚠ CoreML not installed. Click 'Setup CoreML…' first.")
            self.status_label.setStyleSheet("color: orange;")
            return

        from PySide6.QtWidgets import QInputDialog

        alias, ok = QInputDialog.getItem(
            self,
            "Download CoreML model",
            "Pre-converted CoreML models on Hugging Face:",
            list(core_coreml.PRECONVERTED_HF_MODELS.keys()),
            3,  # default to sd-2-1-base-palettized
            False,
        )
        if not ok or not alias:
            return

        self.coreml_setup_btn.setEnabled(False)
        self.coreml_pull_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.status_label.setText(f"📥 Downloading {alias}…")
        self.status_label.setStyleSheet("color: blue;")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()

        self.coreml_pull_worker = CoreMLDownloadWorker(alias=alias)
        self.coreml_pull_worker.status.connect(self._on_status)
        self.coreml_pull_worker.finished.connect(self._on_coreml_pull_finished)
        self.coreml_pull_worker.start()

    def _on_coreml_pull_finished(self, success: bool, message: str):
        self.coreml_setup_btn.setEnabled(True)
        self.coreml_pull_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.progress_bar.hide()
        self.progress_bar.reset()
        if success:
            self.status_label.setText(f"✅ {message}")
            self.status_label.setStyleSheet("color: green;")
        else:
            self.status_label.setText(f"❌ {message}")
            self.status_label.setStyleSheet("color: red;")
        self._refresh_coreml_status()

    # -- MFLUX model path & save -----------------------------------------

    def _on_browse_model_path(self):
        """Open a folder picker for a saved MFLUX model directory."""
        start_dir = str(MODELS_DIR) if MODELS_DIR.exists() else str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Select Saved Model Directory", start_dir)
        if path:
            self.model_path_input.setText(path)
            self._on_model_path_changed()

    def _on_browse_hf_cache(self):
        """Open a folder picker for the HuggingFace cache directory."""
        current = self.hf_cache_input.text().strip()
        start_dir = current if current else str(Path.home() / ".cache" / "huggingface" / "hub")
        path = QFileDialog.getExistingDirectory(
            self, "Select HuggingFace Cache Directory", start_dir
        )
        if path:
            self.hf_cache_input.setText(path)

    def _on_model_path_changed(self):
        """Validate the model path and update the status label."""
        self._refresh_model_path_status()
        self._update_backend_dependent_controls()

    def _refresh_model_path_status(self):
        """Update model_path_status label based on the current path.

        Also disables the Model and Quantize fields when a valid saved model
        path is set, since the saved model's weights/quantization take
        precedence over those UI values.
        """
        path = self.model_path_input.text().strip()
        if not path:
            self.model_path_status.setText("")
            self.model_input.setEnabled(True)
            self.model_input.setToolTip("Hugging Face model ID or OllamaDiffuser model name")
            self.quantize_combo.setEnabled(True)
            self.quantize_combo.setToolTip("MFLUX runtime quantization level")
            return
        valid, meta = validate_saved_model(path)
        if valid and meta:
            ql = meta.get("quantization_level")
            q_str = f"{ql}-bit" if ql else "full precision"
            ver = meta.get("mflux_version") or "unknown"
            self.model_path_status.setText(f"✅ Valid model — {q_str} (mflux {ver})")
            self.model_path_status.setStyleSheet("color: green;")
        elif valid:
            self.model_path_status.setText("✅ Model directory found (no metadata)")
            self.model_path_status.setStyleSheet("color: #666;")
        else:
            self.model_path_status.setText("⚠ Not a valid saved model directory")
            self.model_path_status.setStyleSheet("color: orange;")

        # Grey out Model + Quantize when a valid saved model is active
        saved_model_active = valid
        self.model_input.setEnabled(not saved_model_active)
        self.model_input.setToolTip(
            "Determined by saved model path"
            if saved_model_active
            else "Hugging Face model ID or OllamaDiffuser model name"
        )
        self.quantize_combo.setEnabled(not saved_model_active)
        self.quantize_combo.setToolTip(
            "Determined by saved model path"
            if saved_model_active
            else "MFLUX runtime quantization level"
        )

    def _on_save_model(self):
        """Open the Save Model dialog and start the save worker."""
        if (
            hasattr(self, "_save_worker")
            and self._save_worker is not None
            and self._save_worker.isRunning()
        ):
            self.status_label.setText("⚠ A model save is already in progress")
            self.status_label.setStyleSheet("color: orange;")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Save Model")
        dlg.setMinimumWidth(420)
        layout = QVBoxLayout(dlg)

        layout.addWidget(QLabel("Model alias:"))
        alias_combo = QComboBox()
        try:
            models = list_mflux_models()
            for m in models:
                alias_combo.addItem(f"{m['alias']}  ({m['model_name']})", m["alias"])
        except (OSError, ValueError, ImportError, AttributeError):
            for a in ("dev", "schnell", "fibo", "z-image"):
                alias_combo.addItem(a, a)
        # Pre-select current model if it's in the list
        current = self.model_input.text().strip()
        idx = alias_combo.findData(current)
        if idx >= 0:
            alias_combo.setCurrentIndex(idx)
        layout.addWidget(alias_combo)

        layout.addWidget(QLabel("Quantization:"))
        q_combo = QComboBox()
        q_combo.addItem("4-bit (recommended)", 4)
        q_combo.addItem("8-bit", 8)
        q_combo.addItem("None (full precision)", 0)
        layout.addWidget(q_combo)

        layout.addWidget(QLabel("Save directory:"))
        path_row = QHBoxLayout()
        path_input = QLineEdit()

        def _update_default_path():
            alias = alias_combo.currentData()
            q = q_combo.currentData()
            q_label = f"{q}bit" if q else "fp"
            path_input.setText(str(MODELS_DIR / f"{alias}-{q_label}"))

        _update_default_path()
        alias_combo.currentIndexChanged.connect(lambda _: _update_default_path())
        q_combo.currentIndexChanged.connect(lambda _: _update_default_path())
        path_row.addWidget(path_input)
        pick_btn = QPushButton("…")
        pick_btn.setFixedWidth(30)
        pick_btn.clicked.connect(
            lambda: path_input.setText(
                QFileDialog.getExistingDirectory(dlg, "Choose Directory", str(MODELS_DIR))
                or path_input.text()
            )
        )
        path_row.addWidget(pick_btn)
        layout.addLayout(path_row)

        note = QLabel(
            "⚠ This downloads the full model from HuggingFace, quantizes it,\n"
            "and saves to disk. This is a one-time operation that may take\n"
            "several minutes and use significant disk space."
        )
        note.setWordWrap(True)
        note.setProperty("class", "hint")
        layout.addWidget(note)

        from PySide6.QtWidgets import QDialogButtonBox

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() != QDialog.Accepted:
            return

        alias = alias_combo.currentData()
        q = q_combo.currentData()
        quantize = q if q != 0 else None
        out_path = path_input.text().strip()
        if not out_path:
            return

        self.save_model_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()

        self._save_worker = SaveModelWorker(
            alias, quantize, out_path, hf_cache_dir=self.hf_cache_input.text().strip() or None
        )
        self._save_worker.status.connect(self._on_status)
        self._save_worker.finished.connect(self._on_save_model_finished)
        self._save_worker.start()

    def _on_save_model_finished(self, success: bool, message: str, saved_path: str):
        self.save_model_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.progress_bar.hide()
        self.progress_bar.reset()
        if success:
            self.status_label.setText(f"✅ {message}")
            self.status_label.setStyleSheet("color: green;")
            self.model_path_input.setText(saved_path)
            self._on_model_path_changed()
        else:
            self.status_label.setText(f"❌ Save failed: {message}")
            self.status_label.setStyleSheet("color: red;")

    def _on_hf_login(self):
        dlg = HFLoginDialog(self)
        dlg.exec()
        self._refresh_hf_button()

    def _refresh_hf_button(self):
        """Update the HF button label to reflect login state."""
        info = _cached_hf_status()
        if info:
            name = info.get("name", "unknown")
            self.hf_btn.setText(f"✅  HuggingFace: {name}")
        else:
            self.hf_btn.setText("🔑  HuggingFace Login")

    def _on_browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff)",
        )
        if path:
            self.image_path_input.setText(path)
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                scaled = pixmap.scaledToHeight(80, Qt.SmoothTransformation)
                self.image_thumbnail.setPixmap(scaled)
                self.image_thumbnail.show()

    def _scale_preview(self):
        if self._current_pixmap:
            scaled = self._current_pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled)

    # ----- State persistence ----------------------------------------------

    def _collect_state(self) -> dict:
        return {
            "prompt": self.prompt_input.toPlainText(),
            "negative_prompt": self.negative_prompt_input.toPlainText(),
            "steps": self.steps_spin.value(),
            "guidance": self.guidance_spin.value(),
            "width": str(self.width_combo.currentText()),
            "height": str(self.height_combo.currentText()),
            "seed": self.seed_input.text(),
            "use_t5": self.t5_check.isChecked(),
            "model": self.model_input.text(),
            "mode_tab": self.mode_tabs.currentIndex(),
            "image_path": self.image_path_input.text(),
            "denoise": self.denoise_spin.value(),
            "backend": self.backend_combo.currentData(),
            "mflux_quantize": self.quantize_combo.currentData(),
            "mflux_model_path": self.model_path_input.text(),
            "hf_cache_dir": self.hf_cache_input.text(),
            "bonsai_model_path": self.model_input.text()
            if self._resolved_backend() == BACKEND_BONSAI
            else "",
            "coreml_model_path": self.model_input.text()
            if self._resolved_backend() == BACKEND_COREML
            else "",
            "coreml_compute_unit": self.config.get("coreml_compute_unit", "CPU_AND_NE"),
        }

    def _restore_state(self):
        s = self._gui_state
        if not s:
            return
        self._restoring_state = True
        if "prompt" in s:
            self.prompt_input.setPlainText(s["prompt"])
        if "negative_prompt" in s:
            self.negative_prompt_input.setPlainText(s["negative_prompt"])
        if "steps" in s:
            self.steps_spin.setValue(s["steps"])
        if "guidance" in s:
            self.guidance_spin.setValue(s["guidance"])
        if "width" in s:
            self.width_combo.setCurrentText(str(s["width"]))
        if "height" in s:
            self.height_combo.setCurrentText(str(s["height"]))
        if "seed" in s:
            self.seed_input.setText(s["seed"])
        if "use_t5" in s:
            self.t5_check.setChecked(s["use_t5"])
        if "model" in s:
            self.model_input.setText(s["model"])
        if "mode_tab" in s:
            self.mode_tabs.setCurrentIndex(s["mode_tab"])
        if "image_path" in s:
            path = s["image_path"]
            if path and Path(path).exists():
                self.image_path_input.setText(path)
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    scaled = pixmap.scaledToHeight(80, Qt.SmoothTransformation)
                    self.image_thumbnail.setPixmap(scaled)
                    self.image_thumbnail.show()
        if "denoise" in s:
            self.denoise_spin.setValue(s["denoise"])
        if "backend" in s:
            idx = self.backend_combo.findData(s["backend"])
            if idx >= 0:
                self.backend_combo.setCurrentIndex(idx)
        if "mflux_quantize" in s:
            idx = self.quantize_combo.findData(s["mflux_quantize"])
            if idx >= 0:
                self.quantize_combo.setCurrentIndex(idx)
        if "mflux_model_path" in s and s["mflux_model_path"]:
            self.model_path_input.setText(s["mflux_model_path"])
        if "hf_cache_dir" in s and s["hf_cache_dir"]:
            self.hf_cache_input.setText(s["hf_cache_dir"])
        self._restoring_state = False
        # Apply backend-dependent visibility after all state is restored
        self._update_backend_dependent_controls()

    # ----- Generation -----------------------------------------------------

    def _on_generate(self):
        # If a generation is running, stop it
        if self.worker is not None and self.worker.isRunning():
            self._stop_generation()
            return

        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            self.status_label.setText("⚠ Enter a prompt first")
            self.status_label.setStyleSheet("color: orange;")
            return

        is_img2img = self.mode_tabs.currentIndex() == 1
        image_path: Optional[str] = None
        denoise: float = 1.0

        width = self.width_combo.currentData()
        height = self.height_combo.currentData()

        if is_img2img:
            image_path = self.image_path_input.text().strip() or None
            if not image_path:
                self.status_label.setText("⚠ Select an input image first")
                self.status_label.setStyleSheet("color: orange;")
                return
            err = validate_image_path(image_path)
            if err:
                self.status_label.setText(f"⚠ {err}")
                self.status_label.setStyleSheet("color: orange;")
                return
            denoise = self.denoise_spin.value()
        else:
            err = validate_dimensions(width, height)
            if err:
                self.status_label.setText(f"⚠ {err}")
                self.status_label.setStyleSheet("color: red;")
                return

        seed_text = self.seed_input.text().strip()
        if seed_text:
            try:
                seed = int(seed_text)
            except ValueError:
                self.status_label.setText("⚠ Seed must be a valid integer")
                self.status_label.setStyleSheet("color: red;")
                return
        else:
            seed = None

        config = dict(self.config)
        config["model"] = self.model_input.text().strip() or DEFAULT_CONFIG["model"]
        config["height"] = height
        config["width"] = width
        config["num_inference_steps"] = self.steps_spin.value()
        config["guidance_scale"] = self.guidance_spin.value()
        config["backend"] = self.backend_combo.currentData()
        config["mflux_quantize"] = self.quantize_combo.currentData()

        resolved_backend = self._resolved_backend()

        # Pass MFLUX saved model path through config
        model_path = self.model_path_input.text().strip()
        if model_path and resolved_backend == BACKEND_MFLUX:
            config["mflux_model_path"] = model_path
        else:
            config["mflux_model_path"] = None

        # Pass HF cache dir through config (applies to all backends)
        config["hf_cache_dir"] = self.hf_cache_input.text().strip() or None

        # Clean, validate, and cast variables using EyeGenConfig
        try:
            cfg_obj = EyeGenConfig.from_dict(config)
            errors = cfg_obj.validate()
            if errors:
                self.status_label.setText(f"⚠ {errors[0]}")
                self.status_label.setStyleSheet("color: red;")
                return
            config = cfg_obj.to_dict()
        except (ValueError, TypeError) as e:
            log.warning("Config cast failed: %s", e)
            self.status_label.setText(f"⚠ Invalid config: {e}")
            self.status_label.setStyleSheet("color: red;")
            return

        self._status_clear_timer.stop()
        self._generation_id += 1

        self.generate_btn.setText("⏹  Stop")
        self.generate_btn.setStyleSheet("background-color: #cc3333; color: white;")
        self.progress_bar.setRange(0, 0)  # indeterminate until step info arrives
        self.progress_bar.show()
        self._elapsed_seconds = 0
        self._current_phase = "Starting…"
        self.status_label.setText("Starting…")
        self.status_label.setStyleSheet("color: #853D4F;")
        self._elapsed_timer.start()

        # Resolve MFLUX quantize
        q = self.quantize_combo.currentData()
        mflux_quantize = q if q != 0 else None

        self.worker = GenerationWorker(
            prompt=prompt,
            negative_prompt=self.negative_prompt_input.toPlainText().strip(),
            cfg_weight=self.guidance_spin.value(),
            num_steps=self.steps_spin.value(),
            width=width,
            height=height,
            seed=seed,
            config=config,
            use_t5=self.t5_check.isChecked(),
            image_path=image_path,
            denoise=denoise,
            backend=resolved_backend,
            mflux_quantize=mflux_quantize,
        )
        self.worker.status.connect(self._on_status)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.quantize_failed.connect(self._on_quantize_failed)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.start()

    def _on_status(self, msg: str):
        self._current_phase = msg
        elapsed = f" ({self._elapsed_seconds}s)" if self._elapsed_seconds > 0 else ""
        self.status_label.setText(f"{msg}{elapsed}")

    def _on_progress(self, step: int, total: int):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(step)
        self._current_phase = f"Denoising step {step}/{total}"
        elapsed = f" ({self._elapsed_seconds}s)" if self._elapsed_seconds > 0 else ""
        self.status_label.setText(f"{self._current_phase}{elapsed}")

    def _on_finished(self, pil_image, output_path: str):
        self._elapsed_timer.stop()
        self.progress_bar.hide()
        self.progress_bar.reset()
        self._reset_generate_btn()
        elapsed = f" ({self._elapsed_seconds}s)" if self._elapsed_seconds > 0 else ""
        self.status_label.setText(f"✅ Done{elapsed}")
        self.status_label.setStyleSheet("color: green;")
        self.output_label.setText(f"Saved: {output_path}")

        self._current_pixmap = pil_to_pixmap(pil_image)
        self._scale_preview()
        self._arm_status_autoclear()

    def _on_error(self, full_traceback: str):
        self._elapsed_timer.stop()
        self.progress_bar.hide()
        self.progress_bar.reset()
        self._reset_generate_btn()

        # Short summary for the status bar
        lines = [line for line in full_traceback.strip().splitlines() if line.strip()]
        last_line = lines[-1] if lines else "Unknown error"
        self.status_label.setText("❌ Error — see details")
        self.status_label.setStyleSheet("color: red;")

        # Full traceback in a scrollable dialog
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Generation Error")
        dlg.setIcon(QMessageBox.Critical)
        dlg.setText(last_line)
        dlg.setDetailedText(full_traceback)
        dlg.setInformativeText(f"Full log: {LOG_FILE}")
        dlg.exec()
        self._arm_status_autoclear()

    def _on_quantize_failed(self, model_name: str):
        """Handle QuantizationError — offer retry without quantization."""
        self._elapsed_timer.stop()
        self.progress_bar.hide()
        self.progress_bar.reset()
        self._reset_generate_btn()
        self.status_label.setText("⚠️ Quantization failed")
        self.status_label.setStyleSheet("color: orange;")

        dlg = QMessageBox(self)
        dlg.setWindowTitle("Quantization Error")
        dlg.setIcon(QMessageBox.Warning)
        dlg.setText(
            f"The quantized weights for '{model_name}' are incompatible "
            f"with the current MLX version."
        )
        dlg.setInformativeText(
            "You can retry with full precision (uses more memory) or "
            "clear the model cache to force a fresh download."
        )
        retry_btn = dlg.addButton("Retry (full precision)", QMessageBox.AcceptRole)
        clear_btn = dlg.addButton("Clear cache && retry", QMessageBox.ActionRole)
        dlg.addButton(QMessageBox.Cancel)
        dlg.exec()

        clicked = dlg.clickedButton()
        if clicked == clear_btn:
            try:
                removed = clear_mflux_cache()
                log.info("Cleared %d cached revision(s)", len(removed))
            except (OSError, ValueError) as exc:
                log.error("Cache clear failed: %s", exc)
            self.quantize_combo.setCurrentIndex(
                self.quantize_combo.findData(0)
            )  # "None (full precision)"
            self._on_generate()
        elif clicked == retry_btn:
            self.quantize_combo.setCurrentIndex(
                self.quantize_combo.findData(0)
            )  # "None (full precision)"
            self._on_generate()

    def _stop_generation(self):
        """Cancel the running generation cooperatively without unsafe OS terminates."""
        if self.worker is None or not self.worker.isRunning():
            return
        log.info("User requested generation stop (backend=%s)", self.worker.backend)
        self.worker.cancel()

        self.status_label.setText("Cancelling… (cleaning up safely)")
        self.status_label.setStyleSheet("color: orange;")
        self.generate_btn.setEnabled(False)  # prevent double-clicks

        # For backends that wrap a long-lived subprocess (bonsai/coreml),
        # ask the pipeline to terminate the child so the user does not
        # have to wait for it to finish naturally. No-op for backends
        # whose pipeline does not expose cancel().
        if self.worker.backend in (BACKEND_BONSAI, BACKEND_COREML):
            self.worker.cancel()
        if self.worker.backend == BACKEND_MLX:
            # MLX has cooperative step-level interrupts and stops instantly
            pass
        else:
            # MFLUX / OllamaDiffuser run very quickly to completion in background.
            # Discard results and reset UI state immediately to keep GUI responsive.
            _clear_pipeline_cache()
            self._on_cancelled(force_terminated=False)

    def _on_cancelled(self, force_terminated=False):
        """Handle generation cancellation."""
        self._elapsed_timer.stop()
        self.progress_bar.hide()
        self.progress_bar.reset()
        self._reset_generate_btn()
        elapsed = f" ({self._elapsed_seconds}s)" if self._elapsed_seconds > 0 else ""
        self.status_label.setText(f"⏹ Cancelled{elapsed}")
        self.status_label.setStyleSheet("color: gray;")
        self._arm_status_autoclear()

    def _on_elapsed_tick(self):
        """Update status label with elapsed seconds."""
        self._elapsed_seconds += 1
        elapsed = f" ({self._elapsed_seconds}s)"
        self.status_label.setText(f"{self._current_phase}{elapsed}")

    def _arm_status_autoclear(self):
        """Start a 5s timer to reset the status label to 'Ready'."""
        self._autoclear_generation_id = self._generation_id
        self._status_clear_timer.start()

    def _clear_status(self):
        """Reset status to 'Ready' unless a new generation is in progress."""
        if self._generation_id != self._autoclear_generation_id:
            return
        if self.worker is not None and self.worker.isRunning():
            return
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet("color: gray;")

    def _reset_generate_btn(self):
        """Reset the generate button to its default state."""
        self.generate_btn.setText("✨  Generate")
        self.generate_btn.setStyleSheet("")
        self.generate_btn.setEnabled(True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    log.info(
        "EyeGen GUI starting (Python %s, arch=%s)",
        sys.version.split()[0],
        __import__("platform").machine(),
    )
    app = QApplication(sys.argv)
    app.setApplicationName("EyeGen")
    if sys.platform == "darwin":
        app.setFont(QFont(".AppleSystemUIFont", 13))
    app.setStyleSheet("""
        [class="hint"]    { font-size: 11px; color: #666666; }
        [class="display"] { font-size: 16px; }

        /* Accent: oxblood #4A1F2A (session palette, 600 step) */
        QPushButton {
            background-color: #853D4F;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
        }
        QPushButton:hover    { background-color: #AC4962; }
        QPushButton:pressed  { background-color: #5D2D39; }
        QPushButton:disabled { background-color: #BCB5AE; color: #6B6157; }

        QTabBar::tab:selected   { background-color: #853D4F; color: white; }
        QTabBar::tab:!selected  { background-color: transparent; color: #4A1F2A; }

        QProgressBar {
            background-color: #EAE8E6;
            border: none;
            border-radius: 8px;
        }
        QProgressBar::chunk {
            background-color: #853D4F;
            border-radius: 8px;
        }
    """)

    # Set app icon from icon.png (next to this script)
    icon_path = Path(__file__).resolve().parent / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
