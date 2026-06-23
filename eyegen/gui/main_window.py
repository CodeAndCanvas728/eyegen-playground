"""Main window class for the EyeGen GUI."""

import logging
from typing import Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMainWindow

from eyegen import (
    CONFIG_DIR,
    DEFAULT_CONFIG,
    EyeGenConfig,
    load_config,
)
from eyegen.config import Backend
from eyegen.gui.main_window_backend_handlers import MainWindowBackendHandlersMixin
from eyegen.gui.main_window_controls import MainWindowControlsMixin
from eyegen.gui.main_window_handlers import MainWindowHandlersMixin
from eyegen.gui.main_window_img2img import MainWindowImg2ImgMixin
from eyegen.gui.main_window_lifecycle import MainWindowLifecycleMixin
from eyegen.gui.main_window_save_model import MainWindowSaveModelMixin
from eyegen.gui.main_window_settings import MainWindowSettingsMixin
from eyegen.gui.main_window_state import MainWindowStateMixin
from eyegen.gui.main_window_ui import MainWindowUIMixin
from eyegen.gui.state import load_gui_state

log = logging.getLogger("eyegen")


class MainWindow(
    QMainWindow,
    MainWindowUIMixin,
    MainWindowSettingsMixin,
    MainWindowImg2ImgMixin,
    MainWindowControlsMixin,
    MainWindowHandlersMixin,
    MainWindowBackendHandlersMixin,
    MainWindowSaveModelMixin,
    MainWindowLifecycleMixin,
    MainWindowStateMixin,
):
    """Main window class for the EyeGen GUI.

    This class coordinates the application's main interface by composing the
    following mixin classes:
    - MainWindowUIMixin: Builds and sets up UI components/layouts.
    - MainWindowSettingsMixin: Manages inputs and updates configuration properties.
    - MainWindowImg2ImgMixin: Validates and manages inputs for Image-to-Image mode.
    - MainWindowControlsMixin: Populates combo-boxes and initializes controls values.
    - MainWindowHandlersMixin: Triggers backend changes, prompts checks, and UI resets.
    - MainWindowBackendHandlersMixin: Installs backends and triggers model download workers.
    - MainWindowSaveModelMixin: Connects GUI actions to MFLUX local model saving logic.
    - MainWindowLifecycleMixin: Handles elapsed time and state updates during generation.
    - MainWindowStateMixin: Restores and saves UI state history.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EyeGen")
        self.setMinimumSize(950, 650)
        self.resize(1200, 800)

        self.worker: Optional[object] = None
        self.pull_worker: Optional[object] = None
        self._save_worker: Optional[object] = None
        self.config = load_config()
        self._gui_state = load_gui_state()
        self._log_file = CONFIG_DIR / "eyegen.log"
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

    def _on_generate(self):
        if self.worker is not None and self.worker.isRunning():
            self._stop_generation()
            return

        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            self._set_status("⚠ Enter a prompt first", "orange")
            return

        width = self.width_combo.currentData()
        height = self.height_combo.currentData()
        image_path, denoise, ok = self._validate_inputs(width, height)
        if not ok:
            return

        seed, ok = self._parse_seed()
        if not ok:
            return

        config, ok = self._build_generation_config(width, height)
        if not ok:
            return

        resolved_backend = self._resolved_backend()
        if resolved_backend is None:
            self._set_status("⚠ Unrecognized model for the selected backend", "red")
            return
        self._start_generation(
            prompt, config, width, height, seed, image_path, denoise, resolved_backend
        )

    def _validate_inputs(self, width: int, height: int):
        from eyegen import validate_dimensions, validate_image_path

        is_img2img = self.mode_tabs.currentIndex() == 1
        if is_img2img:
            image_path = self.image_path_input.text().strip() or None
            if not image_path:
                self._set_status("⚠ Select an input image first", "orange")
                return None, 1.0, False
            err = validate_image_path(image_path)
            if err:
                self._set_status(f"⚠ {err}", "orange")
                return None, 1.0, False
            return image_path, self.denoise_spin.value(), True

        err = validate_dimensions(width, height)
        if err:
            self._set_status(f"⚠ {err}", "red")
            return None, 1.0, False
        return None, 1.0, True

    def _parse_seed(self):
        seed_text = self.seed_input.text().strip()
        if not seed_text:
            return None, True
        try:
            return int(seed_text), True
        except ValueError:
            self._set_status("⚠ Seed must be a valid integer", "red")
            return None, False

    def _set_status(self, message: str, color: str):
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {color};")

    def _build_generation_config(self, width: int, height: int):
        config = dict(self.config)
        config["model"] = self.model_input.text().strip() or DEFAULT_CONFIG["model"]
        config["height"] = height
        config["width"] = width
        config["num_inference_steps"] = self.steps_spin.value()
        config["guidance_scale"] = self.guidance_spin.value()
        config["backend"] = self.backend_combo.currentData()
        config["mflux_quantize"] = self.quantize_combo.currentData()

        resolved_backend = self._resolved_backend()
        model_path = self.model_path_input.text().strip()
        if model_path and resolved_backend == Backend.MFLUX:
            config["mflux_model_path"] = model_path
        else:
            config["mflux_model_path"] = None

        config["hf_cache_dir"] = self.hf_cache_input.text().strip() or None

        try:
            cfg_obj = EyeGenConfig.from_dict(config)
            errors = cfg_obj.validate()
            if errors:
                self._set_status(f"⚠ {errors[0]}", "red")
                return None, False
            return cfg_obj.to_dict(), True
        except (ValueError, TypeError) as e:
            log.warning("Config cast failed: %s", e)
            self._set_status(f"⚠ Invalid config: {e}", "red")
            return None, False

    def _start_generation(
        self,
        prompt,
        config,
        width,
        height,
        seed,
        image_path,
        denoise,
        resolved_backend,
    ):
        self._status_clear_timer.stop()
        self._generation_id += 1

        self.generate_btn.setText("⏹  Stop")
        self.generate_btn.setStyleSheet("background-color: #cc3333; color: white;")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self._elapsed_seconds = 0
        self._current_phase = "Starting…"
        self.status_label.setText("Starting…")
        self.status_label.setStyleSheet("color: #853D4F;")
        self._elapsed_timer.start()

        q = self.quantize_combo.currentData()
        mflux_quantize = q if q != 0 else None

        from eyegen.gui.workers import GenerationWorker

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
