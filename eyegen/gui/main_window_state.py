"""State persistence mixin for MainWindow."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from eyegen.config import Backend


class MainWindowStateMixin:
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
            "backend": self.backend_combo.currentData().value,
            "mflux_quantize": self.quantize_combo.currentData(),
            "mflux_model_path": self.model_path_input.text(),
            "hf_cache_dir": self.hf_cache_input.text(),
            "bonsai_model_path": self.model_input.text()
            if self._resolved_backend() == Backend.BONSAI
            else "",
            "coreml_model_path": self.model_input.text()
            if self._resolved_backend() == Backend.COREML
            else "",
            "coreml_compute_unit": self.config.get("coreml_compute_unit", "CPU_AND_NE"),
        }

    def _restore_state(self):
        s = self._gui_state
        if not s:
            return
        self._restoring_state = True
        self._restore_text_state(s)
        self._restore_numeric_state(s)
        self._restore_model_backend_state(s)
        self._restore_image_state(s)
        self._restore_optional_fields(s)
        self._restoring_state = False
        self._update_backend_dependent_controls()

    def _restore_text_state(self, s: dict):
        if "prompt" in s:
            self.prompt_input.setPlainText(s["prompt"])
        if "negative_prompt" in s:
            self.negative_prompt_input.setPlainText(s["negative_prompt"])
        if "model" in s:
            self.model_input.setText(s["model"])

    def _restore_numeric_state(self, s: dict):
        if "steps" in s:
            self.steps_spin.setValue(s["steps"])
        if "guidance" in s:
            self.guidance_spin.setValue(s["guidance"])
        if "width" in s:
            self.width_combo.setCurrentText(str(s["width"]))
        if "height" in s:
            self.height_combo.setCurrentText(str(s["height"]))
        if "denoise" in s:
            self.denoise_spin.setValue(s["denoise"])

    def _restore_model_backend_state(self, s: dict):
        if "seed" in s:
            self.seed_input.setText(s["seed"])
        if "use_t5" in s:
            self.t5_check.setChecked(s["use_t5"])
        if "mode_tab" in s:
            self.mode_tabs.setCurrentIndex(s["mode_tab"])
        if "backend" in s:
            try:
                backend = Backend(s["backend"])
            except ValueError:
                backend = None
            idx = self.backend_combo.findData(backend) if backend else -1
            if idx >= 0:
                self.backend_combo.setCurrentIndex(idx)
        if "mflux_quantize" in s:
            idx = self.quantize_combo.findData(s["mflux_quantize"])
            if idx >= 0:
                self.quantize_combo.setCurrentIndex(idx)

    def _restore_image_state(self, s: dict):
        if "image_path" not in s:
            return
        path = s["image_path"]
        if not path or not Path(path).exists():
            return
        self.image_path_input.setText(path)
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return
        scaled = pixmap.scaledToHeight(80, Qt.SmoothTransformation)
        self.image_thumbnail.setPixmap(scaled)
        self.image_thumbnail.show()

    def _restore_optional_fields(self, s: dict):
        if "mflux_model_path" in s and s["mflux_model_path"]:
            self.model_path_input.setText(s["mflux_model_path"])
        if "hf_cache_dir" in s and s["hf_cache_dir"]:
            self.hf_cache_input.setText(s["hf_cache_dir"])
