"""General event handlers mixin for MainWindow."""

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFileDialog

from eyegen import MODELS_DIR, validate_saved_model

log = logging.getLogger("eyegen")


class MainWindowHandlersMixin:
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_preview()

    def closeEvent(self, event):
        self._elapsed_timer.stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)
        from eyegen.gui.state import save_gui_state

        try:
            save_gui_state(self._collect_state())
            log.info("GUI state saved")
        except (ValueError, OSError) as exc:
            # An unrecognized model name makes detect_backend raise; never let
            # state persistence abort window shutdown.
            log.warning("Could not save GUI state: %s", exc)
        super().closeEvent(event)

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
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()

        from eyegen.gui.workers import PullWorker

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

    def _on_browse_model_path(self):
        start_dir = str(MODELS_DIR) if MODELS_DIR.exists() else str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Select Saved Model Directory", start_dir)
        if path:
            self.model_path_input.setText(path)
            self._on_model_path_changed()

    def _on_browse_hf_cache(self):
        current = self.hf_cache_input.text().strip()
        start_dir = current if current else str(Path.home() / ".cache" / "huggingface" / "hub")
        path = QFileDialog.getExistingDirectory(
            self, "Select HuggingFace Cache Directory", start_dir
        )
        if path:
            self.hf_cache_input.setText(path)

    def _on_model_path_changed(self):
        self._refresh_model_path_status()
        self._update_backend_dependent_controls()

    def _refresh_model_path_status(self):
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
