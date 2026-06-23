"""Generation lifecycle handlers mixin for MainWindow."""

import logging

from PySide6.QtWidgets import QMessageBox

from eyegen.gui.cache import _clear_pipeline_cache
from eyegen.gui.utils import pil_to_pixmap

log = logging.getLogger("eyegen")


class MainWindowLifecycleMixin:
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

        lines = [line for line in full_traceback.strip().splitlines() if line.strip()]
        last_line = lines[-1] if lines else "Unknown error"
        self.status_label.setText("❌ Error — see details")
        self.status_label.setStyleSheet("color: red;")

        dlg = QMessageBox(self)
        dlg.setWindowTitle("Generation Error")
        dlg.setIcon(QMessageBox.Critical)
        dlg.setText(last_line)
        dlg.setDetailedText(full_traceback)
        dlg.setInformativeText(f"Full log: {self._log_file}")
        dlg.exec()
        self._arm_status_autoclear()

    def _on_quantize_failed(self, model_name: str):
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
                from eyegen import clear_mflux_cache

                removed = clear_mflux_cache()
                log.info("Cleared %d cached revision(s)", len(removed))
            except (OSError, ValueError) as exc:
                log.error("Cache clear failed: %s", exc)
            self.quantize_combo.setCurrentIndex(self.quantize_combo.findData(0))
            self._on_generate()
        elif clicked == retry_btn:
            self.quantize_combo.setCurrentIndex(self.quantize_combo.findData(0))
            self._on_generate()

    def _stop_generation(self):
        if self.worker is None or not self.worker.isRunning():
            return
        log.info("User requested generation stop (backend=%s)", self.worker.backend)
        self.worker.cancel()

        self.status_label.setText("Cancelling… (cleaning up safely)")
        self.status_label.setStyleSheet("color: orange;")
        self.generate_btn.setEnabled(False)
        # The worker resets the UI by emitting `cancelled` once it actually
        # stops; do not reset here or the user could launch a second run while
        # the first is still terminating.

    def _on_cancelled(self, force_terminated=False):
        _clear_pipeline_cache()
        self._elapsed_timer.stop()
        self.progress_bar.hide()
        self.progress_bar.reset()
        self._reset_generate_btn()
        elapsed = f" ({self._elapsed_seconds}s)" if self._elapsed_seconds > 0 else ""
        self.status_label.setText(f"⏹ Cancelled{elapsed}")
        self.status_label.setStyleSheet("color: gray;")
        self._arm_status_autoclear()

    def _on_elapsed_tick(self):
        self._elapsed_seconds += 1
        elapsed = f" ({self._elapsed_seconds}s)"
        self.status_label.setText(f"{self._current_phase}{elapsed}")

    def _arm_status_autoclear(self):
        self._autoclear_generation_id = self._generation_id
        self._status_clear_timer.start()

    def _clear_status(self):
        if self._generation_id != self._autoclear_generation_id:
            return
        if self.worker is not None and self.worker.isRunning():
            return
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet("color: gray;")

    def _reset_generate_btn(self):
        self.generate_btn.setText("✨  Generate")
        self.generate_btn.setStyleSheet("")
        self.generate_btn.setEnabled(True)
