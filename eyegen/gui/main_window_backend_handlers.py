"""Bonsai/CoreML backend handlers mixin for MainWindow."""

import logging

from PySide6.QtWidgets import QInputDialog

from eyegen import PROJECT_ROOT
from eyegen.backends import bonsai, coreml

log = logging.getLogger("eyegen")


class MainWindowBackendHandlersMixin:
    def _refresh_bonsai_status(self):
        try:
            from eyegen.backends import bonsai

            status = bonsai.validate_bonsai_install()
            self.bonsai_status_label.setText(status.message)
            self.bonsai_pull_btn.setEnabled(status.installed)
        except (OSError, ValueError, ImportError, AttributeError) as exc:
            self.bonsai_status_label.setText(f"⚠ {exc}")

    def _on_bonsai_setup(self):
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

        from eyegen.gui.backend_workers import BonsaiSetupWorker

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
        status = bonsai.validate_bonsai_install()
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

        from eyegen.gui.backend_workers import BonsaiDownloadWorker

        self.bonsai_pull_worker = BonsaiDownloadWorker(variant=bonsai.DEFAULT_VARIANT)
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

    def _refresh_coreml_status(self):
        try:
            from eyegen.backends import coreml

            status = coreml.validate_coreml_install()
            self.coreml_status_label.setText(status.message)
            self.coreml_pull_btn.setEnabled(status.installed)
        except (OSError, ValueError, ImportError, AttributeError) as exc:
            self.coreml_status_label.setText(f"⚠ {exc}")

    def _on_coreml_setup(self):
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

        from eyegen.gui.backend_workers import CoreMLSetupWorker

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
        status = coreml.validate_coreml_install()
        if not status.installed:
            self.status_label.setText("⚠ CoreML not installed. Click 'Setup CoreML…' first.")
            self.status_label.setStyleSheet("color: orange;")
            return

        alias, ok = QInputDialog.getItem(
            self,
            "Download CoreML model",
            "Pre-converted CoreML models on Hugging Face:",
            list(coreml.PRECONVERTED_HF_MODELS.keys()),
            3,
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

        from eyegen.gui.backend_workers import CoreMLDownloadWorker

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
