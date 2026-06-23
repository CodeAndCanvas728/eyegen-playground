"""Save-model dialog and worker launcher mixin for MainWindow."""

from PySide6.QtWidgets import QFileDialog, QLabel, QVBoxLayout

from eyegen import MODELS_DIR, list_mflux_models
from eyegen.gui.dialogs import HFLoginDialog
from eyegen.gui.utils import _cached_hf_status


class MainWindowSaveModelMixin:
    def _on_save_model(self):
        if (
            hasattr(self, "_save_worker")
            and self._save_worker is not None
            and self._save_worker.isRunning()
        ):
            self.status_label.setText("⚠ A model save is already in progress")
            self.status_label.setStyleSheet("color: orange;")
            return

        result = self._show_save_model_dialog()
        if result is None:
            return
        alias, quantize, out_path = result
        self._run_save_model_worker(alias, quantize, out_path)

    def _show_save_model_dialog(self):
        from PySide6.QtWidgets import QDialog, QDialogButtonBox

        dlg = QDialog(self)
        dlg.setWindowTitle("Save Model")
        dlg.setMinimumWidth(420)
        layout = QVBoxLayout(dlg)

        layout.addWidget(QLabel("Model alias:"))
        alias_combo = self._build_alias_combo()
        layout.addWidget(alias_combo)

        layout.addWidget(QLabel("Quantization:"))
        q_combo = self._build_quantize_combo()
        layout.addWidget(q_combo)

        layout.addWidget(QLabel("Save directory:"))
        path_row, path_input = self._build_save_path_row(dlg, alias_combo, q_combo)
        layout.addLayout(path_row)

        note = QLabel(
            "⚠ This downloads the full model from HuggingFace, quantizes it,\n"
            "and saves to disk. This is a one-time operation that may take\n"
            "several minutes and use significant disk space."
        )
        note.setWordWrap(True)
        note.setProperty("class", "hint")
        layout.addWidget(note)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() != QDialog.Accepted:
            return None

        alias = alias_combo.currentData()
        q = q_combo.currentData()
        quantize = q if q != 0 else None
        out_path = path_input.text().strip()
        if not out_path:
            return None
        return alias, quantize, out_path

    def _run_save_model_worker(self, alias, quantize, out_path):
        self.save_model_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()

        from eyegen.gui.workers import SaveModelWorker

        self._save_worker = SaveModelWorker(
            alias, quantize, out_path, hf_cache_dir=self.hf_cache_input.text().strip() or None
        )
        self._save_worker.status.connect(self._on_status)
        self._save_worker.finished.connect(self._on_save_model_finished)
        self._save_worker.start()

    def _build_alias_combo(self):
        from PySide6.QtWidgets import QComboBox

        alias_combo = QComboBox()
        try:
            models = list_mflux_models()
            for m in models:
                alias_combo.addItem(f"{m['alias']}  ({m['model_name']})", m["alias"])
        except (OSError, ValueError, ImportError, AttributeError):
            for a in ("dev", "schnell", "fibo", "z-image"):
                alias_combo.addItem(a, a)
        current = self.model_input.text().strip()
        idx = alias_combo.findData(current)
        if idx >= 0:
            alias_combo.setCurrentIndex(idx)
        return alias_combo

    def _build_quantize_combo(self):
        from PySide6.QtWidgets import QComboBox

        q_combo = QComboBox()
        q_combo.addItem("4-bit (recommended)", 4)
        q_combo.addItem("8-bit", 8)
        q_combo.addItem("None (full precision)", 0)
        return q_combo

    def _build_save_path_row(self, dlg, alias_combo, q_combo):
        from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QPushButton

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
        return path_row, path_input

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
        info = _cached_hf_status()
        if info:
            name = info.get("name", "unknown")
            self.hf_btn.setText(f"✅  HuggingFace: {name}")
        else:
            self.hf_btn.setText("🔑  HuggingFace Login")
