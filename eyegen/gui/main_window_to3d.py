"""Image-to-3D (Idol) page mixin for MainWindow — the fourth nav tab."""

import shutil
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from eyegen import OUTPUT_DIR
from eyegen.backends import idol
from eyegen.backends.idol.constants import (
    IDOL_DEVICES,
    IDOL_GAUSSIAN_CHOICES,
    IDOL_IMAGE_EXTENSIONS,
)
from eyegen.gui.backend_workers import To3DWorker
from eyegen.validation import validate_image_path


class MainWindowTo3DMixin:
    def _build_to3d_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Image to 3D")
        title.setProperty("class", "section")
        layout.addWidget(title)

        hint = QLabel(
            "One image of an object → a 3D Gaussian splat (.ply), via the Idol sidecar. "
            "Open the .ply in a splat viewer (e.g. SuperSplat)."
        )
        hint.setWordWrap(True)
        hint.setProperty("hint", True)
        layout.addWidget(hint)

        self.to3d_install_warning = QLabel()
        self.to3d_install_warning.setWordWrap(True)
        self.to3d_install_warning.setProperty("class", "warning")
        layout.addWidget(self.to3d_install_warning)
        self._refresh_to3d_install_warning()

        self._build_to3d_image_section(layout)
        self._build_to3d_options(layout)
        self._build_to3d_run_section(layout)
        layout.addStretch()
        self.content_stack.addWidget(page)

    def _build_to3d_run_section(self, layout):
        self.to3d_run_btn = QPushButton("Generate 3D")
        self.to3d_run_btn.setMinimumHeight(48)
        self.to3d_run_btn.setProperty("gradient", True)
        self.to3d_run_btn.setToolTip("Turn the image into a 3D splat (runs for minutes on MPS)")
        self.to3d_run_btn.clicked.connect(self._on_to3d_run)
        layout.addWidget(self.to3d_run_btn)

        self.to3d_status_label = QLabel("")
        self.to3d_status_label.setWordWrap(True)
        layout.addWidget(self.to3d_status_label)

        self.to3d_result_preview = QLabel()
        self.to3d_result_preview.setAlignment(Qt.AlignCenter)
        self.to3d_result_preview.setMaximumHeight(160)
        self.to3d_result_preview.hide()
        layout.addWidget(self.to3d_result_preview)

        result_row = QHBoxLayout()
        self.to3d_ply_label = QLabel("")
        self.to3d_ply_label.setWordWrap(True)
        result_row.addWidget(self.to3d_ply_label, 1)
        self.to3d_reveal_btn = QPushButton("Reveal in Finder")
        self.to3d_reveal_btn.setCursor(Qt.PointingHandCursor)
        self.to3d_reveal_btn.clicked.connect(self._on_to3d_reveal)
        self.to3d_reveal_btn.hide()
        result_row.addWidget(self.to3d_reveal_btn)
        layout.addLayout(result_row)

    def _build_to3d_image_section(self, layout):
        image_label = QLabel("Input Image")
        image_label.setProperty("class", "section-heading")
        layout.addWidget(image_label)
        image_row = QHBoxLayout()
        self.to3d_image_input = QLineEdit()
        self.to3d_image_input.setReadOnly(True)
        self.to3d_image_input.setPlaceholderText("No image selected…")
        image_row.addWidget(self.to3d_image_input)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._on_to3d_browse_image)
        image_row.addWidget(browse_btn)
        layout.addLayout(image_row)

        self.to3d_thumbnail = QLabel()
        self.to3d_thumbnail.setAlignment(Qt.AlignCenter)
        self.to3d_thumbnail.setMaximumHeight(100)
        self.to3d_thumbnail.hide()
        layout.addWidget(self.to3d_thumbnail)

    def _build_to3d_options(self, layout):
        self._build_to3d_mode_row(layout)
        self._build_to3d_steps_row(layout)
        self._build_to3d_gaussians_row(layout)
        self._build_to3d_seed_row(layout)
        self._build_to3d_device_row(layout)
        self._build_to3d_out_row(layout)

    def _build_to3d_mode_row(self, layout):
        mode_row = QHBoxLayout()
        mode_label = QLabel("Mode")
        mode_label.setProperty("class", "input-label")
        mode_row.addWidget(mode_label)
        self.to3d_mode_combo = QComboBox()
        self.to3d_mode_combo.addItem("synthetic (renders / AI images)", "synthetic")
        self.to3d_mode_combo.addItem("real (photos)", "real")
        mode_row.addWidget(self.to3d_mode_combo)
        layout.addLayout(mode_row)

    def _build_to3d_steps_row(self, layout):
        steps_row = QHBoxLayout()
        steps_label = QLabel("Steps")
        steps_label.setProperty("class", "input-label")
        steps_row.addWidget(steps_label)
        self.to3d_steps_spin = QSpinBox()
        self.to3d_steps_spin.setRange(1, 100)
        self.to3d_steps_spin.setValue(self.config.idol_steps)
        self.to3d_steps_spin.setToolTip(
            "Sampling steps. Fewer are faster but lose detail. On MPS ~30 s/step."
        )
        steps_row.addWidget(self.to3d_steps_spin)
        layout.addLayout(steps_row)

    def _build_to3d_gaussians_row(self, layout):
        gaussians_row = QHBoxLayout()
        gaussians_label = QLabel("Gaussians")
        gaussians_label.setProperty("class", "input-label")
        gaussians_row.addWidget(gaussians_label)
        self.to3d_gaussians_combo = QComboBox()
        for choice in IDOL_GAUSSIAN_CHOICES:
            self.to3d_gaussians_combo.addItem(str(choice), choice)
        idx = self.to3d_gaussians_combo.findData(self.config.idol_num_gaussians)
        self.to3d_gaussians_combo.setCurrentIndex(idx if idx >= 0 else 2)
        self.to3d_gaussians_combo.setToolTip(
            "How many 3D Gaussians to generate (more = finer detail)"
        )
        gaussians_row.addWidget(self.to3d_gaussians_combo)
        layout.addLayout(gaussians_row)

    def _build_to3d_seed_row(self, layout):
        seed_row = QHBoxLayout()
        seed_label = QLabel("Seed")
        seed_label.setProperty("class", "input-label")
        seed_row.addWidget(seed_label)
        self.to3d_seed_input = QLineEdit()
        self.to3d_seed_input.setPlaceholderText("42")
        seed_row.addWidget(self.to3d_seed_input)
        layout.addLayout(seed_row)

    def _build_to3d_device_row(self, layout):
        device_row = QHBoxLayout()
        device_label = QLabel("Device")
        device_label.setProperty("class", "input-label")
        device_row.addWidget(device_label)
        self.to3d_device_combo = QComboBox()
        for dev in IDOL_DEVICES:
            self.to3d_device_combo.addItem(dev, dev)
        idx = self.to3d_device_combo.findData(self.config.idol_device)
        self.to3d_device_combo.setCurrentIndex(idx if idx >= 0 else 0)
        device_row.addWidget(self.to3d_device_combo)
        layout.addLayout(device_row)

    def _build_to3d_out_row(self, layout):
        out_row = QHBoxLayout()
        out_label = QLabel("Output")
        out_label.setProperty("class", "input-label")
        out_row.addWidget(out_label)
        self.to3d_out_input = QLineEdit()
        self.to3d_out_input.setText(str(Path(str(OUTPUT_DIR)) / "to3d"))
        out_row.addWidget(self.to3d_out_input)
        out_browse_btn = QPushButton("Browse…")
        out_browse_btn.setFixedWidth(70)
        out_browse_btn.clicked.connect(self._on_to3d_browse_out)
        out_row.addWidget(out_browse_btn)
        layout.addLayout(out_row)

    def _refresh_to3d_install_warning(self):
        status = idol.validate_idol_install(self.config)
        if status.installed:
            self.to3d_install_warning.hide()
        else:
            self.to3d_install_warning.setText(
                "⚠ Idol sidecar not installed. Run in a terminal:  ./generate.py setup-idol"
            )
            self.to3d_install_warning.show()

    def _on_to3d_browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Image",
            str(Path.home()),
            "Images (" + " ".join(f"*{e}" for e in IDOL_IMAGE_EXTENSIONS) + ")",
        )
        if path:
            self.to3d_image_input.setText(path)
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                scaled = pixmap.scaledToHeight(100, Qt.SmoothTransformation)
                self.to3d_thumbnail.setPixmap(scaled)
                self.to3d_thumbnail.show()

    def _on_to3d_browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory", str(Path.home()))
        if path:
            self.to3d_out_input.setText(path)

    def _on_to3d_run(self):
        if self.to3d_worker is not None and self.to3d_worker.isRunning():
            self.to3d_worker.cancel()
            self.to3d_status_label.setText("Cancelling…")
            return

        image = self.to3d_image_input.text().strip()
        err = validate_image_path(image) if image else "Select an input image first."
        if err:
            self.to3d_status_label.setText(f"⚠ {err}")
            return

        status = idol.validate_idol_install(self.config)
        if not status.installed:
            self.to3d_status_label.setText(f"⚠ {status.message}")
            return

        seed_text = self.to3d_seed_input.text().strip()
        try:
            seed = int(seed_text) if seed_text else 42
        except ValueError:
            self.to3d_status_label.setText("⚠ Seed must be an integer.")
            return

        # Stays enabled as the Cancel control while the worker runs.
        self.to3d_run_btn.setText("Cancel")
        self.to3d_status_label.setText("Starting Idol… (first run downloads ~5 GB)")
        self.to3d_reveal_btn.hide()
        self.to3d_result_preview.hide()

        self.to3d_worker = To3DWorker(
            image=Path(image),
            out_dir=Path(self.to3d_out_input.text().strip() or Path(str(OUTPUT_DIR)) / "to3d"),
            mode=self.to3d_mode_combo.currentData(),
            steps=self.to3d_steps_spin.value(),
            num_gaussians=self.to3d_gaussians_combo.currentData(),
            seed=seed,
            device=self.to3d_device_combo.currentData(),
            config=self.config,
        )
        self.to3d_worker.finished.connect(self._on_to3d_finished)
        self.to3d_worker.error.connect(self._on_to3d_error)
        self.to3d_worker.status.connect(self._on_to3d_status)
        self.to3d_worker.start()

    def _on_to3d_status(self, line: str):
        # tqdm redraws with \r; show only the latest segment.
        segment = line.split("\r")[-1].strip()
        if segment:
            self.to3d_status_label.setText(segment)

    def _on_to3d_finished(self, result, message: str):
        self.to3d_run_btn.setText("Generate 3D")
        self.to3d_status_label.setText(f"✅ {message}")
        self.to3d_ply_label.setText(str(result.ply))
        self.to3d_reveal_btn.show()
        self._to3d_result = result
        if result.input_png.is_file():
            pixmap = QPixmap(str(result.input_png))
            if not pixmap.isNull():
                scaled = pixmap.scaledToHeight(160, Qt.SmoothTransformation)
                self.to3d_result_preview.setPixmap(scaled)
                self.to3d_result_preview.show()

    def _on_to3d_error(self, message: str):
        self.to3d_run_btn.setText("Generate 3D")
        self.to3d_status_label.setText(f"❌ {message}")

    def _on_to3d_reveal(self):
        result = getattr(self, "_to3d_result", None)
        if result is not None:
            subprocess.run([shutil.which("open") or "/usr/bin/open", "-R", str(result.ply)])  # noqa: S603
