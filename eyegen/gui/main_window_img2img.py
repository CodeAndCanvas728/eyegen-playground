"""img2img controls builder mixin for MainWindow."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class MainWindowImg2ImgMixin:
    def _build_img2img_controls(self) -> QWidget:
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
        self.denoise_slider.setRange(5, 100)
        self.denoise_slider.setValue(75)
        self.denoise_slider.valueChanged.connect(lambda v: self.denoise_spin.setValue(v / 100.0))
        self.denoise_spin.valueChanged.connect(
            lambda v: self.denoise_slider.setValue(int(round(v * 100)))
        )
        img2img_layout.addWidget(self.denoise_slider)

        widget.hide()
        return widget
