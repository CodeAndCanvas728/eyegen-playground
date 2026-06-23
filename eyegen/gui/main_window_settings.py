"""Settings panel builder mixin for MainWindow."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from eyegen import DEFAULT_CONFIG
from eyegen.config import Backend
from eyegen.gui.constants import DIMENSION_PRESETS


class MainWindowSettingsMixin:
    def _build_settings_panel(self) -> QGroupBox:
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(8)

        self._build_steps_row(settings_layout)
        self._build_guidance_row(settings_layout)
        self._build_width_row(settings_layout)
        self._build_height_row(settings_layout)
        self._build_seed_row(settings_layout)
        self._build_t5_row(settings_layout)
        self._build_model_row(settings_layout)
        self._build_backend_row(settings_layout)
        self._build_quantize_row(settings_layout)
        self._build_model_path_row(settings_layout)
        self._build_backend_hint(settings_layout)
        self._build_bonsai_row(settings_layout)
        self._build_coreml_row(settings_layout)
        self._build_hf_cache_row(settings_layout)

        return settings_group

    def _build_steps_row(self, layout: QVBoxLayout):
        row = QHBoxLayout()
        row.addWidget(QLabel("Steps"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 100)
        self.steps_spin.setValue(self.config.get("num_inference_steps", 30))
        row.addWidget(self.steps_spin)
        layout.addLayout(row)

        self.steps_slider = QSlider(Qt.Horizontal)
        self.steps_slider.setRange(1, 100)
        self.steps_slider.setValue(self.steps_spin.value())
        self.steps_slider.valueChanged.connect(self.steps_spin.setValue)
        self.steps_spin.valueChanged.connect(self.steps_slider.setValue)
        layout.addWidget(self.steps_slider)

    def _build_guidance_row(self, layout: QVBoxLayout):
        row = QHBoxLayout()
        row.addWidget(QLabel("Guidance"))
        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(1.0, 15.0)
        self.guidance_spin.setSingleStep(0.5)
        self.guidance_spin.setValue(self.config.get("guidance_scale", 7.5))
        row.addWidget(self.guidance_spin)
        layout.addLayout(row)

        self.guidance_slider = QSlider(Qt.Horizontal)
        self.guidance_slider.setRange(10, 150)
        self.guidance_slider.setValue(int(self.guidance_spin.value() * 10))
        self.guidance_slider.valueChanged.connect(lambda v: self.guidance_spin.setValue(v / 10.0))
        self.guidance_spin.valueChanged.connect(
            lambda v: self.guidance_slider.setValue(int(v * 10))
        )
        layout.addWidget(self.guidance_slider)

    def _build_width_row(self, layout: QVBoxLayout):
        row = QHBoxLayout()
        row.addWidget(QLabel("Width"))
        self.width_combo = QComboBox()
        for d in DIMENSION_PRESETS:
            self.width_combo.addItem(str(d), d)
        self.width_combo.setCurrentText(str(self.config.get("width", 1024)))
        row.addWidget(self.width_combo)
        layout.addLayout(row)

    def _build_height_row(self, layout: QVBoxLayout):
        row = QHBoxLayout()
        row.addWidget(QLabel("Height"))
        self.height_combo = QComboBox()
        for d in DIMENSION_PRESETS:
            self.height_combo.addItem(str(d), d)
        self.height_combo.setCurrentText(str(self.config.get("height", 1024)))
        row.addWidget(self.height_combo)
        layout.addLayout(row)

    def _build_seed_row(self, layout: QVBoxLayout):
        row = QHBoxLayout()
        row.addWidget(QLabel("Seed"))
        self.seed_input = QLineEdit()
        self.seed_input.setPlaceholderText("Random")
        row.addWidget(self.seed_input)
        layout.addLayout(row)

    def _build_t5_row(self, layout: QVBoxLayout):
        self.t5_check = QCheckBox("Use T5 encoder (better quality, slower)")
        self.t5_check.setChecked(True)
        layout.addWidget(self.t5_check)

    def _build_model_row(self, layout: QVBoxLayout):
        layout.addWidget(QLabel("Model"))
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
        layout.addLayout(model_row)

    def _build_backend_row(self, layout: QVBoxLayout):
        row = QHBoxLayout()
        row.addWidget(QLabel("Backend"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Auto", Backend.AUTO)
        self.backend_combo.addItem("MLX (diffusionkit)", Backend.MLX)
        self.backend_combo.addItem("MFLUX (FLUX/FIBO/Z-Image)", Backend.MFLUX)
        self.backend_combo.addItem("OllamaDiffuser (GGUF)", Backend.OLLAMA)
        self.backend_combo.addItem("Bonsai (PrismML ternary 1.58-bit)", Backend.BONSAI)
        self.backend_combo.addItem("CoreML (Apple Neural Engine)", Backend.COREML)
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        row.addWidget(self.backend_combo)
        layout.addLayout(row)

    def _build_quantize_row(self, layout: QVBoxLayout):
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
        layout.addWidget(self.quantize_row)

    def _build_model_path_row(self, layout: QVBoxLayout):
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
        layout.addWidget(self.model_path_row)

    def _build_backend_hint(self, layout: QVBoxLayout):
        self.backend_hint = QLabel()
        self.backend_hint.setWordWrap(True)
        self.backend_hint.setProperty("class", "hint")
        self.backend_hint.hide()
        layout.addWidget(self.backend_hint)

    def _build_bonsai_row(self, layout: QVBoxLayout):
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
        layout.addWidget(self.bonsai_row)

    def _build_coreml_row(self, layout: QVBoxLayout):
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
        layout.addWidget(self.coreml_row)

    def _build_hf_cache_row(self, layout: QVBoxLayout):
        layout.addWidget(QLabel("HF Cache Dir"))
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
        layout.addLayout(hf_cache_row)
