"""Settings panel builder mixin for MainWindow.

Widgets now created in main_window_ui.py (Home page):
  steps_spin/slider, guidance_spin/slider, t5_check,
  model_stack/dropdown/input, pull_btn, backend_combo, seed_input,
  width_combo, height_combo, theme_combo.

This module creates the backend-specific config widgets (quantize, model_path,
bonsai, coreml, hf_cache) and places them in self.settings_page_layout.
"""

import logging

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from eyegen import DEFAULT_CONFIG
from eyegen.gui.constants import DIMENSION_PRESETS

log = logging.getLogger("eyegen")


class MainWindowSettingsMixin:
    def _build_backend_settings(self):
        """Build backend-specific config widgets in the Settings tab."""
        layout = self.settings_page_layout

        card = QWidget()
        card.setProperty("card", True)
        card_l = QVBoxLayout(card)
        card_l.setContentsMargins(0, 0, 0, 0)
        card_l.setSpacing(8)
        backend_config_label = QLabel("Backend Configuration")
        backend_config_label.setProperty("class", "section-heading")
        card_l.addWidget(backend_config_label)

        # Quantize
        self._build_quantize_row(card_l)

        # Saved model path
        self._build_model_path_row(card_l)

        # Bonsai
        self._build_bonsai_row(card_l)

        # CoreML
        self._build_coreml_row(card_l)

        # HF Cache
        self._build_hf_cache_row(card_l)

        layout.addWidget(card)

        # Populate dimension combos
        for combo in (self.width_combo, self.height_combo):
            combo.blockSignals(True)
        for d in DIMENSION_PRESETS:
            self.width_combo.addItem(str(d), d)
            self.height_combo.addItem(str(d), d)
        self.width_combo.setCurrentText(str(DEFAULT_CONFIG.width))
        self.height_combo.setCurrentText(str(DEFAULT_CONFIG.height))
        for combo in (self.width_combo, self.height_combo):
            combo.blockSignals(False)

    def _build_quantize_row(self, layout):
        self.quantize_row = QWidget()
        q_layout = QHBoxLayout(self.quantize_row)
        q_layout.setContentsMargins(0, 0, 0, 0)
        quantize_label = QLabel("Quantize")
        quantize_label.setProperty("class", "input-label")
        q_layout.addWidget(quantize_label)
        self.quantize_combo = QComboBox()
        self.quantize_combo.addItem("4-bit (recommended)", 4)
        self.quantize_combo.addItem("8-bit", 8)
        self.quantize_combo.addItem("None (full precision)", 0)
        self.quantize_combo.setToolTip("MFLUX runtime quantization level")
        q_layout.addWidget(self.quantize_combo)
        layout.addWidget(self.quantize_row)

    def _build_model_path_row(self, layout):
        self.model_path_row = QWidget()
        mp_layout = QVBoxLayout(self.model_path_row)
        mp_layout.setContentsMargins(0, 0, 0, 0)
        mp_top = QHBoxLayout()
        saved_model_label = QLabel("Saved model")
        saved_model_label.setProperty("class", "input-label")
        mp_top.addWidget(saved_model_label)
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
        layout.addWidget(self.model_path_row)

    def _build_bonsai_row(self, layout):
        self.bonsai_row = QWidget()
        bonsai_layout = QHBoxLayout(self.bonsai_row)
        bonsai_layout.setContentsMargins(0, 0, 0, 0)
        self.bonsai_status_label = QLabel()
        self.bonsai_status_label.setProperty("class", "hint")
        self.bonsai_status_label.setWordWrap(True)
        bonsai_layout.addWidget(self.bonsai_status_label, 1)
        self.bonsai_setup_btn = QPushButton("Setup Bonsai\u2026")
        self.bonsai_setup_btn.setFixedWidth(130)
        self.bonsai_setup_btn.setToolTip(
            "One-time install: clones the Bonsai-Image-Demo repo and runs its\n"
            "setup.sh, which creates a dedicated Python 3.11 venv with the\n"
            "patched mflux + MLX kernels needed for ternary 1.58-bit weights."
        )
        self.bonsai_setup_btn.clicked.connect(self._on_bonsai_setup)
        bonsai_layout.addWidget(self.bonsai_setup_btn)
        self.bonsai_pull_btn = QPushButton("Download Model\u2026")
        self.bonsai_pull_btn.setFixedWidth(140)
        self.bonsai_pull_btn.setToolTip(
            "Download a bonsai model via the bonsai-demo's download script."
        )
        self.bonsai_pull_btn.clicked.connect(self._on_bonsai_pull)
        bonsai_layout.addWidget(self.bonsai_pull_btn)
        layout.addWidget(self.bonsai_row)

    def _build_coreml_row(self, layout):
        self.coreml_row = QWidget()
        coreml_layout = QHBoxLayout(self.coreml_row)
        coreml_layout.setContentsMargins(0, 0, 0, 0)
        self.coreml_status_label = QLabel()
        self.coreml_status_label.setProperty("class", "hint")
        self.coreml_status_label.setWordWrap(True)
        coreml_layout.addWidget(self.coreml_status_label, 1)
        self.coreml_setup_btn = QPushButton("Setup CoreML\u2026")
        self.coreml_setup_btn.setFixedWidth(130)
        self.coreml_setup_btn.setToolTip(
            "One-time install: creates a sidecar Python 3.11 venv at\n"
            "~/models/eyegen/.coreml-venv/ with Apple's python_coreml_stable_diffusion."
        )
        self.coreml_setup_btn.clicked.connect(self._on_coreml_setup)
        coreml_layout.addWidget(self.coreml_setup_btn)
        self.coreml_pull_btn = QPushButton("Download Model\u2026")
        self.coreml_pull_btn.setFixedWidth(140)
        self.coreml_pull_btn.setToolTip(
            "Download a pre-converted CoreML model from Hugging Face.\n"
            "Or use ./generate.py convert-coreml to convert a PyTorch model from scratch."
        )
        self.coreml_pull_btn.clicked.connect(self._on_coreml_pull)
        coreml_layout.addWidget(self.coreml_pull_btn)
        layout.addWidget(self.coreml_row)

    def _build_hf_cache_row(self, layout):
        hf_cache_label = QLabel("HF Cache Dir")
        hf_cache_label.setProperty("class", "input-label")
        layout.addWidget(hf_cache_label)
        hf_cache_row = QHBoxLayout()
        self.hf_cache_input = QLineEdit()
        self.hf_cache_input.setPlaceholderText("Default (~/.cache/huggingface/hub)")
        self.hf_cache_input.setToolTip(
            "Directory where HuggingFace caches downloaded model weights.\n"
            "Leave blank to use the default (~/.cache/huggingface/hub)."
        )
        hf_cache_row.addWidget(self.hf_cache_input)
        browse_btn = QPushButton("Browse\u2026")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._on_browse_hf_cache)
        hf_cache_row.addWidget(browse_btn)
        layout.addLayout(hf_cache_row)

    def _build_backend_hint(self, layout):
        """Create the backend hint label."""
        self.backend_hint = QLabel()
        self.backend_hint.setWordWrap(True)
        self.backend_hint.setProperty("class", "hint")
        self.backend_hint.hide()
        layout.addWidget(self.backend_hint)
