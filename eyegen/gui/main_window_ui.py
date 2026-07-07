"""Main window UI layout builder mixin."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabBar,
    QVBoxLayout,
    QWidget,
)


class MainWindowUIMixin:
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        nav_bg = QWidget()
        nav_bg.setProperty("nav", True)
        nav_layout = QVBoxLayout(nav_bg)
        nav_layout.setContentsMargins(12, 0, 12, 0)
        self.nav_bar = QTabBar()
        self.nav_bar.setProperty("nav", True)
        self.nav_bar.addTab("Home")
        self.nav_bar.addTab("History")
        self.nav_bar.addTab("Settings")
        self.nav_bar.setExpanding(False)
        self.nav_bar.currentChanged.connect(self._on_nav_changed)
        nav_layout.addWidget(self.nav_bar)
        root.addWidget(nav_bg)

        self.content_stack = QStackedWidget()
        root.addWidget(self.content_stack, 1)

        self._build_home_page()
        self._build_history_page()
        self._build_settings_page()

        self._current_pixmap = None

    def _on_nav_changed(self, index: int):
        self.content_stack.setCurrentIndex(index)
        if index == 0:
            self._scale_preview()

    def _build_home_page(self):  # noqa: PLR0915
        home = QWidget()
        home_layout = QHBoxLayout(home)
        home_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        home_layout.addWidget(splitter)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setMaximumWidth(420)
        controls_scroll.setMinimumWidth(300)

        controls_container = QWidget()
        self.controls_layout = QVBoxLayout(controls_container)
        self.controls_layout.setContentsMargins(8, 8, 8, 8)
        self.controls_layout.setSpacing(8)

        self.mode_tabs = QTabBar()
        self.mode_tabs.addTab("Text to Image")
        self.mode_tabs.addTab("Image to Image")
        self.mode_tabs.currentChanged.connect(self._on_mode_changed)
        self.controls_layout.addWidget(self.mode_tabs)

        self.img2img_controls = self._build_img2img_controls()
        self.controls_layout.addWidget(self.img2img_controls)

        prompt_card = QWidget()
        prompt_card.setProperty("card", True)
        prompt_card_layout = QVBoxLayout(prompt_card)
        prompt_card_layout.setContentsMargins(0, 0, 0, 0)
        prompt_card_layout.setSpacing(8)
        self._build_prompt_section(prompt_card_layout)
        self.controls_layout.addWidget(prompt_card)

        self.generate_btn = QPushButton("✨  Generate")
        self.generate_btn.setMinimumHeight(44)
        self.generate_btn.setProperty("gradient", True)
        self.generate_btn.setToolTip("Generate (Cmd+Return / Ctrl+Return)")
        self.generate_btn.clicked.connect(self._on_generate)
        self._bind_generate_shortcut(self.prompt_input)
        self._bind_generate_shortcut(self.negative_prompt_input)
        self.controls_layout.addWidget(self.generate_btn)

        self._build_status_section(self.controls_layout)

        self._build_advanced_settings(self.controls_layout)

        bottom_card = QWidget()
        bottom_card.setProperty("card", True)
        bottom_card_layout = QVBoxLayout(bottom_card)
        bottom_card_layout.setContentsMargins(0, 0, 0, 0)
        bottom_card_layout.setSpacing(8)
        self._build_model_backend_seed(bottom_card_layout)
        bottom_card_layout.addStretch()
        self.controls_layout.addWidget(bottom_card)

        preview = self._build_preview_panel()

        splitter.addWidget(controls_scroll)
        splitter.addWidget(preview)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        controls_scroll.setWidget(controls_container)

        self.content_stack.addWidget(home)

    def _build_advanced_settings(self, layout):
        self.advanced_toggle = QPushButton("▼  Advanced Settings")
        self.advanced_toggle.setFlat(True)
        self.advanced_toggle.setCursor(Qt.PointingHandCursor)
        self.advanced_toggle.clicked.connect(self._toggle_advanced)
        layout.addWidget(self.advanced_toggle)

        self.advanced_container = QWidget()
        self.advanced_container.setVisible(False)
        advanced_layout = QVBoxLayout(self.advanced_container)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(6)

        steps_row = QHBoxLayout()
        steps_row.addWidget(QLabel("Steps"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 100)
        self.steps_spin.setValue(self.config.num_inference_steps)
        steps_row.addWidget(self.steps_spin)
        advanced_layout.addLayout(steps_row)

        self.steps_slider = QSlider(Qt.Horizontal)
        self.steps_slider.setRange(1, 100)
        self.steps_slider.setValue(self.steps_spin.value())
        self.steps_slider.valueChanged.connect(self.steps_spin.setValue)
        self.steps_spin.valueChanged.connect(self.steps_slider.setValue)
        advanced_layout.addWidget(self.steps_slider)

        guidance_row = QHBoxLayout()
        guidance_row.addWidget(QLabel("Guidance"))
        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(1.0, 15.0)
        self.guidance_spin.setSingleStep(0.5)
        self.guidance_spin.setValue(self.config.guidance_scale)
        guidance_row.addWidget(self.guidance_spin)
        advanced_layout.addLayout(guidance_row)

        self.guidance_slider = QSlider(Qt.Horizontal)
        self.guidance_slider.setRange(10, 150)
        self.guidance_slider.setValue(int(self.guidance_spin.value() * 10))
        self.guidance_slider.valueChanged.connect(lambda v: self.guidance_spin.setValue(v / 10.0))
        self.guidance_spin.valueChanged.connect(
            lambda v: self.guidance_slider.setValue(int(v * 10))
        )
        advanced_layout.addWidget(self.guidance_slider)

        self._build_t5_row(advanced_layout)

        layout.addWidget(self.advanced_container)

    def _toggle_advanced(self):
        visible = not self.advanced_container.isVisible()
        self.advanced_container.setVisible(visible)
        self.advanced_toggle.setText(("▲" if visible else "▼") + "  Advanced Settings")

    def _build_t5_row(self, layout):
        self.t5_check = QCheckBox("Use T5 encoder (better quality, slower)")
        self.t5_check.setChecked(True)
        layout.addWidget(self.t5_check)

    def _build_model_backend_seed(self, layout):
        layout.addWidget(QLabel("Model"))
        model_row = QHBoxLayout()
        self.model_stack = QStackedWidget()
        self.model_stack.setMaximumWidth(310)

        self.model_dropdown = QComboBox()
        self.model_dropdown.setToolTip("Select a discovered model or 'Custom...' to type your own")
        self.model_dropdown.currentIndexChanged.connect(self._on_model_dropdown_changed)
        self.model_stack.addWidget(self.model_dropdown)

        self.model_input = QLineEdit()
        self.model_input.setText(self.config.model)
        self.model_input.setToolTip("Hugging Face model ID or OllamaDiffuser model name")
        self.model_input.editingFinished.connect(self._update_backend_dependent_controls)
        self.model_stack.addWidget(self.model_input)

        model_row.addWidget(self.model_stack)

        self.pull_btn = QPushButton("Pull…")
        self.pull_btn.setFixedWidth(50)
        self.pull_btn.setToolTip("Download this GGUF model via OllamaDiffuser")
        self.pull_btn.clicked.connect(self._on_pull_model)
        model_row.addWidget(self.pull_btn)
        layout.addLayout(model_row)

        layout.addWidget(QLabel("Backend"))
        self.backend_combo = QComboBox()
        self._populate_backend_combo()
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        layout.addWidget(self.backend_combo)

        layout.addWidget(QLabel("Seed"))
        self.seed_input = QLineEdit()
        self.seed_input.setPlaceholderText("Random")
        layout.addWidget(self.seed_input)

    def _populate_backend_combo(self):
        from eyegen.config import Backend

        self.backend_combo.addItem("Auto", Backend.AUTO)
        self.backend_combo.addItem("MLX (diffusionkit)", Backend.MLX)
        self.backend_combo.addItem("MFLUX (FLUX/FIBO/Z-Image)", Backend.MFLUX)
        self.backend_combo.addItem("OllamaDiffuser (GGUF)", Backend.OLLAMA)
        self.backend_combo.addItem("Bonsai (PrismML ternary 1.58-bit)", Backend.BONSAI)
        self.backend_combo.addItem("CoreML (Apple Neural Engine)", Backend.COREML)


