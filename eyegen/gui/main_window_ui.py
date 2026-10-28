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
        # The name is what the stylesheet paints: without it the page shows
        # the platform's unpainted window colour behind the content.
        central.setObjectName("pageHost")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        nav_bg = QWidget()
        nav_bg.setProperty("nav", True)
        nav_layout = QVBoxLayout(nav_bg)
        nav_layout.setContentsMargins(16, 0, 16, 0)
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
        labels = ["Home", "History", "Settings"]
        self.setWindowTitle(f"EyeGen — {labels[index]}")
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
        controls_scroll.setFrameShape(QScrollArea.NoFrame)
        controls_scroll.setMaximumWidth(500)
        controls_scroll.setMinimumWidth(380)

        controls_container = QWidget()
        self.controls_layout = QVBoxLayout(controls_container)
        self.controls_layout.setContentsMargins(28, 28, 28, 28)
        self.controls_layout.setSpacing(22)

        self.mode_tabs = QTabBar()
        self.mode_tabs.setProperty("segment", True)
        self.mode_tabs.addTab("Text to Image")
        self.mode_tabs.addTab("Image to Image")
        self.mode_tabs.currentChanged.connect(self._on_mode_changed)
        self.controls_layout.addWidget(self.mode_tabs)

        self.img2img_controls = self._build_img2img_controls()
        self.controls_layout.addWidget(self.img2img_controls)

        # A section is headed and spaced, not boxed: the ground is one
        # surface, and space is what separates one thing from another.
        self._build_prompt_section(self.controls_layout)

        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setMinimumHeight(48)
        self.generate_btn.setProperty("gradient", True)
        self.generate_btn.setToolTip("Generate (Cmd+Return / Ctrl+Return)")
        self.generate_btn.clicked.connect(self._on_generate)
        self._bind_generate_shortcut(self.prompt_input)
        self._bind_generate_shortcut(self.negative_prompt_input)
        self.controls_layout.addWidget(self.generate_btn)

        self._build_status_section(self.controls_layout)

        self._build_advanced_settings(self.controls_layout)

        self._build_model_backend_seed(self.controls_layout)
        self.controls_layout.addStretch()

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
        advanced_layout.setSpacing(12)

        steps_row = QHBoxLayout()
        steps_label = QLabel("Steps")
        steps_label.setProperty("class", "input-label")
        steps_row.addWidget(steps_label)
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
        guidance_label = QLabel("Guidance")
        guidance_label.setProperty("class", "input-label")
        guidance_row.addWidget(guidance_label)
        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(1.0, 15.0)
        self.guidance_spin.setSingleStep(0.5)
        self.guidance_spin.setValue(self.config.guidance_scale)
        guidance_row.addWidget(self.guidance_spin)
        advanced_layout.addLayout(guidance_row)

        self._build_guidance_slider(advanced_layout)

        self._build_t5_row(advanced_layout)

        layout.addWidget(self.advanced_container)

    def _toggle_advanced(self):
        visible = not self.advanced_container.isVisible()
        self.advanced_container.setVisible(visible)
        self.advanced_toggle.setText(("▲" if visible else "▼") + "  Advanced Settings")

    def _build_t5_row(self, layout):
        self.t5_check = QCheckBox("Use T5 encoder (better quality, slower)")
        self.t5_check.setChecked(self.config.use_t5)
        layout.addWidget(self.t5_check)

    def _build_guidance_slider(self, advanced_layout):
        self.guidance_slider = QSlider(Qt.Horizontal)
        self.guidance_slider.setRange(10, 150)
        self.guidance_slider.setValue(int(self.guidance_spin.value() * 10))
        self.guidance_slider.valueChanged.connect(lambda v: self.guidance_spin.setValue(v / 10.0))
        self.guidance_spin.valueChanged.connect(
            lambda v: self.guidance_slider.setValue(int(v * 10))
        )
        advanced_layout.addWidget(self.guidance_slider)

    def _build_model_backend_seed(self, layout):
        layout.addSpacing(8)
        self._add_section_label(layout, "Model")
        self.model_stack = QStackedWidget()

        self.model_dropdown = QComboBox()
        self.model_dropdown.setToolTip("Select a discovered model or 'Custom...' to type your own")
        self.model_dropdown.currentIndexChanged.connect(self._on_model_dropdown_changed)
        self.model_stack.addWidget(self.model_dropdown)

        self.model_input = QLineEdit()
        self.model_input.setText(self.config.model)
        self.model_input.setToolTip(
            "Type a model name, or a path to a model file or folder on this machine"
        )
        self.model_input.editingFinished.connect(self._update_backend_dependent_controls)
        self.model_stack.addWidget(self.model_input)

        layout.addWidget(self.model_stack)
        layout.addLayout(self._build_model_actions())

        layout.addSpacing(8)
        self._add_section_label(layout, "Backend")
        self.backend_combo = QComboBox()
        self._populate_backend_combo()
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        layout.addWidget(self.backend_combo)

        layout.addSpacing(8)
        self._add_section_label(layout, "Seed")
        self.seed_input = QLineEdit()
        self.seed_input.setPlaceholderText("Random")
        layout.addWidget(self.seed_input)

        layout.addSpacing(8)
        self.low_memory_check = QCheckBox("Low Memory mode")
        self.low_memory_check.setChecked(self.config.sdcli_offload_to_cpu)
        self.low_memory_check.setToolTip(
            "Offloads the model's weights to the CPU instead of keeping them in "
            "GPU memory, for edge devices.\n"
            "Trade-off: generation is slower, because every layer is copied from "
            "RAM to the GPU as it is needed."
        )
        self.low_memory_check.toggled.connect(self._on_low_memory_toggled)
        layout.addWidget(self.low_memory_check)

    def _on_low_memory_toggled(self, checked: bool):
        """Low Memory mode is a per-run preference; it lives in the run's config."""
        self.config.sdcli_offload_to_cpu = bool(checked)

    def _add_section_label(self, layout, text):
        label = QLabel(text)
        label.setProperty("class", "section-heading")
        layout.addWidget(label)

    def _build_model_actions(self):
        """The little actions under the field, not squeezed beside it:
        a long path or a long name may use the whole width."""
        actions_row = QHBoxLayout()
        actions_row.setSpacing(10)
        self.browse_model_btn = QPushButton("Browse Model…")
        self.browse_model_btn.setCursor(Qt.PointingHandCursor)
        self.browse_model_btn.setToolTip("Choose a model file or folder on this machine")
        self.browse_model_btn.clicked.connect(self._on_browse_model)
        actions_row.addWidget(self.browse_model_btn)

        self.pull_btn = QPushButton("Download…")
        self.pull_btn.setCursor(Qt.PointingHandCursor)
        self.pull_btn.setToolTip("Download this GGUF model via OllamaDiffuser")
        self.pull_btn.clicked.connect(self._on_pull_model)
        actions_row.addWidget(self.pull_btn)
        actions_row.addStretch()
        return actions_row

    def _populate_backend_combo(self):
        from eyegen.config import Backend

        self.backend_combo.addItem("Auto", Backend.AUTO)
        self.backend_combo.addItem("MLX (diffusionkit)", Backend.MLX)
        self.backend_combo.addItem("MFLUX (FLUX/FIBO/Z-Image)", Backend.MFLUX)
        self.backend_combo.addItem("OllamaDiffuser (GGUF)", Backend.OLLAMA)
        self.backend_combo.addItem("Bonsai (PrismML ternary 1.58-bit)", Backend.BONSAI)
        self.backend_combo.addItem("CoreML (Apple Neural Engine)", Backend.COREML)
        self.backend_combo.addItem("stable-diffusion.cpp (FLUX.3)", Backend.SDCLI)
