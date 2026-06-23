"""Main window UI layout builder mixin."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class MainWindowUIMixin:
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        controls = self._build_controls_panel()
        preview = self._build_preview_panel()

        splitter.addWidget(controls)
        splitter.addWidget(preview)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._current_pixmap = None

    def _build_controls_panel(self) -> QWidget:
        controls = QWidget()
        controls.setMaximumWidth(420)
        controls.setMinimumWidth(300)
        ctrl_layout = QVBoxLayout(controls)
        ctrl_layout.setContentsMargins(8, 8, 8, 8)
        ctrl_layout.setSpacing(8)

        self.mode_tabs = QTabBar()
        self.mode_tabs.addTab("Text to Image")
        self.mode_tabs.addTab("Image to Image")
        self.mode_tabs.currentChanged.connect(self._on_mode_changed)
        ctrl_layout.addWidget(self.mode_tabs)

        self.img2img_controls = self._build_img2img_controls()
        ctrl_layout.addWidget(self.img2img_controls)

        self._build_prompt_section(ctrl_layout)
        self.generate_btn = QPushButton("✨  Generate")
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.setToolTip("Generate (Cmd+Return / Ctrl+Return)")
        self.generate_btn.clicked.connect(self._on_generate)
        self._bind_generate_shortcut(self.prompt_input)
        self._bind_generate_shortcut(self.negative_prompt_input)
        ctrl_layout.addWidget(self.generate_btn)

        self._build_status_section(ctrl_layout)

        settings_group = self._build_settings_panel()
        ctrl_layout.addWidget(settings_group)

        self.hf_btn = QPushButton("🔑  HuggingFace Login")
        self.hf_btn.setToolTip("Log in to download gated models (e.g. FLUX.1-Kontext)")
        self.hf_btn.clicked.connect(self._on_hf_login)
        ctrl_layout.addWidget(self.hf_btn)
        self._refresh_hf_button()

        ctrl_layout.addStretch()

        self.output_label = QLabel("")
        self.output_label.setWordWrap(True)
        self.output_label.setProperty("class", "hint")
        self.output_label.setStyleSheet("color: gray;")
        ctrl_layout.addWidget(self.output_label)

        return controls

    def _build_status_section(self, ctrl_layout: QVBoxLayout):
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray;")
        ctrl_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(8)
        self.progress_bar.hide()
        ctrl_layout.addWidget(self.progress_bar)

    def _build_prompt_section(self, ctrl_layout: QVBoxLayout):
        self.prompt_input, prompt_container = self._build_prompt_input(
            "Prompt", "Describe the image you want to generate...", 120
        )
        ctrl_layout.addWidget(prompt_container)

        self.negative_prompt_input, neg_container = self._build_prompt_input(
            "Negative Prompt", "What to avoid (optional)...", 60
        )
        ctrl_layout.addWidget(neg_container)

    def _build_prompt_input(self, label: str, placeholder: str, max_height: int):
        layout = QVBoxLayout()
        layout.addWidget(QLabel(label))
        input_field = QTextEdit()
        input_field.setPlaceholderText(placeholder)
        input_field.setMaximumHeight(max_height)
        layout.addWidget(input_field)
        container = QWidget()
        container.setLayout(layout)
        return input_field, container

    def _bind_generate_shortcut(self, widget):
        shortcut = QShortcut(QKeySequence("Ctrl+Return"), widget, self._on_generate)
        shortcut.setContext(Qt.WidgetWithChildrenShortcut)

    def _build_preview_panel(self) -> QWidget:
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel("No image yet")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setProperty("class", "display")
        self.image_label.setStyleSheet(
            "background-color: #1e1e1e; color: #888; border-radius: 8px;"
        )
        preview_layout.addWidget(self.image_label)

        return preview
