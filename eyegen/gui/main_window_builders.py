"""Home page sub-builders and preview panel for MainWindow."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QLabel,
    QProgressBar,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class MainWindowBuildersMixin:
    def _build_status_section(self, layout):
        self.status_label = QLabel("Ready")
        self.status_label.setProperty("hint", True)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(6)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

    def _build_prompt_section(self, layout):
        self.prompt_input, prompt_container = self._build_prompt_input(
            "Prompt", "Describe the image you want to generate...", 120
        )
        layout.addWidget(prompt_container)

        self.negative_prompt_input, neg_container = self._build_prompt_input(
            "Negative Prompt", "What to avoid (optional)...", 60
        )
        layout.addWidget(neg_container)

    def _build_prompt_input(self, label, placeholder, max_height):
        wrapper = QVBoxLayout()
        label_widget = QLabel(label)
        label_widget.setProperty("class", "section-heading")
        wrapper.addWidget(label_widget)
        input_field = QTextEdit()
        input_field.setPlaceholderText(placeholder)
        input_field.setMaximumHeight(max_height)
        wrapper.addWidget(input_field)
        container = QWidget()
        container.setLayout(wrapper)
        return input_field, container

    def _bind_generate_shortcut(self, widget):
        shortcut = QShortcut(QKeySequence("Ctrl+Return"), widget, self._on_generate)
        shortcut.setContext(Qt.WidgetWithChildrenShortcut)

    def _build_preview_panel(self):
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel("No image yet")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setProperty("class", "preview-placeholder")
        self.image_label.setMinimumSize(400, 400)
        preview_layout.addWidget(self.image_label)

        return preview
