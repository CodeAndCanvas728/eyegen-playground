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
            "Prompt", "Describe the image you want to generate...", 140
        )
        self.negative_prompt_input, neg_container = self._build_prompt_input(
            "Negative Prompt", "What to avoid (optional)...", 80
        )

        # The two boxes share a vertical splitter, so the user can drag the
        # boundary and give the prompt as much room as the window allows.
        from PySide6.QtWidgets import QSplitter

        prompt_splitter = QSplitter(Qt.Vertical)
        prompt_splitter.setChildrenCollapsible(False)
        prompt_splitter.addWidget(prompt_container)
        prompt_splitter.addWidget(neg_container)
        prompt_splitter.setSizes([240, 120])
        prompt_splitter.setMinimumHeight(300)
        prompt_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(prompt_splitter, 1)

    def _build_prompt_input(self, label, placeholder, min_height):
        wrapper = QVBoxLayout()
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.setSpacing(8)
        label_widget = QLabel(label)
        label_widget.setProperty("class", "input-label")
        wrapper.addWidget(label_widget)
        input_field = QTextEdit()
        input_field.setPlaceholderText(placeholder)
        input_field.setMinimumHeight(min_height)
        input_field.setMaximumHeight(min_height * 4)
        input_field.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        wrapper.addWidget(input_field, 1)
        container = QWidget()
        container.setLayout(wrapper)
        return input_field, container

    def _bind_generate_shortcut(self, widget):
        shortcut = QShortcut(QKeySequence("Ctrl+Return"), widget, self._on_generate)
        shortcut.setContext(Qt.WidgetWithChildrenShortcut)

    def _build_preview_panel(self):
        preview = QWidget()
        preview.setProperty("preview", True)
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(28, 28, 28, 28)

        self.image_label = QLabel(
            "Your image will appear here\n\nGenerate from the controls on the left."
        )
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setProperty("class", "preview-placeholder")
        self.image_label.setMinimumSize(400, 400)
        preview_layout.addWidget(self.image_label)

        return preview
