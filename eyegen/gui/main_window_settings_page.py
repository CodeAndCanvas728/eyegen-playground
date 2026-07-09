"""Settings page builder mixin for MainWindow.

This is separate from main_window_settings.py (which handles backend-specific widgets).
This module builds the outer Settings page layout.
"""

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class MainWindowSettingsPageMixin:
    def _build_settings_page(self):  # noqa: PLR0915
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(16, 16, 16, 16)
        page_layout.setSpacing(16)

        theme_card = QWidget()
        theme_card.setProperty("card", True)
        theme_card_l = QVBoxLayout(theme_card)
        theme_card_l.setContentsMargins(0, 0, 0, 0)
        theme_card_l.setSpacing(8)
        appearance_label = QLabel("Appearance")
        appearance_label.setProperty("class", "section-heading")
        theme_card_l.addWidget(appearance_label)
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Dark")
        self.theme_combo.addItem("Light")
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        theme_card_l.addWidget(self.theme_combo)
        dim_row = QHBoxLayout()
        width_label = QLabel("Width")
        width_label.setProperty("class", "input-label")
        dim_row.addWidget(width_label)
        self.width_combo = QComboBox()
        dim_row.addWidget(self.width_combo)
        height_label = QLabel("Height")
        height_label.setProperty("class", "input-label")
        dim_row.addWidget(height_label)
        self.height_combo = QComboBox()
        dim_row.addWidget(self.height_combo)
        dimensions_label = QLabel("Dimensions")
        dimensions_label.setProperty("class", "section-heading")
        theme_card_l.addWidget(dimensions_label)
        theme_card_l.addLayout(dim_row)
        page_layout.addWidget(theme_card)

        self.settings_page_widget = QWidget()
        self.settings_page_layout = QVBoxLayout(self.settings_page_widget)
        self.settings_page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(self.settings_page_widget)

        self._build_backend_hint(page_layout)

        account_card = QWidget()
        account_card.setProperty("card", True)
        account_card_l = QVBoxLayout(account_card)
        account_card_l.setContentsMargins(0, 0, 0, 0)
        account_card_l.setSpacing(8)
        account_label = QLabel("Account")
        account_label.setProperty("class", "section-heading")
        account_card_l.addWidget(account_label)
        self.hf_btn = QPushButton("🔑  HuggingFace Login")
        self.hf_btn.setToolTip("Log in to download gated models (e.g. FLUX.1-Kontext)")
        self.hf_btn.clicked.connect(self._on_hf_login)
        account_card_l.addWidget(self.hf_btn)
        self._refresh_hf_button()

        self.output_label = QLabel("")
        self.output_label.setWordWrap(True)
        self.output_label.setProperty("hint", True)
        account_card_l.addWidget(self.output_label)
        page_layout.addWidget(account_card)

        page_layout.addStretch()

        self.content_stack.addWidget(page)
