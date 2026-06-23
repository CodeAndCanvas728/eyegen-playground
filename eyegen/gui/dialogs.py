"""GUI dialogs."""

import logging
from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from eyegen import hf_login, hf_logout
from eyegen.gui.utils import _cached_hf_status

log = logging.getLogger("eyegen")


class HFLoginDialog(QDialog):
    """Modal dialog for HuggingFace authentication."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HuggingFace Login")
        self.setMinimumWidth(400)
        self._has_unsaved_token = False

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        self.status_label = QLabel("Checking login status…")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addWidget(QLabel("Access Token"))
        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.Password)
        self.token_input.setPlaceholderText("hf_...")
        self.token_input.textChanged.connect(self._on_token_changed)
        layout.addWidget(self.token_input)

        hint = QLabel(
            '<a href="https://huggingface.co/settings/tokens">'
            "Get a token at huggingface.co/settings/tokens</a>"
        )
        hint.setOpenExternalLinks(True)
        hint.setProperty("class", "hint")
        layout.addWidget(hint)

        btn_row = QHBoxLayout()
        self.login_btn = QPushButton("Login")
        self.login_btn.clicked.connect(self._on_login)
        btn_row.addWidget(self.login_btn)

        self.logout_btn = QPushButton("Logout")
        self.logout_btn.clicked.connect(self._on_logout)
        btn_row.addWidget(self.logout_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._on_close_clicked)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self._refresh_status()

    def _on_token_changed(self, text: str):
        self._has_unsaved_token = bool(text.strip())

    def _on_close_clicked(self):
        self.close()

    def _confirm_discard_token(self) -> bool:
        if not self._has_unsaved_token:
            return True
        parent = self.parentWidget()
        if parent and not parent.isVisible():
            self._has_unsaved_token = False
            return True
        reply = QMessageBox.question(
            self,
            "Discard token?",
            "A token was entered but not submitted. Discard it?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._has_unsaved_token = False
            return True
        return False

    def reject(self):
        if self._confirm_discard_token():
            super().reject()

    def closeEvent(self, event):
        if self._confirm_discard_token():
            self._has_unsaved_token = False
            event.accept()
        else:
            event.ignore()

    def done(self, r):
        self.token_input.clear()
        super().done(r)

    def _refresh_status(self):
        info = _cached_hf_status()
        if info:
            name = info.get("name", "unknown")
            self.status_label.setText(f"✅ Logged in as <b>{name}</b>")
            self.status_label.setStyleSheet("color: green;")
            self.logout_btn.setEnabled(True)
        else:
            self.status_label.setText("Not logged in — enter a token to access gated models")
            self.status_label.setStyleSheet("color: gray;")
            self.logout_btn.setEnabled(False)

    def _on_login(self):
        token = self.token_input.text().strip()
        if not token:
            self.status_label.setText("⚠ Enter a token first")
            self.status_label.setStyleSheet("color: orange;")
            return
        try:
            info = hf_login(token)
            name = info.get("name", "unknown")
            self.status_label.setText(f"✅ Logged in as <b>{name}</b>")
            self.status_label.setStyleSheet("color: green;")
            self.token_input.clear()
            self._has_unsaved_token = False
            self.logout_btn.setEnabled(True)
        except (OSError, ValueError) as e:
            self.status_label.setText(f"❌ Login failed: {e}")
            self.status_label.setStyleSheet("color: red;")

    def _on_logout(self):
        try:
            hf_logout()
        except (OSError, ValueError) as e:
            log.warning("Logout failed: %s", e)
        self.token_input.clear()
        self._refresh_status()

    def get_username(self) -> Optional[str]:
        """Return current HF username if logged in, else None."""
        info = _cached_hf_status()
        return info.get("name") if info else None
