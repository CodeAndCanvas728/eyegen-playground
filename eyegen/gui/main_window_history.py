"""History page builder mixin for MainWindow."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from eyegen import OUTPUT_DIR


class MainWindowHistoryMixin:
    def _build_history_page(self):
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Generation History")
        title.setProperty("class", "section")
        page_layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.history_grid = QWidget()
        self.history_grid_layout = QVBoxLayout(self.history_grid)
        self.history_grid.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll.setWidget(self.history_grid)
        page_layout.addWidget(scroll, 1)

        self.refresh_history_btn = QPushButton("Refresh History")
        self.refresh_history_btn.clicked.connect(self._refresh_history)
        page_layout.addWidget(self.refresh_history_btn)

        self.content_stack.addWidget(page)
        self._refresh_history()

    def _refresh_history(self):
        while self.history_grid_layout.count():
            item = self.history_grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        outputs = Path(str(OUTPUT_DIR))
        if not outputs.is_dir():
            label = QLabel("No output directory found.")
            label.setProperty("hint", True)
            label.setAlignment(Qt.AlignCenter)
            self.history_grid_layout.addWidget(label)
            return

        images = sorted(
            [
                f
                for f in outputs.iterdir()
                if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            ],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        if not images:
            label = QLabel("No generated images yet. Generate one first!")
            label.setProperty("hint", True)
            label.setAlignment(Qt.AlignCenter)
            self.history_grid_layout.addWidget(label)
            cta = QPushButton("Go Generate")
            cta.setCursor(Qt.PointingHandCursor)
            cta.clicked.connect(lambda: self.nav_bar.setCurrentIndex(0))
            self.history_grid_layout.addWidget(cta, alignment=Qt.AlignCenter)
            return

        row_widget = None
        row_layout = None
        max_per_row = 3

        for idx, img_path in enumerate(images):
            if idx % max_per_row == 0:
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 8)
                self.history_grid_layout.addWidget(row_widget)

            thumb = QLabel()
            thumb.setProperty("thumbnail", True)
            pixmap = QPixmap(str(img_path))
            if not pixmap.isNull():
                thumb.setPixmap(pixmap.scaledToWidth(200, Qt.SmoothTransformation))
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setCursor(Qt.PointingHandCursor)
            thumb.mousePressEvent = lambda e, p=img_path: self._open_history_image(p)
            row_layout.addWidget(thumb)

    def _open_history_image(self, path):
        import subprocess

        subprocess.Popen(["open", "-R", str(path)])  # noqa: S603, S607
