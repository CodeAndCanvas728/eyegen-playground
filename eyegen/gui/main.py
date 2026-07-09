"""EyeGen GUI entry point."""

import logging
import sys
from pathlib import Path

from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication

from eyegen import CONFIG_DIR
from eyegen.gui.main_window import MainWindow

LOG_FILE = CONFIG_DIR / "eyegen.log"


def main():
    log = logging.getLogger("eyegen")
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )

    log.info(
        "EyeGen GUI starting (Python %s, arch=%s)",
        sys.version.split()[0],
        __import__("platform").machine(),
    )
    app = QApplication(sys.argv)
    app.setApplicationName("EyeGen")
    if sys.platform == "darwin":
        app.setFont(QFont(".AppleSystemUIFont", 13))

    icon_path = Path(__file__).resolve().parent.parent.parent / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
