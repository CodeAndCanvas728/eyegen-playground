"""GUI utility helpers."""

import io
import logging
import time
from typing import Optional

from PySide6.QtGui import QImage, QPixmap

from eyegen import hf_status

log = logging.getLogger("eyegen")

def _cached_hf_status() -> Optional[dict]:
    """Return cached hf_status."""
    return hf_status()


def pil_to_pixmap(pil_image) -> QPixmap:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    qimg = QImage()
    qimg.loadFromData(buf.read())
    return QPixmap.fromImage(qimg)
