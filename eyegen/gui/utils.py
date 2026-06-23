"""GUI utility helpers."""

import io
import logging

from PySide6.QtGui import QImage, QPixmap

log = logging.getLogger("eyegen")



def pil_to_pixmap(pil_image) -> QPixmap:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    qimg = QImage()
    qimg.loadFromData(buf.read())
    return QPixmap.fromImage(qimg)
