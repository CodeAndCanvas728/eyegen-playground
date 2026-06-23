"""GUI utility helpers."""

import io
import logging
import time
from typing import Optional

from PySide6.QtGui import QImage, QPixmap

from eyegen import hf_status

log = logging.getLogger("eyegen")

_HF_STATUS_CACHE: dict = {"result": None, "timestamp": 0.0}
_HF_CACHE_TTL = 30.0  # seconds


def _cached_hf_status() -> Optional[dict]:
    """Return cached hf_status, refreshing from network every _HF_CACHE_TTL seconds."""
    now = time.monotonic()
    cache_age = now - _HF_STATUS_CACHE["timestamp"]
    if cache_age < _HF_CACHE_TTL and _HF_STATUS_CACHE["result"] is not None:
        return _HF_STATUS_CACHE["result"]
    result = hf_status()
    _HF_STATUS_CACHE["result"] = result
    _HF_STATUS_CACHE["timestamp"] = now
    return result


def pil_to_pixmap(pil_image) -> QPixmap:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    qimg = QImage()
    qimg.loadFromData(buf.read())
    return QPixmap.fromImage(qimg)
