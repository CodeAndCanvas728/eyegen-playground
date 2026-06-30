"""Pipeline cache to avoid reloading models between generations."""

import threading
from typing import Any, Optional, Tuple


class _PipelineCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._pipeline: Optional[Any] = None
        self._key: Optional[Any] = None
        self._generation: int = 0

    def clear(self) -> None:
        with self._lock:
            self._pipeline = None
            self._key = None
            self._generation += 1

    def get_generation(self) -> int:
        with self._lock:
            return self._generation

    def get_cached(self) -> Tuple[Optional[Any], Optional[Any], int]:
        with self._lock:
            return self._pipeline, self._key, self._generation

    def set_cached(self, pipeline: Any, key: Any) -> None:
        with self._lock:
            self._pipeline = pipeline
            self._key = key


_pipeline_cache = _PipelineCache()


def _clear_pipeline_cache():
    _pipeline_cache.clear()


def get_cache_generation() -> int:
    return _pipeline_cache.get_generation()
