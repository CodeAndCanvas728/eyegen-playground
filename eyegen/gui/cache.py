"""Pipeline cache to avoid reloading models between generations."""

import threading

_pipeline_cache: dict = {"pipeline": None, "key": None, "generation": 0}
_pipeline_cache_lock = threading.Lock()


def _clear_pipeline_cache():
    with _pipeline_cache_lock:
        _pipeline_cache["pipeline"] = None
        _pipeline_cache["key"] = None
        _pipeline_cache["generation"] += 1


def get_cache_generation() -> int:
    with _pipeline_cache_lock:
        return _pipeline_cache.get("generation", 0)

