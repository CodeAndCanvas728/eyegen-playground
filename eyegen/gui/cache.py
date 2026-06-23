"""Pipeline cache to avoid reloading models between generations."""

import threading

_pipeline_cache: dict = {"pipeline": None, "key": None}
_pipeline_cache_lock = threading.Lock()


def _clear_pipeline_cache():
    with _pipeline_cache_lock:
        _pipeline_cache["pipeline"] = None
        _pipeline_cache["key"] = None
