"""HuggingFace authentication helpers with caching."""

from __future__ import annotations

import logging
import time
from typing import Optional

log = logging.getLogger(__name__)

# Core HF status caching
_HF_STATUS_CACHE: dict = {"result": None, "timestamp": 0.0}
_HF_CACHE_TTL = 30.0  # seconds


def hf_login(token: str) -> dict:
    """Log in to HuggingFace, update the cache, and return user info."""
    from huggingface_hub import login, whoami

    login(token=token)
    info = whoami()
    _HF_STATUS_CACHE["result"] = info
    _HF_STATUS_CACHE["timestamp"] = time.monotonic()
    return info


def hf_status() -> Optional[dict]:
    """Return HuggingFace user info if logged in, or None. Caches the result."""
    now = time.monotonic()
    cache_age = now - _HF_STATUS_CACHE["timestamp"]
    if cache_age < _HF_CACHE_TTL and _HF_STATUS_CACHE["result"] is not None:
        return _HF_STATUS_CACHE["result"]

    from huggingface_hub import errors as hf_errors
    from huggingface_hub import get_token, whoami

    if get_token() is None:
        _HF_STATUS_CACHE["result"] = None
        _HF_STATUS_CACHE["timestamp"] = now
        return None

    import requests

    catch_types = [
        hf_errors.LocalTokenNotFoundError,
        hf_errors.HTTPError,
        ValueError,
        requests.exceptions.RequestException,
    ]
    offline_err = getattr(hf_errors, "OfflineModeIsEnabled", None)
    if offline_err is not None:
        catch_types.append(offline_err)
    try:
        info = whoami()
        _HF_STATUS_CACHE["result"] = info
        _HF_STATUS_CACHE["timestamp"] = now
        return info
    except tuple(catch_types) as exc:
        log.debug("HF status check failed: %s", exc)
        _HF_STATUS_CACHE["result"] = None
        _HF_STATUS_CACHE["timestamp"] = now
        return None


def hf_logout():
    """Remove the stored HuggingFace token and clear the cache."""
    from huggingface_hub import logout

    logout()
    _HF_STATUS_CACHE["result"] = None
    _HF_STATUS_CACHE["timestamp"] = 0.0
