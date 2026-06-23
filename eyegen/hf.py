"""HuggingFace authentication helpers."""

import logging
from typing import Optional

log = logging.getLogger(__name__)


def hf_login(token: str) -> dict:
    """Log in to HuggingFace and return the user info dict."""
    from huggingface_hub import login, whoami

    login(token=token)
    return whoami()


def hf_status() -> Optional[dict]:
    """Return HuggingFace user info if logged in, or None."""
    from huggingface_hub import errors as hf_errors
    from huggingface_hub import get_token, whoami

    if get_token() is None:
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
        return whoami()
    except tuple(catch_types) as exc:
        log.debug("HF status check failed: %s", exc)
        return None


def hf_logout():
    """Remove the stored HuggingFace token."""
    from huggingface_hub import logout

    logout()
