"""Bonsai model listing and downloading."""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

from .constants import DEFAULT_VARIANT, SUPPORTED_VARIANTS
from .install import validate_bonsai_install

log = logging.getLogger(__name__)


def list_bonsai_models() -> list[dict]:
    status = validate_bonsai_install()
    out = []
    for name in status.has_models:
        out.append(
            {
                "name": name,
                "path": str(status.bonsai_dir / "models" / name),
                "alias": f"bonsai-{name.split('bonsai-image-4B-')[-1]}",
            }
        )
    return out


def _run_bonsai_script(
    script_name: str,
    *args: str,
    progress_callback: Optional[Callable[[str], None]] = None,
    timeout: Optional[float] = None,
) -> int:
    status = validate_bonsai_install()
    script_path = status.bonsai_dir / "scripts" / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"Bonsai script not found: {script_path}")

    cmd = [str(script_path), *args]
    log.info("Running bonsai script: %s", " ".join(cmd))

    env = os.environ.copy()
    env["BONSAI_PACKAGE_MIN_AGE_DAYS"] = env.get("BONSAI_PACKAGE_MIN_AGE_DAYS", "0")

    from eyegen.backends.runner import BaseSubprocessRunner
    from eyegen.config import load_config

    cfg = load_config()
    runner = BaseSubprocessRunner(cfg)
    run_timeout = timeout if timeout is not None else float(cfg.download_timeout)
    try:
        returncode, _, _ = runner._execute_subprocess(
            cmd,
            cwd=str(status.bonsai_dir),
            env=env,
            stream_stdout=True,
            stream_stderr=True,
            log_prefix="bonsai",
            progress_callback=progress_callback,
            timeout=run_timeout,
        )
    except RuntimeError as exc:
        if "timed out" in str(exc):
            raise RuntimeError(f"Bonsai subprocess timed out after {run_timeout} seconds") from exc
        raise
    return returncode


def download_bonsai_model(
    variant: str = DEFAULT_VARIANT,
    progress_callback: Optional[Callable[[str], None]] = None,
    timeout: Optional[float] = None,
) -> bool:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"Unknown bonsai variant {variant!r}. Supported: {', '.join(SUPPORTED_VARIANTS)}"
        )
    rc = _run_bonsai_script(
        "download_model.sh",
        "--model",
        variant,
        progress_callback=progress_callback,
        timeout=timeout,
    )
    return rc == 0
