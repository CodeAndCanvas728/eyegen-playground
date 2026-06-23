"""Bonsai model listing and downloading."""

from __future__ import annotations

import logging
import os
import subprocess
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
    script_name: str, *args: str, progress_callback: Optional[Callable[[str], None]] = None
) -> int:
    status = validate_bonsai_install()
    script_path = status.bonsai_dir / "scripts" / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"Bonsai script not found: {script_path}")

    cmd = [str(script_path), *args]
    log.info("Running bonsai script: %s", " ".join(cmd))

    env = os.environ.copy()
    env["BONSAI_PACKAGE_MIN_AGE_DAYS"] = env.get("BONSAI_PACKAGE_MIN_AGE_DAYS", "0")

    proc = subprocess.Popen(  # noqa: S603
        cmd,
        cwd=str(status.bonsai_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if proc.stdout is None:
        raise RuntimeError("bonsai subprocess stdout pipe was not created")
    try:
        for line in proc.stdout:
            line = line.rstrip()
            log.info("[bonsai] %s", line)
            if progress_callback is not None:
                progress_callback(line)
        try:
            proc.wait(timeout=1800.0)
        except subprocess.TimeoutExpired as exc:
            log.warning("Bonsai subprocess timed out, terminating pid=%s", proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                log.warning("Bonsai subprocess did not terminate, killing pid=%s", proc.pid)
                proc.kill()
                proc.wait(timeout=5.0)
            raise RuntimeError("Bonsai subprocess timed out after 30 minutes") from exc
    except KeyboardInterrupt:
        proc.terminate()
        raise
    return proc.returncode


def _spawn_bonsai_subprocess(cmd: list[str]) -> subprocess.Popen:
    status = validate_bonsai_install()
    script_path = status.bonsai_dir / "scripts" / cmd[0]
    if not script_path.is_file():
        raise FileNotFoundError(f"Bonsai script not found: {script_path}")

    full_cmd = [str(script_path), *cmd[1:]]
    log.info("Running bonsai script: %s", " ".join(full_cmd))

    env = os.environ.copy()
    env["BONSAI_PACKAGE_MIN_AGE_DAYS"] = env.get("BONSAI_PACKAGE_MIN_AGE_DAYS", "0")

    proc = subprocess.Popen(  # noqa: S603
        full_cmd,
        cwd=str(status.bonsai_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if proc.stdout is None:
        raise RuntimeError("bonsai subprocess stdout pipe was not created")
    return proc


def download_bonsai_model(
    variant: str = DEFAULT_VARIANT, progress_callback: Optional[Callable[[str], None]] = None
) -> bool:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"Unknown bonsai variant {variant!r}. Supported: {', '.join(SUPPORTED_VARIANTS)}"
        )
    rc = _run_bonsai_script(
        "download_model.sh", "--model", variant, progress_callback=progress_callback
    )
    return rc == 0
