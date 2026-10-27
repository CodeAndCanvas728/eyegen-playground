"""CoreML model listing, downloading, and conversion."""

from __future__ import annotations

import logging
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from . import model_index
from .assets import add_pipeline_assets
from .constants import VALID_COMPUTE_UNITS
from .install import (
    _sidecar_has_coreml,
    _sidecar_python,
    get_coreml_models_dir,
)
from .offline_runner import has_models

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def list_coreml_models() -> list[dict]:
    models_dir = get_coreml_models_dir()
    out = []
    if not models_dir.is_dir():
        return out
    for child in sorted(models_dir.iterdir()):
        if not child.is_dir():
            continue
        mlmodelc = list(child.glob("*.mlmodelc"))
        mlpackage = list(child.glob("*.mlpackage"))
        if not (mlmodelc or mlpackage):
            continue
        meta = model_index.read(child)
        out.append(
            {
                "name": child.name,
                "path": str(child),
                "resources_path": str(child),
                "model_version": meta.get("model_version", child.name),
                "compute_unit": meta.get("compute_unit", "CPU_AND_NE"),
                "format": "mlmodelc" if mlmodelc else "mlpackage",
            }
        )
    return out


def convert_to_coreml(
    hf_model_id: str,
    output_dir: Path,
    compute_unit: str = "CPU_AND_NE",
    attention_implementation: str = "SPLIT_EINSUM",
    quantize_nbits: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    timeout: Optional[float] = None,
) -> bool:
    if compute_unit not in VALID_COMPUTE_UNITS:
        raise ValueError(
            f"Invalid compute_unit {compute_unit!r}. Must be one of {VALID_COMPUTE_UNITS}."
        )
    output_dir = Path(output_dir).expanduser()
    if has_models(output_dir):
        log.info("The models in %s are already converted; adding the files a run needs", output_dir)
        return add_pipeline_assets(hf_model_id, output_dir)

    py = _sidecar_python()
    if py is None or not _sidecar_has_coreml():
        raise RuntimeError("CoreML sidecar venv not ready. Run: ./scripts/setup-coreml.sh")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(py),
        "-m",
        "python_coreml_stable_diffusion.torch2coreml",
        "--convert-unet",
        "--convert-text-encoder",
        "--convert-vae-decoder",
        "--convert-safety-checker",
        "--model-version",
        hf_model_id,
        "-o",
        str(output_dir),
        "--compute-unit",
        compute_unit,
        "--attention-implementation",
        attention_implementation,
    ]
    if quantize_nbits is not None:
        cmd.extend(["--quantize-nbits", str(quantize_nbits)])

    if timeout is None:
        from eyegen.config import load_config

        timeout = float(load_config().convert_timeout)

    log.info("coreml-convert: %s", " ".join(cmd))
    returncode, timed_out = _run_conversion(cmd, progress_callback, timeout=timeout)

    if timed_out:
        raise subprocess.TimeoutExpired(cmd, timeout)

    if returncode != 0:
        return False
    # Recorded before the assets are fetched: a conversion that cannot reach
    # Hugging Face for the tokenizer has still converted every model, and the
    # next run must find it rather than convert it all over again.
    model_index.record(
        output_dir,
        {
            "model_version": hf_model_id,
            "compute_unit": compute_unit,
            "attention_implementation": attention_implementation,
            "quantize_nbits": quantize_nbits,
            "converted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    list_coreml_models.cache_clear()
    return add_pipeline_assets(hf_model_id, output_dir)


def _run_conversion(
    cmd: list[str],
    progress_callback: Optional[Callable[[str], None]] = None,
    timeout: Optional[float] = None,
) -> tuple[int, bool]:
    """Run the CoreML conversion subprocess.

    Streams combined stdout/stderr to the log and *progress_callback*. Returns
    ``(returncode, timed_out)``.
    """
    from eyegen.backends.runner import BaseSubprocessRunner
    from eyegen.config import load_config

    cfg = load_config()
    runner = BaseSubprocessRunner(cfg)
    timed_out = False
    run_timeout = timeout if timeout is not None else float(cfg.convert_timeout)
    try:
        returncode, _, _ = runner._execute_subprocess(
            cmd,
            stream_stdout=True,
            stream_stderr=True,
            log_prefix="coreml-convert",
            progress_callback=progress_callback,
            timeout=run_timeout,
        )
    except RuntimeError as exc:
        if "timed out" in str(exc):
            timed_out = True
            returncode = -1
        else:
            raise
    return returncode, timed_out
