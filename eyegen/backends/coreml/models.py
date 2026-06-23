"""CoreML model listing, downloading, and conversion."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from .constants import PRECONVERTED_HF_MODELS, VALID_COMPUTE_UNITS
from .install import _sidecar_has_coreml, _sidecar_python, get_coreml_models_dir

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
        meta_path = child / "model_index.json"
        meta = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, ValueError):
                meta = {}
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


def pull_preconverted_coreml_model(
    alias: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Optional[Path]:
    from huggingface_hub import snapshot_download

    repo_id = PRECONVERTED_HF_MODELS.get(alias, alias)
    target_name = repo_id.split("/")[-1]
    target_dir = get_coreml_models_dir() / target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    def _cb(msg):
        log.info("[coreml-pull] %s", msg)
        if progress_callback:
            progress_callback(msg)

    try:
        _cb(f"Downloading {repo_id} ...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
    except (OSError, ValueError) as exc:
        log.error("Failed to download %s: %s", repo_id, exc)
        return None

    meta = {
        "model_version": repo_id,
        "compute_unit": "CPU_AND_NE",
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (target_dir / "model_index.json").write_text(json.dumps(meta, indent=2) + "\n")
    list_coreml_models.cache_clear()
    _cb(f"Done. Model at {target_dir}")
    return target_dir


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
    py = _sidecar_python()
    if py is None or not _sidecar_has_coreml():
        raise RuntimeError("CoreML sidecar venv not ready. Run: ./scripts/setup-coreml.sh")
    output_dir = Path(output_dir).expanduser()
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

        timeout = float(load_config().get("convert_timeout", 1800.0))

    log.info("coreml-convert: %s", " ".join(cmd))
    returncode, timed_out = _run_conversion(cmd, progress_callback, timeout=timeout)

    if timed_out:
        raise subprocess.TimeoutExpired(cmd, timeout)

    if returncode == 0:
        meta = {
            "model_version": hf_model_id,
            "compute_unit": compute_unit,
            "attention_implementation": attention_implementation,
            "converted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (output_dir / "model_index.json").write_text(json.dumps(meta, indent=2) + "\n")
    return returncode == 0


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
    run_timeout = timeout if timeout is not None else float(cfg.get("convert_timeout", 1800.0))
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
