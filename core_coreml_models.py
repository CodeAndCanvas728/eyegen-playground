"""CoreML model listing, downloading, and conversion."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

from core_coreml_constants import PRECONVERTED_HF_MODELS, VALID_COMPUTE_UNITS
from core_coreml_install import _sidecar_has_coreml, _sidecar_python, get_coreml_models_dir

log = logging.getLogger(__name__)


def list_coreml_models() -> list[dict]:
    models_dir = get_coreml_models_dir()
    out = []
    if not models_dir.is_dir():
        return out
    for child in sorted(models_dir.iterdir()):
        if not child.is_dir():
            continue
        mlmodelc = list(child.rglob("*.mlmodelc"))
        mlpackage = list(child.rglob("*.mlpackage"))
        if not (mlmodelc or mlpackage):
            continue
        meta_path = child / "model_index.json"
        meta = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
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
    except Exception as exc:
        log.error("Failed to download %s: %s", repo_id, exc)
        return None

    meta = {
        "model_version": repo_id,
        "compute_unit": "CPU_AND_NE",
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (target_dir / "model_index.json").write_text(json.dumps(meta, indent=2) + "\n")
    _cb(f"Done. Model at {target_dir}")
    return target_dir


def convert_to_coreml(
    hf_model_id: str,
    output_dir: Path,
    compute_unit: str = "CPU_AND_NE",
    attention_implementation: str = "SPLIT_EINSUM",
    quantize_nbits: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
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

    log.info("coreml-convert: %s", " ".join(cmd))
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    if proc.stdout is None:
        raise RuntimeError("CoreML conversion stdout pipe was not created")
    for line in proc.stdout:
        line = line.rstrip()
        log.info("[coreml-convert] %s", line)
        if progress_callback:
            progress_callback(line)
    proc.wait()

    if proc.returncode == 0:
        meta = {
            "model_version": hf_model_id,
            "compute_unit": compute_unit,
            "attention_implementation": attention_implementation,
            "converted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (output_dir / "model_index.json").write_text(json.dumps(meta, indent=2) + "\n")
    return proc.returncode == 0
