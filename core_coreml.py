"""
CoreML backend — subprocess wrapper for the Apple ``python_coreml_stable_diffusion``
package, which provides Stable Diffusion inference on Apple Silicon via Core ML
(Neural Engine + GPU + CPU).

Why a subprocess wrapper and not an in-process import?
    Apple's ``python_coreml_stable_diffusion`` package pins a specific old
    dependency set: ``diffusers==0.30.2``, ``transformers==4.44.2``,
    ``huggingface-hub==0.24.6``, ``numpy<1.24``, ``diffusionkit==0.4.0``.
    These collide with EyeGen's main 3.14 venv (which uses diffusers 0.38,
    numpy 2.x, etc.). Rather than fragment EyeGen's main venv, we install
    Apple's package in a sidecar Python 3.11 venv at
    ``~/models/eyegen/.coreml-venv/`` and shell out to it.

Supported models:
    Stable Diffusion 1.4, 1.5, 2-base, 2.1-base. SDXL and SD3 have CoreML
    conversions in Apple's package but require more GPU memory and longer
    conversion times. For first-class support, start with the 1.x/2.x family.
    Pre-converted weights on HF (e.g. ``apple/coreml-stable-diffusion-2-1-base-palettized``)
    can be downloaded directly without running the conversion step.

Lifecycle:
    1. User runs ``./scripts/setup-coreml.sh`` (creates the sidecar venv and
       installs ``python_coreml_stable_diffusion``).
    2. User runs ``./generate.py convert-coreml stabilityai/stable-diffusion-2-1-base``
       to convert a model from PyTorch to CoreML (15-20 min on M1 Pro).
       OR skips conversion by using a pre-converted HF model:
       ``./generate.py pull-coreml apple/coreml-stable-diffusion-2-1-base-palettized``.
    3. User can now select "coreml" in the backend dropdown and pick the
       model directory.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_COREML_VENV = Path.home() / "models" / "eyegen" / ".coreml-venv"
DEFAULT_COREML_MODELS_DIR = Path.home() / "models" / "eyegen" / "coreml"
PYTHON_REQUIRED = "3.11"

# Known pre-converted CoreML model repos on Hugging Face (Apple's conversions).
# These can be downloaded with snapshot_download and used without running the
# conversion step. Keys are short aliases, values are the HF repo IDs.
PRECONVERTED_HF_MODELS = {
    "sd-1-4": "apple/coreml-stable-diffusion-v1-4",
    "sd-1-5": "apple/coreml-stable-diffusion-v1-5",
    "sd-1-5-palettized": "apple/coreml-stable-diffusion-v1-5-palettized",
    "sd-2-base": "apple/coreml-stable-diffusion-2-base",
    "sd-2-1-base": "apple/coreml-stable-diffusion-2-1-base",
    "sd-2-1-base-palettized": "apple/coreml-stable-diffusion-2-1-base-palettized",
    "sdxl-base": "apple/coreml-stable-diffusion-xl-base",
    "sdxl-ios": "apple/coreml-stable-diffusion-xl-base-ios",
}

VALID_COMPUTE_UNITS = ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL")


# ---------------------------------------------------------------------------
# Install validation
# ---------------------------------------------------------------------------


@dataclass
class CoreMLInstallStatus:
    installed: bool
    venv_python: Optional[Path]
    models_dir: Path
    model_count: int
    message: str


def get_coreml_models_dir() -> Path:
    """Return where converted/downloaded CoreML model bundles live."""
    override = os.environ.get("COREML_MODELS_DIR")
    return Path(override).expanduser() if override else DEFAULT_COREML_MODELS_DIR


def get_coreml_venv() -> Path:
    """Return the sidecar venv for the CoreML backend."""
    override = os.environ.get("COREML_VENV")
    return Path(override).expanduser() if override else DEFAULT_COREML_VENV


def _sidecar_python() -> Optional[Path]:
    """Return the path to the sidecar venv's python, or None if missing."""
    candidates = [
        get_coreml_venv() / "bin" / "python",
        get_coreml_venv() / "bin" / "python3",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _sidecar_has_coreml() -> bool:
    """Return True if the sidecar venv has python_coreml_stable_diffusion installed."""
    py = _sidecar_python()
    if py is None:
        return False
    try:
        r = subprocess.run(
            [str(py), "-c", "import python_coreml_stable_diffusion"],
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode == 0
    except Exception:
        return False


def _validate_coreml_install_uncached() -> CoreMLInstallStatus:
    """Inspect the sidecar venv and model dir, return a status report."""
    py = _sidecar_python()
    has_coreml = _sidecar_has_coreml() if py else False
    models_dir = get_coreml_models_dir()
    model_count = 0
    if models_dir.is_dir():
        model_count = sum(1 for p in models_dir.iterdir() if p.is_dir())

    if py is None:
        msg = (
            f"CoreML sidecar venv not found at {get_coreml_venv()}. "
            "Run: ./scripts/setup-coreml.sh"
        )
        installed = False
    elif not has_coreml:
        msg = (
            f"CoreML sidecar venv exists at {py} but "
            "python_coreml_stable_diffusion is not installed. "
            "Re-run: ./scripts/setup-coreml.sh"
        )
        installed = False
    else:
        msg = (
            f"CoreML ready (venv={py}, models={model_count} in {models_dir})"
        )
        installed = True

    return CoreMLInstallStatus(
        installed=installed,
        venv_python=py,
        models_dir=models_dir,
        model_count=model_count,
        message=msg,
    )


# Cached wrapper: the venv layout and model count only change when the
# user runs setup/pull scripts, but the GUI calls this on every status
# poll. The cache lives for the process lifetime; if the user runs
# ``setup-coreml.sh`` mid-session and needs a fresh check, they must
# relaunch the GUI.
@lru_cache(maxsize=1)
def validate_coreml_install() -> CoreMLInstallStatus:
    return _validate_coreml_install_uncached()


# ---------------------------------------------------------------------------
# Model listing + downloading + conversion
# ---------------------------------------------------------------------------


def list_coreml_models() -> list[dict]:
    """Return metadata for each CoreML model bundle under the models dir.

    A valid bundle is a directory containing the .mlmodelc / .mlpackage
    subdirectories produced by Apple's converter. We accept any directory
    under the models dir that has at least one .mlmodelc.
    """
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
        # Try to discover model_version from a sidecar metadata file.
        meta_path = child / "model_index.json"
        meta = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
        out.append({
            "name": child.name,
            "path": str(child),
            "resources_path": str(child),  # Apple pipeline expects this
            "model_version": meta.get("model_version", child.name),
            "compute_unit": meta.get("compute_unit", "CPU_AND_NE"),
            "format": "mlmodelc" if mlmodelc else "mlpackage",
        })
    return out


def pull_preconverted_coreml_model(
    alias: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Optional[Path]:
    """Download a pre-converted CoreML model from HF.

    *alias* is a key of :data:`PRECONVERTED_HF_MODELS` (or a full HF repo ID).
    Returns the local model directory on success, or None on failure.
    """
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
    """Run Apple's torch2coreml conversion in the sidecar venv.

    Returns True on success, False on failure.
    This is a long-running operation (15-20 min on M1 Pro for SD 1.x/2.x).
    """
    if compute_unit not in VALID_COMPUTE_UNITS:
        raise ValueError(
            f"Invalid compute_unit {compute_unit!r}. "
            f"Must be one of {VALID_COMPUTE_UNITS}."
        )
    py = _sidecar_python()
    if py is None or not _sidecar_has_coreml():
        raise RuntimeError(
            "CoreML sidecar venv not ready. Run: ./scripts/setup-coreml.sh"
        )
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(py), "-m", "python_coreml_stable_diffusion.torch2coreml",
        "--convert-unet", "--convert-text-encoder",
        "--convert-vae-decoder", "--convert-safety-checker",
        "--model-version", hf_model_id,
        "-o", str(output_dir),
        "--compute-unit", compute_unit,
        "--attention-implementation", attention_implementation,
    ]
    if quantize_nbits is not None:
        cmd.extend(["--quantize-nbits", str(quantize_nbits)])

    log.info("coreml-convert: %s", " ".join(cmd))
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
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


# ---------------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------------


class CoreMLWrapper:
    """Adapter that runs Apple's Stable Diffusion pipeline via subprocess."""

    def __init__(self, config: dict):
        self.config = config
        self.model_path = self._resolve_model_path(config)
        self.compute_unit = self._resolve_compute_unit(config)
        self._validate()
        self._proc: Optional[subprocess.Popen] = None
        self._cancelled = False

    def _resolve_model_path(self, config: dict) -> Path:
        explicit = config.get("coreml_model_path")
        if explicit:
            p = Path(explicit).expanduser()
            if not p.is_dir():
                raise FileNotFoundError(
                    f"coreml_model_path {p} is not a directory. "
                    "Run: ./generate.py pull-coreml <alias>, "
                    "or ./generate.py convert-coreml <hf-model> --output <dir>."
                )
            return p
        # Fallback: try the model name as a coreml model dir name.
        model_name = config.get("model", "")
        if model_name:
            candidate = get_coreml_models_dir() / model_name
            if candidate.is_dir():
                return candidate
        raise RuntimeError(
            "No coreml_model_path set and no CoreML model found at "
            f"{get_coreml_models_dir() / '<model-name>'}. "
            "Set coreml_model_path in config or run "
            "./generate.py pull-coreml <alias>."
        )

    def _resolve_compute_unit(self, config: dict) -> str:
        cu = config.get("coreml_compute_unit", "CPU_AND_NE")
        if cu not in VALID_COMPUTE_UNITS:
            log.warning("Unknown coreml_compute_unit %r; falling back to CPU_AND_NE", cu)
            return "CPU_AND_NE"
        return cu

    def _validate(self) -> None:
        status = validate_coreml_install()
        if not status.installed:
            raise RuntimeError(
                f"{status.message} "
                "Run ./scripts/setup-coreml.sh, then ./generate.py pull-coreml <alias>."
            )
        if not self.model_path.exists():
            raise RuntimeError(f"CoreML model dir does not exist: {self.model_path}")

    def generate_image(self, prompt: str, cfg_weight: float, num_steps: int,
                       width: int, height: int, seed: Optional[int] = None,
                       negative_prompt: str = "",
                       image_path: Optional[str] = None,
                       denoise: float = 1.0) -> "PIL.Image.Image":
        """Run Apple's pipeline via subprocess and return a PIL Image."""
        from PIL import Image

        # Apple's pipeline uses fixed 512x512 and ignores negative_prompt when
        # using v1.5/v2-base; we forward everything we can and warn on ignored.
        if image_path:
            log.warning(
                "CoreML backend's SD 1.x/2.x pipeline does not support img2img in this "
                "wrapper. Ignoring image_path=%r. (Re-convert with --convert-vae-encoder "
                "and add an img2img wrapper to enable.)", image_path,
            )
        if width != 512 or height != 512:
            log.warning(
                "CoreML SD 1.x/2.x models are converted at a fixed 512x512. "
                "Requested %dx%d may produce odd results.", width, height,
            )
        if width % 8 or height % 8:
            raise ValueError(
                f"CoreML requires width and height to be multiples of 8 "
                f"(got {width}x{height})."
            )

        from core import OUTPUT_DIR
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"coreml_{seed or uuid.uuid4().hex[:8]}.png"

        py = _sidecar_python()
        if py is None:
            raise RuntimeError("CoreML sidecar Python not found; run ./scripts/setup-coreml.sh")
        cmd = [
            str(py), "-m", "python_coreml_stable_diffusion.pipeline",
            "--prompt", prompt,
            "--compute-unit", self.compute_unit,
            "--num-inference-steps", str(num_steps),
            "--guidance-scale", str(cfg_weight),
            "--image-height", str(height),
            "--image-width", str(width),
            "-i", str(self.model_path),
            "-o", str(out_path),
        ]
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        if negative_prompt:
            cmd.extend(["--negative-prompt", negative_prompt])

        log.info("coreml-pipeline: %s", " ".join(cmd))
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        self._proc = subprocess.Popen(cmd, env=env,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       text=True)
        try:
            _stdout, stderr = self._proc.communicate()
            returncode = self._proc.returncode
        finally:
            self._proc = None
        if self._cancelled:
            raise RuntimeError("CoreML generation was cancelled by the user.")
        if returncode != 0:
            log.error("coreml stderr: %s", (stderr or "")[-2000:])
            raise RuntimeError(
                f"CoreML generation failed (exit {returncode}). "
                "See eyegen.log for details."
            )
        if not out_path.is_file():
            raise RuntimeError(
                f"CoreML pipeline did not produce expected output: {out_path}"
            )
        return Image.open(out_path).convert("RGB")

    def cancel(self) -> None:
        """Terminate the running CoreML subprocess, if any.

        Called by the GUI's ``Stop`` button. Tries ``terminate()`` (SIGTERM)
        first; if the process has not exited after a short grace period,
        escalates to ``kill()`` (SIGKILL). No-op when no generation is in
        flight.
        """
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        log.info("coreml cancel: terminating pid=%s", proc.pid)
        self._cancelled = True
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                log.warning("coreml cancel: terminate timed out, killing pid=%s", proc.pid)
                proc.kill()
                proc.wait(timeout=2.0)
        except Exception as exc:
            log.warning("coreml cancel: %s", exc)


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


def get_coreml_pipeline(config: dict) -> CoreMLWrapper:
    """Build a CoreMLWrapper. The subprocess runs lazily on first generate call."""
    return CoreMLWrapper(config)
