"""
Bonsai (PrismML) backend — subprocess wrapper for the PrismML 1.58-bit
ternary + 1-bit binary diffusion transformer models.

This module does NOT import the bonsai kernels directly. Instead, it shells
out to the Bonsai-Image-Demo's own ``scripts/generate.sh``, which runs in
its dedicated Python 3.11 venv (``~/models/eyegen/bonsai-demo/.venv/``) with
the patched mflux-prism + MLX binaries that the 1.58-bit weight format needs.

Why a subprocess wrapper and not an in-process import?
    ``prism-image-studio`` (the public image-studio repo on GitHub) pins a
    patched mflux and a patched MLX in its ``pyproject.toml`` that conflict
    with the upstream ``mflux`` and ``mlx`` EyeGen uses for the existing
    MFLUX + MLX backends. The bonsai-demo's ``setup.sh`` resolves this in
    an isolated venv. Keeping that venv separate means EyeGen's existing
    backends are never affected by the bonsai vendor.

Tradeoff: every call pays the bonsai-demo's cold-start cost (imports +
~4 GiB weight load + first-shape MLX kernel compile), which is ~5-30s
depending on hardware. The bonsai-demo's own docs note this is a CPU/GPU
issue; on Apple Silicon the MLX kernels are precompiled Metal binaries
and cold-start is closer to 5s.

Lifecycle:
    1. User runs ``./scripts/setup-bonsai.sh`` (clones the bonsai-demo into
       ``~/models/eyegen/bonsai-demo/`` and runs its ``setup.sh``).
    2. User runs ``./generate.py pull-bonsai ternary-mlx`` (or the GUI
       equivalent) which invokes ``scripts/download_model.sh``.
    3. User can now select "bonsai" in the backend dropdown. The first
       ``generate`` call pays the cold-start; subsequent calls at the same
       shape benefit from the MLX metallib cache.
"""

from __future__ import annotations

import logging
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default install location. Overridable via $BONSAI_DIR for tests / power users.
DEFAULT_BONSAI_DIR = Path.home() / "models" / "eyegen" / "bonsai-demo"

# Subset of bonsai-demo's variant list. Only the macOS/MLX variants are
# supported from this wrapper — gemlite variants need a Linux GPU and the
# bonsai-demo's full serve.sh daemon, not a one-shot subprocess.
SUPPORTED_VARIANTS = ("ternary-mlx", "binary-mlx")
DEFAULT_VARIANT = "ternary-mlx"

# Bonsai's sampler is fixed: 4 steps, guidance=1.0, no CFG, no negative
# prompt, no img2img. We forward steps/seed/size to the subprocess and
# log a warning when the user passes kwargs that bonsai will ignore.
BONSAI_DEFAULT_STEPS = 4
BONSAI_DEFAULT_GUIDANCE = 1.0


# ---------------------------------------------------------------------------
# Install validation
# ---------------------------------------------------------------------------


@dataclass
class BonsaiInstallStatus:
    installed: bool
    bonsai_dir: Path
    has_venv: bool
    has_generate_sh: bool
    has_models: list[str]
    message: str


def get_bonsai_dir() -> Path:
    """Return the bonsai-demo install directory, honoring $BONSAI_DIR override."""
    override = os.environ.get("BONSAI_DIR")
    return Path(override).expanduser() if override else DEFAULT_BONSAI_DIR


def validate_bonsai_install() -> BonsaiInstallStatus:
    """Inspect the bonsai-demo install and return a status report.

    Never raises. The caller decides whether the install is "good enough"
    for the current operation (e.g. download only needs the repo + scripts,
    generation needs the venv + a model).
    """
    bonsai_dir = get_bonsai_dir()
    has_venv = (bonsai_dir / ".venv" / "bin" / "python").is_file()
    has_generate_sh = (bonsai_dir / "scripts" / "generate.sh").is_file()
    has_download_sh = (bonsai_dir / "scripts" / "download_model.sh").is_file()

    models = []
    models_dir = bonsai_dir / "models"
    if models_dir.is_dir():
        for child in sorted(models_dir.iterdir()):
            if child.is_dir() and any(child.glob("*.safetensors")):
                models.append(child.name)

    installed = has_generate_sh and has_download_sh
    if not installed:
        msg = (
            f"Bonsai is not installed at {bonsai_dir}. "
            "Run: ./scripts/setup-bonsai.sh"
        )
    elif not has_venv:
        msg = (
            f"Bonsai venv not found at {bonsai_dir}/.venv/. "
            "Run: ./scripts/setup-bonsai.sh"
        )
    elif not models:
        msg = (
            "Bonsai is installed but no models are downloaded. "
            "Run: ./generate.py pull-bonsai ternary-mlx"
        )
    else:
        msg = f"Bonsai ready at {bonsai_dir} (models: {', '.join(models)})"

    return BonsaiInstallStatus(
        installed=installed and has_venv,
        bonsai_dir=bonsai_dir,
        has_venv=has_venv,
        has_generate_sh=has_generate_sh,
        has_models=models,
        message=msg,
    )


# ---------------------------------------------------------------------------
# Model listing + downloading
# ---------------------------------------------------------------------------


def list_bonsai_models() -> list[dict]:
    """Return metadata for each bonsai model dir under the install.

    Returns a list of dicts with ``name`` and ``path`` keys, plus a
    synthetic ``bonsai-<variant>`` alias for each installed variant so
    the user can type it in the Model field.
    """
    status = validate_bonsai_install()
    out = []
    for name in status.has_models:
        out.append({
            "name": name,
            "path": str(status.bonsai_dir / "models" / name),
            "alias": f"bonsai-{name.split('bonsai-image-4B-')[-1]}",
        })
    return out


def _run_bonsai_script(script_name: str, *args: str,
                        progress_callback: Optional[Callable[[str], None]] = None
                        ) -> int:
    """Run a bonsai-demo shell script with optional streaming progress.

    Returns the process exit code. Raises FileNotFoundError if the script
    is missing.
    """
    status = validate_bonsai_install()
    script_path = status.bonsai_dir / "scripts" / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"Bonsai script not found: {script_path}")

    cmd = [str(script_path), *args]
    log.info("Running bonsai script: %s", " ".join(cmd))

    env = os.environ.copy()
    # Suppress interactive prompts; bonsai-demo's scripts do not need a TTY
    # for these operations.
    env["BONSAI_PACKAGE_MIN_AGE_DAYS"] = env.get("BONSAI_PACKAGE_MIN_AGE_DAYS", "0")

    proc = subprocess.Popen(
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
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        raise
    return proc.returncode


def _spawn_bonsai_subprocess(cmd: list[str]) -> subprocess.Popen:
    """Spawn a bonsai-demo shell script and return the live Popen handle.

    Used by ``BonsaiWrapper`` so the GUI's ``Stop`` button can terminate a
    running generation. The caller is responsible for iterating the stdout
    pipe and clearing the handle. Raises FileNotFoundError if the script
    is missing, RuntimeError if the stdout pipe could not be created.
    """
    status = validate_bonsai_install()
    script_path = status.bonsai_dir / "scripts" / cmd[0]
    if not script_path.is_file():
        raise FileNotFoundError(f"Bonsai script not found: {script_path}")

    full_cmd = [str(script_path), *cmd[1:]]
    log.info("Running bonsai script: %s", " ".join(full_cmd))

    env = os.environ.copy()
    env["BONSAI_PACKAGE_MIN_AGE_DAYS"] = env.get("BONSAI_PACKAGE_MIN_AGE_DAYS", "0")

    proc = subprocess.Popen(
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


def download_bonsai_model(variant: str = DEFAULT_VARIANT,
                           progress_callback: Optional[Callable[[str], None]] = None
                           ) -> bool:
    """Download a bonsai model via the bonsai-demo's download_model.sh.

    *variant* is one of ``SUPPORTED_VARIANTS``.
    Returns True on success, False if the script failed.
    """
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"Unknown bonsai variant {variant!r}. "
            f"Supported: {', '.join(SUPPORTED_VARIANTS)}"
        )
    rc = _run_bonsai_script("download_model.sh", "--model", variant,
                            progress_callback=progress_callback)
    return rc == 0


# ---------------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------------


class BonsaiWrapper:
    """Adapter exposing EyeGen's uniform ``generate_image()`` over a bonsai subprocess.

    Construction does NOT spawn the bonsai venv — only ``generate_image()`` does.
    This keeps the "pipeline cache" pattern in the GUI's GenerationWorker working
    (one wrapper instance, many generations).
    """

    def __init__(self, config: dict):
        self.config = config
        self.variant = self._resolve_variant(config)
        self._validate()
        self._proc: Optional[subprocess.Popen] = None

    def _resolve_variant(self, config: dict) -> str:
        """Pick the bonsai variant from the config or the model name."""
        explicit = config.get("bonsai_model_path")
        model_name = config.get("model", "")
        if explicit:
            # bonsai_model_path is a local model dir, e.g.
            # "~/models/eyegen/bonsai-demo/models/bonsai-image-4B-ternary-mlx"
            return Path(explicit).expanduser().name.split("bonsai-image-4B-")[-1]
        if model_name.startswith("bonsai-") or "bonsai-image-4B-" in model_name:
            tail = model_name.split("bonsai-image-4B-")[-1] or model_name.replace("bonsai-", "")
            if tail in SUPPORTED_VARIANTS:
                return tail
        return DEFAULT_VARIANT

    def _validate(self) -> None:
        status = validate_bonsai_install()
        if not status.installed:
            raise RuntimeError(
                f"{status.message} "
                "Run ./scripts/setup-bonsai.sh, or use the GUI's Setup Bonsai button."
            )
        if not status.has_models:
            raise RuntimeError(
                "Bonsai is installed but no models are downloaded. "
                f"Run: ./generate.py pull-bonsai {self.variant}"
            )

    def generate_image(self, prompt: str, cfg_weight: float, num_steps: int,
                       width: int, height: int, seed: Optional[int] = None,
                       negative_prompt: str = "",
                       image_path: Optional[str] = None,
                       denoise: float = 1.0) -> "PIL.Image.Image":
        """Run the bonsai subprocess and return a PIL Image.

        Ignores ``cfg_weight``, ``negative_prompt``, ``image_path``,
        ``denoise`` with a logged warning (bonsai uses fixed guidance=1.0,
        no CFG, no negative prompt, no img2img).
        """
        from PIL import Image

        if cfg_weight not in (None, BONSAI_DEFAULT_GUIDANCE):
            log.warning(
                "Bonsai backend ignores cfg_weight=%.2f; using fixed guidance=%.1f. "
                "Clear the Guidance field in the GUI or omit --guidance on the CLI.",
                cfg_weight, BONSAI_DEFAULT_GUIDANCE,
            )
        if negative_prompt:
            log.warning("Bonsai backend ignores negative_prompt (not supported by this model).")
        if image_path:
            log.warning("Bonsai backend does not support img2img; ignoring image_path=%r", image_path)
        if denoise != 1.0:
            log.warning("Bonsai backend does not support img2img; ignoring denoise=%.2f", denoise)

        if width % 32 or height % 32:
            raise ValueError(
                f"Bonsai requires width and height to be multiples of 32 "
                f"(got {width}x{height})."
            )
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        # Bonsai writes to a path we control; use a temp file under OUTPUT_DIR
        # so cleanup is straightforward and the user can find it if something
        # goes wrong.
        from core import OUTPUT_DIR
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"bonsai_{self.variant}_{seed or uuid.uuid4().hex[:8]}.png"

        cmd = [
            "--model", self.variant,
            "--prompt", prompt,
            "--size", f"{width}x{height}",
            "--steps", str(num_steps),
            "--output", str(out_path),
        ]
        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        log.info("bonsai subprocess: %s", " ".join(cmd))
        self._proc = _spawn_bonsai_subprocess(["generate.sh", *cmd])
        try:
            if self._proc.stdout is None:
                raise RuntimeError("bonsai subprocess stdout pipe was not created")
            for line in self._proc.stdout:
                line = line.rstrip()
                log.info("[bonsai] %s", line)
            self._proc.wait()
            rc = self._proc.returncode
        finally:
            self._proc = None
        if rc != 0:
            raise RuntimeError(
                f"Bonsai generation failed (exit {rc}). "
                f"Inspect logs at {get_bonsai_dir()}/outputs/"
            )
        if not out_path.is_file():
            raise RuntimeError(
                f"Bonsai generation did not produce expected output: {out_path}"
            )
        return Image.open(out_path).convert("RGB")

    def cancel(self) -> None:
        """Terminate the running bonsai subprocess, if any.

        Called by the GUI's ``Stop`` button. Tries ``terminate()`` (SIGTERM)
        first; if the process has not exited after a short grace period,
        escalates to ``kill()`` (SIGKILL). No-op when no generation is in
        flight.
        """
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        log.info("bonsai cancel: terminating pid=%s", proc.pid)
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                log.warning("bonsai cancel: terminate timed out, killing pid=%s", proc.pid)
                proc.kill()
                proc.wait(timeout=2.0)
        except Exception as exc:
            log.warning("bonsai cancel: %s", exc)


# ---------------------------------------------------------------------------
# Pipeline factory (matches the get_<backend>_pipeline() convention)
# ---------------------------------------------------------------------------


def get_bonsai_pipeline(config: dict) -> BonsaiWrapper:
    """Build a BonsaiWrapper. The subprocess model load happens lazily on first
    ``generate_image()`` call."""
    return BonsaiWrapper(config)
