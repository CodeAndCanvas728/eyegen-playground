"""CoreML pipeline wrapper and factory."""

from __future__ import annotations

import logging
import os
import subprocess
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

from PIL import Image

from .constants import VALID_COMPUTE_UNITS
from .install import (
    _sidecar_python,
    get_coreml_models_dir,
    validate_coreml_install,
)

log = logging.getLogger(__name__)


def _terminate(proc: subprocess.Popen, grace: float = 5.0) -> None:
    """Terminate *proc*, escalating to kill if it does not exit within *grace*."""
    proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        log.error("CoreML subprocess terminate timed out. Killing...")
        proc.kill()
        proc.wait()


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
            from eyegen.validation import validate_safe_path
            p = validate_safe_path(explicit, "coreml_model_path")
            if not p.is_dir():
                raise FileNotFoundError(
                    f"coreml_model_path {p} is not a directory. "
                    "Run: ./generate.py pull-coreml <alias>, "
                    "or ./generate.py convert-coreml <hf-model> --output <dir>."
                )
            return p
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

    def generate_image(  # noqa: C901, PLR0915
        self,
        prompt: str,
        cfg_weight: float,
        num_steps: int,
        width: int,
        height: int,
        seed: Optional[int] = None,
        negative_prompt: str = "",
        image_path: Optional[str] = None,
        denoise: float = 1.0,
    ) -> Image.Image:
        self._check_request(width, height, image_path)

        from eyegen.config import OUTPUT_DIR

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        seed_str = str(seed) if seed is not None else uuid.uuid4().hex[:8]
        out_path = OUTPUT_DIR / f"coreml_{seed_str}.png"

        py = _sidecar_python()
        if py is None:
            raise RuntimeError("CoreML sidecar Python not found; run ./scripts/setup-coreml.sh")
        cmd = [
            str(py),
            "-m",
            "python_coreml_stable_diffusion.pipeline",
            "--prompt",
            prompt,
            "--compute-unit",
            self.compute_unit,
            "--num-inference-steps",
            str(num_steps),
            "--guidance-scale",
            str(cfg_weight),
            "--image-height",
            str(height),
            "--image-width",
            str(width),
            "-i",
            str(self.model_path),
            "-o",
            str(out_path),
        ]
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        if negative_prompt:
            cmd.extend(["--negative-prompt", negative_prompt])

        log.info("coreml-pipeline: %s", " ".join(cmd))
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        self._proc = subprocess.Popen(  # noqa: S603
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        import threading
        stderr_lines: deque = deque(maxlen=500)
        def read_stderr():
            if self._proc and self._proc.stderr:
                try:
                    for line in self._proc.stderr:
                        line_stripped = line.rstrip()
                        log.info("[coreml] %s", line_stripped)
                        stderr_lines.append(line)
                except Exception as e:
                    log.debug("Error reading coreml stderr: %s", e)

        t = threading.Thread(target=read_stderr, daemon=True)
        t.start()

        try:
            try:
                self._proc.wait(timeout=300.0)
            except subprocess.TimeoutExpired as exc:
                log.warning("CoreML subprocess timed out, terminating pid=%s", self._proc.pid)
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    log.warning(
                        "CoreML subprocess did not terminate, killing pid=%s", self._proc.pid
                    )
                    self._proc.kill()
                    self._proc.wait(timeout=5.0)
                raise RuntimeError("CoreML subprocess timed out after 300 seconds") from exc
            returncode = self._proc.returncode
        finally:
            self._proc = None
            t.join(timeout=1.0)

        if self._cancelled:
            raise RuntimeError("CoreML generation was cancelled by the user.")
        if returncode != 0:
            stderr = "".join(stderr_lines)
            log.error("coreml stderr: %s", stderr[-2000:])
            raise RuntimeError(
                f"CoreML generation failed (exit {returncode}). See eyegen.log for details."
            )
        if not out_path.is_file():
            raise RuntimeError(f"CoreML pipeline did not produce expected output: {out_path}")
        return Image.open(out_path).convert("RGB")

    @staticmethod
    def _check_request(width: int, height: int, image_path: Optional[str]) -> None:
        """Warn about unsupported options and reject invalid dimensions."""
        if image_path:
            log.warning(
                "CoreML backend's SD 1.x/2.x pipeline does not support img2img in this "
                "wrapper. Ignoring image_path=%r. (Re-convert with --convert-vae-encoder "
                "and add an img2img wrapper to enable.)",
                image_path,
            )
        if width != 512 or height != 512:
            log.warning(
                "CoreML SD 1.x/2.x models are converted at a fixed 512x512. "
                "Requested %dx%d may produce odd results.",
                width,
                height,
            )
        if width % 8 or height % 8:
            raise ValueError(
                f"CoreML requires width and height to be multiples of 8 (got {width}x{height})."
            )

    def _run_subprocess(self, cmd: list[str]) -> tuple[int, list[str], bool]:
        """Run the CoreML pipeline subprocess with a hard timeout.

        Streams stderr (rather than buffering via ``communicate()``) and returns
        ``(returncode, stderr_lines, timed_out)``.
        """
        import threading

        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        self._proc = subprocess.Popen(  # noqa: S603
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        timed_out = False

        def target_timeout():
            nonlocal timed_out
            timed_out = True
            proc = self._proc
            if proc:
                log.error("CoreML subprocess timed out after 600 seconds. Terminating...")
                _terminate(proc)

        timer = threading.Timer(600.0, target_timeout)
        timer.start()
        stderr_lines: list[str] = []
        try:
            if self._proc.stderr is None:
                raise RuntimeError("coreml subprocess stderr pipe was not created")
            for line in self._proc.stderr:
                line = line.rstrip()
                log.info("[coreml] %s", line)
                stderr_lines.append(line)
            self._proc.wait()
            returncode = self._proc.returncode
        except Exception:
            if self._proc:
                _terminate(self._proc)
            raise
        finally:
            timer.cancel()
            self._proc = None
        return returncode, stderr_lines, timed_out

    def cancel(self) -> None:
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
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            log.warning("coreml cancel: %s", exc)


def get_coreml_pipeline(config: dict) -> CoreMLWrapper:
    """Build a CoreMLWrapper. The subprocess runs lazily on first generate call."""
    return CoreMLWrapper(config)
