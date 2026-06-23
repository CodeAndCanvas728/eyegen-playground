"""Bonsai pipeline wrapper and factory."""

from __future__ import annotations

import logging
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image

from .constants import (
    BONSAI_DEFAULT_GUIDANCE,
    DEFAULT_VARIANT,
    SUPPORTED_VARIANTS,
)
from .install import get_bonsai_dir, validate_bonsai_install
from .models import _spawn_bonsai_subprocess

log = logging.getLogger(__name__)


class BonsaiWrapper:
    """Adapter exposing EyeGen's uniform ``generate_image()`` over a bonsai subprocess."""

    def __init__(self, config: dict):
        self.config = config
        self.variant = self._resolve_variant(config)
        self._validate()
        self._proc: Optional[subprocess.Popen] = None

    def _resolve_variant(self, config: dict) -> str:
        explicit = config.get("bonsai_model_path")
        model_name = config.get("model", "")
        if explicit:
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

    def generate_image(  # noqa: C901
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
        if cfg_weight not in (None, BONSAI_DEFAULT_GUIDANCE):
            log.warning(
                "Bonsai backend ignores cfg_weight=%.2f; using fixed guidance=%.1f. "
                "Clear the Guidance field in the GUI or omit --guidance on the CLI.",
                cfg_weight,
                BONSAI_DEFAULT_GUIDANCE,
            )
        if negative_prompt:
            log.warning("Bonsai backend ignores negative_prompt (not supported by this model).")
        if image_path:
            log.warning(
                "Bonsai backend does not support img2img; ignoring image_path=%r", image_path
            )
        if denoise != 1.0:
            log.warning("Bonsai backend does not support img2img; ignoring denoise=%.2f", denoise)

        if width % 32 or height % 32:
            raise ValueError(
                f"Bonsai requires width and height to be multiples of 32 (got {width}x{height})."
            )
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        from eyegen.config import OUTPUT_DIR

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
<<<<<<< Updated upstream
        seed_suffix = str(seed) if seed is not None else uuid.uuid4().hex[:8]
        out_path = OUTPUT_DIR / f"bonsai_{self.variant}_{seed_suffix}.png"
=======
        seed_str = str(seed) if seed is not None else uuid.uuid4().hex[:8]
        out_path = OUTPUT_DIR / f"bonsai_{self.variant}_{seed_str}.png"
>>>>>>> Stashed changes

        cmd = [
            "--model",
            self.variant,
            "--prompt",
            prompt,
            "--size",
            f"{width}x{height}",
            "--steps",
            str(num_steps),
            "--output",
            str(out_path),
        ]
        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        log.info("bonsai subprocess: %s", " ".join(cmd))
<<<<<<< Updated upstream
        rc = self._run_subprocess(["generate.sh", *cmd])
=======
        self._proc = _spawn_bonsai_subprocess(["generate.sh", *cmd])
        try:
            if self._proc.stdout is None:
                raise RuntimeError("bonsai subprocess stdout pipe was not created")
            for line in self._proc.stdout:
                line = line.rstrip()
                log.info("[bonsai] %s", line)
            try:
                self._proc.wait(timeout=300.0)
            except subprocess.TimeoutExpired as exc:
                log.warning("Bonsai subprocess timed out, terminating pid=%s", self._proc.pid)
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    log.warning("Bonsai subprocess did not terminate, killing pid=%s", self._proc.pid)
                    self._proc.kill()
                    self._proc.wait(timeout=5.0)
                raise RuntimeError("Bonsai subprocess timed out after 300 seconds") from exc
            rc = self._proc.returncode
        finally:
            self._proc = None
>>>>>>> Stashed changes
        if rc != 0:
            raise RuntimeError(
                f"Bonsai generation failed (exit {rc}). Inspect logs at {get_bonsai_dir()}/outputs/"
            )
        if not out_path.is_file():
            raise RuntimeError(f"Bonsai generation did not produce expected output: {out_path}")
        return Image.open(out_path).convert("RGB")

    def _run_subprocess(self, args: list[str]) -> int:
        """Spawn the bonsai subprocess, stream its stdout, and return the exit code.

        Enforces a hard timeout so a hung generation cannot block the worker thread.
        """
        self._proc = _spawn_bonsai_subprocess(args)
        try:
            if self._proc.stdout is None:
                raise RuntimeError("bonsai subprocess stdout pipe was not created")
            for line in self._proc.stdout:
                log.info("[bonsai] %s", line.rstrip())
            try:
                self._proc.wait(timeout=300.0)
            except subprocess.TimeoutExpired:
                log.error("Bonsai subprocess timed out. Terminating...")
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    log.error("Bonsai subprocess terminate timed out. Killing...")
                    self._proc.kill()
                    self._proc.wait()
                raise
            return self._proc.returncode
        finally:
            self._proc = None

    def cancel(self) -> None:
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
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            log.warning("bonsai cancel: %s", exc)


def get_bonsai_pipeline(config: dict) -> BonsaiWrapper:
    """Build a BonsaiWrapper. The subprocess model load happens lazily on first
    ``generate_image()`` call."""
    return BonsaiWrapper(config)
