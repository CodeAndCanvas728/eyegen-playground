"""CoreML pipeline wrapper and factory."""

from __future__ import annotations

import logging
import pathlib
import re
import secrets
from pathlib import Path
from typing import Optional

from PIL import Image

from eyegen.backends.runner import BaseSubprocessRunner
from eyegen.config import EyeGenConfig
from eyegen.model_specs import ModelSpec

from . import model_index
from .constants import VALID_COMPUTE_UNITS
from .conversion import converted_if_needed
from .install import (
    _sidecar_python,
    get_coreml_model_dir,
    validate_coreml_install,
)

log = logging.getLogger(__name__)

# The sizes this tool converts to: SD 1.x/2.x and SDXL. A bundle that records
# the size its models were exported at is held to that instead.
_CONVERTED_SIZES = ((512, 512), (1024, 1024))
_CONVERTED_SIZE_NAMES = " and ".join(f"{width}x{height}" for width, height in _CONVERTED_SIZES)


class CoreMLWrapper(BaseSubprocessRunner):
    """Adapter that runs Apple's Stable Diffusion pipeline via subprocess."""

    def __init__(self, config: EyeGenConfig, spec: Optional[ModelSpec] = None):
        super().__init__(config)
        self.spec = spec
        self.compute_unit = self._resolve_compute_unit(config)
        self.model_path = converted_if_needed(self._resolve_model_path(config), self.compute_unit)
        self._validate()

    def _resolve_model_path(self, config: EyeGenConfig) -> Path:
        if config.coreml_model_path:
            from eyegen.validation import validate_safe_path

            p = validate_safe_path(config.coreml_model_path, "coreml_model_path")
            if not p.is_dir():
                raise FileNotFoundError(
                    f"coreml_model_path {p} is not a directory. "
                    "Run: ./generate.py convert-coreml <hf-model> --output <dir>."
                )
            return p
        if self.spec is not None and self.spec.repo:
            candidate = get_coreml_model_dir(self.spec.repo.split("/")[-1])
            if candidate.is_dir():
                return candidate
            raise RuntimeError(
                f"CoreML model '{self.spec.repo}' not found at {candidate}. "
                f"Run ./generate.py convert-coreml {self.spec.repo} to make it."
            )
        if not config.model:
            raise RuntimeError(
                "No model selected. Set a model name or provide a coreml_model_path."
            )
        if not re.match(r"^[a-zA-Z0-9_\-\.]+$", config.model):
            raise ValueError(f"Invalid characters in CoreML model name: {config.model}")
        candidate = get_coreml_model_dir(config.model)
        if candidate.is_dir():
            return candidate
        raise RuntimeError(
            f"CoreML model '{config.model}' not found at {candidate}. "
            "Run ./generate.py convert-coreml <hf-model> to make it."
        )

    def _resolve_compute_unit(self, config: EyeGenConfig) -> str:
        cu = config.coreml_compute_unit or "CPU_AND_NE"
        if cu not in VALID_COMPUTE_UNITS:
            raise ValueError(
                f"Unsupported CoreML compute unit: {cu!r}. Must be one of {VALID_COMPUTE_UNITS}."
            )
        return cu

    def _validate(self) -> None:
        status = validate_coreml_install(force=True)
        if not status.installed:
            raise RuntimeError(
                f"{status.message} "
                "Run ./scripts/setup-coreml.sh, then ./generate.py convert-coreml <hf-model>."
            )
        if not self.model_path.exists():
            raise RuntimeError(f"CoreML model dir does not exist: {self.model_path}")

    def generate_image(
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

        self._validate_parameters(num_steps, cfg_weight, seed, denoise)
        return self._run_coreml_pipeline(
            prompt, cfg_weight, num_steps, width, height, seed, negative_prompt
        )

    def _validate_parameters(
        self, num_steps: int, cfg_weight: float, seed: Optional[int], denoise: float
    ) -> None:
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        if cfg_weight < 0:
            raise ValueError(f"cfg_weight must be >= 0, got {cfg_weight}")
        if seed is not None and not isinstance(seed, int):
            raise ValueError(f"seed must be an integer, got {type(seed).__name__}")
        if seed is not None and not 0 <= seed < 2**32:
            raise ValueError(f"seed must be between 0 and 2**32-1, got {seed}")
        if not (0.0 <= denoise <= 1.0):
            raise ValueError(f"denoise must be in [0, 1], got {denoise}")

    def _run_coreml_pipeline(
        self,
        prompt: str,
        cfg_weight: float,
        num_steps: int,
        width: int,
        height: int,
        seed: Optional[int],
        negative_prompt: str,
    ) -> Image.Image:
        from eyegen.config import OUTPUT_DIR

        output_dir = OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        py = _sidecar_python()
        if py is None:
            raise RuntimeError("CoreML sidecar Python not found; run ./scripts/setup-coreml.sh")

        if seed is None:
            seed = secrets.randbelow(2**32)
        out_path = output_dir / f".tmp_coreml_{seed}.png"
        self._temp_output_path = out_path

        cmd = self._build_coreml_command(
            prompt, cfg_weight, num_steps, width, height, seed, negative_prompt, out_path
        )

        returncode, stdout_lines, stderr_lines = self._execute_subprocess(
            cmd, stream_stdout=False, stream_stderr=True, log_prefix="coreml"
        )

        self._handle_coreml_result(returncode, stderr_lines, out_path)
        return self._load_and_cleanup_image(out_path)

    def _build_coreml_command(
        self,
        prompt: str,
        cfg_weight: float,
        num_steps: int,
        width: int,
        height: int,
        seed: Optional[int],
        negative_prompt: str,
        out_path: pathlib.Path,
    ) -> list:
        runner = Path(__file__).with_name("offline_runner.py")
        cmd = [
            str(_sidecar_python()),
            str(runner),
            "--prompt",
            prompt,
            "--compute-unit",
            self.compute_unit,
            "--num-inference-steps",
            str(num_steps),
            "--guidance-scale",
            str(cfg_weight),
            "--width",
            str(width),
            "--height",
            str(height),
            "-i",
            str(self.model_path),
            "-o",
            str(out_path),
        ]
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        if negative_prompt:
            cmd.extend(["--negative-prompt", negative_prompt])
        return cmd

    def _handle_coreml_result(
        self, returncode: int, stderr_lines: list, out_path: pathlib.Path
    ) -> None:
        if returncode != 0:
            stderr = "".join(stderr_lines)
            log.error("coreml stderr: %s", stderr[-2000:])
            raise RuntimeError(_failure_message(returncode, stderr))
        if not out_path.is_file():
            raise RuntimeError(f"CoreML pipeline did not produce expected output: {out_path}")

    def _load_and_cleanup_image(self, out_path: pathlib.Path) -> Image.Image:
        try:
            return Image.open(out_path).convert("RGB")
        finally:
            try:
                out_path.unlink(missing_ok=True)
            except (OSError, PermissionError):
                pass

    def _check_request(self, width: int, height: int, image_path: Optional[str]) -> None:
        """Reject unsupported options, and a size the bundle cannot generate.

        The models are exported at one resolution, so this is refused here
        rather than after the sidecar has loaded them; the runner checks it
        again against the loaded models, which is what settles it.
        """
        if image_path:
            raise ValueError("CoreML backend's SD 1.x/2.x pipeline does not support img2img.")
        exported = model_index.resolution(self.model_path)
        if exported is None:
            if (width, height) not in _CONVERTED_SIZES:
                raise ValueError(
                    f"{self.model_path.name} does not record the size its models were exported "
                    f"at, so only {_CONVERTED_SIZE_NAMES} can be generated (got {width}x{height}). "
                    "Re-run ./generate.py convert-coreml to record it."
                )
            return
        if (width, height) != exported:
            raise ValueError(
                f"{self.model_path.name} generates {exported[0]}x{exported[1]} images only; "
                f"asked for {width}x{height}. Set width and height to "
                f"{exported[0]}x{exported[1]} to use it."
            )


def _failure_message(returncode: int, stderr: str) -> str:
    """What the runner reported, and its output when it reported nothing."""
    for line in reversed(stderr.strip().splitlines()):
        if line.startswith("Error: "):
            return line[len("Error: ") :]
    return f"CoreML generation failed (exit {returncode}). Stderr: {stderr[-500:]}"


def get_coreml_pipeline(config: EyeGenConfig, spec: Optional[ModelSpec] = None) -> CoreMLWrapper:
    """Build a CoreMLWrapper. The subprocess runs lazily on first generate call."""
    return CoreMLWrapper(config, spec=spec)
