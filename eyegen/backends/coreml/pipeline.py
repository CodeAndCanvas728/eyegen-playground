"""CoreML pipeline wrapper and factory."""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image

from eyegen.backends.runner import BaseSubprocessRunner
from eyegen.config import EyeGenConfig

from .constants import VALID_COMPUTE_UNITS
from .install import (
    _sidecar_python,
    get_coreml_models_dir,
    validate_coreml_install,
)

log = logging.getLogger(__name__)


class CoreMLWrapper(BaseSubprocessRunner):
    """Adapter that runs Apple's Stable Diffusion pipeline via subprocess."""

    def __init__(self, config: EyeGenConfig):
        super().__init__(config)
        self.model_path = self._resolve_model_path(config)
        self.compute_unit = self._resolve_compute_unit(config)
        self._validate()

    def _resolve_model_path(self, config: EyeGenConfig) -> Path:
        if config.coreml_model_path:
            from eyegen.validation import validate_safe_path

            p = validate_safe_path(config.coreml_model_path, "coreml_model_path")
            if not p.is_dir():
                raise FileNotFoundError(
                    f"coreml_model_path {p} is not a directory. "
                    "Run: ./generate.py pull-coreml <alias>, "
                    "or ./generate.py convert-coreml <hf-model> --output <dir>."
                )
            return p
        if not config.model:
            raise RuntimeError(
                "No model selected. Set a model name or provide a coreml_model_path."
            )
        if not re.match(r"^[a-zA-Z0-9_\-\.]+$", config.model):
            raise ValueError(f"Invalid characters in CoreML model name: {config.model}")
        candidate = get_coreml_models_dir() / config.model
        if candidate.is_dir():
            return candidate
        raise RuntimeError(
            f"CoreML model '{config.model}' not found at {candidate}. "
            "Run ./generate.py pull-coreml <alias> to download it."
        )

    def _resolve_compute_unit(self, config: EyeGenConfig) -> str:
        cu = config.coreml_compute_unit or "CPU_AND_NE"
        if cu not in VALID_COMPUTE_UNITS:
            raise ValueError(
                f"Unsupported CoreML compute unit: {cu!r}. Must be one of {VALID_COMPUTE_UNITS}."
            )
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

        returncode, stdout_lines, stderr_lines = self._execute_subprocess(
            cmd,
            stream_stdout=False,
            stream_stderr=True,
            log_prefix="coreml",
        )

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
        """Reject unsupported options and invalid dimensions."""
        if image_path:
            raise ValueError("CoreML backend's SD 1.x/2.x pipeline does not support img2img.")
        if width != 512 or height != 512:
            raise ValueError(
                "CoreML SD 1.x/2.x models only support a fixed 512x512 resolution "
                f"(got {width}x{height})."
            )


def get_coreml_pipeline(config: EyeGenConfig) -> CoreMLWrapper:
    """Build a CoreMLWrapper. The subprocess runs lazily on first generate call."""
    return CoreMLWrapper(config)
