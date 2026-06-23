"""Bonsai pipeline wrapper and factory."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image

from eyegen.backends.runner import BaseSubprocessRunner

from .constants import (
    BONSAI_DEFAULT_GUIDANCE,
    DEFAULT_VARIANT,
    SUPPORTED_VARIANTS,
)
from .install import get_bonsai_dir, validate_bonsai_install

log = logging.getLogger(__name__)


class BonsaiWrapper(BaseSubprocessRunner):
    """Adapter exposing EyeGen's uniform ``generate_image()`` over a bonsai subprocess."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.variant = self._resolve_variant(config)
        self._validate()

    def _resolve_variant(self, config: dict) -> str:
        explicit = config.get("bonsai_model_path")
        model_name = config.get("model", "")
        variant = DEFAULT_VARIANT
        if explicit:
            name = Path(explicit).expanduser().name
            if "bonsai-image-4B-" in name:
                variant = name.split("bonsai-image-4B-")[-1]
            else:
                variant = name
        elif "bonsai-image-4B-" in model_name:
            variant = model_name.split("bonsai-image-4B-")[-1]
        elif model_name.startswith("bonsai-"):
            variant = model_name[len("bonsai-"):]
        else:
            variant = model_name

        # Security validation for variant to avoid arbitrary arg/command injection
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported Bonsai variant: {variant!r}. Must be one of {SUPPORTED_VARIANTS}"
            )
        return variant

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

        # Auto-adjust width and height to nearest multiple of 32
        adjusted_width = max(32, ((width + 16) // 32) * 32)
        adjusted_height = max(32, ((height + 16) // 32) * 32)
        if adjusted_width != width or adjusted_height != height:
            log.info(
                "Auto-adjusting width/height for Bonsai from %dx%d to %dx%d (multiples of 32)",
                width, height, adjusted_width, adjusted_height
            )
            width = adjusted_width
            height = adjusted_height

        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        from eyegen.config import OUTPUT_DIR

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        seed_str = str(seed) if seed is not None else uuid.uuid4().hex[:8]
        out_path = OUTPUT_DIR / f"bonsai_{self.variant}_{seed_str}.png"

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

        # Construct generation script command safely
        status = validate_bonsai_install()
        script_path = status.bonsai_dir / "scripts" / "generate.sh"
        if not script_path.is_file():
            raise FileNotFoundError(f"Bonsai script not found: {script_path}")

        full_cmd = [str(script_path), *cmd]

        env = {"BONSAI_PACKAGE_MIN_AGE_DAYS": "0"}

        returncode, stdout_lines, stderr_lines = self._execute_subprocess(
            full_cmd,
            cwd=str(status.bonsai_dir),
            env=env,
            stream_stdout=True,
            stream_stderr=False,
            log_prefix="bonsai",
        )

        if returncode != 0:
            raise RuntimeError(
                f"Bonsai generation failed (exit {returncode}). Inspect logs at {get_bonsai_dir()}/outputs/"
            )
        if not out_path.is_file():
            raise RuntimeError(f"Bonsai generation did not produce expected output: {out_path}")
        return Image.open(out_path).convert("RGB")


def get_bonsai_pipeline(config: dict) -> BonsaiWrapper:
    """Build a BonsaiWrapper. The subprocess model load happens lazily on first
    ``generate_image()`` call."""
    return BonsaiWrapper(config)
