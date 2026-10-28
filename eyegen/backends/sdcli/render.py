"""Rendering one image: the sd-cli run loop and its warnings."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image

from .constants import resolve_settings

log = logging.getLogger(__name__)


class SdCliRenderMixin:
    def _warn_about_img2img(
        self,
        image_path: Optional[str],
        denoise: float,
    ) -> None:
        """Say what the image and the denoise actually do.

        A reference image conditions the model: the picture is an input,
        not a source to be redrawn. There is no strength to set, so a
        denoise that says otherwise is said out loud rather than dropped
        in silence.
        """
        if image_path:
            log.info(
                "[sdcli] %r is used as a reference image: the model conditions on it "
                "rather than repainting it.",
                image_path,
            )
        if denoise != 1.0:
            log.warning(
                "The stable-diffusion.cpp backend has no denoise: a reference image is "
                "conditioning, not a starting point. Ignoring denoise=%.2f.",
                denoise,
            )

    def generate_image(
        self,
        prompt: str,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        negative_prompt: str = "",
        image_path: Optional[str] = None,
        denoise: float = 1.0,
        **kwargs,
    ) -> Image.Image:
        """Render *prompt* and return the image."""
        steps, guidance = resolve_settings(
            self.model, num_inference_steps, guidance_scale, spec=self.spec
        )
        self._warn_about_img2img(image_path, denoise)

        with tempfile.TemporaryDirectory(prefix="eyegen-sdcli-") as tmp:
            output = Path(tmp) / "output.png"
            cmd = self._build_command(
                prompt,
                output,
                steps,
                guidance,
                width,
                height,
                seed,
                image_path,
                negative_prompt,
            )
            log.info("[sdcli] %s", " ".join(cmd))
            returncode, _stdout, stderr = self._execute_subprocess(
                cmd, log_prefix="sd-cli", timeout=self.timeout
            )
            if returncode != 0:
                detail = "\n".join(stderr[-8:]) if stderr else ""
                raise RuntimeError(
                    f"sd-cli exited with status {returncode}" + (f":\n{detail}" if detail else "")
                )
            if not output.is_file():
                raise RuntimeError("sd-cli finished without writing an image")
            with Image.open(output) as img:
                return img.convert("RGB")
