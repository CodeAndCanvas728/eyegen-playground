"""Run a model through the stable-diffusion.cpp command line."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

from PIL import Image

from eyegen.backends.runner import BaseSubprocessRunner
from eyegen.config import EyeGenConfig

from .constants import SDCLI_MODELS, TOOL_DEFAULT_GUIDANCE, TOOL_DEFAULT_STEPS
from .install import get_sdcli_timeout, validate_sdcli_install

log = logging.getLogger(__name__)


def resolve_settings(
    model: str,
    steps: Optional[int],
    guidance: Optional[float],
) -> tuple[int, float]:
    """The steps and guidance to use for *model*.

    A setting the user has left at the tool-wide default is the model's to
    choose; one they have changed is theirs.
    """
    spec = SDCLI_MODELS.get(model) or {}
    if steps is None or steps == TOOL_DEFAULT_STEPS:
        steps = spec.get("steps", TOOL_DEFAULT_STEPS)
    if guidance is None or guidance == TOOL_DEFAULT_GUIDANCE:
        guidance = spec.get("cfg_scale", TOOL_DEFAULT_GUIDANCE)
    return steps, guidance


class SdCliWrapper(BaseSubprocessRunner):
    """Adapter that runs a model through the ``sd-cli`` binary.

    The binary is a separate program rather than a Python library on
    purpose: FLUX.3 needs a build of stable-diffusion.cpp that understands
    the architecture, and the Python bindings on PyPI are older than the
    model. The command line is the interface that upstream documents, and it
    is the one that works on Metal.
    """

    # The flags this backend passes, and which of them take a value. Only
    # the ones `_build_command` emits: the guard is a check on what this
    # backend builds, not a list of what the binary could take.
    FLAGS_WITH_ARGS = {
        "--diffusion-model",
        "--llm",
        "--vae",
        "-p",
        "--cfg-scale",
        "--steps",
        "--sampling-method",
        "-s",
        "-W",
        "-H",
        "-o",
        "-r",
        "-n",
    }
    FLAGS_WITHOUT_ARGS = {"--offload-to-cpu"}

    def __init__(self, config: EyeGenConfig, model: str):
        super().__init__(config)
        self.model = model
        self.spec = SDCLI_MODELS.get(model)
        if self.spec is None:
            raise ValueError(f"Unknown stable-diffusion.cpp model: {model}")

        status = validate_sdcli_install(model)
        if not status.installed:
            raise RuntimeError(status.message)
        self.binary = status.binary
        self.model_dir = status.model_dir
        self.timeout = get_sdcli_timeout()

    def _validate_cmd_args(self, cmd: List[str]) -> None:
        """Allow only the flags this backend passes."""
        i = 1
        while i < len(cmd):
            arg = cmd[i]
            if arg.startswith("-"):
                if arg in self.FLAGS_WITH_ARGS:
                    if i + 1 >= len(cmd):
                        raise ValueError(f"Flag {arg} requires a value argument.")
                    i += 2
                elif arg in self.FLAGS_WITHOUT_ARGS:
                    i += 1
                else:
                    raise ValueError(f"Unsafe subprocess flag detected: {arg}")
            else:
                i += 1

    def _build_command(
        self,
        prompt: str,
        output_path: Path,
        steps: int,
        guidance: float,
        width: int,
        height: int,
        seed: Optional[int],
        image_path: Optional[str],
        negative_prompt: str = "",
    ) -> List[str]:
        spec = self.spec
        cmd = [
            str(self.binary),
            "--diffusion-model",
            str(self.model_dir / spec["diffusion_model"]),
            "--llm",
            str(self.model_dir / spec["llm"]),
            "--vae",
            str(self.model_dir / spec["vae"]),
            "-p",
            prompt,
            "--cfg-scale",
            str(guidance),
            "--steps",
            str(steps),
            "--sampling-method",
            spec.get("sampling_method", "euler"),
            "-W",
            str(width),
            "-H",
            str(height),
            "-o",
            str(output_path),
            "--offload-to-cpu",
        ]
        if seed is not None:
            cmd += ["-s", str(seed)]
        if image_path:
            cmd += ["-r", image_path]
        if negative_prompt:
            cmd += ["-n", negative_prompt]
        return cmd

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
        steps, guidance = resolve_settings(self.model, num_inference_steps, guidance_scale)
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


def get_sdcli_pipeline(config: EyeGenConfig, model: Optional[str] = None) -> SdCliWrapper:
    """Build the wrapper for the configured model."""
    name = model or config.model or ""
    return SdCliWrapper(config, name)
