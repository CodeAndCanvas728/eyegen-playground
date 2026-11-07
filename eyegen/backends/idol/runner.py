"""Run an image through the Idol SDK (image → 3D Gaussian splat)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from eyegen.backends.runner import BaseSubprocessRunner
from eyegen.config import EyeGenConfig

from .constants import (
    IDOL_DEFAULT_DEVICE,
    IDOL_DEFAULT_MODE,
    IDOL_DEFAULT_NUM_GAUSSIANS,
    IDOL_DEFAULT_STEPS,
)
from .install import validate_idol_install

log = logging.getLogger(__name__)


@dataclass
class IdolResult:
    """What one idol.sample run wrote."""

    image: Path
    out_dir: Path
    ply: Path
    input_png: Path
    camera_json: Path


class IdolRunner(BaseSubprocessRunner):
    """Adapter that runs one image through ``python -m idol.sample``.

    The SDK is a separate program in its own sidecar venv on purpose: it needs
    torch 2.15 + transformers 4.57, which conflict with the main venv's
    stack (the CoreML/Bonsai sidecar pattern, DD-009/DD-020). The command
    line is the SDK's documented interface.
    """

    # The flags this backend passes, and which of them take a value. Only
    # the ones `_build_command` emits: the guard is a check on what this
    # backend builds, not a list of what idol.sample could take.
    FLAGS_WITH_ARGS = {
        "-m",
        "--image",
        "--out",
        "--mode",
        "--device",
        "--steps",
        "--num_gaussians",
        "--seed",
    }
    FLAGS_WITHOUT_ARGS: set = set()

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

    def __init__(self, config: EyeGenConfig):
        super().__init__(config)
        status = validate_idol_install(config)
        if not status.installed:
            raise RuntimeError(status.message)
        self.venv_python = str(status.venv_python)

    def _build_command(
        self,
        image: Path,
        out_dir: Path,
        mode: str,
        steps: int,
        num_gaussians: int,
        seed: int,
        device: str,
    ) -> List[str]:
        return [
            self.venv_python,
            "-m",
            "idol.sample",
            "--image",
            str(image),
            "--mode",
            mode,
            "--out",
            str(out_dir),
            "--device",
            device,
            "--steps",
            str(steps),
            "--num_gaussians",
            str(num_gaussians),
            "--seed",
            str(seed),
        ]

    def run(
        self,
        image: Path,
        out_dir: Path,
        mode: str = IDOL_DEFAULT_MODE,
        steps: int = IDOL_DEFAULT_STEPS,
        num_gaussians: int = IDOL_DEFAULT_NUM_GAUSSIANS,
        seed: int = 42,
        device: str = IDOL_DEFAULT_DEVICE,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> IdolResult:
        """Run idol.sample on *image* and return the files it wrote.

        The SDK writes ``<stem>.ply``, ``<stem>_input.png`` and
        ``<stem>_camera.json`` into *out_dir*. A non-zero exit raises with
        the tail of the subprocess output.
        """
        image = Path(image).expanduser()
        out_dir = Path(out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._build_command(image, out_dir, mode, steps, num_gaussians, seed, device)
        timeout = float(getattr(self.config, "idol_timeout", 3600) or 3600)
        returncode, stdout_lines, stderr_lines = self._execute_subprocess(
            cmd,
            cwd=str(out_dir),
            stream_stdout=True,
            stream_stderr=True,
            log_prefix="idol",
            progress_callback=progress_callback,
            timeout=timeout,
        )
        if returncode != 0:
            tail = (stderr_lines or stdout_lines or [""])[-5:]
            raise RuntimeError(f"idol.sample failed (exit {returncode}):\n" + "\n".join(tail))
        stem = image.stem
        ply = out_dir / f"{stem}.ply"
        if not ply.is_file():
            raise RuntimeError(f"idol.sample exited 0 but wrote no {ply.name} in {out_dir}")
        return IdolResult(
            image=image,
            out_dir=out_dir,
            ply=ply,
            input_png=out_dir / f"{stem}_input.png",
            camera_json=out_dir / f"{stem}_camera.json",
        )


def run_to3d(
    config: EyeGenConfig,
    image: Path,
    out_dir: Path,
    mode: str = IDOL_DEFAULT_MODE,
    steps: int = IDOL_DEFAULT_STEPS,
    num_gaussians: int = IDOL_DEFAULT_NUM_GAUSSIANS,
    seed: int = 42,
    device: str = IDOL_DEFAULT_DEVICE,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> IdolResult:
    """Build the runner for *config* and run one image through it."""
    return IdolRunner(config).run(
        image,
        out_dir,
        mode=mode,
        steps=steps,
        num_gaussians=num_gaussians,
        seed=seed,
        device=device,
        progress_callback=progress_callback,
    )
