"""Idol (image-to-3D) CLI commands: setup-idol and to3d."""

import subprocess
from pathlib import Path
from typing import Annotated, Optional

import typer

from eyegen import OUTPUT_DIR, PROJECT_ROOT
from eyegen.backends import idol
from eyegen.backends.idol.constants import (
    IDOL_DEFAULT_DEVICE,
    IDOL_DEFAULT_MODE,
    IDOL_DEFAULT_NUM_GAUSSIANS,
    IDOL_DEFAULT_STEPS,
    IDOL_DEVICES,
    IDOL_GAUSSIAN_CHOICES,
    IDOL_MODES,
)
from eyegen.config import load_config
from eyegen.validation import validate_image_path


def setup_idol_cmd():
    """One-time install: clones the Idol SDK and creates its sidecar venv."""
    script = PROJECT_ROOT / "scripts" / "setup-idol.sh"
    if not script.is_file():
        typer.echo(f"❌ Setup script not found: {script}", err=True)
        raise typer.Exit(1)

    typer.echo("🧊 Setting up the Idol (image-to-3D) backend (one-time install) ...")
    typer.echo("   This clones speridlabs/idol-sdk (MPS branch) into ~/models/eyegen/idol/")
    typer.echo("   and creates a sidecar Python 3.11 venv with torch.")
    typer.echo("   May take several minutes on first run.")
    typer.echo()

    rc = subprocess.run([str(script)], cwd=PROJECT_ROOT).returncode  # noqa: S603
    if rc != 0:
        typer.echo(f"❌ Idol setup failed (exit {rc})", err=True)
        raise typer.Exit(1)


def _resolve_to3d_args(config, mode, steps, num_gaussians, seed, device, out):
    """Merge CLI options over the config defaults and validate them.

    Argument errors exit before the sidecar install is checked: a bad flag
    is the user's, a missing install is the machine's, and the first should
    be said first.
    """
    mode = (mode or config.idol_mode or IDOL_DEFAULT_MODE).lower()
    if mode not in IDOL_MODES:
        typer.echo(f"❌ Invalid mode '{mode}'. Must be one of: {', '.join(IDOL_MODES)}", err=True)
        raise typer.Exit(1)
    device = (device or config.idol_device or IDOL_DEFAULT_DEVICE).lower()
    if device not in IDOL_DEVICES:
        typer.echo(
            f"❌ Invalid device '{device}'. Must be one of: {', '.join(IDOL_DEVICES)}", err=True
        )
        raise typer.Exit(1)
    steps = steps if steps is not None else config.idol_steps or IDOL_DEFAULT_STEPS
    num_gaussians = (
        num_gaussians
        if num_gaussians is not None
        else config.idol_num_gaussians or IDOL_DEFAULT_NUM_GAUSSIANS
    )
    if num_gaussians % 32 != 0 or not 32768 <= num_gaussians <= 1048576:
        choices = ", ".join(str(c) for c in IDOL_GAUSSIAN_CHOICES)
        typer.echo(
            "❌ --num-gaussians must be a multiple of 32 between 32768 and 1048576 "
            f"(choices: {choices})",
            err=True,
        )
        raise typer.Exit(1)
    seed = seed if seed is not None else 42
    out_dir = (out or OUTPUT_DIR / "to3d").expanduser()
    return mode, steps, num_gaussians, seed, device, out_dir


def to3d(
    image: Annotated[Path, typer.Argument(..., help="Input image (PNG/JPG/JPEG/WEBP/BMP)")],
    mode: str = typer.Option(
        None,
        "--mode",
        help="'synthetic' for renders/AI images (default), 'real' for photos",
    ),
    steps: Optional[int] = typer.Option(None, "--steps", help="Sampling steps (default: 25)"),
    num_gaussians: Optional[int] = typer.Option(
        None,
        "--num-gaussians",
        help="Gaussians to generate: 32768-1048576, a multiple of 32 (default: 131072)",
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed (default: 42)"),
    device: Optional[str] = typer.Option(None, "--device", help="cuda, mps (default) or cpu"),
    out: Annotated[
        Optional[Path],
        typer.Option("--out", "-o", help="Output directory (default: outputs/to3d)"),
    ] = None,
):
    """Turn one image into a 3D Gaussian splat (.ply) with Idol."""
    config = load_config()

    err = validate_image_path(str(image))
    if err:
        typer.echo(f"❌ {err}", err=True)
        raise typer.Exit(1)

    mode, steps, num_gaussians, seed, device, out_dir = _resolve_to3d_args(
        config, mode, steps, num_gaussians, seed, device, out
    )

    status = idol.validate_idol_install(config)
    if not status.installed:
        typer.echo(f"❌ {status.message}", err=True)
        raise typer.Exit(1)

    typer.echo(f"🧊 Idol image-to-3D: {image}")
    typer.echo(
        f"   mode={mode} steps={steps} gaussians={num_gaussians} seed={seed} device={device}"
    )
    typer.echo(f"   out={out_dir}")
    typer.echo(
        "   First run downloads ~5 GB (weights + gated encoders); on MPS this takes minutes."
    )
    typer.echo()

    try:
        result = idol.run_to3d(
            config,
            image.expanduser(),
            out_dir,
            mode=mode,
            steps=steps,
            num_gaussians=num_gaussians,
            seed=seed,
            device=device,
            progress_callback=lambda line: typer.echo(f"   {line}"),
        )
    except (RuntimeError, OSError, ValueError) as exc:
        typer.echo(f"❌ {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo()
    typer.echo("✅ 3D splat written:")
    typer.echo(f"   {result.ply}")
    typer.echo(f"   {result.input_png}  (the image the model saw)")
    typer.echo(f"   {result.camera_json}  (predicted camera)")
    typer.echo("   Open the .ply in a Gaussian splat viewer (e.g. SuperSplat).")
