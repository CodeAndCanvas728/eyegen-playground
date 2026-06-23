"""Bonsai (PrismML) backend CLI commands."""

import subprocess

import typer

from eyegen import PROJECT_ROOT
from eyegen.backends import bonsai


def setup_bonsai_cmd():
    """One-time install: clones the Bonsai-Image-Demo and runs its setup.sh."""
    script = PROJECT_ROOT / "scripts" / "setup-bonsai.sh"
    if not script.is_file():
        typer.echo(f"❌ Setup script not found: {script}", err=True)
        raise typer.Exit(1)

    typer.echo("🌳 Setting up Bonsai backend (one-time install) ...")
    typer.echo("   This clones the Bonsai-Image-Demo repo and creates a dedicated")
    typer.echo("   Python 3.11 venv with the patched mflux + MLX kernels.")
    typer.echo("   May take several minutes on first run.")
    typer.echo()

    rc = subprocess.run([str(script)], cwd=PROJECT_ROOT).returncode  # noqa: S603
    if rc != 0:
        typer.echo(f"❌ Bonsai setup failed (exit {rc})", err=True)
        raise typer.Exit(1)


def pull_bonsai_cmd(
    variant: str = typer.Option(
        bonsai.DEFAULT_VARIANT,
        "--variant",
        "-v",
        help=f"Variant to download: {', '.join(bonsai.SUPPORTED_VARIANTS)}",
    ),
):
    """Download a Bonsai (PrismML) model via the bonsai-demo's download script."""
    if variant not in bonsai.SUPPORTED_VARIANTS:
        typer.echo(
            f"❌ Unknown variant '{variant}'. Supported: {', '.join(bonsai.SUPPORTED_VARIANTS)}",
            err=True,
        )
        raise typer.Exit(1)

    status = bonsai.validate_bonsai_install()
    if not status.installed:
        typer.echo(f"❌ {status.message}", err=True)
        raise typer.Exit(1)

    typer.echo(f"📥 Downloading bonsai variant: {variant}")
    ok = bonsai.download_bonsai_model(
        variant,
        progress_callback=lambda m: typer.echo(f"   {m}"),
    )
    if ok:
        typer.echo(f"✅ Bonsai variant '{variant}' is ready at {bonsai.get_bonsai_dir()}/models/")
    else:
        typer.echo(f"❌ Failed to download bonsai variant '{variant}'", err=True)
        raise typer.Exit(1)


def list_bonsai_models_cmd():
    """List installed Bonsai (PrismML) models."""
    status = bonsai.validate_bonsai_install()
    if not status.installed:
        typer.echo(f"❌ {status.message}", err=True)
        typer.echo("   Run: ./generate.py setup-bonsai", err=True)
        raise typer.Exit(1)

    models = bonsai.list_bonsai_models()
    if not models:
        typer.echo("📦 No bonsai models installed yet.")
        typer.echo(f"   Available variants: {', '.join(bonsai.SUPPORTED_VARIANTS)}")
        typer.echo("   Run: ./generate.py pull-bonsai")
    else:
        typer.echo(f"🌳 Installed bonsai models ({len(models)}):")
        for m in models:
            typer.echo(f"  • {m['alias']:20s}  {m['path']}")
