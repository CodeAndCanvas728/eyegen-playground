"""CoreML (Apple Neural Engine) backend CLI commands."""

import subprocess
from pathlib import Path
from typing import Annotated, Optional

import typer

from eyegen import PROJECT_ROOT
from eyegen.backends import coreml


def setup_coreml_cmd():
    """One-time install: creates the CoreML sidecar Python 3.11 venv."""
    script = PROJECT_ROOT / "scripts" / "setup-coreml.sh"
    if not script.is_file():
        typer.echo(f"❌ Setup script not found: {script}", err=True)
        raise typer.Exit(1)

    typer.echo("🍎 Setting up CoreML backend (one-time install) ...")
    typer.echo("   This creates a sidecar Python 3.11 venv and installs Apple's")
    typer.echo("   python_coreml_stable_diffusion + its pinned dependencies.")
    typer.echo("   May take several minutes on first run.")
    typer.echo()

    rc = subprocess.run([str(script)], cwd=PROJECT_ROOT).returncode  # noqa: S603
    if rc != 0:
        typer.echo(f"❌ CoreML setup failed (exit {rc})", err=True)
        raise typer.Exit(1)


def pull_coreml_cmd(
    alias: str = typer.Argument(
        "sd-2-1-base-palettized",
        help=(
            "Alias (sd-1-4, sd-1-5, sd-2-1-base, sd-2-1-base-palettized, "
            "sdxl-base, ...) or a full HuggingFace repo id."
        ),
    ),
):
    """Download a pre-converted CoreML model from Hugging Face."""
    status = coreml.validate_coreml_install()
    if not status.installed:
        typer.echo(f"❌ {status.message}", err=True)
        typer.echo("   Run: ./generate.py setup-coreml", err=True)
        raise typer.Exit(1)

    typer.echo(f"📥 Downloading pre-converted CoreML model: {alias}")
    target = coreml.pull_preconverted_coreml_model(
        alias,
        progress_callback=lambda m: typer.echo(f"   {m}"),
    )
    if target:
        typer.echo(f"✅ CoreML model downloaded to {target}")
        typer.echo("   Use it via: ./generate.py generate 'prompt' --backend coreml \\")
        typer.echo(f"       --model {alias}")
    else:
        typer.echo(f"❌ Failed to download CoreML model '{alias}'", err=True)
        raise typer.Exit(1)


def convert_coreml_cmd(
    hf_model_id: str = typer.Argument(
        "stabilityai/stable-diffusion-2-1-base",
        help="HuggingFace model id of the PyTorch model to convert",
    ),
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output", "-o", help="Output directory (default: ~/models/eyegen/coreml/<name>)"
        ),
    ] = None,
    compute_unit: str = typer.Option(
        "CPU_AND_NE",
        "--compute-unit",
        "-c",
        help="CoreML compute unit: CPU_AND_NE (mobile), CPU_AND_GPU (Mac), CPU_ONLY, ALL",
    ),
    quantize_nbits: Optional[int] = typer.Option(
        None,
        "--quantize-nbits",
        "-q",
        help="Palettization bit-width: 2, 4, 6, or 8 (reduces size; quality may drop)",
    ),
    attention: str = typer.Option(
        "SPLIT_EINSUM",
        "--attention",
        "-a",
        help=(
            "Attention implementation: SPLIT_EINSUM (NE), "
            "SPLIT_EINSUM_V2 (NE, slower compile), ORIGINAL (CPU/GPU)"
        ),
    ),
):
    """Convert a PyTorch Stable Diffusion model to CoreML."""
    if output is None:
        output = coreml.get_coreml_models_dir() / hf_model_id.split("/")[-1]
    output = output.expanduser()

    status = coreml.validate_coreml_install()
    if not status.installed:
        typer.echo(f"❌ {status.message}", err=True)
        typer.echo("   Run: ./generate.py setup-coreml", err=True)
        raise typer.Exit(1)

    typer.echo(f"🔄 Converting {hf_model_id} → CoreML at {output}")
    typer.echo(f"   compute-unit: {compute_unit}")
    typer.echo(f"   attention:    {attention}")
    if quantize_nbits:
        typer.echo(f"   quantize:     {quantize_nbits}-bit (palettization)")
    typer.echo("   This may take 15-20 minutes on first run.")
    typer.echo()

    ok = coreml.convert_to_coreml(
        hf_model_id=hf_model_id,
        output_dir=output,
        compute_unit=compute_unit,
        attention_implementation=attention,
        quantize_nbits=quantize_nbits,
        progress_callback=lambda m: typer.echo(f"   {m}"),
    )
    if ok:
        typer.echo(f"\n✅ Converted model at {output}")
        typer.echo("   Use it via: ./generate.py generate 'prompt' --backend coreml \\")
        typer.echo(f"       --model {output.name}")
    else:
        typer.echo("\n❌ Conversion failed. See eyegen.log for details.", err=True)
        raise typer.Exit(1)


def list_coreml_models_cmd():
    """List installed CoreML model bundles."""
    status = coreml.validate_coreml_install()
    if not status.installed:
        typer.echo(f"❌ {status.message}", err=True)
        typer.echo("   Run: ./generate.py setup-coreml", err=True)
        raise typer.Exit(1)

    models = coreml.list_coreml_models()
    if not models:
        typer.echo("📦 No CoreML models installed yet.")
        typer.echo("   Pre-converted (fast):  ./generate.py pull-coreml sd-2-1-base-palettized")
        typer.echo(
            "   Convert from PyTorch:  ./generate.py convert-coreml "
            "stabilityai/stable-diffusion-2-1-base"
        )
    else:
        typer.echo(f"🍎 Installed CoreML models ({len(models)}):")
        for m in models:
            typer.echo(f"  • {m['name']:30s}  {m['path']}")
            typer.echo(f"      model_version: {m['model_version']}")
            typer.echo(f"      compute_unit: {m['compute_unit']}  format: {m['format']}")
