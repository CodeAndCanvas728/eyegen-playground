"""Model management CLI commands."""

from pathlib import Path
from typing import Annotated, Optional

import typer

from eyegen import (
    clear_mflux_cache,
    list_mflux_models,
    list_ollama_models,
    save_mflux_model,
    validate_saved_model,
)
from eyegen.model_catalogue import load_catalogue
from eyegen.paths import saved_mflux_dir


def list_models():
    """List available models for MLX, MFLUX, OllamaDiffuser (GGUF), Bonsai, and CoreML backends."""
    _print_own_models()

    typer.echo("🔷 MLX Native Models (diffusionkit):")
    typer.echo("  • argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized  (Default, ~3GB)")
    typer.echo(
        "  • mlx-community/Lance-3B-AWQ-INT4                          (Multimodal Image Specialist)"
    )
    typer.echo("  (Downloads from HuggingFace on first use — cached locally)")
    typer.echo()

    try:
        mflux_models = list_mflux_models()
        typer.echo(f"🔷 MFLUX models ({len(mflux_models)}):")
        for m in mflux_models:
            if m["alias"] != m["model_name"]:
                typer.echo(f"  • {m['alias']:30s} → {m['model_name']}")
            else:
                typer.echo(f"  • {m['alias']}")
        typer.echo("  (Auto-download from HuggingFace on first use — no pull needed)")
    except (OSError, ValueError, ImportError, AttributeError) as e:
        typer.echo(f"❌ Failed to list MFLUX models: {e}", err=True)

    typer.echo()

    try:
        models = list_ollama_models()
    except (OSError, ValueError, ImportError, AttributeError) as e:
        typer.echo(f"❌ Failed to list OllamaDiffuser models: {e}", err=True)
        raise typer.Exit(1) from e

    installed = models.get("installed", [])
    available = models.get("available", [])

    if installed:
        typer.echo(f"📦 Installed GGUF models ({len(installed)}):")
        for m in sorted(installed):
            typer.echo(f"  ✓ {m}")
    else:
        typer.echo("📦 No GGUF models installed yet.")

    if available:
        not_installed = [m for m in available if m not in installed]
        if not_installed:
            typer.echo(f"\n☁️  Available to pull ({len(not_installed)}):")
            for m in sorted(not_installed):
                typer.echo(f"  • {m}")
    typer.echo("\nPull a GGUF model:  ./generate.py pull <model-name>")

    _print_sdcli_models()


def _print_own_models():
    """The models in the user's own catalogue."""
    try:
        specs = load_catalogue()
    except ValueError as exc:
        typer.echo(f"❌ {exc}", err=True)
        return
    if not specs:
        return
    typer.echo(f"🔷 Your own models ({len(specs)}):")
    for name, spec in sorted(specs.items()):
        detail = spec.description or spec.repo or ", ".join(sorted(spec.sources))
        typer.echo(f"  • {name:24s} {spec.backend.value:16s} {detail}")
    typer.echo("  (Use one with: ./generate.py generate 'your prompt' --model <name>)")


def _print_sdcli_models():
    """List the stable-diffusion.cpp models and whether their files are here."""
    typer.echo()
    try:
        from eyegen.backends.sdcli.models import list_sdcli_models

        sdcli_models = list_sdcli_models()
    except (OSError, ValueError, ImportError, AttributeError) as e:
        typer.echo(f"❌ Failed to list stable-diffusion.cpp models: {e}", err=True)
        return

    typer.echo(f"🔷 stable-diffusion.cpp models ({len(sdcli_models)}):")
    for m in sdcli_models:
        mark = "✓" if m["installed"] else "•"
        typer.echo(f"  {mark} {m['name']:24s} {m['description']}")
        if not m["installed"]:
            typer.echo(f"      fetch it:  ./generate.py pull {m['name']}")
    typer.echo("  (Needs the sd-cli binary — see eyegen/backends/sdcli/__init__.py)")


def save_model_cmd(
    model: str = typer.Argument(
        ...,
        help="MFLUX model alias (e.g. 'dev', 'schnell', 'fibo')",
    ),
    quantize_bits: Optional[int] = typer.Option(
        4,
        "--quantize",
        "-q",
        help="Quantization bits: 4 (default), 8, or omit for full precision",
    ),
    path: Annotated[
        Optional[Path],
        typer.Option("--path", "-p", help="Output directory (default: models/<alias>-<q>bit)"),
    ] = None,
):
    """Save a pre-quantized MFLUX model to disk for fast local loading."""
    if path is None:
        q_label = f"{quantize_bits}bit" if quantize_bits else "fp"
        path = saved_mflux_dir() / f"{model}-{q_label}"

    typer.echo(f"💾 Saving model '{model}' to {path}")
    q_label = f"{quantize_bits}-bit" if quantize_bits else "full precision"
    typer.echo(f"   Quantization: {q_label}")
    typer.echo("   This may take a while (downloads the full model)…")

    try:
        result_path = save_mflux_model(
            model_alias=model,
            quantize=quantize_bits,
            output_path=path,
            progress_callback=lambda msg: typer.echo(f"   {msg}"),
        )
        typer.echo(f"\n✅ Model saved to: {result_path}")
        validation = validate_saved_model(result_path)
        if validation.valid and validation.meta:
            ql = validation.meta.get("quantization_level")
            typer.echo(f"   Quantization: {ql}-bit" if ql else "   Quantization: full precision")
        typer.echo("\n💡 To use this model, set the path in your config:")
        typer.echo(f"   ./generate.py config-set mflux_model_path {result_path}")
    except (OSError, ValueError, RuntimeError) as e:
        typer.echo(f"\n❌ Save failed: {e}", err=True)
        raise typer.Exit(1) from e


def clear_cache(
    model: Optional[str] = typer.Argument(
        None,
        help="Model alias or HF repo-id to clear (e.g. 'dev'). Omit to clear all FLUX caches.",
    ),
):
    """Clear cached model files from HuggingFace to force a fresh download."""
    typer.echo(f"🗑  Clearing model cache{f' for {model}' if model else ''}...")
    try:
        removed = clear_mflux_cache(model)
        if removed:
            for r in removed:
                typer.echo(f"  ✓ Removed: {r}")
            typer.echo(
                f"✅ Cleared {len(removed)} cached revision(s). "
                "Models will re-download on next use."
            )
        else:
            typer.echo("ℹ  No matching cached models found.")
    except (OSError, ValueError) as e:
        typer.echo(f"❌ Cache clear failed: {e}", err=True)
        raise typer.Exit(1) from e
