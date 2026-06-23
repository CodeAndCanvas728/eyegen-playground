"""Model management CLI commands."""

from pathlib import Path
from typing import Optional

import typer

from eyegen import (
    MODELS_DIR,
    clear_mflux_cache,
    list_mflux_models,
    list_ollama_models,
    save_mflux_model,
    validate_saved_model,
)


def pull(
    model_name: str = typer.Argument(
        ..., help="OllamaDiffuser model name (e.g. flux.1-dev-gguf-q4ks)"
    ),
):
    """Download a GGUF model via OllamaDiffuser."""
    typer.echo(f"📥 Pulling model: {model_name}")

    def on_progress(msg):
        if isinstance(msg, str):
            typer.echo(f"   {msg}")

    try:
        from eyegen import pull_model

        ok = pull_model(model_name, progress_callback=on_progress)
        if ok:
            typer.echo(f"✅ Model '{model_name}' is ready")
        else:
            typer.echo(f"❌ Failed to pull model '{model_name}'", err=True)
            raise typer.Exit(1)
    except (OSError, ValueError) as e:
        typer.echo(f"❌ Pull failed: {e}", err=True)
        raise typer.Exit(1)


def list_models():
    """List available models for MLX, MFLUX, OllamaDiffuser (GGUF), Bonsai, and CoreML backends."""
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
        raise typer.Exit(1)

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
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        "-p",
        help="Output directory (default: models/<alias>-<q>bit)",
    ),
):
    """Save a pre-quantized MFLUX model to disk for fast local loading."""
    if path is None:
        q_label = f"{quantize_bits}bit" if quantize_bits else "fp"
        path = MODELS_DIR / f"{model}-{q_label}"

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
        valid, meta = validate_saved_model(result_path)
        if valid and meta:
            ql = meta.get("quantization_level")
            typer.echo(f"   Quantization: {ql}-bit" if ql else "   Quantization: full precision")
        typer.echo("\n💡 To use this model, set the path in your config:")
        typer.echo(f"   ./generate.py config-set mflux_model_path {result_path}")
    except (OSError, ValueError, RuntimeError) as e:
        typer.echo(f"\n❌ Save failed: {e}", err=True)
        raise typer.Exit(1)


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
        raise typer.Exit(1)
