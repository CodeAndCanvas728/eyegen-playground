#!/usr/bin/env python3
"""
EyeGen — Multi-backend Image Generator CLI
Command-line interface for quick image generation on Apple Silicon.
Supports MLX (diffusionkit), OllamaDiffuser (GGUF), and MFLUX backends.
"""

import typer
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from core import (
    load_config, save_config, get_pipeline, get_ollama_pipeline,
    get_mflux_pipeline,
    sanitize_prompt, validate_dimensions, validate_image_path,
    generate_image, detect_backend, pull_model,
    list_ollama_models, list_mflux_models, clear_mflux_cache,
    save_mflux_model, validate_saved_model,
    hf_login, hf_status, hf_logout,
    PROJECT_ROOT, CONFIG_FILE, OUTPUT_DIR, MODELS_DIR, DEFAULT_CONFIG,
    BACKEND_AUTO, BACKEND_MLX, BACKEND_OLLAMA, BACKEND_MFLUX,
    VALID_BACKENDS,
    QuantizationError,
)

app = typer.Typer(
    help="Generate images using MLX SD3.5, MFLUX (FLUX/FIBO/Z-Image), or OllamaDiffuser GGUF models",
    no_args_is_help=True
)


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Image description"),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file path (default: outputs/timestamp.png)"
    ),
    steps: Optional[int] = typer.Option(
        None,
        "--steps",
        help="Number of inference steps (default: 30, faster: 20, better: 40)"
    ),
    guidance: Optional[float] = typer.Option(
        None,
        "--guidance",
        help="Guidance scale for prompt adherence (default: 7.5, range: 1.0-15.0)"
    ),
    height: Optional[int] = typer.Option(
        None,
        "--height",
        help="Image height in pixels (default: 1024, must be multiple of 8)"
    ),
    width: Optional[int] = typer.Option(
        None,
        "--width",
        help="Image width in pixels (default: 1024, must be multiple of 8)"
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for reproducibility"
    ),
    image: Optional[Path] = typer.Option(
        None,
        "--image", "-i",
        help="Input image for img2img mode (PNG/JPG/JPEG/BMP/WEBP/TIFF)",
    ),
    denoise: Optional[float] = typer.Option(
        None,
        "--denoise", "-d",
        help="Denoise strength for img2img (0.05=keep original, 1.0=full redraw; default: 0.75)",
        min=0.05, max=1.0,
    ),
    backend: str = typer.Option(
        "auto",
        "--backend", "-b",
        help="Generation backend: auto (detect by model name), mlx, mflux, ollamadiffuser",
    ),
    quantize: Optional[int] = typer.Option(
        None,
        "--quantize", "-q",
        help="MFLUX quantization: 4 (default), 8, or omit for no quantization",
    ),
):
    """Generate an image from a text prompt."""
    if backend not in VALID_BACKENDS:
        typer.echo(f"❌ Invalid backend '{backend}'. Choose from: {', '.join(VALID_BACKENDS)}", err=True)
        raise typer.Exit(1)

    config = load_config()

    model = config.get("model", DEFAULT_CONFIG["model"])
    resolved_backend = detect_backend(model, backend)

    num_steps = steps or config.get("num_inference_steps", 30)
    guidance_scale = guidance or config.get("guidance_scale", 7.5)
    h = height or config.get("height", 1024)
    w = width or config.get("width", 1024)

    # img2img mode setup
    image_path: Optional[str] = None
    denoise_value: float = 1.0
    if image is not None:
        err = validate_image_path(str(image))
        if err:
            typer.echo(f"❌ {err}", err=True)
            raise typer.Exit(1)
        image_path = str(image)
        denoise_value = denoise if denoise is not None else 0.75
        if resolved_backend == BACKEND_MLX:
            typer.echo("⚠  Note: img2img with 4-bit quantized MLX models is known to produce output identical to the input (denoise may have no effect).")
        if (height is not None or width is not None):
            typer.echo("ℹ  Note: --width/--height are ignored in img2img mode (input image dimensions are used)")
    elif denoise is not None:
        typer.echo("⚠  Warning: --denoise has no effect without --image")

    if image_path is None:
        err = validate_dimensions(w, h)
        if err:
            typer.echo(f"❌ {err}", err=True)
            raise typer.Exit(1)

    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = OUTPUT_DIR / f"{timestamp}.png"
    else:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

    backend_labels = {
        BACKEND_MLX: "MLX (diffusionkit)",
        BACKEND_OLLAMA: "OllamaDiffuser (GGUF)",
        BACKEND_MFLUX: "MFLUX (MLX FLUX)",
    }
    backend_label = backend_labels.get(resolved_backend, resolved_backend)
    typer.echo(f"✨ Generating image...")
    typer.echo(f"   Backend: {backend_label}")
    typer.echo(f"   Model: {model}")
    local_model = config.get("mflux_model_path")
    if local_model and resolved_backend == BACKEND_MFLUX:
        typer.echo(f"   Local model: {local_model}")
    typer.echo(f"   Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
    if image_path:
        typer.echo(f"   Mode: img2img | Denoise: {denoise_value:.2f} | Input: {image_path}")
    else:
        typer.echo(f"   Steps: {num_steps} | Guidance: {guidance_scale} | Size: {w}x{h}")

    try:
        typer.echo(f"📦 Loading model...")
        if resolved_backend == BACKEND_OLLAMA:
            pipeline = get_ollama_pipeline(config)
        elif resolved_backend == BACKEND_MFLUX:
            q = quantize if quantize is not None else config.get("mflux_quantize", 4)
            if q is not None:
                typer.echo(f"   Quantize: {q}-bit")
            pipeline = get_mflux_pipeline(config, quantize=q)
        else:
            pipeline = get_pipeline(config)

        if seed is not None:
            typer.echo(f"   Seed: {seed}")

        gen_image = generate_image(
            pipeline, prompt, guidance_scale, num_steps, w, h, seed,
            image_path=image_path, denoise=denoise_value,
            backend=resolved_backend,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        gen_image.save(output)

        typer.echo(f"\n✅ Image saved to: {output}")
        if not image_path:
            typer.echo(f"   Size: {w}x{h} pixels")

    except ImportError as exc:
        if resolved_backend == BACKEND_OLLAMA:
            typer.echo(
                "❌ ollamadiffuser not installed. Install with:\n"
                "  pip install ollamadiffuser",
                err=True,
            )
        elif resolved_backend == BACKEND_MFLUX:
            typer.echo(
                "❌ mflux not installed. Install with:\n"
                "  pip install mflux",
                err=True,
            )
        else:
            typer.echo(
                "❌ diffusionkit not installed. Install with:\n"
                "  pip install -r requirements.txt",
                err=True,
            )
        raise typer.Exit(1)
    except QuantizationError as qe:
        typer.echo(f"\n⚠️  Quantization failed: {qe.original}", err=True)
        typer.echo("   Retrying with full precision (no quantization)...", err=True)
        try:
            pipeline = get_mflux_pipeline(config, quantize=None)
            gen_image = generate_image(
                pipeline, prompt, guidance_scale, num_steps, w, h, seed,
                image_path=image_path, denoise=denoise_value,
                backend=resolved_backend,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            gen_image.save(output)
            typer.echo(f"\n✅ Image saved to: {output}")
            if not image_path:
                typer.echo(f"   Size: {w}x{h} pixels")
            typer.echo("\n💡 Tip: To avoid this warning, set quantize to None:")
            typer.echo("   ./generate.py config-set mflux_quantize null")
            typer.echo("   Or clear the model cache: ./generate.py clear-cache")
        except (OSError, ValueError, QuantizationError, RuntimeError) as retry_err:
            typer.echo(f"\n❌ Retry also failed: {retry_err}", err=True)
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"\n❌ Generation failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def pull(
    model_name: str = typer.Argument(..., help="OllamaDiffuser model name (e.g. flux.1-dev-gguf-q4ks)"),
):
    """Download a GGUF model via OllamaDiffuser."""
    typer.echo(f"📥 Pulling model: {model_name}")

    def on_progress(msg):
        if isinstance(msg, str):
            typer.echo(f"   {msg}")

    try:
        ok = pull_model(model_name, progress_callback=on_progress)
        if ok:
            typer.echo(f"✅ Model '{model_name}' is ready")
        else:
            typer.echo(f"❌ Failed to pull model '{model_name}'", err=True)
            raise typer.Exit(1)
    except (OSError, ValueError) as e:
        typer.echo(f"❌ Pull failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def list_models():
    """List available models for MLX, OllamaDiffuser (GGUF), and MFLUX backends."""

    # MLX Native models
    typer.echo("🔷 MLX Native Models (diffusionkit):")
    typer.echo("  • argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized  (Default, ~3GB)")
    typer.echo("  • mlx-community/Lance-3B-AWQ-INT4                          (Multimodal Image Specialist)")
    typer.echo("  (Downloads from HuggingFace on first use — cached locally)")
    typer.echo()

    # MFLUX models
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

    # OllamaDiffuser models
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
    typer.echo(f"\nPull a GGUF model:  ./generate.py pull <model-name>")


@app.command()
def config_show():
    """Display current configuration."""
    config = load_config()
    typer.echo("📋 Current Configuration:")
    typer.echo(json.dumps(config, indent=2))


@app.command()
def config_set(
    key: str = typer.Argument(..., help="Config key (e.g., 'num_inference_steps')"),
    value: str = typer.Argument(..., help="Config value"),
):
    """Update a configuration value."""
    config = load_config()

    # Try to parse as JSON (for numbers, booleans, etc.)
    try:
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        parsed_value = value

    config[key] = parsed_value
    try:
        save_config(config)
        typer.echo(f"✓ Config saved to {CONFIG_FILE}")
        typer.echo(f"✓ Set {key} = {parsed_value}")
    except ValueError as val_err:
        typer.echo(f"❌ Configuration validation failed:\n   {val_err}", err=True)
        raise typer.Exit(1)
    except OSError as e:
        typer.echo(f"❌ Failed to save config: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def config_reset():
    """Reset configuration to defaults."""
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
    typer.echo("✓ Configuration reset to defaults")


@app.command()
def list_outputs():
    """List all generated images."""
    images = sorted(OUTPUT_DIR.glob("*.png"))

    if not images:
        typer.echo("No images generated yet.")
        return

    typer.echo(f"📁 Generated images ({len(images)}):")
    for img in images:
        size = img.stat().st_size / 1024  # KB
        typer.echo(f"  • {img.name} ({size:.1f} KB)")


@app.command()
def status():
    """Check system status and model availability."""
    typer.echo("🔍 MLX SD 3.5 Workspace Status")
    typer.echo(f"  Project root: {PROJECT_ROOT}")
    typer.echo(f"  Config file: {CONFIG_FILE}")
    typer.echo(f"  Output directory: {OUTPUT_DIR}")

    config = load_config()
    model = config.get("model", "Not set")
    backend_setting = config.get("backend", BACKEND_AUTO)
    resolved = detect_backend(model, backend_setting)

    typer.echo(f"\n⚙️  Configuration:")
    typer.echo(f"  Model: {model}")
    typer.echo(f"  Backend: {backend_setting} → {resolved}")
    typer.echo(f"  Default steps: {config.get('num_inference_steps', 30)}")
    typer.echo(f"  Default guidance: {config.get('guidance_scale', 7.5)}")
    typer.echo(f"  Default size: {config.get('width', 1024)}x{config.get('height', 1024)}")

    # Check diffusionkit
    try:
        import diffusionkit
        typer.echo(f"\n✅ diffusionkit: Installed (v{diffusionkit.__version__})")
    except ImportError:
        typer.echo(f"\n❌ diffusionkit: Not installed")

    # Check ollamadiffuser
    try:
        import ollamadiffuser
        typer.echo(f"✅ ollamadiffuser: Installed (v{ollamadiffuser.__version__})")
        try:
            models = list_ollama_models()
            installed = models.get("installed", [])
            typer.echo(f"   GGUF models installed: {len(installed)}")
            for m in installed[:5]:
                typer.echo(f"     • {m}")
            if len(installed) > 5:
                typer.echo(f"     ... and {len(installed) - 5} more")
        except (OSError, ValueError, ImportError, AttributeError):
            pass
    except ImportError:
        typer.echo(f"❌ ollamadiffuser: Not installed")

    # Check mflux
    try:
        import mflux
        typer.echo(f"✅ mflux: Installed")
        mflux_models = list_mflux_models()
        typer.echo(f"   Available models: {len(mflux_models)}")
        for m in mflux_models[:5]:
            typer.echo(f"     • {m['alias']}")
        if len(mflux_models) > 5:
            typer.echo(f"     ... and {len(mflux_models) - 5} more (run list-models to see all)")
    except ImportError:
        typer.echo(f"❌ mflux: Not installed")

    # Count generated images
    images = list(OUTPUT_DIR.glob("*.png"))
    typer.echo(f"\n📊 Images generated: {len(images)}")

    # HuggingFace login status
    info = hf_status()
    if info:
        typer.echo(f"🔑 HuggingFace: logged in as {info.get('name', 'unknown')}")
    else:
        typer.echo(f"🔑 HuggingFace: not logged in (run hf-login for gated models)")


@app.command(name="clear-cache")
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
            typer.echo(f"✅ Cleared {len(removed)} cached revision(s). Models will re-download on next use.")
        else:
            typer.echo("ℹ  No matching cached models found.")
    except (OSError, ValueError) as e:
        typer.echo(f"❌ Cache clear failed: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="save-model")
def save_model_cmd(
    model: str = typer.Argument(
        ...,
        help="MFLUX model alias (e.g. 'dev', 'schnell', 'fibo')",
    ),
    quantize_bits: Optional[int] = typer.Option(
        4,
        "--quantize", "-q",
        help="Quantization bits: 4 (default), 8, or omit for full precision",
    ),
    path: Optional[Path] = typer.Option(
        None,
        "--path", "-p",
        help="Output directory (default: models/<alias>-<q>bit)",
    ),
):
    """Save a pre-quantized MFLUX model to disk for fast local loading.

    Downloads the model from HuggingFace, quantizes it, and saves weights
    + tokenizers to a local directory.  Subsequent runs can load from this
    path instantly — no download or runtime quantization needed.

    \b
    Examples:
      ./generate.py save-model dev --quantize 4
      ./generate.py save-model schnell --quantize 8 --path ~/models/schnell-8bit
    """
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
        typer.echo(f"\n💡 To use this model, set the path in your config:")
        typer.echo(f"   ./generate.py config-set mflux_model_path {result_path}")
    except Exception as e:
        typer.echo(f"\n❌ Save failed: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="hf-login")
def hf_login_cmd(
    token: Optional[str] = typer.Option(
        None, "--token", "-t",
        help="HuggingFace access token (prompted if not provided)",
    ),
):
    """Log in to HuggingFace to access gated models."""
    if token is None:
        token = typer.prompt("Enter your HuggingFace token", hide_input=True)

    try:
        info = hf_login(token)
        typer.echo(f"✅ Logged in as {info.get('name', 'unknown')}")
    except Exception as e:
        typer.echo(f"❌ Login failed: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="hf-status")
def hf_status_cmd():
    """Check HuggingFace login status."""
    info = hf_status()
    if info:
        typer.echo(f"✅ Logged in as {info.get('name', 'unknown')}")
    else:
        typer.echo("Not logged in. Run hf-login to authenticate.")


@app.command(name="hf-logout")
def hf_logout_cmd():
    """Log out of HuggingFace."""
    try:
        hf_logout()
        typer.echo("✅ Logged out of HuggingFace")
    except Exception as e:
        typer.echo(f"❌ Logout failed: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
