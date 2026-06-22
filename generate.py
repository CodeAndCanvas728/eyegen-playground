#!/usr/bin/env python3
"""
EyeGen — Multi-backend Image Generator CLI
Command-line interface for quick image generation on Apple Silicon.
Supports MLX (diffusionkit), OllamaDiffuser (GGUF), MFLUX, Bonsai (PrismML),
and CoreML (Apple Neural Engine) backends.
"""

import typer
import json
import subprocess
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
    BACKEND_BONSAI, BACKEND_COREML,
    VALID_BACKENDS,
    QuantizationError,
)
import core_bonsai
import core_coreml

app = typer.Typer(
    help="Generate images using MLX SD3.5, MFLUX (FLUX/FIBO/Z-Image), OllamaDiffuser GGUF, Bonsai (PrismML ternary), or CoreML (Apple Neural Engine) backends",
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
        help="Generation backend: auto (detect by model name), mlx, mflux, ollamadiffuser, bonsai, coreml",
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
    resolved_backend = detect_backend(model, backend, config=config)

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
        BACKEND_BONSAI: "Bonsai (PrismML ternary 1.58-bit)",
        BACKEND_COREML: "CoreML (Apple Neural Engine)",
    }
    backend_label = backend_labels.get(resolved_backend, resolved_backend)
    typer.echo(f"✨ Generating image...")
    typer.echo(f"   Backend: {backend_label}")
    typer.echo(f"   Model: {model}")
    local_model = config.get("mflux_model_path")
    if local_model and resolved_backend == BACKEND_MFLUX:
        typer.echo(f"   Local model: {local_model}")
    if resolved_backend == BACKEND_COREML:
        coreml_path = config.get("coreml_model_path")
        if coreml_path:
            typer.echo(f"   CoreML model: {coreml_path}")
        typer.echo(f"   Compute unit: {config.get('coreml_compute_unit', 'CPU_AND_NE')}")
    if resolved_backend == BACKEND_BONSAI:
        bonsai_path = config.get("bonsai_model_path")
        if bonsai_path:
            typer.echo(f"   Bonsai model: {bonsai_path}")
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
        elif resolved_backend == BACKEND_BONSAI:
            pipeline = core_bonsai.get_bonsai_pipeline(config)
        elif resolved_backend == BACKEND_COREML:
            pipeline = core_coreml.get_coreml_pipeline(config)
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
        except Exception as retry_err:
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
    except Exception as e:
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
    except Exception as e:
        typer.echo(f"❌ Failed to list MFLUX models: {e}", err=True)

    typer.echo()

    # OllamaDiffuser models
    try:
        models = list_ollama_models()
    except Exception as e:
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

    # Bonsai (PrismML)
    typer.echo()
    bonsai_status = core_bonsai.validate_bonsai_install()
    if bonsai_status.installed:
        bonsai_models = core_bonsai.list_bonsai_models()
        typer.echo(f"🌳 Bonsai (PrismML) — installed ({len(bonsai_models)} model(s)):")
        for m in bonsai_models:
            typer.echo(f"  • {m['alias']:20s}  {m['path']}")
    else:
        typer.echo("🌳 Bonsai (PrismML) — not installed")
        typer.echo("   Install:  ./generate.py setup-bonsai")
        typer.echo("   Then:     ./generate.py pull-bonsai ternary-mlx")

    # CoreML (Apple Neural Engine)
    typer.echo()
    coreml_status = core_coreml.validate_coreml_install()
    if coreml_status.installed:
        coreml_models = core_coreml.list_coreml_models()
        typer.echo(f"🍎 CoreML (Apple Neural Engine) — installed ({len(coreml_models)} model(s)):")
        for m in coreml_models:
            typer.echo(f"  • {m['name']}")
    else:
        typer.echo("🍎 CoreML (Apple Neural Engine) — not installed")
        typer.echo("   Install:  ./generate.py setup-coreml")
        typer.echo("   Then:     ./generate.py pull-coreml sd-2-1-base-palettized")
        typer.echo("   Or:       ./generate.py convert-coreml stabilityai/stable-diffusion-2-1-base")


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
    except:
        parsed_value = value

    config[key] = parsed_value
    try:
        save_config(config)
        typer.echo(f"✓ Config saved to {CONFIG_FILE}")
        typer.echo(f"✓ Set {key} = {parsed_value}")
    except ValueError as val_err:
        typer.echo(f"❌ Configuration validation failed:\n   {val_err}", err=True)
        raise typer.Exit(1)
    except Exception as e:
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
    resolved = detect_backend(model, backend_setting, config=config)

    typer.echo(f"\n⚙️  Configuration:")
    typer.echo(f"  Model: {model}")
    typer.echo(f"  Backend: {backend_setting} → {resolved}")
    typer.echo(f"  Default steps: {config.get('num_inference_steps', 30)}")
    typer.echo(f"  Default guidance: {config.get('guidance_scale', 7.5)}")
    typer.echo(f"  Default size: {config.get('width', 1024)}x{config.get('height', 1024)}")

    # Check diffusionkit
    try:
        import diffusionkit
        from importlib.metadata import version as _pkg_version
        try:
            v = _pkg_version("diffusionkit")
        except Exception:
            v = "unknown"
        typer.echo(f"\n✅ diffusionkit: Installed (v{v})")
    except ImportError:
        typer.echo(f"\n❌ diffusionkit: Not installed")

    # Check ollamadiffuser
    try:
        import ollamadiffuser
        from importlib.metadata import version as _pkg_version
        try:
            v = _pkg_version("ollamadiffuser")
        except Exception:
            v = "unknown"
        typer.echo(f"✅ ollamadiffuser: Installed (v{v})")
        try:
            models = list_ollama_models()
            installed = models.get("installed", [])
            typer.echo(f"   GGUF models installed: {len(installed)}")
            for m in installed[:5]:
                typer.echo(f"     • {m}")
            if len(installed) > 5:
                typer.echo(f"     ... and {len(installed) - 5} more")
        except Exception:
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

    # Check bonsai
    bonsai_status = core_bonsai.validate_bonsai_install()
    if bonsai_status.installed:
        bonsai_models = core_bonsai.list_bonsai_models()
        typer.echo(f"\n🌳 Bonsai (PrismML): Installed")
        typer.echo(f"   Vendor: {bonsai_status.bonsai_dir}")
        typer.echo(f"   Models: {len(bonsai_models)}")
        for m in bonsai_models:
            typer.echo(f"     • {m['alias']}")
    else:
        typer.echo(f"\n🌳 Bonsai (PrismML): Not installed (run setup-bonsai)")

    # Check coreml
    coreml_status = core_coreml.validate_coreml_install()
    if coreml_status.installed:
        coreml_models = core_coreml.list_coreml_models()
        typer.echo(f"\n🍎 CoreML: Installed (venv={coreml_status.venv_python})")
        typer.echo(f"   Models: {len(coreml_models)}")
        for m in coreml_models:
            typer.echo(f"     • {m['name']}")
    else:
        typer.echo(f"\n🍎 CoreML: Not installed (run setup-coreml)")

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
    except Exception as e:
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
    """Log out from HuggingFace."""
    try:
        hf_logout()
        typer.echo("✅ Logged out from HuggingFace")
    except Exception as e:
        typer.echo(f"❌ Logout failed: {e}", err=True)
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Bonsai (PrismML) commands
# ---------------------------------------------------------------------------


@app.command(name="setup-bonsai")
def setup_bonsai_cmd():
    """One-time install: clones the Bonsai-Image-Demo and runs its setup.sh.

    Installs a dedicated Python 3.11 venv at ~/models/eyegen/bonsai-demo/.venv/
    with the patched mflux + MLX kernels needed for the 1.58-bit ternary and
    1-bit binary weight formats.

    Re-run to update the bonsai vendor to the latest commit.
    """
    script = PROJECT_ROOT / "scripts" / "setup-bonsai.sh"
    if not script.is_file():
        typer.echo(f"❌ Setup script not found: {script}", err=True)
        raise typer.Exit(1)

    typer.echo("🌳 Setting up Bonsai backend (one-time install) ...")
    typer.echo("   This clones the Bonsai-Image-Demo repo and creates a dedicated")
    typer.echo("   Python 3.11 venv with the patched mflux + MLX kernels.")
    typer.echo("   May take several minutes on first run.")
    typer.echo()

    rc = subprocess.run([str(script)], cwd=PROJECT_ROOT).returncode
    if rc != 0:
        typer.echo(f"❌ Bonsai setup failed (exit {rc})", err=True)
        raise typer.Exit(1)


@app.command(name="pull-bonsai")
def pull_bonsai_cmd(
    variant: str = typer.Option(
        core_bonsai.DEFAULT_VARIANT,
        "--variant", "-v",
        help=f"Variant to download: {', '.join(core_bonsai.SUPPORTED_VARIANTS)}",
    ),
):
    """Download a Bonsai (PrismML) model via the bonsai-demo's download script.

    \b
    Examples:
      ./generate.py pull-bonsai                  # default: ternary-mlx
      ./generate.py pull-bonsai --variant binary-mlx
    """
    if variant not in core_bonsai.SUPPORTED_VARIANTS:
        typer.echo(
            f"❌ Unknown variant '{variant}'. Supported: "
            f"{', '.join(core_bonsai.SUPPORTED_VARIANTS)}",
            err=True,
        )
        raise typer.Exit(1)

    status = core_bonsai.validate_bonsai_install()
    if not status.installed:
        typer.echo(f"❌ {status.message}", err=True)
        raise typer.Exit(1)

    typer.echo(f"📥 Downloading bonsai variant: {variant}")
    ok = core_bonsai.download_bonsai_model(
        variant, progress_callback=lambda m: typer.echo(f"   {m}"),
    )
    if ok:
        typer.echo(f"✅ Bonsai variant '{variant}' is ready at {core_bonsai.get_bonsai_dir()}/models/")
    else:
        typer.echo(f"❌ Failed to download bonsai variant '{variant}'", err=True)
        raise typer.Exit(1)


@app.command(name="list-bonsai-models")
def list_bonsai_models_cmd():
    """List installed Bonsai (PrismML) models."""
    status = core_bonsai.validate_bonsai_install()
    if not status.installed:
        typer.echo(f"❌ {status.message}", err=True)
        typer.echo("   Run: ./generate.py setup-bonsai", err=True)
        raise typer.Exit(1)

    models = core_bonsai.list_bonsai_models()
    if not models:
        typer.echo("📦 No bonsai models installed yet.")
        typer.echo(f"   Available variants: {', '.join(core_bonsai.SUPPORTED_VARIANTS)}")
        typer.echo("   Run: ./generate.py pull-bonsai")
    else:
        typer.echo(f"🌳 Installed bonsai models ({len(models)}):")
        for m in models:
            typer.echo(f"  • {m['alias']:20s}  {m['path']}")


# ---------------------------------------------------------------------------
# CoreML (Apple Neural Engine) commands
# ---------------------------------------------------------------------------


@app.command(name="setup-coreml")
def setup_coreml_cmd():
    """One-time install: creates the CoreML sidecar Python 3.11 venv.

    Installs Apple's ``python_coreml_stable_diffusion`` and its pinned deps
    in a sidecar venv at ``~/models/eyegen/.coreml-venv/`` so it does not
    conflict with EyeGen's main Python 3.14 venv.

    Requires ``/opt/homebrew/bin/python3.11`` (or set ``PYTHON_BIN`` to
    another 3.11 interpreter).
    """
    script = PROJECT_ROOT / "scripts" / "setup-coreml.sh"
    if not script.is_file():
        typer.echo(f"❌ Setup script not found: {script}", err=True)
        raise typer.Exit(1)

    typer.echo("🍎 Setting up CoreML backend (one-time install) ...")
    typer.echo("   This creates a sidecar Python 3.11 venv and installs Apple's")
    typer.echo("   python_coreml_stable_diffusion + its pinned dependencies.")
    typer.echo("   May take several minutes on first run.")
    typer.echo()

    rc = subprocess.run([str(script)], cwd=PROJECT_ROOT).returncode
    if rc != 0:
        typer.echo(f"❌ CoreML setup failed (exit {rc})", err=True)
        raise typer.Exit(1)


@app.command(name="pull-coreml")
def pull_coreml_cmd(
    alias: str = typer.Argument(
        "sd-2-1-base-palettized",
        help=(
            "Alias (sd-1-4, sd-1-5, sd-2-1-base, sd-2-1-base-palettized, "
            "sdxl-base, sd-3-medium, ...) or a full HuggingFace repo id."
        ),
    ),
):
    """Download a pre-converted CoreML model from Hugging Face.

    \b
    Examples:
      ./generate.py pull-coreml                                # default
      ./generate.py pull-coreml sd-2-1-base-palettized
      ./generate.py pull-coreml apple/coreml-stable-diffusion-v1-5
    """
    status = core_coreml.validate_coreml_install()
    if not status.installed:
        typer.echo(f"❌ {status.message}", err=True)
        typer.echo("   Run: ./generate.py setup-coreml", err=True)
        raise typer.Exit(1)

    typer.echo(f"📥 Downloading pre-converted CoreML model: {alias}")
    target = core_coreml.pull_preconverted_coreml_model(
        alias, progress_callback=lambda m: typer.echo(f"   {m}"),
    )
    if target:
        typer.echo(f"✅ CoreML model downloaded to {target}")
        typer.echo(f"   Use it via: ./generate.py generate 'prompt' --backend coreml \\")
        typer.echo(f"       --model {alias}")
    else:
        typer.echo(f"❌ Failed to download CoreML model '{alias}'", err=True)
        raise typer.Exit(1)


@app.command(name="convert-coreml")
def convert_coreml_cmd(
    hf_model_id: str = typer.Argument(
        "stabilityai/stable-diffusion-2-1-base",
        help="HuggingFace model id of the PyTorch model to convert (e.g. stabilityai/stable-diffusion-2-1-base)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory (default: ~/models/eyegen/coreml/<name>)",
    ),
    compute_unit: str = typer.Option(
        "CPU_AND_NE",
        "--compute-unit", "-c",
        help="CoreML compute unit: CPU_AND_NE (mobile), CPU_AND_GPU (Mac), CPU_ONLY, ALL",
    ),
    quantize_nbits: Optional[int] = typer.Option(
        None,
        "--quantize-nbits", "-q",
        help="Palettization bit-width: 2, 4, 6, or 8 (reduces size; quality may drop)",
    ),
    attention: str = typer.Option(
        "SPLIT_EINSUM",
        "--attention", "-a",
        help="Attention implementation: SPLIT_EINSUM (NE), SPLIT_EINSUM_V2 (NE, slower compile), ORIGINAL (CPU/GPU)",
    ),
):
    """Convert a PyTorch Stable Diffusion model to CoreML (15-20 min on M1 Pro).

    \b
    Examples:
      ./generate.py convert-coreml stabilityai/stable-diffusion-2-1-base
      ./generate.py convert-coreml stabilityai/stable-diffusion-2-1-base --quantize-nbits 6
      ./generate.py convert-coreml stabilityai/stable-diffusion-xl-base-1.0 --compute-unit CPU_AND_GPU
    """
    if output is None:
        output = core_coreml.get_coreml_models_dir() / hf_model_id.split("/")[-1]
    output = output.expanduser()

    status = core_coreml.validate_coreml_install()
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

    ok = core_coreml.convert_to_coreml(
        hf_model_id=hf_model_id,
        output_dir=output,
        compute_unit=compute_unit,
        attention_implementation=attention,
        quantize_nbits=quantize_nbits,
        progress_callback=lambda m: typer.echo(f"   {m}"),
    )
    if ok:
        typer.echo(f"\n✅ Converted model at {output}")
        typer.echo(f"   Use it via: ./generate.py generate 'prompt' --backend coreml \\")
        typer.echo(f"       --model {output.name}")
    else:
        typer.echo(f"\n❌ Conversion failed. See eyegen.log for details.", err=True)
        raise typer.Exit(1)


@app.command(name="list-coreml-models")
def list_coreml_models_cmd():
    """List installed CoreML model bundles."""
    status = core_coreml.validate_coreml_install()
    if not status.installed:
        typer.echo(f"❌ {status.message}", err=True)
        typer.echo("   Run: ./generate.py setup-coreml", err=True)
        raise typer.Exit(1)

    models = core_coreml.list_coreml_models()
    if not models:
        typer.echo("📦 No CoreML models installed yet.")
        typer.echo("   Pre-converted (fast):  ./generate.py pull-coreml sd-2-1-base-palettized")
        typer.echo("   Convert from PyTorch:  ./generate.py convert-coreml stabilityai/stable-diffusion-2-1-base")
    else:
        typer.echo(f"🍎 Installed CoreML models ({len(models)}):")
        for m in models:
            typer.echo(f"  • {m['name']:30s}  {m['path']}")
            typer.echo(f"      model_version: {m['model_version']}")
            typer.echo(f"      compute_unit: {m['compute_unit']}  format: {m['format']}")


if __name__ == "__main__":
    app()
