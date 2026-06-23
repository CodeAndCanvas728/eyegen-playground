"""Internal helpers for the `generate` CLI command."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import typer

from eyegen import (
    DEFAULT_CONFIG,
    OUTPUT_DIR,
    Backend,
    QuantizationError,
    detect_backend,
    generate_image,
    get_mflux_pipeline,
    get_ollama_pipeline,
    get_pipeline,
    load_config,
    validate_dimensions,
    validate_image_path,
)
from eyegen.backends import bonsai, coreml


def validate_cli_backend(backend: str) -> str:
    valid = {b.value for b in Backend}
    if backend not in valid:
        typer.echo(
            f"❌ Invalid backend '{backend}'. Choose from: {', '.join(sorted(valid))}", err=True
        )
        raise typer.Exit(1)
    return backend


def resolve_output(output: Optional[Path]) -> Path:
    if output is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return OUTPUT_DIR / f"{timestamp}.png"
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def setup_img2img(
    image: Optional[Path],
    denoise: Optional[float],
    resolved_backend: Backend,
    height: Optional[int],
    width: Optional[int],
) -> Tuple[Optional[str], float]:
    if image is None:
        if denoise is not None:
            typer.echo("⚠  Warning: --denoise has no effect without --image")
        return None, 1.0

    err = validate_image_path(str(image))
    if err:
        typer.echo(f"❌ {err}", err=True)
        raise typer.Exit(1)

    image_path = str(image)
    denoise_value = denoise if denoise is not None else 0.75
    if resolved_backend == Backend.MLX:
        typer.echo(
            "⚠  Note: img2img with 4-bit quantized MLX models is known to produce "
            "output identical to the input (denoise may have no effect)."
        )
    if height is not None or width is not None:
        typer.echo(
            "ℹ  Note: --width/--height are ignored in img2img mode "
            "(input image dimensions are used)"
        )
    return image_path, denoise_value


def print_generation_settings(
    model: str,
    resolved_backend: Backend,
    config: dict,
    prompt: str,
    image_path: Optional[str],
    denoise_value: float,
    w: int,
    h: int,
    num_steps: int,
    guidance_scale: float,
    seed: Optional[int],
):
    backend_labels = {
        Backend.MLX: "MLX (diffusionkit)",
        Backend.OLLAMA: "OllamaDiffuser (GGUF)",
        Backend.MFLUX: "MFLUX (MLX FLUX)",
        Backend.BONSAI: "Bonsai (PrismML ternary 1.58-bit)",
        Backend.COREML: "CoreML (Apple Neural Engine)",
    }
    backend_label = backend_labels.get(resolved_backend, resolved_backend.value)
    typer.echo("✨ Generating image...")
    typer.echo(f"   Backend: {backend_label}")
    typer.echo(f"   Model: {model}")
    local_model = config.get("mflux_model_path")
    if local_model and resolved_backend == Backend.MFLUX:
        typer.echo(f"   Local model: {local_model}")
    if resolved_backend == Backend.COREML:
        coreml_path = config.get("coreml_model_path")
        if coreml_path:
            typer.echo(f"   CoreML model: {coreml_path}")
        typer.echo(f"   Compute unit: {config.get('coreml_compute_unit', 'CPU_AND_NE')}")
    if resolved_backend == Backend.BONSAI:
        bonsai_path = config.get("bonsai_model_path")
        if bonsai_path:
            typer.echo(f"   Bonsai model: {bonsai_path}")
    typer.echo(f"   Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
    if image_path:
        typer.echo(f"   Mode: img2img | Denoise: {denoise_value:.2f} | Input: {image_path}")
    else:
        typer.echo(f"   Steps: {num_steps} | Guidance: {guidance_scale} | Size: {w}x{h}")
    if seed is not None:
        typer.echo(f"   Seed: {seed}")


def load_pipeline(resolved_backend: Backend, config: dict, quantize: Optional[int]):
    typer.echo("📦 Loading model...")
    if resolved_backend == Backend.OLLAMA:
        return get_ollama_pipeline(config)
    if resolved_backend == Backend.MFLUX:
        q = quantize if quantize is not None else config.get("mflux_quantize", 4)
        if q is not None:
            typer.echo(f"   Quantize: {q}-bit")
        return get_mflux_pipeline(config, quantize=q)
    if resolved_backend == Backend.BONSAI:
        return bonsai.get_bonsai_pipeline(config)
    if resolved_backend == Backend.COREML:
        return coreml.get_coreml_pipeline(config)
    return get_pipeline(config)


def save_output_image(gen_image, output: Path, w: int, h: int, image_path: Optional[str]):
    output.parent.mkdir(parents=True, exist_ok=True)
    gen_image.save(output)
    typer.echo(f"\n✅ Image saved to: {output}")
    if not image_path:
        typer.echo(f"   Size: {w}x{h} pixels")


def handle_import_error(resolved_backend: Backend):
    if resolved_backend == Backend.OLLAMA:
        typer.echo(
            "❌ ollamadiffuser not installed. Install with:\n  pip install ollamadiffuser",
            err=True,
        )
    elif resolved_backend == Backend.MFLUX:
        typer.echo(
            "❌ mflux not installed. Install with:\n  pip install mflux",
            err=True,
        )
    else:
        typer.echo(
            "❌ diffusionkit not installed. Install with:\n  pip install -r requirements.txt",
            err=True,
        )
    raise typer.Exit(1)


def handle_quantization_error(
    qe: QuantizationError,
    config: dict,
    prompt: str,
    guidance_scale: float,
    num_steps: int,
    w: int,
    h: int,
    seed: Optional[int],
    image_path: Optional[str],
    denoise_value: float,
    resolved_backend: Backend,
    output: Path,
):
    typer.echo(f"\n⚠️  Quantization failed: {qe.original}", err=True)
    typer.echo("   Retrying with full precision (no quantization)...", err=True)
    try:
        pipeline = get_mflux_pipeline(config, quantize=None)
        gen_image = generate_image(
            pipeline,
            prompt,
            guidance_scale,
            num_steps,
            w,
            h,
            seed,
            image_path=image_path,
            denoise=denoise_value,
            backend=resolved_backend,
        )
        save_output_image(gen_image, output, w, h, image_path)
        typer.echo("\n💡 Tip: To avoid this warning, set quantize to None:")
        typer.echo("   ./generate.py config-set mflux_quantize null")
        typer.echo("   Or clear the model cache: ./generate.py clear-cache")
    except (OSError, ValueError, QuantizationError, RuntimeError) as retry_err:
        typer.echo(f"\n❌ Retry also failed: {retry_err}", err=True)
        raise typer.Exit(1) from retry_err


def build_generation_params(
    backend: str,
    output: Optional[Path],
    steps: Optional[int],
    guidance: Optional[float],
    height: Optional[int],
    width: Optional[int],
    image: Optional[Path],
    denoise: Optional[float],
    quantize: Optional[int],
):
    validate_cli_backend(backend)
    config = load_config()

    model = config.get("model", DEFAULT_CONFIG["model"])
    resolved_backend = detect_backend(model, backend, config=config)

    num_steps = steps or config.get("num_inference_steps", 30)
    guidance_scale = guidance or config.get("guidance_scale", 7.5)
    h = height or config.get("height", 1024)
    w = width or config.get("width", 1024)

    image_path, denoise_value = setup_img2img(image, denoise, resolved_backend, height, width)

    if image_path is None:
        err = validate_dimensions(w, h)
        if err:
            typer.echo(f"❌ {err}", err=True)
        raise typer.Exit(1)

    output_path = resolve_output(output)
    return (
        config,
        model,
        resolved_backend,
        num_steps,
        guidance_scale,
        h,
        w,
        image_path,
        denoise_value,
        output_path,
        quantize,
    )
