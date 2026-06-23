"""The `generate` command."""

from pathlib import Path
from typing import Optional

import typer

from eyegen import Backend, QuantizationError, generate_image
from eyegen.cli._generate_helpers import (
    build_generation_params,
    handle_import_error,
    handle_quantization_error,
    load_pipeline,
    print_generation_settings,
    save_output_image,
)


def generate(
    prompt: str = typer.Argument(..., help="Image description"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path (default: outputs/timestamp.png)"
    ),
    steps: Optional[int] = typer.Option(
        None, "--steps", help="Number of inference steps (default: 30, faster: 20, better: 40)"
    ),
    guidance: Optional[float] = typer.Option(
        None,
        "--guidance",
        help="Guidance scale for prompt adherence (default: 7.5, range: 1.0-15.0)",
    ),
    height: Optional[int] = typer.Option(
        None, "--height", help="Image height in pixels (default: 1024, must be multiple of 8)"
    ),
    width: Optional[int] = typer.Option(
        None, "--width", help="Image width in pixels (default: 1024, must be multiple of 8)"
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducibility"),
    image: Optional[Path] = typer.Option(
        None,
        "--image",
        "-i",
        help="Input image for img2img mode (PNG/JPG/JPEG/BMP/WEBP/TIFF)",
    ),
    denoise: Optional[float] = typer.Option(
        None,
        "--denoise",
        "-d",
        help="Denoise strength for img2img (0.05=keep original, 1.0=full redraw; default: 0.75)",
        min=0.05,
        max=1.0,
    ),
    backend: str = typer.Option(
        Backend.AUTO.value,
        "--backend",
        "-b",
        help="Generation backend: auto (detect by model name), mlx, mflux, ollamadiffuser, bonsai, coreml",
    ),
    quantize: Optional[int] = typer.Option(
        None,
        "--quantize",
        "-q",
        help="MFLUX quantization: 4 (default), 8, or omit for no quantization",
    ),
):
    """Generate an image from a text prompt."""
    (
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
    ) = build_generation_params(
        backend,
        output,
        steps,
        guidance,
        height,
        width,
        image,
        denoise,
        quantize,
    )

    print_generation_settings(
        model,
        resolved_backend,
        config,
        prompt,
        image_path,
        denoise_value,
        w,
        h,
        num_steps,
        guidance_scale,
        seed,
    )

    try:
        pipeline = load_pipeline(resolved_backend, config, quantize)
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
        save_output_image(gen_image, output_path, w, h, image_path)
    except ImportError:
        handle_import_error(resolved_backend)
    except QuantizationError as qe:
        handle_quantization_error(
            qe,
            config,
            prompt,
            guidance_scale,
            num_steps,
            w,
            h,
            seed,
            image_path,
            denoise_value,
            resolved_backend,
            output_path,
        )
    except Exception as e:
        typer.echo(f"\n❌ Generation failed: {e}", err=True)
        raise typer.Exit(1)
