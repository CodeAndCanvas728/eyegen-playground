"""The `generate` command."""

from pathlib import Path
from typing import Annotated, Optional

import typer

from eyegen import (
    Backend,
    QuantizationError,
    generate_image,
)
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
    negative_prompt: Optional[str] = typer.Option(
        None, "--negative-prompt", "-n", help="What to avoid (optional)"
    ),
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path (default: outputs/timestamp.png)"),
    ] = None,
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
    image: Annotated[
        Optional[Path],
        typer.Option(
            "--image", "-i", help="Input image for img2img mode (PNG/JPG/JPEG/BMP/WEBP/TIFF)"
        ),
    ] = None,
    denoise: Optional[float] = typer.Option(
        None,
        "--denoise",
        "-d",
        help="Denoise strength for img2img (0.05=keep original, 1.0=full redraw; default: 0.75)",
        min=0.05,
        max=1.0,
    ),
    use_t5: Optional[bool] = typer.Option(
        None, "--use-t5/--no-t5", help="Use the T5 text encoder (better quality, slower)"
    ),
    offload_to_cpu: Optional[bool] = typer.Option(
        None,
        "--offload-to-cpu/--no-offload-to-cpu",
        help=(
            "sd-cli: place weights in RAM and stream them to the GPU (slower, smaller "
            "footprint; edge devices). Overrides the configured Low Memory mode."
        ),
    ),
    backend: str = typer.Option(
        Backend.AUTO.value,
        "--backend",
        "-b",
        help=(
            "Generation backend: auto (detect by model name), mlx, mflux,"
            " ollamadiffuser, bonsai, coreml"
        ),
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use for this run, instead of the configured default",
    ),
    quantize: Optional[int] = typer.Option(
        None,
        "--quantize",
        "-q",
        help="MFLUX quantization: 4 (default), 8, or omit for no quantization",
    ),
):
    """Generate an image from a text prompt."""
    try:
        (
            config,
            resolved,
            num_steps,
            guidance_scale,
            h,
            w,
            image_path,
            denoise_value,
            output_path,
            quantize,
            negative_prompt_value,
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
            model,
            negative_prompt=negative_prompt,
            use_t5=use_t5,
            offload_to_cpu=offload_to_cpu,
        )
    except ValueError as exc:
        # A model the tool cannot run, a backend that contradicts the model,
        # and a catalogue the user edited badly are all the user's to fix,
        # and all say what is wrong.
        typer.echo(f"❌ {exc}", err=True)
        raise typer.Exit(1) from exc
    resolved_backend = resolved.backend

    for note in resolved.notes:
        typer.echo(f"📁 {note}")

    print_generation_settings(
        resolved.name,
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
        pipeline = load_pipeline(resolved, config, quantize, use_t5=config.use_t5)
        gen_image = generate_image(
            pipeline,
            prompt,
            guidance_scale,
            num_steps,
            w,
            h,
            seed,
            negative_prompt=negative_prompt_value,
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
    except (OSError, ValueError, RuntimeError) as e:
        typer.echo(f"\n❌ Generation failed: {e}", err=True)
        raise typer.Exit(1) from e
