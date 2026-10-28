"""The lines a generate run prints about the settings it will use."""

from __future__ import annotations

from typing import Optional

import typer

from eyegen.config import Backend, EyeGenConfig


def print_generation_settings(
    model: str,
    resolved_backend: Backend,
    config: EyeGenConfig,
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
        Backend.SDCLI: "stable-diffusion.cpp (FLUX.3)",
    }
    backend_label = backend_labels.get(resolved_backend, resolved_backend.value)
    typer.echo("✨ Generating image...")
    typer.echo(f"   Backend: {backend_label}")
    typer.echo(f"   Model: {model}")
    if config.mflux_model_path and resolved_backend == Backend.MFLUX:
        typer.echo(f"   Local model: {config.mflux_model_path}")
    if resolved_backend == Backend.COREML:
        if config.coreml_model_path:
            typer.echo(f"   CoreML model: {config.coreml_model_path}")
        typer.echo(f"   Compute unit: {config.coreml_compute_unit or 'CPU_AND_NE'}")
    if resolved_backend == Backend.BONSAI:
        if config.bonsai_model_path:
            typer.echo(f"   Bonsai model: {config.bonsai_model_path}")
    typer.echo(f"   Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
    if image_path:
        typer.echo(f"   Mode: img2img | Denoise: {denoise_value:.2f} | Input: {image_path}")
    else:
        typer.echo(f"   Steps: {num_steps} | Guidance: {guidance_scale} | Size: {w}x{h}")
    if seed is not None:
        typer.echo(f"   Seed: {seed}")
