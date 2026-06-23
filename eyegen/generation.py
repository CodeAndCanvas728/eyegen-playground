"""Image generation dispatcher and backend-specific helpers."""

import logging
import random
from typing import Optional

from eyegen.config import Backend
from eyegen.errors import QuantizationError, _is_quantization_error
from eyegen.validation import sanitize_prompt

log = logging.getLogger(__name__)


def _generate_image_mlx(
    pipeline,
    prompt: str,
    cfg_weight: float,
    num_steps: int,
    width: int,
    height: int,
    seed: Optional[int] = None,
    negative_prompt: str = "",
    image_path: Optional[str] = None,
    denoise: float = 1.0,
):
    """Run the MLX/diffusionkit pipeline and return a PIL Image."""
    image, _latent = pipeline.generate_image(
        prompt,
        cfg_weight=cfg_weight,
        num_steps=num_steps,
        latent_size=(height // 8, width // 8),
        seed=seed,
        negative_text=negative_prompt,
        image_path=image_path,
        denoise=denoise,
    )
    return image


def _generate_image_ollama(
    engine,
    prompt: str,
    cfg_weight: float,
    num_steps: int,
    width: int,
    height: int,
    seed: Optional[int] = None,
    negative_prompt: str = "",
    image_path: Optional[str] = None,
    denoise: float = 1.0,
):
    """Run ollamadiffuser engine and return a PIL Image."""
    kwargs = {}

    if image_path:
        from PIL import Image as PILImage

        kwargs["image"] = PILImage.open(image_path).convert("RGB")
        kwargs["strength"] = denoise

    if seed is not None:
        try:
            import torch

            device = engine.device if hasattr(engine, "device") else "cpu"
            if device == "mps":
                kwargs["generator"] = torch.Generator("cpu").manual_seed(seed)
            else:
                kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)
        except Exception as exc:
            log.debug("Seed not supported by engine, continuing: %s", exc)

    return engine.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt
        or "low quality, bad anatomy, worst quality, low resolution",
        num_inference_steps=num_steps,
        guidance_scale=cfg_weight,
        width=width,
        height=height,
        **kwargs,
    )


def _generate_image_mflux(
    model,
    prompt: str,
    cfg_weight: float,
    num_steps: int,
    width: int,
    height: int,
    seed: int | None = None,
    negative_prompt: str = "",
    image_path: str | None = None,
    denoise: float = 1.0,
):
    """Run an MFLUX model and return a PIL Image."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)  # noqa: S311

    kwargs = {
        "seed": seed,
        "prompt": prompt,
        "num_inference_steps": num_steps,
        "width": width,
        "height": height,
        "guidance": cfg_weight,
    }

    if image_path:
        kwargs["image_path"] = image_path
        kwargs["image_strength"] = denoise

    if negative_prompt:
        try:
            import inspect

            sig = inspect.signature(model.generate_image)
            if "negative_prompt" in sig.parameters:
                kwargs["negative_prompt"] = negative_prompt
        except Exception as exc:
            log.debug("Could not check negative_prompt support: %s", exc)

    try:
        result = model.generate_image(**kwargs)
    except ValueError as exc:
        if _is_quantization_error(exc):
            model_config = getattr(model, "model_config", None)
            model_name = (
                getattr(model_config, "model_name", "unknown") if model_config else "unknown"
            )
            raise QuantizationError(exc, model_name) from exc
        raise

    if hasattr(result, "image"):
        return result.image
    return result


def generate_image(
    pipeline,
    prompt: str,
    cfg_weight: float,
    num_steps: int,
    width: int,
    height: int,
    seed: Optional[int] = None,
    negative_prompt: str = "",
    image_path: Optional[str] = None,
    denoise: float = 1.0,
    backend: Backend = Backend.MLX,
):
    """Run the diffusion pipeline and return a PIL Image."""
    prompt = sanitize_prompt(prompt)
    negative_prompt = sanitize_prompt(negative_prompt) if negative_prompt else ""

    if backend == Backend.OLLAMA:
        return _generate_image_ollama(
            pipeline,
            prompt,
            cfg_weight,
            num_steps,
            width,
            height,
            seed,
            negative_prompt,
            image_path,
            denoise,
        )

    if backend == Backend.MFLUX:
        return _generate_image_mflux(
            pipeline,
            prompt,
            cfg_weight,
            num_steps,
            width,
            height,
            seed,
            negative_prompt,
            image_path,
            denoise,
        )

    if backend in (Backend.BONSAI, Backend.COREML):
        return pipeline.generate_image(
            prompt=prompt,
            cfg_weight=cfg_weight,
            num_steps=num_steps,
            width=width,
            height=height,
            seed=seed,
            negative_prompt=negative_prompt,
            image_path=image_path,
            denoise=denoise,
        )

    return _generate_image_mlx(
        pipeline,
        prompt,
        cfg_weight,
        num_steps,
        width,
        height,
        seed,
        negative_prompt,
        image_path,
        denoise,
    )
