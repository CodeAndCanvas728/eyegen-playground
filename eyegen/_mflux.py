"""Internal MFLUX helpers."""

import logging

log = logging.getLogger(__name__)

_mflux_aliases: set | None = None


def _get_mflux_aliases() -> set:
    """Return the set of all known MFLUX model aliases and HF model names."""
    global _mflux_aliases
    if _mflux_aliases is not None:
        return _mflux_aliases

    _mflux_aliases = {
        "dev",
        "schnell",
        "dev-kontext",
        "dev-fill",
        "dev-redux",
        "dev-depth",
        "dev-controlnet-canny",
        "schnell-controlnet-canny",
        "dev-controlnet-upscaler",
        "dev-fill-catvton",
        "krea-dev",
        "flux2-klein-4b",
        "flux2-klein-9b",
        "flux2-klein-base-4b",
        "flux2-klein-base-9b",
        "flux2-klein-4B",
        "flux2-klein-9B",
        "klein-4b",
        "klein-9b",
        "klein-base-4b",
        "klein-base-9b",
        "qwen-image",
        "qwen-image-edit",
        "qwen",
        "qwen-edit",
        "fibo",
        "fibo-lite",
        "fibo-edit",
        "fibo-edit-rmbg",
        "z-image",
        "z-image-turbo",
        "zimage",
        "zimage-turbo",
        "seedvr2-3b",
        "seedvr2-7b",
        "seedvr2",
    }

    try:
        from mflux.models.common.config.model_config import AVAILABLE_MODELS

        for cfg in AVAILABLE_MODELS.values():
            _mflux_aliases.update(cfg.aliases)
            _mflux_aliases.add(cfg.model_name)
    except (ImportError, AttributeError) as exc:
        log.debug("Could not augment MFLUX aliases from live package: %s", exc)

    return _mflux_aliases


def _resolve_mflux_class(model_config):
    """Given a resolved ModelConfig, return the appropriate model class."""
    name = model_config.model_name
    aliases = set(model_config.aliases)

    klein_names = {
        "black-forest-labs/FLUX.2-klein-4B",
        "black-forest-labs/FLUX.2-klein-9B",
        "black-forest-labs/FLUX.2-klein-base-4B",
        "black-forest-labs/FLUX.2-klein-base-9B",
    }
    klein_aliases = {
        "flux2-klein-4b", "flux2-klein-4B", "flux2-klein", "klein-4b", "klein-4B",
        "flux2-klein-9b", "flux2-klein-9B", "klein-9b", "klein-9B",
        "flux2-klein-base-4b", "flux2-klein-base-4B", "flux2-base-4b", "flux2-base-4B",
        "klein-base-4b", "klein-base-4B", "flux2-klein-base-9b", "flux2-klein-base-9B",
        "flux2-base-9b", "flux2-base-9B", "klein-base-9b", "klein-base-9B",
    }
    if name in klein_names or not aliases.isdisjoint(klein_aliases):
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
        return Flux2Klein

    z_names = {
        "Tongyi-MAI/Z-Image",
        "Tongyi-MAI/Z-Image-Turbo",
    }
    z_aliases = {
        "z-image", "zimage", "z-image-turbo", "zimage-turbo",
    }
    if name in z_names or not aliases.isdisjoint(z_aliases):
        from mflux.models.z_image.variants.z_image import ZImage
        return ZImage

    fibo_names = {
        "briaai/FIBO",
        "briaai/Fibo-lite",
        "briaai/Fibo-Edit",
        "briaai/Fibo-Edit-RMBG",
    }
    fibo_aliases = {
        "fibo", "fibo-lite", "fibo_lite", "fibo-edit", "fiboedit",
        "fibo-edit-rmbg", "fiboedit-rmbg",
    }
    if name in fibo_names or not aliases.isdisjoint(fibo_aliases):
        from mflux.models.fibo.variants.txt2img.fibo import FIBO
        return FIBO

    qwen_names = {
        "Qwen/Qwen-Image",
        "Qwen/Qwen-Image-Edit-2509",
    }
    qwen_aliases = {
        "qwen-image", "qwen", "qwen-image-edit", "qwen-edit", "qwen-edit-plus", "qwen-edit-2509",
    }
    if name in qwen_names or not aliases.isdisjoint(qwen_aliases):
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
        return QwenImage

    seed_names = {
        "numz/SeedVR2_comfyUI",
    }
    seed_aliases = {
        "seedvr2-3b", "seedvr2", "seedvr2-7b", "seedvr2-7B",
    }
    if name in seed_names or not aliases.isdisjoint(seed_aliases):
        from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
        return SeedVR2

    from mflux.models.flux.variants.txt2img.flux import Flux1
    return Flux1


def _format_mflux_model_error(model_name: str, original: Exception) -> ValueError:
    """Build a clear error when MFLUX does not recognize a model name."""
    aliases = ", ".join(sorted(_get_mflux_aliases()))
    return ValueError(
        f"Model '{model_name}' is not recognized by MFLUX. "
        f"Known aliases: {aliases}. "
        f"Original error: {original}"
    )
