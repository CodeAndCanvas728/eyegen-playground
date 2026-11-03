"""stable-diffusion.cpp CLI backend constants."""

import logging
from pathlib import Path
from typing import Optional

from eyegen.config import DEFAULT_CONFIG

log = logging.getLogger(__name__)

DEFAULT_SDCLI_BIN = Path.home() / ".local" / "bin" / "sd-cli"

# A render takes minutes, not seconds: sixteen steps at 1024x1024 is about
# fifteen minutes on an M-series Mac. The generic subprocess timeout of
# five minutes would cut it off half way through.
DEFAULT_SDCLI_TIMEOUT = 3600

# The tool-wide defaults. A model that knows better than these wins, because
# for a distilled model the difference is not a preference but a wrong
# picture: FLUX.3 Klein is trained for sixteen steps at a guidance of 1.0,
# and asking it for thirty steps at 7.5 is asking it for something it was
# not trained to do.
TOOL_DEFAULT_STEPS = DEFAULT_CONFIG.num_inference_steps
TOOL_DEFAULT_GUIDANCE = DEFAULT_CONFIG.guidance_scale

# The role a file plays in a model, and the flag the binary takes for it.
# The names are the binary's own, so a model described in the upstream
# documentation can be written down here without translating it.
SDCLI_ROLE_FLAGS = {
    "diffusion_model": "--diffusion-model",
    "high_noise_diffusion_model": "--high-noise-diffusion-model",
    "uncond_diffusion_model": "--uncond-diffusion-model",
    "vae": "--vae",
    "llm": "--llm",
    "llm_vision": "--llm_vision",
    "t5xxl": "--t5xxl",
    "clip_l": "--clip_l",
    "clip_g": "--clip_g",
    "clip_vision": "--clip_vision",
    "tokenizer": "--tokenizer",
    "control_net": "--control-net",
}
# The models this backend knows how to run.
#
# FLUX.3 Image Klein is three files, not one: the transformer, the Qwen3-VL
# text encoder that conditions it, and the VAE that decodes it. The GGUF
# repository holds only the transformer, so the other two come from elsewhere
# and the sizes are what make the model expensive rather than the quality.
#
# ``guidance`` is 1.0 and ``steps`` is 16 because the model is distilled: it
# is trained to answer in that many steps, and anything more is wasted work.
#
# The entries are model specifications, the same shape as the ones a user
# writes in models.json, so a model added there and a model shipped here
# are read and run the same way.
SDCLI_MODELS = {
    "flux-3-image-klein-gguf": {
        "backend": "sdcli",
        "sources": {
            "diffusion_model": {
                "repo": "unsloth/flux-3-image-klein-GGUF",
                "file": "flux-3-image-klein-UD-Q4_K_XL.gguf",
            },
            "llm": {
                "repo": "Qwen/Qwen3-VL-4B-Instruct-GGUF",
                "file": "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
            },
            "vae": {
                "repo": "black-forest-labs/flux-3-image-klein",
                "file": "ae.safetensors",
            },
        },
        "steps": 16,
        "guidance": 1.0,
        "sampling_method": "euler",
        "description": "FLUX.3 Image Klein, 16-step distilled, GGUF transformer",
    },
}

# The upstream documentation for this model is the stable-diffusion.cpp
# repository's docs/flux3.md.
SDCLI_DOC_URL = "https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/flux3.md"

# The settings a model family wants when neither the specification nor the
# user says otherwise, and the fewest steps it can be run at and still make
# a picture. The key is the family with its punctuation left out, matched
# anywhere in a model's name or repository — the way the text encoder's
# companion is looked up in pick.py. The more specific family comes first:
# "qwenimage21" is also a "qwenimage", and the two are not the same model.
#
# Qwen Image 2.1: twenty steps is sd-cli's own default and what upstream's
# text-to-image example runs, at a guidance of 6.0. Below ten steps the flow
# sampler has not left the noise, so the VAE decodes an almost untouched
# latent: the picture is a flat field of colour, not a poor drawing of the
# prompt.
SDCLI_FAMILY_SETTINGS = {
    "qwenimage21": {
        "name": "Qwen Image 2.1",
        "steps": 20,
        "guidance": 6.0,
        "min_steps": 10,
    },
}

# The Turbo weights answer to the same family name and are a different
# model, run through a fixed eight-step sigma schedule of their own; they are
# not grown this family's defaults.
SDCLI_FAMILY_EXCEPTIONS = ("turbo",)


def family_settings(*names: Optional[str]) -> Optional[dict]:
    """The settings of the model family *names* mention, if it is one we know.

    A family is named by any of: the model, its repository, the transformer's
    file. Local discovery gives a model neither repository nor catalogue
    entry, so its name is often the only signal there is.
    """
    from .pick import normalise

    signals = [normalise(name) for name in names if name]
    for family, settings in SDCLI_FAMILY_SETTINGS.items():
        if not any(family in signal for signal in signals):
            continue
        if any(bad in signal for signal in signals for bad in SDCLI_FAMILY_EXCEPTIONS):
            return None
        return dict(settings)
    return None


def _family_names(model: str, spec) -> list[str]:
    """Every name *model* is known by, for the family lookup."""
    names = [model]
    if spec is not None:
        names.append(spec.repo or "")
        source = spec.sources.get("diffusion_model")
        if source is not None:
            names += [source.file, source.name, source.path]
    return names


def resolve_settings(
    model: str,
    steps: Optional[int],
    guidance: Optional[float],
    spec=None,
) -> tuple[int, float]:
    """The steps and guidance the model wants.

    A setting the user has left at the tool-wide default is the model's to
    choose; one they have changed is theirs. A model that states nothing is
    asked for what its family states, and a step count below what the family
    can render at is raised to it: a one-step run of a model trained for
    twenty is not a faster picture, it is a field of colour.
    """
    if spec is None and model in SDCLI_MODELS:
        from eyegen.model_specs import ModelSpec

        spec = ModelSpec.from_dict(model, SDCLI_MODELS[model])
    family = family_settings(*_family_names(model, spec))
    # A setting the model states is the model's, even when it is zero.
    if steps is None or steps == TOOL_DEFAULT_STEPS:
        steps = TOOL_DEFAULT_STEPS
        if spec is not None and spec.steps is not None:
            steps = spec.steps
        elif family is not None:
            steps = family["steps"]
    if guidance is None or guidance == TOOL_DEFAULT_GUIDANCE:
        guidance = TOOL_DEFAULT_GUIDANCE
        if spec is not None and spec.guidance is not None:
            guidance = spec.guidance
        elif family is not None:
            guidance = family["guidance"]
    if family is not None and steps < family["min_steps"]:
        log.warning(
            "Asked for %d step(s) on %s, which renders at %d or more: below that the "
            "sampler has not left the noise. Running %d.",
            steps,
            family["name"],
            family["min_steps"],
            family["min_steps"],
        )
        steps = family["min_steps"]
    return steps, guidance
