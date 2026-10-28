"""stable-diffusion.cpp CLI backend constants."""

from pathlib import Path
from typing import Optional

from eyegen.config import DEFAULT_CONFIG

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


def resolve_settings(
    model: str,
    steps: Optional[int],
    guidance: Optional[float],
    spec=None,
) -> tuple[int, float]:
    """The steps and guidance the model wants.

    A setting the user has left at the tool-wide default is the model's to
    choose; one they have changed is theirs.
    """
    if spec is None and model in SDCLI_MODELS:
        from eyegen.model_specs import ModelSpec

        spec = ModelSpec.from_dict(model, SDCLI_MODELS[model])
    # A setting the model states is the model's, even when it is zero.
    if steps is None or steps == TOOL_DEFAULT_STEPS:
        steps = TOOL_DEFAULT_STEPS
        if spec is not None and spec.steps is not None:
            steps = spec.steps
    if guidance is None or guidance == TOOL_DEFAULT_GUIDANCE:
        guidance = TOOL_DEFAULT_GUIDANCE
        if spec is not None and spec.guidance is not None:
            guidance = spec.guidance
    return steps, guidance
