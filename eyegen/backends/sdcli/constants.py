"""stable-diffusion.cpp CLI backend constants."""

from pathlib import Path

from eyegen.config import DEFAULT_CONFIG

DEFAULT_SDCLI_BIN = Path.home() / ".local" / "bin" / "sd-cli"
DEFAULT_SDCLI_MODEL_DIR = Path.home() / "models" / "eyegen" / "flux3-klein"

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

# The models this backend knows how to run.
#
# FLUX.3 Image Klein is three files, not one: the transformer, the Qwen3-VL
# text encoder that conditions it, and the VAE that decodes it. The GGUF
# repository holds only the transformer, so the other two come from elsewhere
# and the sizes are what make the model expensive rather than the quality.
#
# ``cfg_scale`` is 1.0 and ``steps`` is 16 because the model is distilled: it
# is trained to answer in that many steps, and anything more is wasted work.
SDCLI_MODELS = {
    "flux-3-image-klein-gguf": {
        "diffusion_model": "flux-3-image-klein-UD-Q4_K_XL.gguf",
        "llm": "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
        "vae": "ae.safetensors",
        "sources": {
            "flux-3-image-klein-UD-Q4_K_XL.gguf": (
                "unsloth/flux-3-image-klein-GGUF",
                "flux-3-image-klein-UD-Q4_K_XL.gguf",
            ),
            "Qwen3VL-4B-Instruct-Q4_K_M.gguf": (
                "Qwen/Qwen3-VL-4B-Instruct-GGUF",
                "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
            ),
            "ae.safetensors": (
                "black-forest-labs/flux-3-image-klein",
                "ae.safetensors",
            ),
        },
        "cfg_scale": 1.0,
        "steps": 16,
        "sampling_method": "euler",
        "description": "FLUX.3 Image Klein, 16-step distilled, GGUF transformer",
    },
}

# The upstream documentation for this model is the stable-diffusion.cpp
# repository's docs/flux3.md.
SDCLI_DOC_URL = "https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/flux3.md"
