"""
stable-diffusion.cpp CLI backend — runs a model through the ``sd-cli``
binary that upstream stable-diffusion.cpp builds.

Why a command line rather than a Python library?
    FLUX.3 Image Klein is newer than the Python bindings on PyPI: the
    bindings bundle a stable-diffusion.cpp from before the model existed, so
    they cannot load it at all. The command line is built from the current
    upstream source, and it is what upstream documents for this model
    (docs/flux3.md). It also runs on Metal, which is what this machine has.

Supported models:
    FLUX.3 Image Klein, the 16-step distilled checkpoint, as a GGUF
    transformer from unsloth. The transformer is one file; the Qwen3-VL text
    encoder that conditions it and the VAE that decodes it are two more,
    and the text encoder is not small.

Lifecycle:
    1. Build the binary:
       ``git clone https://github.com/leejet/stable-diffusion.cpp && \\
        cd stable-diffusion.cpp && git submodule update --init --recursive && \\
        cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release && \\
        cmake --build build --target sd-cli -j $(sysctl -n hw.ncpu)``
       Put the binary at ``~/.local/bin/sd-cli``, or set ``SDCLI_BIN``.
    2. Fetch the model files:
       ``./generate.py pull flux-3-image-klein-gguf``
       That is about 6.7 GB: the transformer, the text encoder and the VAE.
"""

from .constants import SDCLI_DOC_URL, SDCLI_MODELS
from .install import (
    SdCliInstallStatus,
    get_sdcli_bin,
    get_sdcli_model_dir,
    missing_model_files,
    validate_sdcli_install,
)
from .models import list_sdcli_models, pull_sdcli_model
from .pipeline import SdCliWrapper, get_sdcli_pipeline

__all__ = [
    "SDCLI_DOC_URL",
    "SDCLI_MODELS",
    "SdCliInstallStatus",
    "SdCliWrapper",
    "get_sdcli_bin",
    "get_sdcli_model_dir",
    "get_sdcli_pipeline",
    "list_sdcli_models",
    "missing_model_files",
    "pull_sdcli_model",
    "validate_sdcli_install",
]
