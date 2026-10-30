"""Which repository file fills which role: VAE, text encoder, transformer.

Picking is pure: a list of file names in, one file out. Nothing here
touches the network or the disk, so a repository can be picked apart
without downloading a byte of it.
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

QUANTIZED_HINTS = ("int8", "convrot", "q4_", "q5_", "q6_", "q8_", "q4_k", "q5_k", "q6_k", "q8_0")
BULKY_HINTS = ("bf16", "fp16", "f16", "fp32", "float32", "f32")
VAE_HINTS = ("vae", "ae.safetensors", "flux_vae")
LLM_HINTS = ("t5xxl", "clip_l", "clip_g", "qwen3vl", "llm", "instruct", "text_encoder", "/llm")
# int8 tensorwise/convrot weights run on CPU, CUDA, HIP and Vulkan only:
# sd.cpp's Metal backend has no kernel for them, so such a file aborts a Mac.
UNRUNNABLE_HINTS = ("int8", "convrot")
# A runnable encoder for a repository that ships none: keyed by a fragment
# of the encoder's name, as documented by sd.cpp's model guides.
TEXT_ENCODER_COMPANIONS = {
    "qwen3vl_8b": ("Qwen/Qwen3-VL-8B-Instruct-GGUF", "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
}


def pick_by_hints(
    files: list[str], hints: tuple[str, ...], exclude: tuple[str, ...] = ()
) -> Optional[str]:
    """The one file of *files* a role names, or nothing.

    Weights are called by what they do: a file with ``vae`` in its name or
    path is the VAE. The first hint that matches decides, and the match is
    the first file the hint names, so the answer does not depend on the
    order a repository happens to list its files in.

    A quantized file beats a full-precision one when a repository ships
    both: a text encoder in BF16 is enormous, and on a laptop it is the
    file that takes the machine down. ``int8``, the ``Q*`` quantisations
    and GGUF are preferred over ``bf16``/``fp16``/``fp32``. A name holding
    one of *exclude* is never chosen, however slim: the text-encoder role
    passes the formats this machine's backend cannot run.
    """
    lowered = [f.lower() for f in files]
    for hint in hints:
        matches = [
            (index, name)
            for index, name in enumerate(lowered)
            if hint in name and not any(bad in name for bad in exclude)
        ]
        if not matches:
            continue
        by_slimness = {index: slimness(name) for index, name in matches}
        chosen = min(matches, key=lambda pair: by_slimness[pair[0]])
        return files[chosen[0]]
    return None


def resolve_llm(files: list[str], repo: str) -> tuple[Optional[str], str]:
    """The text encoder *repo* offers that this machine can run, and where.

    Returns the encoder's file and the repository it comes from — the model's
    own, or a documented companion's. An int8 convrot encoder aborts on this
    machine's Metal backend and a bf16 one is enormous, so either sends the
    choice to the companion; a repository of another model family keeps its
    full-precision encoder, with a warning.
    """
    llm = pick_by_hints(files, LLM_HINTS, exclude=UNRUNNABLE_HINTS)
    if llm is not None and slimness(llm.lower()) < 2:
        return llm, repo
    companion = text_encoder_companion(files)
    if companion is not None:
        log.info("The text encoder comes from the companion repository %s: %s", *companion)
        return companion[1], companion[0]
    if llm is not None:
        log.warning(
            "The only text encoder %s ships is full precision (%s): it runs, but it is"
            " enormous. An int8/convrot encoder cannot be used here — the Metal backend"
            " has no kernel for it.",
            repo,
            llm,
        )
    return llm, repo


def text_encoder_companion(files: list[str]) -> Optional[tuple[str, str]]:
    """The (repository, file) of a documented GGUF companion encoder.

    sd.cpp's own model guide names where a runnable encoder lives when a
    repository ships none: Qwen-Image-2.1 pairs with the quantised GGUF of
    Qwen3-VL-8B-Instruct. A fragment of the repository's own encoder name
    selects the entry, so another model family is not redirected.
    """
    lowered = [f.lower() for f in files]
    for fragment, companion in TEXT_ENCODER_COMPANIONS.items():
        if any(fragment in name for name in lowered):
            return companion
    return None


def slimness(name: str) -> int:
    """How much memory a file's format wants: lower is slimmer."""
    if any(h in name for h in QUANTIZED_HINTS):
        return 0
    if any(h in name for h in BULKY_HINTS) or is_bulky_gguf(name):
        return 2
    return 1


def is_bulky_gguf(name: str) -> bool:
    """Whether a GGUF carries its weights at full precision."""
    return name.endswith(".gguf") and any(h in name for h in BULKY_HINTS)
