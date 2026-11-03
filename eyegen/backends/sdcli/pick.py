"""Which repository file fills which role: VAE, text encoder, transformer.

Picking is pure: a list of file names in, one file out. Nothing here
touches the network or the disk, so a repository can be picked apart
without downloading a byte of it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

QUANTIZED_HINTS = ("int8", "convrot", "q4_", "q5_", "q6_", "q8_", "q4_k", "q5_k", "q6_k", "q8_0")
BULKY_HINTS = ("bf16", "fp16", "f16", "fp32", "float32", "f32")
VAE_HINTS = ("vae", "ae.safetensors", "flux_vae")
LLM_HINTS = ("t5xxl", "clip_l", "clip_g", "qwen3vl", "llm", "instruct", "text_encoder", "/llm")
# int8 tensorwise/convrot weights run on CPU, CUDA, HIP and Vulkan only:
# sd.cpp's Metal backend has no kernel for them, so such a file aborts a Mac.
UNRUNNABLE_HINTS = ("int8", "convrot")
# sd.cpp's own model guide pairs a model family with the encoder that runs
# on it: Qwen-Image 2.1 is conditioned by Qwen3-VL-8B. The key is the
# family as it is written, with the punctuation left out, so that the name
# of a repository and the name of a file can both name it.
TEXT_ENCODER_COMPANIONS = {
    "qwenimage21": ("Qwen/Qwen3-VL-8B-Instruct-GGUF", "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
}
# The encoder's own name is a second signal for the same pairing, for a
# repository that does not say which model family it holds.
ENCODER_NAME_FAMILIES = {
    "qwen3vl8b": "qwenimage21",
}


@dataclass(frozen=True)
class EncoderChoice:
    """The text encoder a repository offers, and what was passed over.

    ``file`` is the encoder to use and ``repo`` the repository it comes
    from — the model's own, or a documented companion's. ``rejected`` names
    the encoder-shaped files that were passed over because this machine's
    backend cannot run them.
    """

    file: Optional[str]
    repo: str
    rejected: tuple[str, ...] = ()


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


def resolve_llm(files: list[str], repo: str, transformer: Optional[str] = None) -> EncoderChoice:
    """The text encoder *repo* offers that this machine can run, and where.

    Returns the encoder's file, the repository it comes from — the model's
    own, or a documented companion's — and the encoder-shaped files that
    were passed over. An int8 convrot encoder aborts on this machine's
    Metal backend and a bf16 one is enormous, so either sends the choice
    to the companion; a repository of another model family keeps its
    full-precision encoder, with a warning.
    """
    candidates = [f for f in files if not _claims_transformer(f, transformer, files)]
    rejected = tuple(
        f
        for f in candidates
        if _is_encoder_shaped(f) and any(bad in f.lower() for bad in UNRUNNABLE_HINTS)
    )
    llm = pick_by_hints(candidates, LLM_HINTS, exclude=UNRUNNABLE_HINTS)
    if llm is not None and slimness(llm.lower()) < 2:
        return EncoderChoice(llm, repo, rejected)
    companion = text_encoder_companion(files, repo, transformer)
    if companion is not None:
        log.info("The text encoder comes from the companion repository %s: %s", *companion)
        return EncoderChoice(companion[1], companion[0], rejected)
    if llm is None:
        return EncoderChoice(None, repo, rejected)
    log.warning(
        "The only text encoder %s ships is full precision (%s): it runs, but it is enormous.%s",
        repo,
        llm,
        _skipped_note(rejected),
    )
    return EncoderChoice(llm, repo, rejected)


def text_encoder_companion(
    files: list[str], repo: str, transformer: Optional[str] = None
) -> Optional[tuple[str, str]]:
    """The (repository, file) of a documented GGUF companion encoder.

    sd.cpp's own model guide names where a runnable encoder lives when a
    repository ships none: Qwen-Image-2.1 pairs with the quantised GGUF of
    Qwen3-VL-8B-Instruct. The pairing is looked up by the family of the
    model — the repository's own name, or the transformer's file name — so
    that the spelling of a file name cannot decide it. The encoder's own
    name is a second signal, for a repository that does not say.
    """
    signals = [normalise(repo), normalise(transformer or "")]
    signals += [normalise(f) for f in files]
    for family, companion in TEXT_ENCODER_COMPANIONS.items():
        if any(family in signal for signal in signals):
            return companion
    for name, family in ENCODER_NAME_FAMILIES.items():
        if any(name in signal for signal in signals):
            return TEXT_ENCODER_COMPANIONS[family]
    return None


def _claims_transformer(name: str, transformer: Optional[str], files: list[str]) -> bool:
    """Whether *name* is the transformer rather than a text encoder.

    The transformer is the model itself: the file whose name the caller
    knows, and otherwise the only GGUF in the listing — one GGUF is a
    model of one file, and that file is the model. A name may claim to be
    an encoder only when another GGUF is left to be the transformer.
    """
    if transformer and Path(name).name == Path(transformer).name:
        return True
    if name.lower().endswith(".gguf"):
        ggufs = [f for f in files if f.lower().endswith(".gguf")]
        if len(ggufs) == 1:
            return True
    return False


def _is_encoder_shaped(name: str) -> bool:
    """Whether a file's name says that it is a text encoder."""
    lowered = name.lower()
    return any(hint in lowered for hint in LLM_HINTS)


def _skipped_note(rejected: tuple[str, ...]) -> str:
    """What to say about the encoders that were passed over, if there were any."""
    if not rejected:
        return ""
    return (
        f" The repository also ships {', '.join(rejected)}, which was passed over:"
        " the Metal backend has no kernel for int8/convrot weights."
    )


def normalise(name: str) -> str:
    """A name as its letters and digits only, in lower case.

    "Qwen-Image-2.1" and "qwen_image_2_1" are one family written two
    ways; the punctuation is what differs, not the family.
    """
    return "".join(ch for ch in name.lower() if ch.isalnum())


def slimness(name: str) -> int:
    """How much memory a file's format wants: lower is slimmer."""
    if any(h in name for h in QUANTIZED_HINTS):
        return 0
    if any(h in name for h in BULKY_HINTS):
        return 2
    return 1
