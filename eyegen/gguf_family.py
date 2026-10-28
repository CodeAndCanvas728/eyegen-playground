"""The family a GGUF model belongs to, and what it is called here.

OllamaDiffuser reads the family off the name of the directory a model is
in, and fetches different parts for each family. A name that says neither
FLUX nor SD 3.5 is given the family the package uses for most of its
models.
"""

# The names that say which family a GGUF model belongs to.
FLUX_HINTS = ("flux", "flux1")
SD35_HINTS = ("3.5", "sd3.5", "stable-diffusion-3-5")


def local_model_name(stem: str) -> str:
    """The name a local GGUF file is known by.

    The package reads the family off the name of the directory the model is
    in — FLUX or SD 3.5 — and fetches different parts for each. A name that
    says neither is given the family the package uses for most of its models.
    """
    lowered = stem.lower()
    if any(hint in lowered for hint in (*FLUX_HINTS, *SD35_HINTS)):
        return stem
    return f"flux-{stem}"


def family_of(name: str) -> str:
    """The family a GGUF model belongs to, as the package names it."""
    lowered = name.lower()
    if any(hint in lowered for hint in SD35_HINTS):
        return "sd3.5"
    return "flux"
