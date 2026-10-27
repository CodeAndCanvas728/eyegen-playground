"""Which backend runs a model, and the specification behind it.

One resolver, asked by the CLI and by the GUI, so that the two cannot
answer the same question two ways. The order is fixed (DD-010): the
specific patterns are checked before the broad catalogues, because a
substring match is ambiguous — "gguf" in a name must not be allowed to
hijack a bonsai or a HuggingFace name.

A model that is already on this machine is not a name at all: it is a
path, and what it is comes from what is there rather than from how it is
spelled. That is a different question, not a new rung in this chain.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from eyegen._mflux import _get_mflux_aliases
from eyegen._mlx import _get_mlx_supported_models
from eyegen.artifacts import describe
from eyegen.config import Backend, EyeGenConfig
from eyegen.errors import ModelBackendConflict, UnsupportedModelError
from eyegen.model_catalogue import find_spec, load_catalogue
from eyegen.model_specs import ModelSpec, ResolvedModel


def detect_backend(
    model: str,
    override: Backend = Backend.AUTO,
    config: Optional[EyeGenConfig] = None,
) -> Backend:
    """Resolve which backend to use.

    If *override* is ``Backend.AUTO`` (priority order):
      0. A model in the user's own catalogue → that model's backend
      1. A model in the stable-diffusion.cpp catalogue → SDCLI
      2. ``"gguf"`` in model name → OLLAMA  (substr match first — bonsai/hf "gguf" can't hijack)
      3. Bonsai/PrismML identifier pattern → BONSAI
      4. A name the CoreML backend answers to / coreml_model_path set → COREML
      5. Known MFLUX alias → MFLUX
      6. diffusionkit MMDIT_CKPT entry → MLX
      7. otherwise → UnsupportedModelError
    """
    if isinstance(override, str):
        override = Backend(override)
    if override != Backend.AUTO:
        return override
    resolution = resolve_spec(model)
    return _detect_backend(resolution.name, config, resolution.spec)


@dataclass(frozen=True)
class Resolution:
    """A model as it was resolved: its name, its specification, and what was decided."""

    name: str
    spec: Optional[ModelSpec] = None
    notes: tuple[str, ...] = ()


def resolve_spec(model: str) -> Resolution:
    """The model as it is named, and the specification behind it.

    A model in the user's catalogue carries that specification. A model
    that is already on this machine is described by what is there: a path
    is not a name, and what it is comes from its shape.
    """
    spec = find_spec(model)
    if spec is not None:
        return Resolution(model, spec)
    artifact = describe(Path(model))
    if artifact is None:
        return Resolution(model)
    name = _a_name_that_is_free(artifact.name)
    if name == artifact.name:
        return Resolution(name, artifact.spec, artifact.notes)
    note = f"the name '{artifact.name}' is already taken, so this one is '{name}'"
    return Resolution(name, replace(artifact.spec, name=name), (*artifact.notes, note))


def _a_name_that_is_free(name: str) -> str:
    """*name*, or the first suffix of it that nothing else answers to.

    A model on disk is not in a catalogue, but it can arrive with the name
    of one, and a catalogue is asked first: keeping both under one name
    would make one of them unreachable. The suffix is derived from the
    collision alone, so the same model is called the same thing on the next
    run. An entry with no repository is one we wrote for a local model, so
    it does not collide with itself.
    """
    from eyegen.backends.sdcli.constants import SDCLI_MODELS
    from eyegen.ollama_registry import read_entries

    taken = set(SDCLI_MODELS)
    try:
        taken |= set(load_catalogue())
    except ValueError:
        # A catalogue that cannot be read is reported where it is shown.
        # It cannot lend a name to a model here.
        pass
    taken |= {
        key
        for key, entry in read_entries().items()
        if isinstance(entry, dict) and entry.get("repo_id")
    }
    if name not in taken:
        return name
    suffix = 2
    while f"{name}-{suffix}" in taken:
        suffix += 1
    return f"{name}-{suffix}"


def resolve_model(
    model: str,
    override: Backend = Backend.AUTO,
    config: Optional[EyeGenConfig] = None,
) -> ResolvedModel:
    """The backend that runs *model*, and the specification behind it.

    A model the user described in their own catalogue carries its
    specification; a model from a catalogue that ships with the tool carries
    that catalogue's entry, when there is one; a model that is only a name
    carries nothing, and the backend knows it by name alone.
    """
    if isinstance(override, str):
        override = Backend(override)
    resolution = resolve_spec(model)
    name, spec = resolution.name, resolution.spec
    if override != Backend.AUTO:
        if spec is not None and spec.backend != override:
            raise ModelBackendConflict(
                f"The model '{name}' is a {spec.backend.value} model; "
                f"--backend {override.value} cannot run it. "
                f"Drop --backend, or use --backend {spec.backend.value}."
            )
        return ResolvedModel(name, override, spec, notes=resolution.notes)
    backend = _detect_backend(name, config, spec)
    return ResolvedModel(
        name,
        backend,
        spec or _built_in_spec(name, backend),
        notes=resolution.notes,
    )


def _detect_backend(
    model: str,
    config: Optional[EyeGenConfig],
    spec: Optional[ModelSpec],
) -> Backend:
    """The backend for *model*, given the specification already looked up.

    The catalogue is read once by the caller and handed in, so that a
    resolution reads the file once rather than once per step.
    """
    # (0) The user's own catalogue — a model the user described runs the way
    # they described it, whatever its name looks like.
    if spec is not None:
        return spec.backend
    # (1) The stable-diffusion.cpp catalogue — an explicit list beats a pattern
    if _is_sdcli_model(model):
        return Backend.SDCLI
    # (2) GGUF check — substring match so it must run before other patterns
    if "gguf" in model.lower():
        return Backend.OLLAMA
    # (3) Bonsai — model names start with "bonsai-" or "prism-"
    if _is_bonsai_model(model):
        return Backend.BONSAI
    # (4) CoreML — known aliases or coreml_model_path set in config
    if _is_coreml_model(model, config=config):
        return Backend.COREML
    # (5) MFLUX — check live aliases from package, fall back to static set
    if model in _get_mflux_aliases():
        return Backend.MFLUX
    # (6) MLX/diffusionkit — registered MMDIT_CKPT keys
    mlx_keys = _get_mlx_supported_models()
    if mlx_keys is not None and model in mlx_keys:
        return Backend.MLX
    raise UnsupportedModelError(_format_unsupported_error(model, "auto-detected"))


def _built_in_spec(model: str, backend: Backend) -> Optional[ModelSpec]:
    """The specification a catalogue that ships with the tool holds for *model*."""
    if backend != Backend.SDCLI:
        return None
    from eyegen.backends.sdcli.constants import SDCLI_MODELS

    data = SDCLI_MODELS.get(model)
    return ModelSpec.from_dict(model, data) if data else None


def _is_sdcli_model(model: str) -> bool:
    """Return True if *model* is one the stable-diffusion.cpp backend runs.

    The catalogue decides, not a pattern: MFLUX knows a model called
    "black-forest-labs/flux-3-image-klein", and that is a different route
    to a different build of the same weights.
    """
    from eyegen.backends.sdcli.constants import SDCLI_MODELS

    return model in SDCLI_MODELS


def _is_bonsai_model(model: str) -> bool:
    """Return True if *model* looks like a bonsai (PrismML) identifier."""
    m = model.lower()
    return (
        m.startswith("bonsai-")
        or m.startswith("bonsai-image-4b-")
        or m.startswith("prism-ml/bonsai-")
        or "bonsai-image-4b" in m
    )


def _is_coreml_model(model: str, config: Optional[EyeGenConfig] = None) -> bool:
    """Return True if *model* is a name the CoreML backend answers to."""
    if config and config.coreml_model_path:
        return True
    return model.lower() in {
        "sd-1-4",
        "sd-1-5",
        "sd-1-5-palettized",
        "sd-2-base",
        "sd-2-1-base",
        "sd-2-1-base-palettized",
        "sdxl-base",
        "sdxl-ios",
    }


def _format_unsupported_error(model: str, attempted_backend: str) -> str:
    """Build a clear 'model not supported' message listing what *is* supported."""
    mlx_keys = _get_mlx_supported_models() or set()
    mflux_aliases = _get_mflux_aliases() - {""}
    parts = [f"Model '{model}' is not supported by the {attempted_backend} backend."]
    if mlx_keys:
        mlx_list = ", ".join(sorted(mlx_keys))
        parts.append(f"MLX (diffusionkit) supports: {mlx_list}.")
    if mflux_aliases:
        aliases_list = ", ".join(sorted(mflux_aliases))
        parts.append(f"MFLUX supports (use --backend mflux): {aliases_list}.")
    parts.append(
        "For OllamaDiffuser (GGUF) models, pull one with `./generate.py pull <name>` first."
    )
    parts.append(
        "For Bonsai (PrismML), use a model name starting with 'bonsai-' "
        "(e.g. bonsai-ternary-mlx) after running ./scripts/setup-bonsai.sh."
    )
    parts.append(
        "For CoreML, run ./scripts/setup-coreml.sh, convert a model with "
        "./generate.py convert-coreml <hf-model>, then name the bundle it made "
        "(e.g. stable-diffusion-2-1-base)."
    )
    parts.append(
        "For stable-diffusion.cpp, use 'flux-3-image-klein-gguf' "
        "after fetching it with ./generate.py pull flux-3-image-klein-gguf."
    )
    parts.append(
        "To run a model the tool does not know, add it to models.json next to "
        "your config, or fetch one with ./generate.py pull --from-hf <repo>."
    )
    return " ".join(parts)
