"""What a model on disk is.

A model that is already on this machine has no name in a catalogue, and
the tool cannot tell what it is by looking at the name: a directory of
weights is a model however it is named. These are the rules that say what
a thing on disk is. They answer two questions: what is this path, and
what is under the models root.

The rules are about the shape of what is there — the files, and what they
are called — and not about the name of the directory it is in. Where a
shape fits two backends, the more specific one wins, and what was decided
is said out loud, because a role the binary takes by path is a role the
user may want to correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from eyegen.config import Backend
from eyegen.model_sources import ModelSource
from eyegen.model_specs import ModelSpec

# The text encoders the OllamaDiffuser package fetches next to a GGUF
# model. A directory that has one of them is laid out the way that package
# lays a model out; the VAE alone is not a sign, because a stable-diffusion.cpp
# model has one too.
OLLAMA_ENCODER_FILES = ("clip_l.safetensors", "t5xxl_fp16.safetensors", "t5xxl.safetensors")

# The names that say which family a GGUF model belongs to. The package reads
# the family off the name of the directory the model is in.
FLUX_HINTS = ("flux", "flux1")
SD35_HINTS = ("3.5", "sd3.5", "stable-diffusion-3-5")


@dataclass(frozen=True)
class Artifact:
    """A model on disk: what it is, where it is, and how it is run."""

    name: str
    backend: Backend
    spec: ModelSpec
    path: Path
    notes: tuple[str, ...] = ()


def describe(path: Path) -> Optional[Artifact]:
    """What the thing at *path* is, or nothing when it is not a model.

    The path is made absolute first: it is handed to a runtime, linked into
    another program's directory, and compared with the paths in a
    specification, and a relative path means a different file to each of
    them.
    """
    target = Path(path).expanduser().resolve()
    if target.is_file():
        return _describe_file(target)
    if target.is_dir():
        return _describe_directory(target)
    return None


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


def _describe_file(path: Path) -> Optional[Artifact]:
    """A file: a GGUF is a model, and nothing else the tool can run is."""
    if path.suffix.lower() != ".gguf":
        return None
    name = local_model_name(path.stem)
    spec = ModelSpec(
        name=name,
        backend=Backend.OLLAMA,
        sources={"diffusion_model": ModelSource(path=str(path))},
        description=f"a GGUF file at {path}",
    )
    return Artifact(
        name=name,
        backend=Backend.OLLAMA,
        spec=spec,
        path=path,
        notes=(
            f"a GGUF file: {path}",
            "it will be linked into the OllamaDiffuser model directory",
        ),
    )


def _describe_directory(path: Path) -> Optional[Artifact]:
    """A directory: a bundle, a snapshot, or a set of weights."""
    if is_coreml_bundle(path):
        return Artifact(
            name=path.name,
            backend=Backend.COREML,
            spec=ModelSpec(
                name=path.name,
                backend=Backend.COREML,
                path=str(path),
                description=f"a CoreML bundle at {path}",
            ),
            path=path,
            notes=(f"a CoreML bundle: {path}",),
        )

    if is_diffusers_snapshot(path):
        return Artifact(
            name=path.name,
            backend=Backend.COREML,
            spec=ModelSpec(
                name=path.name,
                backend=Backend.COREML,
                path=str(path),
                description=f"a diffusers model at {path}: converted to CoreML on first use",
            ),
            path=path,
            notes=(
                f"a diffusers model: {path}",
                "it will be converted to CoreML the first time it runs (~15-20 min)",
            ),
        )

    ggufs = sorted(path.glob("*.gguf"), key=_size, reverse=True)
    if not ggufs:
        return None

    if is_ollama_layout(path):
        name = local_model_name(path.name)
        return Artifact(
            name=name,
            backend=Backend.OLLAMA,
            spec=ModelSpec(
                name=name,
                backend=Backend.OLLAMA,
                sources={"diffusion_model": ModelSource(path=str(path / ggufs[0].name))},
                path=str(path),
                description=f"a GGUF model at {path}",
            ),
            path=path,
            notes=(
                f"a GGUF model laid out the way OllamaDiffuser lays one out: {path}",
                "it will be registered with the package",
            ),
        )

    roles, inferred = infer_sdcli_roles(path)
    if "diffusion_model" not in roles:
        return None
    return Artifact(
        name=path.name,
        backend=Backend.SDCLI,
        spec=ModelSpec(
            name=path.name,
            backend=Backend.SDCLI,
            sources={role: ModelSource(path=str(file)) for role, file in roles.items()},
            description=f"a stable-diffusion.cpp model at {path}",
        ),
        path=path,
        notes=(f"a stable-diffusion.cpp model: {path}", *inferred),
    )


def infer_sdcli_roles(directory: Path) -> tuple[dict[str, Path], list[str]]:
    """The roles the files in *directory* fill, and what was read off them.

    The binary takes a path per role, and nothing about a file says which
    role it fills. The names do: a file that says it is a VAE is the VAE, a
    GGUF that says it is an encoder is the encoder, and the transformer is
    the largest GGUF that is left. What was decided is returned with it,
    because a role the user meant differently is one line of a
    specification away.
    """
    ggufs = sorted(directory.glob("*.gguf"), key=_size, reverse=True)
    safetensors = sorted(directory.glob("*.safetensors"), key=_size, reverse=True)
    roles: dict[str, Path] = {}
    inferred: list[str] = []

    for path in ggufs:
        role = _gguf_role(path.name)
        if role and role not in roles:
            roles[role] = path
            inferred.append(f"{role} → {path.name}")

    for path in safetensors:
        role = _safetensors_role(path.name)
        if role and role not in roles:
            roles[role] = path
            inferred.append(f"{role} → {path.name}")

    if "diffusion_model" not in roles:
        assigned = set(roles.values())
        left = [path for path in ggufs if path not in assigned]
        if left:
            roles["diffusion_model"] = left[0]
            inferred.append(f"diffusion_model → {left[0].name} (the largest GGUF that is left)")

    return roles, inferred


def _gguf_role(name: str) -> Optional[str]:
    """The role a GGUF file fills, from its name."""
    lowered = name.lower()
    for hint, role in (
        ("t5xxl", "t5xxl"),
        ("clip_l", "clip_l"),
        ("clip_g", "clip_g"),
        ("clip_vision", "clip_vision"),
        ("llm_vision", "llm_vision"),
        ("control", "control_net"),
        ("qwen", "llm"),
        ("llm", "llm"),
    ):
        if hint in lowered:
            return role
    return None


def _safetensors_role(name: str) -> Optional[str]:
    """The role a safetensors file fills, from its name."""
    lowered = name.lower()
    if lowered in ("ae.safetensors", "flux_vae.safetensors"):
        return "vae"
    for hint, role in (
        ("vae", "vae"),
        ("t5xxl", "t5xxl"),
        ("clip_l", "clip_l"),
        ("clip_g", "clip_g"),
        ("control", "control_net"),
    ):
        if hint in lowered:
            return role
    return None


def is_coreml_bundle(path: Path) -> bool:
    """Whether the directory holds a converted CoreML model."""
    return any(path.glob("*.mlmodelc")) or any(path.glob("*.mlpackage"))


def is_diffusers_snapshot(path: Path) -> bool:
    """Whether the directory holds a diffusers model."""
    if not (path / "config.json").is_file():
        return False
    return (path / "unet").is_dir() or (path / "diffusion_model").is_dir()


def is_ollama_layout(path: Path) -> bool:
    """Whether the directory is laid out the way OllamaDiffuser lays one out."""
    return any((path / name).is_file() for name in OLLAMA_ENCODER_FILES)


def _size(path: Path) -> int:
    """The size of a file, or zero when it cannot be read."""
    try:
        return path.stat().st_size
    except OSError:
        return 0
