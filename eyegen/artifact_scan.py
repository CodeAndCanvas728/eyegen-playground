"""The models under the models root, and the roots that other programs keep.

The root holds a directory per model, each under the name of what it
holds: ``sdcli`` for the stable-diffusion.cpp models, ``coreml`` for the
converted bundles, ``saved-mflux`` for the models saved for a faster load.
A directory is looked at, not named: what is in it says what it is.

The rest of the root is walked too: a HuggingFace cache keeps a model
under ``models--<org>--<repo>/snapshots/<revision>/``, and a user may drop
a model anywhere. What is in a directory still says what it is, wherever
the directory is.

The same holds for the roots that other programs keep: the HuggingFace
hub caches, and the directory OllamaDiffuser keeps its own models in. A
model that another program downloaded is still a model on this machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from eyegen.artifacts import Artifact, describe, infer_sdcli_roles, is_coreml_bundle
from eyegen.config import Backend
from eyegen.model_sources import ModelSource
from eyegen.model_specs import ModelSpec

# What the walk does not look inside: the Bonsai checkout, the CoreML
# sidecar, and the two halves of a HuggingFace cache that hold a blob and
# a pointer rather than a model.
_WALK_SKIPS = ("bonsai-demo", ".coreml-venv", "blobs", "refs", "snapshots-lock")

_WALK_DEPTH = 4


def scan_all(config=None) -> list[Artifact]:
    """Every model on this machine, whatever root holds it.

    The models root, each HuggingFace hub cache, and the OllamaDiffuser
    model directory are scanned, and the same directory is not reported
    twice.
    """
    from eyegen.paths import hf_hub_dirs, models_dir, ollama_models_dir

    found = list(scan_undescribed(models_dir(config)))
    seen = {artifact.path for artifact in found}
    for hub in hf_hub_dirs(config):
        for artifact in scan_hf_hub(hub):
            if artifact.path not in seen:
                found.append(artifact)
                seen.add(artifact.path)
    for artifact in scan_ollama_models(ollama_models_dir()):
        if artifact.path not in seen:
            found.append(artifact)
            seen.add(artifact.path)
    return found


def scan_hf_hub(hub: Path) -> list[Artifact]:
    """The models a HuggingFace hub cache holds.

    A hub keeps one directory per repository, ``models--<org>--<repo>``,
    and one snapshot per revision under ``snapshots/``. The revision's
    own name is a hash and says nothing; the model is named after the
    repository it came from.
    """
    found: list[Artifact] = []
    if not hub.is_dir():
        return found
    for repo_dir in sorted(hub.iterdir()):
        if not repo_dir.is_dir() or not repo_dir.name.startswith("models--"):
            continue
        # A slash-free name: the name of a model finds its files, so it is
        # one directory name, and the directories it may live under are not
        # folded into it.
        repo_name = repo_dir.name[len("models--") :].replace("--", "__")
        snapshots = repo_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        for revision in sorted(snapshots.iterdir()):
            if not revision.is_dir():
                continue
            artifact = describe(revision)
            if artifact is None:
                continue
            if artifact.name != repo_name:
                artifact = _rename_artifact(artifact, repo_name)
            found.append(artifact)
    return found


def scan_ollama_models(root: Path) -> list[Artifact]:
    """The models OllamaDiffuser keeps in its own directory, one per model."""
    found: list[Artifact] = []
    if not root.is_dir():
        return found
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        artifact = describe(child)
        if artifact is not None:
            found.append(artifact)
    return found


def _rename_artifact(artifact: Artifact, name: str) -> Artifact:
    """*artifact*, known by *name* instead of by the directory it sat in."""
    from dataclasses import replace

    spec = replace(artifact.spec, name=name)
    return replace(artifact, name=name, spec=spec)


def scan_root(root: Path) -> list[Artifact]:
    """The models under the models root, and what each of them is."""
    found: list[Artifact] = []
    for directory, builder in (
        (root / "sdcli", _describe_sdcli_directory),
        (root / "coreml", _describe_coreml_directory),
        (root / "saved-mflux", _describe_saved_mflux_directory),
    ):
        if not directory.is_dir():
            continue
        for child in sorted(directory.iterdir()):
            if not child.is_dir():
                continue
            artifact = builder(child)
            if artifact is not None:
                found.append(artifact)
    found.extend(_walk_elsewhere(root))
    return found


def _walk_elsewhere(root: Path) -> list[Artifact]:
    """The models under the root that the convention does not place.

    The three conventional directories are handled above and are not
    looked at again. Everything else is walked, and a directory that is
    a model is reported without being descended into — its files are its
    roles, not models of their own.
    """
    from eyegen.paths import is_conventional_subdir

    found: list[Artifact] = []
    if not root.is_dir():
        return found

    def walk(directory: Path, depth: int):
        if depth > _WALK_DEPTH:
            return
        try:
            children = sorted(directory.iterdir())
        except OSError:
            return
        for child in children:
            if not child.is_dir() or child.name.startswith("."):
                continue
            if child.name in _WALK_SKIPS:
                continue
            if depth == 0 and is_conventional_subdir(child):
                continue
            artifact = describe(child)
            if artifact is not None:
                found.append(artifact)
                continue
            walk(child, depth + 1)

    walk(root, 0)
    return found


def scan_undescribed(root: Path) -> list[Artifact]:
    """The models under the root that no catalogue describes.

    A catalogue that cannot be read costs the filter, not the listing: the
    shape of a directory still says what it is. The error is reported where
    the catalogue itself is shown.
    """
    from eyegen.backends.sdcli.constants import SDCLI_MODELS
    from eyegen.model_catalogue import load_catalogue

    described = set(SDCLI_MODELS)
    try:
        described |= set(load_catalogue())
    except ValueError:
        pass
    return [artifact for artifact in scan_root(root) if artifact.name not in described]


def _describe_sdcli_directory(directory: Path) -> Optional[Artifact]:
    """A directory under the stable-diffusion.cpp root: a model of that backend.

    The catalogue says what the model is when it is in the catalogue; the
    files say what it is when it is not. A catalogue that cannot be read is
    left to the section that shows it: the shape still says what is here.
    """
    from eyegen.model_catalogue import find_spec

    try:
        spec = find_spec(directory.name)
    except ValueError:
        spec = None
    if spec is not None and spec.backend == Backend.SDCLI:
        return Artifact(
            name=directory.name,
            backend=Backend.SDCLI,
            spec=spec,
            path=directory,
            notes=(f"a stable-diffusion.cpp model: {directory}",),
        )
    roles, inferred = infer_sdcli_roles(directory)
    if "diffusion_model" not in roles:
        return None
    return Artifact(
        name=directory.name,
        backend=Backend.SDCLI,
        spec=ModelSpec(
            name=directory.name,
            backend=Backend.SDCLI,
            sources={role: ModelSource(path=str(file)) for role, file in roles.items()},
            description=f"a stable-diffusion.cpp model at {directory}",
        ),
        path=directory,
        notes=(f"a stable-diffusion.cpp model: {directory}", *inferred),
    )


def _describe_coreml_directory(directory: Path) -> Optional[Artifact]:
    """A directory under the CoreML root: a converted bundle."""
    if not is_coreml_bundle(directory):
        return None
    return Artifact(
        name=directory.name,
        backend=Backend.COREML,
        spec=ModelSpec(
            name=directory.name,
            backend=Backend.COREML,
            path=str(directory),
            description=f"a CoreML bundle at {directory}",
        ),
        path=directory,
        notes=(f"a CoreML bundle: {directory}",),
    )


def _describe_saved_mflux_directory(directory: Path) -> Optional[Artifact]:
    """A directory under the saved-mflux root: a model saved for a faster load.

    A saved model is a directory of component directories, each holding
    safetensors — the same shape ``validate_saved_model`` reads. Anything
    else under the root is not a saved model.
    """
    if not any(directory.glob("*/*.safetensors")):
        return None
    return Artifact(
        name=directory.name,
        backend=Backend.MFLUX,
        spec=ModelSpec(
            name=directory.name,
            backend=Backend.MFLUX,
            path=str(directory),
            description=f"a saved MFLUX model at {directory}",
        ),
        path=directory,
        notes=(f"a saved MFLUX model: {directory}",),
    )
