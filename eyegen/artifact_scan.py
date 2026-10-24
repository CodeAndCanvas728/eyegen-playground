"""The models under the models root.

The root holds a directory per model, each under the name of what it
holds: ``sdcli`` for the stable-diffusion.cpp models, ``coreml`` for the
converted bundles, ``saved-mflux`` for the models saved for a faster load.
A directory is looked at, not named: what is in it says what it is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from eyegen.artifacts import Artifact, infer_sdcli_roles, is_coreml_bundle
from eyegen.config import Backend
from eyegen.model_sources import ModelSource
from eyegen.model_specs import ModelSpec


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
