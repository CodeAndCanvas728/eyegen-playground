"""Model specifications: what a model is, where its files come from, how it runs.

A specification is data. It names a backend, the files the model is made of,
and the settings the model wants. The catalogues that ship with the tool are
specifications too, so a model the user adds and a model that ships are the
same kind of thing and travel the same path.

The user's own specifications live in ``models.json`` next to the config
file. They are read at the moment a model is looked up, never cached, so a
hand-edited file takes effect on the next command.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Mapping, Optional

from eyegen.config import Backend, EyeGenConfig
from eyegen.model_sources import ModelSource
from eyegen.validation import validate_model_name

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """A model the tool can run.

    ``sources`` maps a role to the file that fills it: ``diffusion_model``,
    ``vae``, ``llm`` and the rest. Which roles a backend understands is the
    backend's business; the specification only says which files exist.

    ``steps``, ``guidance`` and ``sampling_method`` are the settings a model
    wants when the user has not chosen their own. A distilled model is
    trained for a number of steps and a guidance, and asking it for anything
    else produces a wrong picture rather than a slower one.

    ``alias`` names an entry in the backend's own catalogue, for the backends
    whose runtime only knows a fixed set of models. ``path`` is a directory
    on this machine, for the models that are already here. ``repo`` is a
    whole repository, for the backends that take one rather than a set of
    files.
    """

    name: str
    backend: Backend
    sources: Mapping[str, ModelSource] = field(default_factory=dict)
    steps: Optional[int] = None
    guidance: Optional[float] = None
    sampling_method: Optional[str] = None
    path: Optional[str] = None
    alias: Optional[str] = None
    repo: Optional[str] = None
    variant: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self) -> None:
        """Refuse a name that is not one directory name.

        The name is where the model's files are written, so a name with a
        path in it would write them somewhere else. It is checked here, at
        the making of the specification, so that every way in — a catalogue
        the user edited, the command line, the catalogues that ship with the
        tool — is checked the same way.
        """
        validate_model_name(self.name)

    @property
    def is_local(self) -> bool:
        """Whether every file of the model is already on this machine.

        A model whose files are all local has no directory of its own: the
        files are used where they are, so there is nowhere to look for them
        and nothing to fetch.
        """
        return bool(self.sources) and all(source.is_local for source in self.sources.values())

    @classmethod
    def from_dict(cls, name: str, data: Mapping) -> "ModelSpec":
        """Build a specification from its JSON form."""
        if not isinstance(data, Mapping):
            raise ValueError(f"the model '{name}' must be an object, got {type(data).__name__}")

        backend_value = data.get("backend")
        try:
            backend = Backend(backend_value)
        except ValueError as exc:
            known = ", ".join(sorted(b.value for b in Backend))
            raise ValueError(
                f"the model '{name}' names the backend '{backend_value}'. Known backends: {known}."
            ) from exc

        raw_sources = data.get("sources") or {}
        if not isinstance(raw_sources, Mapping):
            raise ValueError(f"the sources of '{name}' must be an object")
        sources = {role: ModelSource.from_dict(entry) for role, entry in raw_sources.items()}

        return cls(
            name=name,
            backend=backend,
            sources=sources,
            steps=_steps(data.get("steps"), name),
            guidance=_guidance(data.get("guidance"), name),
            sampling_method=_optional_str(data.get("sampling_method"), name, "sampling_method"),
            path=_optional_str(data.get("path"), name, "path"),
            alias=_optional_str(data.get("alias"), name, "alias"),
            repo=_optional_str(data.get("repo"), name, "repo"),
            variant=_optional_str(data.get("variant"), name, "variant"),
            description=_optional_str(data.get("description"), name, "description"),
        )

    def to_dict(self) -> dict:
        """Return the JSON form of this specification."""
        data: dict = {"backend": self.backend.value}
        if self.sources:
            data["sources"] = {role: src.to_dict() for role, src in self.sources.items()}
        for key, value in (
            ("steps", self.steps),
            ("guidance", self.guidance),
            ("sampling_method", self.sampling_method),
            ("path", self.path),
            ("alias", self.alias),
            ("repo", self.repo),
            ("variant", self.variant),
            ("description", self.description),
        ):
            if value is not None:
                data[key] = value
        return data


@dataclass(frozen=True)
class ResolvedModel:
    """A model name, the backend that runs it, and the specification behind it.

    The specification is ``None`` for a model that is only a name: the
    backend knows it by name alone, from a catalogue that ships with the
    tool or from the package it uses.

    ``notes`` says what was decided about a model that was resolved from
    what is on this machine: which file is the transformer, where the model
    was linked into the package's own directory.
    """

    name: str
    backend: Backend
    spec: Optional[ModelSpec] = None
    notes: tuple[str, ...] = ()


def _optional_int(value, name: str, key: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"the '{key}' of '{name}' must be a whole number")
    return value


def _optional_float(value, name: str, key: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"the '{key}' of '{name}' must be a number")
    return float(value)


def _steps(value, name: str) -> Optional[int]:
    """The steps a model is run at. Fewer than one is not a setting."""
    steps = _optional_int(value, name, "steps")
    if steps is not None and steps < 1:
        raise ValueError(f"the 'steps' of '{name}' must be at least 1, got {steps}")
    return steps


def _guidance(value, name: str) -> Optional[float]:
    """The guidance a model is run at. Zero is a value a model can be trained for."""
    guidance = _optional_float(value, name, "guidance")
    if guidance is not None and guidance < 0:
        raise ValueError(f"the 'guidance' of '{name}' must not be negative, got {guidance}")
    return guidance


def _optional_str(value, name: str, key: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"the '{key}' of '{name}' must be a string")
    return value


def apply_spec(config: EyeGenConfig, spec: Optional[ModelSpec]) -> EyeGenConfig:
    """The configuration that runs *spec*.

    A specification can name the model its backend knows — ``alias`` — and a
    directory on this machine — ``path``. Both replace what the
    configuration says, for this run only; the file on disk is untouched.

    The backends whose runtime chooses the model by name are the ones that
    read an alias: MFLUX, MLX and Bonsai. The others are told what their
    model is by the specification itself.
    """
    if spec is None:
        return config
    updates: dict = {}
    if spec.alias and spec.backend in (Backend.MFLUX, Backend.MLX, Backend.BONSAI):
        updates["model"] = spec.alias
    if spec.path:
        if spec.backend == Backend.MFLUX:
            updates["mflux_model_path"] = spec.path
        elif spec.backend == Backend.BONSAI:
            updates["bonsai_model_path"] = spec.path
        elif spec.backend == Backend.COREML:
            updates["coreml_model_path"] = spec.path
    if not updates:
        return config
    return replace(config, **updates)
