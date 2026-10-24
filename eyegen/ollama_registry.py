"""The model registry OllamaDiffuser reads from outside its own package.

OllamaDiffuser keeps its models in a table inside the package. It also
reads a registry file of its own, named by an environment variable, and
that file is how a model the tool does not know reaches it: the entry says
which repository the model is in and what kind of model it is.

The file is written before the package is imported, because the package
reads it once, when it is imported.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Mapping

from eyegen.artifacts import family_of, is_ollama_layout
from eyegen.config import CONFIG_DIR
from eyegen.model_specs import ModelSpec

log = logging.getLogger(__name__)

OLLAMA_REGISTRY_FILE = CONFIG_DIR / "ollamadiffuser-models.json"


def registry_path() -> Path:
    """Where the registry file lives."""
    return OLLAMA_REGISTRY_FILE


def apply_registry() -> None:
    """Point OllamaDiffuser at the entries we keep for it.

    Nothing is done when there is nothing to say: a file with no entries
    would stand in for the package's own table, and the models that come
    with it are the ones that work today.
    """
    entries = read_entries()
    if not entries:
        return
    os.environ["OLLAMADIFFUSER_MODEL_CONFIG"] = str(registry_path())
    try:
        from ollamadiffuser.core.config.model_registry import model_registry
    except ImportError:
        return
    # The package read the file when it was imported; this makes it read
    # the file again, now that it knows where it is.
    model_registry.reload()


def entry_for(spec: ModelSpec) -> dict:
    """The registry entry for a GGUF model."""
    source = spec.sources.get("diffusion_model")
    if source is None:
        raise ValueError(
            f"The model '{spec.name}' names no GGUF file. A GGUF model needs a "
            "'diffusion_model' source."
        )
    return {
        "repo_id": source.repo,
        "model_type": spec.alias or "flux",
        "variant": spec.variant or f"gguf-{spec.name}",
    }


def read_entries() -> dict[str, dict]:
    """The entries we have written so far."""
    path = registry_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("Could not read %s: %s", path, exc)
        return {}
    models = raw.get("models") if isinstance(raw, dict) else None
    return models if isinstance(models, dict) else {}


def write_entries(entries: Mapping[str, dict]) -> Path:
    """Write the registry file, atomically."""
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".ollama-", suffix=".json")
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump({"models": dict(entries)}, stream, indent=2)
            stream.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return path


def register(spec: ModelSpec) -> Path:
    """Record a GGUF model where OllamaDiffuser will find it."""
    entries = read_entries()
    entries[spec.name] = entry_for(spec)
    path = write_entries(entries)
    apply_registry()
    return path


def register_local_gguf(spec: ModelSpec, name: str) -> str:
    """Make the package able to run a GGUF file that is already on this machine.

    The package keeps its models in a directory of its own, one per model,
    and reads the family off the name of that directory. A file that is
    somewhere else is *linked* into the directory the package expects — the
    file stays where it is, and nothing is copied — and the package's own
    settings are told about it, so the model stays installed for the runs
    after this one.

    A directory that is already laid out the way the package lays one out
    is used as it is; there is nothing to link.

    Returns the name the package knows the model by.
    """
    source = spec.sources.get("diffusion_model")
    if source is None or not source.is_local:
        return name

    # A relative path means a different file to the package, whose own
    # directory is where the link is made.
    gguf = Path(source.path).expanduser().resolve()
    directory = gguf.parent
    if not is_ollama_layout(directory):
        directory = _link_into_the_package(gguf, name)
    _tell_the_package(name, directory)
    _write_registry_entry(name, family_of(name))
    apply_registry()
    log.info("Registered the local model %s at %s", name, directory)
    return name


def _link_into_the_package(gguf: Path, name: str) -> Path:
    """The directory the package keeps this model in, with the file linked in."""
    from ollamadiffuser.core.config.settings import settings

    directory = settings.get_model_path(name)
    directory.mkdir(parents=True, exist_ok=True)
    link = directory / gguf.name
    if link.is_symlink() and link.resolve() == gguf.resolve():
        return directory
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(gguf)
    return directory


def _tell_the_package(name: str, directory: Path) -> None:
    """Add the model to the package's own settings, if it is not there."""
    from ollamadiffuser.core.config.settings import ModelConfig, settings

    existing = settings.models.get(name)
    if existing is not None and existing.path == str(directory):
        return
    settings.add_model(
        ModelConfig(
            name=name,
            path=str(directory),
            model_type=family_of(name),
            variant=f"gguf-{name}",
        )
    )
    settings.save_config()


def _write_registry_entry(name: str, family: str) -> None:
    """The entry the package's own table reads, for a model with no repository."""
    entries = read_entries()
    entries[name] = {"repo_id": "", "model_type": family, "variant": f"gguf-{name}"}
    write_entries(entries)
