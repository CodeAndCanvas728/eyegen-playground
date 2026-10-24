"""Where the user's own model specifications are kept.

The catalogue is a file next to the configuration. It is read when a model
is looked up and never cached, so a hand-edited file takes effect on the
next command — the file is the interface, not a cache of one.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Mapping, Optional

from eyegen.config import CONFIG_DIR
from eyegen.model_specs import ModelSpec

log = logging.getLogger(__name__)

MODELS_FILE = CONFIG_DIR / "models.json"


def catalogue_path() -> Path:
    """Where the user's own model specifications live."""
    return MODELS_FILE


def load_catalogue(path: Optional[Path] = None) -> dict[str, ModelSpec]:
    """Read the user's model specifications.

    A file that is not there is an empty catalogue: the tool ships with
    models of its own, and none of them come from here. A file that is
    there and cannot be read is an error, because silently ignoring it
    would look like the model had been deleted.
    """
    target = path or catalogue_path()
    if not target.exists():
        return {}
    try:
        raw = json.loads(target.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read the model catalogue {target}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"The model catalogue {target} must hold an object.")
    models = raw.get("models")
    if models is None:
        return {}
    if not isinstance(models, Mapping):
        raise ValueError(f"The 'models' in {target} must be an object.")
    return {name: ModelSpec.from_dict(name, data) for name, data in models.items()}


def save_catalogue(specs: Mapping[str, ModelSpec], path: Optional[Path] = None) -> Path:
    """Write the user's model specifications, atomically."""
    target = path or catalogue_path()
    payload = {"models": {name: spec.to_dict() for name, spec in specs.items()}}
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".models-", suffix=".json")
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return target


def upsert_spec(spec: ModelSpec, path: Optional[Path] = None) -> Path:
    """Add a specification to the catalogue, or replace one with the same name."""
    specs = load_catalogue(path)
    specs[spec.name] = spec
    return save_catalogue(specs, path)


def remove_spec(name: str, path: Optional[Path] = None) -> bool:
    """Remove a specification from the catalogue. True when one was there."""
    specs = load_catalogue(path)
    if name not in specs:
        return False
    del specs[name]
    save_catalogue(specs, path)
    return True


def find_spec(name: str, path: Optional[Path] = None) -> Optional[ModelSpec]:
    """Return the user's specification for *name*, if there is one."""
    return load_catalogue(path).get(name)
