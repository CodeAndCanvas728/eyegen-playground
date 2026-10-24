"""Tests for the registry OllamaDiffuser reads.

OllamaDiffuser keeps its models in a table inside the package, and it also
reads a registry file of its own, named by an environment variable. That
file is how a model the tool does not know reaches it. The package is not
vendored here, so these check both halves of the contract: the shape of
what we write, and that the package reads it.
"""

import json
import os

import pytest

from eyegen.config import Backend
from eyegen.model_specs import ModelSource, ModelSpec
from eyegen.ollama_registry import (
    apply_registry,
    entry_for,
    read_entries,
    register,
    registry_path,
    write_entries,
)


def _require_ollamadiffuser():
    """The package the contract is with — or a failure where it is required.

    The package is not vendored here, so a machine without it cannot check
    the other half of the contract. Where the environment says the package
    is part of the run — the CI job that installs the requirements — a
    missing package is a failure rather than a skip: a green run that
    verified nothing is worse than a red one.
    """
    try:
        import ollamadiffuser
    except ImportError as exc:
        if os.environ.get("EYEGEN_REQUIRE_OLLAMADIFFUSER") == "1":
            raise AssertionError(
                "ollamadiffuser is required by this run but is not installed"
            ) from exc
        pytest.skip(f"ollamadiffuser is not installed: {exc}")
    return ollamadiffuser


def _gguf(name: str = "my-gguf") -> ModelSpec:
    return ModelSpec(
        name=name,
        backend=Backend.OLLAMA,
        sources={"diffusion_model": ModelSource(repo="owner/repo", file="model.gguf")},
    )


def test_the_entry_says_where_the_model_is():
    entry = entry_for(_gguf())
    assert entry["repo_id"] == "owner/repo"
    assert entry["model_type"] == "flux"
    assert entry["variant"] == "gguf-my-gguf"


def test_the_file_holds_the_entries_under_models():
    write_entries({"my-gguf": entry_for(_gguf())})
    written = json.loads(registry_path().read_text())
    assert "my-gguf" in written["models"]


def test_nothing_is_applied_when_there_is_nothing_to_say(monkeypatch):
    """An empty file would stand in for the package's own table."""
    monkeypatch.delenv("OLLAMADIFFUSER_MODEL_CONFIG", raising=False)
    apply_registry()
    assert "OLLAMADIFFUSER_MODEL_CONFIG" not in os.environ


def test_the_package_is_pointed_at_the_file_when_there_is_one(monkeypatch):
    register(_gguf())
    assert os.environ["OLLAMADIFFUSER_MODEL_CONFIG"] == str(registry_path())
    assert read_entries()["my-gguf"]["repo_id"] == "owner/repo"


def test_the_package_reads_what_we_write():
    """The package is the other half of the contract; it is not vendored here."""
    _require_ollamadiffuser()
    from ollamadiffuser.core.config.model_registry import model_registry

    register(_gguf())
    entry = model_registry.get_model("my-gguf")
    assert entry is not None
    assert entry["repo_id"] == "owner/repo"


def test_an_unknown_variant_downloads_every_gguf():
    """The fallback the README relies on: a variant we made up takes all."""
    _require_ollamadiffuser()
    from ollamadiffuser.core.models.gguf_loader import GGUFModelLoader

    patterns = GGUFModelLoader().get_gguf_download_patterns("gguf-my-gguf")
    assert "*.gguf" in patterns["allow_patterns"]
