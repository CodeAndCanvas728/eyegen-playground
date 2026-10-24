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
    register_local_gguf,
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


def _package_in_a_temporary_directory(monkeypatch, tmp_path):
    """The package's own configuration, somewhere that is not the user's."""
    _require_ollamadiffuser()
    from ollamadiffuser.core.config.settings import settings

    monkeypatch.setattr(settings, "config_dir", tmp_path)
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    monkeypatch.setattr(settings, "config_file", tmp_path / "config.json")
    (tmp_path / "models").mkdir()
    return settings


def test_a_local_gguf_is_linked_into_the_package_directory(tmp_path, monkeypatch):
    """The package looks for a model in a directory of its own."""
    settings = _package_in_a_temporary_directory(monkeypatch, tmp_path)
    gguf = tmp_path / "flux1-dev-Q4_K_M.gguf"
    gguf.write_bytes(b"x")
    spec = ModelSpec(
        name="flux1-dev-Q4_K_M",
        backend=Backend.OLLAMA,
        sources={"diffusion_model": ModelSource(path=str(gguf))},
    )

    name = register_local_gguf(spec, spec.name)

    directory = settings.get_model_path(name)
    link = directory / gguf.name
    assert link.is_symlink()
    assert link.resolve() == gguf.resolve()
    assert settings.models[name].path == str(directory)


def test_registering_a_local_gguf_twice_changes_nothing(tmp_path, monkeypatch):
    """The model stays installed for the runs after this one."""
    settings = _package_in_a_temporary_directory(monkeypatch, tmp_path)
    gguf = tmp_path / "flux1-dev-Q4_K_M.gguf"
    gguf.write_bytes(b"x")
    spec = ModelSpec(
        name="flux1-dev-Q4_K_M",
        backend=Backend.OLLAMA,
        sources={"diffusion_model": ModelSource(path=str(gguf))},
    )

    register_local_gguf(spec, spec.name)
    register_local_gguf(spec, spec.name)

    assert settings.models["flux1-dev-Q4_K_M"].path == str(
        settings.get_model_path("flux1-dev-Q4_K_M")
    )


def test_a_model_that_is_not_local_is_left_alone():
    """A model in a repository is fetched; there is nothing to link."""
    spec = ModelSpec(
        name="my-gguf",
        backend=Backend.OLLAMA,
        sources={"diffusion_model": ModelSource(repo="owner/repo", file="model.gguf")},
    )

    assert register_local_gguf(spec, "my-gguf") == "my-gguf"


def test_a_model_the_package_ships_is_ours_when_we_name_it():
    """A name the package already has is replaced, not refused.

    A user can take a name the package ships and make it theirs — their
    repository, their quantisation — and what they wrote is what runs.
    The name is one both 2.0.17 and 2.5.0 ship, in code in the first and
    in the bundled models.yaml in the second, so this holds on either.
    """
    _require_ollamadiffuser()
    from ollamadiffuser.core.config.model_registry import model_registry

    theirs = model_registry.get_model("flux.1-dev-gguf-q2k")
    assert theirs is not None
    assert theirs["repo_id"] == "city96/FLUX.1-dev-gguf"

    register(_gguf(name="flux.1-dev-gguf-q2k"))

    entry = model_registry.get_model("flux.1-dev-gguf-q2k")
    assert entry is not None
    assert entry["repo_id"] == "owner/repo"


def test_a_local_model_whose_name_the_registry_knows_gets_a_suffix(tmp_path):
    """A repository model is what a name fetches; the local one stays reachable."""
    from eyegen.backends import resolve_model

    gguf = tmp_path / "flux-1-dev.gguf"
    gguf.write_bytes(b"x")
    register(_gguf(name="flux-1-dev"))

    resolved = resolve_model(str(gguf))

    assert resolved.name == "flux-1-dev-2"
    assert resolved.spec is not None
    assert resolved.spec.name == "flux-1-dev-2"
    assert any("already taken" in note for note in resolved.notes)


def test_the_suffix_is_stable_when_the_local_model_is_registered(tmp_path):
    """Our own entry has no repository, so it does not collide with itself."""
    from eyegen.backends import resolve_model

    gguf = tmp_path / "flux-1-dev.gguf"
    gguf.write_bytes(b"x")
    register(_gguf(name="flux-1-dev"))
    first = resolve_model(str(gguf)).name

    write_entries(
        {
            "flux-1-dev": entry_for(_gguf(name="flux-1-dev")),
            "flux-1-dev-2": {
                "repo_id": "",
                "model_type": "flux",
                "variant": "gguf-flux-1-dev-2",
            },
        }
    )

    assert first == "flux-1-dev-2"
    assert resolve_model(str(gguf)).name == "flux-1-dev-2"


def test_a_local_model_does_not_take_a_catalogue_name(tmp_path):
    """A catalogue is asked first, so the local model gets its own name."""
    from eyegen.backends import resolve_model
    from eyegen.model_catalogue import upsert_spec

    upsert_spec(ModelSpec(name="klein", backend=Backend.SDCLI))
    directory = tmp_path / "klein"
    directory.mkdir()
    (directory / "transformer.gguf").write_bytes(b"x")
    (directory / "ae.safetensors").write_bytes(b"x")

    assert resolve_model(str(directory)).name == "klein-2"


def test_a_local_model_does_not_take_a_built_in_name(tmp_path):
    """The names the tool ships with are names a catalogue answers too."""
    from eyegen.backends import resolve_model

    directory = tmp_path / "flux-3-image-klein-gguf"
    directory.mkdir()
    (directory / "transformer.gguf").write_bytes(b"x")
    (directory / "ae.safetensors").write_bytes(b"x")

    assert resolve_model(str(directory)).name == "flux-3-image-klein-gguf-2"
