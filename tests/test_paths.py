"""Tests for where the tool keeps what it downloads.

One root, and everything under it by convention. These check that the
root moves when it is configured — from the settings file, from the
environment, from a configuration handed in — and that the directories
under it move with it.

The collision is the reason the layout exists at all: two models have
files with the same names, and one directory for both means one
overwriting the other.
"""

import json
from pathlib import Path

import pytest

from eyegen.config import EyeGenConfig
from eyegen.paths import (
    DEFAULT_MODELS_DIR,
    MODELS_DIR_ENV,
    bonsai_dir,
    coreml_models_dir,
    coreml_venv_dir,
    models_dir,
    saved_mflux_dir,
    sdcli_model_dir,
)


def _settings_file(root, monkeypatch, tmp_path):
    """A settings file naming *root*, and the file the tool reads."""
    settings = tmp_path / "config.json"
    settings.write_text(json.dumps({"models_dir": str(root)}))
    import eyegen.config as config_module

    monkeypatch.setattr(config_module, "CONFIG_FILE", settings)
    monkeypatch.delenv(MODELS_DIR_ENV, raising=False)
    return settings


def test_the_default_is_one_directory_for_everything(tmp_path, monkeypatch):
    import eyegen.config as config_module

    monkeypatch.setattr(config_module, "CONFIG_FILE", tmp_path / "absent.json")
    monkeypatch.delenv(MODELS_DIR_ENV, raising=False)
    assert models_dir() == DEFAULT_MODELS_DIR


def test_the_settings_file_moves_the_root(tmp_path, monkeypatch):
    """The setting is read where it lives: the file, when there is no object."""
    _settings_file(tmp_path / "roots", monkeypatch, tmp_path)
    assert models_dir() == tmp_path / "roots"


def test_the_environment_overrides_the_settings_file(tmp_path, monkeypatch):
    """A variable is how one run is pointed elsewhere without editing a file."""
    _settings_file(tmp_path / "from-file", monkeypatch, tmp_path)
    monkeypatch.setenv(MODELS_DIR_ENV, str(tmp_path / "from-env"))
    assert models_dir() == tmp_path / "from-env"


def test_a_configuration_handed_in_wins(tmp_path, monkeypatch):
    _settings_file(tmp_path / "from-file", monkeypatch, tmp_path)
    monkeypatch.setenv(MODELS_DIR_ENV, str(tmp_path / "from-env"))
    config = EyeGenConfig(models_dir=str(tmp_path / "from-config"))
    assert models_dir(config) == tmp_path / "from-config"


def test_a_settings_file_that_is_not_there_is_nothing(tmp_path, monkeypatch):
    import eyegen.config as config_module

    monkeypatch.setattr(config_module, "CONFIG_FILE", tmp_path / "absent.json")
    monkeypatch.delenv(MODELS_DIR_ENV, raising=False)
    assert models_dir() == DEFAULT_MODELS_DIR


def test_a_settings_file_that_is_broken_is_nothing(tmp_path, monkeypatch):
    """A broken settings file must not send the root somewhere else."""
    _settings_file(tmp_path / "roots", monkeypatch, tmp_path)
    (tmp_path / "config.json").write_text("{not json")
    assert models_dir() == DEFAULT_MODELS_DIR


def test_every_directory_is_under_the_root(tmp_path, monkeypatch):
    _settings_file(tmp_path, monkeypatch, tmp_path)
    root = models_dir()
    assert sdcli_model_dir("one") == root / "sdcli" / "one"
    assert coreml_models_dir() == root / "coreml"
    assert coreml_venv_dir() == root / ".coreml-venv"
    assert bonsai_dir() == root / "bonsai-demo"
    assert saved_mflux_dir() == root / "saved-mflux"


def test_every_directory_follows_a_configuration_handed_in(tmp_path, monkeypatch):
    _settings_file(tmp_path / "from-file", monkeypatch, tmp_path)
    config = EyeGenConfig(models_dir=str(tmp_path / "from-config"))
    root = tmp_path / "from-config"
    assert sdcli_model_dir("one", config) == root / "sdcli" / "one"
    assert coreml_models_dir(config) == root / "coreml"
    assert coreml_venv_dir(config) == root / ".coreml-venv"
    assert bonsai_dir(config) == root / "bonsai-demo"
    assert saved_mflux_dir(config) == root / "saved-mflux"


def test_a_model_name_with_a_path_in_it_is_refused(tmp_path, monkeypatch):
    """The name is the directory the model's files are written to."""
    _settings_file(tmp_path / "roots", monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="not a path"):
        sdcli_model_dir("../../etc")


def test_two_models_with_the_same_file_keep_their_own(tmp_path, monkeypatch):
    """``ae.safetensors`` is a VAE whichever model it belongs to."""
    _settings_file(tmp_path, monkeypatch, tmp_path)

    for model in ("one", "two"):
        directory = sdcli_model_dir(model)
        directory.mkdir(parents=True)
        (directory / "ae.safetensors").write_bytes(model.encode())

    assert (sdcli_model_dir("one") / "ae.safetensors").read_bytes() == b"one"
    assert (sdcli_model_dir("two") / "ae.safetensors").read_bytes() == b"two"


def test_the_default_root_is_under_the_users_home():
    assert str(DEFAULT_MODELS_DIR).startswith(str(Path.home()))
