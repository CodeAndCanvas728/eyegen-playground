"""Smoke tests for the Bonsai backend helpers."""

from pathlib import Path
from unittest import mock

import pytest

from eyegen.backends import bonsai
from eyegen.backends.bonsai import BonsaiInstallStatus


class TestBonsaiDir:
    def test_default_returns_constant(self):
        from eyegen.backends.bonsai.constants import DEFAULT_BONSAI_DIR

        assert bonsai.get_bonsai_dir() == DEFAULT_BONSAI_DIR

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("BONSAI_DIR", "~/custom/bonsai")
        assert bonsai.get_bonsai_dir() == Path.home() / "custom" / "bonsai"


class TestValidateBonsaiInstall:
    def test_not_installed_when_scripts_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BONSAI_DIR", str(tmp_path))
        status = bonsai.validate_bonsai_install()
        assert isinstance(status, BonsaiInstallStatus)
        assert not status.installed
        assert not status.has_venv
        assert status.has_models == []
        assert "not installed" in status.message

    def test_installed_with_model(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BONSAI_DIR", str(tmp_path))
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "generate.sh").write_text("#!/bin/sh\n")
        (scripts / "download_model.sh").write_text("#!/bin/sh\n")
        venv_python = tmp_path / ".venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.write_text("")

        model_dir = tmp_path / "models" / "ternary-mlx"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_text("dummy")

        status = bonsai.validate_bonsai_install()
        assert status.installed
        assert status.has_venv
        assert status.has_models == ["ternary-mlx"]


class TestListBonsaiModels:
    def test_list_returns_expected_shape(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BONSAI_DIR", str(tmp_path))
        model_dir = tmp_path / "models" / "ternary-mlx"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_text("dummy")

        models = bonsai.list_bonsai_models()
        assert len(models) == 1
        assert models[0]["name"] == "ternary-mlx"
        assert models[0]["alias"] == "bonsai-ternary-mlx"


class TestDownloadBonsaiModel:
    def test_invalid_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown bonsai variant"):
            bonsai.download_bonsai_model("not-a-variant")

    def test_valid_variant_calls_script(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BONSAI_DIR", str(tmp_path))
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "generate.sh").write_text("#!/bin/sh\n")
        (scripts / "download_model.sh").write_text("#!/bin/sh\necho ok")
        venv_python = tmp_path / ".venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.write_text("")
        model_dir = tmp_path / "models" / "ternary-mlx"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_text("dummy")

        with mock.patch("eyegen.backends.bonsai.models.subprocess.Popen") as popen_mock:
            popen_mock.return_value.stdout = []
            popen_mock.return_value.returncode = 0
            assert bonsai.download_bonsai_model("ternary-mlx") is True
