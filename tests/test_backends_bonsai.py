"""Smoke tests for the Bonsai backend helpers."""

from pathlib import Path
from unittest import mock

import pytest

from eyegen.backends import bonsai
from eyegen.backends.bonsai import BonsaiInstallStatus
from eyegen.config import EyeGenConfig


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

        with mock.patch(
            "eyegen.backends.runner.BaseSubprocessRunner._execute_subprocess"
        ) as exec_mock:
            exec_mock.return_value = (0, ["ok\n"], [])
            assert bonsai.download_bonsai_model("ternary-mlx") is True

    def test_download_timeout(self, monkeypatch, tmp_path):
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

        with mock.patch(
            "eyegen.backends.runner.BaseSubprocessRunner._execute_subprocess"
        ) as exec_mock:
            exec_mock.side_effect = RuntimeError("bonsai subprocess timed out after 1800 seconds")
            with pytest.raises(RuntimeError, match="Bonsai subprocess timed out"):
                bonsai.download_bonsai_model("ternary-mlx")
            assert exec_mock.called


class TestBonsaiWrapper:
    @mock.patch("eyegen.backends.bonsai.pipeline.validate_bonsai_install")
    @mock.patch("eyegen.backends.bonsai.pipeline.BonsaiWrapper._execute_subprocess")
    @mock.patch("eyegen.backends.bonsai.pipeline.Image.open")
    def test_seed_zero(self, mock_image_open, mock_execute, mock_validate, tmp_path, monkeypatch):
        mock_status = mock.Mock()
        mock_status.installed = True
        mock_status.has_models = True
        mock_status.bonsai_dir = tmp_path
        mock_validate.return_value = mock_status

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / "generate.sh").touch()

        mock_execute.return_value = (0, ["line1\n", "line2\n"], [])

        mock_image = mock.Mock()
        mock_image_open.return_value.convert.return_value = mock_image

        monkeypatch.setattr("eyegen.config.OUTPUT_DIR", tmp_path)
        expected_path = tmp_path / "bonsai_ternary-mlx_0.png"
        expected_path.touch()

        from eyegen.backends.bonsai.pipeline import BonsaiWrapper
        from eyegen.config import EyeGenConfig

        wrapper = BonsaiWrapper(EyeGenConfig(model="bonsai-ternary-mlx"))
        img = wrapper.generate_image(
            prompt="test", cfg_weight=1.0, num_steps=10, width=64, height=64, seed=0
        )
        assert img == mock_image
        assert expected_path.is_file()

    @mock.patch("eyegen.backends.bonsai.pipeline.validate_bonsai_install")
    @mock.patch("eyegen.backends.bonsai.pipeline.BonsaiWrapper._execute_subprocess")
    def test_subprocess_timeout(self, mock_execute, mock_validate, tmp_path, monkeypatch):
        mock_status = mock.Mock()
        mock_status.installed = True
        mock_status.has_models = True
        mock_status.bonsai_dir = tmp_path
        mock_validate.return_value = mock_status

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / "generate.sh").touch()

        mock_execute.side_effect = RuntimeError("Bonsai subprocess timed out after 300 seconds")

        monkeypatch.setattr("eyegen.config.OUTPUT_DIR", tmp_path)

        from eyegen.backends.bonsai.pipeline import BonsaiWrapper
        from eyegen.config import EyeGenConfig

        wrapper = BonsaiWrapper(EyeGenConfig(model="bonsai-ternary-mlx"))
        with pytest.raises(RuntimeError, match="Bonsai subprocess timed out"):
            wrapper.generate_image(
                prompt="test", cfg_weight=1.0, num_steps=10, width=64, height=64, seed=42
            )

    @mock.patch("eyegen.backends.bonsai.pipeline.validate_bonsai_install")
    @mock.patch("eyegen.backends.bonsai.pipeline.BonsaiWrapper._execute_subprocess")
    @mock.patch("eyegen.backends.bonsai.pipeline.Image.open")
    def test_dimensions_auto_adjust(
        self, mock_image_open, mock_execute, mock_validate, tmp_path, monkeypatch
    ):
        mock_status = mock.Mock()
        mock_status.installed = True
        mock_status.has_models = True
        mock_status.bonsai_dir = tmp_path
        mock_validate.return_value = mock_status

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / "generate.sh").touch()

        mock_execute.return_value = (0, [], [])
        mock_image = mock.Mock()
        mock_image_open.return_value.convert.return_value = mock_image

        monkeypatch.setattr("eyegen.config.OUTPUT_DIR", tmp_path)
        expected_path = tmp_path / "bonsai_ternary-mlx_42.png"
        expected_path.touch()

        from eyegen.backends.bonsai.pipeline import BonsaiWrapper
        from eyegen.config import EyeGenConfig

        wrapper = BonsaiWrapper(EyeGenConfig(model="bonsai-ternary-mlx"))
        # Bonsai requires multiples of 32; non-compliant dims raise ValueError
        with pytest.raises(ValueError, match="multiples of 32"):
            wrapper.generate_image(
                prompt="test", cfg_weight=1.0, num_steps=10, width=500, height=500, seed=42
            )

    @mock.patch("eyegen.backends.bonsai.pipeline.validate_bonsai_install")
    @mock.patch("eyegen.backends.bonsai.pipeline.BonsaiWrapper._execute_subprocess")
    @mock.patch("eyegen.backends.bonsai.pipeline.Image.open")
    def test_dimensions_accepts_32_multiple(
        self, mock_image_open, mock_execute, mock_validate, tmp_path, monkeypatch
    ):
        mock_status = mock.Mock()
        mock_status.installed = True
        mock_status.has_models = True
        mock_status.bonsai_dir = tmp_path
        mock_validate.return_value = mock_status

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / "generate.sh").touch()

        mock_execute.return_value = (0, ["line1\n", "line2\n"], [])
        mock_image = mock.Mock()
        mock_image_open.return_value.convert.return_value = mock_image

        monkeypatch.setattr("eyegen.config.OUTPUT_DIR", tmp_path)
        expected_path = tmp_path / "bonsai_ternary-mlx_42.png"
        expected_path.touch()

        from eyegen.backends.bonsai.pipeline import BonsaiWrapper

        wrapper = BonsaiWrapper(EyeGenConfig(model="bonsai-ternary-mlx"))
        img = wrapper.generate_image(
            prompt="test", cfg_weight=1.0, num_steps=10, width=512, height=512, seed=42
        )
        assert img == mock_image

    @mock.patch("eyegen.backends.bonsai.pipeline.validate_bonsai_install")
    def test_invalid_variant_raises(self, mock_validate):
        mock_status = mock.Mock()
        mock_status.installed = True
        mock_status.has_models = True
        mock_validate.return_value = mock_status

        from eyegen.backends.bonsai.pipeline import BonsaiWrapper
        from eyegen.config import EyeGenConfig

        with pytest.raises(ValueError, match="Unsupported Bonsai variant"):
            BonsaiWrapper(EyeGenConfig(model="bonsai-invalid-variant-name"))
