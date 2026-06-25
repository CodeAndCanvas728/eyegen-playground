"""Smoke tests for the CoreML backend helpers."""

import subprocess
from pathlib import Path
from unittest import mock

import pytest

from eyegen.backends import coreml
from eyegen.backends.coreml import CoreMLInstallStatus
from eyegen.config import EyeGenConfig


@pytest.fixture(autouse=True)
def _clear_coreml_cache():
    coreml.validate_coreml_install.cache_clear()
    coreml.list_coreml_models.cache_clear()
    yield


class TestCoremlPaths:
    def test_default_models_dir(self):
        from eyegen.backends.coreml.constants import DEFAULT_COREML_MODELS_DIR

        assert coreml.get_coreml_models_dir() == DEFAULT_COREML_MODELS_DIR

    def test_models_dir_env_override(self, monkeypatch):
        monkeypatch.setenv("COREML_MODELS_DIR", "~/custom/coreml")
        assert coreml.get_coreml_models_dir() == Path.home() / "custom" / "coreml"

    def test_default_venv(self):
        from eyegen.backends.coreml.constants import DEFAULT_COREML_VENV

        assert coreml.get_coreml_venv() == DEFAULT_COREML_VENV

    def test_venv_env_override(self, monkeypatch):
        monkeypatch.setenv("COREML_VENV", "~/custom/venv")
        assert coreml.get_coreml_venv() == Path.home() / "custom" / "venv"


class TestValidateCoremlInstall:
    def test_missing_venv(self, tmp_path, monkeypatch):
        monkeypatch.setenv("COREML_VENV", str(tmp_path / "missing"))
        monkeypatch.setenv("COREML_MODELS_DIR", str(tmp_path / "models"))
        status = coreml.validate_coreml_install()
        assert isinstance(status, CoreMLInstallStatus)
        assert not status.installed
        assert status.venv_python is None

    def test_venv_without_coreml_package(self, tmp_path, monkeypatch):
        monkeypatch.setenv("COREML_VENV", str(tmp_path))
        monkeypatch.setenv("COREML_MODELS_DIR", str(tmp_path / "models"))
        py = tmp_path / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("#!/bin/sh\n")

        status = coreml.validate_coreml_install()
        assert not status.installed
        assert "python_coreml_stable_diffusion is not installed" in status.message


class TestListCoremlModels:
    def test_empty_when_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("COREML_MODELS_DIR", str(tmp_path / "missing"))
        assert coreml.list_coreml_models() == []

    def test_skips_entries_without_model_files(self, tmp_path, monkeypatch):
        monkeypatch.setenv("COREML_MODELS_DIR", str(tmp_path))
        (tmp_path / "empty-dir").mkdir()
        assert coreml.list_coreml_models() == []

    def test_finds_mlmodelc(self, tmp_path, monkeypatch):
        monkeypatch.setenv("COREML_MODELS_DIR", str(tmp_path))
        model_dir = tmp_path / "sd-2-1-base"
        mlmodelc = model_dir / "SD2_1_base.mlmodelc"
        mlmodelc.mkdir(parents=True)
        assert len(coreml.list_coreml_models()) == 1
        assert coreml.list_coreml_models()[0]["name"] == "sd-2-1-base"


class TestConvertToCoreml:
    def test_invalid_compute_unit_raises(self):
        with pytest.raises(ValueError, match="Invalid compute_unit"):
            coreml.convert_to_coreml("foo/bar", Path("output"), compute_unit="INVALID")

    @mock.patch("eyegen.backends.coreml.models._sidecar_has_coreml")
    @mock.patch("eyegen.backends.coreml.models._sidecar_python")
    @mock.patch("eyegen.backends.runner.BaseSubprocessRunner._execute_subprocess")
    def test_convert_to_coreml_timeout(self, mock_exec, mock_sidecar, mock_has_coreml, tmp_path):
        mock_has_coreml.return_value = True
        mock_sidecar.return_value = Path("/fake/python")
        mock_exec.side_effect = RuntimeError(
            "coreml-convert subprocess timed out after 1800 seconds"
        )

        with pytest.raises(subprocess.TimeoutExpired):
            coreml.convert_to_coreml("foo/bar", tmp_path, compute_unit="CPU_AND_NE")

        assert mock_exec.called


class TestCoreMLWrapper:
    @mock.patch("eyegen.backends.coreml.pipeline.validate_coreml_install")
    @mock.patch("eyegen.backends.coreml.pipeline._sidecar_python")
    @mock.patch("eyegen.backends.coreml.pipeline.CoreMLWrapper._execute_subprocess")
    def test_seed_zero(self, mock_execute, mock_sidecar, mock_validate, tmp_path, monkeypatch):
        mock_status = mock.Mock()
        mock_status.installed = True
        mock_validate.return_value = mock_status
        mock_sidecar.return_value = Path("/fake/python")

        mock_execute.return_value = (0, [], ["done\n"])

        monkeypatch.setattr("eyegen.config.OUTPUT_DIR", tmp_path)
        model_dir = tmp_path / "sd-2-1-base"
        model_dir.mkdir()
        expected_path = tmp_path / "coreml_0.png"
        expected_path.touch()

        from eyegen.backends.coreml.pipeline import CoreMLWrapper

        wrapper = CoreMLWrapper(EyeGenConfig(coreml_model_path=str(model_dir)))
        with mock.patch("eyegen.backends.coreml.pipeline.Image.open") as mock_img_open:
            mock_img_open.return_value.convert.return_value = mock.Mock()
            wrapper.generate_image(
                prompt="test",
                cfg_weight=7.5,
                num_steps=10,
                width=512,
                height=512,
                seed=0,
            )
        assert expected_path.is_file()

    @mock.patch("eyegen.backends.coreml.pipeline.validate_coreml_install")
    @mock.patch("eyegen.backends.coreml.pipeline._sidecar_python")
    @mock.patch("eyegen.backends.coreml.pipeline.CoreMLWrapper._execute_subprocess")
    def test_subprocess_timeout(  # noqa: E501
        self, mock_execute, mock_sidecar, mock_validate, tmp_path, monkeypatch
    ):
        mock_status = mock.Mock()
        mock_status.installed = True
        mock_validate.return_value = mock_status
        mock_sidecar.return_value = Path("/fake/python")

        mock_execute.side_effect = RuntimeError("CoreML subprocess timed out after 300 seconds")

        monkeypatch.setattr("eyegen.config.OUTPUT_DIR", tmp_path)

        model_dir = tmp_path / "sd-2-1-base"
        model_dir.mkdir()

        from eyegen.backends.coreml.pipeline import CoreMLWrapper

        wrapper = CoreMLWrapper(EyeGenConfig(coreml_model_path=str(model_dir)))

        with pytest.raises(RuntimeError, match="CoreML subprocess timed out"):
            wrapper.generate_image(
                prompt="test", cfg_weight=7.5, num_steps=10, width=512, height=512, seed=42
            )

    @mock.patch("eyegen.backends.coreml.pipeline.validate_coreml_install")
    def test_reject_unsupported_options(self, mock_validate, tmp_path):
        mock_status = mock.Mock()
        mock_status.installed = True
        mock_validate.return_value = mock_status

        model_dir = tmp_path / "sd-2-1-base"
        model_dir.mkdir()

        from eyegen.backends.coreml.pipeline import CoreMLWrapper

        wrapper = CoreMLWrapper(EyeGenConfig(coreml_model_path=str(model_dir)))

        # 1. Reject non-512 dimensions (e.g. 256x256)
        with pytest.raises(ValueError, match="CoreML models only support these resolutions"):
            wrapper.generate_image(
                prompt="test", cfg_weight=7.5, num_steps=10, width=256, height=256
            )

        # 2. Reject img2img (non-None image_path)
        with pytest.raises(ValueError, match="does not support img2img"):
            wrapper.generate_image(
                prompt="test",
                cfg_weight=7.5,
                num_steps=10,
                width=512,
                height=512,
                image_path="dummy.png",
            )

    @mock.patch("eyegen.backends.coreml.pipeline.validate_coreml_install")
    def test_invalid_characters_in_model_name(self, mock_validate):
        mock_status = mock.Mock()
        mock_status.installed = True
        mock_validate.return_value = mock_status

        from eyegen.backends.coreml.pipeline import CoreMLWrapper

        with pytest.raises(ValueError, match="Invalid characters in CoreML model name"):
            CoreMLWrapper(EyeGenConfig(model="sd-2-1-base; rm -rf /"))


class TestCoreMLSubprocessIntegration:
    @mock.patch("eyegen.backends.coreml.pipeline.validate_coreml_install")
    def test_coreml_subprocess_integration(self, mock_validate, tmp_path):
        import sys

        from eyegen.backends.coreml.pipeline import CoreMLWrapper

        mock_status = mock.Mock()
        mock_status.installed = True
        mock_validate.return_value = mock_status

        model_dir = tmp_path / "sd-2-1-base"
        model_dir.mkdir()

        wrapper = CoreMLWrapper(
            EyeGenConfig(subprocess_timeout=5, coreml_model_path=str(model_dir))
        )

        with mock.patch.object(wrapper, "_validate_cmd_args"):
            returncode, stdout, stderr = wrapper._execute_subprocess(
                [sys.executable, "-c", "import sys; print('ok'); sys.stdout.flush()"], timeout=2.0
            )
        assert returncode == 0
        assert any("ok" in line for line in stdout)

    @mock.patch("eyegen.backends.coreml.pipeline.validate_coreml_install")
    def test_coreml_subprocess_timeout(self, mock_validate, tmp_path):
        import sys

        from eyegen.backends.coreml.pipeline import CoreMLWrapper

        mock_status = mock.Mock()
        mock_status.installed = True
        mock_validate.return_value = mock_status

        model_dir = tmp_path / "sd-2-1-base"
        model_dir.mkdir()

        wrapper = CoreMLWrapper(
            EyeGenConfig(subprocess_timeout=5, coreml_model_path=str(model_dir))
        )

        with mock.patch.object(wrapper, "_validate_cmd_args"):
            with pytest.raises(RuntimeError, match="timed out"):
                wrapper._execute_subprocess(
                    [
                        sys.executable,
                        "-c",
                        "import time, sys; print('running'); sys.stdout.flush(); time.sleep(10)",
                    ],
                    timeout=0.5,
                )

    @mock.patch("eyegen.backends.coreml.pipeline.validate_coreml_install")
    def test_coreml_subprocess_cancellation(self, mock_validate, tmp_path):
        import sys
        import threading
        import time

        from eyegen.backends.coreml.pipeline import CoreMLWrapper

        mock_status = mock.Mock()
        mock_status.installed = True
        mock_validate.return_value = mock_status

        model_dir = tmp_path / "sd-2-1-base"
        model_dir.mkdir()

        wrapper = CoreMLWrapper(
            EyeGenConfig(subprocess_timeout=5, coreml_model_path=str(model_dir))
        )

        def cancel_after_delay():
            time.sleep(0.2)
            wrapper.cancel()

        t = threading.Thread(target=cancel_after_delay)
        t.start()

        with mock.patch.object(wrapper, "_validate_cmd_args"):
            with pytest.raises(RuntimeError, match="cancelled"):
                wrapper._execute_subprocess(
                    [
                        sys.executable,
                        "-c",
                        "import time, sys; print('waiting'); sys.stdout.flush(); time.sleep(5)",
                    ],
                    timeout=3.0,
                )
        t.join()
