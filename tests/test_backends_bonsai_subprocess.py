"""Subprocess integration tests for the Bonsai backend."""

import sys
import threading
import time
from unittest import mock

import pytest


class TestBonsaiSubprocessIntegration:
    @mock.patch("eyegen.backends.bonsai.pipeline.validate_bonsai_install")
    def test_bonsai_subprocess_integration(self, mock_validate, tmp_path):
        mock_status = mock.Mock()
        mock_status.installed = True
        mock_status.has_models = True
        mock_status.bonsai_dir = tmp_path
        mock_validate.return_value = mock_status

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / "generate.sh").touch()

        from eyegen.backends.bonsai.pipeline import BonsaiWrapper
        from eyegen.config import EyeGenConfig

        wrapper = BonsaiWrapper(EyeGenConfig(subprocess_timeout=5, model="bonsai-ternary-mlx"))

        with mock.patch.object(wrapper, "_validate_cmd_args"):
            returncode, stdout, stderr = wrapper._execute_subprocess(
                [sys.executable, "-c", "import sys; print('ok'); sys.stdout.flush()"], timeout=2.0
            )
        assert returncode == 0
        assert any("ok" in line for line in stdout)

    @mock.patch("eyegen.backends.bonsai.pipeline.validate_bonsai_install")
    def test_bonsai_subprocess_timeout(self, mock_validate, tmp_path):
        mock_status = mock.Mock()
        mock_status.installed = True
        mock_status.has_models = True
        mock_status.bonsai_dir = tmp_path
        mock_validate.return_value = mock_status

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / "generate.sh").touch()

        from eyegen.backends.bonsai.pipeline import BonsaiWrapper
        from eyegen.config import EyeGenConfig

        wrapper = BonsaiWrapper(EyeGenConfig(subprocess_timeout=5, model="bonsai-ternary-mlx"))

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

    @mock.patch("eyegen.backends.bonsai.pipeline.validate_bonsai_install")
    def test_bonsai_subprocess_cancellation(self, mock_validate, tmp_path):
        mock_status = mock.Mock()
        mock_status.installed = True
        mock_status.has_models = True
        mock_status.bonsai_dir = tmp_path
        mock_validate.return_value = mock_status

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / "generate.sh").touch()

        from eyegen.backends.bonsai.pipeline import BonsaiWrapper
        from eyegen.config import EyeGenConfig

        wrapper = BonsaiWrapper(EyeGenConfig(subprocess_timeout=5, model="bonsai-ternary-mlx"))

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
