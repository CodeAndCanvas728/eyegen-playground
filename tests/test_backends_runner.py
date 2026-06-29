"""Tests for BaseSubprocessRunner."""

import subprocess
from unittest import mock

import pytest

from eyegen.backends.runner import BaseSubprocessRunner
from eyegen.config import EyeGenConfig


class DummyRunner(BaseSubprocessRunner):
    """Subclass of BaseSubprocessRunner for testing."""

    def _validate_cmd_args(self, cmd):
        super()._validate_cmd_args(cmd)


def test_runner_execute_success():
    cfg = EyeGenConfig(subprocess_timeout=5)
    runner = DummyRunner(cfg)

    mock_proc = mock.Mock()
    mock_proc.returncode = 0
    mock_proc.stdout.readline.side_effect = ["stdout_line\n", ""]
    mock_proc.stderr.readline.side_effect = ["stderr_line\n", ""]

    with mock.patch("eyegen.backends.runner.subprocess.Popen") as mock_popen:
        mock_popen.return_value = mock_proc
        rc, out, err = runner._execute_subprocess(["dummy_cmd"])
        assert rc == 0
        assert out == ["stdout_line\n"]
        assert err == ["stderr_line\n"]


def test_runner_timeout():
    cfg = EyeGenConfig(subprocess_timeout=0.1)
    runner = DummyRunner(cfg)

    mock_proc = mock.Mock()
    mock_proc.pid = 9999
    mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="dummy_cmd", timeout=0.1)

    with mock.patch("eyegen.backends.runner.subprocess.Popen") as mock_popen:
        mock_popen.return_value = mock_proc
        with pytest.raises(RuntimeError, match="subprocess timed out after 0.1 seconds"):
            runner._execute_subprocess(["dummy_cmd"])
        assert mock_proc.terminate.called


def test_runner_cancel():
    cfg = EyeGenConfig(subprocess_timeout=5)
    runner = DummyRunner(cfg)

    mock_proc = mock.Mock()
    mock_proc.pid = 1234
    mock_proc.returncode = 0

    # We trigger runner.cancel() during the self._proc.wait() call
    # Use a flag to avoid infinite recursion
    cancel_called = False

    def mock_wait(*args, **kwargs):
        nonlocal cancel_called
        if not cancel_called:
            cancel_called = True
            runner.cancel()
        # For subsequent calls from _terminate_proc, just return
        return 0

    mock_proc.wait.side_effect = mock_wait

    with mock.patch("eyegen.backends.runner.subprocess.Popen") as mock_popen:
        mock_popen.return_value = mock_proc
        with pytest.raises(RuntimeError, match="generation was cancelled by the user"):
            runner._execute_subprocess(["dummy_cmd"])
        assert mock_proc.terminate.called


def test_runner_integration_real_process():
    import sys

    cfg = EyeGenConfig(subprocess_timeout=5)
    runner = DummyRunner(cfg)
    cmd = [
        sys.executable,
        "-c",
        "import sys; print('hello from stdout'); print('hello from stderr', file=sys.stderr)",
    ]
    with mock.patch.object(runner, "_validate_cmd_args"):
        rc, out, err = runner._execute_subprocess(
            cmd,
            stream_stdout=True,
            stream_stderr=True,
            log_prefix="test-integration",
        )
    assert rc == 0
    assert any("hello from stdout" in line for line in out)
    assert any("hello from stderr" in line for line in err)


def test_runner_integration_timeout():
    import sys

    cfg = EyeGenConfig(subprocess_timeout=0.5)
    runner = DummyRunner(cfg)
    cmd = [
        sys.executable,
        "-c",
        "import time; time.sleep(10)",
    ]
    with mock.patch.object(runner, "_validate_cmd_args"):
        with pytest.raises(RuntimeError, match="subprocess timed out after 0.5 seconds"):
            runner._execute_subprocess(cmd, log_prefix="test-timeout")


def test_runner_integration_cancel():
    import sys
    import threading
    import time

    cfg = EyeGenConfig(subprocess_timeout=5)
    runner = DummyRunner(cfg)
    cmd = [
        sys.executable,
        "-c",
        "import time; time.sleep(10)",
    ]

    def trigger_cancel():
        time.sleep(0.2)
        runner.cancel()

    t = threading.Thread(target=trigger_cancel)
    t.start()
    try:
        with mock.patch.object(runner, "_validate_cmd_args"):
            with pytest.raises(RuntimeError, match="generation was cancelled by the user"):
                runner._execute_subprocess(cmd, log_prefix="test-cancel")
    finally:
        t.join()


def test_runner_integration_injection_rejection():
    import sys

    cfg = EyeGenConfig(subprocess_timeout=5)
    runner = DummyRunner(cfg)
    cmd = [
        sys.executable,
        "--unsafe-flag",
    ]
    with pytest.raises(ValueError, match="Unsafe subprocess flag detected"):
        runner._execute_subprocess(cmd)
