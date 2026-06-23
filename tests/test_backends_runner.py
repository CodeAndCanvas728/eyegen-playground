"""Tests for BaseSubprocessRunner."""

import subprocess
from unittest import mock

import pytest

from eyegen.backends.runner import BaseSubprocessRunner


class DummyRunner(BaseSubprocessRunner):
    """Subclass of BaseSubprocessRunner for testing."""
    pass


def test_runner_execute_success():
    runner = DummyRunner({"subprocess_timeout": 5})

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
    runner = DummyRunner({"subprocess_timeout": 0.1})

    mock_proc = mock.Mock()
    mock_proc.pid = 9999
    mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="dummy_cmd", timeout=0.1)

    with mock.patch("eyegen.backends.runner.subprocess.Popen") as mock_popen:
        mock_popen.return_value = mock_proc
        with pytest.raises(RuntimeError, match="subprocess timed out after 0.1 seconds"):
            runner._execute_subprocess(["dummy_cmd"])
        assert mock_proc.terminate.called


def test_runner_cancel():
    runner = DummyRunner({"subprocess_timeout": 5})

    mock_proc = mock.Mock()
    mock_proc.pid = 1234

    # We trigger runner.cancel() during the self._proc.wait() call
    def mock_wait(*args, **kwargs):
        runner.cancel()
        return 0

    mock_proc.wait.side_effect = mock_wait

    with mock.patch("eyegen.backends.runner.subprocess.Popen") as mock_popen:
        mock_popen.return_value = mock_proc

        with pytest.raises(RuntimeError, match="generation was cancelled by the user"):
            runner._execute_subprocess(["dummy_cmd"])
        assert mock_proc.terminate.called


def test_runner_integration_real_process():
    import sys
    runner = DummyRunner({"subprocess_timeout": 5})
    cmd = [
        sys.executable,
        "-c",
        "import sys; print('hello from stdout'); print('hello from stderr', file=sys.stderr)",
    ]
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
    runner = DummyRunner({"subprocess_timeout": 0.5})
    cmd = [
        sys.executable,
        "-c",
        "import time; time.sleep(10)",
    ]
    with pytest.raises(RuntimeError, match="subprocess timed out after 0.5 seconds"):
        runner._execute_subprocess(cmd, log_prefix="test-timeout")


def test_runner_integration_cancel():
    import sys
    import threading
    import time
    runner = DummyRunner({"subprocess_timeout": 5})
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
        with pytest.raises(RuntimeError, match="generation was cancelled by the user"):
            runner._execute_subprocess(cmd, log_prefix="test-cancel")
    finally:
        t.join()


def test_runner_integration_injection_rejection():
    import sys
    runner = DummyRunner({"subprocess_timeout": 5})
    cmd = [
        sys.executable,
        "--unsafe-flag",
    ]
    with pytest.raises(ValueError, match="Unsafe subprocess flag detected"):
        runner._execute_subprocess(cmd)

