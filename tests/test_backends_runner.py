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
