import subprocess
from unittest import mock
import pytest


pytest.importorskip("PySide6")


from eyegen.gui.backend_workers import BonsaiSetupWorker


@mock.patch("eyegen.gui.backend_workers.subprocess.run")
def test_script_worker_timeout(mock_run):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="setup-bonsai.sh", timeout=600.0)

    worker = BonsaiSetupWorker("/path/to/setup-bonsai.sh")

    finished_called = []
    def on_finished(success, message):
        finished_called.append((success, message))

    worker.finished.connect(on_finished)
    worker.run()

    assert len(finished_called) == 1
    success, message = finished_called[0]
    assert not success
    assert "timed out after 10 minutes" in message
