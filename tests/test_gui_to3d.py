"""Tests for the Image-to-3D (Idol) GUI tab and its worker."""

import os
import types
from unittest import mock

import pytest


def test_to3d_worker_runs_runner(tmp_path):
    pytest.importorskip("PySide6")
    from eyegen.config import EyeGenConfig
    from eyegen.gui.backend_workers import To3DWorker

    ply = tmp_path / "out" / "photo.ply"
    fake_result = mock.Mock()
    fake_result.ply = ply

    with mock.patch("eyegen.backends.idol.IdolRunner") as runner_cls:
        runner_cls.return_value.run.return_value = fake_result
        worker = To3DWorker(
            image=tmp_path / "photo.jpg",
            out_dir=tmp_path / "out",
            mode="real",
            steps=5,
            num_gaussians=32768,
            seed=42,
            device="mps",
            config=EyeGenConfig(),
        )
        finished = []
        worker.finished.connect(lambda result, msg: finished.append((result, msg)))
        worker.run()

    assert finished == [(fake_result, f"3D splat written: {ply}")]
    runner_cls.return_value.run.assert_called_once()
    kwargs = runner_cls.return_value.run.call_args.kwargs
    assert kwargs["mode"] == "real"
    assert kwargs["num_gaussians"] == 32768


def test_to3d_worker_error(tmp_path):
    pytest.importorskip("PySide6")
    from eyegen.config import EyeGenConfig
    from eyegen.gui.backend_workers import To3DWorker

    with mock.patch("eyegen.backends.idol.IdolRunner") as runner_cls:
        runner_cls.return_value.run.side_effect = RuntimeError("setup-idol first")
        worker = To3DWorker(
            image=tmp_path / "photo.jpg",
            out_dir=tmp_path / "out",
            mode="synthetic",
            steps=5,
            num_gaussians=32768,
            seed=42,
            device="mps",
            config=EyeGenConfig(),
        )
        errors = []
        worker.error.connect(errors.append)
        worker.run()

    assert errors == ["setup-idol first"]


def test_to3d_worker_cancel_calls_runner(tmp_path):
    pytest.importorskip("PySide6")
    from eyegen.config import EyeGenConfig
    from eyegen.gui.backend_workers import To3DWorker

    worker = To3DWorker(
        image=tmp_path / "photo.jpg",
        out_dir=tmp_path / "out",
        mode="real",
        steps=5,
        num_gaussians=32768,
        seed=42,
        device="mps",
        config=EyeGenConfig(),
    )
    runner = mock.Mock()
    with worker._runner_lock:
        worker._runner = runner
    worker.cancel()
    runner.cancel.assert_called_once()


def test_to3d_worker_cancel_without_runner_is_noop(tmp_path):
    pytest.importorskip("PySide6")
    from eyegen.config import EyeGenConfig
    from eyegen.gui.backend_workers import To3DWorker

    worker = To3DWorker(
        image=tmp_path / "photo.jpg",
        out_dir=tmp_path / "out",
        mode="real",
        steps=5,
        num_gaussians=32768,
        seed=42,
        device="mps",
        config=EyeGenConfig(),
    )
    worker.cancel()  # _runner is None: must not raise


def _require_qt_widgets():
    """Skip where Qt's GUI libraries cannot load.

    importorskip("PySide6") only imports the top-level package, which works
    without libGL; QtWidgets does not, and the CI runner has no libGL.
    """
    try:
        from PySide6.QtWidgets import QApplication  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"PySide6.QtWidgets cannot load here: {exc}")


def _make_window(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from eyegen.config import EyeGenConfig

    app = QApplication.instance() or QApplication([])
    import eyegen.gui.main_window as mw

    monkeypatch.setattr(mw, "load_config", lambda: EyeGenConfig())
    monkeypatch.setattr(mw, "load_gui_state", lambda: {})
    monkeypatch.setattr(mw, "pop_config_warnings", lambda: [])
    return mw.MainWindow(), app


def test_to3d_run_button_stays_enabled_and_toggles(monkeypatch, tmp_path):
    _require_qt_widgets()
    from eyegen.backends import idol

    window, app = _make_window(monkeypatch)
    try:
        image = tmp_path / "photo.jpg"
        image.write_bytes(b"x")
        window.to3d_image_input.setText(str(image))
        monkeypatch.setattr(idol, "validate_idol_install", lambda config: mock.Mock(installed=True))
        fake_worker = mock.Mock()
        fake_worker.isRunning.return_value = False
        with mock.patch("eyegen.gui.main_window_to3d.To3DWorker", return_value=fake_worker):
            window._on_to3d_run()
        # While the worker runs the button is the Cancel control, not disabled.
        assert window.to3d_run_btn.isEnabled()
        assert window.to3d_run_btn.text() == "Cancel"
        result = types.SimpleNamespace(
            ply=tmp_path / "out" / "photo.ply", input_png=tmp_path / "nope.png"
        )
        window._on_to3d_finished(result, "done")
        assert window.to3d_run_btn.text() == "Generate 3D"
        window._on_to3d_error("boom")
        assert window.to3d_run_btn.text() == "Generate 3D"
    finally:
        window.close()
        del app


def test_to3d_run_cancel_branch_reaches_worker(monkeypatch):
    _require_qt_widgets()
    window, app = _make_window(monkeypatch)
    try:
        fake_worker = mock.Mock()
        fake_worker.isRunning.return_value = True
        window.to3d_worker = fake_worker
        window._on_to3d_run()
        fake_worker.cancel.assert_called_once()
        assert window.to3d_status_label.text() == "Cancelling…"
    finally:
        window.close()
        del app


def test_main_window_has_to3d_tab(monkeypatch):
    _require_qt_widgets()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from eyegen.config import EyeGenConfig

    app = QApplication.instance() or QApplication([])
    import eyegen.gui.main_window as mw

    monkeypatch.setattr(mw, "load_config", lambda: EyeGenConfig())
    monkeypatch.setattr(mw, "load_gui_state", lambda: {})
    monkeypatch.setattr(mw, "pop_config_warnings", lambda: [])

    window = mw.MainWindow()
    labels = [window.nav_bar.tabText(i) for i in range(window.nav_bar.count())]
    assert labels == ["Home", "History", "Settings", "Image to 3D"]
    assert window.content_stack.count() == 4
    assert hasattr(window, "to3d_run_btn")
    assert hasattr(window, "to3d_image_input")
    window.close()
    del app
