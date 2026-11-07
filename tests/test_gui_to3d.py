"""Tests for the Image-to-3D (Idol) GUI tab and its worker."""

import os
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


def test_main_window_has_to3d_tab(monkeypatch):
    pytest.importorskip("PySide6")
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
