"""Tests for the Idol (image-to-3D) sidecar backend: command building,
install validation, config validation and the to3d CLI error paths."""

import pytest

from eyegen.backends import idol
from eyegen.backends.idol.runner import IdolRunner
from eyegen.cli import to3d as to3d_cli
from eyegen.config import EyeGenConfig


@pytest.fixture
def fake_sidecar(tmp_path, monkeypatch):
    """An Idol sidecar that looks installed: venv python + SDK checkout."""
    root = tmp_path / "idol"
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").write_text("#!/bin/sh\n")
    (root / "idol-sdk" / "idol").mkdir(parents=True)
    (root / "idol-sdk" / "idol" / "sample.py").write_text("# sample\n")
    monkeypatch.setenv("IDOL_DIR", str(root))
    return root


class TestValidateIdolInstall:
    def test_missing_everything(self, tmp_path, monkeypatch):
        monkeypatch.setenv("IDOL_DIR", str(tmp_path / "nope"))
        status = idol.validate_idol_install()
        assert not status.installed
        assert "setup-idol" in status.message

    def test_installed(self, fake_sidecar):
        status = idol.validate_idol_install()
        assert status.installed
        assert status.venv_python == fake_sidecar / ".venv" / "bin" / "python"


class TestIdolRunnerCommand:
    def test_build_command(self, fake_sidecar, tmp_path):
        runner = IdolRunner(EyeGenConfig())
        image = tmp_path / "photo.jpg"
        image.write_bytes(b"x")
        out = tmp_path / "out"
        cmd = runner._build_command(image, out, "real", 20, 65536, 7, "mps")
        assert cmd[0] == str(fake_sidecar / ".venv" / "bin" / "python")
        assert cmd[1:3] == ["-m", "idol.sample"]
        assert cmd[cmd.index("--image") + 1] == str(image)
        assert cmd[cmd.index("--mode") + 1] == "real"
        assert cmd[cmd.index("--out") + 1] == str(out)
        assert cmd[cmd.index("--device") + 1] == "mps"
        assert cmd[cmd.index("--steps") + 1] == "20"
        assert cmd[cmd.index("--num_gaussians") + 1] == "65536"
        assert cmd[cmd.index("--seed") + 1] == "7"

    def test_run_returns_result_paths(self, fake_sidecar, tmp_path, monkeypatch):
        runner = IdolRunner(EyeGenConfig())
        image = tmp_path / "photo.jpg"
        image.write_bytes(b"x")
        out = tmp_path / "out"

        def fake_exec(cmd, **kwargs):
            out.mkdir(parents=True, exist_ok=True)
            (out / "photo.ply").write_bytes(b"ply")
            (out / "photo_input.png").write_bytes(b"png")
            (out / "photo_camera.json").write_text("{}")
            return 0, ["ok"], []

        monkeypatch.setattr(runner, "_execute_subprocess", fake_exec)
        result = runner.run(image, out, mode="real", steps=5, num_gaussians=32768, seed=1)
        assert result.ply == out / "photo.ply"
        assert result.input_png == out / "photo_input.png"
        assert result.camera_json == out / "photo_camera.json"

    def test_run_nonzero_exit_raises(self, fake_sidecar, tmp_path, monkeypatch):
        runner = IdolRunner(EyeGenConfig())
        image = tmp_path / "photo.jpg"
        image.write_bytes(b"x")
        monkeypatch.setattr(runner, "_execute_subprocess", lambda cmd, **kw: (1, [], ["boom"]))
        with pytest.raises(RuntimeError, match="exit 1"):
            runner.run(image, tmp_path / "out")

    def test_run_missing_ply_raises(self, fake_sidecar, tmp_path, monkeypatch):
        runner = IdolRunner(EyeGenConfig())
        image = tmp_path / "photo.jpg"
        image.write_bytes(b"x")
        monkeypatch.setattr(runner, "_execute_subprocess", lambda cmd, **kw: (0, [], []))
        with pytest.raises(RuntimeError, match="wrote no"):
            runner.run(image, tmp_path / "out")

    def test_not_installed_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("IDOL_DIR", str(tmp_path / "nope"))
        with pytest.raises(RuntimeError, match="setup-idol"):
            IdolRunner(EyeGenConfig())


class TestIdolConfigValidation:
    def test_defaults_are_valid(self):
        assert EyeGenConfig().validate() == []

    def test_bad_gaussians(self):
        errors = EyeGenConfig(idol_num_gaussians=100).validate()
        assert any("idol_num_gaussians" in e for e in errors)

    def test_bad_mode(self):
        errors = EyeGenConfig(idol_mode="weird").validate()
        assert any("idol_mode" in e for e in errors)

    def test_bad_device(self):
        errors = EyeGenConfig(idol_device="tpu").validate()
        assert any("idol_device" in e for e in errors)

    def test_from_dict_coerces(self):
        cfg = EyeGenConfig.from_dict({"idol_steps": "30", "idol_num_gaussians": "65536"})
        assert cfg.idol_steps == 30
        assert cfg.idol_num_gaussians == 65536


class TestTo3dCli:
    @pytest.fixture
    def cli(self, monkeypatch):
        from typer.testing import CliRunner

        from eyegen.cli.app import app

        monkeypatch.setattr(to3d_cli, "load_config", lambda: EyeGenConfig())
        return CliRunner(), app

    def test_missing_image_exits(self, cli, tmp_path):
        runner, app = cli
        result = runner.invoke(app, ["to3d", str(tmp_path / "nope.jpg")])
        assert result.exit_code == 1

    def test_bad_mode_exits(self, cli, tmp_path):
        runner, app = cli
        image = tmp_path / "photo.jpg"
        image.write_bytes(b"x")
        result = runner.invoke(app, ["to3d", str(image), "--mode", "weird"])
        assert result.exit_code == 1
        assert "Invalid mode" in result.output

    def test_bad_gaussians_exits(self, cli, tmp_path):
        runner, app = cli
        image = tmp_path / "photo.jpg"
        image.write_bytes(b"x")
        result = runner.invoke(app, ["to3d", str(image), "--num-gaussians", "100"])
        assert result.exit_code == 1
        assert "num-gaussians" in result.output

    def test_not_installed_exits(self, cli, tmp_path, monkeypatch):
        runner, app = cli
        monkeypatch.setenv("IDOL_DIR", str(tmp_path / "nope"))
        image = tmp_path / "photo.jpg"
        image.write_bytes(b"x")
        result = runner.invoke(app, ["to3d", str(image)])
        assert result.exit_code == 1
        assert "setup-idol" in result.output
