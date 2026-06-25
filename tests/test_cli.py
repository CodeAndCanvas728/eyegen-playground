"""Smoke tests for CLI helper functions that don't require subprocess or real backends."""

from unittest.mock import patch

import pytest
import typer

from eyegen.cli._generate_helpers import (
    build_generation_params,
    handle_import_error,
    load_pipeline,
    print_generation_settings,
    resolve_output,
    setup_img2img,
    validate_cli_backend,
)
from eyegen.config import Backend, EyeGenConfig


class TestValidateCliBackend:
    def test_valid_backend(self):
        assert validate_cli_backend("mflux") == "mflux"

    def test_invalid_backend_exits(self):
        with pytest.raises(typer.Exit):
            validate_cli_backend("nonexistent")


class TestResolveOutput:
    def test_default_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr("eyegen.cli._generate_helpers.OUTPUT_DIR", tmp_path)
        p = resolve_output(None)
        assert p.parent == tmp_path
        assert p.suffix == ".png"

    def test_custom_path(self, tmp_path):
        custom = tmp_path / "sub" / "out.png"
        p = resolve_output(custom)
        assert p == custom
        assert custom.parent.exists()


class TestSetupImg2img:
    def test_no_image(self, tmp_path):
        image_path, denoise = setup_img2img(None, None, Backend.MLX, None, None)
        assert image_path is None
        assert denoise == 1.0

    def test_denoise_without_image_warns(self, capsys):
        image_path, denoise = setup_img2img(None, 0.75, Backend.MLX, None, None)
        captured = capsys.readouterr()
        assert "denoise has no effect" in captured.out
        assert image_path is None
        assert denoise == 1.0

    def test_valid_image_path(self, tmp_path):
        p = tmp_path / "input.png"
        p.write_text("fake")
        image_path, denoise = setup_img2img(p, None, Backend.MLX, None, None)
        assert image_path == str(p)
        assert denoise == 0.75

    def test_missing_image_exits(self, tmp_path):
        with pytest.raises(typer.Exit):
            setup_img2img(tmp_path / "missing.png", None, Backend.MLX, None, None)


class TestBuildGenerationParams:
    @patch("eyegen.cli._generate_helpers.load_config")
    def test_basic_flow(self, mock_load_config, tmp_path, monkeypatch):
        mock_load_config.return_value = EyeGenConfig()
        monkeypatch.setattr("eyegen.cli._generate_helpers.OUTPUT_DIR", tmp_path)
        result = build_generation_params("mlx", None, None, None, None, None, None, None, None)
        (
            config,
            model,
            backend,
            steps,
            guidance,
            h,
            w,
            img_path,
            denoise,
            out_path,
            quantize,
        ) = result
        assert backend == Backend.MLX
        assert model == EyeGenConfig().model
        assert denoise == 1.0
        assert out_path.parent == tmp_path


class TestPrintGenerationSettings:
    def test_output_contains_expected_labels(self, capsys):
        config = EyeGenConfig(backend=Backend.MFLUX)
        print_generation_settings(
            "dev", Backend.MFLUX, config, "a cat", None, 1.0, 1024, 1024, 30, 7.5, 42
        )
        captured = capsys.readouterr()
        assert "MFLUX" in captured.out
        assert "a cat" in captured.out

    def test_bonsai_label(self, capsys):
        config = EyeGenConfig(backend=Backend.BONSAI, bonsai_model_path="bonsai-ternary-mlx")
        print_generation_settings(
            "bonsai-ternary-mlx", Backend.BONSAI, config, "test", None, 1.0, 512, 512, 10, 1.0, None
        )
        captured = capsys.readouterr()
        assert "Bonsai" in captured.out


class TestHandleImportError:
    def test_ollama(self):
        with pytest.raises(typer.Exit):
            handle_import_error(Backend.OLLAMA)

    def test_mflux(self):
        with pytest.raises(typer.Exit):
            handle_import_error(Backend.MFLUX)

    def test_default(self):
        with pytest.raises(typer.Exit):
            handle_import_error(Backend.MLX)


class TestLoadPipeline:
    @patch("eyegen.cli._generate_helpers.get_ollama_pipeline")
    def test_ollama(self, mock_get):
        mock_get.return_value = "engine"
        result = load_pipeline(Backend.OLLAMA, EyeGenConfig(), None)
        assert result == "engine"

    @patch("eyegen.cli._generate_helpers.get_mflux_pipeline")
    def test_mflux(self, mock_get):
        mock_get.return_value = "model"
        result = load_pipeline(Backend.MFLUX, EyeGenConfig(), 4)
        assert result == "model"
        mock_get.assert_called_once()

    @patch("eyegen.cli._generate_helpers.bonsai.get_bonsai_pipeline")
    def test_bonsai(self, mock_get):
        mock_get.return_value = "wrapper"
        result = load_pipeline(Backend.BONSAI, EyeGenConfig(), None)
        assert result == "wrapper"

    @patch("eyegen.cli._generate_helpers.coreml.get_coreml_pipeline")
    def test_coreml(self, mock_get):
        mock_get.return_value = "wrapper"
        result = load_pipeline(Backend.COREML, EyeGenConfig(), None)
        assert result == "wrapper"

    @patch("eyegen.cli._generate_helpers.get_pipeline")
    def test_default_mlx(self, mock_get):
        mock_get.return_value = "pipeline"
        result = load_pipeline(Backend.MLX, EyeGenConfig(), None)
        assert result == "pipeline"
