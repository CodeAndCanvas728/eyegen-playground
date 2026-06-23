"""Tests for eyegen.generation."""

from unittest.mock import MagicMock, patch

from eyegen.config import Backend
from eyegen.generation import generate_image


def test_generate_image_dispatches_to_mlx():
    pipeline = MagicMock()
    pipeline.generate_image.return_value = MagicMock()
    with patch("eyegen.generation._generate_image_mlx") as mock_mlx:
        generate_image(
            pipeline,
            "a cat",
            7.5,
            30,
            512,
            512,
            backend=Backend.MLX,
        )
        mock_mlx.assert_called_once()


def test_generate_image_dispatches_to_ollama():
    pipeline = MagicMock()
    with patch("eyegen.generation._generate_image_ollama") as mock_ollama:
        generate_image(
            pipeline,
            "a cat",
            7.5,
            30,
            512,
            512,
            backend=Backend.OLLAMA,
        )
        mock_ollama.assert_called_once()


def test_generate_image_dispatches_to_mflux():
    pipeline = MagicMock()
    with patch("eyegen.generation._generate_image_mflux") as mock_mflux:
        generate_image(
            pipeline,
            "a cat",
            7.5,
            30,
            512,
            512,
            backend=Backend.MFLUX,
        )
        mock_mflux.assert_called_once()


def test_generate_image_dispatches_to_bonsai():
    pipeline = MagicMock()
    generate_image(
        pipeline,
        "a cat",
        7.5,
        30,
        512,
        512,
        backend=Backend.BONSAI,
    )
    pipeline.generate_image.assert_called_once()
