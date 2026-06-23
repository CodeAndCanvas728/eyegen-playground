"""Tests for eyegen.backends."""

from unittest.mock import patch

import pytest

from eyegen.backends import _is_bonsai_model, _is_coreml_model, detect_backend
from eyegen.config import Backend


def test_detect_backend_gguf():
    assert detect_backend("foo-gguf", Backend.AUTO) == Backend.OLLAMA


def test_detect_backend_bonsai():
    assert detect_backend("bonsai-ternary-mlx", Backend.AUTO) == Backend.BONSAI


def test_detect_backend_coreml_alias():
    assert detect_backend("sd-2-1-base", Backend.AUTO) == Backend.COREML


def test_detect_backend_mflux_alias():
    with patch("eyegen.backends._get_mflux_aliases", return_value={"dev"}):
        assert detect_backend("dev", Backend.AUTO) == Backend.MFLUX


def test_detect_backend_override():
    assert detect_backend("anything", Backend.OLLAMA) == Backend.OLLAMA


def test_detect_backend_unsupported_raises():
    with (
        patch("eyegen.backends._get_mflux_aliases", return_value=set()),
        patch("eyegen.backends._get_mlx_supported_models", return_value=set()),
    ):
        with pytest.raises(ValueError):
            detect_backend("totally-unknown-model", Backend.AUTO)


<<<<<<< Updated upstream
def test_detect_backend_unsupported_raises_when_diffusionkit_missing():
    # When diffusionkit is unavailable, _get_mlx_supported_models returns None;
    # an unsupported model must still raise rather than silently routing to MLX.
=======
def test_detect_backend_mlx_missing_raises():
>>>>>>> Stashed changes
    with (
        patch("eyegen.backends._get_mflux_aliases", return_value=set()),
        patch("eyegen.backends._get_mlx_supported_models", return_value=None),
    ):
        with pytest.raises(ValueError):
            detect_backend("totally-unknown-model", Backend.AUTO)


def test_is_bonsai_model():
    assert _is_bonsai_model("bonsai-ternary-mlx")
    assert not _is_bonsai_model("dev")


def test_is_coreml_model_with_path():
    assert _is_coreml_model("anything", config={"coreml_model_path": "/some/path"})
