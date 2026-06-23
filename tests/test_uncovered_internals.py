"""Tests for the five uncovered internal functions in eyegen."""

import sys
from unittest.mock import MagicMock, patch

# Mock mlx and mflux before importing the module under test
mock_mflux = MagicMock()
mock_mlx = MagicMock()

# Setup all possible import paths that might be accessed
sys.modules["mflux"] = mock_mflux
sys.modules["mflux.models"] = mock_mflux
sys.modules["mflux.models.flux"] = mock_mflux
sys.modules["mflux.models.flux.variants"] = mock_mflux
sys.modules["mflux.models.flux.variants.txt2img"] = mock_mflux
sys.modules["mflux.models.flux.variants.txt2img.flux"] = mock_mflux
sys.modules["mflux.models.flux2"] = mock_mflux
sys.modules["mflux.models.flux2.variants"] = mock_mflux
sys.modules["mflux.models.flux2.variants.txt2img"] = mock_mflux
sys.modules["mflux.models.flux2.variants.txt2img.flux2_klein"] = mock_mflux
sys.modules["mflux.models.z_image"] = mock_mflux
sys.modules["mflux.models.z_image.variants"] = mock_mflux
sys.modules["mflux.models.z_image.variants.z_image"] = mock_mflux
sys.modules["mflux.models.fibo"] = mock_mflux
sys.modules["mflux.models.fibo.variants"] = mock_mflux
sys.modules["mflux.models.fibo.variants.txt2img"] = mock_mflux
sys.modules["mflux.models.fibo.variants.txt2img.fibo"] = mock_mflux
sys.modules["mflux.models.qwen"] = mock_mflux
sys.modules["mflux.models.qwen.variants"] = mock_mflux
sys.modules["mflux.models.qwen.variants.txt2img"] = mock_mflux
sys.modules["mflux.models.qwen.variants.txt2img.qwen_image"] = mock_mflux
sys.modules["mflux.models.seedvr2"] = mock_mflux
sys.modules["mflux.models.seedvr2.variants"] = mock_mflux
sys.modules["mflux.models.seedvr2.variants.upscale"] = mock_mflux
sys.modules["mflux.models.seedvr2.variants.upscale.seedvr2"] = mock_mflux
sys.modules["mlx"] = mock_mlx
sys.modules["mlx.core"] = mock_mlx
sys.modules["mlx.core.fast"] = mock_mlx

# Set fake class references on mock_mflux
mock_mflux.Flux1 = type("Flux1", (), {})
mock_mflux.Flux2Klein = type("Flux2Klein", (), {})
mock_mflux.ZImage = type("ZImage", (), {})
mock_mflux.FIBO = type("FIBO", (), {})
mock_mflux.QwenImage = type("QwenImage", (), {})
mock_mflux.SeedVR2 = type("SeedVR2", (), {})

from eyegen._mflux import (  # noqa: E402
    _format_mflux_model_error,
    _get_mflux_aliases,
    _resolve_mflux_class,
)
from eyegen._mlx import (  # noqa: E402
    _make_patched,
    _patch_mlx_attention,
)


def test_get_mflux_aliases():
    # Make sure we don't fail even if live package can't be introspected
    aliases = _get_mflux_aliases()
    assert isinstance(aliases, set)
    assert "dev" in aliases
    assert "schnell" in aliases


def test_resolve_mflux_class():
    # 1. Test standard/default model Flux1
    mock_cfg = MagicMock()
    mock_cfg.model_name = "black-forest-labs/FLUX.1-schnell"
    mock_cfg.aliases = ["schnell"]
    assert _resolve_mflux_class(mock_cfg) == mock_mflux.Flux1

    # 2. Test Klein variant
    mock_cfg_klein = MagicMock()
    mock_cfg_klein.model_name = "black-forest-labs/FLUX.2-klein-4B"
    mock_cfg_klein.aliases = ["flux2-klein-4b"]
    assert _resolve_mflux_class(mock_cfg_klein) == mock_mflux.Flux2Klein

    # 3. Test ZImage variant
    mock_cfg_z = MagicMock()
    mock_cfg_z.model_name = "Tongyi-MAI/Z-Image"
    mock_cfg_z.aliases = ["z-image"]
    assert _resolve_mflux_class(mock_cfg_z) == mock_mflux.ZImage

    # 4. Test FIBO variant
    mock_cfg_fibo = MagicMock()
    mock_cfg_fibo.model_name = "briaai/FIBO"
    mock_cfg_fibo.aliases = ["fibo"]
    assert _resolve_mflux_class(mock_cfg_fibo) == mock_mflux.FIBO

    # 5. Test QwenImage variant
    mock_cfg_qwen = MagicMock()
    mock_cfg_qwen.model_name = "Qwen/Qwen-Image"
    mock_cfg_qwen.aliases = ["qwen-image"]
    assert _resolve_mflux_class(mock_cfg_qwen) == mock_mflux.QwenImage

    # 6. Test SeedVR2 variant
    mock_cfg_seed = MagicMock()
    mock_cfg_seed.model_name = "numz/SeedVR2_comfyUI"
    mock_cfg_seed.aliases = ["seedvr2"]
    assert _resolve_mflux_class(mock_cfg_seed) == mock_mflux.SeedVR2


def test_format_mflux_model_error():
    orig_exc = Exception("Test exception details")
    val_err = _format_mflux_model_error("bad-model-name", orig_exc)
    assert isinstance(val_err, ValueError)
    assert "bad-model-name" in str(val_err)
    assert "Test exception details" in str(val_err)
    assert "Known aliases:" in str(val_err)


def test_make_patched():
    called_kwargs = {}

    def dummy_func(*args, **kwargs):
        called_kwargs.update(kwargs)
        return "success"

    patched = _make_patched(dummy_func)
    res = patched(1, 2, a=3, memory_efficient_threshold=100)
    assert res == "success"
    assert "a" in called_kwargs
    assert called_kwargs["a"] == 3
    assert "memory_efficient_threshold" not in called_kwargs


def test_patch_mlx_attention():
    mock_mlx_core = MagicMock()
    mock_mlx_core_fast = MagicMock()

    # Set attribute existence to True
    mock_mlx_core.scaled_dot_product_attention = lambda: None
    mock_mlx_core_fast.scaled_dot_product_attention = lambda: None

    with (
        patch("sys.modules", {"mlx.core": mock_mlx_core, "mlx.core.fast": mock_mlx_core_fast}),
        patch("eyegen._mlx._patched_already", False),
    ):
        _patch_mlx_attention()
        assert (
            mock_mlx_core.scaled_dot_product_attention
            != mock_mlx_core_fast.scaled_dot_product_attention
        )
