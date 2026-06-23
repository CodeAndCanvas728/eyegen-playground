"""Tests for eyegen.config."""

import pytest

from eyegen.config import Backend, EyeGenConfig


def test_default_config_values():
    cfg = EyeGenConfig()
    assert cfg.backend == Backend.AUTO
    assert cfg.height == 1024
    assert cfg.width == 1024


def test_config_from_dict_coerces_backend_string():
    cfg = EyeGenConfig.from_dict({"backend": "mflux"})
    assert cfg.backend == Backend.MFLUX


def test_config_to_dict_round_trip():
    cfg = EyeGenConfig(backend=Backend.MFLUX, height=512, width=512)
    data = cfg.to_dict()
    assert data["backend"] == "mflux"
    restored = EyeGenConfig.from_dict(data)
    assert restored.backend == Backend.MFLUX
    assert restored.height == 512


def test_config_validate_catches_bad_dimensions():
    cfg = EyeGenConfig(height=100, width=100)
    errors = cfg.validate()
    assert any("multiples of 8" in e for e in errors)


def test_config_validate_catches_invalid_backend():
    cfg = EyeGenConfig()
    cfg.backend = "not-a-backend"
    errors = cfg.validate()
    assert any("Backend" in e for e in errors)


def test_config_from_dict_coerces_mflux_quantize_none():
    cfg = EyeGenConfig.from_dict({"mflux_quantize": "null"})
    assert cfg.mflux_quantize is None


def test_config_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown configuration key"):
        EyeGenConfig.from_dict({"unknown_key": "value", "height": 512})


def test_config_validate_rejects_zero_dimensions():
    cfg = EyeGenConfig(width=0, height=1024)
    assert any("greater than 0" in e for e in cfg.validate())


def test_config_validate_rejects_negative_dimensions():
    cfg = EyeGenConfig(width=-8, height=1024)
    assert any("greater than 0" in e for e in cfg.validate())
