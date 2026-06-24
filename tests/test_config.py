import json

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

    cfg_zero = EyeGenConfig(height=0, width=1024)
    errors_zero = cfg_zero.validate()
    assert any("greater than 0" in e for e in errors_zero)

    cfg_neg = EyeGenConfig(height=-8, width=1024)
    errors_neg = cfg_neg.validate()
    assert any("greater than 0" in e for e in errors_neg)


def test_config_validate_catches_invalid_backend():
    cfg = EyeGenConfig()
    cfg.backend = "not-a-backend"
    errors = cfg.validate()
    assert any("Backend" in e for e in errors)


def test_config_validate_mflux_model_path_must_exist(tmp_path):
    cfg = EyeGenConfig(mflux_model_path=str(tmp_path / "nonexistent"))
    errors = cfg.validate()
    assert any("does not exist" in e for e in errors)

    sub = tmp_path / "my_model"
    sub.mkdir()
    cfg_ok = EyeGenConfig(mflux_model_path=str(sub))
    assert not any("does not exist" in e for e in cfg_ok.validate())


def test_config_from_dict_coerces_mflux_quantize_none():
    cfg = EyeGenConfig.from_dict({"mflux_quantize": "null"})
    assert cfg.mflux_quantize is None


def test_config_from_dict_filters_unknown_keys(caplog):
    import logging

    caplog.set_level(logging.WARNING)
    cfg = EyeGenConfig.from_dict({"unknown_key": "value", "height": 512})
    assert cfg.height == 512
    assert not hasattr(cfg, "unknown_key")
    assert any("unknown_key" in msg for msg in caplog.messages)


def test_config_to_dict_round_trip_all_fields(tmp_path):
    """Full round-trip with all config fields exercised."""
    mflux_path = str(tmp_path / "models" / "mflux")
    coreml_path = str(tmp_path / "models" / "coreml")
    cfg = EyeGenConfig(
        backend=Backend.BONSAI,
        model="bonsai-ternary-mlx",
        width=768,
        height=768,
        guidance_scale=3.5,
        num_inference_steps=20,
        mflux_quantize=8,
        mflux_model_path=mflux_path,
        coreml_model_path=coreml_path,
        subprocess_timeout=60,
    )
    data = cfg.to_dict()
    restored = EyeGenConfig.from_dict(data)
    assert restored.backend == Backend.BONSAI
    assert restored.model == cfg.model
    assert restored.width == 768
    assert restored.height == 768
    assert restored.guidance_scale == 3.5
    assert restored.num_inference_steps == 20
    assert restored.mflux_quantize == 8
    assert restored.mflux_model_path == mflux_path
    assert restored.coreml_model_path == coreml_path
    assert restored.subprocess_timeout == 60


def test_load_config_migration(tmp_path, monkeypatch):
    from eyegen.config import load_config

    fake_config = tmp_path / "config.json"
    monkeypatch.setattr("eyegen.config.CONFIG_FILE", fake_config)

    # 1. Create a config with old string backend
    data = {"backend": "mlx", "width": 512, "height": 512}
    with open(fake_config, "w") as f:
        json.dump(data, f)

    loaded = load_config()
    assert loaded.backend == "mlx"
    assert loaded.width == 512

    # Verify it saved back the migrated config
    assert fake_config.exists()
    with open(fake_config, "r") as f:
        saved_data = json.load(f)
    assert saved_data["backend"] == "mlx"
    assert "subprocess_timeout" in saved_data


def test_load_config_corrupted_backup(tmp_path, monkeypatch):
    from eyegen.config import load_config

    fake_config = tmp_path / "config.json"
    monkeypatch.setattr("eyegen.config.CONFIG_FILE", fake_config)

    # Write invalid json
    fake_config.write_text("invalid json contents")

    loaded = load_config()
    # Should fallback to defaults
    assert loaded.width == 1024

    # Should have backed up the bad config
    backup = tmp_path / "config.json.bak"
    assert backup.exists()
    assert backup.read_text() == "invalid json contents"

    # Should have rewritten a valid config
    assert fake_config.exists()
