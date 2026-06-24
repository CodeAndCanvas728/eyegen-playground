"""Tests for eyegen._model_ops."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eyegen._model_ops import (
    clear_mflux_cache,
    list_mflux_models,
    save_mflux_model,
    validate_saved_model,
)


class TestListMfluxModels:
    def test_from_live_package(self):
        import sys

        keys = (
            "mflux",
            "mflux.models",
            "mflux.models.common",
            "mflux.models.common.config",
            "mflux.models.common.config.model_config",
        )
        saved = {key: sys.modules.get(key) for key in keys}
        mock_module = MagicMock()
        mock_cfg_a = MagicMock()
        mock_cfg_a.model_name = "black-forest-labs/FLUX.1-dev"
        mock_cfg_a.priority = 0
        mock_cfg_a.aliases = ["dev"]
        mock_cfg_b = MagicMock()
        mock_cfg_b.model_name = "black-forest-labs/FLUX.1-schnell"
        mock_cfg_b.priority = 1
        mock_cfg_b.aliases = ["schnell"]
        mock_module.AVAILABLE_MODELS = {"dev": mock_cfg_a, "schnell": mock_cfg_b}
        for key in keys:
            sys.modules[key] = mock_module

        try:
            result = list_mflux_models()
            assert len(result) == 2
            names = [r["model_name"] for r in result]
            assert "black-forest-labs/FLUX.1-dev" in names
        finally:
            for key in keys:
                if saved[key] is not None:
                    sys.modules[key] = saved[key]
                else:
                    sys.modules.pop(key, None)

    def test_fallback_on_import_error(self):
        with patch("eyegen._model_ops._get_mflux_aliases", return_value={"dev", "schnell"}):
            result = list_mflux_models()
            assert len(result) == 2
            aliases = [r["alias"] for r in result]
            assert "dev" in aliases


class TestClearMfluxCache:
    @pytest.fixture(autouse=True)
    def _mock_hf_hub(self):
        """Mock huggingface_hub which may not be installed in test env."""
        mock_hub = MagicMock()
        mock_scan = MagicMock()
        mock_hub.scan_cache_dir = mock_scan
        with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
            yield mock_scan

    def test_no_matching_repos(self, _mock_hf_hub):
        mock_cache = MagicMock()
        mock_cache.repos = []
        _mock_hf_hub.return_value = mock_cache
        result = clear_mflux_cache()
        assert result == []

    def test_removes_flux_repos(self, _mock_hf_hub):
        repo = MagicMock()
        repo.repo_id = "black-forest-labs/FLUX.1-dev"
        revision = MagicMock()
        revision.commit_hash = "abc123"
        repo.revisions = [revision]
        mock_cache = MagicMock()
        mock_cache.repos = [repo]
        _mock_hf_hub.return_value = mock_cache
        mock_cache.delete_revisions.return_value = mock_cache

        result = clear_mflux_cache()
        assert len(result) == 1
        assert "FLUX.1-dev" in result[0]

    def test_removes_specific_model(self, _mock_hf_hub):
        import sys

        saved = {}
        keys = (
            "mflux",
            "mflux.models",
            "mflux.models.common",
            "mflux.models.common.config",
            "mflux.models.common.config.model_config",
        )
        for key in keys:
            saved[key] = sys.modules.get(key)
            sys.modules[key] = MagicMock()
        mock_config_cls = sys.modules[keys[-1]].ModelConfig = MagicMock()
        mock_config_cls.from_name.return_value.model_name = "black-forest-labs/FLUX.1-dev"

        repo = MagicMock()
        repo.repo_id = "black-forest-labs/FLUX.1-dev"
        revision = MagicMock()
        revision.commit_hash = "abc123"
        repo.revisions = [revision]

        other = MagicMock()
        other.repo_id = "other/model"
        other.revisions = []

        mock_cache = MagicMock()
        mock_cache.repos = [repo, other]
        _mock_hf_hub.return_value = mock_cache
        mock_cache.delete_revisions.return_value = mock_cache

        try:
            result = clear_mflux_cache("dev")
            assert len(result) == 1
            assert "FLUX.1-dev" in result[0]
        finally:
            for key in keys:
                if saved[key] is not None:
                    sys.modules[key] = saved[key]
                else:
                    sys.modules.pop(key, None)


class TestValidateSavedModel:
    def test_not_a_directory(self, tmp_path):
        valid, meta = validate_saved_model(str(tmp_path / "nonexistent"))
        assert valid is False
        assert meta is None

    def test_empty_directory(self, tmp_path):
        valid, meta = validate_saved_model(str(tmp_path))
        assert valid is False
        assert meta is None

    def test_with_safetensors(self, tmp_path):
        sub = tmp_path / "diffusion_pytorch_model-00001-of-00002"
        sub.mkdir(parents=True)
        sf = sub / "model.safetensors"
        sf.write_text("not real safetensors")

        mock_safe = MagicMock()
        mock_f = MagicMock()
        mock_f.metadata.return_value = {"quantization_level": "4", "mflux_version": "0.1.0"}
        mock_safe.safe_open.return_value.__enter__.return_value = mock_f
        mock_safe.SafetensorError = Exception
        with patch.dict("sys.modules", {"safetensors": mock_safe}):
            valid, meta = validate_saved_model(str(tmp_path))
            assert valid is True
            assert meta["quantization_level"] == 4
            assert meta["mflux_version"] == "0.1.0"

    def test_safetensors_not_installed(self, tmp_path):
        with patch.dict("sys.modules", {"safetensors": None}):
            valid, meta = validate_saved_model(str(tmp_path))
            assert valid is False
            assert meta is None


class TestSaveMfluxModel:
    @pytest.fixture(autouse=True)
    def _mock_mflux(self):
        import sys

        saved = {}
        keys = (
            "mflux",
            "mflux.models",
            "mflux.models.common",
            "mflux.models.common.config",
            "mflux.models.common.config.model_config",
        )
        for key in keys:
            saved[key] = sys.modules.get(key)
            sys.modules[key] = MagicMock()
        self.mock_config_cls = sys.modules[keys[-1]].ModelConfig = MagicMock()
        yield
        for key in keys:
            if saved[key] is not None:
                sys.modules[key] = saved[key]
            else:
                sys.modules.pop(key, None)

    @patch("eyegen._model_ops._apply_hf_cache")
    @patch("eyegen._model_ops._resolve_mflux_class")
    def test_save_minimal(self, mock_resolve, mock_hf_cache, tmp_path):
        mock_cfg = MagicMock()
        self.mock_config_cls.from_name.return_value = mock_cfg
        mock_cls = MagicMock()
        mock_resolve.return_value = mock_cls
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_model.save_model.return_value = None

        out_dir = tmp_path / "saved"
        out_dir.mkdir()
        (out_dir / "some_file.bin").write_text("data")

        result = save_mflux_model("dev", 4, str(out_dir))
        assert Path(result) == out_dir.resolve()
        mock_model.save_model.assert_called_once_with(str(out_dir))
