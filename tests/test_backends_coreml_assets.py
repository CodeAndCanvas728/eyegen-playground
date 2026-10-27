"""The files a CoreML bundle needs besides its models.

A bundle that runs must keep running: the tokenizer, scheduler and feature
extractor are staged and checked before they replace anything, and a bundle
that was converted from another model is refused the files of one it did not
come from.
"""

import json
from pathlib import Path
from unittest import mock

import coreml_layouts as layouts
import pytest

from eyegen.config import EyeGenConfig

_SIDECAR = "eyegen.backends.coreml.models._sidecar_python"
_HAS_COREML = "eyegen.backends.coreml.models._sidecar_has_coreml"
_SUBPROCESS = "eyegen.backends.runner.BaseSubprocessRunner._execute_subprocess"


class TestConversionAssets:
    """A conversion keeps the small diffusers files the runner needs."""

    @mock.patch(_SIDECAR)
    @mock.patch(_HAS_COREML)
    @mock.patch(_SUBPROCESS)
    def test_converting_saves_the_assets(self, mock_exec, mock_has_coreml, mock_sidecar, tmp_path):
        mock_sidecar.return_value = Path("/fake/python")
        mock_has_coreml.return_value = True
        mock_exec.return_value = (0, [], [])
        snapshot = layouts.snapshot(tmp_path / "my-model")
        output = tmp_path / "coreml" / "my-model"

        from eyegen.backends.coreml import convert_to_coreml

        assert convert_to_coreml(str(snapshot), output)
        assets = output / "pipeline_assets"
        assert (assets / "tokenizer" / "vocab.json").read_text() == "{}"
        assert (assets / "scheduler" / "scheduler_config.json").is_file()
        meta = json.loads((output / "model_index.json").read_text())
        assert meta["model_version"] == str(snapshot)
        assert meta["force_zeros_for_empty_prompt"] is True
        assert meta["resolution"] == {"height": 512, "width": 512}
        assert [path.name for path in output.iterdir() if path.name.startswith(".")] == []

    @mock.patch("eyegen.backends.coreml.pipeline.validate_coreml_install")
    @mock.patch(_SIDECAR)
    @mock.patch(_HAS_COREML)
    @mock.patch(_SUBPROCESS)
    def test_a_conversion_that_could_not_save_its_assets_is_not_redone(
        self, mock_exec, mock_has_coreml, mock_sidecar, mock_validate, tmp_path, monkeypatch
    ):
        mock_validate.return_value = mock.Mock(installed=True)
        mock_sidecar.return_value = Path("/fake/python")
        mock_has_coreml.return_value = True
        monkeypatch.setenv("COREML_MODELS_DIR", str(tmp_path / "coreml"))
        snapshot = layouts.snapshot(tmp_path / "my-model")
        output = tmp_path / "coreml" / "my-model"

        def convert_models(*_args, **_kwargs):
            for name in layouts.MODEL_NAMES:
                (output / f"{name}.mlpackage").mkdir(parents=True, exist_ok=True)
            return 0, [], []

        mock_exec.side_effect = convert_models

        from eyegen.backends.coreml import convert_to_coreml
        from eyegen.backends.coreml.pipeline import CoreMLWrapper

        with mock.patch(
            "eyegen.backends.coreml.assets._save_pipeline_assets", side_effect=OSError("offline")
        ):
            assert not convert_to_coreml(str(snapshot), output)

        recorded = json.loads((output / "model_index.json").read_text())
        assert recorded["model_version"] == str(snapshot)

        with mock.patch("eyegen.backends.coreml.conversion.convert_to_coreml") as convert:
            wrapper = CoreMLWrapper(EyeGenConfig(coreml_model_path=str(snapshot)))

        assert wrapper.model_path == output
        assert (output / "pipeline_assets" / "tokenizer" / "vocab.json").is_file()
        convert.assert_not_called()

    @mock.patch(_SIDECAR)
    @mock.patch(_HAS_COREML)
    @mock.patch(_SUBPROCESS)
    def test_downloads_the_assets_for_a_hub_model(
        self, mock_exec, mock_has_coreml, mock_sidecar, tmp_path
    ):
        pytest.importorskip("huggingface_hub")
        mock_sidecar.return_value = Path("/fake/python")
        mock_has_coreml.return_value = True
        mock_exec.return_value = (0, [], [])

        def fake_download(repo_id, allow_patterns, local_dir):
            assets = Path(local_dir)
            tokenizer = assets / "tokenizer"
            tokenizer.mkdir(parents=True)
            (tokenizer / "vocab.json").write_text("{}")
            (tokenizer / "merges.txt").write_text("")
            scheduler = assets / "scheduler"
            scheduler.mkdir()
            (scheduler / "scheduler_config.json").write_text("{}")
            (assets / "model_index.json").write_text(
                json.dumps({"force_zeros_for_empty_prompt": False})
            )
            return str(assets)

        output = tmp_path / "coreml" / "hub-model"

        from eyegen.backends.coreml import convert_to_coreml

        with mock.patch("huggingface_hub.snapshot_download") as mock_download:
            mock_download.side_effect = fake_download
            assert convert_to_coreml("owner/model", output)

        kwargs = mock_download.call_args.kwargs
        assert kwargs["repo_id"] == "owner/model"
        # Downloaded aside and moved in, so a failed download leaves the
        # bundle's own files alone.
        assert Path(kwargs["local_dir"]).parent == output
        assert "local_dir_use_symlinks" not in kwargs
        meta = json.loads((output / "model_index.json").read_text())
        assert meta["force_zeros_for_empty_prompt"] is False


class TestAddPipelineAssets:
    """Handing an already-converted bundle the files a run needs."""

    def test_a_failed_fetch_keeps_the_files_the_bundle_has(self, tmp_path):
        bundle = layouts.converted(tmp_path / "bundle", "owner/model")
        vocab = bundle / "pipeline_assets" / "tokenizer" / "vocab.json"
        vocab.write_text('{"old": true}')

        from eyegen.backends.coreml.assets import add_pipeline_assets

        with mock.patch(
            "eyegen.backends.coreml.assets._download_pipeline_assets",
            side_effect=OSError("offline"),
        ):
            assert not add_pipeline_assets("owner/model", bundle)

        assert vocab.read_text() == '{"old": true}'
        assert [path.name for path in bundle.iterdir() if path.name.startswith(".")] == []

    def test_a_bundle_converted_from_another_model_is_refused(self, tmp_path):
        bundle = layouts.converted(tmp_path / "bundle", "owner/one")

        from eyegen.backends.coreml.assets import add_pipeline_assets

        with pytest.raises(ValueError, match="owner/one"):
            add_pipeline_assets("owner/two", bundle)

        assert (bundle / "pipeline_assets" / "tokenizer" / "vocab.json").is_file()

    def test_a_source_missing_a_file_a_run_needs_is_not_called_finished(self, tmp_path):
        snapshot = layouts.snapshot(tmp_path / "my-model", merges=False)
        bundle = tmp_path / "bundle"
        bundle.mkdir()

        from eyegen.backends.coreml.assets import add_pipeline_assets

        assert not add_pipeline_assets(str(snapshot), bundle)
        assert not (bundle / "pipeline_assets").exists()
