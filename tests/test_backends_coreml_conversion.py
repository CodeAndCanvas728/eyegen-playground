"""Converting a diffusers snapshot for the CoreML backend.

The conversion is the only way to run a diffusers model, and it must happen
once: the bundle it writes is found and reused by the next run. The sidecar
and Apple's converter are not run here; the conversion itself is replaced.
"""

import json
from unittest import mock

import coreml_layouts as layouts
import pytest

from eyegen.config import EyeGenConfig


def _diffusers_snapshot(directory):
    """A snapshot with no small files of its own: it must be converted to run."""
    (directory / "unet").mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    return directory


def _fake_conversion(model_id, output, **_kwargs):
    layouts.converted(output, model_id)
    return True


class TestCoreMLConversion:
    """A diffusers model is converted once, where it can be found again."""

    @mock.patch("eyegen.backends.coreml.pipeline.validate_coreml_install")
    def test_a_diffusers_model_is_converted_once(self, mock_validate, tmp_path, monkeypatch):
        mock_validate.return_value = mock.Mock(installed=True)
        monkeypatch.setenv("COREML_MODELS_DIR", str(tmp_path / "coreml"))
        snapshot = _diffusers_snapshot(tmp_path / "my-model")

        from eyegen.backends.coreml.pipeline import CoreMLWrapper

        with mock.patch(
            "eyegen.backends.coreml.conversion.convert_to_coreml",
            side_effect=_fake_conversion,
        ) as convert:
            wrapper = CoreMLWrapper(EyeGenConfig(coreml_model_path=str(snapshot)))

        output = tmp_path / "coreml" / "my-model"
        assert wrapper.model_path == output
        meta = json.loads((output / "model_index.json").read_text())
        assert meta["model_version"] == str(snapshot)
        assert convert.call_count == 1
        assert convert.call_args.args == (str(snapshot), output)

        with mock.patch("eyegen.backends.coreml.conversion.convert_to_coreml") as again:
            second = CoreMLWrapper(EyeGenConfig(coreml_model_path=str(snapshot)))

        assert second.model_path == output
        again.assert_not_called()

    @mock.patch("eyegen.backends.coreml.pipeline.validate_coreml_install")
    def test_a_snapshot_under_the_coreml_root_is_not_its_own_output(
        self, mock_validate, tmp_path, monkeypatch
    ):
        mock_validate.return_value = mock.Mock(installed=True)
        monkeypatch.setenv("COREML_MODELS_DIR", str(tmp_path / "coreml"))
        snapshot = _diffusers_snapshot(tmp_path / "coreml" / "my-model")

        from eyegen.backends.coreml.pipeline import CoreMLWrapper

        with mock.patch(
            "eyegen.backends.coreml.conversion.convert_to_coreml",
            side_effect=_fake_conversion,
        ):
            wrapper = CoreMLWrapper(EyeGenConfig(coreml_model_path=str(snapshot)))

        assert wrapper.model_path == tmp_path / "coreml" / "my-model-2"

    @mock.patch("eyegen.backends.coreml.pipeline.validate_coreml_install")
    def test_a_bundle_from_somewhere_else_keeps_its_name(
        self, mock_validate, tmp_path, monkeypatch
    ):
        mock_validate.return_value = mock.Mock(installed=True)
        monkeypatch.setenv("COREML_MODELS_DIR", str(tmp_path / "coreml"))
        taken = tmp_path / "coreml" / "my-model"
        taken.mkdir(parents=True)
        (taken / "unet.mlpackage").mkdir()
        (taken / "model_index.json").write_text(
            json.dumps({"model_version": "apple/coreml-stable-diffusion-v1-5"})
        )
        snapshot = _diffusers_snapshot(tmp_path / "elsewhere" / "my-model")

        from eyegen.backends.coreml.pipeline import CoreMLWrapper

        with mock.patch(
            "eyegen.backends.coreml.conversion.convert_to_coreml",
            side_effect=_fake_conversion,
        ):
            wrapper = CoreMLWrapper(EyeGenConfig(coreml_model_path=str(snapshot)))

        assert wrapper.model_path == tmp_path / "coreml" / "my-model-2"
        assert (taken / "model_index.json").read_text().count("apple/") == 1

    @mock.patch("eyegen.backends.coreml.pipeline.validate_coreml_install")
    def test_a_conversion_that_fails_says_how_to_run_it(self, mock_validate, tmp_path, monkeypatch):
        mock_validate.return_value = mock.Mock(installed=True)
        monkeypatch.setenv("COREML_MODELS_DIR", str(tmp_path / "coreml"))
        snapshot = _diffusers_snapshot(tmp_path / "my-model")

        from eyegen.backends.coreml.pipeline import CoreMLWrapper

        with (
            mock.patch("eyegen.backends.coreml.conversion.convert_to_coreml", return_value=False),
            pytest.raises(RuntimeError, match="convert-coreml"),
        ):
            CoreMLWrapper(EyeGenConfig(coreml_model_path=str(snapshot)))

    @mock.patch("eyegen.backends.coreml.pipeline.validate_coreml_install")
    def test_an_old_bundle_is_given_its_assets_without_converting_again(
        self, mock_validate, tmp_path, monkeypatch
    ):
        mock_validate.return_value = mock.Mock(installed=True)
        monkeypatch.setenv("COREML_MODELS_DIR", str(tmp_path / "coreml"))
        snapshot = layouts.snapshot(tmp_path / "my-model")
        output = tmp_path / "coreml" / "my-model"
        for name in layouts.MODEL_NAMES:
            (output / f"{name}.mlpackage").mkdir(parents=True, exist_ok=True)
        (output / "model_index.json").write_text(json.dumps({"model_version": str(snapshot)}))

        from eyegen.backends.coreml.pipeline import CoreMLWrapper

        with mock.patch("eyegen.backends.coreml.conversion.convert_to_coreml") as convert:
            wrapper = CoreMLWrapper(EyeGenConfig(coreml_model_path=str(snapshot)))

        assert wrapper.model_path == output
        assert (output / "pipeline_assets" / "tokenizer" / "vocab.json").is_file()
        assert json.loads((output / "model_index.json").read_text())["resolution"] == {
            "height": 512,
            "width": 512,
        }
        convert.assert_not_called()

    @mock.patch("eyegen.backends.coreml.pipeline.validate_coreml_install")
    def test_an_old_bundle_gains_the_size_its_models_run_at(
        self, mock_validate, tmp_path, monkeypatch
    ):
        mock_validate.return_value = mock.Mock(installed=True)
        monkeypatch.setenv("COREML_MODELS_DIR", str(tmp_path / "coreml"))
        snapshot = layouts.snapshot(tmp_path / "my-model")
        output = layouts.converted(tmp_path / "coreml" / "my-model", str(snapshot))

        from eyegen.backends.coreml.pipeline import CoreMLWrapper

        with mock.patch("eyegen.backends.coreml.conversion.convert_to_coreml") as convert:
            CoreMLWrapper(EyeGenConfig(coreml_model_path=str(snapshot)))

        assert json.loads((output / "model_index.json").read_text())["resolution"] == {
            "height": 512,
            "width": 512,
        }
        convert.assert_not_called()

    @mock.patch("eyegen.backends.coreml.pipeline.validate_coreml_install")
    def test_a_bundle_that_lost_a_model_is_finished_with_the_recorded_settings(
        self, mock_validate, tmp_path, monkeypatch
    ):
        mock_validate.return_value = mock.Mock(installed=True)
        monkeypatch.setenv("COREML_MODELS_DIR", str(tmp_path / "coreml"))
        snapshot = layouts.snapshot(tmp_path / "my-model")
        output = tmp_path / "coreml" / "my-model"
        (output / "unet.mlpackage").mkdir(parents=True)
        (output / "model_index.json").write_text(
            json.dumps(
                {
                    "model_version": str(snapshot),
                    "attention_implementation": "SPLIT_EINSUM_V2",
                    "quantize_nbits": 6,
                }
            )
        )

        from eyegen.backends.coreml.pipeline import CoreMLWrapper

        with mock.patch(
            "eyegen.backends.coreml.conversion.convert_to_coreml",
            side_effect=_fake_conversion,
        ) as convert:
            CoreMLWrapper(EyeGenConfig(coreml_model_path=str(snapshot)))

        assert convert.call_args.kwargs["attention_implementation"] == "SPLIT_EINSUM_V2"
        assert convert.call_args.kwargs["quantize_nbits"] == 6

    @mock.patch("eyegen.backends.coreml.pipeline.validate_coreml_install")
    def test_a_bundle_takes_the_size_its_conversion_recorded(self, mock_validate, tmp_path):
        mock_validate.return_value = mock.Mock(installed=True)
        bundle = layouts.converted(tmp_path / "bundle", "owner/model")
        meta = json.loads((bundle / "model_index.json").read_text())
        meta["resolution"] = {"height": 512, "width": 512}
        (bundle / "model_index.json").write_text(json.dumps(meta))

        from eyegen.backends.coreml.pipeline import CoreMLWrapper

        wrapper = CoreMLWrapper(EyeGenConfig(coreml_model_path=str(bundle)))

        with pytest.raises(ValueError, match="512x512 images only"):
            wrapper.generate_image(
                prompt="a fox", cfg_weight=7.5, num_steps=10, width=1024, height=1024
            )
