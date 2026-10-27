"""Running a CoreML bundle without the diffusers model it came from."""

import sys
import types
from pathlib import Path
from unittest import mock

import pytest

from eyegen.backends.coreml import offline_runner
from eyegen.config import EyeGenConfig


def _touch(path, text="{}"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _package_bundle(directory):
    directory.mkdir(parents=True)
    for name in ("text_encoder", "unet", "vae_decoder"):
        (directory / f"Stable_Diffusion_version_test-model_{name}.mlpackage").mkdir()
    return directory


def _compiled_bundle(directory):
    directory.mkdir(parents=True)
    for name in ("TextEncoder", "Unet", "VAEDecoder"):
        (directory / f"{name}.mlmodelc").mkdir()
    return directory


def _assets(bundle):
    assets = bundle / "pipeline_assets"
    _touch(assets / "tokenizer" / "vocab.json")
    _touch(assets / "tokenizer" / "merges.txt", "")
    _touch(assets / "scheduler" / "scheduler_config.json")
    return assets


class TestComponentDiscovery:
    def test_finds_packages(self, tmp_path):
        bundle = _package_bundle(tmp_path / "bundle")
        components, sources = offline_runner.discover_components(bundle)
        assert sources == "packages"
        assert components["unet"].name.endswith("_unet.mlpackage")
        assert components["safety_checker"] is None

    def test_finds_compiled_models(self, tmp_path):
        bundle = _compiled_bundle(tmp_path / "bundle")
        components, sources = offline_runner.discover_components(bundle)
        assert sources == "compiled"
        assert components["vae_decoder"].name == "VAEDecoder.mlmodelc"

    def test_refuses_a_mixed_bundle(self, tmp_path):
        bundle = _package_bundle(tmp_path / "bundle")
        (bundle / "Unet.mlmodelc").mkdir()
        with pytest.raises(offline_runner.BundleError, match="both"):
            offline_runner.discover_components(bundle)

    def test_names_the_missing_models(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "Stable_Diffusion_version_test-model_text_encoder.mlpackage").mkdir()
        with pytest.raises(offline_runner.BundleError, match="unet, vae_decoder"):
            offline_runner.discover_components(bundle)


class TestAssets:
    def test_missing_assets_say_how_to_add_them(self, tmp_path):
        bundle = _package_bundle(tmp_path / "bundle")
        with pytest.raises(offline_runner.BundleError, match="convert-coreml"):
            offline_runner.pipeline_assets_dir(bundle)

    def test_finds_the_tokenizer_and_scheduler(self, tmp_path):
        bundle = _package_bundle(tmp_path / "bundle")
        _assets(bundle)
        assert offline_runner.pipeline_assets_dir(bundle) == bundle / "pipeline_assets"


class _FakeImage:
    def __init__(self, saved):
        self.saved = saved

    def save(self, path):
        self.saved["path"] = Path(path)


class _FakePipe:
    """Apple's pipeline, reduced to the shape and the call the runner makes."""

    height = 512
    width = 512

    def __init__(self, saved):
        self.saved = saved
        self.kwargs = {}
        self.drawn = None

    def __call__(self, **kwargs):
        import numpy as np

        # The real pipeline draws its latents here, from the global generator.
        self.drawn = np.random.rand()
        self.kwargs = kwargs
        return {"images": [_FakeImage(self.saved)]}


class TestRun:
    @staticmethod
    def _args(tmp_path, *extra):
        return offline_runner.parse_args(
            [
                "--prompt",
                "a fox",
                "-i",
                str(tmp_path / "bundle"),
                "-o",
                str(tmp_path / "out.png"),
                *extra,
            ]
        )

    @staticmethod
    def _pipe(monkeypatch):
        pipe = _FakePipe({})
        monkeypatch.setattr(offline_runner, "build_pipeline", lambda _args: pipe)
        return pipe

    def test_writes_the_exact_output_path(self, tmp_path, monkeypatch):
        pytest.importorskip("numpy")
        pipe = self._pipe(monkeypatch)
        out = tmp_path / "nested" / "out.png"
        args = offline_runner.parse_args(
            ["--prompt", "a fox", "-i", str(tmp_path / "bundle"), "-o", str(out)]
        )

        offline_runner.run(args)

        assert pipe.saved["path"] == out

    def test_generates_at_the_size_the_models_were_exported_at(self, tmp_path, monkeypatch):
        pytest.importorskip("numpy")
        pipe = self._pipe(monkeypatch)

        offline_runner.run(self._args(tmp_path))

        assert (pipe.kwargs["width"], pipe.kwargs["height"]) == (512, 512)

    def test_refuses_a_size_the_models_cannot_make(self, tmp_path, monkeypatch):
        pytest.importorskip("numpy")
        self._pipe(monkeypatch)

        with pytest.raises(offline_runner.BundleError, match=r"512x512"):
            offline_runner.run(self._args(tmp_path, "--width", "1024", "--height", "1024"))

    def test_the_seed_is_what_seeds_the_noise_before_generating(self, tmp_path, monkeypatch):
        np = pytest.importorskip("numpy")
        pipe = self._pipe(monkeypatch)

        offline_runner.run(self._args(tmp_path, "--seed", "7"))

        np.random.seed(7)
        assert pipe.drawn == np.random.rand()

    def test_refuses_a_seed_the_generator_cannot_take(self, tmp_path):
        pytest.importorskip("numpy")

        with pytest.raises(offline_runner.BundleError, match=r"0 to 2\*\*32-1"):
            offline_runner.run(self._args(tmp_path, "--seed", str(2**32)))


class TestBuildPipeline:
    def test_builds_the_pipeline_from_bundle_files(self, tmp_path, monkeypatch):
        bundle = _package_bundle(tmp_path / "bundle")
        assets = _assets(bundle)
        _touch(assets / "feature_extractor" / "preprocessor_config.json")
        _touch(
            assets / "scheduler" / "scheduler_config.json",
            '{"_class_name": "PNDMScheduler"}',
        )
        _touch(bundle / "model_index.json", '{"force_zeros_for_empty_prompt": true}')

        created = []

        class FakeCoreMLModel:
            def __init__(self, path, compute_unit, sources="packages"):
                self.path = Path(path)
                self.compute_unit = compute_unit
                self.sources = sources
                created.append(self)

        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, _path, **_kwargs):
                return cls()

        class FakeScheduler(FakeTokenizer):
            pass

        class FakePipeline:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        coreml = types.ModuleType("python_coreml_stable_diffusion")
        coreml.__path__ = []
        coreml_model = types.ModuleType("python_coreml_stable_diffusion.coreml_model")
        coreml_model.CoreMLModel = FakeCoreMLModel
        pipeline = types.ModuleType("python_coreml_stable_diffusion.pipeline")
        pipeline.CoreMLStableDiffusionPipeline = FakePipeline
        coreml.coreml_model = coreml_model
        coreml.pipeline = pipeline
        transformers = types.ModuleType("transformers")
        transformers.CLIPTokenizer = FakeTokenizer
        transformers.CLIPFeatureExtractor = FakeTokenizer
        diffusers = types.ModuleType("diffusers")
        diffusers.PNDMScheduler = FakeScheduler
        for name, module in {
            "python_coreml_stable_diffusion": coreml,
            "python_coreml_stable_diffusion.coreml_model": coreml_model,
            "python_coreml_stable_diffusion.pipeline": pipeline,
            "transformers": transformers,
            "diffusers": diffusers,
        }.items():
            monkeypatch.setitem(sys.modules, name, module)

        args = offline_runner.parse_args(
            ["--prompt", "a fox", "-i", str(bundle), "-o", str(tmp_path / "out.png")]
        )
        pipe = offline_runner.build_pipeline(args)

        assert isinstance(pipe, FakePipeline)
        assert pipe.kwargs["xl"] is False
        assert pipe.kwargs["force_zeros_for_empty_prompt"] is True
        assert pipe.kwargs["unet"].path.name.endswith("_unet.mlpackage")
        assert pipe.kwargs["safety_checker"] is None
        assert len(created) == 3


class TestCoreMLCommand:
    @mock.patch("eyegen.backends.coreml.pipeline.validate_coreml_install")
    def test_runs_the_offline_runner_at_the_requested_path(self, mock_validate, tmp_path):
        mock_validate.return_value = mock.Mock(installed=True)
        bundle = tmp_path / "bundle"
        bundle.mkdir()

        from eyegen.backends.coreml.pipeline import CoreMLWrapper

        wrapper = CoreMLWrapper(EyeGenConfig(coreml_model_path=str(bundle)))
        out = tmp_path / "out.png"
        sidecar = "eyegen.backends.coreml.pipeline._sidecar_python"
        with mock.patch(sidecar, return_value=Path("/fake/python")):
            cmd = wrapper._build_coreml_command("a fox", 7.5, 10, 1024, 768, 42, "", out)

        assert cmd[0] == "/fake/python"
        assert Path(cmd[1]).name == "offline_runner.py"
        assert cmd[cmd.index("-o") + 1] == str(out)
        assert cmd[cmd.index("--seed") + 1] == "42"
        assert cmd[cmd.index("--width") + 1] == "1024"
        assert cmd[cmd.index("--height") + 1] == "768"

    @mock.patch("eyegen.backends.coreml.pipeline.validate_coreml_install")
    @mock.patch("eyegen.backends.coreml.pipeline._sidecar_python")
    @mock.patch("eyegen.backends.coreml.pipeline.CoreMLWrapper._execute_subprocess")
    def test_a_random_seed_is_passed_to_the_run(
        self, mock_execute, mock_sidecar, mock_validate, tmp_path, monkeypatch
    ):
        mock_validate.return_value = mock.Mock(installed=True)
        mock_sidecar.return_value = Path("/fake/python")
        monkeypatch.setattr("eyegen.config.OUTPUT_DIR", tmp_path)
        bundle = tmp_path / "bundle"
        bundle.mkdir()

        def fake_execute(cmd, **_kwargs):
            seed = int(cmd[cmd.index("--seed") + 1])
            (tmp_path / f".tmp_coreml_{seed}.png").touch()
            return 0, [], []

        mock_execute.side_effect = fake_execute

        from eyegen.backends.coreml.pipeline import CoreMLWrapper

        wrapper = CoreMLWrapper(EyeGenConfig(coreml_model_path=str(bundle)))
        with mock.patch("eyegen.backends.coreml.pipeline.Image.open") as mock_img_open:
            mock_img_open.return_value.convert.return_value = mock.Mock()
            wrapper.generate_image(
                prompt="a fox",
                cfg_weight=7.5,
                num_steps=10,
                width=512,
                height=512,
                seed=None,
            )

        cmd = mock_execute.call_args.args[0]
        seed = int(cmd[cmd.index("--seed") + 1])
        assert 0 <= seed < 2**32
