"""Tests for the stable-diffusion.cpp backend's model handling.

These cover the paths the first round of review found missing: the pull,
the GUI's route into it, and the class MFLUX uses for FLUX.3.
"""

from unittest.mock import patch

import pytest

from eyegen.backends.sdcli.constants import SDCLI_MODELS
from eyegen.backends.sdcli.models import list_sdcli_models, pull_sdcli_model
from eyegen.config import Backend

MODEL = "flux-3-image-klein-gguf"
MODEL_FILES = (
    "flux-3-image-klein-UD-Q4_K_XL.gguf",
    "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
    "ae.safetensors",
)


def test_the_catalogue_holds_the_three_files():
    spec = SDCLI_MODELS[MODEL]
    assert set(spec["sources"]) == {"diffusion_model", "llm", "vae"}
    names = {source["file"] for source in spec["sources"].values()}
    assert names == set(MODEL_FILES)


def test_pulling_an_unknown_model_says_so():
    assert pull_sdcli_model("not-a-model") is None


def test_pulling_downloads_the_files(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    monkeypatch.setenv("SDCLI_MODEL_DIR", str(tmp_path))
    downloaded = []

    def fake_download(repo_id, filename, local_dir):
        downloaded.append((repo_id, filename))
        path = tmp_path / MODEL / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return str(path)

    with patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
        directory = pull_sdcli_model(MODEL)

    assert directory == tmp_path / MODEL
    assert len(downloaded) == 3
    for name in MODEL_FILES:
        assert (tmp_path / MODEL / name).is_file()


def test_pulling_skips_a_file_that_is_already_there(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    monkeypatch.setenv("SDCLI_MODEL_DIR", str(tmp_path))
    directory = tmp_path / MODEL
    directory.mkdir()
    for name in MODEL_FILES:
        (directory / name).write_bytes(b"x")

    with patch("huggingface_hub.hf_hub_download") as download:
        pulled = pull_sdcli_model(MODEL)

    assert pulled == directory
    download.assert_not_called()


def test_the_listing_says_whether_the_files_are_there(tmp_path, monkeypatch):
    monkeypatch.setenv("SDCLI_MODEL_DIR", str(tmp_path))
    directory = tmp_path / MODEL
    directory.mkdir()
    (directory / MODEL_FILES[0]).write_bytes(b"x")

    models = list_sdcli_models()
    entry = next(m for m in models if m["name"] == MODEL)
    assert not entry["installed"]
    assert entry["present"] == [MODEL_FILES[0]]


def test_the_users_own_models_are_in_the_listing(tmp_path, monkeypatch):
    """A model the user described is one this backend can run."""
    from eyegen.model_catalogue import upsert_spec
    from eyegen.model_specs import ModelSource, ModelSpec

    monkeypatch.setenv("SDCLI_MODEL_DIR", str(tmp_path))
    upsert_spec(
        ModelSpec(
            name="mine",
            backend=Backend.SDCLI,
            sources={"diffusion_model": ModelSource(repo="owner/repo", file="mine.gguf")},
        )
    )

    models = list_sdcli_models()
    entry = next(m for m in models if m["name"] == "mine")
    assert entry["files"] == ["mine.gguf"]
    assert not entry["installed"]


def test_a_local_model_is_installed_where_its_files_are(tmp_path, monkeypatch):
    """A file already on the machine is not looked for in a model directory."""
    from eyegen.model_catalogue import upsert_spec
    from eyegen.model_specs import ModelSource, ModelSpec

    monkeypatch.setenv("SDCLI_MODEL_DIR", str(tmp_path / "downloads"))
    transformer = tmp_path / "transformer.gguf"
    vae = tmp_path / "ae.safetensors"
    transformer.write_bytes(b"x")
    vae.write_bytes(b"x")
    upsert_spec(
        ModelSpec(
            name="mine",
            backend=Backend.SDCLI,
            sources={
                "diffusion_model": ModelSource(path=str(transformer)),
                "vae": ModelSource(path=str(vae)),
            },
        )
    )

    models = list_sdcli_models()
    entry = next(m for m in models if m["name"] == "mine")

    assert entry["present"] == ["transformer.gguf", "ae.safetensors"]
    assert entry["installed"]


def test_the_gui_pulls_through_the_stable_diffusion_puller(tmp_path):
    """The GUI must not send a stable-diffusion.cpp model to OllamaDiffuser."""
    try:
        from eyegen.gui.workers import PullWorker
    except ImportError:
        pytest.skip("PySide6 not available")

    worker = PullWorker(MODEL)
    with (
        patch("eyegen.backends.sdcli.models.pull_sdcli_model", return_value=tmp_path) as sdcli,
        patch("eyegen._model_ops.pull_model") as ollama,
    ):
        worker.run()

    sdcli.assert_called_once()
    ollama.assert_not_called()


def test_the_gui_still_pulls_an_ollama_model_through_ollama():
    try:
        from eyegen.gui.workers import PullWorker
    except ImportError:
        pytest.skip("PySide6 not available")

    worker = PullWorker("flux.1-dev-gguf-q4ks")
    with (
        patch("eyegen.backends.sdcli.models.pull_sdcli_model") as sdcli,
        patch("eyegen._model_ops.pull_model", return_value=True) as ollama,
    ):
        worker.run()

    ollama.assert_called_once()
    sdcli.assert_not_called()


def test_mflux_resolves_the_flux3_class():
    """The import path is checked here rather than at first use.

    Another test module mocks mflux when it is imported, and pytest
    imports every test module before it runs any of them, so the real
    package has to be put back for the length of this test.
    """
    import importlib
    import sys

    mocked = {key: value for key, value in sys.modules.items() if _is_mocked(key)}
    for key in mocked:
        del sys.modules[key]
    try:
        try:
            model_config = importlib.import_module("mflux.models.common.config.model_config")
            flux3 = importlib.import_module("mflux.models.flux3.variants.txt2img.flux3_klein")
        except ImportError as exc:
            pytest.skip(f"mflux is not importable here: {exc}")

        from eyegen._mflux import _resolve_mflux_class

        config = model_config.ModelConfig.from_name(model_name="flux3-image-klein")
        resolved = _resolve_mflux_class(config)
        assert resolved is flux3.Flux3Klein
    finally:
        # The mocks go back, skipped or not: the tests that come after
        # this one expect them.
        for key in [k for k in sys.modules if _is_mocked(k)]:
            del sys.modules[key]
        sys.modules.update(mocked)


def _is_mocked(key: str) -> bool:
    """True for the packages another test module mocks when it is imported."""
    return key.split(".")[0] in ("mflux", "mlx")


def test_a_lone_gguf_from_a_huggingface_cache_is_completed_from_its_repository(
    tmp_path, monkeypatch
):
    """A transformer alone is not the whole model; its repository ships the rest."""
    from eyegen.backends.sdcli.models import complete_local_spec
    from eyegen.model_specs import ModelSource, ModelSpec

    repo_dir = tmp_path / "models--abenzerps--Qwen-Image-2.1-Uncensored-Quantized"
    snapshot = repo_dir / "snapshots" / "7959e658915f52274897fb573ffeda49de8dc16b"
    snapshot.mkdir(parents=True)
    gguf = snapshot / "qwen-image-2.1-UC-Q4_K_M.gguf"
    gguf.write_bytes(b"x" * 8)

    spec = ModelSpec(
        name="qwen-image-2.1-UC-Q4_K_M",
        backend=Backend.SDCLI,
        sources={"diffusion_model": ModelSource(path=str(gguf))},
    )

    monkeypatch.setattr(
        "eyegen.backends.sdcli.models._repo_files",
        lambda repo: [
            "README.md",
            "qwen-image-2.1-UC-Q4_K_M.gguf",
            "text_encoders/qwen3vl_8b_bf16.safetensors",
            "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
            "vae/qwen_image_2.1_vae_bf16.safetensors",
        ],
    )

    completed = complete_local_spec(spec)

    assert completed.sources["vae"].repo == "abenzerps/Qwen-Image-2.1-Uncensored-Quantized"
    assert completed.sources["vae"].file == "vae/qwen_image_2.1_vae_bf16.safetensors"
    # The quantised text encoder, not the 17 GB BF16 one.
    assert completed.sources["llm"].file == "text_encoders/qwen3vl_8b_int8_convrot.safetensors"


def test_a_gguf_outside_a_cache_is_left_alone(tmp_path):
    """Without a repository, nothing can be fetched: the spec is what it is."""
    from eyegen.backends.sdcli.models import complete_local_spec
    from eyegen.model_specs import ModelSource, ModelSpec

    directory = tmp_path / "elsewhere"
    directory.mkdir()
    gguf = directory / "transformer.gguf"
    gguf.write_bytes(b"x" * 8)
    spec = ModelSpec(
        name="local",
        backend=Backend.SDCLI,
        sources={"diffusion_model": ModelSource(path=str(gguf))},
    )

    assert complete_local_spec(spec) is spec


def test_a_text_encoder_in_full_precision_is_not_the_one_picked():
    """A BF16 encoder is tens of gigabytes; the quantized one runs on a laptop."""
    from eyegen.backends.sdcli.models import _pick_by_hints

    files = [
        "text_encoders/qwen3vl_8b_bf16.safetensors",
        "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
    ]
    hints = ("t5xxl", "clip_l", "clip_g", "qwen3vl", "llm", "instruct", "text_encoder", "/llm")

    assert _pick_by_hints(files, hints) == "text_encoders/qwen3vl_8b_int8_convrot.safetensors"


def test_a_gguf_transformer_beats_a_bf16_one():
    """A BF16 transformer is a fallback, not a first choice."""
    from eyegen.backends.sdcli.models import _pick_by_hints

    files = [
        "qwen-image-2.1-UC-BF16.gguf",
        "qwen-image-2.1-UC-Q4_K_M.gguf",
    ]
    hints = ("qwen", "diffusion", "transformer")

    assert _pick_by_hints(files, hints) == "qwen-image-2.1-UC-Q4_K_M.gguf"
