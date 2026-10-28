"""How a model that is on this machine reaches the listing and the GUI.

A path is not a name: the listing shows what is there, the GUI offers the
path rather than a name a catalogue could answer to first, and a model is
shown once.
"""

import pytest

from eyegen.artifact_scan import scan_root
from eyegen.config import Backend


@pytest.fixture(autouse=True)
def _isolated_roots(tmp_path, monkeypatch):
    """Point every scanned root at a fresh, empty directory.

    The listing walks the models root, the HuggingFace hub caches, and the
    OllamaDiffuser model directory: a test must not see the machine's own
    models, or what a previous test left behind.
    """
    for env in (
        "EYEGEN_MODELS_DIR",
        "EYEGEN_HF_HUB_DIRS",
        "EYEGEN_OLLAMA_MODELS_DIR",
    ):
        monkeypatch.setenv(env, str(tmp_path / env))
    yield


def _gguf(directory, name="flux1-dev-Q4_K_M.gguf", size=8):
    path = directory / name
    path.write_bytes(b"x" * size)
    return path


def _mixin():
    """The GUI mixin, where the GUI's dependency can be loaded."""
    try:
        from eyegen.gui.main_window_model_dropdown import MainWindowModelDropdownMixin
    except ImportError:
        pytest.skip("PySide6 not available")
    return MainWindowModelDropdownMixin


def test_the_scan_finds_the_models_under_the_root(tmp_path):
    """A directory is named after what it holds, and looked at rather than named."""
    root = tmp_path / "models"
    (root / "sdcli" / "klein").mkdir(parents=True)
    _gguf(root / "sdcli" / "klein", "flux-3-image-klein-UD-Q4_K_XL.gguf")
    bundle = root / "coreml" / "sd-2-1-base"
    bundle.mkdir(parents=True)
    (bundle / "model.mlmodelc").mkdir()
    saved = root / "saved-mflux" / "dev-4bit" / "unet"
    saved.mkdir(parents=True)
    (saved / "diffusion.safetensors").write_bytes(b"x")

    found = {artifact.name: artifact.backend for artifact in scan_root(root)}

    assert found == {
        "klein": Backend.SDCLI,
        "sd-2-1-base": Backend.COREML,
        "dev-4bit": Backend.MFLUX,
    }


def test_the_scan_says_what_it_found(tmp_path):
    """What was decided is said, so that a role can be corrected."""
    root = tmp_path / "models"
    directory = root / "sdcli" / "klein"
    directory.mkdir(parents=True)
    _gguf(directory, "transformer.gguf", size=64)
    (directory / "ae.safetensors").write_bytes(b"x")

    found = scan_root(root)

    assert len(found) == 1
    assert "vae → ae.safetensors" in " ".join(found[0].notes)


def test_a_saved_mflux_directory_needs_safetensors_under_a_component(tmp_path):
    """A directory with the name is not a model; the weights make it one."""
    root = tmp_path / "models"
    directory = root / "saved-mflux" / "dev-4bit"
    directory.mkdir(parents=True)
    (directory / "README.txt").write_text("not weights")

    assert scan_root(root) == []


def test_the_gui_offers_a_local_model_by_path(tmp_path, monkeypatch):
    """A name would have to be resolved back to the directory it came from."""
    mixin = _mixin()
    monkeypatch.setenv("EYEGEN_MODELS_DIR", str(tmp_path))
    directory = tmp_path / "sdcli" / "klein"
    directory.mkdir(parents=True)
    _gguf(directory, "transformer.gguf", size=64)
    (directory / "ae.safetensors").write_bytes(b"x")

    models = mixin()._get_local_models()

    assert ("klein", str(directory)) in models


def test_the_gui_leaves_a_saved_mflux_model_to_its_own_list(tmp_path, monkeypatch):
    """A saved model is offered by name, and a path cannot name it."""
    mixin_type = _mixin()
    monkeypatch.setenv("EYEGEN_MODELS_DIR", str(tmp_path))
    saved = tmp_path / "saved-mflux" / "dev-4bit" / "unet"
    saved.mkdir(parents=True)
    (saved / "diffusion.safetensors").write_bytes(b"x")
    mixin = mixin_type()

    assert mixin._get_local_models() == []
    assert ("dev-4bit", "dev-4bit") in mixin._get_saved_mflux_models()


def test_auto_offers_a_name_once_and_prefers_the_path(tmp_path, monkeypatch):
    """The local entry is the one that can be run without resolving a name."""
    mixin = _mixin()()
    monkeypatch.setattr(mixin, "_get_local_models", lambda: [("klein", "/models/klein")])
    monkeypatch.setattr(mixin, "_get_own_models", lambda: [("klein", "klein")])
    monkeypatch.setattr(mixin, "_get_sdcli_models", lambda: [])
    monkeypatch.setattr(mixin, "_get_coreml_models", lambda: [])
    monkeypatch.setattr(mixin, "_get_saved_mflux_models", lambda: [])

    assert mixin._get_auto_models() == [("klein", "/models/klein")]


def test_the_listing_shows_what_is_on_this_machine(tmp_path, monkeypatch):
    """A model that is here, and that no catalogue describes, is listed."""
    from typer.testing import CliRunner

    from eyegen.cli.app import app

    monkeypatch.setenv("EYEGEN_MODELS_DIR", str(tmp_path))
    directory = tmp_path / "sdcli" / "klein"
    directory.mkdir(parents=True)
    _gguf(directory, "transformer.gguf", size=64)
    (directory / "ae.safetensors").write_bytes(b"x")

    result = CliRunner().invoke(app, ["list-models"])

    assert result.exit_code == 0, result.output
    assert "On this machine" in result.output
    assert "klein" in result.output


def test_a_catalogue_that_cannot_be_read_still_shows_what_is_here(tmp_path, monkeypatch):
    """A broken file costs its own section, not the models on the machine."""
    from typer.testing import CliRunner

    from eyegen import model_catalogue
    from eyegen.cli.app import app

    models_root = tmp_path / "models-root"
    monkeypatch.setenv("EYEGEN_MODELS_DIR", str(models_root))
    directory = models_root / "sdcli" / "klein"
    directory.mkdir(parents=True)
    _gguf(directory, "transformer.gguf", size=64)
    (directory / "ae.safetensors").write_bytes(b"x")
    model_catalogue.catalogue_path().write_text("{not json")

    result = CliRunner().invoke(app, ["list-models"])

    assert result.exit_code == 0, result.output
    assert "Could not read the model catalogue" in result.output
    assert "On this machine" in result.output
    assert "klein" in result.output


def test_the_scan_finds_a_model_in_a_huggingface_cache_under_the_root(tmp_path, monkeypatch):
    """A cache under the root holds the model too: what is there says what it is."""
    monkeypatch.setenv("EYEGEN_MODELS_DIR", str(tmp_path))
    cache = tmp_path / "models--abenzerps--Qwen-Image-2.1-Uncensored-Quantized"
    blob = cache / "blobs" / "e79c8a009f2ecbdb6c70fd663d9aea9ee304a0d91f347e4169a756b8ad141b41"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"x" * 8)
    snapshot = cache / "snapshots" / "7959e658915f52274897fb573ffeda49de8dc16b"
    snapshot.mkdir(parents=True)
    (snapshot / "qwen-image-2.1-UC-Q4_K_M.gguf").symlink_to(blob)

    found = {artifact.name: artifact.backend for artifact in scan_root(tmp_path)}

    assert found == {"qwen-image-2.1-UC-Q4_K_M": Backend.SDCLI}


def test_the_gui_offers_a_local_model_under_its_own_backend(tmp_path, monkeypatch):
    """A model that is here is offered by path whichever backend runs it."""
    mixin = _mixin()
    monkeypatch.setenv("EYEGEN_MODELS_DIR", str(tmp_path))
    directory = tmp_path / "sdcli" / "klein"
    directory.mkdir(parents=True)
    _gguf(directory, "transformer.gguf", size=64)
    (directory / "ae.safetensors").write_bytes(b"x")

    models = mixin()._get_models_for_backend(Backend.SDCLI)

    assert ("klein", str(directory)) in models


def _hub_cache(root, repo_id, files):
    """A HuggingFace hub cache holding one snapshot of *repo_id*."""
    repo = root / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo / "snapshots" / "abcdef123456"
    snapshot.mkdir(parents=True)
    for name, size in files.items():
        (snapshot / name).write_bytes(b"x" * size)
    return repo


def test_the_scan_finds_a_model_in_a_huggingface_hub(tmp_path):
    """A hub cache is scanned as the root it is, named after the repository."""
    _hub_cache(
        tmp_path / "hub",
        "unsloth/flux-3-image-klein-GGUF",
        {
            "flux-3-image-klein-UD-Q4_K_XL.gguf": 16,
            "Qwen3VL-4B-Instruct-Q4_K_M.gguf": 8,
            "ae.safetensors": 4,
        },
    )

    from eyegen.artifact_scan import scan_hf_hub

    found = scan_hf_hub(tmp_path / "hub")

    assert len(found) == 1
    assert found[0].name == "unsloth__flux-3-image-klein-GGUF"
    assert found[0].backend == Backend.SDCLI
    assert set(found[0].spec.sources) == {"diffusion_model", "llm", "vae"}


def test_the_scan_finds_a_model_ollamadiffuser_keeps(tmp_path):
    """The package's model directory is a root of its own."""
    from eyegen.artifact_scan import scan_ollama_models

    directory = tmp_path / "ollama" / "flux1-dev-Q4_K_M"
    directory.mkdir(parents=True)
    _gguf(directory, "flux1-dev-Q4_K_M.gguf", size=16)
    (directory / "clip_l.safetensors").write_bytes(b"x")

    found = scan_ollama_models(tmp_path / "ollama")

    assert len(found) == 1
    assert found[0].name == "flux1-dev-Q4_K_M"
    assert found[0].backend == Backend.OLLAMA


def test_the_gui_offers_models_from_the_hub_and_ollama_dirs(tmp_path, monkeypatch):
    """The dropdown shows what the other roots hold, by path."""
    mixin = _mixin()
    monkeypatch.setenv("EYEGEN_HF_HUB_DIRS", str(tmp_path / "hub"))
    monkeypatch.setenv("EYEGEN_OLLAMA_MODELS_DIR", str(tmp_path / "ollama"))
    _hub_cache(
        tmp_path / "hub",
        "unsloth/flux-3-image-klein-GGUF",
        {"flux-3-image-klein-UD-Q4_K_XL.gguf": 16, "ae.safetensors": 4},
    )
    ollama_dir = tmp_path / "ollama" / "flux1-dev-Q4_K_M"
    ollama_dir.mkdir(parents=True)
    _gguf(ollama_dir, "flux1-dev-Q4_K_M.gguf", size=16)
    (ollama_dir / "clip_l.safetensors").write_bytes(b"x")

    models = {display: value for display, value in mixin()._get_local_models()}

    assert models.get("unsloth__flux-3-image-klein-GGUF") is not None
    assert models.get("flux1-dev-Q4_K_M") is not None


def test_the_cli_listing_shows_what_the_other_roots_hold(tmp_path, monkeypatch):
    """list-models reports a model the hub or the package holds."""
    from typer.testing import CliRunner

    from eyegen.cli.app import app

    monkeypatch.setenv("EYEGEN_HF_HUB_DIRS", str(tmp_path / "hub"))
    _hub_cache(
        tmp_path / "hub",
        "unsloth/flux-3-image-klein-GGUF",
        {"flux-3-image-klein-UD-Q4_K_XL.gguf": 16, "ae.safetensors": 4},
    )

    result = CliRunner().invoke(app, ["list-models"])

    assert result.exit_code == 0, result.output
    assert "On this machine" in result.output
    assert "unsloth__flux-3-image-klein-GGUF" in result.output
