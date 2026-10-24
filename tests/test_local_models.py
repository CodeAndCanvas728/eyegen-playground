"""How a model that is on this machine reaches the listing and the GUI.

A path is not a name: the listing shows what is there, the GUI offers the
path rather than a name a catalogue could answer to first, and a model is
shown once.
"""

import pytest

from eyegen.artifact_scan import scan_root
from eyegen.config import Backend


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
