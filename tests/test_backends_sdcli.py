"""Tests for the stable-diffusion.cpp backend."""

from unittest.mock import patch

import pytest

from eyegen.backends import detect_backend
from eyegen.backends.sdcli.constants import SDCLI_MODELS
from eyegen.backends.sdcli.install import (
    get_sdcli_model_dir,
    get_sdcli_timeout,
    missing_model_files,
    validate_sdcli_install,
)
from eyegen.backends.sdcli.pipeline import SdCliWrapper, resolve_settings
from eyegen.config import Backend

MODEL_FILES = (
    "flux-3-image-klein-UD-Q4_K_XL.gguf",
    "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
    "ae.safetensors",
)


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """A directory with the three model files in it."""
    for name in MODEL_FILES:
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setenv("SDCLI_MODEL_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def binary(tmp_path, monkeypatch):
    """An sd-cli binary in a directory of its own."""
    path = tmp_path / "sd-cli"
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    monkeypatch.setenv("SDCLI_BIN", str(path))
    return path


def test_detect_backend_flux3_gguf():
    assert detect_backend("flux-3-image-klein-gguf", Backend.AUTO) == Backend.SDCLI


def test_detect_backend_mflux_flux3():
    """The same weights through MFLUX have MFLUX's name."""
    assert detect_backend("flux3-image-klein", Backend.AUTO) == Backend.MFLUX


def test_detect_backend_mflux_flux3_model_name():
    assert detect_backend("black-forest-labs/flux-3-image-klein", Backend.AUTO) == Backend.MFLUX


def test_gguf_still_goes_to_ollama():
    """A GGUF name is the OllamaDiffuser backend, not this one."""
    assert detect_backend("flux.1-dev-gguf-q4ks", Backend.AUTO) == Backend.OLLAMA


def test_missing_model_files_lists_what_is_absent(tmp_path):
    missing = missing_model_files("flux-3-image-klein-gguf", tmp_path)
    assert set(missing) == set(MODEL_FILES)


def test_missing_model_files_is_empty_when_everything_is_there(model_dir):
    assert missing_model_files("flux-3-image-klein-gguf", model_dir) == []


def test_validate_reports_a_missing_binary(tmp_path, monkeypatch):
    monkeypatch.setenv("SDCLI_BIN", str(tmp_path / "not-here"))
    with patch("eyegen.backends.sdcli.install.shutil.which", return_value=None):
        status = validate_sdcli_install("flux-3-image-klein-gguf", model_dir=tmp_path)
    assert not status.installed
    assert "sd-cli not found" in status.message


def test_validate_reports_missing_files(tmp_path, binary):
    status = validate_sdcli_install("flux-3-image-klein-gguf", model_dir=tmp_path)
    assert not status.installed
    assert status.missing_files


def test_the_model_dir_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SDCLI_MODEL_DIR", str(tmp_path))
    assert get_sdcli_model_dir() == tmp_path


def test_the_timeout_follows_the_environment(monkeypatch):
    monkeypatch.setenv("SDCLI_TIMEOUT", "120")
    assert get_sdcli_timeout() == 120.0


def test_a_timeout_that_is_not_a_number_is_ignored(monkeypatch):
    monkeypatch.setenv("SDCLI_TIMEOUT", "soon")
    assert get_sdcli_timeout() > 0


def test_the_model_settings_win_over_the_tool_defaults():
    """A distilled model knows how many steps it was trained for."""
    steps, guidance = resolve_settings("flux-3-image-klein-gguf", 30, 7.5)
    assert steps == SDCLI_MODELS["flux-3-image-klein-gguf"]["steps"]
    assert guidance == SDCLI_MODELS["flux-3-image-klein-gguf"]["cfg_scale"]


def test_a_changed_setting_is_the_users():
    steps, guidance = resolve_settings("flux-3-image-klein-gguf", 8, 3.0)
    assert steps == 8
    assert guidance == 3.0


def test_the_command_only_carries_flags_the_backend_passes(model_dir, binary):
    """The subprocess guard must not let an unexpected flag through."""
    from eyegen.config import EyeGenConfig

    wrapper = SdCliWrapper(EyeGenConfig(), "flux-3-image-klein-gguf")

    command = wrapper._build_command("a prompt", model_dir / "out.png", 16, 1.0, 512, 512, 7, None)
    wrapper._validate_cmd_args(command)

    with pytest.raises(ValueError):
        wrapper._validate_cmd_args([str(binary), "--rm-rf"])


def test_the_command_carries_a_negative_prompt(model_dir, binary):
    from eyegen.config import EyeGenConfig

    wrapper = SdCliWrapper(EyeGenConfig(), "flux-3-image-klein-gguf")
    command = wrapper._build_command(
        "a prompt",
        model_dir / "out.png",
        16,
        1.0,
        512,
        512,
        7,
        None,
        "blurry, low quality",
    )
    joined = " ".join(command)
    assert "-n blurry, low quality" in joined
    wrapper._validate_cmd_args(command)


def test_the_command_carries_no_negative_prompt_when_there_is_none(model_dir, binary):
    from eyegen.config import EyeGenConfig

    wrapper = SdCliWrapper(EyeGenConfig(), "flux-3-image-klein-gguf")
    command = wrapper._build_command("a prompt", model_dir / "out.png", 16, 1.0, 512, 512, 7, None)
    assert "-n" not in command


def test_the_command_carries_the_three_model_files(model_dir, binary):
    from eyegen.config import EyeGenConfig

    wrapper = SdCliWrapper(EyeGenConfig(), "flux-3-image-klein-gguf")

    command = wrapper._build_command("a prompt", model_dir / "out.png", 16, 1.0, 512, 512, 7, None)
    joined = " ".join(command)
    for name in MODEL_FILES:
        assert name in joined
    assert "--offload-to-cpu" in command
