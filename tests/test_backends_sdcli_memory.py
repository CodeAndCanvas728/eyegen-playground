"""Tests for the stable-diffusion.cpp backend's memory behaviour.

A model sd-cli would load whole is refused before the machine dies, and
the Low Memory mode toggle decides whether the weights stream from RAM.
"""

import pytest

from eyegen.backends.sdcli.pipeline import SdCliWrapper
from eyegen.config import Backend, EyeGenConfig
from eyegen.model_specs import ModelSource, ModelSpec

MODEL = "flux-3-image-klein-gguf"
MODEL_FILES = (
    "flux-3-image-klein-UD-Q4_K_XL.gguf",
    "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
    "ae.safetensors",
)


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """A directory with the three model files in it."""
    monkeypatch.setenv("SDCLI_MODEL_DIR", str(tmp_path))
    directory = tmp_path / MODEL
    directory.mkdir()
    for name in MODEL_FILES:
        (directory / name).write_bytes(b"x")
    return directory


@pytest.fixture
def binary(tmp_path, monkeypatch):
    """An sd-cli binary in a directory of its own."""
    path = tmp_path / "sd-cli"
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    monkeypatch.setenv("SDCLI_BIN", str(path))
    return path


def test_a_model_too_large_for_the_machine_is_refused(tmp_path, binary, monkeypatch):
    """A model sd-cli would load whole is refused before the machine dies."""
    from eyegen.backends.sdcli import runner

    directory = tmp_path / "huge"
    directory.mkdir()
    (directory / "transformer.gguf").write_bytes(b"x" * 1024)
    (directory / "ae.safetensors").write_bytes(b"x" * 1024)
    spec = ModelSpec(
        name="too-big",
        backend=Backend.SDCLI,
        sources={
            "diffusion_model": ModelSource(path=str(directory / "transformer.gguf")),
            "vae": ModelSource(path=str(directory / "ae.safetensors")),
        },
    )

    monkeypatch.setattr(runner, "physical_memory", lambda: 1024)  # a 1 KB machine

    with pytest.raises(RuntimeError, match="more memory"):
        SdCliWrapper(EyeGenConfig(), "too-big", spec=spec)


def test_offload_to_cpu_is_omitted_when_low_memory_mode_is_off(model_dir, binary):
    from eyegen.config import EyeGenConfig

    config = EyeGenConfig(sdcli_offload_to_cpu=False)
    wrapper = SdCliWrapper(config, "flux-3-image-klein-gguf")
    command = wrapper._build_command("a prompt", model_dir / "out.png", 16, 1.0, 512, 512, 7, None)

    assert "--offload-to-cpu" not in command


def test_offload_to_cpu_is_passed_when_low_memory_mode_is_on(model_dir, binary):
    from eyegen.config import EyeGenConfig

    config = EyeGenConfig(sdcli_offload_to_cpu=True)
    wrapper = SdCliWrapper(config, "flux-3-image-klein-gguf")
    command = wrapper._build_command("a prompt", model_dir / "out.png", 16, 1.0, 512, 512, 7, None)

    assert "--offload-to-cpu" in command


def test_physical_memory_reads_the_machine_size():
    """The machine's memory is a positive number, on macOS and elsewhere."""
    from eyegen.backends.sdcli.install import physical_memory

    assert physical_memory() > 0


def test_physical_memory_falls_back_to_sysconf_on_macos(monkeypatch):
    """When sysctl cannot answer on macOS, the generic path still can."""
    from eyegen.backends.sdcli import install

    monkeypatch.setattr(install, "_sysctl_hw_memsize", lambda: None)
    monkeypatch.setattr(install, "_sysconf_physical_pages_bytes", lambda: 512)

    assert install.physical_memory() == 512


def test_undeterminable_memory_is_refused_loudly(monkeypatch):
    """A guard that guesses when it cannot see would never fire; say so."""
    from eyegen.backends.sdcli import install

    monkeypatch.setattr(install, "_sysctl_hw_memsize", lambda: None)
    monkeypatch.setattr(install, "_sysconf_physical_pages_bytes", lambda: None)

    with pytest.raises(RuntimeError, match="Could not determine"):
        install.physical_memory()


def test_an_unreadable_machine_refuses_the_run(tmp_path, binary, monkeypatch):
    """The refusal reaches the user as the run being refused, not a guess."""
    from eyegen.backends.sdcli import runner

    directory = tmp_path / "small"
    directory.mkdir()
    (directory / "transformer.gguf").write_bytes(b"x")

    def broken() -> int:
        raise RuntimeError("Could not determine")

    monkeypatch.setattr(runner, "physical_memory", broken)

    with pytest.raises(RuntimeError, match="Could not determine"):
        SdCliWrapper(
            EyeGenConfig(),
            "small",
            spec=ModelSpec(
                name="small",
                backend=Backend.SDCLI,
                sources={"diffusion_model": ModelSource(path=str(directory / "transformer.gguf"))},
            ),
        )
