"""Tests for completing a lone GGUF with its repository companions.

A transformer alone is half a model: sd-cli also needs the text encoder
and the VAE, and when they cannot be named the run must say so instead
of handing the failure to the binary.
"""

import logging

import pytest

from eyegen.backends.sdcli import models
from eyegen.backends.sdcli.models import complete_local_spec
from eyegen.backends.sdcli.pipeline import SdCliWrapper
from eyegen.config import Backend, EyeGenConfig
from eyegen.model_specs import ModelSource, ModelSpec


def _cache_gguf(tmp_path):
    """A lone GGUF in a HuggingFace cache layout, and a spec for it."""
    transformer = tmp_path / "models--owner--repo" / "snapshots" / "abc123" / "flux.gguf"
    transformer.parent.mkdir(parents=True)
    transformer.write_bytes(b"x")
    spec = ModelSpec(
        name="lone-gguf",
        backend=Backend.SDCLI,
        sources={"diffusion_model": ModelSource(path=str(transformer))},
    )
    return transformer, spec


COMPANION_FILES = ["ae.safetensors", "t5xxl_fp8.safetensors"]


def test_completing_a_lone_gguf_preserves_every_spec_field(tmp_path, monkeypatch):
    """Completion changes the sources and the repo, and nothing else."""
    spec = ModelSpec(
        name="lone-gguf",
        backend=Backend.SDCLI,
        sources=_cache_gguf(tmp_path)[1].sources,
        alias="catalogue-alias",
        description="a lone GGUF",
        variant="q4",
    )
    monkeypatch.setattr(models, "_repo_files", lambda repo: COMPANION_FILES)

    completed = complete_local_spec(spec)

    assert completed is not spec
    assert set(completed.sources) == {"diffusion_model", "vae", "llm"}
    assert completed.repo == "owner/repo"
    assert completed.alias == "catalogue-alias"
    assert completed.variant == "q4"
    assert completed.description == "a lone GGUF"


def test_companions_that_cannot_be_listed_refuse_with_the_reason(tmp_path, monkeypatch):
    """An unreachable repository must not look like one with no companions.

    A lone GGUF is a half a model; running it silently hands the failure
    to sd-cli. Saying the listing failed tells the user what to fix.
    """
    monkeypatch.setattr(models, "_repo_files", lambda repo: None)

    with pytest.raises(RuntimeError, match="could not be listed"):
        SdCliWrapper(EyeGenConfig(), "lone-gguf", spec=_cache_gguf(tmp_path)[1])


def test_a_first_run_fetch_lands_in_the_configured_tree(tmp_path, monkeypatch):
    """The companion fetch honours the config tree, not the env override.

    A bare pull would resolve the root on its own — the environment wins
    there — and the companions would land in a tree the validation never
    looks at, failing with "could not be fetched" for files that are there.
    """
    config_tree = tmp_path / "config-tree"
    env_tree = tmp_path / "env-tree"
    monkeypatch.setenv("EYEGEN_MODELS_DIR", str(env_tree))
    binary = tmp_path / "sd-cli"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setenv("SDCLI_BIN", str(binary))

    spec = _cache_gguf(tmp_path)[1]
    monkeypatch.setattr(models, "_repo_files", lambda repo: COMPANION_FILES)
    fetched_into = []

    def fake_pull(model, spec=None, model_dir=None, **kwargs):
        fetched_into.append(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        for source in spec.sources.values():
            if not source.is_local:
                (model_dir / source.name).write_bytes(b"x")
        return model_dir

    monkeypatch.setattr(models, "pull_sdcli_model", fake_pull)

    wrapper = SdCliWrapper(EyeGenConfig(models_dir=str(config_tree)), "lone-gguf", spec=spec)

    expected = config_tree / "sdcli" / "lone-gguf"
    assert fetched_into == [expected]
    assert wrapper.model_dir == expected
    assert not (env_tree / "sdcli" / "lone-gguf").exists()


def test_a_failed_fetch_names_the_repository_each_file_comes_from(tmp_path, monkeypatch):
    """The advice after a failed download points at the right repository.

    The text encoder may come from a companion repository, and telling the
    user to fetch it from the model's own repository sends them for a file
    that is not there.
    """
    transformer = (
        tmp_path
        / "models--owner--Qwen-Image-2.1-Uncensored-Quantized"
        / "snapshots"
        / "abc123"
        / "qwen-image-2.1-UC-Q4_K_M.gguf"
    )
    transformer.parent.mkdir(parents=True)
    transformer.write_bytes(b"x")
    spec = ModelSpec(
        name="qwen-image-2.1-UC-Q4_K_M",
        backend=Backend.SDCLI,
        sources={"diffusion_model": ModelSource(path=str(transformer))},
    )
    monkeypatch.setattr(
        models,
        "_repo_files",
        lambda repo: [
            "qwen-image-2.1-UC-Q4_K_M.gguf",
            "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
        ],
    )
    binary = tmp_path / "sd-cli"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setenv("SDCLI_BIN", str(binary))
    # The model's directory must start empty: a real download of the same
    # model on this machine would hold exactly the files the fetch is meant
    # to be missing, and the run would not come to the advice at all.
    monkeypatch.setenv("SDCLI_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(models, "pull_sdcli_model", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="Qwen/Qwen3-VL-8B-Instruct-GGUF"):
        SdCliWrapper(EyeGenConfig(), "qwen-image-2.1-UC-Q4_K_M", spec=spec)


def test_an_encoder_the_machine_cannot_run_refuses_the_run(tmp_path, monkeypatch):
    """An int8 encoder is no kernel on Metal: the run says so, or fails oddly."""
    snapshot = tmp_path / "models--owner--flux-3-image-klein-GGUF" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    gguf = snapshot / "flux-3-image-klein-UD-Q4_K_XL.gguf"
    gguf.write_bytes(b"x" * 8)

    spec = ModelSpec(
        name="flux-3-image-klein-UD-Q4_K_XL",
        backend=Backend.SDCLI,
        sources={"diffusion_model": ModelSource(path=str(gguf))},
    )
    monkeypatch.setattr(
        models,
        "_repo_files",
        lambda repo: [
            "flux-3-image-klein-UD-Q4_K_XL.gguf",
            "text_encoders/clip_l_int8_convrot.safetensors",
            "ae.safetensors",
        ],
    )

    with pytest.raises(RuntimeError, match="clip_l_int8_convrot") as raised:
        complete_local_spec(spec)

    assert "Metal backend has no kernel" in str(raised.value)


def test_a_repository_with_no_encoder_at_all_is_warned_about(tmp_path, monkeypatch, caplog):
    """No encoder and no companion: the absence is said out loud."""
    snapshot = tmp_path / "models--owner--repo" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    gguf = snapshot / "transformer.gguf"
    gguf.write_bytes(b"x" * 8)

    spec = ModelSpec(
        name="transformer",
        backend=Backend.SDCLI,
        sources={"diffusion_model": ModelSource(path=str(gguf))},
    )
    monkeypatch.setattr(
        models,
        "_repo_files",
        lambda repo: ["transformer.gguf", "ae.safetensors"],
    )

    with caplog.at_level(logging.WARNING):
        completed = complete_local_spec(spec)

    assert "llm" not in completed.sources
    assert "no text encoder" in caplog.text


def test_a_text_encoder_is_not_run_as_the_model(tmp_path):
    """The companions live in a directory of their own; it is not a model.

    A directory of one GGUF is that file's model, so the encoder the tool
    fetched next to a model whose transformer lives elsewhere would be run
    as the model, and sd-cli fails on a file it cannot read a version from.
    """
    directory = tmp_path / "qwen-image-2.1-UC-Q4_K_M"
    directory.mkdir()
    encoder = directory / "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
    encoder.write_bytes(b"x")
    spec = ModelSpec(
        name="qwen-image-2.1-UC-Q4_K_M",
        backend=Backend.SDCLI,
        sources={"diffusion_model": ModelSource(path=str(encoder))},
    )

    with pytest.raises(ValueError, match="is a text encoder, not a model"):
        SdCliWrapper(EyeGenConfig(), "qwen-image-2.1-UC-Q4_K_M", spec=spec)
