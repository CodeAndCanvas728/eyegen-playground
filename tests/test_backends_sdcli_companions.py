"""Tests for completing a lone GGUF with its repository companions.

A transformer alone is half a model: sd-cli also needs the text encoder
and the VAE, and when they cannot be named the run must say so instead
of handing the failure to the binary.
"""

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
