"""Tests for the model specifications: what a model is and how it is run.

A specification is what makes a model the tool does not know possible: it
says which files the model is made of, where they come from, and how it
wants to be run. These check that a specification survives the round trip,
that the user's own catalogue wins over the name patterns, and that the
command line can turn a repository into a model.
"""

import json

import pytest
from typer.testing import CliRunner

from eyegen.backends import detect_backend, resolve_model
from eyegen.backends.sdcli.pipeline import SdCliWrapper
from eyegen.cli.app import app
from eyegen.config import Backend, EyeGenConfig
from eyegen.model_catalogue import upsert_spec
from eyegen.model_specs import ModelSource, ModelSpec

runner = CliRunner()


def _spec(name: str = "klein", backend: Backend = Backend.SDCLI) -> ModelSpec:
    return ModelSpec(
        name=name,
        backend=backend,
        sources={
            "diffusion_model": ModelSource(repo="unsloth/flux", file="transformer.gguf"),
            "vae": ModelSource(repo="black-forest-labs/flux", file="ae.safetensors"),
        },
        steps=16,
        guidance=1.0,
        sampling_method="euler",
    )


def test_a_spec_survives_the_round_trip():
    spec = _spec()
    again = ModelSpec.from_dict(spec.name, spec.to_dict())
    assert again == spec


def test_a_spec_with_a_wrong_backend_says_what_is_known():
    with pytest.raises(ValueError, match="Known backends"):
        ModelSpec.from_dict("klein", {"backend": "nonsense"})


def test_a_spec_with_impossible_settings_says_so():
    """A negative number of steps would reach the process as an argument."""
    with pytest.raises(ValueError, match="at least 1"):
        ModelSpec.from_dict("klein", {"backend": "sdcli", "steps": -5})
    with pytest.raises(ValueError, match="not be negative"):
        ModelSpec.from_dict("klein", {"backend": "sdcli", "guidance": -1.0})


def test_a_guidance_of_zero_is_a_value():
    """A model trained without guidance is a real thing."""
    spec = ModelSpec.from_dict("klein", {"backend": "sdcli", "guidance": 0})
    assert spec.guidance == 0.0


def test_a_spec_without_a_repository_says_so():
    with pytest.raises(ValueError, match="repo"):
        ModelSpec.from_dict("klein", {"backend": "sdcli", "sources": {"vae": {"file": "a"}}})


def test_an_alias_becomes_the_model_the_backend_knows():
    """A specification for a fixed-set backend names one of its own models."""
    from eyegen.model_specs import apply_spec

    spec = ModelSpec(
        name="my-klein",
        backend=Backend.MFLUX,
        alias="flux3-image-klein",
        path="models/klein",
    )
    config = apply_spec(EyeGenConfig(model="my-klein"), spec)
    assert config.model == "flux3-image-klein"
    assert config.mflux_model_path == "models/klein"


def test_a_path_is_the_local_model_the_backend_reads():
    from eyegen.model_specs import apply_spec

    spec = ModelSpec(name="my-bundle", backend=Backend.COREML, path="models/bundle")
    config = apply_spec(EyeGenConfig(), spec)
    assert config.coreml_model_path == "models/bundle"


def test_a_spec_without_those_fields_changes_nothing():
    from eyegen.model_specs import apply_spec

    config = EyeGenConfig(model="something")
    assert apply_spec(config, None) is config
    assert apply_spec(config, _spec()) is config


def test_a_model_name_with_a_path_in_it_is_refused():
    """The name is the directory the model's files are written to."""
    with pytest.raises(ValueError, match="not a path"):
        ModelSpec(name="../../etc", backend=Backend.SDCLI)


def test_a_catalogue_entry_with_a_path_in_its_name_is_refused():
    """A catalogue the user edited is an input like any other."""
    with pytest.raises(ValueError, match="not a path"):
        ModelSpec.from_dict("../../etc", {"backend": "sdcli"})


def test_a_name_that_is_not_a_directory_is_refused():
    for name in ("", ".", "..", "-dashed"):
        with pytest.raises(ValueError):
            ModelSpec(name=name, backend=Backend.SDCLI)


def test_a_catalogue_with_a_bad_name_says_so():
    """The user is told what is wrong with the file, not shown a traceback."""
    from eyegen import model_catalogue

    model_catalogue.MODELS_FILE.write_text(
        json.dumps({"models": {"../../etc": {"backend": "sdcli"}}})
    )
    result = runner.invoke(app, ["generate", "a fox"])

    assert result.exit_code == 1
    assert "not a path" in result.output


def test_a_model_with_a_role_the_backend_does_not_know_is_refused(tmp_path, monkeypatch):
    """The role would be dropped, and the model would run without the file."""
    from eyegen.backends.sdcli.pipeline import SdCliWrapper

    monkeypatch.setenv("SDCLI_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("SDCLI_BIN", str(tmp_path / "sd-cli"))
    spec = ModelSpec(
        name="typo",
        backend=Backend.SDCLI,
        sources={"difusion_model": ModelSource(repo="owner/repo", file="model.gguf")},
    )
    with pytest.raises(ValueError, match="does not know"):
        SdCliWrapper(EyeGenConfig(), "typo", spec=spec)


def test_a_saved_model_is_reported_as_it_is():
    """The validation returns a record; unpacking it raised a TypeError."""
    from eyegen.saved_models import validate_saved_model

    validation = validate_saved_model("nowhere")
    assert validation.valid is False
    assert validation.error


def test_the_user_catalogue_wins_over_the_name_patterns():
    """A name that looks like a GGUF model runs the way the user said."""
    upsert_spec(_spec(name="flux.1-dev-gguf-q4ks", backend=Backend.COREML))
    assert detect_backend("flux.1-dev-gguf-q4ks", Backend.AUTO) == Backend.COREML


def test_a_model_of_the_catalogue_that_ships_still_resolves():
    assert detect_backend("flux-3-image-klein-gguf", Backend.AUTO) == Backend.SDCLI


def test_resolving_a_model_carries_its_specification():
    upsert_spec(_spec())
    resolved = resolve_model("klein")
    assert resolved.backend == Backend.SDCLI
    assert resolved.spec is not None
    assert resolved.spec.steps == 16


def test_a_backend_that_contradicts_the_specification_is_an_error():
    """Running a spec's files through another runtime would render garbage."""
    from eyegen.errors import ModelBackendConflict

    upsert_spec(_spec(name="klein"))
    with pytest.raises(ModelBackendConflict, match="cannot run it"):
        resolve_model("klein", Backend.MFLUX)


def test_a_backend_that_agrees_with_the_specification_is_kept():
    upsert_spec(_spec(name="klein"))
    resolved = resolve_model("klein", Backend.SDCLI)
    assert resolved.backend == Backend.SDCLI
    assert resolved.spec is not None


def test_a_backend_without_a_specification_is_the_one_asked_for():
    resolved = resolve_model("flux-3-image-klein-gguf", Backend.MFLUX)
    assert resolved.backend == Backend.MFLUX
    assert resolved.spec is None


def test_the_command_reports_a_conflicting_backend():
    """The user is told what happened, not shown a traceback."""
    upsert_spec(_spec(name="klein"))
    result = runner.invoke(app, ["generate", "a fox", "--model", "klein", "--backend", "mflux"])
    assert result.exit_code == 1
    assert "cannot run it" in result.output


def test_the_model_option_is_used_for_the_run_only(monkeypatch):
    """The configuration file is left as it was."""
    from eyegen.cli._generate_helpers import build_generation_params

    config = EyeGenConfig(model="argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized")
    monkeypatch.setattr("eyegen.cli._generate_helpers.load_config", lambda: config)

    run_config, resolved, *_ = build_generation_params(
        "auto", None, None, None, None, None, None, None, None, "flux-3-image-klein-gguf"
    )

    assert resolved.name == "flux-3-image-klein-gguf"
    assert run_config.model == "flux-3-image-klein-gguf"
    assert config.model == "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized"


def test_a_model_with_one_file_carries_only_that_flag(tmp_path, monkeypatch):
    """A model that is one GGUF file needs one flag, not three."""
    monkeypatch.setenv("SDCLI_MODEL_DIR", str(tmp_path))
    model_dir = tmp_path / "one-file"
    model_dir.mkdir()
    (model_dir / "transformer.gguf").write_bytes(b"x")
    binary = tmp_path / "sd-cli"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setenv("SDCLI_BIN", str(binary))

    spec = ModelSpec(
        name="one-file",
        backend=Backend.SDCLI,
        sources={"diffusion_model": ModelSource(repo="owner/repo", file="transformer.gguf")},
    )
    wrapper = SdCliWrapper(EyeGenConfig(), "one-file", spec=spec)
    command = wrapper._build_command("a prompt", model_dir / "out.png", 16, 1.0, 512, 512, 7, None)

    joined = " ".join(command)
    assert "--diffusion-model" in joined
    assert "--llm" not in joined
    assert "--vae" not in joined
    wrapper._validate_cmd_args(command)


def test_the_settings_of_a_model_come_from_its_specification():
    from eyegen.backends.sdcli.pipeline import resolve_settings

    spec = _spec()
    steps, guidance = resolve_settings("klein", 30, 7.5, spec=spec)
    assert steps == 16
    assert guidance == 1.0


def test_a_guidance_of_zero_is_not_replaced_by_the_default():
    """A model trained without guidance must not be given 7.5."""
    from eyegen.backends.sdcli.pipeline import resolve_settings

    spec = ModelSpec(name="klein", backend=Backend.SDCLI, guidance=0.0)
    _, guidance = resolve_settings("klein", 30, 7.5, spec=spec)
    assert guidance == 0.0
