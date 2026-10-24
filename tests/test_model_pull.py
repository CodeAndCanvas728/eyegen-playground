"""Tests for fetching a model: by name, and from a repository.

A model the tool does not know comes from a repository: the command line
turns the repository and its files into a specification, writes it to the
catalogue and fetches the files. These check that the specification is
what it says, and that a model which cannot be fetched says why.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from eyegen.backends.sdcli.models import pull_sdcli_model
from eyegen.cli.app import app
from eyegen.config import Backend
from eyegen.model_catalogue import load_catalogue
from eyegen.model_specs import ModelSource, ModelSpec

runner = CliRunner()


def _spec(name: str = "klein") -> ModelSpec:
    return ModelSpec(
        name=name,
        backend=Backend.SDCLI,
        sources={
            "diffusion_model": ModelSource(repo="unsloth/flux", file="transformer.gguf"),
            "vae": ModelSource(repo="black-forest-labs/flux", file="ae.safetensors"),
        },
        steps=16,
        guidance=1.0,
        sampling_method="euler",
    )


def test_the_pull_downloads_every_file_of_the_specification(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    monkeypatch.setenv("SDCLI_MODEL_DIR", str(tmp_path))
    downloaded = []

    def fake_download(repo_id, filename, local_dir):
        downloaded.append((repo_id, filename))
        path = tmp_path / spec.name / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return str(path)

    spec = _spec()
    with patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
        directory = pull_sdcli_model(spec.name, spec=spec)

    assert directory == tmp_path / spec.name
    assert set(downloaded) == {
        ("unsloth/flux", "transformer.gguf"),
        ("black-forest-labs/flux", "ae.safetensors"),
    }
    assert (tmp_path / spec.name / "transformer.gguf").is_file()
    assert (tmp_path / spec.name / "ae.safetensors").is_file()


def test_the_pull_writes_the_specification_and_fetches_the_model(tmp_path, monkeypatch):
    """The command line turns a repository into a model."""
    pytest.importorskip("huggingface_hub")
    monkeypatch.setenv("SDCLI_MODEL_DIR", str(tmp_path))

    def fake_download(repo_id, filename, local_dir):
        path = Path(local_dir) / filename
        path.write_bytes(b"x")
        return str(path)

    with patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
        result = runner.invoke(
            app,
            [
                "pull",
                "--from-hf",
                "owner/repo",
                "--as",
                "little-model",
                "--file",
                "diffusion_model=owner/repo:model.gguf",
                "--steps",
                "8",
            ],
        )

    assert result.exit_code == 0, result.output
    spec = load_catalogue()["little-model"]
    assert spec.backend == Backend.SDCLI
    assert spec.sources["diffusion_model"].repo == "owner/repo"
    assert spec.sources["diffusion_model"].file == "model.gguf"
    assert spec.steps == 8
    assert (tmp_path / "little-model" / "model.gguf").is_file()


def test_a_pull_without_a_name_or_a_repository_says_so():
    result = runner.invoke(app, ["pull"])
    assert result.exit_code == 1
    assert "--from-hf" in result.output


def test_a_file_without_a_repository_says_so():
    from eyegen.cli.pull import _parse_source

    with pytest.raises(ValueError, match="no repository"):
        _parse_source("vae=ae.safetensors", None)


def test_a_model_with_no_files_at_all_says_so():
    result = runner.invoke(app, ["pull", "--from-hf", "owner/repo"])
    assert result.exit_code == 1
    assert "at least one file" in result.output


def test_a_role_the_backend_does_not_know_says_so():
    """A typo would drop the file, and the model would run without it."""
    result = runner.invoke(
        app,
        [
            "pull",
            "--from-hf",
            "owner/repo",
            "--file",
            "difusion_model=owner/repo:model.gguf",
        ],
    )
    assert result.exit_code == 1
    assert "does not know the role" in result.output


def test_a_role_given_twice_says_so():
    """The second file would silently win, and the first would be fetched."""
    result = runner.invoke(
        app,
        [
            "pull",
            "--from-hf",
            "owner/repo",
            "--file",
            "diffusion_model=owner/repo:one.gguf",
            "--file",
            "diffusion_model=owner/repo:two.gguf",
        ],
    )
    assert result.exit_code == 1
    assert "given twice" in result.output


def test_a_file_name_with_a_path_in_it_is_refused():
    """The name becomes a path under the model's own directory."""
    from eyegen.model_specs import ModelSource

    with pytest.raises(ValueError, match="not a path"):
        ModelSource(repo="owner/repo", file="a.gguf", name="../../../etc/passwd")


def test_a_model_name_with_a_path_in_it_is_refused():
    """The model's name is the directory its files are written to."""
    result = runner.invoke(
        app,
        [
            "pull",
            "--from-hf",
            "owner/repo",
            "--as",
            "../../x",
            "--file",
            "diffusion_model=owner/repo:model.gguf",
        ],
    )

    assert result.exit_code == 1
    assert "not a path" in result.output
    assert load_catalogue() == {}


def test_a_fixed_set_backend_says_so():
    """MFLUX runs a fixed set of models; it cannot fetch one from a repository."""
    result = runner.invoke(
        app,
        [
            "pull",
            "--from-hf",
            "owner/repo",
            "--backend",
            "mflux",
            "--file",
            "diffusion_model=owner/repo:model.gguf",
        ],
    )

    assert result.exit_code == 1
    assert "fixed set of models" in result.output
    assert load_catalogue() == {}


def test_a_file_name_that_is_a_directory_is_refused():
    from eyegen.model_specs import ModelSource

    with pytest.raises(ValueError, match="not a file name"):
        ModelSource(repo="owner/repo", file="a.gguf", name="..")


def test_the_gguf_entry_is_written_where_ollamadiffuser_reads_it(tmp_path):
    """OllamaDiffuser reads an external registry file; this is that file."""
    from eyegen.ollama_registry import entry_for, write_entries

    spec = ModelSpec(
        name="my-gguf",
        backend=Backend.OLLAMA,
        sources={"diffusion_model": ModelSource(repo="owner/repo", file="model.gguf")},
    )
    write_entries({"my-gguf": entry_for(spec)})

    written = json.loads((tmp_path / "ollamadiffuser-models.json").read_text())
    entry = written["models"]["my-gguf"]
    assert entry["repo_id"] == "owner/repo"
    assert "gguf" in entry["variant"]


def test_a_gguf_model_without_a_file_says_so():
    from eyegen.ollama_registry import entry_for

    spec = ModelSpec(name="my-gguf", backend=Backend.OLLAMA)
    with pytest.raises(ValueError, match="GGUF"):
        entry_for(spec)
