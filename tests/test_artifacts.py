"""Tests for what a model on disk is.

A model the tool does not know by name can still be a model: a path on
this machine is described by what is there. These check the rules that
say what a thing is — a bundle, a snapshot, a set of weights — and that a
model is run the way what is there says.
"""

from pathlib import Path

import pytest

from eyegen.artifacts import describe, infer_sdcli_roles, local_model_name
from eyegen.backends import resolve_model
from eyegen.config import Backend


def _gguf(directory, name="flux1-dev-Q4_K_M.gguf", size=8):
    path = directory / name
    path.write_bytes(b"x" * size)
    return path


def test_a_gguf_file_is_a_model_for_the_package_that_reads_those_files(tmp_path):
    """A GGUF is OllamaDiffuser's own format."""
    gguf = _gguf(tmp_path)

    artifact = describe(gguf)

    assert artifact is not None
    assert artifact.backend == Backend.OLLAMA
    assert artifact.spec.sources["diffusion_model"].path == str(gguf)


def test_the_name_of_a_local_model_carries_its_family(tmp_path):
    """The package reads the family off the name of the model's directory."""
    assert local_model_name("flux1-dev-Q4_K_M") == "flux1-dev-Q4_K_M"
    assert local_model_name("sd3.5-large") == "sd3.5-large"
    assert local_model_name("some-model") == "flux-some-model"


def test_a_coreml_bundle_is_a_coreml_model(tmp_path):
    """A converted bundle is what it looks like."""
    bundle = tmp_path / "sd-2-1-base"
    bundle.mkdir()
    (bundle / "model.mlmodelc").mkdir()

    artifact = describe(bundle)

    assert artifact is not None
    assert artifact.backend == Backend.COREML
    assert artifact.spec.path == str(bundle)


def test_a_diffusers_model_is_a_coreml_model_that_is_converted_on_first_use(tmp_path):
    """The MLX backend cannot run it: its checkpoints are a table in a package."""
    snapshot = tmp_path / "my-model"
    (snapshot / "unet").mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")

    artifact = describe(snapshot)

    assert artifact is not None
    assert artifact.backend == Backend.COREML
    assert any("converted to CoreML" in note for note in artifact.notes)


def test_a_directory_of_weights_is_a_stable_diffusion_model(tmp_path):
    """The transformer, the encoder and the VAE: the three roles of a model."""
    directory = tmp_path / "klein"
    directory.mkdir()
    _gguf(directory, "flux-3-image-klein-UD-Q4_K_XL.gguf", size=64)
    _gguf(directory, "Qwen3VL-4B-Instruct-Q4_K_M.gguf", size=32)
    (directory / "ae.safetensors").write_bytes(b"x")

    artifact = describe(directory)

    assert artifact is not None
    assert artifact.backend == Backend.SDCLI
    sources = artifact.spec.sources
    assert sources["diffusion_model"].path.endswith("flux-3-image-klein-UD-Q4_K_XL.gguf")
    assert sources["llm"].path.endswith("Qwen3VL-4B-Instruct-Q4_K_M.gguf")
    assert sources["vae"].path.endswith("ae.safetensors")


def test_what_was_inferred_is_said_out_loud(tmp_path):
    """A role the user meant differently is one line of a specification away."""
    directory = tmp_path / "klein"
    directory.mkdir()
    _gguf(directory, "flux-3-image-klein-UD-Q4_K_XL.gguf", size=64)
    _gguf(directory, "Qwen3VL-4B-Instruct-Q4_K_M.gguf", size=32)
    (directory / "ae.safetensors").write_bytes(b"x")

    artifact = describe(directory)

    assert artifact is not None
    joined = " ".join(artifact.notes)
    assert "diffusion_model → flux-3-image-klein-UD-Q4_K_XL.gguf" in joined
    assert "llm → Qwen3VL-4B-Instruct-Q4_K_M.gguf" in joined
    assert "vae → ae.safetensors" in joined


def test_a_directory_that_is_not_a_model_is_not_one(tmp_path):
    directory = tmp_path / "notes"
    directory.mkdir()
    (directory / "readme.txt").write_text("hello")

    assert describe(directory) is None


def test_a_gguf_with_a_text_encoder_next_to_it_is_the_package_layout(tmp_path):
    """The package fetches the encoders itself, so a model looks like this."""
    directory = tmp_path / "flux1-dev"
    directory.mkdir()
    _gguf(directory)
    (directory / "clip_l.safetensors").write_bytes(b"x")

    artifact = describe(directory)

    assert artifact is not None
    assert artifact.backend == Backend.OLLAMA


def test_the_roles_are_read_off_the_names(tmp_path):
    directory = tmp_path / "klein"
    directory.mkdir()
    _gguf(directory, "transformer.gguf", size=64)
    _gguf(directory, "t5xxl-Q4_K_M.gguf", size=32)
    (directory / "ae.safetensors").write_bytes(b"x")

    roles, inferred = infer_sdcli_roles(directory)

    assert roles["diffusion_model"].name == "transformer.gguf"
    assert roles["t5xxl"].name == "t5xxl-Q4_K_M.gguf"
    assert roles["vae"].name == "ae.safetensors"
    assert len(inferred) == 3


def test_the_transformer_is_a_gguf_no_other_role_claimed(tmp_path):
    """The largest GGUF is the transformer only when it fills no other role."""
    directory = tmp_path / "klein"
    directory.mkdir()
    _gguf(directory, "Qwen3VL-4B-Instruct-Q4_K_M.gguf", size=64)
    _gguf(directory, "weights.gguf", size=32)

    roles, _ = infer_sdcli_roles(directory)

    assert roles["llm"].name == "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
    assert roles["diffusion_model"].name == "weights.gguf"


def test_a_path_on_the_command_line_is_a_model(tmp_path):
    """The path is not a name: what it is comes from what is there."""
    gguf = _gguf(tmp_path)

    resolved = resolve_model(str(gguf))

    assert resolved.backend == Backend.OLLAMA
    assert resolved.name == "flux1-dev-Q4_K_M"
    assert resolved.spec is not None
    assert resolved.spec.sources["diffusion_model"].path == str(gguf)


def test_a_path_is_resolved_the_same_way_in_the_gui(tmp_path):
    """The GUI asks the same resolver, so it says the same thing."""
    from eyegen.backends import resolve_spec

    gguf = _gguf(tmp_path)

    resolution = resolve_spec(str(gguf))

    assert resolution.name == "flux1-dev-Q4_K_M"
    assert resolution.spec is not None


def test_a_relative_path_is_made_absolute(tmp_path, monkeypatch):
    """The path is given to a runtime and used for a link: it cannot be relative."""
    monkeypatch.chdir(tmp_path)
    gguf = _gguf(tmp_path)

    artifact = describe(Path("flux1-dev-Q4_K_M.gguf"))

    assert artifact is not None
    assert artifact.path == gguf
    assert artifact.spec.sources["diffusion_model"].path == str(gguf)


def test_the_order_of_the_detection_is_not_moved(tmp_path):
    """A path is a different question, not a new rung in the chain."""
    from eyegen.backends import detect_backend

    assert detect_backend("flux-3-image-klein-gguf") == Backend.SDCLI
    assert detect_backend("flux.1-dev-gguf-q4ks") == Backend.OLLAMA
    assert detect_backend("bonsai-ternary-mlx") == Backend.BONSAI
    assert detect_backend("sd-2-1-base") == Backend.COREML

    gguf = _gguf(tmp_path)
    assert detect_backend(str(gguf)) == Backend.OLLAMA


def test_a_backend_that_contradicts_a_path_is_an_error(tmp_path):
    """The path is a GGUF model; the MFLUX runtime cannot run one."""
    from eyegen.errors import ModelBackendConflict

    gguf = _gguf(tmp_path)

    with pytest.raises(ModelBackendConflict, match="cannot run it"):
        resolve_model(str(gguf), Backend.MFLUX)


def test_a_symlinked_gguf_keeps_its_own_name(tmp_path):
    """A HuggingFace cache links a blob whose name says nothing."""
    blob = tmp_path / "blobs" / "e79c8a009f2ecbdb6c70fd663d9aea9ee304a0d91f347e4169a756b8ad141b41"
    blob.parent.mkdir()
    blob.write_bytes(b"x" * 8)
    snapshot = tmp_path / "snapshots" / "7959e658915f52274897fb573ffeda49de8dc16b"
    snapshot.mkdir(parents=True)
    link = snapshot / "qwen-image-2.1-UC-Q4_K_M.gguf"
    link.symlink_to(blob)

    artifact = describe(link)

    assert artifact is not None
    assert artifact.backend == Backend.OLLAMA
    assert artifact.spec.sources["diffusion_model"].path == str(link)


def test_a_lone_gguf_is_the_transformer_not_an_encoder(tmp_path):
    """One file is one model; a name may not claim it as its encoder."""
    directory = tmp_path / "qwen-image"
    directory.mkdir()
    _gguf(directory, "qwen-image-2.1-UC-Q4_K_M.gguf")

    roles, inferred = infer_sdcli_roles(directory)

    assert roles["diffusion_model"].name == "qwen-image-2.1-UC-Q4_K_M.gguf"
    assert "llm" not in roles
    assert any("diffusion_model" in line for line in inferred)


def test_a_directory_of_one_gguf_is_named_after_the_file(tmp_path):
    """A revision hash is where the model is, not what it is called."""
    snapshot = tmp_path / "snapshots" / "7959e658915f52274897fb573ffeda49de8dc16b"
    snapshot.mkdir(parents=True)
    _gguf(snapshot, "qwen-image-2.1-UC-Q4_K_M.gguf")

    artifact = describe(snapshot)

    assert artifact is not None
    assert artifact.name == "qwen-image-2.1-UC-Q4_K_M"
