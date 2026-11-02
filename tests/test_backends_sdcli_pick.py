"""Tests for the text encoder a stable-diffusion.cpp model gets.

The encoder is the file a repository most easily gets wrong: the slim one
can be a format this machine's Metal backend has no kernel for (int8
convrot), the model itself can be mistaken for an encoder, and the
runnable one can be 17 GB of full precision. These cover the resolution:
what runs, what is warned about, what is passed over, and where the
documented companion steps in.
"""

import logging

from eyegen.backends.sdcli.pick import resolve_llm

QWEN3VL_8B_GGUF = ("Qwen/Qwen3-VL-8B-Instruct-GGUF", "Qwen3VL-8B-Instruct-Q4_K_M.gguf")
REPO = "abenzerps/Qwen-Image-2.1-Uncensored-Quantized"
TRANSFORMER = "qwen-image-2.1-UC-Q4_K_M.gguf"


def test_an_int8_only_repository_gets_the_documented_companion():
    """No encoder the Metal backend can run: the companion's GGUF does it."""
    files = [
        TRANSFORMER,
        "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
        "vae/qwen_image_2.1_vae_bf16.safetensors",
    ]

    choice = resolve_llm(files, REPO, transformer=TRANSFORMER)

    assert choice.file == QWEN3VL_8B_GGUF[1]
    assert choice.repo == QWEN3VL_8B_GGUF[0]


def test_a_full_precision_only_repository_gets_the_companion_too():
    """A 5 GB quantised GGUF beats a 17 GB bf16 encoder from the same repo."""
    files = [
        "qwen-image-2.1-UC-BF16.gguf",
        "text_encoders/qwen3vl_8b_bf16.safetensors",
        "vae/qwen_image_2.1_vae_bf16.safetensors",
    ]

    choice = resolve_llm(files, REPO, transformer="qwen-image-2.1-UC-BF16.gguf")

    assert choice.file == QWEN3VL_8B_GGUF[1]
    assert choice.repo == QWEN3VL_8B_GGUF[0]


def test_a_quantised_gguf_in_the_repository_itself_wins():
    """The model's own repository ships a runnable encoder: it is used."""
    files = [
        TRANSFORMER,
        "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
        "text_encoders/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        "vae/qwen_image_2.1_vae_bf16.safetensors",
    ]

    choice = resolve_llm(files, REPO, transformer=TRANSFORMER)

    assert choice.file == "text_encoders/Qwen3VL-8B-Instruct-Q4_K_M.gguf"
    assert choice.repo == REPO


def test_another_model_family_keeps_its_full_precision_encoder(caplog):
    """No companion for, say, a Flux repository: its bf16 encoder is kept."""
    files = [
        "flux-3-image-klein-UD-Q4_K_XL.gguf",
        "t5xxl_fp16.safetensors",
        "ae.safetensors",
    ]

    with caplog.at_level(logging.WARNING):
        choice = resolve_llm(files, "unsloth/flux-3-image-klein-GGUF")

    assert choice.file == "t5xxl_fp16.safetensors"
    assert choice.repo == "unsloth/flux-3-image-klein-GGUF"
    assert "full precision" in caplog.text
    # The repository ships no int8 encoder: the warning must not say it does.
    assert "int8" not in caplog.text


def test_the_encoder_that_cannot_run_is_named(caplog):
    """An int8 encoder that was passed over is named in the warning."""
    files = [
        "flux-3-image-klein-UD-Q4_K_XL.gguf",
        "t5xxl_fp16.safetensors",
        "text_encoders/clip_l_int8_convrot.safetensors",
    ]

    with caplog.at_level(logging.WARNING):
        choice = resolve_llm(files, "unsloth/flux-3-image-klein-GGUF")

    assert choice.file == "t5xxl_fp16.safetensors"
    assert "text_encoders/clip_l_int8_convrot.safetensors" in caplog.text
    assert "no kernel" in caplog.text


def test_a_repository_with_no_encoder_at_all_is_left_without_one():
    """Nothing encoder-shaped and no companion: the absence is honest."""
    files = ["flux-3-image-klein-UD-Q4_K_XL.gguf", "ae.safetensors"]

    choice = resolve_llm(files, "unsloth/flux-3-image-klein-GGUF")

    assert choice.file is None
    assert choice.repo == "unsloth/flux-3-image-klein-GGUF"
    assert choice.rejected == ()


def test_the_companion_is_chosen_by_the_model_family_not_by_the_file_name():
    """The spelling of a file name must not decide which encoder runs."""
    files = [
        "Qwen-Image-2.1-UC-Q4_K_M.gguf",
        "text_encoders/qwen3_vl_8b_int8_convrot.safetensors",
    ]

    choice = resolve_llm(files, "owner/quantized", transformer=files[0])

    assert choice.file == QWEN3VL_8B_GGUF[1]
    assert choice.repo == QWEN3VL_8B_GGUF[0]


def test_a_repository_that_ships_no_encoder_gets_the_companion():
    """A family with a documented companion does not need an encoder of its own."""
    files = [TRANSFORMER, "vae/qwen_image_2.1_vae_bf16.safetensors"]

    choice = resolve_llm(files, REPO, transformer=TRANSFORMER)

    assert choice.file == QWEN3VL_8B_GGUF[1]
    assert choice.repo == QWEN3VL_8B_GGUF[0]


def test_the_transformer_is_never_taken_for_the_encoder():
    """A model whose name says 'instruct' is the model, not its encoder."""
    files = [
        "flux-3-image-klein-Instruct-Q4_K_XL.gguf",
        "t5xxl_fp16.safetensors",
    ]

    choice = resolve_llm(files, "owner/repo", transformer=files[0])

    assert choice.file == "t5xxl_fp16.safetensors"


def test_the_only_gguf_in_a_listing_is_the_transformer():
    """One GGUF is a model of one file, and that file is the model."""
    files = [
        "transformer.gguf",
        "text_encoders/clip_l_int8_convrot.safetensors",
    ]

    choice = resolve_llm(files, "owner/repo")

    assert choice.file is None
    assert choice.rejected == ("text_encoders/clip_l_int8_convrot.safetensors",)
