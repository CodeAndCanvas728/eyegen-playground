"""Tests for the text encoder a stable-diffusion.cpp model gets.

The encoder is the file a repository most easily gets wrong: the slim one
can be a format this machine's Metal backend has no kernel for (int8
convrot), and the runnable one can be 17 GB of full precision. These
cover the resolution: what runs, what is warned about, and where the
documented companion steps in.
"""

import logging

from eyegen.backends.sdcli.pick import resolve_llm

QWEN3VL_8B_GGUF = ("Qwen/Qwen3-VL-8B-Instruct-GGUF", "Qwen3VL-8B-Instruct-Q4_K_M.gguf")
REPO = "abenzerps/Qwen-Image-2.1-Uncensored-Quantized"


def test_an_int8_only_repository_gets_the_documented_companion():
    """No encoder the Metal backend can run: the companion's GGUF does it."""
    files = [
        "qwen-image-2.1-UC-Q4_K_M.gguf",
        "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
        "vae/qwen_image_2.1_vae_bf16.safetensors",
    ]

    assert resolve_llm(files, REPO) == ("Qwen3VL-8B-Instruct-Q4_K_M.gguf", QWEN3VL_8B_GGUF[0])


def test_a_full_precision_only_repository_gets_the_companion_too():
    """A 5 GB quantised GGUF beats a 17 GB bf16 encoder from the same repo."""
    files = [
        "qwen-image-2.1-UC-BF16.gguf",
        "text_encoders/qwen3vl_8b_bf16.safetensors",
        "vae/qwen_image_2.1_vae_bf16.safetensors",
    ]

    assert resolve_llm(files, REPO) == ("Qwen3VL-8B-Instruct-Q4_K_M.gguf", QWEN3VL_8B_GGUF[0])


def test_a_quantised_gguf_in_the_repository_itself_wins():
    """The model's own repository ships a runnable encoder: it is used."""
    files = [
        "qwen-image-2.1-UC-Q4_K_M.gguf",
        "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
        "text_encoders/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        "vae/qwen_image_2.1_vae_bf16.safetensors",
    ]

    assert resolve_llm(files, REPO) == ("text_encoders/Qwen3VL-8B-Instruct-Q4_K_M.gguf", REPO)


def test_another_model_family_keeps_its_full_precision_encoder(caplog):
    """No companion for, say, a Flux repository: its bf16 encoder is kept."""
    files = [
        "flux-3-image-klein-UD-Q4_K_XL.gguf",
        "t5xxl_fp16.safetensors",
        "ae.safetensors",
    ]

    with caplog.at_level(logging.WARNING):
        llm, llm_repo = resolve_llm(files, "unsloth/flux-3-image-klein-GGUF")

    assert llm == "t5xxl_fp16.safetensors"
    assert llm_repo == "unsloth/flux-3-image-klein-GGUF"
    assert "full precision" in caplog.text
    assert "no kernel" in caplog.text


def test_a_repository_with_no_encoder_at_all_is_left_without_one():
    """Nothing encoder-shaped and no companion: the absence is honest."""
    files = ["qwen-image-2.1-UC-Q4_K_M.gguf", "vae/qwen_image_2.1_vae_bf16.safetensors"]

    assert resolve_llm(files, REPO) == (None, REPO)
