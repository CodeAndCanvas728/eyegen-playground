---
title: "int8_convrot has no Metal kernel; the runnable Qwen3-VL-8B encoder is the Qwen GGUF companion"
author: CodeAndCanvas728
date: 2026-10-29
id: l-int8-convrot-metal
scope: repo
status: active
src: agent
touches:
  - eyegen/backends/sdcli/models.py
  - eyegen/backends/sdcli/pick.py
  - tests/test_backends_sdcli_pick.py
---

sd.cpp's int8 tensorwise/convrot weights (e.g. `text_encoders/qwen3vl_8b_int8_convrot.safetensors` in abenzerps/Qwen-Image-2.1-Uncensored-Quantized) have **no Metal kernel** in sd.cpp's patched ggml — `docs/int8_convrot.md` lists CPU/CUDA/HIP/Vulkan only. On a Mac the run aborts mid-text-encoding with `kernel_mul_mm_i8_i8 ... kernel not found in any metal library` (exit -6). Verified against upstream leejet/ggml master: no i8×i8 mul_mm kernel exists, so updating sd-cli does NOT help. The auto-completion's slimness ranking (`int8` outranks bf16 and GGUF) picked exactly this file. Fix: int8/convrot are excluded for the `llm` role (`UNRUNNABLE_HINTS` in `eyegen/backends/sdcli/pick.py`), and when a repository ships no runnable encoder, the documented companion from sd.cpp's own `docs/qwen_image_2.1.md` is used instead: `Qwen3VL-8B-Instruct-Q4_K_M.gguf` from Qwen/Qwen3-VL-8B-Instruct-GGUF (~5 GB, runs natively on Metal; the bf16 encoder is 17.5 GB — worse than useless on a 16 GB Mac). The same crash would hit an int8_convrot *diffusion* model; not guarded yet (would need the same treatment for the transformer role).
