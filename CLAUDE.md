# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Always activate the venv before running any Python scripts: `source venv/bin/activate`

## Running

**CLI:**
```bash
./generate.py generate "your prompt here"
./generate.py generate "prompt" --steps 20 --guidance 7.5 --width 1024 --height 1024 --seed 42
./generate.py generate "prompt" --backend mflux              # force MFLUX backend
./generate.py generate "prompt" --backend mflux --quantize 8 # MFLUX with 8-bit quantization
./generate.py generate "prompt" --backend ollamadiffuser     # force GGUF backend
./generate.py generate "prompt" --backend bonsai             # force Bonsai (PrismML) backend
./generate.py generate "prompt" --backend coreml             # force CoreML (Apple Neural Engine) backend
./generate.py save-model dev --quantize 4                    # save pre-quantized MFLUX model locally
./generate.py save-model schnell -q 8 -p ~/models/schnell-8b # save to custom path
./generate.py pull flux.1-dev-gguf-q4ks                      # download a GGUF model
./generate.py list-models                                    # list MFLUX + GGUF + Bonsai + CoreML models
./generate.py status          # check system/model status
./generate.py config-show     # view current config
./generate.py config-set num_inference_steps 25
./generate.py config-set model dev                           # switch to MFLUX model
./generate.py config-set mflux_model_path ~/models/eyegen/saved-mflux/dev-4bit  # use saved local model
./generate.py config-reset
./generate.py hf-login        # log in to HuggingFace for gated models
./generate.py hf-status       # check HF login status
./generate.py hf-logout       # remove stored HF token

# Bonsai (PrismML ternary 1.58-bit)
./generate.py setup-bonsai              # one-time install (clones bonsai-demo, runs its setup.sh)
./generate.py pull-bonsai               # download ternary-mlx (default)
./generate.py pull-bonsai -v binary-mlx # download binary 1-bit variant
./generate.py list-bonsai-models

# CoreML (Apple Neural Engine)
./generate.py setup-coreml                                # one-time install (creates sidecar Py 3.11 venv)
./generate.py pull-coreml                                 # download pre-converted sd-2-1-base-palettized (default)
./generate.py pull-coreml apple/coreml-stable-diffusion-v1-5
./generate.py convert-coreml stabilityai/stable-diffusion-2-1-base           # convert from PyTorch (15-20 min)
./generate.py list-coreml-models
```

**GUI (terminal):**
```bash
./gui.py
```

**macOS App bundle** (installs to `~/Applications/EyeGen.app`):
```bash
./create_app.sh
```

The .app bundle is a launcher for the workspace's `gui.py` + venv. Bonsai and CoreML work in the .app because their deps live in the venv (`core_bonsai.py` shells out to `~/models/eyegen/bonsai-demo/`; `core_coreml.py` shells out to `~/models/eyegen/.coreml-venv/`).

## Architecture

Five Python files share a clean separation of concerns:

- **`core.py`** — all shared logic: config loading/saving, backend detection (`detect_backend`), pipeline construction for MLX (`get_pipeline`), MFLUX (`get_mflux_pipeline`), and OllamaDiffuser (`get_ollama_pipeline`), unified image generation dispatcher (`generate_image`), model pulling (`pull_model`), model saving (`save_mflux_model`, `validate_saved_model`), model listing (`list_mflux_models`, `list_ollama_models`), HuggingFace authentication (`hf_login`, `hf_status`, `hf_logout`), dimension validation, and Unicode prompt sanitization. Both CLI and GUI import from here exclusively.
- **`core_bonsai.py`** — Bonsai (PrismML) backend. Subprocess wrapper around the bonsai-demo's `scripts/generate.sh` (no direct MLX kernel imports — the bonsai-demo's own venv has the patched mflux + MLX). Provides `BonsaiWrapper` exposing the standard `generate_image()` interface, plus `validate_bonsai_install`, `list_bonsai_models`, `download_bonsai_model`, `get_bonsai_dir`.
- **`core_coreml.py`** — CoreML (Apple Neural Engine) backend. Subprocess wrapper around Apple's `python_coreml_stable_diffusion` in a sidecar Py 3.11 venv at `~/models/eyegen/.coreml-venv/`. Provides `CoreMLWrapper` exposing the standard `generate_image()` interface, plus `validate_coreml_install`, `list_coreml_models`, `pull_preconverted_coreml_model`, `convert_to_coreml`.
- **`generate.py`** — Typer-based CLI. Thin wrapper around `core.py` functions. Commands: `generate`, `pull`, `save-model`, `list-models`, `config-show`, `config-set`, `config-reset`, `list-outputs`, `status`, `clear-cache`, `hf-login`, `hf-status`, `hf-logout`, `setup-bonsai`, `pull-bonsai`, `list-bonsai-models`, `setup-coreml`, `pull-coreml`, `convert-coreml`, `list-coreml-models`.
- **`gui.py`** — PySide6 GUI. Runs generation in a `QThread` (`GenerationWorker`), model pulling in a `PullWorker` thread, model saving in a `SaveModelWorker` thread, and bonsai/coreml setup + download in dedicated `BonsaiSetupWorker`/`BonsaiDownloadWorker`/`CoreMLSetupWorker`/`CoreMLDownloadWorker` threads. Includes `HFLoginDialog` for HuggingFace authentication. Persists UI state to `config/gui_state.json` on close and restores it on next launch. Monkey-patches `diffusionkit.mlx.sample_euler` to inject a per-step progress callback (MLX only).

### Five-backend system

Five generation backends are supported:
- **MLX (diffusionkit)** — `get_pipeline()` + `_generate_image_mlx()`. Apple Silicon native. Default for SD3.5.
- **MFLUX** — `get_mflux_pipeline()` + `_generate_image_mflux()`. MLX-native FLUX ecosystem (20+ models). Uses `_resolve_mflux_class()` to map model aliases to the correct class (Flux1, Flux2Klein, ZImage, Fibo, QwenImage, SeedVR2).
- **OllamaDiffuser (GGUF)** — `get_ollama_pipeline()` + `_generate_image_ollama()`. 40+ quantized models.
- **Bonsai (PrismML)** — `core_bonsai.get_bonsai_pipeline()`. 1.58-bit ternary + 1-bit binary FLUX.2 Klein 4B for Apple Silicon. Subprocess wrapper around `~/models/eyegen/bonsai-demo/scripts/generate.sh`. Fixed `guidance=1.0`, no CFG, no negative prompt, no img2img. Dimensions must be multiples of 32.
- **CoreML (Apple Neural Engine)** — `core_coreml.get_coreml_pipeline()`. SD 1.x/2.x via Apple's `python_coreml_stable_diffusion` in a sidecar Py 3.11 venv. Supports `CPU_AND_NE`, `CPU_AND_GPU`, `CPU_ONLY`, `ALL` compute units. No img2img (would need VAE encoder conversion). Dimensions must be multiples of 8.

`detect_backend(model, override, config)` resolves "auto" → concrete backend. Rules (in order):
1. `"gguf"` in model name → ollamadiffuser
2. bonsai identifier (`bonsai-*`, `bonsai-image-4B-*`, `prism-ml/bonsai-*`) → bonsai
3. coreml identifier (known alias, `apple/coreml-stable-diffusion-*`, or `coreml_model_path` set) → coreml
4. model matches MFLUX aliases → mflux
5. model is in `MMDIT_CKPT` (diffusionkit) → mlx
6. otherwise → `ValueError` listing supported models per backend (defensive — `core.py:_format_unsupported_error`)

Manual override always wins.

`generate_image()` is the unified dispatcher — it accepts a `backend` parameter and routes to the appropriate internal function. For MFLUX, it also accepts `mflux_quantize` (int or None). For bonsai/coreml, the pipeline object is itself the wrapper, so `generate_image()` calls `pipeline.generate_image()`.

### Path layout (unified)

All model artifacts live under `~/models/`:
- `~/models/.hf-cache/hub/` — HF_HUB_CACHE for HF downloads (default; overridable via `hf_cache_dir` config or `HF_HUB_CACHE` env var)
- `~/models/eyegen/` — non-HF-cached artifacts:
  - `saved-mflux/` — output of `./generate.py save-model` (renamed from project-local `models/`)
  - `bonsai-demo/` — bonsai-demo vendor checkout (created by `./scripts/setup-bonsai.sh`)
  - `coreml/` — converted/downloaded CoreML model bundles (created on first pull)
  - `.coreml-venv/` — sidecar Py 3.11 venv with `python_coreml_stable_diffusion` (created by `./scripts/setup-coreml.sh`)

`core.py:23-47` sets `MODELS_DIR = ~/models/eyegen` and `HF_CACHE_DIR = ~/models/.hf-cache/hub` in both dev and bundled modes. `_apply_hf_cache()` (core.py:325) respects priority: explicit `config["hf_cache_dir"]` > `$HF_HUB_CACHE` env var > `HF_CACHE_DIR` default.

Config and output paths retain the dev/bundled split:
- **Dev/CLI**: config → `config/config.json`, outputs → `outputs/`
- **Bundled .app**: config → `~/Library/Application Support/EyeGen/`, outputs → `~/Pictures/EyeGen/`

### MLX compatibility shim

`core.py:_patch_mlx_attention()` strips the `memory_efficient_threshold` kwarg from `mlx.core.fast.scaled_dot_product_attention` before each pipeline load. This exists because newer MLX versions removed that parameter but `diffusionkit` still passes it.

### Key constraints

- Image dimensions must be multiples of 8 (MLX/CoreML/GGUF) or 32 (Bonsai)
- Requires Apple Silicon (arm64); `create_app.sh` explicitly runs `arch -arm64` in the launcher to prevent Rosetta issues
- Default MLX model: `argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized` (~3GB, downloaded on first run and cached by HuggingFace)
- MFLUX models auto-download from HuggingFace on first use (no pull needed). Runtime quantization: 4-bit (default), 8-bit, or full precision. Config key: `mflux_quantize`.
- MFLUX models can be pre-quantized and saved locally with `save-model` to avoid re-downloading. Config key: `mflux_model_path`. Saved models include weights + tokenizers and load with no network access. The `validate_saved_model()` function checks directory integrity and reads quantization metadata from safetensors headers. When a valid saved model path is entered in the GUI, the **Model** and **Quantize** fields are automatically disabled (greyed out) because the saved model's architecture and quantization level take precedence over those values.
- The HuggingFace model cache directory can be customized via config key `hf_cache_dir` (default: `~/models/.hf-cache/hub/`). When set, `_apply_hf_cache(config)` sets the `HF_HUB_CACHE` env var before any pipeline load or download. Exposed in the GUI as "HF Cache Dir" and configurable via `./generate.py config-set hf_cache_dir /path`.
- GGUF models must be pulled before use: `./generate.py pull <model-name>`
- T5 encoder can be disabled in the GUI for faster (lower quality) generation (MLX only, not applicable to MFLUX/GGUF/Bonsai/CoreML)
- img2img with MLX 4-bit quantized models is known to produce output identical to the input; MFLUX and GGUF models are not affected. Bonsai and CoreML SD 1.x/2.x do not support img2img in this wrapper.
- Gated models (e.g. FLUX.1-Kontext) require HuggingFace login: `./generate.py hf-login` or the GUI login button. Uses `huggingface_hub.login()` to store token in `~/.cache/huggingface/token`.
- GUI errors write full tracebacks to `~/Library/Logs/EyeGen.log`

### Bonsai (PrismML) — opt-in, third-party

- Vendor: `PrismML-Eng/Bonsai-Image-Demo` (Apache 2.0). Installed at `~/models/eyegen/bonsai-demo/` by `scripts/setup-bonsai.sh` (one-time).
- Uses the bonsai-demo's own Py 3.11 venv at `~/models/eyegen/bonsai-demo/.venv/` with the patched `mflux-prism` + `mlx` binaries. **Do not** try to install bonsai's kernels into EyeGen's main venv — version conflicts with upstream `mflux` are guaranteed.
- EyeGen's `core_bonsai.py` shells out to `~/models/eyegen/bonsai-demo/scripts/generate.sh` per generation. Cold-start (~5s on M-series) is paid each call.
- `bonsai-ternary-mlx` (1.58-bit, 1.21 GB) and `bonsai-binary-mlx` (1-bit, 0.93 GB) variants. Other bonsai variants (gemlite) need Linux.
- Fixed sampler: 4 steps, `guidance=1.0`, `shift=3.0`, no negative prompt, no img2img. The GUI grays out img2img when bonsai is the resolved backend.

### CoreML (Apple Neural Engine) — opt-in, sidecar venv

- Vendor: Apple's `python_coreml_stable_diffusion` (Apple Inc. license). Installed in a sidecar Py 3.11 venv at `~/models/eyegen/.coreml-venv/` by `scripts/setup-coreml.sh` (one-time).
- Sidecar venv exists because Apple's package pins `diffusers==0.30.2`, `transformers==4.44.2`, `huggingface-hub==0.24.6`, `numpy<1.24`, `diffusionkit==0.4.0` — incompatible with EyeGen's Py 3.14 venv.
- Requires `/opt/homebrew/bin/python3.11` (set `PYTHON_BIN` to override). Will not work on Py 3.12+.
- Supports SD 1.x/2.x via pre-converted HF models (`apple/coreml-stable-diffusion-*`) or convert from scratch (`./generate.py convert-coreml`). Conversion is 15-20 min on M1 Pro.
- `coreml_compute_unit` config key: `CPU_AND_NE` (default, mobile + ANE), `CPU_AND_GPU` (Mac), `CPU_ONLY`, `ALL`.
- CoreML SD 1.x/2.x does not support img2img in this wrapper (would require converting a VAE encoder — out of scope for first cut).

@AGENTS.md
