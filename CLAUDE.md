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
./generate.py save-model dev --quantize 4                    # save pre-quantized MFLUX model locally
./generate.py save-model schnell -q 8 -p ~/models/schnell-8b # save to custom path
./generate.py pull flux.1-dev-gguf-q4ks                      # download a GGUF model
./generate.py list-models                                    # list MFLUX + GGUF models
./generate.py status          # check system/model status
./generate.py config-show     # view current config
./generate.py config-set num_inference_steps 25
./generate.py config-set model dev                           # switch to MFLUX model
./generate.py config-set mflux_model_path ~/models/dev-4bit  # use saved local model
./generate.py config-reset
./generate.py hf-login        # log in to HuggingFace for gated models
./generate.py hf-status       # check HF login status
./generate.py hf-logout       # remove stored HF token
```

**GUI (terminal):**
```bash
./gui.py
```

**macOS App bundle** (installs to `~/Applications/EyeGen.app`):
```bash
./create_app.sh
```

## Architecture

Three Python files share a clean separation of concerns:

- **`core.py`** — all shared logic: config loading/saving, backend detection (`detect_backend`), pipeline construction for MLX (`get_pipeline`), MFLUX (`get_mflux_pipeline`), and OllamaDiffuser (`get_ollama_pipeline`), unified image generation dispatcher (`generate_image`), model pulling (`pull_model`), model saving (`save_mflux_model`, `validate_saved_model`), model listing (`list_mflux_models`, `list_ollama_models`), HuggingFace authentication (`hf_login`, `hf_status`, `hf_logout`), dimension validation, and Unicode prompt sanitization. Both CLI and GUI import from here exclusively.
- **`generate.py`** — Typer-based CLI. Thin wrapper around `core.py` functions. Commands: `generate`, `pull`, `save-model`, `list-models`, `config-show`, `config-set`, `config-reset`, `list-outputs`, `status`, `clear-cache`, `hf-login`, `hf-status`, `hf-logout`.
- **`gui.py`** — PySide6 GUI. Runs generation in a `QThread` (`GenerationWorker`), model pulling in a `PullWorker` thread, and model saving in a `SaveModelWorker` thread. Includes `HFLoginDialog` for HuggingFace authentication. Persists UI state to `config/gui_state.json` on close and restores it on next launch. Monkey-patches `diffusionkit.mlx.sample_euler` to inject a per-step progress callback (MLX only).

### Three-backend system

Three generation backends are supported:
- **MLX (diffusionkit)** — `get_pipeline()` + `_generate_image_mlx()`. Apple Silicon native. Default for SD3.5.
- **MFLUX** — `get_mflux_pipeline()` + `_generate_image_mflux()`. MLX-native FLUX ecosystem (20+ models). Uses `_resolve_mflux_class()` to map model aliases to the correct class (Flux1, Flux2Klein, ZImage, Fibo, QwenImage, SeedVR2).
- **OllamaDiffuser (GGUF)** — `get_ollama_pipeline()` + `_generate_image_ollama()`. 40+ quantized models.

`detect_backend(model, override)` resolves "auto" → concrete backend. Rule: "gguf" in model name → ollamadiffuser; model matches _mflux_aliases → mflux; else → mlx. Manual override always wins.

`generate_image()` is the unified dispatcher — it accepts a `backend` parameter and routes to the appropriate internal function. For MFLUX, it also accepts `mflux_quantize` (int or None).

### Path handling (bundled vs. dev)

`core.py` detects `sys.frozen == 'macosx_app'` (set by py2app) to switch between two path modes:
- **Dev/CLI**: config → `config/config.json`, outputs → `outputs/`, saved models → `models/`
- **Bundled .app**: config → `~/Library/Application Support/EyeGen/`, outputs → `~/Pictures/EyeGen/`, saved models → `~/Library/Application Support/EyeGen/models/`

### MLX compatibility shim

`core.py:_patch_mlx_attention()` strips the `memory_efficient_threshold` kwarg from `mlx.core.fast.scaled_dot_product_attention` before each pipeline load. This exists because newer MLX versions removed that parameter but `diffusionkit` still passes it.

### Key constraints

- Image dimensions must be multiples of 8
- Requires Apple Silicon (arm64); `create_app.sh` explicitly runs `arch -arm64` in the launcher to prevent Rosetta issues
- Default MLX model: `argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized` (~3GB, downloaded on first run and cached by HuggingFace)
- MFLUX models auto-download from HuggingFace on first use (no pull needed). Runtime quantization: 4-bit (default), 8-bit, or full precision. Config key: `mflux_quantize`.
- MFLUX models can be pre-quantized and saved locally with `save-model` to avoid re-downloading. Config key: `mflux_model_path`. Saved models include weights + tokenizers and load with no network access. The `validate_saved_model()` function checks directory integrity and reads quantization metadata from safetensors headers. When a valid saved model path is entered in the GUI, the **Model** and **Quantize** fields are automatically disabled (greyed out) because the saved model's architecture and quantization level take precedence over those values.
- The HuggingFace model cache directory can be customized via config key `hf_cache_dir` (default: `None` = `~/.cache/huggingface/hub`). When set, `_apply_hf_cache(config)` sets the `HF_HUB_CACHE` env var before any pipeline load or download. Exposed in the GUI as "HF Cache Dir" and configurable via `./generate.py config-set hf_cache_dir /path`.
- GGUF models must be pulled before use: `./generate.py pull <model-name>`
- T5 encoder can be disabled in the GUI for faster (lower quality) generation (MLX only, not applicable to MFLUX or GGUF)
- img2img with MLX 4-bit quantized models is known to produce output identical to the input; MFLUX and GGUF models are not affected
- Gated models (e.g. FLUX.1-Kontext) require HuggingFace login: `./generate.py hf-login` or the GUI login button. Uses `huggingface_hub.login()` to store token in `~/.cache/huggingface/token`.
- GUI errors write full tracebacks to `~/Library/Logs/EyeGen.log`
