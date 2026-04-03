# Copilot Instructions

## Project Overview

Single-file CLI tool (`generate.py`) and native PySide6 GUI (`gui.py`) for local image generation on Apple Silicon. Named **EyeGen**. Built with Typer for the CLI, PySide6 for the desktop GUI, diffusionkit for the ML pipeline, and MLX as the compute backend.

## Setup & Commands

```bash
# First-time setup
./setup.sh              # or: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Activate environment (required before any command)
source venv/bin/activate

# Build the macOS app bundle (installs to ~/Applications)
./create_app.sh         # re-run if you move the workspace folder

# Launch the GUI from terminal
./gui.py                # or: python gui.py

# Generate an image (CLI)
./generate.py generate "prompt" --steps 30 --guidance 7.5 --width 1024 --height 1024 --seed 42

# Config management
./generate.py config-show
./generate.py config-set num_inference_steps 25
./generate.py config-reset

# Utilities
./generate.py list-outputs
./generate.py status
```

No tests or linter are configured.

## Architecture

Shared generation logic lives in `core.py`, consumed by both entry points:

- `core.py` — config loading/saving, MLX attention patching, pipeline initialization, prompt sanitization, dimension validation, image generation
- `generate.py` — Typer CLI with subcommands: `generate`, `config-show`, `config-set`, `config-reset`, `list-outputs`, `status`
- `gui.py` — PySide6 desktop app with controls panel (prompt, sliders, toggles, dropdowns) and image preview. Uses `QThread` (`GenerationWorker`) to run inference off the main thread.

- **Config**: `config/config.json` stores defaults (model, steps, guidance, dimensions). CLI flags override config values per-run. `load_config()` falls back to hardcoded defaults if the file is missing.
- **Pipeline**: `get_pipeline()` lazy-loads `diffusionkit.mlx.DiffusionPipeline` only when generating. It applies `_patch_mlx_attention()` first to strip the removed `memory_efficient_threshold` kwarg from newer MLX versions.
- **Output**: Images go to `outputs/YYYYMMDD_HHMMSS.png` by default.

## Key Conventions

- **Prompt sanitization**: `sanitize_prompt()` in `core.py` replaces Unicode punctuation (em/en dashes, smart quotes, ellipsis) with ASCII equivalents before passing to the T5 tokenizer.
- **Dimension validation**: Width and height must be multiples of 8. Latent size is passed as `(h // 8, w // 8)`. The GUI enforces this via preset dropdowns; the CLI validates with `validate_dimensions()`.
- **Compatibility shim**: `_patch_mlx_attention()` uses a blacklist approach (only strips `memory_efficient_threshold`) rather than signature introspection, because `inspect.signature()` on MLX C extensions is unreliable.
- **App bundle**: `create_app.sh` builds a minimal macOS `.app` (shell launcher + `Info.plist`) installed to `~/Applications/EyeGen.app`. No Python libraries are bundled — the venv is used directly. Re-run if the workspace is moved. `sys.frozen = 'macosx_app'` is set in this context, which `core.py` uses to redirect paths.
- **GUI threading**: Image generation runs in a `GenerationWorker(QThread)` that emits `status`, `finished`, and `error` signals. Never call the pipeline from the main Qt thread.
- **Requires Apple Silicon**: MLX only runs on M-series Macs. 16 GB+ RAM recommended for the quantized model (~3-4 GB).
