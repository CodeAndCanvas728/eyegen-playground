# EyeGen Playground

Local image generation on Apple Silicon — runs entirely offline, no API key needed.  
Comes with a native macOS GUI and a full-featured CLI.

Three generation backends:
- **MLX (diffusionkit)** — Apple Silicon native, SD3.5 quantized
- **MFLUX** — MLX-native FLUX, FLUX.2, Z-Image, FIBO, Qwen, SeedVR2 (20+ models)
- **OllamaDiffuser (GGUF)** — 40+ quantized models (FLUX, SDXL, SD1.5, SD3.5, PixArt-Sigma, etc.)

The backend is auto-detected from the model name, or you can choose manually.

**Requires:** Apple Silicon Mac (M1/M2/M3/M4) · 16 GB+ RAM recommended

---

## Setup (one-time)

```bash
cd mlx-sd35-workspace
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The MLX model (~3 GB) downloads automatically on first use and is cached by HuggingFace.  
MFLUX models also auto-download from HuggingFace on first use — no pull step needed.  
GGUF models are pulled separately — see [GGUF Models](#gguf-models-ollamadiffuser) below.

---

## GUI

The GUI is the primary interface. It has two modes — **Text to Image** and **Image to Image** — switchable via tabs at the top of the controls panel.

### Launch options

**As a native macOS app** (Dock / Spotlight / Finder):
```bash
./create_app.sh
```
Builds `EyeGen.app` and installs it to `~/Applications`. Launch from Finder, add it to your Dock, or find it via Spotlight (⌘Space → "EyeGen").

> **First launch:** right-click → Open to bypass Gatekeeper (one-time only for unsigned apps).  
> Re-run `create_app.sh` if you move the workspace folder.

**From the terminal:**
```bash
source venv/bin/activate
./gui.py
```

### Text to Image

The default mode. Enter a prompt (and optional negative prompt), adjust the settings below, and click **Generate**.

| Control | Description |
|---------|-------------|
| **Prompt** | What you want in the image |
| **Negative Prompt** | What to avoid (optional) |
| **Steps** | Inference steps — 20 = fast, 30 = default, 40 = best quality |
| **Guidance** | How closely to follow the prompt (1.0–15.0, default 7.5) |
| **Width / Height** | Output dimensions — presets: 512, 640, 768, 896, 1024 px |
| **Seed** | Leave blank for random; set a number to reproduce a result |
| **T5 encoder** | Better prompt understanding at the cost of slower load time (MLX only) |
| **Model** | HuggingFace model ID, MFLUX alias (dev, schnell, etc.), or OllamaDiffuser model name |
| **Pull…** | Download a GGUF model (next to Model field) |
| **Backend** | Auto (detect from model name), MLX, MFLUX, or OllamaDiffuser |
| **Quantize** | MFLUX quantization: 4-bit (recommended), 8-bit, or None (shown only for MFLUX) |
| **🔑 HuggingFace Login** | Log in to access gated models (e.g. FLUX.1-Kontext). Shows login status. |

A progress bar tracks each denoising step (MLX) or shows indeterminate progress (MFLUX/OllamaDiffuser). Generated images are saved to `outputs/` (or `~/Pictures/EyeGen/` in the .app bundle) and displayed immediately. UI settings are automatically restored on next launch.

### Image to Image

Switch to the **Image to Image** tab to restyle or modify an existing image.

| Control | Description |
|---------|-------------|
| **Input Image** | Browse for a PNG/JPG/JPEG/BMP/WEBP/TIFF file |
| **Denoise** | How much to change the image — 0.05 = barely touched, 1.0 = fully redrawn (default 0.75) |

The prompt still guides the output style. Width/Height controls are disabled in this mode — output dimensions match the input image.

> **Known limitation (MLX only):** img2img may produce output identical to the input when using the 4-bit quantized MLX model. This does not affect MFLUX or OllamaDiffuser models — img2img works correctly with those backends.

---

## CLI

```bash
source venv/bin/activate

# Basic generation (MLX backend, auto-detected)
./generate.py generate "a serene mountain landscape at sunset"

# Custom size, steps, guidance, seed
./generate.py generate "a detailed portrait" --steps 40 --guidance 8.0 --width 768 --height 1024 --seed 42

# Save to a specific path
./generate.py generate "a cat wearing sunglasses" --output ~/Desktop/cat.png

# Image to image (restyle an existing image)
./generate.py generate "watercolor painting style" --image outputs/photo.png --denoise 0.7

# Use an MFLUX model (auto-detects mflux backend)
./generate.py generate "a futuristic city" --steps 8

# Use a GGUF model (auto-detects ollamadiffuser backend)
./generate.py generate "a futuristic city" --steps 24

# Force a specific backend
./generate.py generate "a cat" --backend mflux
./generate.py generate "a cat" --backend ollamadiffuser
./generate.py generate "a cat" --backend mlx

# MFLUX with custom quantization
./generate.py generate "a cat" --backend mflux --quantize 8
```

### All `generate` options

| Flag | Short | Description |
|------|-------|-------------|
| `--steps` | | Inference steps (default: 30) |
| `--guidance` | | Guidance scale 1.0–15.0 (default: 7.5) |
| `--width` | | Output width in pixels, multiple of 8 (default: 1024) |
| `--height` | | Output height in pixels, multiple of 8 (default: 1024) |
| `--seed` | | Random seed for reproducibility |
| `--output` | `-o` | Output file path (default: `outputs/YYYYMMDD_HHMMSS.png`) |
| `--image` | `-i` | Input image path for img2img mode |
| `--denoise` | `-d` | Denoise strength for img2img, 0.05–1.0 (default: 0.75) |
| `--backend` | `-b` | `auto` (default), `mlx`, `mflux`, or `ollamadiffuser` |
| `--quantize` | `-q` | MFLUX quantization: `4` (default), `8`, or omit for full precision |

> `--width`/`--height` are ignored when `--image` is provided.  
> **Known limitation (MLX only):** img2img with the 4-bit quantized MLX model may produce output identical to the input. MFLUX and GGUF models are not affected.

### Configuration commands

```bash
./generate.py config-show                        # view current defaults
./generate.py config-set num_inference_steps 25  # change a default
./generate.py config-set guidance_scale 8.0
./generate.py config-set backend mflux           # set default backend
./generate.py config-set mflux_quantize 8        # set MFLUX quantization
./generate.py config-reset                       # restore factory defaults
```

Configurable keys: `model`, `num_inference_steps`, `guidance_scale`, `width`, `height`, `backend`, `mflux_quantize`.

### Utility commands

```bash
./generate.py list-outputs   # list all generated images
./generate.py list-models    # list MFLUX + GGUF models
./generate.py status         # system info, backends, model availability
```

### HuggingFace authentication

Some models (e.g. FLUX.1-Kontext) are gated and require a HuggingFace account.

```bash
./generate.py hf-login                   # prompts for token
./generate.py hf-login --token hf_...    # pass token directly
./generate.py hf-status                  # check login status
./generate.py hf-logout                  # remove stored token
```

Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). In the GUI, click **🔑 HuggingFace Login**.

---

## Performance

| Steps | Speed | Quality | Use case |
|-------|-------|---------|----------|
| 20 | Fast | Good | Prompt iteration |
| 30 | Normal | Great | Default — balanced |
| 40 | Slow | Excellent | Final renders |

- **MLX model size:** ~3–4 GB (cached after first download)
- **MFLUX model size:** varies (1–12 GB depending on model and quantization)
- **GGUF model size:** varies (3–6 GB typical for quantized models)
- **Per image:** ~3 MB (1024×1024 PNG)
- **T5 encoder:** disable it in the GUI for faster loads at slightly lower prompt fidelity (MLX only)

---

## MFLUX Models

MFLUX is an MLX-native implementation of state-of-the-art image generation models. It runs entirely on Apple Silicon via MLX — no torch overhead — and supports 20+ models across multiple architectures.

### Available models

| Family | Models | Typical steps |
|--------|--------|---------------|
| **FLUX.1** | dev, schnell, kontext, fill, redux, depth | 4–50 |
| **FLUX.2** | klein-4b, klein-9b, klein-base-4b, klein-base-9b | 4–12 |
| **Z-Image** | z-image, z-image-turbo | 4–20 |
| **FIBO** | fibo, fibo-lite, fibo-edit | 20–50 |
| **Qwen** | qwen-image, qwen-image-edit | 20–50 |
| **SeedVR2** | seedvr2-3b, seedvr2-7b | 20–50 |

Run `./generate.py list-models` for the full list with HuggingFace model IDs.

### Using MFLUX

```bash
# Set model to an MFLUX alias — backend auto-detects
./generate.py config-set model dev

# Or pass per-run
./generate.py generate "a cat" --backend mflux

# With 8-bit quantization instead of default 4-bit
./generate.py generate "a cat" --backend mflux --quantize 8

# FLUX.2 Klein (fast, high quality)
./generate.py config-set model flux2-klein-4b
./generate.py generate "a sunset over the ocean" --steps 8
```

In the GUI, type the model alias (e.g. `dev`, `flux2-klein-4b`, `z-image`) in the **Model** field. The backend auto-detects, or select **MFLUX** from the **Backend** dropdown. Use the **Quantize** dropdown to choose 4-bit (default), 8-bit, or full precision.

### Quantization

| Level | Memory | Speed | Quality |
|-------|--------|-------|---------|
| 4-bit | Lowest (~3 GB) | Fast | Good — recommended for 16 GB RAM |
| 8-bit | Medium (~6 GB) | Medium | Better |
| None | Highest (~12 GB) | Slow | Best — requires 32 GB+ RAM |

### Auto-download

MFLUX models download from HuggingFace automatically on first use. No pull step needed. Models are cached locally by HuggingFace for subsequent runs.

---

## GGUF Models (OllamaDiffuser)

GGUF is a quantized model format that dramatically reduces VRAM requirements. OllamaDiffuser provides access to 40+ diffusion models in GGUF format — models that aren't available through MLX/diffusionkit.

### Why GGUF?

- **More models:** FLUX, SDXL, SD1.5, SD3.5, PixArt-Sigma, Kolors, CogView4, and more
- **img2img works:** unlike MLX 4-bit quantized models, GGUF denoise/strength works correctly
- **Low VRAM:** runs on machines with as little as 4 GB VRAM

### Pulling models

```bash
# CLI
./generate.py pull flux.1-dev-gguf-q4ks
./generate.py pull stable-diffusion-xl-gguf-q4ks

# List what's available
./generate.py list-models
```

In the GUI, enter the model name in the **Model** field and click **Pull…**.

### Auto-detection

The backend is chosen automatically based on the model name:
- Contains `gguf` → OllamaDiffuser
- Matches a known MFLUX alias (dev, schnell, flux2-klein-4b, z-image, fibo, etc.) → MFLUX
- Everything else → MLX (diffusionkit)

Override with `--backend mlx`, `--backend mflux`, or `--backend ollamadiffuser` on the CLI, or use the **Backend** dropdown in the GUI.

### Using a GGUF model

```bash
# Set your default model to a GGUF model
./generate.py config-set model flux.1-dev-gguf-q4ks

# Or pass it per-run (backend auto-detects from "gguf" in the name)
./generate.py generate "a cat" --backend ollamadiffuser
```

### Common GGUF models

| Model | Pull name |
|-------|-----------|
| FLUX.1 Dev | `flux.1-dev-gguf-q4ks` |
| FLUX.1 Schnell | `flux.1-schnell` |
| SDXL | `stable-diffusion-xl-gguf-q4ks` |
| SDXL Lightning | `sdxl-lightning-gguf-q4ks` |
| SD 1.5 | `stable-diffusion-1.5` |

Run `./generate.py list-models` for the full list.

---

## File Structure

```
mlx-sd35-workspace/
├── gui.py                # PySide6 GUI (primary interface)
├── generate.py           # Typer CLI
├── core.py               # Shared generation logic
├── create_app.sh         # Builds ~/Applications/EyeGen.app
├── requirements.txt      # Python dependencies
├── config/
│   ├── config.json       # Generation defaults
│   └── gui_state.json    # GUI state (auto-saved on close)
├── outputs/              # Generated images (auto-created)
└── venv/                 # Virtual environment (after setup)
```

---

## Troubleshooting

**Model downloads every run / "not installed" error**
```bash
source venv/bin/activate   # venv must be active
```

**First run is slow**  
The model (~3 GB) is downloading and caching. Subsequent runs load from the local cache.

**"Height and width must be multiples of 8"**  
Use one of the preset values: 512, 640, 768, 896, 1024.

**Out of memory**
- Use a smaller size: `--width 512 --height 512`
- Use fewer steps: `--steps 20`
- Disable the T5 encoder in the GUI
- Use 4-bit quantization for MFLUX (`--quantize 4`)
- Restart the terminal to clear cached models

**GGUF model not found**  
Pull it first: `./generate.py pull <model-name>` or click "Pull…" in the GUI.

**"Access denied" or "gated model" error**  
Log in to HuggingFace first: `./generate.py hf-login` or click **🔑 HuggingFace Login** in the GUI. Some models (e.g. FLUX.1-Kontext) require accepting terms on the model's HuggingFace page.

**GUI errors**  
Full tracebacks are written to `~/Library/Logs/EyeGen.log`.

**Moved the workspace folder**  
Re-run `./create_app.sh` — the `.app` launcher uses an absolute path to the venv.

---

## Batch generation (CLI)

```bash
#!/bin/bash
source venv/bin/activate
prompts=("a red sunset" "a blue ocean" "a green forest")
for prompt in "${prompts[@]}"; do
  ./generate.py generate "$prompt"
done
```

---

## Model & License

**MLX backend:** Uses **[argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized](https://huggingface.co/argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized)** — a 4-bit quantized version of Stable Diffusion 3.5 Large, optimized for Apple Silicon via MLX. Licensed under [CreativeML OpenRAIL-M](https://huggingface.co/argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized/blob/main/LICENSE.md).

**MFLUX backend:** Models are downloaded from HuggingFace via the [mflux](https://github.com/filipstrand/mflux) package. FLUX models are licensed by Black Forest Labs. Other models (FIBO, Z-Image, Qwen, etc.) have their own licenses — check the model card before commercial use.

**OllamaDiffuser backend:** GGUF models are pulled from the [OllamaDiffuser registry](https://github.com/ollamadiffuser/ollamadiffuser). Each model has its own license — check the model card before commercial use.

Free for personal and research use; review license terms before any commercial application.

---

MLX docs: https://ml-explore.github.io/mlx/build/latest/  
MFLUX docs: https://github.com/filipstrand/mflux  
OllamaDiffuser docs: https://github.com/ollamadiffuser/ollamadiffuser
