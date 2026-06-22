# EyeGen Playground

Local image generation on Apple Silicon — runs entirely offline, no API key needed.  
Comes with a native macOS GUI and a full-featured CLI.

Five generation backends:
- **MLX (diffusionkit)** — Apple Silicon native, SD3.5 quantized
- **MFLUX** — MLX-native FLUX, FLUX.2, Z-Image, FIBO, Qwen, SeedVR2 (20+ models)
- **OllamaDiffuser (GGUF)** — 40+ quantized models (FLUX, SDXL, SD1.5, SD3.5, PixArt-Sigma, etc.)
- **Bonsai (PrismML)** — 1.58-bit ternary + 1-bit binary FLUX.2 Klein 4B for Apple Silicon (third-party, opt-in)
- **CoreML (Apple Neural Engine)** — SD 1.x/2.x via Apple's `python_coreml_stable_diffusion` (opt-in, sidecar venv)

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

# Use a Bonsai model (auto-detects bonsai backend, requires setup-bonsai)
./generate.py generate "a tiny bonsai tree" --steps 4

# Use a CoreML model (auto-detects coreml, requires setup-coreml)
./generate.py generate "a photo of a cat" --steps 20

# Force a specific backend
./generate.py generate "a cat" --backend mflux
./generate.py generate "a cat" --backend ollamadiffuser
./generate.py generate "a cat" --backend mlx
./generate.py generate "a cat" --backend bonsai
./generate.py generate "a cat" --backend coreml

# MFLUX with custom quantization
./generate.py generate "a cat" --backend mflux --quantize 8
```

### All `generate` options

| Flag | Short | Description |
|------|-------|-------------|
| `--steps` | | Inference steps (default: 30, 4 for Bonsai) |
| `--guidance` | | Guidance scale 1.0–15.0 (default: 7.5, 1.0 for Bonsai) |
| `--width` | | Output width in pixels, multiple of 8 (MLX/CoreML/GGUF) or 32 (Bonsai) |
| `--height` | | Output height in pixels, multiple of 8 (MLX/CoreML/GGUF) or 32 (Bonsai) |
| `--seed` | | Random seed for reproducibility |
| `--output` | `-o` | Output file path (default: `outputs/YYYYMMDD_HHMMSS.png`) |
| `--image` | `-i` | Input image path for img2img mode (not supported by Bonsai or CoreML) |
| `--denoise` | `-d` | Denoise strength for img2img, 0.05–1.0 (default: 0.75) |
| `--backend` | `-b` | `auto` (default), `mlx`, `mflux`, `ollamadiffuser`, `bonsai`, or `coreml` |
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

## Bonsai Models (PrismML ternary 1.58-bit)

Bonsai is a third-party backend by [Prism ML](https://huggingface.co/prism-ml) that runs
[FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) at extremely
small footprint via 1.58-bit ternary weights with custom MLX kernels. The 4B model
fits in 1.21 GB instead of 7.75 GB (FP16), making it runnable on M1/M2/M3 with
low memory pressure.

**Bonsai is opt-in** because it uses a patched mflux and MLX from
`PrismML-Eng/mflux-prism` and `PrismML-Eng/mlx` that conflict with the
upstream `mflux` and `mlx` EyeGen uses for the MFLUX and MLX backends.
EyeGen shells out to the bonsai-demo's own Python 3.11 venv to keep that
isolation clean.

### One-time setup

```bash
./generate.py setup-bonsai    # clones Bonsai-Image-Demo to ~/models/eyegen/bonsai-demo/
                              # and runs its setup.sh (~3-5 min, installs Py 3.11 venv)
./generate.py pull-bonsai     # downloads the ternary-mlx model (~1.2 GB)
```

You can also use the GUI: select **Bonsai** in the Backend dropdown → click
**Setup Bonsai…** → click **Download Model…**.

### Using Bonsai

```bash
./generate.py generate "a tiny bonsai tree in a quiet ceramic studio" \
    --model bonsai-ternary-mlx --steps 4
```

Or type `bonsai-ternary-mlx` (or `bonsai-image-4B-ternary-mlx`, or
`prism-ml/bonsai-image-ternary-4B-mlx-2bit`) in the **Model** field of the
GUI and select **Bonsai** in the Backend dropdown. Backend auto-detects
from the model name.

### Bonsai constraints

- **Fixed sampler**: 4 steps, `guidance=1.0`, `shift=3.0`. No CFG, no
  negative prompt, no img2img. The GUI grays out img2img / negative prompt
  / denoise when Bonsai is the resolved backend.
- **Dimensions**: must be multiples of 32 (e.g. 512×512, 1024×1024,
  1248×832, 832×1248).
- **Cold-start**: each call pays ~5s of imports + weight load on M-series.
  Subsequent calls at the same shape benefit from the MLX metallib cache.
- **License**: Apache 2.0. The 4B backbone is FLUX.2 Klein 4B — check
  the model card before commercial use.

---

## CoreML Models (Apple Neural Engine)

The CoreML backend runs Stable Diffusion 1.x / 2.x models via Apple's
[`python_coreml_stable_diffusion`](https://github.com/apple/ml-stable-diffusion)
on the Apple Neural Engine (ANE) — fast and power-efficient on M-series chips.

**CoreML is opt-in** because Apple's package pins an older dependency set
(`diffusers==0.30.2`, `transformers==4.44.2`, `numpy<1.24`,
`diffusionkit==0.4.0`) that's incompatible with EyeGen's main Python 3.14
venv. EyeGen installs Apple's package in a sidecar Python 3.11 venv at
`~/models/eyegen/.coreml-venv/` and shells out to it.

### One-time setup

```bash
brew install python@3.11      # if not already installed
./generate.py setup-coreml    # creates the sidecar venv + installs Apple's package (~3-5 min)
```

You can also use the GUI: select **CoreML** in the Backend dropdown → click
**Setup CoreML…**.

### Pull a pre-converted model (fast)

Apple has pre-converted Stable Diffusion models on Hugging Face. Pulling
these is the fastest path:

```bash
./generate.py pull-coreml                          # default: sd-2-1-base-palettized
./generate.py pull-coreml sd-1-5-palettized
./generate.py pull-coreml apple/coreml-stable-diffusion-v1-4
```

Or in the GUI: select **CoreML** → **Download Model…** → pick from a list.

### Convert a PyTorch model (15-20 min on M1 Pro)

```bash
./generate.py convert-coreml stabilityai/stable-diffusion-2-1-base
./generate.py convert-coreml stabilityai/stable-diffusion-2-1-base --quantize-nbits 6
```

Run `./generate.py list-coreml-models` to see installed models.

### CoreML constraints

- **SD 1.x/2.x only**. SD3 and FLUX have CoreML conversions in Apple's
  package but require more memory; not first-class supported here.
- **Image dimensions**: should be 512×512 and multiples of 8. Larger
  sizes are technically supported but require custom conversion.
- **No img2img** in this wrapper (would need VAE encoder conversion —
  out of scope for first cut). The GUI switches back to txt2img if you
  select CoreML while on the img2img tab.
- **First call** at a new shape pays the CoreML compile cost (~5-30s).
  Subsequent calls are fast.

---

## File Structure

```
mlx-sd35-workspace/
├── gui.py                # PySide6 GUI (primary interface)
├── generate.py           # Typer CLI
├── core.py               # Shared generation logic (5-backend dispatcher)
├── core_bonsai.py        # Bonsai (PrismML) backend wrapper
├── core_coreml.py        # CoreML (Apple Neural Engine) backend wrapper
├── create_app.sh         # Builds ~/Applications/EyeGen.app
├── scripts/
│   ├── setup-bonsai.sh   # One-time installer for Bonsai
│   └── setup-coreml.sh   # One-time installer for CoreML sidecar venv
├── requirements.txt      # Python dependencies
├── config/
│   ├── config.json       # Generation defaults
│   └── gui_state.json    # GUI state (auto-saved on close)
├── outputs/              # Generated images (auto-created)
└── venv/                 # Virtual environment (after setup)

~/models/                 # Unified model artifact tree
├── .hf-cache/hub/        # HuggingFace download cache (HF_HUB_CACHE)
└── eyegen/               # All EyeGen-specific artifacts
    ├── saved-mflux/      # Output of ./generate.py save-model
    ├── bonsai-demo/      # Bonsai-Image-Demo vendor (created by setup-bonsai)
    │   ├── .venv/        # Py 3.11 venv with patched mflux-prism + mlx
    │   └── models/       # Downloaded bonsai models
    ├── coreml/           # Downloaded/converted CoreML model bundles
    └── .coreml-venv/     # Py 3.11 sidecar venv with python_coreml_stable_diffusion
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

**Bonsai (PrismML) backend:** Models from [prism-ml/bonsai-image-*](https://huggingface.co/prism-ml) — Apache 2.0. The base architecture is FLUX.2 Klein 4B (Black Forest Labs license). Review the model card on Hugging Face before commercial use.

**CoreML (Apple Neural Engine) backend:** Uses Apple's [python_coreml_stable_diffusion](https://github.com/apple/ml-stable-diffusion) (Apple Inc. license). Underlying Stable Diffusion models have their own licenses — see the model card for each (CompVis SD 1.4/1.5: CreativeML OpenRAIL-M; Stability AI SD 2.x: their own terms).

Free for personal and research use; review license terms before any commercial application.

---

MLX docs: https://ml-explore.github.io/mlx/build/latest/  
MFLUX docs: https://github.com/filipstrand/mflux  
Bonsai demo: https://github.com/PrismML-Eng/Bonsai-Image-Demo  
CoreML Stable Diffusion: https://github.com/apple/ml-stable-diffusion
OllamaDiffuser docs: https://github.com/ollamadiffuser/ollamadiffuser
