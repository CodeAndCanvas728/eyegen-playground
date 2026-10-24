# EyeGen Playground

Local image generation on Apple Silicon — runs entirely offline, no API key needed.
Comes with a native macOS GUI and a full-featured CLI.

Six generation backends:
- **MLX (diffusionkit)** — Apple Silicon native, SD3.5 quantized
- **MFLUX** — MLX-native FLUX, FLUX.2, FLUX.3, Z-Image, FIBO, Qwen, Krea, Ideogram, SeedVR2 (40+ models)
- **OllamaDiffuser (GGUF)** — 40+ quantized models (FLUX, SDXL, SD1.5, SD3.5, PixArt-Sigma, etc.)
- **Bonsai (PrismML)** — 1.58-bit ternary + 1-bit binary FLUX.2 Klein 4B for Apple Silicon (third-party, opt-in)
- **CoreML (Apple Neural Engine)** — SD 1.x/2.x via Apple's `python_coreml_stable_diffusion` (opt-in, sidecar venv)
- **stable-diffusion.cpp** — FLUX.3 Image Klein as a GGUF transformer, through the `sd-cli` binary (opt-in)

The backend is auto-detected from the model name, or you can choose manually.

A model the tool does not know can be described in `models.json` and run like
any other — see [Adding a model the tool does not know](#adding-a-model-the-tool-does-not-know).

**Requires:** Apple Silicon Mac (M1/M2/M3/M4) · 16 GB+ RAM recommended

---

## Setup (one-time)

```bash
cd mygen-playground
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The MLX model (~3 GB) downloads automatically on first use and is cached by HuggingFace.
MFLUX models also auto-download from HuggingFace on first use — no pull step needed.
GGUF models are pulled separately — see [GGUF Models](#gguf-models-ollamadiffuser) below.

### Or install with Homebrew

```bash
brew tap CodeAndCanvas728/eyegen
brew install eyegen
```

The formula puts the CLI in a virtual environment in `~/.local/share/eyegen/venv`, not
in the Cellar, so an upgrade does not take the backends with it.

The formula arrives with the first release, because a Homebrew formula without a
checksum is not a formula: the checksum is of the tarball the release publishes, and
there is nothing to check against until there is a release.

---

## Keeping the backends up to date

The backends are Python packages, and a new model arrives when one of them is
released: FLUX.3 came with mflux 0.23. Nothing here knows that on its own, so the
tool can be asked:

```bash
./generate.py update-backends
```

```
The backends and what is out there:

  ↑ diffusionkit     0.5.1        wants ≥0.1.0      newest 0.5.2
  ✓ mlx              0.32.4       wants ≥0.32.0     newest 0.32.4
  ✓ mflux            0.23.0       wants ≥0.23.0     newest 0.23.0
  ✓ ollamadiffuser   2.5.0        wants ≥2.0.17     newest 2.5.0
  ✓ Pillow           12.3.0       wants ≥10.0.0     newest 13.0.0
```

`--apply` installs what it finds:

```bash
./generate.py update-backends --apply
```

The requirements file is what decides. It carries the floors — mflux 0.23 for
FLUX.3 — and the ceilings that come from a broken release, like Pillow under 13,
which removed the module mflux imports. The update moves within those bounds, so
it does not walk into the next broken release.

---

## GUI

The GUI is the primary interface with three tabs: **Home**, **History**, and **Settings**. Each generation session has two modes — **Text to Image** and **Image to Image** — switchable via tabs in the controls panel.

### Launch options

**As a native macOS app** (Dock / Spotlight / Finder):
```bash
./create_app.sh
```
Builds `EyeGen.app` and installs it to `~/Applications`. Launch from Finder, add it to your Dock, or find it via Spotlight (⌘Space → "EyeGen"). The app auto-detects macOS dark/light mode; switch themes in **Settings**.

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
| **Steps** | Inference steps — 20 = fast, 30 = default, 40 = best quality (collapsible Advanced Settings) |
| **Guidance** | How closely to follow the prompt (1.0–15.0, default 7.5) (collapsible Advanced Settings) |
| **Width / Height** | Output dimensions — presets: 512, 640, 768, 896, 1024 px (in Settings tab) |
| **Seed** | Leave blank for random; set a number to reproduce a result |
| **T5 encoder** | Better prompt understanding at the cost of slower load time (MLX only) (collapsible Advanced Settings) |
| **Model** | HuggingFace model ID, MFLUX alias (dev, schnell, etc.), or OllamaDiffuser model name |
| **Pull…** | Download a GGUF model (next to Model field) |
| **Backend** | Auto (detect from model name), MLX, MFLUX, or OllamaDiffuser |
| **Quantize** | MFLUX quantization: 4-bit (recommended), 8-bit, or None (in Settings tab) |
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

# Use FLUX.3 Image Klein (auto-detects sdcli, requires the sd-cli binary)
./generate.py generate "a red fox in a snow-covered forest" --steps 16 --guidance 1.0

# Force a specific backend
./generate.py generate "a cat" --backend mflux
./generate.py generate "a cat" --backend ollamadiffuser
./generate.py generate "a cat" --backend mlx
./generate.py generate "a cat" --backend bonsai
./generate.py generate "a cat" --backend coreml
./generate.py generate "a cat" --backend sdcli

# Use a model for this run only, without changing the configured default
./generate.py generate "a cat" --model flux-3-image-klein-gguf
./generate.py generate "a cat" --model my-own-model

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
| `--model` | `-m` | Model to use for this run, instead of the configured default |
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

Configurable keys: `model`, `num_inference_steps`, `guidance_scale`, `width`, `height`, `backend`, `mflux_quantize`, `hf_cache_dir`, `models_dir`.

### Utility commands

```bash
./generate.py list-outputs   # list all generated images
./generate.py list-models    # list your own models, MFLUX, GGUF and stable-diffusion.cpp models
./generate.py status         # system info, backends, model availability
./generate.py --version      # the version of the installed package
```

### Fetching a model

```bash
# A model the tool already knows
./generate.py pull flux.1-dev-gguf-q4ks
./generate.py pull flux-3-image-klein-gguf

# A model the tool does not know: a repository and its files
./generate.py pull --from-hf owner/repo --as my-model \
    --file diffusion_model=owner/repo:model.gguf

# Keep the backends up to date
./generate.py update-backends
```

See [Adding a model the tool does not know](#adding-a-model-the-tool-does-not-know) for the whole story.

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

MFLUX is an MLX-native implementation of state-of-the-art image generation models. It runs entirely on Apple Silicon via MLX — no torch overhead — and supports 40+ models across a dozen architectures.

### Available models

MFLUX 0.23 knows 43 models. The families:

| Family | Models | Typical steps |
|--------|--------|---------------|
| **FLUX.1** | dev, schnell, kontext, fill, redux, depth, controlnet-canny, controlnet-upscaler | 4–50 |
| **FLUX.2** | klein-4b, klein-9b, klein-base-4b, klein-base-9b, klein-9b-kv | 4–12 |
| **FLUX.3** | image-klein, image-klein-base | 16–50 |
| **Z-Image** | z-image, z-image-turbo | 4–20 |
| **FIBO** | fibo, fibo-lite, fibo-edit, fibo-edit-rmbg | 20–50 |
| **Qwen Image** | qwen-image, qwen-image-edit, qwen-image-2.1, qwen-image-2.1-turbo | 8–50 |
| **Krea 2** | krea-2, krea-2-raw, krea-dev | 4–28 |
| **Ideogram 4** | ideogram-4-fp8 | 20–50 |
| **ERNIE-Image** | ernie-image, ernie-image-turbo | 4–50 |
| **Boogu** | boogu-image-turbo | 4 |
| **Ming-Image** | ming-image-design | 20–50 |
| **Lens** | lens-turbo | 4 |
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

The tool works with OllamaDiffuser 2.0.17 and later, and is developed and tested against 2.5.0. `./generate.py update-backends` compares what you have with what is out there.

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

The backend is chosen automatically from the model name:

1. **Your own catalogue** (`models.json`) — a model you described runs the way you described it, whatever its name looks like.
2. The stable-diffusion.cpp catalogue.
3. Contains `gguf` → OllamaDiffuser.
4. A Bonsai identifier (`bonsai-*`, `prism-ml/bonsai-*`) → Bonsai.
5. A CoreML alias, an `apple/coreml-stable-diffusion-*` id, or `coreml_model_path` → CoreML.
6. A known MFLUX alias → MFLUX.
7. A diffusionkit checkpoint → MLX.

Override with `--backend mlx`, `--backend mflux`, `--backend ollamadiffuser`, `--backend bonsai`, `--backend coreml` or `--backend sdcli` on the CLI, or the **Backend** dropdown in the GUI. A backend that contradicts your own catalogue is an error rather than a silent choice: the model would run through a runtime that cannot read it.

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

## FLUX.3 Image Klein (stable-diffusion.cpp)

[FLUX.3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) is
Black Forest Labs' newest image model: a 5.6B transformer, distilled to answer in
16 steps. It runs here through [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp),
which is the only runtime that understands the architecture and runs on Metal.

**Why a command line rather than a Python library:** the `stable-diffusion-cpp-python`
bindings on PyPI bundle a stable-diffusion.cpp from before FLUX.3 existed, so they
cannot load the model at all. The command line is built from current upstream source,
and it is what upstream documents for this model (`docs/flux3.md`).

### One-time setup

```bash
# 1. Build the binary (a few minutes; Metal is what makes it fast)
git clone https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp
git submodule update --init --recursive
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sd-cli -j $(sysctl -n hw.ncpu)
cp build/bin/sd-cli ~/.local/bin/sd-cli

# 2. Fetch the model files (~6.7 GB)
cd -
./generate.py pull flux-3-image-klein-gguf
```

The model is three files, and the pull step fetches each from where it lives:

| file | what it is | size |
|---|---|---|
| `flux-3-image-klein-UD-Q4_K_XL.gguf` | the transformer, from [unsloth](https://huggingface.co/unsloth/flux-3-image-klein-GGUF) | 3.9 GB |
| `Qwen3VL-4B-Instruct-Q4_K_M.gguf` | the Qwen3-VL text encoder that conditions it | 2.5 GB |
| `ae.safetensors` | the FLUX.2 VAE that decodes it | 336 MB |

The VAE comes from Black Forest Labs' own repository, which asks you to
accept its terms the first time you fetch from it. So sign in once before
the pull:

```bash
./generate.py hf-login
```

The other two are not gated, and the pull works without an account for them.

### Using it

```bash
./generate.py generate "a red fox in a snow-covered forest clearing at golden hour"
```

The model is distilled, so it wants 16 steps at a guidance of 1.0. EyeGen uses those
settings unless you change them yourself — a setting you leave at the tool-wide
default is the model's to choose.

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

## Adding a model the tool does not know

Every model EyeGen runs is described by a **specification**: the backend that
runs it, the files it is made of, the repository each file comes from, and the
steps and guidance it was trained for. The catalogues that ship with the tool
are specifications too, and your own models live in `models.json` next to your
config file (`config/models.json` in a checkout, the application-support
folder for the bundled app).

### From a repository, in one command

```bash
./generate.py pull --from-hf unsloth/flux-3-image-klein-GGUF \
    --as klein-q4 \
    --file diffusion_model=unsloth/flux-3-image-klein-GGUF:flux-3-image-klein-UD-Q4_K_XL.gguf \
    --file llm=Qwen/Qwen3-VL-4B-Instruct-GGUF:Qwen3VL-4B-Instruct-Q4_K_M.gguf \
    --file vae=black-forest-labs/flux-3-image-klein:ae.safetensors \
    --steps 16 --guidance 1.0 --sampler euler
```

The specification is written to `models.json` and the files are fetched. Then
use it, for one run or for good:

```bash
./generate.py generate "a red fox in a snow-covered forest" --model klein-q4
./generate.py config-set model klein-q4
```

`--file` takes `role=repository:filename`. The role is the part the file plays,
and the names are the ones `sd-cli` itself uses: `diffusion_model`, `vae`,
`llm`, `t5xxl`, `clip_l`, `clip_g`, `clip_vision`, `llm_vision`, `tokenizer`,
`control_net`. The repository may be left out when it is the one `--from-hf`
names. A model of one file needs one `--file`; the roles it does not name are
not passed to the binary.

### By hand

`config/models.json` is a plain file, and editing it is supported:

```json
{
  "models": {
    "klein-q4": {
      "backend": "sdcli",
      "sources": {
        "diffusion_model": {"repo": "unsloth/flux-3-image-klein-GGUF", "file": "flux-3-image-klein-UD-Q4_K_XL.gguf"},
        "vae": {"repo": "black-forest-labs/flux-3-image-klein", "file": "ae.safetensors"}
      },
      "steps": 16,
      "guidance": 1.0,
      "sampling_method": "euler",
      "description": "FLUX.3 Image Klein, 16-step distilled"
    }
  }
}
```

A model in this file runs the way it is written here, whatever its name looks
like: a model called `flux.1-dev-gguf-q4ks` with `"backend": "coreml"` runs
through CoreML.

A model's name is one plain name — the directory its files are written to,
under `models_dir/sdcli/`. A name with a path in it (`../etc`) is refused.

Two fields are for the backends whose runtime knows a fixed set of models:
`alias` names one of that backend's own models, and `path` is a directory on
this machine that holds a local copy of it.

### What each backend can do with a specification

| Backend | An arbitrary repository |
|---|---|
| stable-diffusion.cpp | **Yes.** Any model `sd-cli` can run: the roles map to the binary's own flags. |
| CoreML | **Yes.** The repository is fetched whole; `convert-coreml` turns a PyTorch model into one. |
| OllamaDiffuser | **Yes, for GGUF models.** The specification is written to the registry file the package reads, and a variant it has never heard of takes every `.gguf` in the repository. The text encoders and VAE come from the FLUX and SD 3.5 families, so a GGUF of another family is not runnable through it. |
| MLX (diffusionkit) | **No.** Its checkpoints are a table inside that package, and the architecture of each is code there. A specification names one of the models diffusionkit has. |
| MFLUX | **No.** The architecture is chosen by the alias, and the aliases are the package's. A specification names one of its aliases (`"alias"`), and `path` points it at a local copy. |
| Bonsai | **No.** Its variants are a closed list in the Bonsai demo repository. A specification names one of them. |

### A model that is already on this machine

A model you already have does not have to be written down first. Give the
tool the path:

```bash
./generate.py generate "a red fox" --model ~/models/flux1-dev-Q4_K_M.gguf
./generate.py generate "a red fox" --model ~/models/my-model-directory
```

What a thing on disk is comes from what is there, not from how it is
named:

| What is on disk | What it is | Who runs it |
|---|---|---|
| a `.gguf` file | a GGUF model | OllamaDiffuser — the file is linked into the directory the package keeps its models in, and the package is told about it |
| a directory with a `.gguf` and the text encoders next to it | a GGUF model laid out the way OllamaDiffuser lays one out | OllamaDiffuser |
| a directory of weights (a transformer, an encoder, a VAE) | a stable-diffusion.cpp model | stable-diffusion.cpp — the roles are read off the names, and what was decided is printed |
| a directory with `*.mlmodelc` or `*.mlpackage` | a converted bundle | CoreML |
| a directory with `config.json` and `unet/` | a diffusers model | CoreML — converted on first use, then reused |

A name a catalogue already answers to is suffixed (`klein-2`), so the
catalogue entry and the model on disk both stay reachable. A diffusers
model is converted the first time it runs; that takes 15–20 minutes.

The MLX backend cannot run a local model: its checkpoints are a table
inside diffusionkit, and the architecture of each is code there.

`./generate.py list-models` shows what is under the models root, including
the models no catalogue describes.

---

## File Structure

```
mygen-playground/
├── gui.py                # PySide6 GUI entry (primary interface)
├── generate.py           # Typer CLI entry
├── eyegen/               # The package
│   ├── cli/              # The commands: generate, pull, models, config, status, hf
│   ├── gui/              # The window, its mixins, its workers
│   ├── backends/         # One subpackage per backend
│   │   ├── bonsai/       # PrismML ternary models
│   │   ├── coreml/       # Apple Neural Engine, via a sidecar venv
│   │   ├── sdcli/        # stable-diffusion.cpp, through its command line
│   │   └── runner.py     # The subprocess plumbing they share
│   ├── model_specs.py    # What a model is: its files, its settings
│   ├── model_catalogue.py# Where the user's own models are kept
│   ├── paths.py          # Where the models are kept: one root, by convention
│   ├── config.py         # The configuration and the paths
│   └── generation.py     # The dispatcher that runs a backend
├── scripts/
│   ├── setup-bonsai.sh   # One-time installer for Bonsai
│   ├── setup-coreml.sh   # One-time installer for CoreML sidecar venv
│   └── generate_theme.py # Generates gui/style/*.qss from the tokens
├── config/
│   ├── config.json       # Generation defaults
│   ├── models.json       # Your own model specifications (see above)
│   └── gui_state.json    # GUI state (auto-saved on close)
├── tests/                # pytest
├── outputs/              # Generated images (auto-created)
└── venv/                 # Virtual environment (after setup)

~/models/                 # Unified model artifact tree
├── .hf-cache/hub/        # HuggingFace download cache (HF_HUB_CACHE)
└── eyegen/               # All EyeGen-specific artifacts
    ├── sdcli/            # One directory per model: transformer, encoder, VAE
    ├── saved-mflux/      # Output of ./generate.py save-model
    ├── bonsai-demo/      # Bonsai-Image-Demo vendor (created by setup-bonsai)
    │   ├── .venv/        # Py 3.11 venv with patched mflux-prism + mlx
    │   └── models/       # Downloaded bonsai models
    ├── coreml/           # Downloaded/converted CoreML model bundles
    └── .coreml-venv/     # Py 3.11 sidecar venv with python_coreml_stable_diffusion
```

The root is one setting. `models_dir` in `config/config.json` moves everything under it — `./generate.py config-set models_dir /Volumes/Models` — and `EYEGEN_MODELS_DIR` does the same for one run, which is the way to use a different disk without editing a file.

The stable-diffusion.cpp models each have a directory of their own, because their files have the same names: `ae.safetensors` is a VAE whichever model it belongs to. An install from before that change keeps its files in one flat directory and is asked for them in a per-model one, so the tool will say the files are missing. Move them, and the model is back:

```bash
mkdir -p ~/models/eyegen/sdcli/flux-3-image-klein-gguf
mv ~/models/eyegen/flux3-klein/* ~/models/eyegen/sdcli/flux-3-image-klein-gguf/
```

The same is true of `./generate.py save-model`, which now writes under `saved-mflux/` rather than into the root itself.

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
