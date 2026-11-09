"""Constants for the Idol (image-to-3D) sidecar backend."""

#: The upstream SDK repository.
IDOL_SDK_REPO = "https://github.com/speridlabs/idol-sdk.git"

#: The git ref that adds Apple Silicon (MPS) device selection. It is an open
#: pull request (PR #5), not merged into main: main falls back to CPU on a
#: Mac. Pinned to the PR head's commit SHA (was ``pull/5/head``) so a re-run
#: of setup-idol.sh cannot silently pull whatever the PR holds later.
IDOL_SDK_REF = "69b947bd3856f39f6230408c8fd3b76b47e66a60"
IDOL_SDK_BRANCH = "mps"

#: The Hugging Face repo the weights download from on first run.
IDOL_WEIGHTS_REPO = "speridlabs/idol"

#: transformers pin: the SDK passes ``torch_dtype=``, which transformers v5
#: removed (open PR #7 upstream); 4.57 only warns.
IDOL_TRANSFORMERS_PIN = ">=4.56,<5"

#: Crop modes idol.sample accepts.
IDOL_MODES = ("synthetic", "real")

#: Devices idol.sample accepts on this branch.
IDOL_DEVICES = ("cuda", "mps", "cpu")

#: --steps bounds, matching config validation.
IDOL_MIN_STEPS = 1
IDOL_MAX_STEPS = 100

#: --num_gaussians bounds from the SDK: 32,768 to 1,048,576, a multiple of 32.
IDOL_MIN_GAUSSIANS = 32768
IDOL_MAX_GAUSSIANS = 1048576

#: The choices the GUI offers, all multiples of 32 within the bounds.
IDOL_GAUSSIAN_CHOICES = (32768, 65536, 131072, 262144, 524288, 1048576)

#: Image extensions idol.sample reads.
IDOL_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

#: Interactive defaults, measured on the 16 GB M2 Air (AI Lab 2026-11-06):
#: 25 steps x 131072 gaussians is ~15 min on MPS; the paper defaults
#: (50 x 262144) take ~33 min and still fit, but are not interactive.
IDOL_DEFAULT_MODE = "synthetic"
IDOL_DEFAULT_STEPS = 25
IDOL_DEFAULT_NUM_GAUSSIANS = 131072
IDOL_DEFAULT_DEVICE = "mps"
IDOL_DEFAULT_TIMEOUT = 3600
