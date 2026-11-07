#!/bin/bash
# setup-idol.sh — One-time install for the Idol (image-to-3D) backend
#
# Creates a sidecar Python 3.11 venv at ~/models/eyegen/idol/.venv/ and
# clones speridlabs/idol-sdk into ~/models/eyegen/idol/idol-sdk, checked out
# at the MPS branch (open PR #5: --device mps, CUDA→MPS→CPU auto-select).
#
# Why a sidecar venv?
#   The SDK needs torch 2.15 + transformers 4.57, which conflict with
#   EyeGen's main venv (same conflict class as the CoreML sidecar). The
#   system python3 may be too new for torch wheels, so 3.11 is required.
#
# Usage:
#   ./scripts/setup-idol.sh
#   PYTHON_BIN=/opt/homebrew/bin/python3.11 IDOL_DIR=/path/to/idol ./scripts/setup-idol.sh
#
# Re-run to upgrade: the SDK is re-fetched and pip installs with --upgrade.

set -e

IDOL_ROOT="${IDOL_DIR:-$HOME/models/eyegen/idol}"
VENV_DIR="$IDOL_ROOT/.venv"
SDK_DIR="$IDOL_ROOT/idol-sdk"
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.11}"
IDOL_SDK_REPO="${IDOL_SDK_REPO:-https://github.com/speridlabs/idol-sdk.git}"
IDOL_SDK_REF="${IDOL_SDK_REF:-pull/5/head}"
IDOL_SDK_BRANCH="${IDOL_SDK_BRANCH:-mps}"

echo "Idol (image-to-3D) setup for EyeGen"
echo "==================================="
echo ""
echo "Sidecar root:  $IDOL_ROOT"
echo "Venv:          $VENV_DIR"
echo "SDK checkout:  $SDK_DIR"
echo "Python binary: $PYTHON_BIN"
echo ""

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python 3.11 not found at $PYTHON_BIN"
    echo ""
    echo "Install it with Homebrew:"
    echo "  brew install python@3.11"
    echo ""
    echo "Or set PYTHON_BIN to another 3.11 interpreter, e.g.:"
    echo "  PYTHON_BIN=/path/to/python3.11 ./scripts/setup-idol.sh"
    exit 1
fi

PY_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$PY_VERSION" != "3.11" ]; then
    echo "ERROR: expected Python 3.11, found $PY_VERSION (torch has no wheels for newer)."
    exit 1
fi

# Create the venv
echo "Creating sidecar venv ..."
mkdir -p "$IDOL_ROOT"
"$PYTHON_BIN" -m venv "$VENV_DIR"
echo "Created $VENV_DIR"

# Clone (or update) the SDK at the MPS branch
echo ""
if [ -d "$SDK_DIR/.git" ]; then
    echo "Updating existing SDK checkout at $SDK_DIR ..."
    git -C "$SDK_DIR" fetch --depth 1 origin "$IDOL_SDK_REF"
    git -C "$SDK_DIR" checkout -q FETCH_HEAD
else
    echo "Cloning idol-sdk ($IDOL_SDK_REPO) ..."
    git clone --depth 1 "$IDOL_SDK_REPO" "$SDK_DIR"
    git -C "$SDK_DIR" fetch --depth 1 origin "$IDOL_SDK_REF"
    git -C "$SDK_DIR" checkout -q FETCH_HEAD
fi
echo "SDK at ref $IDOL_SDK_REF (branch $IDOL_SDK_BRANCH)"

# Install the SDK and its deps
echo ""
echo "Upgrading pip ..."
"$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip

echo "Installing idol-sdk + torch (this may take a few minutes) ..."
# transformers is pinned below v5: the SDK passes torch_dtype=, removed in
# transformers v5 (open PR #7 upstream); 4.57 only warns.
"$VENV_DIR/bin/pip" install --quiet --upgrade \
    -e "$SDK_DIR" \
    accelerate \
    "transformers>=4.56,<5"

# Sanity check: the SDK imports and resolves the device
if ! "$VENV_DIR/bin/python" -c "import idol; from idol.device import resolve_device" 2>/dev/null; then
    echo ""
    echo "ERROR: Installation completed but the idol package is not importable."
    echo "Try activating the venv and running it manually to see the error:"
    echo "  source $VENV_DIR/bin/activate"
    echo "  python -c 'import idol'"
    exit 1
fi
echo "Device auto-select on this machine: $(cd "$SDK_DIR" && "$VENV_DIR/bin/python" -c 'from idol.device import resolve_device; print(resolve_device(None))')"

echo ""
echo "Idol sidecar ready at $IDOL_ROOT"
echo ""
echo "Note: the first run downloads ~5 GB into the HuggingFace cache (the idol"
echo "weights plus two GATED encoders — accept their terms on HuggingFace with"
echo "the account your HF token belongs to first):"
echo "  https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m"
echo "  https://huggingface.co/black-forest-labs/FLUX.2-dev"
echo ""
echo "Then: ./generate.py to3d photo.jpg --mode real"
