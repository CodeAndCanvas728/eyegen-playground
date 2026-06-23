#!/bin/bash
# setup-coreml.sh — One-time install for the CoreML (Apple Neural Engine) backend
#
# Creates a sidecar Python 3.11 venv at ~/models/eyegen/.coreml-venv/ and
# installs Apple's python_coreml_stable_diffusion package + its pinned deps.
#
# Why a sidecar venv?
#   Apple's package pins diffusers==0.30.2, transformers==4.44.2,
#   huggingface-hub==0.24.6, numpy<1.24, diffusionkit==0.4.0. These conflict
#   with EyeGen's main Python 3.14 venv (diffusers 0.38, numpy 2.x, etc.).
#   Keeping Apple's pinned stack in a sidecar 3.11 venv avoids any conflict
#   and means we don't touch EyeGen's main venv.
#
# Usage:
#   ./scripts/setup-coreml.sh
#   PYTHON_BIN=/opt/homebrew/bin/python3.11 ./scripts/setup-coreml.sh   # override
#
# Re-run to upgrade the sidecar venv (it will pip install --upgrade).

set -e

VENV_DIR="${COREML_VENV:-$HOME/models/eyegen/.coreml-venv}"
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.11}"

echo "CoreML Setup for EyeGen"
echo "=================================="
echo ""
echo "Sidecar venv:  $VENV_DIR"
echo "Python binary: $PYTHON_BIN"
echo ""

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python 3.11 not found at $PYTHON_BIN"
    echo ""
    echo "Install it with Homebrew:"
    echo "  brew install python@3.11"
    echo ""
    echo "Or set PYTHON_BIN to another 3.11 interpreter, e.g.:"
    echo "  PYTHON_BIN=/path/to/python3.11 ./scripts/setup-coreml.sh"
    exit 1
fi

PY_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$PY_VERSION" != "3.11" ]; then
    echo "WARN: expected Python 3.11, found $PY_VERSION."
    echo "Apple's pinned deps (numpy<1.24) require 3.11 or older. Continuing anyway."
fi

# Create the venv
echo "Creating sidecar venv ..."
mkdir -p "$(dirname "$VENV_DIR")"
"$PYTHON_BIN" -m venv "$VENV_DIR"
echo "Created $VENV_DIR"

# Activate and install
echo ""
echo "Upgrading pip ..."
"$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip

echo "Installing pinned dependencies (this may take a few minutes) ..."
"$VENV_DIR/bin/pip" install --quiet \
    "diffusers==0.30.2" \
    "transformers==4.44.2" \
    "huggingface-hub==0.24.6" \
    "numpy<1.24" \
    "coremltools" \
    "diffusionkit==0.4.0" \
    "invisible-watermark" \
    "scipy" \
    "scikit-learn" \
    "matplotlib"

echo "Installing python_coreml_stable_diffusion from apple/ml-stable-diffusion ..."
<<<<<<< Updated upstream
COREML_COMMIT="5a170d29cf38e674b80541d7ce22929c6a11cdde"
"$VENV_DIR/bin/pip" install --quiet "git+https://github.com/apple/ml-stable-diffusion.git@$COREML_COMMIT"
=======
"$VENV_DIR/bin/pip" install --quiet "git+https://github.com/apple/ml-stable-diffusion.git@e12202c1f6405b83918b58a5d097cd61e3e1f702"
>>>>>>> Stashed changes

# Sanity check
if ! "$VENV_DIR/bin/python" -c "import python_coreml_stable_diffusion" 2>/dev/null; then
    echo ""
    echo "ERROR: Installation completed but python_coreml_stable_diffusion is not importable."
    echo "Try activating the venv and running it manually to see the error:"
    echo "  source $VENV_DIR/bin/activate"
    echo "  python -c 'import python_coreml_stable_diffusion'"
    exit 1
fi

echo ""
echo "CoreML sidecar venv ready at $VENV_DIR"
echo ""
echo "Next: download or convert a model."
echo "  Pull a pre-converted model from Hugging Face:"
echo "    ./generate.py pull-coreml sd-2-1-base-palettized"
echo ""
echo "  Or convert a PyTorch model from scratch (15-20 min):"
echo "    ./generate.py convert-coreml stabilityai/stable-diffusion-2-1-base"
echo ""
echo "Then select 'coreml' in the GUI's Backend dropdown, or:"
echo "    ./generate.py generate 'a photo of a cat' --backend coreml"
