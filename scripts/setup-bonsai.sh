#!/bin/bash
# setup-bonsai.sh — One-time install for the Bonsai (PrismML) backend
#
# Clones the Bonsai-Image-Demo repo into ~/models/eyegen/bonsai-demo/ and runs
# its setup.sh. This installs a dedicated Python 3.11 venv at
# ~/models/eyegen/bonsai-demo/.venv/ that contains the patched mflux + MLX
# kernels needed for the 1.58-bit ternary + 1-bit binary weight formats.
#
# Re-run this script to update the bonsai vendor to the latest commit.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")")"
EYEGEN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BONSAI_DIR="${BONSAI_DIR:-$HOME/models/eyegen/bonsai-demo}"
BONSAI_REPO="https://github.com/PrismML-Eng/Bonsai-Image-Demo.git"
PYTHON_VERSION="3.11"

echo "Bonsai Setup for EyeGen"
echo "=================================="
echo ""
echo "Install location: $BONSAI_DIR"
echo ""

# macOS-only: bonsai relies on MLX which is Apple Silicon native
if [ "$(uname -s)" != "Darwin" ]; then
    echo "Note: bonsai MLX kernels are macOS/Apple-Silicon only."
    echo "On Linux, the bonsai subprocess will use the gemlite backend."
    echo ""
fi

# Clone or update the bonsai-demo repo
if [ ! -d "$BONSAI_DIR" ]; then
    echo "Cloning Bonsai-Image-Demo ..."
    mkdir -p "$(dirname "$BONSAI_DIR")"
    git clone "$BONSAI_REPO" "$BONSAI_DIR"
    echo "Cloned."
elif [ -d "$BONSAI_DIR/.git" ]; then
    echo "Bonsai-Image-Demo already cloned. Pulling latest ..."
    (cd "$BONSAI_DIR" && git pull --ff-only || echo "  (pull skipped — local changes or no network)")
else
    echo "ERROR: $BONSAI_DIR exists but is not a git repo."
    echo "Move it aside and re-run this script."
    exit 1
fi

# Run the bonsai-demo's own setup.sh
echo ""
echo "Running bonsai-demo setup.sh (this installs Python $PYTHON_VERSION + uv + venv) ..."
echo "This may take several minutes on first run."
echo ""

cd "$BONSAI_DIR"
chmod +x setup.sh
./setup.sh

# Sanity check
if [ ! -x "$BONSAI_DIR/.venv/bin/python" ]; then
    echo "ERROR: bonsai venv was not created at $BONSAI_DIR/.venv/"
    exit 1
fi

echo ""
echo "Bonsai installed at $BONSAI_DIR"
echo ""
echo "Next: download a model:"
echo "  $BONSAI_DIR/scripts/download_model.sh          # ternary (default)"
echo "  $BONSAI_DIR/scripts/download_model.sh binary   # 1-bit binary"
echo ""
echo "Or use the GUI: Settings -> Bonsai -> Download Bonsai Model"
echo "Or the CLI:    ./generate.py pull-bonsai ternary-mlx"
