#!/bin/bash
# Setup script for MLX SD 3.5 workspace

set -e

echo "🚀 EyeGen Setup"
echo "=================================="

# Check Python
find_supported_python() {
    for candidate in python3.14 python3.13 python3.12 python3.11 python3; do
        if command -v "$candidate" &>/dev/null; then
            local cmd
            cmd=$(command -v "$candidate")
            if [[ "$cmd" == *"$PWD"* ]]; then
                continue
            fi
            local ver
            ver=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
            local major=$(echo "$ver" | cut -d. -f1)
            local minor=$(echo "$ver" | cut -d. -f2)
            if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -lt 15 ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    for py in "$HOME"/.local/share/uv/python/*/bin/python3; do
        if [ -x "$py" ]; then
            local ver
            ver=$("$py" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
            local major=$(echo "$ver" | cut -d. -f1)
            local minor=$(echo "$ver" | cut -d. -f2)
            if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -lt 15 ]; then
                echo "$py"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON_BIN=$(find_supported_python || true)
if [ -z "$PYTHON_BIN" ]; then
    echo "❌ Compatible Python (3.10-3.14) not found. PyTorch and MLX do not yet support Python 3.15+."
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_BIN" --version 2>&1 | awk '{print $2}')
echo "✓ Python $PYTHON_VERSION found ($PYTHON_BIN)"

# Create venv
echo ""
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    if [ ! -x "venv/bin/python" ] || ! "venv/bin/python" -c "pass" 2>/dev/null; then
        echo "⚠️  Existing venv is broken or non-executable. Recreating..."
        rm -rf "venv"
    fi
fi

if [ ! -d "venv" ]; then
    "$PYTHON_BIN" -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists and is healthy"
fi

# Activate venv
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "🔄 Upgrading pip..."
pip install --quiet --upgrade pip

# Install requirements
echo ""
echo "📥 Installing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Make script executable
echo ""
echo "🔐 Setting permissions..."
chmod +x generate.py
echo "✓ generate.py is executable"

# Create directories
echo ""
echo "📁 Creating directories..."
mkdir -p outputs
mkdir -p config
echo "✓ Directories ready"

# Create the unified model artifact tree under ~/models/eyegen/
EYEGEN_MODELS="$HOME/models/eyegen"
echo ""
echo "📁 Creating model artifact tree at $EYEGEN_MODELS ..."
mkdir -p "$EYEGEN_MODELS/saved-mflux"
mkdir -p "$EYEGEN_MODELS/coreml"
echo "✓ Model tree ready"
echo "  Saved MFLUX models  → $EYEGEN_MODELS/saved-mflux/"
echo "  CoreML models       → $EYEGEN_MODELS/coreml/"
echo "  Bonsai vendor       → $EYEGEN_MODELS/bonsai-demo/  (created by setup-bonsai)"
echo "  CoreML sidecar venv → $EYEGEN_MODELS/.coreml-venv/ (created by setup-coreml)"
echo "  HF download cache   → $HOME/models/.hf-cache/hub/  (EyeGen default)"

# Check MLX
echo ""
echo "🔍 Checking MLX installation..."
if python3 -c "import mlx; print(f'MLX {mlx.__version__} ✓')" 2>/dev/null; then
    echo "✓ MLX ready to use"
else
    echo "⚠️  MLX not yet available (will be downloaded on first use)"
fi

# Display next steps
echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate the environment (if not already activated):"
echo "     source venv/bin/activate"
echo ""
echo "  2. Try generating an image:"
echo "     ./generate.py generate \"a beautiful landscape\""
echo ""
echo "  3. View all commands:"
echo "     ./generate.py --help"
echo ""
echo "For more info, see README.md"
