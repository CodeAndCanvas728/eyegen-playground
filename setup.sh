#!/bin/bash
# Setup script for MLX SD 3.5 workspace

set -e

echo "🚀 EyeGen Setup"
echo "=================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $PYTHON_VERSION found"

# Create venv
echo ""
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
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
