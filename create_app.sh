#!/bin/bash
# create_app.sh — Create "EyeGen.app" and install it to ~/Applications
#
# This builds a minimal macOS .app bundle that launches gui.py through the
# project venv. No bundling of Python libraries — the venv is used directly,
# so the workspace folder must stay in place. Re-run this script if you move
# the workspace.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="EyeGen"
DEST="$HOME/Applications/$APP_NAME.app"

echo "🔧 EyeGen — App Bundle Creator"
echo "==============================="

# Sanity checks
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "❌ No venv found. Run ./setup.sh first."
    exit 1
fi
if [ ! -f "$SCRIPT_DIR/gui.py" ]; then
    echo "❌ gui.py not found in $SCRIPT_DIR"
    exit 1
fi

# Resolve the Python interpreter inside the venv
PYTHON="$SCRIPT_DIR/venv/bin/python"

echo "✓ Workspace: $SCRIPT_DIR"
echo "✓ Python:    $PYTHON"

# Create bundle directory structure
echo ""
echo "📦 Building $APP_NAME.app..."
rm -rf "$DEST"
mkdir -p "$DEST/Contents/MacOS"
mkdir -p "$DEST/Contents/Resources"

# ── Info.plist ────────────────────────────────────────────────────────────────
cat > "$DEST/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>EyeGen</string>
    <key>CFBundleDisplayName</key>
    <string>EyeGen</string>
    <key>CFBundleIdentifier</key>
    <string>com.local.eyegen</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleExecutable</key>
    <string>EyeGen</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
</dict>
</plist>
PLIST

# ── Launcher script ───────────────────────────────────────────────────────────
# Forces arm64 explicitly so the app works even when launched from a Rosetta
# (x86_64) terminal. MLX is arm64-only and dlopen fails without this.
cat > "$DEST/Contents/MacOS/$APP_NAME" << LAUNCHER
#!/bin/bash
exec arch -arm64 "$PYTHON" "$SCRIPT_DIR/gui.py" "\$@"
LAUNCHER

chmod +x "$DEST/Contents/MacOS/$APP_NAME"

# ── App icon ──────────────────────────────────────────────────────────────────
if [ -f "$SCRIPT_DIR/icon.png" ]; then
    echo "🎨 Generating app icon from icon.png..."
    ICONSET=$(mktemp -d)/AppIcon.iconset
    mkdir -p "$ICONSET"
    for size in 16 32 64 128 256 512; do
        sips -z $size $size "$SCRIPT_DIR/icon.png" --out "$ICONSET/icon_${size}x${size}.png" 2>/dev/null
        double=$((size * 2))
        sips -z $double $double "$SCRIPT_DIR/icon.png" --out "$ICONSET/icon_${size}x${size}@2x.png" 2>/dev/null
    done
    iconutil -c icns "$ICONSET" -o "$SCRIPT_DIR/icon.icns" 2>/dev/null
    rm -rf "$(dirname "$ICONSET")"
    cp "$SCRIPT_DIR/icon.icns" "$DEST/Contents/Resources/AppIcon.icns"
    echo "✓ App icon installed"
else
    echo "⚠  icon.png not found — app will use a default icon"
fi

echo "✅ Created: $DEST"
echo ""
echo "🎉 Done! You can now:"
echo "   • Open Finder → ~/Applications → double-click EyeGen"
echo "   • Add to Dock: drag from ~/Applications into your Dock"
echo "   • Spotlight: press ⌘Space and type 'EyeGen'"
echo ""
echo "   ⚠️  First launch: right-click → Open (Gatekeeper bypass, one time only)"
echo "   First generation downloads the model (~3GB) — subsequent runs are instant."
