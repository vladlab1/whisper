#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_BUNDLE="$REPO_DIR/AudioCapture.app"
BIN_SRC="$REPO_DIR/AudioCapture/.build/release/AudioCapture"
BIN_DST="$APP_BUNDLE/Contents/MacOS/AudioCapture"
PLIST_DST="$APP_BUNDLE/Contents/Info.plist"

cd "$REPO_DIR/AudioCapture"
echo "Building AudioCapture (release)..."
swift build -c release

mkdir -p "$APP_BUNDLE/Contents/MacOS"
cp "$BIN_SRC" "$BIN_DST"

cat > "$PLIST_DST" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>com.github.whisper-app.AudioCapture</string>
    <key>CFBundleName</key>
    <string>AudioCapture</string>
    <key>CFBundleDisplayName</key>
    <string>AudioCapture</string>
    <key>CFBundleExecutable</key>
    <string>AudioCapture</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>AudioCapture records microphone audio for the whisper-app transcription tool.</string>
    <key>NSScreenCaptureUsageDescription</key>
    <string>AudioCapture records system audio output for the whisper-app transcription tool.</string>
    <key>NSAudioCaptureUsageDescription</key>
    <string>AudioCapture records system audio output for the whisper-app transcription tool.</string>
</dict>
</plist>
PLIST

echo "Ad-hoc codesigning the bundle..."
codesign --force --deep --sign - "$APP_BUNDLE"

echo ""
echo "Bundle: $APP_BUNDLE"
echo "Binary: $BIN_DST"
codesign -dv "$APP_BUNDLE" 2>&1 | sed 's/^/  /'
echo "Done."
