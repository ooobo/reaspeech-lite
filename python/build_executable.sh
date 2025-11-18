#!/bin/bash
# Build standalone executables for ReaSpeech Parakeet transcription
# This creates executables for all platforms using PyInstaller

set -e

echo "Building Parakeet Transcription Executables"
echo "==========================================="

# Check if we're in the right directory
if [ ! -f "parakeet_transcribe.py" ]; then
    echo "Error: Must run from python/ directory"
    exit 1
fi

# Check if PyInstaller is installed
if ! command -v pyinstaller &> /dev/null; then
    echo "PyInstaller not found. Installing..."
    pip install pyinstaller
fi

# Install dependencies
echo "Installing Python dependencies..."
pip install onnx-asr onnxruntime huggingface-hub soundfile resampy

# Determine platform
PLATFORM=$(uname -s)
case "$PLATFORM" in
    Linux*)     PLATFORM_NAME="linux";;
    Darwin*)    PLATFORM_NAME="macos";;
    MINGW*|MSYS*|CYGWIN*)  PLATFORM_NAME="windows";;
    *)          PLATFORM_NAME="unknown";;
esac

echo "Building for platform: $PLATFORM_NAME"

# Build executable
echo "Running PyInstaller..."
pyinstaller \
    --clean \
    --noconfirm \
    parakeet_transcribe.spec

echo ""
echo "Build complete!"
echo "Executable location: dist/parakeet-transcribe-$PLATFORM_NAME"
echo ""
echo "To test:"
echo "  ./dist/parakeet-transcribe-$PLATFORM_NAME /path/to/audio.wav"
