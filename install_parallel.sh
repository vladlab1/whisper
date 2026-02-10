#!/bin/bash

# Installation script for parallel transcription on Mac M4
# This script installs the recommended backend (MLX Whisper) for parallel transcription

set -e  # Exit on error

echo "============================================================"
echo "Parallel Transcription Setup for Mac M4"
echo "============================================================"
echo ""

# Check if we're on Apple Silicon
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "⚠️  Warning: You're not on Apple Silicon (detected: $ARCH)"
    echo "   MLX Whisper is optimized for Apple Silicon Macs only."
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "Detected Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "❌ Error: Python 3.10 or higher is required"
    echo "   Please upgrade your Python version or activate the correct conda environment."
    exit 1
fi

echo "✓ Python version OK"
echo ""

# Check if conda environment is activated
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "⚠️  Warning: No conda environment detected"
    echo "   It's recommended to activate your conda environment first:"
    echo "   conda activate venvVideoTranscription"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ Conda environment: $CONDA_DEFAULT_ENV"
fi

echo ""
echo "============================================================"
echo "Installation Options"
echo "============================================================"
echo ""
echo "1. MLX Whisper (Recommended)"
echo "   - Best performance on Apple Silicon"
echo "   - Easiest to install"
echo "   - 3-8x speedup with parallel processing"
echo ""
echo "2. whisper.cpp with CoreML"
echo "   - Good performance"
echo "   - Requires Homebrew"
echo "   - 2-5x speedup with parallel processing"
echo ""
echo "3. Skip (use faster-whisper parallel only)"
echo "   - No additional installation"
echo "   - 1.5-2x speedup from parallel processing"
echo ""

read -p "Select option (1/2/3): " -n 1 -r OPTION
echo ""
echo ""

case $OPTION in
    1)
        echo "Installing MLX Whisper..."
        pip install mlx-whisper
        
        # Test installation
        if python -c "import mlx_whisper" 2>/dev/null; then
            echo ""
            echo "✓ MLX Whisper installed successfully!"
        else
            echo ""
            echo "❌ Installation failed. Please check the error messages above."
            exit 1
        fi
        ;;
    2)
        echo "Installing whisper.cpp..."
        
        # Check if Homebrew is installed
        if ! command -v brew &> /dev/null; then
            echo "❌ Error: Homebrew is not installed"
            echo "   Install Homebrew first: https://brew.sh"
            exit 1
        fi
        
        # Install whisper.cpp
        echo "Installing whisper.cpp via Homebrew..."
        brew install whisper.cpp
        
        # Install Python bindings
        echo "Installing Python bindings..."
        pip install pywhispercpp
        
        # Test installation
        if python -c "from pywhispercpp.model import Model" 2>/dev/null; then
            echo ""
            echo "✓ whisper.cpp installed successfully!"
        else
            echo ""
            echo "❌ Installation failed. Please check the error messages above."
            exit 1
        fi
        ;;
    3)
        echo "Skipping additional installation."
        echo "You can still use parallel transcription with faster-whisper (CPU only)."
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Open Jupyter Lab: jupyter lab"
echo "2. Open transcribe_parallel.ipynb"
echo "3. Configure your audio file path in Cell 2"
echo "4. Run all cells (Cell → Run All)"
echo ""
echo "For detailed instructions, see: PARALLEL_SETUP.md"
echo ""
echo "Happy transcribing! 🎉"

