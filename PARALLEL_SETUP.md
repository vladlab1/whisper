# Parallel Transcription Setup Guide

This guide explains how to set up and use the parallel transcription system for accelerated audio transcription on Mac M4.

## Overview

The parallel transcription system splits your audio into 2 chunks and processes them simultaneously:
- **Chunk 1**: Processed on CPU using faster-whisper
- **Chunk 2**: Processed on CoreML/Neural Engine using MLX Whisper or whisper.cpp

Expected speedup: **3-8x faster** than single-process transcription.

## Prerequisites

- Mac with Apple Silicon (M1/M2/M3/M4)
- Python 3.10 or higher
- Existing conda environment: `venvVideoTranscription`
- ffmpeg (already installed in your conda env)

## Installation Options

### Option 1: MLX Whisper (Recommended - Easiest & Fastest)

MLX Whisper is optimized specifically for Apple Silicon and provides the best performance.

```bash
# Activate your conda environment
conda activate venvVideoTranscription

# Install MLX Whisper
pip install mlx-whisper

# Test installation
python -c "import mlx_whisper; print('MLX Whisper installed successfully!')"
```

**Pros:**
- Easiest to install (single pip command)
- Best performance on Apple Silicon
- Native Neural Engine support
- No additional model downloads needed

**Expected Performance:**
- 2-4x faster than faster-whisper alone
- Combined with parallel processing: 3-8x total speedup

### Option 2: whisper.cpp with CoreML

whisper.cpp is a high-performance C++ implementation with CoreML support.

```bash
# Install whisper.cpp via Homebrew
brew install whisper.cpp

# Install Python bindings
pip install pywhispercpp

# Test installation
python -c "from pywhispercpp.model import Model; print('whisper.cpp installed successfully!')"
```

**Note:** Model files will be downloaded automatically on first use.

**Pros:**
- Good performance with CoreML
- Mature and well-maintained
- Lower memory usage

**Cons:**
- Requires Homebrew installation
- More complex setup

### Option 3: Parallel faster-whisper (Fallback)

If neither MLX nor whisper.cpp are available, the system will automatically fall back to using faster-whisper on both CPU processes.

**No additional installation needed** - uses your existing faster-whisper setup.

**Expected Performance:**
- 1.5-2x speedup from parallel processing only
- Less improvement than CoreML options but still helpful

## Quick Start

1. **Install a backend** (choose Option 1, 2, or skip for fallback)

2. **Open the notebook:**
   ```bash
   jupyter lab transcribe_parallel.ipynb
   ```

3. **Configure your audio file** in Cell 2:
   ```python
   input_audio_path = '/path/to/your/audio.m4a'
   output_audio_name = "output_16khz.wav"
   output_transcription_name = "transcription_parallel.txt"
   meeting_language = "en"  # or "ru", "fr", etc.
   ```

4. **Run all cells** (Cell → Run All)

5. **Monitor progress** - You'll see output like:
   ```
   ✓ MLX Whisper available (optimal for Mac M4)
   ✓ faster-whisper available
   
   Starting parallel transcription...
   Process 1: faster-whisper (CPU)
   Process 2: MLX Whisper (CoreML/Neural Engine)
   ```

6. **Check results:**
   - Transcription text: `transcription_parallel.txt`
   - Detailed segments: `transcription_parallel_segments.json`

## Configuration Options

### Model Size

In Cell 2, adjust `MODEL_SIZE` for speed vs. accuracy tradeoff:

```python
MODEL_SIZE = "medium"     # 2-3x faster, good accuracy (recommended for testing)
MODEL_SIZE = "large-v2"   # Best accuracy, slower (current default)
MODEL_SIZE = "large-v3"   # Latest model, best accuracy
```

### Overlap Duration

Adjust `OVERLAP_SECONDS` to handle word boundaries:

```python
OVERLAP_SECONDS = 2  # Default - good for most cases
OVERLAP_SECONDS = 3  # Better for fast speech or complex audio
OVERLAP_SECONDS = 4  # Best accuracy at split points, slightly slower
```

## Performance Benchmarking

The notebook automatically tracks and displays:
- Preprocessing time
- Audio splitting time
- Parallel transcription time
- Total time and real-time factor

Example output:
```
============================================================
PERFORMANCE SUMMARY
============================================================
Audio duration: 3955.82s (65.93 minutes)

Preprocessing: 2.45s
Audio splitting: 0.87s
Parallel transcription: 842.31s

Total processing time: 845.63s
Real-time factor: 0.21x
  (Lower is better - 0.21x means 4.7x faster than real-time)
============================================================
```

## Troubleshooting

### MLX Whisper Issues

**Problem:** `import mlx_whisper` fails

**Solutions:**
1. Verify Python version: `python --version` (must be 3.10+)
2. Verify Apple Silicon: `uname -m` (should show `arm64`)
3. Reinstall: `pip uninstall mlx-whisper -y && pip install mlx-whisper`
4. Check for conflicts: `pip list | grep whisper`

### whisper.cpp Issues

**Problem:** CoreML backend not working

**Solutions:**
1. Install Xcode Command Line Tools: `xcode-select --install`
2. Verify installation: `which whisper-cpp`
3. Try CPU mode first before CoreML

### Both Processes Using CPU

**Problem:** Warning message "No CoreML backend available, using CPU for both chunks"

**Solution:** This means neither MLX nor whisper.cpp with CoreML is available. Install MLX Whisper (Option 1) for best results.

### Poor Accuracy at Split Point

**Problem:** Transcription has errors or repeated words around the middle

**Solution:** Increase `OVERLAP_SECONDS` to 3 or 4 in Cell 2.

### Out of Memory Errors

**Problem:** Process crashes or runs out of memory

**Solutions:**
1. Use smaller model: `MODEL_SIZE = "medium"` or `"small"`
2. Close other applications
3. Monitor Activity Monitor during transcription

## Performance Tips

1. **Close unnecessary applications** before running transcription to free up RAM
2. **Use SSD storage** for temp files (default location is with your audio file)
3. **Monitor CPU/GPU usage** in Activity Monitor to verify parallel execution
4. **Use medium model for testing** then switch to large for final transcription
5. **Keep your Mac plugged in** for consistent performance

## File Outputs

The notebook creates several files:

1. **`*_output16khz.wav`** - Preprocessed audio (16kHz, mono)
2. **`transcription_parallel.txt`** - Final transcription text
3. **`transcription_parallel_segments.json`** - Detailed segments with timestamps
4. **`temp_chunks/`** - Temporary directory with audio chunks (can be deleted)

## Comparison with Original Method

| Method | Speed | Accuracy | Setup |
|--------|-------|----------|-------|
| Original (faster-whisper) | 1x | Good | Already installed |
| Parallel (CPU only) | 1.5-2x | Good | No setup needed |
| Parallel (MLX) | 3-8x | Good | `pip install mlx-whisper` |
| Parallel (whisper.cpp) | 2-5x | Good | Brew + pip install |

## Next Steps

1. **Test with a short audio file** first (< 5 minutes) to verify setup
2. **Compare with original method** by running both notebooks on the same file
3. **Experiment with model sizes** to find your optimal speed/accuracy balance
4. **Monitor performance** to ensure parallel execution is working

## Support & Resources

- **MLX Whisper:** https://github.com/ml-explore/mlx-examples/tree/main/whisper
- **whisper.cpp:** https://github.com/ggerganov/whisper.cpp
- **faster-whisper:** https://github.com/guillaumekln/faster-whisper

## Questions?

If you encounter issues not covered here, check:
1. Backend detection output in Cell 3
2. Error messages in parallel transcription cell output
3. System Activity Monitor during transcription

