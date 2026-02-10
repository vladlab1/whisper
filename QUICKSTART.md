# Quick Start Guide - Parallel Transcription

Get up and running with parallel transcription in 5 minutes!

## Prerequisites Check

✓ Mac with Apple Silicon (M1/M2/M3/M4)  
✓ Conda environment activated: `conda activate venvVideoTranscription`  
✓ 10 GB free RAM  
✓ Audio file ready to transcribe  

## Step 1: Install MLX Whisper (2 minutes)

```bash
pip install mlx-whisper
```

That's it! One command.

## Step 2: Open Notebook

```bash
jupyter lab transcribe_parallel.ipynb
```

## Step 3: Configure (30 seconds)

In **Cell 2**, update your audio file path:

```python
input_audio_path = '/path/to/your/audio.m4a'
meeting_language = "en"  # or "ru", "fr", "es", etc.
```

## Step 4: Run (2 minutes)

Click: **Cell → Run All**

Watch the progress:
```
✓ MLX Whisper available (optimal for Mac M4)
✓ faster-whisper available

Preprocessing audio...
Audio preprocessed in 2.45s

Splitting audio into 2 chunks with overlap...
Audio split completed in 0.87s

============================================================
Starting parallel transcription...
============================================================
Process 1: faster-whisper (CPU)
Process 2: MLX Whisper (CoreML/Neural Engine)

[Chunk 1] Starting faster-whisper transcription (CPU)...
[Chunk 2] Starting MLX transcription (CoreML/Neural Engine)...
```

## Step 5: Get Results

Files created:
- `transcription_parallel.txt` - Your transcription
- `transcription_parallel_segments.json` - Detailed segments with timestamps

## Performance Example

**Before (original method):**
- 60 min audio → 66 min processing

**After (parallel with MLX):**
- 60 min audio → 15 min processing
- **4.4x faster!**

## Troubleshooting

### MLX Install Failed?

```bash
# Check Python version (must be 3.10+)
python --version

# Check you're on Apple Silicon
uname -m  # Should show: arm64

# Try reinstalling
pip uninstall mlx-whisper -y
pip install mlx-whisper
```

### Backend Not Detected?

Run **Cell 3** to see which backends are available. If MLX shows "not available", see troubleshooting above.

### Still Issues?

See full guide: [PARALLEL_SETUP.md](PARALLEL_SETUP.md)

## Tips for Best Performance

1. **Close other apps** before transcribing
2. **Use "medium" model** for 2-3x faster transcription with good accuracy:
   ```python
   MODEL_SIZE = "medium"  # Change in Cell 2
   ```
3. **Keep Mac plugged in** for consistent performance
4. **First run is slower** (downloads model), subsequent runs are fast

## Compare Performance

Want to see the speedup yourself?

1. Run original: `transcribe.ipynb`
2. Run parallel: `transcribe_parallel.ipynb`
3. Compare the times!

## Next Steps

- Read [COMPARISON.md](COMPARISON.md) for detailed comparison
- Read [PARALLEL_SETUP.md](PARALLEL_SETUP.md) for advanced options
- Check [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for technical details

## Questions?

Common questions answered in [PARALLEL_SETUP.md](PARALLEL_SETUP.md#troubleshooting)

---

**That's it! You're now transcribing 3-8x faster. 🚀**

