# Whisper Audio Transcription

Audio transcription tools using OpenAI Whisper models.

## Setup

Clone repository and open terminal in the folder with repository.

Create venv venvVideoTranscription using conda:

```bash
conda env create -f environment.yml
```

Activate venv:
```bash
conda activate venvVideoTranscription
```

## Notebooks

### `transcribe.ipynb` - Single-Process Transcription
Standard transcription using faster-whisper. Suitable for most use cases.

### `transcribe_parallel.ipynb` - Parallel Transcription (NEW)
**Recommended for Mac M4** - Up to 3-8x faster than single-process transcription!

Splits audio into chunks and processes them in parallel using:
- CPU (faster-whisper)
- CoreML/Neural Engine (MLX Whisper or whisper.cpp)

**See [PARALLEL_SETUP.md](PARALLEL_SETUP.md) for detailed setup instructions.**

Quick install for parallel transcription:
```bash
conda activate venvVideoTranscription
pip install mlx-whisper
```

### `wordCloud.ipynb`
Generate word clouds from transcriptions.

## Performance Comparison

| Method | 1-hour audio | Speedup |
|--------|-------------|---------|
| Single-process (faster-whisper) | ~60-90 min | 1x |
| Parallel (CPU only) | ~40-60 min | 1.5x |
| **Parallel (MLX on M4)** | **~15-25 min** | **3-8x** |

## Troubleshooting

For parallel transcription issues, see [PARALLEL_SETUP.md](PARALLEL_SETUP.md)