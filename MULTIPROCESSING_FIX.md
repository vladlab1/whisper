# Multiprocessing Fix for Jupyter Notebook

## Problem
When running parallel transcription in `transcribe_parallel.ipynb`, you encountered this error:

```
AttributeError: Can't get attribute 'transcribe_with_faster_whisper' on <module '__main__' (<class '_frozen_importlib.BuiltinImporter'>)>
```

## Root Cause
On macOS, Python's `multiprocessing` uses the 'spawn' start method by default. When spawning new processes, it needs to pickle (serialize) the function objects. However, functions defined in Jupyter notebooks exist in the `__main__` module, which cannot be properly pickled and imported by spawned child processes.

## Solution
The transcription worker functions have been extracted into a separate Python module (`transcribe_workers.py`) that can be properly imported by spawned processes.

### Changes Made

1. **Created `transcribe_workers.py`**: A new module containing:
   - `transcribe_with_mlx()`
   - `transcribe_with_faster_whisper()`
   - `transcribe_with_whispercpp()`

2. **Updated `transcribe_parallel.ipynb`** (Cell 7):
   - Removed inline function definitions
   - Added import statement: `from transcribe_workers import ...`

3. **Updated function calls** (Cell 8):
   - Added `MODEL_SIZE` parameter to all worker function calls
   - This ensures the workers can access the model size without relying on global variables

## How to Use

1. **Restart your Jupyter kernel** to clear any cached state:
   - In Jupyter: Kernel → Restart Kernel

2. **Run all cells from the beginning**:
   - The notebook will now import functions from `transcribe_workers.py`
   - Parallel processing will work correctly

3. **Verify the fix**:
   - Look for: `"Transcription functions loaded from transcribe_workers.py"`
   - Both processes should start and run without AttributeError

## Technical Details

### Why This Works
- Separate `.py` modules can be properly imported by spawned processes
- The `transcribe_workers` module is in the same directory, making it easily importable
- All necessary parameters (including `model_size`) are now explicitly passed to workers

### Alternative Solutions (Not Used)
- **Change spawn method to 'fork'**: Has issues on macOS and with Jupyter
- **Use threading**: Limited by Python's GIL (Global Interpreter Lock)
- **Use `dill` for pickling**: Adds dependency and complexity

## Files Modified
- ✅ `transcribe_workers.py` (created)
- ✅ `transcribe_parallel.ipynb` (updated cells 7 and 8)

## Testing
After restarting the kernel and running all cells, you should see:
```
[Chunk 1] Starting faster-whisper transcription (cpu)...
[Chunk 2] Starting MLX transcription (CoreML/Neural Engine)...
```

Both processes will run simultaneously without errors!

