"""
Transcription worker functions for parallel audio processing.
This module is separate to avoid multiprocessing pickling issues in Jupyter notebooks.
"""

import time


def transcribe_with_mlx(audio_path, language, result_queue, chunk_id, model_size):
    """Transcribe using MLX Whisper (optimized for Apple Silicon)."""
    try:
        import mlx_whisper
        print(f"[Chunk {chunk_id}] Starting MLX transcription (CoreML/Neural Engine)...")
        start = time.time()
        
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=f"mlx-community/whisper-{model_size}-mlx",
            language=language
        )
        
        # Extract segments with timestamps
        segments = []
        for segment in result.get('segments', []):
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text']
            })
        
        elapsed = time.time() - start
        print(f"[Chunk {chunk_id}] MLX transcription completed in {elapsed:.2f}s")
        
        result_queue.put({
            'chunk_id': chunk_id,
            'segments': segments,
            'time': elapsed,
            'backend': 'mlx'
        })
    except Exception as e:
        print(f"[Chunk {chunk_id}] MLX transcription failed: {e}")
        result_queue.put({'chunk_id': chunk_id, 'error': str(e)})


def transcribe_with_faster_whisper(audio_path, language, result_queue, chunk_id, model_size, device="cpu", compute_type="int8"):
    """Transcribe using faster-whisper."""
    try:
        from faster_whisper import WhisperModel
        print(f"[Chunk {chunk_id}] Starting faster-whisper transcription ({device})...")
        start = time.time()
        
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            num_workers=1,
            cpu_threads=4  # Use 4 threads per process
        )
        
        segments_iter, info = model.transcribe(audio_path, language=language)
        
        segments = []
        for segment in segments_iter:
            segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text
            })
        
        elapsed = time.time() - start
        print(f"[Chunk {chunk_id}] faster-whisper transcription completed in {elapsed:.2f}s")
        
        result_queue.put({
            'chunk_id': chunk_id,
            'segments': segments,
            'time': elapsed,
            'backend': f'faster-whisper-{device}'
        })
    except Exception as e:
        print(f"[Chunk {chunk_id}] faster-whisper transcription failed: {e}")
        result_queue.put({'chunk_id': chunk_id, 'error': str(e)})


def transcribe_with_whispercpp(audio_path, language, result_queue, chunk_id, model_size, use_coreml=False):
    """Transcribe using whisper.cpp."""
    try:
        from pywhispercpp.model import Model
        print(f"[Chunk {chunk_id}] Starting whisper.cpp transcription (CoreML: {use_coreml})...")
        start = time.time()
        
        # Initialize model
        model = Model(model_size, use_coreml=use_coreml)
        
        # Transcribe
        segments_data = model.transcribe(audio_path, language=language)
        
        segments = []
        for segment in segments_data:
            segments.append({
                'start': segment.t0 / 100.0,  # Convert from centiseconds
                'end': segment.t1 / 100.0,
                'text': segment.text
            })
        
        elapsed = time.time() - start
        print(f"[Chunk {chunk_id}] whisper.cpp transcription completed in {elapsed:.2f}s")
        
        result_queue.put({
            'chunk_id': chunk_id,
            'segments': segments,
            'time': elapsed,
            'backend': f'whispercpp-coreml' if use_coreml else 'whispercpp-cpu'
        })
    except Exception as e:
        print(f"[Chunk {chunk_id}] whisper.cpp transcription failed: {e}")
        result_queue.put({'chunk_id': chunk_id, 'error': str(e)})

