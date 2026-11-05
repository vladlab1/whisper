"""
Transcription worker functions for parallel audio processing.
This module is separate to avoid multiprocessing pickling issues in Jupyter notebooks.
"""

import time
from collections import defaultdict


def _emit(msg, log_queue):
    print(msg)
    if log_queue is not None:
        try:
            log_queue.put(msg)
        except Exception:
            pass


def _mlx_repo(model_size):
    if model_size == "large-v3-turbo":
        return "mlx-community/whisper-large-v3-turbo"
    return f"mlx-community/whisper-{model_size}-mlx"


def transcribe_with_mlx(audio_path, language, result_queue, chunk_id, model_size, log_queue=None):
    """Transcribe using MLX Whisper (optimized for Apple Silicon)."""
    try:
        import mlx_whisper
        _emit(f"[Chunk {chunk_id}] Starting MLX transcription (CoreML/Neural Engine)...", log_queue)
        start = time.time()

        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=_mlx_repo(model_size),
            language=language,
            condition_on_previous_text=False,
            word_timestamps=True,
        )

        # Extract segments with timestamps
        segments = []
        for segment in result.get('segments', []):
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
                'words': [{'start': w['start'], 'end': w['end'], 'word': w['word']}
                          for w in segment.get('words', [])],
            })

        elapsed = time.time() - start
        _emit(f"[Chunk {chunk_id}] MLX transcription completed in {elapsed:.2f}s", log_queue)

        result_queue.put({
            'chunk_id': chunk_id,
            'segments': segments,
            'time': elapsed,
            'backend': 'mlx'
        })
    except Exception as e:
        _emit(f"[Chunk {chunk_id}] MLX transcription failed: {e}", log_queue)
        result_queue.put({'chunk_id': chunk_id, 'error': str(e)})


def transcribe_with_faster_whisper(audio_path, language, result_queue, chunk_id, model_size, device="cpu", compute_type="int8", log_queue=None):
    """Transcribe using faster-whisper."""
    try:
        from faster_whisper import WhisperModel
        _emit(f"[Chunk {chunk_id}] Starting faster-whisper transcription ({device})...", log_queue)
        start = time.time()

        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            num_workers=1,
            cpu_threads=4
        )

        segments_iter, info = model.transcribe(
            audio_path,
            language=language,
            condition_on_previous_text=False,
            word_timestamps=True,
        )

        segments = []
        for segment in segments_iter:
            segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text,
                'words': [{'start': w.start, 'end': w.end, 'word': w.word}
                          for w in segment.words or []],
            })

        elapsed = time.time() - start
        _emit(f"[Chunk {chunk_id}] faster-whisper transcription completed in {elapsed:.2f}s", log_queue)

        result_queue.put({
            'chunk_id': chunk_id,
            'segments': segments,
            'time': elapsed,
            'backend': f'faster-whisper-{device}'
        })
    except Exception as e:
        _emit(f"[Chunk {chunk_id}] faster-whisper transcription failed: {e}", log_queue)
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


def diarize_with_speechbrain(audio_path, num_speakers, device="mps", log=None):
    """
    Run speaker diarization using SpeechBrain ECAPA-TDNN embeddings + clustering.

    Args:
        audio_path: Path to the full preprocessed 16kHz mono WAV file.
        num_speakers: Known number of speakers in the audio.
        device: PyTorch device for embedding extraction ("mps", "cpu").

    Returns:
        List of dicts ``[{start, end, speaker}, ...]`` covering the audio timeline.
    """
    import torch
    import torchaudio
    import numpy as np

    # Compatibility shim: torchaudio 2.9+ removed list_audio_backends()
    # which SpeechBrain 1.0.x still calls on import.
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: [""]

    from speechbrain.inference.speaker import EncoderClassifier
    from sklearn.cluster import AgglomerativeClustering

    def _dlog(msg):
        print(msg)
        if log:
            log(msg)

    _dlog(f"[Diarization] Starting speaker diarization on {device}...")
    start = time.time()

    # Load speaker embedding model on CPU first, then move to target device.
    # SpeechBrain's run_opts only supports "cpu" and "cuda"; MPS is not
    # handled, so we load on CPU and relocate manually.
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )
    if device != "cpu":
        classifier.mods.to(device)
        classifier.device = device
        _dlog(f"[Diarization] Model moved to {device}")

    # Load audio with soundfile to avoid torchaudio's torchcodec requirement (torchaudio 2.9+)
    import soundfile as sf
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # stereo → mono
    if sr != 16000:
        import torchaudio.transforms as T
        resampler = T.Resample(sr, 16000)
        signal = resampler(torch.from_numpy(data).unsqueeze(0)).squeeze(0)
        sr = 16000
    else:
        signal = torch.from_numpy(data)

    # Sliding window parameters
    window_samples = int(1.5 * sr)   # 1.5 s
    step_samples = int(0.75 * sr)    # 0.75 s (50 % overlap)

    embeddings = []
    timestamps = []

    for start_sample in range(0, len(signal) - window_samples + 1, step_samples):
        end_sample = start_sample + window_samples
        window = signal[start_sample:end_sample]

        # Skip near-silent windows
        if window.abs().mean() < 0.005:
            continue

        window_tensor = window.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = classifier.encode_batch(window_tensor)
        embeddings.append(embedding.squeeze().cpu().numpy())
        timestamps.append({
            "start": start_sample / sr,
            "end": end_sample / sr,
        })

    if len(embeddings) < num_speakers:
        _dlog(f"[Diarization] WARNING: only {len(embeddings)} voiced windows found, "
              f"cannot cluster into {num_speakers} speakers. Returning single speaker.")
        diarization = [{"start": t["start"], "end": t["end"], "speaker": 0}
                       for t in timestamps]
        elapsed = time.time() - start
        _dlog(f"[Diarization] Completed in {elapsed:.2f}s ({len(diarization)} windows)")
        return diarization

    embeddings_array = np.array(embeddings)

    clustering = AgglomerativeClustering(n_clusters=num_speakers).fit(embeddings_array)

    diarization = []
    for i, label in enumerate(clustering.labels_):
        diarization.append({
            "start": timestamps[i]["start"],
            "end": timestamps[i]["end"],
            "speaker": int(label),
        })

    elapsed = time.time() - start
    _dlog(f"[Diarization] Completed in {elapsed:.2f}s "
          f"({len(diarization)} windows, {num_speakers} speakers)")
    return diarization


def assign_speakers_to_segments(segments, diarization):
    """
    Assign a speaker label to each transcription segment based on diarization.

    For every segment the function finds all overlapping diarization windows and
    picks the speaker that covers the most time (majority vote by overlap duration).

    Args:
        segments: List of ``{start, end, text, ...}`` transcription segments.
        diarization: List of ``{start, end, speaker}`` from ``diarize_with_speechbrain``.

    Returns:
        The *same* segments list, with an added ``speaker`` key on each entry.
    """
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        # Accumulate overlap time per speaker
        speaker_overlap = defaultdict(float)
        for d in diarization:
            overlap_start = max(seg_start, d["start"])
            overlap_end = min(seg_end, d["end"])
            if overlap_start < overlap_end:
                speaker_overlap[d["speaker"]] += overlap_end - overlap_start

        if speaker_overlap:
            seg["speaker"] = max(speaker_overlap, key=speaker_overlap.get)
        else:
            # No diarization window overlaps this segment; mark unknown
            seg["speaker"] = -1

    return segments

