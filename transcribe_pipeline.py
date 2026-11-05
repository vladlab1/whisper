"""
Clean pipeline wrapper around transcribe_workers.py.
Used by app.py; the original notebook is unchanged.
"""

import os
import re
import shutil
import subprocess
import threading
import time
from multiprocessing import Process, Queue
from typing import Callable, Optional


def _get_gpu_pct() -> float | None:
    """Read Apple Silicon GPU utilization via ioreg (no sudo required)."""
    try:
        out = subprocess.run(
            ["ioreg", "-rc", "IOAccelerator", "-w0"],
            capture_output=True, text=True, timeout=3,
        ).stdout
        for line in out.splitlines():
            if "Device Utilization" in line:
                m = re.search(r'"Device Utilization %"\s*=\s*(\d+)', line)
                if m:
                    return float(m.group(1))
    except Exception:
        pass
    return None

from transcribe_workers import (
    assign_speakers_to_segments,
    diarize_with_speechbrain,
    transcribe_with_faster_whisper,
    transcribe_with_mlx,
)


def detect_backends() -> dict:
    backends = {}
    try:
        import mlx_whisper  # noqa: F401
        backends["mlx"] = True
    except ImportError:
        backends["mlx"] = False
    try:
        from faster_whisper import WhisperModel  # noqa: F401
        backends["faster_whisper"] = True
    except ImportError:
        backends["faster_whisper"] = False
    return backends


def preprocess_audio(input_path: str, output_path: str, normalize: bool = False) -> float:
    """Convert any audio/video to 16 kHz mono WAV. Returns duration in seconds."""
    cmd = ["ffmpeg", "-y", "-vn", "-i", input_path, "-ar", "16000", "-ac", "1"]
    if normalize:
        cmd += ["-af", "loudnorm"]
    cmd.append(output_path)
    subprocess.run(cmd, check=True, capture_output=True)
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", output_path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def split_audio(wav_path: str, temp_dir: str,
                overlap_seconds: int = 2, gpu_ratio: float = 0.78):
    """Split 16 kHz WAV into 2 overlapping chunks. Returns (chunks, split_point)."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
        capture_output=True, text=True, check=True,
    )
    duration = float(result.stdout.strip())
    split_point = duration * (1 - gpu_ratio)

    chunk1 = os.path.join(temp_dir, "chunk_1.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path,
         "-t", str(split_point + overlap_seconds), "-c", "copy", chunk1],
        check=True, capture_output=True,
    )

    chunk2 = os.path.join(temp_dir, "chunk_2.wav")
    chunk2_start = max(0.0, split_point - overlap_seconds)
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path,
         "-ss", str(chunk2_start), "-c", "copy", chunk2],
        check=True, capture_output=True,
    )

    chunks = [
        (chunk1, 0.0, split_point + overlap_seconds),
        (chunk2, chunk2_start, duration),
    ]
    return chunks, split_point


def run_parallel_transcription(chunks, language: str, model_size: str, backends: dict, log_q=None):
    """Transcribe chunk 1 on CPU (faster-whisper), chunk 2 on MLX/Neural Engine."""
    q = Queue()
    chunk1_path = chunks[0][0]
    chunk2_path = chunks[1][0]

    p1 = Process(target=transcribe_with_faster_whisper,
                 args=(chunk1_path, language, q, 1, model_size, "cpu", "int8", log_q))

    if backends.get("mlx"):
        p2 = Process(target=transcribe_with_mlx,
                     args=(chunk2_path, language, q, 2, model_size, log_q))
    else:
        p2 = Process(target=transcribe_with_faster_whisper,
                     args=(chunk2_path, language, q, 2, model_size, "cpu", "int8", log_q))

    t0 = time.time()
    for p in (p1, p2):
        p.start()

    results = []
    while len(results) < 2:
        try:
            results.append(q.get(timeout=10))
        except Exception:
            if not any(p.is_alive() for p in (p1, p2)):
                break

    for p in (p1, p2):
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
            p.join(timeout=2)

    results.sort(key=lambda r: r.get("chunk_id", 0))
    return results, time.time() - t0


def merge_transcriptions(results, chunks, split_point: float, overlap_seconds: int):
    """Drop overlap region and sort all segments by time."""
    if any("error" in r for r in results):
        errors = [r["error"] for r in results if "error" in r]
        raise RuntimeError(f"Transcription failed: {errors}")

    merged = []
    chunk1 = next(r for r in results if r["chunk_id"] == 1)
    offset1 = chunks[0][1]
    for seg in chunk1["segments"]:
        abs_start = seg["start"] + offset1
        if abs_start < split_point:
            merged.append({"start": abs_start, "end": seg["end"] + offset1, "text": seg["text"]})

    chunk2 = next(r for r in results if r["chunk_id"] == 2)
    offset2 = chunks[1][1]
    for seg in chunk2["segments"]:
        abs_start = seg["start"] + offset2
        if abs_start >= split_point:
            merged.append({"start": abs_start, "end": seg["end"] + offset2, "text": seg["text"]})

    merged.sort(key=lambda s: s["start"])
    return merged


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"



def format_with_speakers(segments: list) -> str:
    """Build speaker-labeled transcript with timestamps."""
    lines = []
    prev_speaker = None
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        speaker = seg.get("speaker", -1)
        ts = _fmt_time(seg["start"])
        if speaker != prev_speaker:
            label = f"Speaker {speaker + 1}" if speaker >= 0 else "Unknown"
            lines.append(f"\n[{label} {ts}] {text}")
            prev_speaker = speaker
        else:
            lines.append(f" {text}")
    return "".join(lines).strip()


def _extract_speech(wav_path: str, out_path: str):
    """Write only the voiced parts of a 16 kHz mono WAV to out_path (Silero VAD).

    Returns (speech_seconds, timestamp_map); the map is None when there is no speech.
    """
    import numpy as np
    import soundfile as sf
    from faster_whisper.vad import SpeechTimestampsMap, get_speech_timestamps

    audio, sr = sf.read(wav_path, dtype="float32")
    chunks = get_speech_timestamps(audio, sampling_rate=sr)
    if not chunks:
        return 0.0, None
    speech = np.concatenate([audio[c["start"]:c["end"]] for c in chunks])
    sf.write(out_path, speech, sr)
    return len(speech) / sr, SpeechTimestampsMap(chunks, sr)


def _restore_timestamps(segments: list, ts_map) -> list:
    return [{"start": ts_map.get_original_time(s["start"]),
             "end": ts_map.get_original_time(s["end"], is_end=True),
             "text": s["text"]}
            for s in segments]


# Text Whisper learned from subtitle credits and video outros. It emits it on
# silence or noise, sometimes glued to real speech. Several phrases come from
# Vexa's hand-verified lists (github.com/Vexa-ai/vexa, Apache-2.0).
# Credits are unambiguous, so they are cut out wherever they appear.
_HALLUCINATION_SPAN_RE = re.compile(
    r"(?:редактор\s+субтитров|корректор)\s+[а-яёa-z]\.\s?[а-яёa-z]+|"
    r"субтитр\w*\s+(?:сделал|создавал|подготовил|делал|подогнал)\w*\s+(?:«[^»]*»|\S+)|"
    r"спасибо\s+за\s+субтитры(?:\s+[а-яё]+){0,2}\s*!?|"
    r"dimatorzok|subtitles\s+by\s+the\s+amara\.org\s+community|♪+",
    re.IGNORECASE,
)
# Outros are made of ordinary words, so they are dropped only as a whole sentence.
_HALLUCINATION_SENTENCE_RE = re.compile(
    r"(?:продолжение\s+следует|спасибо\s+за\s+просмотр|динамичная\s+музыка|"
    r"подписывайтесь\s+на\s+(?:(?:наш|мой)\s+)?канал|ставьте\s+лайк\w*(?:\s+и\s+подписывайтесь\w*)?|"
    r"thanks?\s+(?:you\s+)?for\s+watching(?:,?\s+and\s+i'?ll\s+see\s+you\s+(?:in\s+the\s+)?next\s+(?:time|video))?|"
    r"please\s+subscribe(?:\s+to\s+(?:my|our|the)\s+channel)?)[\s.!…]*",
    re.IGNORECASE,
)


def _strip_hallucinations(text: str) -> str:
    text = re.sub(r"\s{2,}", " ", _HALLUCINATION_SPAN_RE.sub(" ", text)).strip()
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    return " ".join(x for x in sentences if not _HALLUCINATION_SENTENCE_RE.fullmatch(x))


def _drop_known_hallucinations(segments: list, label: str, _log: Callable) -> list:
    kept, trimmed = [], 0
    for seg in segments:
        text = _strip_hallucinations(seg["text"])
        if text == seg["text"].strip():
            kept.append(seg)
        elif re.search(r"\w", text):
            kept.append({**seg, "text": " " + text})
            trimmed += 1
    if trimmed:
        _log(f"[{label}] Cut known hallucinations out of {trimmed} segment(s)")
    if len(kept) != len(segments):
        _log(f"[{label}] Dropped {len(segments) - len(kept)} known hallucination segment(s)")
    return kept


def _transcribe_wav(wav_path: str, duration: float, label: str,
                    language: str, model_size: str, backends: dict,
                    gpu_ratio: float, overlap_seconds: int,
                    log_q: Queue, _log: Callable) -> list:
    """Transcribe only the voiced parts of a 16 kHz mono WAV. Returns [{start, end, text}]."""
    speech_wav = os.path.join(os.path.dirname(wav_path), f"{label}_speech.wav")
    speech_duration, ts_map = _extract_speech(wav_path, speech_wav)
    _log(f"[{label}] Speech: {speech_duration / 60:.1f} of {duration / 60:.1f} min")
    if ts_map is None:
        return []
    segs = _transcribe_speech(speech_wav, speech_duration, label, language, model_size,
                              backends, gpu_ratio, overlap_seconds, log_q, _log)
    return _drop_known_hallucinations(_restore_timestamps(segs, ts_map), label, _log)


def _transcribe_speech(wav_path: str, duration: float, label: str,
                       language: str, model_size: str, backends: dict,
                       gpu_ratio: float, overlap_seconds: int,
                       log_q: Queue, _log: Callable) -> list:
    """Transcribe a preprocessed 16 kHz mono WAV. Returns [{start, end, text}]."""
    MIN_SPLIT_DURATION = 180  # seconds
    backend_names = [k for k, v in backends.items() if v]

    if duration < MIN_SPLIT_DURATION:
        _log(f"[{label}] Short ({duration/60:.1f} min) — single chunk via {backend_names[0]}")
        result_q: Queue = Queue()
        if backends.get("mlx"):
            p = Process(target=transcribe_with_mlx,
                        args=(wav_path, language, result_q, 1, model_size, log_q))
        else:
            p = Process(target=transcribe_with_faster_whisper,
                        args=(wav_path, language, result_q, 1, model_size, "cpu", "int8", log_q))
        p.start()
        raw = None
        while p.is_alive() or not result_q.empty():
            try:
                _log(log_q.get(timeout=0.3))
            except Exception:
                pass
            if raw is None:
                try:
                    raw = result_q.get_nowait()
                except Exception:
                    pass
        if raw is None:
            try:
                raw = result_q.get(timeout=5)
            except Exception:
                raw = {"chunk_id": 1, "segments": []}
        p.join(timeout=5)
        if "error" in raw:
            raise RuntimeError(f"[{label}] transcription failed: {raw['error']}")
        segs = [{"start": s["start"], "end": s["end"], "text": s["text"]}
                for s in raw.get("segments", [])]
        return segs
    else:
        _log(f"[{label}] Splitting into 2 chunks...")
        # Use label-prefixed chunk names to avoid collision when called twice
        temp_dir = os.path.dirname(wav_path)
        orig_chunk1 = os.path.join(temp_dir, "chunk_1.wav")
        orig_chunk2 = os.path.join(temp_dir, "chunk_2.wav")
        safe1 = os.path.join(temp_dir, f"{label}_chunk_1.wav")
        safe2 = os.path.join(temp_dir, f"{label}_chunk_2.wav")

        chunks, split_point = split_audio(wav_path, temp_dir, overlap_seconds, gpu_ratio)
        os.rename(orig_chunk1, safe1)
        os.rename(orig_chunk2, safe2)
        chunks = [(safe1, chunks[0][1], chunks[0][2]), (safe2, chunks[1][1], chunks[1][2])]

        stop_drain = threading.Event()
        def _drain():
            while not stop_drain.is_set():
                try:
                    _log(log_q.get(timeout=0.3))
                except Exception:
                    pass
            while True:
                try:
                    _log(log_q.get_nowait())
                except Exception:
                    break

        drain = threading.Thread(target=_drain, daemon=True)
        drain.start()
        results, t = run_parallel_transcription(chunks, language, model_size, backends, log_q)
        stop_drain.set()
        drain.join(timeout=2)
        _log(f"[{label}] Done in {t:.0f}s")
        segs = merge_transcriptions(results, chunks, split_point, overlap_seconds)
        return segs


def run_pipeline(
    input_path: str,
    language: str = "en",
    num_speakers: int = 2,
    model_size: str = "large-v2",
    gpu_ratio: float = 0.78,
    overlap_seconds: int = 2,
    diarize_device: str = "mps",
    log: Optional[Callable[[str], None]] = None,
    mic_path: Optional[str] = None,
) -> dict:
    """
    Full transcription pipeline.

    Returns dict with keys:
      labeled        — speaker-labeled transcript (str)
      segments       — list of {start, end, text, speaker} dicts
      duration       — audio duration in seconds (float)
      processing_time — wall time in seconds (float)
    """
    def _log(msg: str):
        print(msg)
        if log:
            log(msg)

    t_start = time.time()
    temp_dir = input_path + "_pipeline_tmp"
    os.makedirs(temp_dir, exist_ok=True)

    # Background thread: sample CPU/GPU/RAM every 3 s
    stop_monitor = threading.Event()
    def _resource_monitor():
        try:
            import psutil
            psutil.cpu_percent()  # prime (first call always returns 0.0)
            while not stop_monitor.wait(3.0):
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory()
                swp = psutil.swap_memory()
                gpu = _get_gpu_pct()
                gpu_str = f"  GPU {gpu:.0f}%" if gpu is not None else ""
                _log(f"[Resources] CPU {cpu:.0f}%{gpu_str}  RAM {mem.used/1e9:.1f}/{mem.total/1e9:.1f} GB  SWAP {swp.used/1e9:.1f} GB")
        except ImportError:
            pass

    monitor = threading.Thread(target=_resource_monitor, daemon=True)
    monitor.start()

    try:
        backends = detect_backends()
        log_q: Queue = Queue()

        _log("Preprocessing audio...")
        sys_wav = os.path.join(temp_dir, "sys16k.wav")
        duration = preprocess_audio(input_path, sys_wav, normalize=True)
        _log(f"Duration: {duration / 60:.1f} min")

        if mic_path and os.path.exists(mic_path):
            # ── Dual-stream: transcribe each independently ──────────────────
            _log("Dual-stream mode: transcribing system audio and mic separately")
            mic_wav = os.path.join(temp_dir, "mic16k.wav")
            mic_duration = preprocess_audio(mic_path, mic_wav, normalize=True)
            _log(f"Mic duration: {mic_duration / 60:.1f} min")

            _log("Transcribing system audio...")
            sys_segments = _transcribe_wav(
                sys_wav, duration, "sys",
                language, model_size, backends, gpu_ratio, overlap_seconds, log_q, _log,
            )
            if sys_segments:
                _log(f"System audio: {len(sys_segments)} segments  "
                     f"[{_fmt_time(sys_segments[0]['start'])} – {_fmt_time(sys_segments[-1]['end'])}]")
            else:
                _log("System audio: 0 segments")

            _log("Transcribing mic audio...")
            mic_segments = _transcribe_wav(
                mic_wav, mic_duration, "mic",
                language, model_size, backends, gpu_ratio, overlap_seconds, log_q, _log,
            )
            if mic_segments:
                _log(f"Mic audio: {len(mic_segments)} segments  "
                     f"[{_fmt_time(mic_segments[0]['start'])} – {_fmt_time(mic_segments[-1]['end'])}]")
            else:
                _log("Mic audio: 0 segments")

            # The channel already identifies the local user, so diarization is only
            # needed to tell several remote speakers apart on the system audio.
            remote_speakers = max(1, num_speakers - 1)
            if remote_speakers > 1 and sys_segments:
                _log(f"Running speaker diarization on system audio ({remote_speakers} remote speakers)...")
                diarization = diarize_with_speechbrain(
                    sys_wav, remote_speakers, device=diarize_device, log=_log,
                )
                sys_segments = assign_speakers_to_segments(sys_segments, diarization)
            else:
                for seg in sys_segments:
                    seg["speaker"] = 0
            for seg in mic_segments:
                seg["speaker"] = remote_speakers
                seg["label"] = "Me"

            segments = sorted(sys_segments + mic_segments, key=lambda s: s["start"])
            _log(f"Combined: {len(segments)} segments")

            # Mix both for diarization (just needs both voices present)
            diarize_wav = os.path.join(temp_dir, "diarize16k.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", sys_wav, "-i", mic_wav,
                 "-filter_complex", "amix=inputs=2:normalize=0:duration=longest",
                 diarize_wav],
                capture_output=True, check=True,
            )
            wav_for_diarization = diarize_wav
        else:
            # ── Single-stream ───────────────────────────────────────────────
            segments = _transcribe_wav(
                sys_wav, duration, "audio",
                language, model_size, backends, gpu_ratio, overlap_seconds, log_q, _log,
            )
            _log(f"Got {len(segments)} segments")
            wav_for_diarization = sys_wav

        _log(f"Running speaker diarization ({num_speakers} speakers)...")
        diarization = diarize_with_speechbrain(
            wav_for_diarization, num_speakers, device=diarize_device, log=_log,
        )
        segments = assign_speakers_to_segments(segments, diarization)

        labeled = format_with_speakers(segments)
        processing_time = time.time() - t_start
        _log(f"Done in {processing_time:.0f}s  (RTF {processing_time / duration:.2f}x)")

        return {
            "labeled": labeled,
            "segments": segments,
            "duration": duration,
            "processing_time": processing_time,
        }
    finally:
        stop_monitor.set()
        monitor.join(timeout=5)
        shutil.rmtree(temp_dir, ignore_errors=True)
