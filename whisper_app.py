#!/usr/bin/env python3
"""
Whisper Transcription App
  Tab 1 — Transcribe an audio/video file
  Tab 2 — Record system audio via ScreenCaptureKit, then transcribe

Run via:
  uv run whisper_app.py
  ./whisper-app                (shim that wraps the above)
"""

import setproctitle
setproctitle.setproctitle("whisper-app")

import json
import os
import queue
import re
import signal
import subprocess
import threading
import time as _time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from transcribe_pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_DIR = Path(__file__).parent
CAPTURE_BUNDLE = REPO_DIR / "AudioCapture.app"
CAPTURE_BIN = CAPTURE_BUNDLE / "Contents" / "MacOS" / "AudioCapture"
SETTINGS_FILE = REPO_DIR / ".app_settings.json"

# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------

_DEFAULTS = {"language": "en", "num_speakers": 2, "model": "large-v2", "gpu_ratio": 0.78}


def _load_settings() -> dict:
    try:
        return {**_DEFAULTS, **json.loads(SETTINGS_FILE.read_text())}
    except Exception:
        return dict(_DEFAULTS)


def _save_settings(language, num_speakers, model, gpu_ratio):
    try:
        SETTINGS_FILE.write_text(json.dumps({
            "language": language,
            "num_speakers": int(num_speakers),
            "model": model,
            "gpu_ratio": float(gpu_ratio),
        }, indent=2))
    except Exception:
        pass


_settings = _load_settings()

# ---------------------------------------------------------------------------
# Global recording state  (single-user local app)
# ---------------------------------------------------------------------------

_rec_lock = threading.Lock()
_rec_pid: int | None = None
_rec_wav: str | None = None

# ---------------------------------------------------------------------------
# Resource plot helpers
# ---------------------------------------------------------------------------

_RESOURCE_RE = re.compile(
    r"\[Resources\] CPU ([\d.]+)%(?:\s+GPU ([\d.]+)%)?\s+RAM ([\d.]+)/([\d.]+) GB\s+SWAP ([\d.]+) GB"
)


def _make_resource_plot(samples: list) -> plt.Figure | None:
    """samples: list of (elapsed_s, cpu_pct, gpu_pct|None, ram_used_gb, ram_total_gb, swap_gb)"""
    if not samples:
        return None
    times   = [s[0] for s in samples]
    cpu     = [s[1] for s in samples]
    gpu     = [s[2] for s in samples]
    ram_gb  = [s[3] for s in samples]
    ram_tot = samples[-1][4]
    swap_gb = [s[5] for s in samples]
    has_gpu = any(g is not None for g in gpu)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.5), sharex=True)
    fig.patch.set_facecolor("#f9f9f9")

    ax1.plot(times, cpu, color="#2196F3", linewidth=2, label="CPU")
    if has_gpu:
        gpu_vals = [g if g is not None else float("nan") for g in gpu]
        ax1.plot(times, gpu_vals, color="#4CAF50", linewidth=2, label="GPU")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylabel("CPU / GPU %" if has_gpu else "CPU %")
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.25)
    ax1.set_facecolor("#f9f9f9")

    ax2.plot(times, ram_gb,  color="#FF9800", linewidth=2, label="RAM used")
    ax2.plot(times, swap_gb, color="#E53935", linewidth=2, label="Swap", linestyle="--")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_ylabel("GB")
    ax2.set_ylim(0, ram_tot * 1.05)
    ax2.set_xlabel("Elapsed (s)")
    ax2.grid(True, alpha=0.25)
    ax2.set_facecolor("#f9f9f9")

    fig.tight_layout(pad=1.0)
    return fig


# ---------------------------------------------------------------------------
# Shared settings component builder
# ---------------------------------------------------------------------------

def settings_row():
    language = gr.Dropdown(
        ["en", "ru", "de", "fr", "es", "ja", "zh", "uk"],
        value=_settings["language"], label="Language",
    )
    num_speakers = gr.Slider(1, 10, value=_settings["num_speakers"], step=1, label="Speakers")
    model = gr.Dropdown(
        ["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"],
        value=_settings["model"], label="Model",
    )
    gpu_ratio = gr.Slider(0.5, 0.95, value=_settings["gpu_ratio"], step=0.05, label="GPU share (MLX chunk)")
    return language, num_speakers, model, gpu_ratio


# ---------------------------------------------------------------------------
# Pipeline runner — streams (transcript, log_text, plot) tuples
# ---------------------------------------------------------------------------

def _run_pipeline_streaming(input_path, language, num_speakers, model_size, gpu_ratio=0.78,
                            mic_path=None):
    log_lines: list[str] = []
    resource_samples: list[tuple] = []
    result_holder: list[dict] = []
    error_holder: list[str] = []
    progress_q: queue.Queue[str] = queue.Queue()
    t0 = _time.time()

    def worker():
        try:
            result = run_pipeline(
                input_path=input_path,
                language=language,
                num_speakers=int(num_speakers),
                model_size=model_size,
                gpu_ratio=float(gpu_ratio),
                log=lambda msg: progress_q.put(msg),
                mic_path=mic_path,
            )
            result_holder.append(result)
        except Exception as exc:
            error_holder.append(str(exc))
        finally:
            progress_q.put(None)  # sentinel

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while True:
        try:
            msg = progress_q.get(timeout=2)
        except queue.Empty:
            yield "", "\n".join(log_lines), _make_resource_plot(resource_samples)
            continue

        if msg is None:
            break

        m = _RESOURCE_RE.match(msg)
        if m:
            elapsed = _time.time() - t0
            resource_samples.append((
                elapsed,
                float(m.group(1)),                           # cpu
                float(m.group(2)) if m.group(2) else None,  # gpu (optional)
                float(m.group(3)),                           # ram used
                float(m.group(4)),                           # ram total
                float(m.group(5)),                           # swap used
            ))
        else:
            log_lines.append(msg)

        yield "", "\n".join(log_lines), _make_resource_plot(resource_samples)

    t.join()

    plot = _make_resource_plot(resource_samples)
    if error_holder:
        yield f"Error: {error_holder[0]}", "\n".join(log_lines), plot
    elif result_holder:
        yield result_holder[0]["labeled"], "\n".join(log_lines), plot
    else:
        yield "No result returned.", "\n".join(log_lines), plot


# ---------------------------------------------------------------------------
# Tab 1 — File transcription
# ---------------------------------------------------------------------------

def transcribe_file(audio_path, language, num_speakers, model_size, gpu_ratio):
    if not audio_path:
        yield "No file provided.", "", None
        return
    _save_settings(language, num_speakers, model_size, gpu_ratio)
    yield from _run_pipeline_streaming(audio_path, language, num_speakers, model_size, gpu_ratio)


# ---------------------------------------------------------------------------
# Tab 2 — Record & transcribe
# ---------------------------------------------------------------------------

def _find_capture_pid(wav: str) -> int | None:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", f"AudioCapture {re.escape(wav)}"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()
        return int(out[0]) if out else None
    except subprocess.CalledProcessError:
        return None


def start_recording():
    global _rec_pid, _rec_wav

    if not CAPTURE_BIN.exists():
        msg = (
            f"Bundle not found: {CAPTURE_BUNDLE}\n"
            "Run:  ./build_audio_capture.sh"
        )
        return msg, gr.update(interactive=True), gr.update(interactive=False)

    with _rec_lock:
        if _rec_pid is not None:
            return "Already recording.", gr.update(interactive=False), gr.update(interactive=True)

        wav = str(REPO_DIR / "last_recording.wav")
        # Launch via LaunchServices (`open -n -a`) so the AudioCapture.app bundle
        # gets its own TCC attribution rather than inheriting from this python
        # interpreter, whose grants are denied under the launchd context.
        subprocess.run(
            ["/usr/bin/open", "-n", "-a", str(CAPTURE_BUNDLE), "--args", wav],
            check=True,
        )
        pid = None
        for _ in range(30):
            _time.sleep(0.1)
            pid = _find_capture_pid(wav)
            if pid:
                break
        if pid is None:
            return ("Could not start AudioCapture (LaunchServices spawn failed).",
                    gr.update(interactive=True), gr.update(interactive=False))

        _rec_pid = pid
        _rec_wav = wav

    return (
        "Recording system audio…  click Stop when done.",
        gr.update(interactive=False),
        gr.update(interactive=True),
    )


def stop_and_transcribe(language, num_speakers, model_size, gpu_ratio):
    global _rec_pid, _rec_wav

    with _rec_lock:
        pid = _rec_pid
        wav = _rec_wav
        _rec_pid = None
        _rec_wav = None

    if pid is None:
        yield "Not recording.", "", None, None, gr.update(interactive=True), gr.update(interactive=False)
        return

    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    for _ in range(20):
        _time.sleep(0.25)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    if not wav or not os.path.exists(wav) or os.path.getsize(wav) < 1024:
        yield "No audio captured.", "", None, None, gr.update(interactive=True), gr.update(interactive=False)
        return

    wav_path = Path(wav)
    mic_wav_path = wav_path.with_stem(wav_path.stem + "_mic")
    mic_wav = str(mic_wav_path) if mic_wav_path.exists() and mic_wav_path.stat().st_size > 1024 else None

    # Create a mixed preview file for the audio player (both voices, natural levels)
    preview_wav = str(wav_path.with_stem(wav_path.stem + "_preview"))
    if mic_wav:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", wav, "-i", mic_wav,
             "-filter_complex", "amix=inputs=2:normalize=0:duration=longest",
             preview_wav],
            capture_output=True,
        )
        playback_file = preview_wav if r.returncode == 0 else wav
    else:
        playback_file = wav

    # Show audio immediately — don't wait for transcription to finish
    yield "", "", None, playback_file, gr.update(interactive=False), gr.update(interactive=True)

    _save_settings(language, num_speakers, model_size, gpu_ratio)
    transcript = ""
    log = ""
    plot = None
    for transcript, log, plot in _run_pipeline_streaming(
        wav, language, num_speakers, model_size, gpu_ratio, mic_path=mic_wav
    ):
        yield transcript, log, plot, gr.update(), gr.update(interactive=False), gr.update(interactive=True)

    yield transcript, log, plot, gr.update(), gr.update(interactive=True), gr.update(interactive=False)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Whisper Transcription") as demo:
    gr.Markdown("# Whisper Transcription")

    with gr.Tabs():

        # ── Tab 1: File ──────────────────────────────────────────────────
        with gr.Tab("Transcribe File"):
            with gr.Row():
                f_lang, f_speakers, f_model, f_gpu = settings_row()

            f_audio = gr.File(label="Drop audio or video file here", file_types=["audio", "video", ".webm"])
            f_audio_play = gr.Audio(label="Uploaded file", interactive=False)
            f_btn = gr.Button("Transcribe", variant="primary")
            f_out = gr.Textbox(label="Transcript", lines=20)
            f_log = gr.Textbox(label="Progress", lines=14, interactive=False)
            f_plot = gr.Plot(label="CPU / RAM during transcription")

            f_audio.change(lambda f: f, inputs=[f_audio], outputs=[f_audio_play])

            f_btn.click(
                transcribe_file,
                inputs=[f_audio, f_lang, f_speakers, f_model, f_gpu],
                outputs=[f_out, f_log, f_plot],
            )

        # ── Tab 2: Record ─────────────────────────────────────────────────
        with gr.Tab("Record & Transcribe"):
            gr.Markdown(
                "Captures **system audio** (calls, browser, any app) via ScreenCaptureKit.  \n"
                "macOS will ask for **Screen & System Audio Recording** permission on first run."
            )

            with gr.Row():
                r_lang, r_speakers, r_model, r_gpu = settings_row()

            r_status = gr.Textbox(value="Ready", label="Status", interactive=False)

            with gr.Row():
                r_start = gr.Button("Start Recording", variant="primary")
                r_stop = gr.Button("Stop & Transcribe", variant="stop", interactive=False)

            r_audio_play = gr.Audio(label="Recorded audio (last_recording.wav)", interactive=False)
            r_out = gr.Textbox(label="Transcript", lines=20)
            r_log = gr.Textbox(label="Progress", lines=14, interactive=False)
            r_plot = gr.Plot(label="CPU / RAM during transcription")

            r_start.click(
                start_recording,
                outputs=[r_status, r_start, r_stop],
            )
            r_stop.click(
                stop_and_transcribe,
                inputs=[r_lang, r_speakers, r_model, r_gpu],
                outputs=[r_out, r_log, r_plot, r_audio_play, r_start, r_stop],
            )


def main():
    demo.launch(theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
