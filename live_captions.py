"""Live German captions from system audio. v1: chunked ASR (no translation).

Run continuously:   python live_captions.py
Run for N seconds:  python live_captions.py 30
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from nvidia_dll_shim import enable

enable()
from faster_whisper import WhisperModel
from audio_source import make_source

# ==================== TUNABLES ====================
CHUNK_SEC = 5.0
MODEL = "large-v3"
TASK = "transcribe"               # "translate" -> rough English (Opus-MT variant is better)
AUDIO_SOURCE = "loopback"         # "loopback" or "cable"
# =================================================

MAX_RUNTIME = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

src = make_source(AUDIO_SOURCE)
print(f"# source: {src.name}  chunk={CHUNK_SEC}s  model={MODEL}  task={TASK}", flush=True)
print("# loading model ...", flush=True)
model = WhisperModel(MODEL, device="cuda", compute_type="float16")
print("# LIVE — Ctrl+C to stop\n", flush=True)

t0 = time.monotonic()
with src:
    try:
        while True:
            if MAX_RUNTIME and time.monotonic() - t0 > MAX_RUNTIME:
                break
            a16 = src.read(CHUNK_SEC)
            segs, _ = model.transcribe(a16, language="de", task=TASK, beam_size=1, vad_filter=True)
            text = " ".join(s.text.strip() for s in segs).strip()
            if text:
                print(f"[{time.monotonic()-t0:5.1f}s] {text}", flush=True)
    except KeyboardInterrupt:
        pass
print("\n# stopped", flush=True)
