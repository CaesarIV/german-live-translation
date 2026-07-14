"""Live German -> English captions from system audio, STREAMING (sentence-level).

faster-whisper transcribes a growing buffer with LocalAgreement-2; complete stable
German sentences are translated whole by Opus-MT. No more 5 s chunk chopping.

Run:  python live_captions_translate.py       (or double-click LiveTranslate.bat)
      python live_captions_translate.py 30     (stop after 30 s)
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import sys
import time
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from nvidia_dll_shim import enable

enable()
from faster_whisper import WhisperModel
from mt import Translator
from audio_source import make_source
from streaming import StreamingTranscriber

# ==================== TUNABLES ====================
STEP_SEC = 1.5           # how often we re-transcribe (lower = snappier commits, more compute)
MAX_BUF_SEC = 18.0       # force-flush run-on speech past this
ASR_MODEL = "large-v3"
SHOW_GERMAN = True       # False = English only
AUDIO_SOURCE = "loopback"
# =================================================

MAX_RUNTIME = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

src = make_source(AUDIO_SOURCE)
print(f"# source: {src.name}  step={STEP_SEC}s", flush=True)
print("# loading ASR + MT ...", flush=True)
asr = WhisperModel(ASR_MODEL, device="cuda", compute_type="float16")
mt = Translator(device="cuda", compute_type="int8_float16")
print("# LIVE (DE->EN, sentence-streaming) — Ctrl+C to stop\n", flush=True)

t0 = time.monotonic()


def on_sentence(de):
    t_mt = time.monotonic()
    en = mt.translate(de)
    mt_ms = (time.monotonic() - t_mt) * 1000
    stamp = f"[{time.monotonic()-t0:6.1f}s]"
    if SHOW_GERMAN:
        print(f"{stamp} DE: {de}", flush=True)
        print(f"{'':9} EN: {en}   (mt {mt_ms:.0f}ms)", flush=True)
    else:
        print(f"{stamp} {en}", flush=True)


stream = StreamingTranscriber(asr, step_sec=STEP_SEC, max_buf_sec=MAX_BUF_SEC, on_sentence=on_sentence)

silent_s = 0.0
with src:
    try:
        while True:
            if MAX_RUNTIME and time.monotonic() - t0 > MAX_RUNTIME:
                break
            chunk = src.read(0.5)
            stream.add_audio(chunk)
            peak = float(np.abs(chunk).max()) if chunk.size else 0.0
            if peak < 0.003:
                silent_s += 0.5
                if silent_s >= 10.0:
                    print("# …listening (silent — nothing playing)", flush=True)
                    silent_s = 0.0
            else:
                silent_s = 0.0
    except KeyboardInterrupt:
        pass
print("\n# stopped", flush=True)
