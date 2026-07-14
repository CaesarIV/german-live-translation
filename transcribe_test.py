"""Step 2 quality gate: run faster-whisper on a German clip, both tasks, with timing.

Usage:  python transcribe_test.py <audio_file> [model]
        model defaults to large-v3
"""
import sys
import time
from nvidia_dll_shim import enable

enable()
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

AUDIO = sys.argv[1] if len(sys.argv) > 1 else "sample_de.mp3"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "large-v3"
MAX_SEC = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0

audio = decode_audio(AUDIO, sampling_rate=16000)
if MAX_SEC and len(audio) > int(MAX_SEC * 16000):
    audio = audio[: int(MAX_SEC * 16000)]
dur = len(audio) / 16000
print(f"audio: {AUDIO}  ({dur:.1f}s used)", flush=True)

print(f"loading {MODEL} on cuda/float16 ...", flush=True)
t0 = time.time()
m = WhisperModel(MODEL, device="cuda", compute_type="float16")
print(f"model ready in {time.time()-t0:.1f}s", flush=True)

for task in ("transcribe", "translate"):
    print(f"\n===== {task.upper()} =====", flush=True)
    t0 = time.time()
    segs, info = m.transcribe(audio, language="de", task=task, beam_size=5)
    segs = list(segs)
    dt = time.time() - t0
    rtf = dt / dur if dur else float("nan")
    text = " ".join(s.text.strip() for s in segs)
    print(f"[lang={info.language} p={info.language_probability:.2f}]  "
          f"compute={dt:.2f}s  RTF={rtf:.3f}", flush=True)
    print(text or "(no speech detected)", flush=True)
