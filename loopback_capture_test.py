import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import sys
import numpy as np
import soundcard as sc
from nvidia_dll_shim import enable

enable()
from faster_whisper import WhisperModel

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
SR = 48000

spk = sc.default_speaker()
print(f"loopback of DEFAULT speaker: {spk.name}", flush=True)
mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
print(f"capturing {SECS:.0f}s (make sure German audio is playing to this output)...", flush=True)
with mic.recorder(samplerate=SR, channels=2) as rec:
    data = rec.record(numframes=int(SR * SECS))

peak = float(np.max(np.abs(data)))
print(f"peak={peak:.4f} -> {'SIGNAL OK' if peak > 0.005 else 'SILENCE (nothing playing to this endpoint)'}", flush=True)
if peak <= 0.005:
    sys.exit(0)

mono = data.mean(axis=1)
n16 = int(len(mono) * 16000 / SR)
a16 = np.interp(np.linspace(0, 1, n16, endpoint=False),
                np.linspace(0, 1, len(mono), endpoint=False), mono).astype(np.float32)
m = WhisperModel("large-v3", device="cuda", compute_type="float16")
segs, i2 = m.transcribe(a16, language="de", beam_size=5)
print(f"\n[lang={i2.language} p={i2.language_probability:.2f}] GERMAN via LOOPBACK:", flush=True)
print(" ".join(s.text.strip() for s in segs) or "(no speech detected)", flush=True)
