"""Capture N seconds from VB-CABLE Output and transcribe the German with large-v3.
First end-to-end proof: system audio -> German caption.  Usage: python capture_and_transcribe.py [seconds]
"""
import sys
import numpy as np
import sounddevice as sd
from nvidia_dll_shim import enable

enable()
from faster_whisper import WhisperModel

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0


def find_cable_output():
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and "cable output" in d["name"].lower():
            return i, d
    return None, None


dev, info = find_cable_output()
if dev is None:
    print("CABLE Output capture device not found"); sys.exit(1)

sr = int(info["default_samplerate"])
ch = min(2, info["max_input_channels"])
print(f"capture [{dev}] {info['name'].strip()}  sr={sr} ch={ch}  recording {SECS:.0f}s ...", flush=True)

rec = sd.rec(int(SECS * sr), samplerate=sr, channels=ch, dtype="float32", device=dev)
sd.wait()

peak = float(np.max(np.abs(rec))) if rec.size else 0.0
rms_db = 20 * np.log10(float(np.sqrt(np.mean(rec ** 2))) + 1e-9)
print(f"peak={peak:.4f}  rms={rms_db:.1f}dB  -> "
      f"{'SIGNAL OK' if peak > 0.005 else 'SILENCE (check routing / Sonar)'}", flush=True)
if peak <= 0.005:
    sys.exit(0)

# downmix to mono, resample sr -> 16k (linear; fine for a smoke test)
mono = rec.mean(axis=1)
n16 = int(len(mono) * 16000 / sr)
mono16 = np.interp(np.linspace(0, 1, n16, endpoint=False),
                   np.linspace(0, 1, len(mono), endpoint=False), mono).astype(np.float32)

print("loading large-v3 ...", flush=True)
m = WhisperModel("large-v3", device="cuda", compute_type="float16")
segs, i2 = m.transcribe(mono16, language="de", beam_size=5)
text = " ".join(s.text.strip() for s in segs)
print(f"\n[lang={i2.language} p={i2.language_probability:.2f}] GERMAN TRANSCRIPT:", flush=True)
print(text or "(no speech detected)", flush=True)
