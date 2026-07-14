import time
import numpy as np
from nvidia_dll_shim import enable

enable(verbose=True)
from faster_whisper import WhisperModel

audio = np.zeros(16000, dtype=np.float32)  # 1s silence @16k


def run(compute_type):
    print(f">>> cuda/{compute_type} loading...", flush=True)
    m = WhisperModel("tiny", device="cuda", compute_type=compute_type)
    t = time.time()
    segs, info = m.transcribe(audio, language="de")
    n = len(list(segs))
    print(f"    OK cuda/{compute_type}: transcribe {time.time() - t:.2f}s (segs={n})", flush=True)


run("float16")
run("int8_float16")
print("GPU FIX CONFIRMED", flush=True)
