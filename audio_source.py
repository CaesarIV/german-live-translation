"""Audio capture sources -> 16 kHz mono float32 chunks via .read(seconds).

- LoopbackSource (default): WASAPI loopback of the current default speaker.
  Captures whatever you hear, no rerouting. Runs a background capture thread so
  audio is drained continuously (no gaps / dropped speech while ASR+MT run).
- CableSource: VB-CABLE Output via sounddevice (fallback / per-app capture).
"""
import threading
import queue
import warnings
import numpy as np

# soundcard spams "data discontinuity in recording" — silence it (harmless with threaded capture)
warnings.filterwarnings("ignore", message="data discontinuity in recording")
try:
    import soundcard as _sc
    warnings.filterwarnings("ignore", category=_sc.SoundcardRuntimeWarning)
except Exception:
    pass

TARGET_SR = 16000


def _to16k_mono(data, sr):
    mono = data.mean(axis=1) if data.ndim > 1 else data
    if sr == TARGET_SR:
        return mono.astype(np.float32)
    n = int(len(mono) * TARGET_SR / sr)
    if n <= 0:
        return np.zeros(0, np.float32)
    return np.interp(np.linspace(0, 1, n, endpoint=False),
                     np.linspace(0, 1, len(mono), endpoint=False), mono).astype(np.float32)


class LoopbackSource:
    def __init__(self, sr=48000, block_ms=100):
        import soundcard as sc
        self.sr = sr
        self.block = int(sr * block_ms / 1000)
        spk = sc.default_speaker()
        self._mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
        self.name = f"loopback: {spk.name}"
        self._q = queue.Queue()
        self._buf = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        with self._mic.recorder(samplerate=self.sr, channels=2) as rec:
            while not self._stop.is_set():
                self._q.put(np.asarray(rec.record(numframes=self.block)))

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def read(self, seconds):
        need = int(self.sr * seconds)
        while sum(len(b) for b in self._buf) < need:
            self._buf.append(self._q.get())
        data = np.concatenate(self._buf, axis=0)
        out, rem = data[:need], data[need:]
        self._buf = [rem] if len(rem) else []
        return _to16k_mono(out, self.sr)


class CableSource:
    def __init__(self):
        import sounddevice as sd
        self.sd = sd
        dev = info = None
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and "cable output" in d["name"].lower():
                dev, info = i, d
                break
        if dev is None:
            raise SystemExit("CABLE Output not found")
        self.dev, self.sr = dev, int(info["default_samplerate"])
        self.ch = min(2, info["max_input_channels"])
        self.name = f"cable: {info['name'].strip()}"
        self.q = queue.Queue()
        self._stream = None

    def __enter__(self):
        def cb(indata, frames, t, status):
            self.q.put(indata.copy())
        self._stream = self.sd.InputStream(device=self.dev, samplerate=self.sr, channels=self.ch,
                                           dtype="float32", callback=cb, blocksize=int(self.sr * 0.1))
        self._stream.start()
        return self

    def __exit__(self, *a):
        self._stream.stop()

    def read(self, seconds):
        import time
        need = int(self.sr * seconds)
        buf = np.zeros((0, self.ch), np.float32)
        while len(buf) < need:
            while not self.q.empty():
                buf = np.concatenate([buf, self.q.get()], axis=0)
            if len(buf) < need:
                time.sleep(0.02)
        return _to16k_mono(buf[:need], self.sr)


def make_source(kind="loopback"):
    return LoopbackSource() if kind == "loopback" else CableSource()
