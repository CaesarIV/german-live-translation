# Live German → English Captioning Pipeline — Plan & Architecture Options

Fully-local, low-latency **German system audio → English captions** on Windows 11.
No cloud APIs. This doc records the decision, the alternatives, and how to switch.

Last updated: 2026-07-14.

---

## Hardware / environment (verified)

- **GPU:** RTX 4070 Ti, 12 GB, compute capability 8.9 (Ada). Driver 610.47 / CUDA UMD 13.x → any CUDA 12/13 runtime runs; no driver ceiling.
- **No system CUDA toolkit** (`nvcc` absent) — fine, the libs come from pip wheels.
- **Python:** 3.11 (default via `py`), 3.10, 3.8 (too old). No conda.
- **CPU/RAM:** i9-10900KF, 96 GB.
- ⚠️ **Free VRAM before running models** — LM Studio / browsers were eating ~11 GB of 12 GB.

---

## The finding that reshaped the naive plan

The original idea ("Parakeet-TDT for streaming German + Canary for translation") does **not** work as written, and this is the load-bearing research result:

1. **Parakeet-TDT `0.6b-v2` is English-only.** It cannot transcribe German at all.
2. **Parakeet-TDT `0.6b-v3`** is multilingual (25 EU langs incl. German) **but ASR-only** (German→German text, no English) **and OFFLINE** — its "streaming" is *buffered re-decoding* at ~2–4 s latency, not true low-latency streaming.
3. **NeMo has ZERO native Windows support** (official 2.7.3 matrix: *"No support yet"*; blocker = Linux-only Triton). Any NeMo model ⇒ **WSL2**, which forces a Windows→Linux **audio bridge** for the captured system audio.
4. The genuinely low-latency German streaming model is a **different, newer** one: **`nemotron-3.5-asr-streaming-0.6b`** (2026-06-04).

---

## Model landscape (verified against HF cards / NeMo docs, mid-2026)

| Model | Languages | Task | Streaming | German quality | Runs native Windows? |
|---|---|---|---|---|---|
| `parakeet-tdt-0.6b-v2` | English only | ASR | offline | — | via CTranslate2/ONNX ports only |
| `parakeet-tdt-0.6b-v3` | 25 EU incl. German | ASR only (no English out) | **offline** (buffered ~2–4 s) | WER ~5% (offline) | no (NeMo) / sherpa-onnx buffered |
| `canary-1b-flash` | en/de/fr/es | ASR **+ De↔En translate** | offline AED (+new buffered AlignAtt/Wait-K) | De→En BLEU 35.5, WER ~4.4% | no → WSL2 |
| `canary-1b-v2` | 25 EU | ASR + X↔En translate | offline AED | strong (broadest coverage) | no → WSL2 |
| **`nemotron-3.5-asr-streaming-0.6b`** | 40 locales incl. de-DE | ASR only | **TRUE cache-aware** 80–1120 ms | streaming WER 8.3–9.8% | no → WSL2 |
| **`faster-whisper large-v3`** | multi incl. German | ASR **+ translate→En** | buffered | good, a notch below Canary | **YES (native Windows)** |
| `Opus-MT de-en` (CTranslate2) | De→En | text MT | n/a (~20 ms/sentence) | solid; NLLB-600M if more needed | **YES (native Windows)** |

---

## The three architectures

### ✅ Path C — faster-whisper, native Windows  **(CHOSEN)**

**What:** Whisper `large-v3` via CTranslate2, `task=translate` → English. One model, one Windows process.
**Pros:** No WSL2, no Triton, **no audio bridge** — audio capture and model in the same process. Simplest to build and debug. Native CUDA wheels.
**Cons:** English-out only (no reverse); De→En quality a notch below Canary; buffered latency ~1.5–3 s (tunable).
**Latency:** ~1.5–3 s glass-to-glass depending on chunk size.

**Setup:**
```powershell
py -3.11 -m venv d:\LiveTranslation\.venv
d:\LiveTranslation\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install faster-whisper
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12   # CUDA 12 / cuDNN 9 DLLs
```
**Use:**
```python
from faster_whisper import WhisperModel
m = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = m.transcribe("clip.wav", language="de", task="translate")  # → English
```
**Streaming (Step 4):** wrap with `ufal/whisper_streaming` (LocalAgreement-2) or `collabora/WhisperLive` (client/server + VAD).
**Quality upgrade:** if built-in translate underwhelms, switch to `task=transcribe` (German) + a CTranslate2 **Opus-MT de-en** stage (see glue below).

### Path A — Streaming ASR + MT, in WSL2  *(lowest latency; most engineering)*

**What:** `nemotron-3.5-asr-streaming-0.6b` (cache-aware German streaming) → CTranslate2 Opus-MT de→en.
**Pros:** Genuine low latency (ASR ~160–320 ms + MT ~20 ms); modular.
**Cons:** WSL2 + Windows→Linux audio bridge; two models glued with a commit policy; German streaming WER ~8–9%.
**Latency:** ~0.5–1 s achievable.

**Setup (WSL2 Ubuntu 24.04):**
```bash
conda create -n asr python=3.12 && conda activate asr
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126   # torch FIRST
pip install "nemo_toolkit[asr]==2.7.3"                                            # then NeMo, on top
python -c "import torch; print(torch.cuda.is_available())"   # must be True
# ASR: run examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py
#      with --att_context_size=[56,3]  (320 ms sweet spot; [56,1]=160 ms, [56,6]=560 ms)
```
**Audio bridge:** capture system audio on Windows (VB-Cable + sounddevice), stream PCM to WSL2 over a localhost WebSocket; WSL2 forwards localhost automatically.

### Path B — Canary direct, in WSL2  *(best translation quality; simplest NeMo)*

**What:** `canary-1b-flash` does German→English in ONE model.
**Pros:** Best De→En quality (BLEU ~35); single model.
**Cons:** Offline/buffered → ~2–4 s latency; WSL2 + audio bridge.
**Setup:** same WSL2 env as Path A, then:
```python
from nemo.collections.asr.models import EncDecMultiTaskModel
m = EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b-flash")
# transcribe with source_lang="de", target_lang="en"; chunk long audio (<40 s per call)
```
Newer NeMo adds AlignAtt/Wait-K buffered streaming decode if you want lower latency than naive chunking.

---

## Reusable streaming "glue" (applies to any path with a streaming ASR)

1. **Commit policy — LocalAgreement-2:** run ASR on a growing buffer; the longest common prefix over 2 consecutive updates is "committed." Only committed text goes downstream → no re-translating flickering words. (Whisper/transducer models both work with this.)
2. **MT trigger — sentence-bounded re-translation:** German is **verb-final** (SOV→SVO) — you can't correctly finalize an English clause until the German verb arrives. So: re-translate the *unfinished* sentence each update for the volatile display, but **permanently commit** the English only at German sentence-final punctuation (`. ? !`). Optionally flush at commas for very long sentences.
3. **Display — two tier:** committed English rendered solid/stable (never rewritten) + a greyed/italic volatile tail that updates freely. Confines the reader's re-reading to the tail. Reference impls: `ufal/whisper_streaming`, `QuentinFuxa/WhisperLiveKit`.
4. **No separate punctuation model needed** — Whisper and Parakeet/nemotron emit punctuation + capitalization natively.

---

## Build plan (Path C)

1. **Env** — native venv + faster-whisper + CUDA12/cuDNN9; verify a model loads on CUDA. ✅ **DONE (2026-07-14)** — `.venv` (Py 3.11.6), faster-whisper 1.2.1 / ctranslate2 4.8.1 / cuBLAS 12.9 / cuDNN 9.24. GPU confirmed on `float16` + `int8_float16`. Every GPU script must call `nvidia_dll_shim.enable()` first (see gotchas).
2. **German quality check** — run `large-v3` on a real German clip: `transcribe` (German text) and `translate` (English). Judge quality + timing before building further.
3. **System audio capture** — VB-Cable install + Windows output routing; read the virtual device with `sounddevice`.
4. **Live loop** — ring buffer → streaming faster-whisper → captions to stdout. Chunk/buffer size a tunable constant at the top (latency ↔ accuracy).
5. **Overlay** — always-on-top caption window.

---

## Tuning knobs / open questions

- **Model/precision (Step 2):** `large-v3` (best) vs `large-v3-turbo` (much faster, small quality hit) vs `distil-large-v3`; `float16` vs `int8_float16` (less VRAM, tiny quality cost).
- **Windows CUDA-DLL gotcha (SOLVED):** the pip `nvidia-*-cu12` wheels put DLLs in `site-packages\nvidia\*\bin`, off the search path — AND ctranslate2 loads `cublas64_12.dll` without first resolving its `cublasLt64_12.dll` dep, so `os.add_dll_directory` alone fails with *"cublas64_12.dll is not found or cannot be loaded"* (or **hangs** on the int8 path). Fix = **`nvidia_dll_shim.py`**: adds all nvidia bin dirs to PATH + DLL search, then pre-loads the CUDA libs in dependency order. Call `enable()` before any `WhisperModel(device="cuda")`.
- **HF download gotcha:** the bundled `hf-xet` transfer layer hung on Windows; set `HF_HUB_DISABLE_XET=1` (or `pip uninstall hf-xet`) for plain-HTTP downloads. Use `HF_HUB_OFFLINE=1` once a model is cached.
- **Chunk size (Step 4):** trades latency vs accuracy — expose as a top-level constant.
- **Translation quality:** if Whisper `translate` disappoints, add CTranslate2 Opus-MT de-en (or NLLB-600M).

---

## Key sources
- Parakeet/Canary/nemotron HF model cards (`huggingface.co/nvidia/...`)
- NeMo Windows support matrix — `pypi.org/project/nemo-toolkit/` (2.7.3, "No support yet" on Windows)
- `nemotron-3.5-asr-streaming-0.6b` card + NVIDIA fine-tuning blog
- `ufal/whisper_streaming` (LocalAgreement) · `QuentinFuxa/WhisperLiveKit` (two-tier UI) · `OpenNMT/CTranslate2`
