# Live German → English Captions

A **fully-local, real-time German-speech captioning + translation** overlay for Windows.
It listens to whatever is playing on your PC, transcribes the German, translates it to
English, and floats the captions on top of your screen — **no cloud APIs, everything runs
on your own GPU.**

- **Capture:** WASAPI loopback of your default speaker — captures what you already hear, **nothing to reroute**.
- **ASR:** [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) `large-v3` (CTranslate2), streaming with LocalAgreement-2 so captions appear as whole sentences.
- **Translation:** local [Opus-MT](https://huggingface.co/Helsinki-NLP/opus-mt-de-en) `de→en` via CTranslate2 (~20–80 ms/sentence, faithful — no Whisper "translate" hallucinations).
- **Overlay:** always-on-top caption bar — **white = final**, **blue = live partial** that refines as the sentence completes (Chrome-style).

---

## Requirements

- **Windows 10/11.**
- **NVIDIA GPU**, ~4 GB+ free VRAM (developed on an RTX 4070 Ti, 12 GB). A recent NVIDIA driver — no separate CUDA toolkit needed (the pip wheels ship the CUDA runtime).
- **Python 3.11** (get it from [python.org](https://www.python.org/downloads/); the `tkinter` overlay ships with it).
- ~4 GB disk for models (Whisper `large-v3` ≈ 3 GB, downloaded on first run; Opus-MT ≈ 80 MB, built locally).

---

## Setup

From the project folder, in **PowerShell**:

```powershell
# 1. Create the Python 3.11 virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install runtime dependencies
pip install -r requirements.txt

# 3. One-time only: CPU PyTorch — needed *just* to convert the translation model
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. One-time only: convert Opus-MT de->en into CTranslate2 format
ct2-transformers-converter --model Helsinki-NLP/opus-mt-de-en `
  --output_dir models/opus-mt-de-en-ct2 --quantization int8_float16
```

The **Whisper `large-v3`** model downloads automatically from Hugging Face the first time you run the app (~3 GB — give it a minute).

> After step 4 you can uninstall `torch` if you like (`pip uninstall torch`) — it's only used for the one-time conversion; the app runs the model through CTranslate2.

---

## Run

Just **double-click a launcher** (they open a small console — that's your off-switch — then start):

| Launcher | What you get |
|---|---|
| **`Overlay.bat`** | Floating always-on-top caption bar. **Drag** to move, **Esc** to close. |
| **`LiveTranslate.bat`** | Same captions in the terminal (DE + EN lines). |

Then just play any German audio/video. Captions appear ~1.5–3 s after each sentence.

There is **no audio setup** — it captures your current default output device via loopback, so you keep hearing everything normally.

---

## Configuration

Edit the constants at the top of **`overlay.py`** (or `live_captions_translate.py`):

| Setting | Default | Meaning |
|---|---|---|
| `FONT_SIZE` | `22` | Caption text size |
| `OPACITY` | `0.88` | Overlay transparency |
| `MAX_LINES` | `2` | How many finalized lines stay on screen |
| `SHOW_GERMAN` | `False` | Also show the German source line |
| `STEP_SEC` | `1.5` | How often it re-transcribes — lower = snappier live partials, more GPU |
| `SILENCE_PEAK` | `0.005` | Below this audio level it stays quiet (kills silence-hallucination). Lower if quiet speech is missed; raise if silence still captions. |
| `AUDIO_SOURCE` | `"loopback"` | `"loopback"` (default, no setup) or `"cable"` (VB-CABLE, see below) |

---

## How it works

```
system audio ─(WASAPI loopback)─▶ 16 kHz mono
   │
   ▼
faster-whisper large-v3  ──▶  growing-buffer streaming + LocalAgreement-2
   │                           (commit a line only when a German sentence is
   │                            complete & stable → correct verb-final translation)
   ▼
Opus-MT de→en (CTranslate2)  ──▶  English
   │
   ▼
tkinter overlay  (white = committed, blue = live partial)
```

- `audio_source.py` — `LoopbackSource` (soundcard, threaded continuous capture) / `CableSource` (sounddevice + VB-CABLE).
- `streaming.py` — `StreamingTranscriber`: buffering, word-level LocalAgreement, sentence flush, silence/no-speech gating.
- `mt.py` — `Translator`: CTranslate2 + Marian tokenizer.
- `nvidia_dll_shim.py` — makes the pip CUDA wheels loadable on Windows (see Troubleshooting).
- `overlay.py` / `live_captions_translate.py` / `live_captions.py` — the apps.
- Helpers: `list_devices.py`, `loopback_list.py`, `loopback_capture_test.py`, `capture_and_transcribe.py`, `transcribe_test.py`, `_gpu_smoke.py`.
- `PLAN.md` — design notes and the alternative architectures (streaming Parakeet + MT, Canary direct) if you want to swap components.

---

## Troubleshooting

- **`cublas64_12.dll ... cannot be loaded` (or a hang):** the pip CUDA wheels put their DLLs off the search path, and ctranslate2 loads `cublas` before its `cublasLt` dependency. `nvidia_dll_shim.enable()` (called by every app) fixes it — make sure it runs before any CUDA model loads.
- **Hugging Face download hangs on Windows:** the bundled `hf-xet` transfer layer can stall. The apps set `HF_HUB_DISABLE_XET=1` / `HF_HUB_OFFLINE=1` automatically once models are cached.
- **It captions "Thank you" / "Thanks for watching" during silence:** classic Whisper hallucination on quiet audio — handled by `SILENCE_PEAK` + a `no_speech_prob` filter. If it still happens, raise `SILENCE_PEAK`.
- **Captions are sparse / miss quiet speech:** bump the source app's volume, or lower `SILENCE_PEAK`.
- **Only `# …listening (silent)` while audio is clearly playing:** your audio is on a different output than the OS default (common with **SteelSeries Sonar** / virtual mixers — apps may land on a *Media*/*Chat* channel while the default is *Gaming*). Set that app's output to your default device, or target the specific device in `audio_source.py`.
- **VB-CABLE (optional fallback):** to capture one app in isolation instead of everything you hear, install [VB-CABLE](https://vb-audio.com/Cable/), route that app's output into "CABLE Input", set `AUDIO_SOURCE = "cable"`.

---

## Credits & licenses

- **ASR:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (SYSTRAN) + OpenAI Whisper `large-v3`.
- **Translation:** [Helsinki-NLP Opus-MT `de→en`](https://huggingface.co/Helsinki-NLP/opus-mt-de-en) (CC-BY-4.0).
- **Inference engine:** [CTranslate2](https://github.com/OpenNMT/CTranslate2) (OpenNMT).
- **Audio capture:** [soundcard](https://github.com/bastibe/SoundCard).
- **Streaming approach** inspired by [ufal/whisper_streaming](https://github.com/ufal/whisper_streaming) (LocalAgreement).

Everything runs locally; no audio or text leaves your machine.
