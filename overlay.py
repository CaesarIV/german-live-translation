"""Always-on-top caption overlay: white committed lines + blue live partial.

Double-click Overlay.bat. Drag to move. Esc to close.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import threading
import queue
import tkinter as tk

# ==================== CONFIG ====================
FONT_SIZE = 22
OPACITY = 0.88
MAX_LINES = 2            # committed lines kept on screen
SHOW_GERMAN = False      # True -> also show the German (small, grey)
STEP_SEC = 1.5
ASR_MODEL = "large-v3"
AUDIO_SOURCE = "loopback"
SILENCE_PEAK = 0.005     # skip captions when buffer peak is below this (kills silence-hallucination).
                         # lower if quiet speech gets missed; raise if silence still captions.
# ===============================================

msgq = queue.Queue()
stop_evt = threading.Event()


def worker():
    try:
        from nvidia_dll_shim import enable
        enable()
        from faster_whisper import WhisperModel
        from mt import Translator
        from audio_source import make_source
        from streaming import StreamingTranscriber

        msgq.put(("status", "loading models…"))
        asr = WhisperModel(ASR_MODEL, device="cuda", compute_type="float16")
        mt = Translator(device="cuda", compute_type="int8_float16")

        cache = {"de": None, "en": ""}

        def on_sentence(de):
            msgq.put(("commit", de, mt.translate(de)))

        def on_partial(de):
            if not de:
                msgq.put(("partial", "", ""))
                return
            if de != cache["de"]:
                cache["de"], cache["en"] = de, mt.translate(de)
            msgq.put(("partial", de, cache["en"]))

        stream = StreamingTranscriber(asr, sr=16000, step_sec=STEP_SEC,
                                      on_sentence=on_sentence, on_partial=on_partial,
                                      silence_peak=SILENCE_PEAK)
        src = make_source(AUDIO_SOURCE)
        msgq.put(("status", ""))
        with src:
            while not stop_evt.is_set():
                stream.add_audio(src.read(0.5))
    except Exception as e:
        msgq.put(("error", repr(e)))


# ---------------- GUI ----------------
BG = "#0a0a0a"
root = tk.Tk()
root.overrideredirect(True)
root.attributes("-topmost", True)
try:
    root.attributes("-alpha", OPACITY)
except Exception:
    pass
root.configure(bg=BG)
sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
W, H = int(sw * 0.82), 200
root.geometry(f"{W}x{H}+{(sw - W) // 2}+{sh - H - 70}")

txt = tk.Text(root, bg=BG, bd=0, highlightthickness=0, wrap="word",
              font=("Segoe UI", FONT_SIZE), padx=24, pady=16, cursor="fleur")
txt.pack(fill="both", expand=True)
txt.tag_config("commit", foreground="#ffffff")
txt.tag_config("partial", foreground="#8ab4f8")
txt.tag_config("de", foreground="#666666", font=("Segoe UI", int(FONT_SIZE * 0.6)))
txt.tag_config("status", foreground="#888888", font=("Segoe UI", int(FONT_SIZE * 0.7)))
txt.configure(state="disabled")

committed = []          # list of (de, en)
partial = ("", "")
status = "starting…"


def render():
    txt.configure(state="normal")
    txt.delete("1.0", "end")
    if status:
        txt.insert("end", status + "\n", "status")
    for de, en in committed[-MAX_LINES:]:
        if SHOW_GERMAN and de:
            txt.insert("end", de + "\n", "de")
        txt.insert("end", en + "\n", "commit")
    pde, pen = partial
    if pen:
        if SHOW_GERMAN and pde:
            txt.insert("end", pde + "\n", "de")
        txt.insert("end", pen, "partial")
    txt.see("end")
    txt.configure(state="disabled")


def poll():
    global status, partial
    changed = False
    try:
        while True:
            m = msgq.get_nowait()
            kind = m[0]
            if kind == "status":
                status = m[1]
            elif kind == "commit":
                committed.append((m[1], m[2]))
                partial = ("", "")
            elif kind == "partial":
                partial = (m[1], m[2])
            elif kind == "error":
                status = "ERROR: " + m[1]
            changed = True
    except queue.Empty:
        pass
    if changed:
        render()
    root.after(80, poll)


def start_drag(e):
    root._dx, root._dy = e.x, e.y


def do_drag(e):
    root.geometry(f"+{root.winfo_pointerx() - root._dx}+{root.winfo_pointery() - root._dy}")


txt.bind("<Button-1>", start_drag)
txt.bind("<B1-Motion>", do_drag)
root.bind("<Escape>", lambda e: (stop_evt.set(), root.destroy()))

threading.Thread(target=worker, daemon=True).start()
render()
root.after(80, poll)
root.mainloop()
stop_evt.set()
