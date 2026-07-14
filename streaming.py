"""Streaming ASR with LocalAgreement-2 commit + sentence-level flush + live partials.

Feed audio via add_audio(chunk16k). Each step it re-transcribes the growing buffer and:
  - on_partial(german)  -> the current volatile hypothesis (updates continuously)
  - on_sentence(german) -> a COMPLETE, stable sentence (confirmed across two passes)
So a UI can show immediate partial text and finalize it when confirmed.
"""
import re
import numpy as np

_END = re.compile(r'[.!?…]["\'»)\]]?$')
_ORDINAL = re.compile(r'^\d+[.]$')   # German ordinals/dates "18." "9." are NOT sentence ends


def _is_sentence_end(word):
    w = word.strip()
    if not _END.search(w):
        return False
    if _ORDINAL.match(w):
        return False
    return True


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x[0].strip().lower() == y[0].strip().lower():
            n += 1
        else:
            break
    return n


class StreamingTranscriber:
    def __init__(self, asr, sr=16000, step_sec=1.5, max_buf_sec=18.0,
                 on_sentence=None, on_partial=None,
                 silence_peak=0.005, no_speech_max=0.6):
        self.asr = asr
        self.sr = sr
        self.step = int(step_sec * sr)
        self.max_buf = int(max_buf_sec * sr)
        self.on_sentence = on_sentence
        self.on_partial = on_partial
        self.silence_peak = silence_peak      # skip recognition below this buffer peak (kills silence-hallucination)
        self.no_speech_max = no_speech_max    # drop segments Whisper flags as probably-not-speech
        self.abuf = np.zeros(0, np.float32)
        self.prev = []
        self._since = 0

    def add_audio(self, chunk):
        self.abuf = np.concatenate([self.abuf, chunk])
        self._since += len(chunk)
        if self._since >= self.step:
            self._since = 0
            self._process()

    def _transcribe(self):
        segs, _ = self.asr.transcribe(self.abuf, language="de", task="transcribe",
                                      beam_size=1, word_timestamps=True, vad_filter=False)
        words = []
        for s in segs:
            if getattr(s, "no_speech_prob", 0.0) > self.no_speech_max:
                continue                                  # Whisper itself says this isn't speech
            if s.words:
                words.extend((w.word, w.start, w.end) for w in s.words)
            elif s.text.strip():
                words.append((s.text, s.start, s.end))
        return words

    def _emit_partial(self, text):
        if self.on_partial:
            self.on_partial(text)

    def _flush(self, words, upto):
        text = "".join(w[0] for w in words[:upto]).strip()
        end_t = words[upto - 1][2]
        if text and self.on_sentence:
            self.on_sentence(text)
        cut = min(len(self.abuf), int(end_t * self.sr))
        self.abuf = self.abuf[cut:]
        self.prev = []

    def _process(self):
        if len(self.abuf) < int(0.8 * self.sr):
            return
        if float(np.abs(self.abuf).max()) < self.silence_peak:   # near-silence -> skip (kills "Thank you" hallucination)
            if len(self.abuf) > int(4 * self.sr):
                self.abuf = self.abuf[-int(2 * self.sr):]
            self.prev = []
            self._emit_partial("")
            return
        words = self._transcribe()
        if not words:
            if len(self.abuf) > int(4 * self.sr):
                self.abuf = self.abuf[-int(2 * self.sr):]
            self.prev = []
            self._emit_partial("")
            return
        stable = _common_prefix(words, self.prev)
        last_end = -1
        for i in range(stable):
            if _is_sentence_end(words[i][0]):
                last_end = i
        if last_end >= 0:
            self._flush(words, last_end + 1)
            self._emit_partial("".join(w[0] for w in words[last_end + 1:]).strip())
        else:
            self.prev = words
            self._emit_partial("".join(w[0] for w in words).strip())
            if len(self.abuf) > self.max_buf and stable > 0:
                self._flush(words, stable)
                self._emit_partial("")
