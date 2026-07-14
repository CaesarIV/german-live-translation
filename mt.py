"""German -> English translation via CTranslate2 + Opus-MT (Marian).

Faithful NMT (no hallucination like Whisper's translate task), ~10-40 ms/sentence
on GPU. Requires the converted model at models/opus-mt-de-en-ct2 (see convert step).
Call nvidia_dll_shim.enable() before constructing on device="cuda".
"""
import os
import ctranslate2
from transformers import AutoTokenizer

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "opus-mt-de-en-ct2")
_TOK = "Helsinki-NLP/opus-mt-de-en"


class Translator:
    def __init__(self, device="cuda", compute_type="int8_float16", model_dir=_DIR, tok=_TOK):
        self.tokenizer = AutoTokenizer.from_pretrained(tok)
        self.model = ctranslate2.Translator(model_dir, device=device, compute_type=compute_type)

    def translate(self, text):
        text = text.strip()
        if not text:
            return ""
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        result = self.model.translate_batch([tokens], beam_size=2, max_decoding_length=256)
        out = result[0].hypotheses[0]
        return self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(out), skip_special_tokens=True)


if __name__ == "__main__":
    from nvidia_dll_shim import enable
    enable()
    t = Translator()
    for s in ["Guten Morgen, wie geht es dir?",
              "Der Fall der Berliner Mauer war ein Symbol für die Freiheit.",
              "Er entwickelte das Benzinadditiv Tetraethylblei sowie die Fluorchlor-Kohlenwasserstoffe."]:
        print("DE:", s)
        print("EN:", t.translate(s), "\n")
