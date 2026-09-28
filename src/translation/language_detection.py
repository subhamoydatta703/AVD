import os
import fasttext


# 1. Initialize the Language Detector
# Resolve the model path relative to the project root
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "lid.176.ftz")
lang_model = fasttext.load_model(_MODEL_PATH)

# Map standard ISO 2-letter codes to IndicTrans2 FLORES-200 language tags
ISO_TO_FLORES = {
    "hi": "hin_Deva", "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu",
    "mr": "mar_Deva", "gu": "guj_Gujr", "kn": "kan_Knda", "ml": "mal_Mlym",
    "or": "ory_Orya", "pa": "pan_Guru", "as": "asm_Beng", "ne": "nep_Deva",
    "sa": "san_Deva", "ur": "urd_Arab", "sd": "snd_Arab", "ks": "kas_Arab",
    "mni": "mni_Beng", "brx": "brx_Deva", "doi": "doi_Deva", "sat": "sat_Olck",
    "kok": "kok_Deva", "mai": "mai_Deva"
}


def detect_language(text):
    """Detects language and returns the FLORES tag. Defaults to Hindi if unsure."""
    # fasttext outputs tags like '__label__hi'
    predictions = lang_model.predict(text, k=1)
    src_lang_iso = predictions[0][0].replace("__label__", "")

    # 2. If it's already English, don't translate it!
    if src_lang_iso == "en":
        print("Text is already in English, skipping translation.")
        return "eng_Latn"

    # 3. Otherwise, look up the Indic language and return the FLORES tag
    src_lang = ISO_TO_FLORES.get(src_lang_iso)
    if src_lang is None:
        raise ValueError(f"Unsupported language detected: {src_lang_iso}")
    return src_lang

