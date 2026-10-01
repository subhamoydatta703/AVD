"""Explicit ISO/FLORES mappings; model support and tested support differ."""

ISO_TO_FLORES = {
    "as": "asm_Beng", "bn": "ben_Beng", "brx": "brx_Deva",
    "doi": "doi_Deva", "gu": "guj_Gujr", "hi": "hin_Deva",
    "kn": "kan_Knda", "ks": "kas_Arab", "kok": "gom_Deva",
    "mai": "mai_Deva", "ml": "mal_Mlym", "mni": "mni_Beng",
    "mr": "mar_Deva", "ne": "npi_Deva", "or": "ory_Orya",
    "pa": "pan_Guru", "sa": "san_Deva", "sat": "sat_Olck",
    "sd": "snd_Arab", "ta": "tam_Taml", "te": "tel_Telu",
    "ur": "urd_Arab", "en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn",
}
# Script-specific overrides must be explicit rather than guessed from ISO codes.
SCRIPT_VARIANTS = {"kas_Deva", "mni_Mtei", "snd_Deva"}

SCRIPT_RANGES = {
    "Deva": [(0x0900, 0x097F)], "Beng": [(0x0980, 0x09FF)],
    "Guru": [(0x0A00, 0x0A7F)], "Gujr": [(0x0A80, 0x0AFF)],
    "Orya": [(0x0B00, 0x0B7F)], "Taml": [(0x0B80, 0x0BFF)],
    "Telu": [(0x0C00, 0x0C7F)], "Knda": [(0x0C80, 0x0CFF)],
    "Mlym": [(0x0D00, 0x0D7F)], "Arab": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)],
    "Olck": [(0x1C50, 0x1C7F)], "Mtei": [(0xABC0, 0xABFF)],
}


def validate_script(text: str, language: str) -> None:
    """Reject obvious script hallucinations; this does not establish transcript accuracy."""
    import unicodedata
    from .errors import DubbingError

    script = flores_tag(language).split("_")[1]
    if script == "Latn":
        return
    ranges = SCRIPT_RANGES[script]
    letters = [c for c in text if c.isalpha() and "LATIN" not in unicodedata.name(c, "")]
    expected = sum(any(low <= ord(c) <= high for low, high in ranges) for c in letters)
    if letters and expected / len(letters) < 0.8:
        raise DubbingError(f"Transcript script does not match {language}. Review the language "
                           "override or use a more accurate ASR model before translating.")


def flores_tag(language: str) -> str:
    from .errors import DubbingError

    if language in ISO_TO_FLORES.values() or language in SCRIPT_VARIANTS:
        return language
    try:
        return ISO_TO_FLORES[language.lower()]
    except KeyError as exc:
        raise DubbingError(
            f"Unsupported source language: {language}. Specify an Indian ISO code "
            "(German/French also supported with Gemini) or a supported FLORES tag with --language."
        ) from exc


def iso_code(language: str) -> str:
    tag = flores_tag(language)
    variants = {"kas_Deva": "ks", "mni_Mtei": "mni", "snd_Deva": "sd"}
    return variants.get(tag) or next(k for k, v in ISO_TO_FLORES.items() if v == tag)
