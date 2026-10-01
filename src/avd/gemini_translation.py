"""Translate text with the official Google Gen AI SDK."""

import os

from google import genai
from google.genai import errors, types

from .config import DEFAULT_GEMINI_MODEL, gemini_api_key, load_environment
from .errors import DubbingError


def translate_text(text: str, source_language: str = "auto",
                   target_language: str = "English", model: str | None = None) -> str:
    """Return the text translated into the requested language."""
    if not text.strip():
        raise DubbingError("Translation requires nonempty text.")
    load_environment()
    prompt = (
        f"Translate the following text from {source_language} into natural, accurate {target_language}.\n"
        "Preserve its meaning, names, numbers and tone. Return only the translation, without commentary.\n\n"
        f"Text:\n{text}"
    )
    try:
        with genai.Client(api_key=gemini_api_key()) as client:
            response = client.models.generate_content(
                model=model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3),
            )
    except errors.APIError as exc:
        raise DubbingError(f"Gemini translation failed (HTTP {exc.code}).") from None
    if not response.text or not response.text.strip():
        raise DubbingError("Gemini returned an empty translation.")
    return response.text.strip()
