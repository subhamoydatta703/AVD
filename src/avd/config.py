"""Load local credentials without overriding the process environment."""

import os
from pathlib import Path


DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


def load_environment(path: Path | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv(path or Path.cwd() / ".env", override=False)


def gemini_api_key() -> str:
    from .errors import DubbingError

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise DubbingError("Set GEMINI_API_KEY in your .env file or process environment before dubbing.")
    return key
