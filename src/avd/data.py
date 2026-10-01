"""Validated speech segments and atomic JSON artifacts."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from .errors import DubbingError


@dataclass(frozen=True)
class Segment:
    id: int
    start: float
    end: float
    text: str
    english: str = ""
    language: str = ""

    def __post_init__(self) -> None:
        if (type(self.id) is not int or self.id < 0 or not math.isfinite(self.start)
                or not math.isfinite(self.end) or self.start < 0
                or self.end <= self.start or not self.text.strip()):
            raise DubbingError(f"Invalid speech segment: {self.id}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_segments(path: Path, segments: list[Segment], **metadata: Any) -> None:
    write_json(path, {**metadata, "segments": [asdict(s) for s in segments]})


def load_segments(path: Path) -> list[Segment]:
    return [Segment(**s) for s in read_json(path)["segments"]]
