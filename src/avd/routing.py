"""Select a supported source language; uncertain input never becomes English passthrough."""

from pathlib import Path
import wave

from .data import write_json
from .errors import DubbingError
from .languages import flores_tag
from .recognition import Recognizer


def decide_language(samples: list[dict], minimum_confidence: float = 0.8) -> str:
    usable = [row for row in samples if row["speech_present"]]
    if not usable or any(row["probability"] < minimum_confidence for row in usable):
        raise DubbingError("Language detection is uncertain. Review the audio and specify --language.")
    languages = {row["language"] for row in usable}
    if len(languages) != 1:
        raise DubbingError("Sampled passages contain different languages. Specify --language after "
                           "reviewing the source; mixed-language dubbing needs passage-level review.")
    return flores_tag(next(iter(languages)))


def detect_source_language(audio: Path, directory: Path, model_name: str,
                           backend: str, device: str, threads: int,
                           model_root: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    recognizer = Recognizer(model_name, backend, device, threads, model_root)
    samples = []
    try:
        with wave.open(str(audio), "rb") as source:
            rate, frames = source.getframerate(), source.getnframes()
            length = min(frames, rate * 30)
            offsets = sorted({0, max(0, (frames - length) // 2), max(0, frames - length)})
            for index, offset in enumerate(offsets):
                clip = directory / f"sample_{index}.wav"
                source.setpos(offset)
                with wave.open(str(clip), "wb") as output:
                    output.setparams(source.getparams())
                    output.writeframes(source.readframes(length))
                language, probability, speech = recognizer.detect(clip)
                samples.append({"offset_seconds": offset / rate, "language": language,
                                "probability": probability, "speech_present": speech})
                write_json(directory / "samples.json", samples)
                print(f"[detect] {offset / rate:.1f}s: {language} ({probability:.1%}), "
                      f"speech={speech}", flush=True)
        result = decide_language(samples)
        write_json(directory / "result.json", {"language": result, "samples": samples})
        return result
    finally:
        recognizer.close()
