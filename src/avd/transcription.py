"""Bounded-memory Whisper transcription with per-chunk checkpoints."""

from pathlib import Path
import wave

from .data import Segment, load_segments, read_json, save_segments, write_json
from .errors import DubbingError
from .languages import flores_tag, iso_code, validate_script
from .recognition import Recognizer


def transcribe(audio: Path, directory: Path, model_name: str = "large-v3",
               language: str | None = None, device: str = "cpu",
               chunk_seconds: int = 300, backend: str = "faster-whisper",
               threads: int = 8, model_root: Path | None = None) -> tuple[list[Segment], str]:
    from whisper.tokenizer import LANGUAGES

    requested = iso_code(language) if language else None
    if requested and requested not in LANGUAGES:
        raise DubbingError(f"Whisper does not support {requested}; an Indic ASR backend is needed.")
    directory.mkdir(parents=True, exist_ok=True)
    recognizer = Recognizer(model_name, backend, device, threads, model_root or Path("output/models"))
    detected = language
    results: list[Segment] = []
    try:
        with wave.open(str(audio), "rb") as source:
            rate = source.getframerate()
            total_frames = source.getnframes()
            chunk_frames = rate * chunk_seconds
            count = (total_frames + chunk_frames - 1) // chunk_frames
            for index in range(count):
                start_frame = index * chunk_frames
                offset = start_frame / rate
                chunk_duration = min(chunk_frames, total_frames - start_frame) / rate
                checkpoint = directory / f"chunk_{index:05d}.json"
                print(f"[transcribe] Chunk {index + 1}/{count} ({offset:.0f}s)", flush=True)
                if checkpoint.exists():
                    payload = read_json(checkpoint)
                    local = load_segments(checkpoint)
                    chunk_language = payload["language"]
                else:
                    raw_path = directory / f"raw_{index:05d}.json"
                    chunk = directory / "current.wav"
                    source.setpos(start_frame)
                    with wave.open(str(chunk), "wb") as output:
                        output.setparams(source.getparams())
                        output.writeframes(source.readframes(chunk_frames))
                    if raw_path.exists():
                        transcript = read_json(raw_path)
                    else:
                        transcript = recognizer.transcribe(chunk, requested)
                        # A raw file is diagnostic evidence, never an accepted checkpoint.
                        write_json(raw_path, transcript)
                    chunk_language = transcript["language"]
                    local = []
                    for item in transcript["segments"]:
                        text = item["text"].strip()
                        words = item.get("words") or []
                        start = max(0.0, words[0]["start"] if words else item["start"])
                        end = min(chunk_duration, words[-1]["end"] if words else item["end"])
                        if text and end > start:
                            local.append(Segment(len(local), start, end, text))
                    for segment in local:
                        try:
                            validate_script(segment.text, language or chunk_language)
                        except DubbingError as exc:
                            raise DubbingError(
                                f"{exc} Chunk {index + 1}, at {offset + segment.start:.2f}s: "
                                f"{segment.text!r}. Raw output was saved for review."
                            ) from exc
                    save_segments(checkpoint, local, language=chunk_language)
                    chunk.unlink(missing_ok=True)
                if local:
                    for segment in local:
                        validate_script(segment.text, language or chunk_language)
                    if detected is None:
                        detected = chunk_language
                        flores_tag(detected)
                        print(f"[transcribe] Detected source language: {detected}", flush=True)
                    elif not requested and iso_code(detected) != chunk_language:
                        raise DubbingError("Language changed between chunks. Set --language after "
                                           "reviewing the source; automatic code-switch routing is not implemented.")
                for segment in local:
                    results.append(Segment(len(results), offset + segment.start,
                                           offset + segment.end, segment.text))
    finally:
        recognizer.close()
    if not results or detected is None:
        raise DubbingError("No speech detected. Review the audio before continuing.")
    return results, flores_tag(detected)
