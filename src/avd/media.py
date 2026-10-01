"""FFmpeg operations, using argument lists rather than shell interpolation."""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import wave

from .errors import DubbingError


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    video_codec: str
    width: int
    height: int


def require_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
    if missing:
        raise DubbingError(f"Install {', '.join(missing)} and add it to PATH.")


def run(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", check=True)
    except FileNotFoundError as exc:
        raise DubbingError(f"Executable missing: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise DubbingError(f"{command[0]} failed: {exc.stderr[-3000:]}") from exc
    return result.stdout


def probe(path: Path) -> MediaInfo:
    payload = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                              "-of", "json", str(path)]))
    videos = [s for s in payload["streams"] if s["codec_type"] == "video"
              and not s.get("disposition", {}).get("attached_pic")]
    audio = [s for s in payload["streams"] if s["codec_type"] == "audio"]
    if not videos or not audio:
        raise DubbingError("Input/output must contain both video and audio.")
    duration = float(payload["format"].get("duration", 0))
    if duration <= 0:
        raise DubbingError("Could not determine a positive media duration.")
    v = videos[0]
    return MediaInfo(duration, v["codec_name"], v["width"], v["height"])


def extract_audio(video: Path, destination: Path) -> None:
    temporary = destination.with_suffix(".part.wav")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
         "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
         str(temporary)])
    temporary.replace(destination)


def pcm_duration(path: Path, rate: int | None = None) -> float:
    try:
        with wave.open(str(path), "rb") as audio:
            if (audio.getnchannels() != 1 or audio.getsampwidth() != 2
                    or (rate is not None and audio.getframerate() != rate)):
                raise DubbingError(f"Unexpected PCM format: {path}")
            seconds = audio.getnframes() / audio.getframerate()
            if seconds <= 0:
                raise DubbingError(f"Empty audio: {path}")
            return seconds
    except (wave.Error, EOFError) as exc:
        raise DubbingError(f"Invalid WAV audio: {path}") from exc


def fit_audio(source: Path, destination: Path, speed: float, rate: int = 24000) -> None:
    # Speed changes affect tempo while preserving pitch.
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
         "-vn", "-af", f"atempo={speed:.8f}", "-ac", "1", "-ar", str(rate),
         "-c:a", "pcm_s16le", str(destination)])


def passthrough(video: Path, output: Path) -> None:
    """Keep existing English speech and visuals without encoding either stream."""
    if output.suffix.lower() not in {".mkv", ".mp4"}:
        raise DubbingError("Output must use .mkv or .mp4; MKV supports more source codecs.")
    temporary = output.with_name(output.stem + ".part" + output.suffix)
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
         "-map", "0:V:0", "-map", "0:a:0", "-map_metadata", "0", "-c", "copy",
         "-metadata:s:a:0", "language=eng", str(temporary)])
    original, copied = probe(video), probe(temporary)
    if ((original.video_codec, original.width, original.height)
            != (copied.video_codec, copied.width, copied.height)
            or abs(original.duration - copied.duration) > 0.3):
        raise DubbingError("English passthrough validation failed; intermediate output retained.")
    temporary.replace(output)


def mux(video: Path, audio: Path, output: Path) -> None:
    if output.suffix.lower() not in {".mkv", ".mp4"}:
        raise DubbingError("Output must use .mkv or .mp4; MKV supports more source codecs.")
    temporary = output.with_name(output.stem + ".part" + output.suffix)
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
         "-i", str(audio), "-map", "0:V:0", "-map", "1:a:0", "-map_metadata", "0",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-metadata:s:a:0", "language=eng", str(temporary)])
    original, dubbed = probe(video), probe(temporary)
    if ((original.video_codec, original.width, original.height)
            != (dubbed.video_codec, dubbed.width, dubbed.height)
            or abs(original.duration - dubbed.duration) > 0.3):
        raise DubbingError("Final video validation failed; intermediate output retained.")
    temporary.replace(output)
