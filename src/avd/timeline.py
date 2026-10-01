"""Place dubbed PCM on absolute timestamps without holding the full track in RAM."""

from pathlib import Path
import wave

from .data import Segment, write_json
from .errors import DubbingError
from .media import fit_audio, pcm_duration

RATE = 24000


def inspect_timing(segments: list[Segment], clips: list[Path], duration: float,
                   max_speed: float = 1.35) -> list[dict]:
    """Measure every phrase before writing audio, so all conflicts can be repaired together."""
    import math
    if not segments or len(segments) != len(clips) or not math.isfinite(duration) or duration <= 0:
        raise DubbingError("Audio clip count/duration does not match the transcript.")
    if not 1 <= max_speed <= 2 or len({s.id for s in segments}) != len(segments):
        raise DubbingError("Invalid tempo limit or duplicate segment IDs.")
    rows = []
    for index, (segment, clip) in enumerate(zip(segments, clips)):
        next_start = segments[index + 1].start if index + 1 < len(segments) else duration
        if (segment.end > duration + 0.05 or next_start < segment.end - 0.05
                or next_start <= segment.start):
            raise DubbingError(f"Overlapping/out-of-range source segment {segment.id}; review transcript.")
        available = min(duration, next_start) - segment.start
        seconds = pcm_duration(clip, RATE)
        speed = max(1.0, seconds / available)
        rows.append({"id": segment.id, "start": segment.start, "source_end": segment.end,
                     "available_seconds": available, "tts_seconds": seconds,
                     "required_speed": speed, "max_speed": max_speed, "conflict": speed > max_speed})
    return rows


def silence(output: wave.Wave_write, frames: int) -> None:
    block = bytes(RATE * 2)
    while frames:
        count = min(frames, RATE)
        output.writeframesraw(block[:count * 2])
        frames -= count


def assemble(segments: list[Segment], clips: list[Path], duration: float,
             destination: Path, max_speed: float = 1.35) -> None:
    measured = inspect_timing(segments, clips, duration, max_speed)
    write_json(destination.parent / "alignment.json", measured)
    conflicts = [row for row in measured if row["conflict"]]
    if conflicts:
        ids = ", ".join(str(row["id"]) for row in conflicts)
        raise DubbingError(f"Segments {ids} exceed the tempo limit. Review/shorten English text "
                           "in translation.json and resume. Speech has not been truncated.")
    directory = destination.parent / "aligned"
    directory.mkdir(parents=True, exist_ok=True)
    report_path = destination.parent / "alignment.json"
    report = []
    temporary = destination.with_suffix(".part.wav")
    total_frames = round(duration * RATE)
    position = 0
    with wave.open(str(temporary), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(RATE)
        for index, (segment, clip) in enumerate(zip(segments, clips)):
            start = round(segment.start * RATE)
            # Preserve start times; available silence after a phrase can absorb longer English wording.
            next_start = segments[index + 1].start if index + 1 < len(segments) else duration
            if segment.end > duration + 0.05 or next_start < segment.end - 0.05:
                raise DubbingError(f"Overlapping/out-of-range source segment {segment.id}; review transcript.")
            available = min(duration, next_start) - segment.start
            seconds = pcm_duration(clip, RATE)
            speed = max(1.0, seconds / available)
            report.append({"id": segment.id, "start": segment.start,
                           "source_end": segment.end, "available_seconds": available,
                           "tts_seconds": seconds, "required_speed": speed})
            write_json(report_path, report)
            if speed > max_speed:
                raise DubbingError(
                    f"Segment {segment.id} needs {speed:.2f}x speed (limit {max_speed:.2f}x). "
                    "Review/shorten its English text in translation.json and resume. "
                    "Speech has not been truncated."
                )
            aligned = directory / f"{segment.id:06d}.wav"
            fit_audio(clip, aligned, min(max_speed, speed * 1.01) if speed > 1 else 1.0)
            frames = round(pcm_duration(aligned, RATE) * RATE)
            if start < position or start + frames > round(min(duration, next_start) * RATE):
                raise DubbingError(f"Segment {segment.id} still overlaps after tempo adjustment.")
            silence(output, start - position)
            with wave.open(str(aligned), "rb") as audio:
                while block := audio.readframes(RATE):
                    output.writeframesraw(block)
            position = start + frames
            report[-1]["actual_speed"] = min(max_speed, speed * 1.01) if speed > 1 else 1.0
            report[-1]["dub_end"] = position / RATE
            print(f"[align] Segment {index + 1}/{len(segments)}", flush=True)
        silence(output, total_frames - position)
    pcm_duration(temporary, RATE)
    temporary.replace(destination)
    write_json(report_path, report)
