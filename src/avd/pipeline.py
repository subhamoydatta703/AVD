"""Resumable orchestration with explicit stage artifacts and timing records."""

import asyncio
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
from pathlib import Path
import time

from .data import load_segments, read_json, save_segments, write_json
from .download import download_video
from .errors import DubbingError
from .languages import flores_tag
from .media import extract_audio, mux, passthrough, pcm_duration, probe, require_tools
from .routing import detect_source_language
from .synthesis import synthesize
from .timeline import assemble
from .transcription import transcribe
from .config import DEFAULT_GEMINI_MODEL, gemini_api_key
from .gemini_translation import translate_text


@dataclass(frozen=True)
class Settings:
    language: str | None = None
    whisper_model: str = "large-v3"
    asr_backend: str = "faster-whisper"
    asr_threads: int = 8
    device: str = "cpu"
    gemini_model: str = DEFAULT_GEMINI_MODEL
    voice: str = "en-IN-PrabhatNeural"
    rate: str = "+0%"
    height: int = 720
    chunk_seconds: int = 300
    max_speed: float = 1.35


@contextmanager
def job_lock(directory: Path):
    lock = directory / ".running"
    try:
        handle = lock.open("x")
    except FileExistsError as exc:
        raise DubbingError(f"Job is locked: {lock}. If its process stopped, remove this lock and resume.") from exc
    try:
        import os
        with handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def run_pipeline(source: str, root: Path, settings: Settings,
                 output: Path | None = None) -> Path:
    require_tools()
    local = Path(source).expanduser()
    is_local = local.is_file()
    identity = {"source": str(local.resolve()) if is_local else source,
                "settings": asdict(settings), "pipeline_schema": 7}
    if is_local:
        identity["source_size"] = local.stat().st_size
        identity["source_mtime_ns"] = local.stat().st_mtime_ns
    job_id = hashlib.sha256(str(identity).encode()).hexdigest()[:16]
    directory = root / "jobs" / job_id
    directory.mkdir(parents=True, exist_ok=True)
    output = (output or directory / "english.mkv").resolve()
    with job_lock(directory):
        return _run(source, local, is_local, directory, output, settings, identity)


def _run(source: str, local: Path, is_local: bool, directory: Path, output: Path,
         settings: Settings, identity: dict) -> Path:
    report_path = directory / "report.json"
    report = read_json(report_path) if report_path.exists() else {
        "identity": identity, "attempts": [], "stages": {}, "status": "running",
        "voice_mode": "stock English voice; original speaker is not cloned",
    }
    started = time.perf_counter()
    attempt = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stages": {}}
    report["attempts"].append(attempt)
    report["status"] = "running"

    def stage(name, function):
        print(f"[{name}] Starting (job: {directory})", flush=True)
        begin = time.perf_counter()
        try:
            result = function()
        finally:
            seconds = time.perf_counter() - begin
            attempt["stages"][name] = seconds
            report["stages"][name] = report["stages"].get(name, 0) + seconds
            write_json(report_path, report)
        print(f"[{name}] Finished in {seconds:.1f}s", flush=True)
        return result

    try:
        if settings.language:
            flores_tag(settings.language)
        if settings.language and flores_tag(settings.language) != "eng_Latn":
            gemini_api_key()
        source_record = directory / "source.json"
        if is_local:
            video = local.resolve()
        elif source_record.exists():
            video = Path(read_json(source_record)["path"])
            if not video.is_file():
                raise DubbingError("Cached source is missing; restore it or use another work directory.")
        else:
            video = stage("download", lambda: download_video(source, directory / "source", settings.height))
        info = probe(video)
        write_json(source_record, {"path": str(video), "duration": info.duration})
        if output == video:
            raise DubbingError("Output must not overwrite the source video.")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and not (report.get("output") == str(output)):
            raise DubbingError(f"Output already exists: {output}. Choose a new filename.")
        language = flores_tag(settings.language) if settings.language else None

        def return_english():
            print("[route] English input: preserving original English audio and video.", flush=True)
            stage("passthrough", lambda: passthrough(video, output))
            report.pop("error", None)
            report.update(status="complete", output=str(output), language="eng_Latn",
                          mode="english_passthrough", voice_mode="original English voice preserved",
                          duration=info.duration, video=asdict(info))
            print(f"[complete] Saved {output}", flush=True)
            return output

        if language == "eng_Latn":
            return return_english()
        audio = directory / "source.wav"
        if not audio.exists():
            stage("extract", lambda: extract_audio(video, audio))
        pcm_duration(audio, 16000)
        if language is None:
            detection = directory / "detection" / "result.json"
            language = read_json(detection)["language"] if detection.exists() else stage(
                "detect", lambda: detect_source_language(
                    audio, directory / "detection", settings.whisper_model, settings.asr_backend,
                    settings.device, settings.asr_threads, directory.parents[1] / "models"))
            if language == "eng_Latn":
                return return_english()
        gemini_api_key()
        print(f"[route] {language}: source transcription -> Gemini English -> English speech", flush=True)
        transcript_path = directory / "transcript.json"
        if transcript_path.exists():
            segments = load_segments(transcript_path)
            language = read_json(transcript_path)["language"]
        else:
            segments, language = stage("transcribe", lambda: transcribe(
                audio, directory / "transcription", settings.whisper_model, language,
                settings.device, settings.chunk_seconds, settings.asr_backend,
                settings.asr_threads, directory.parents[1] / "models"))
            save_segments(transcript_path, segments, language=language)
        translated_path = directory / "translation.json"

        def translate():
            completed = load_segments(translated_path) if translated_path.exists() else []
            if len(completed) > len(segments) or any(
                    (a.id, a.start, a.end, a.text) != (b.id, b.start, b.end, b.text)
                    or not a.english.strip() for a, b in zip(completed, segments)):
                raise DubbingError("Translation checkpoint does not match its source transcript.")
            for segment in segments[len(completed):]:
                print(f"[translate] Segment {len(completed) + 1}/{len(segments)}", flush=True)
                english = translate_text(segment.text, segment.language or language,
                                         model=settings.gemini_model)
                completed.append(replace(segment, english=english))
                save_segments(translated_path, completed, language=language,
                              provider="gemini", model=settings.gemini_model)
            return completed

        translated = stage("translate", translate)
        report["translation"] = {"provider": "gemini", "model": settings.gemini_model,
                                 "translated_segments": len(translated)}
        # Edits to English text invalidate just that segment's synthesized clip.
        tts_directory = directory / "speech"
        tts_directory.mkdir(exist_ok=True)
        text_record = tts_directory / "texts.json"
        previous = read_json(text_record) if text_record.exists() else {}
        current = {str(s.id): s.english for s in translated}
        for key, text in current.items():
            if previous.get(key) != text:
                (tts_directory / f"{int(key):06d}.wav").unlink(missing_ok=True)
        write_json(text_record, current)
        clips = stage("synthesize", lambda: asyncio.run(synthesize(
            translated, tts_directory, settings.voice, settings.rate)))
        track = directory / "english.wav"
        stage("align", lambda: assemble(translated, clips, info.duration, track, settings.max_speed))
        stage("mux", lambda: mux(video, track, output))
        report.pop("error", None)
        report.update(status="complete", output=str(output), language=language, mode="english_dub",
                      duration=info.duration, video=asdict(info))
        print(f"[complete] Saved {output}", flush=True)
        return output
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        attempt["elapsed_seconds"] = time.perf_counter() - started
        report["total_processing_seconds"] = sum(a.get("elapsed_seconds", 0) for a in report["attempts"])
        report["packages"] = {name: importlib.metadata.version(name)
                              for name in ("yt-dlp", "openai-whisper", "faster-whisper", "torch", "google-genai", "edge-tts")}
        write_json(report_path, report)
