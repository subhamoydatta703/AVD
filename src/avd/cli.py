"""Terminal interface and dependency checks."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

from .data import write_json
from .errors import DubbingError
from .pipeline import Settings, run_pipeline
from .languages import ISO_TO_FLORES
from .config import DEFAULT_GEMINI_MODEL, load_environment
from .gemini_translation import translate_text


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gemini-model", default=os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL)
    parser.add_argument("--env-file", type=Path, help="Credentials file (default: .env in current directory)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--work-dir", type=Path, default=Path("output"))


def doctor(args: argparse.Namespace) -> int:
    import torch

    package_names = ("torch", "edge-tts", "openai-whisper",
                     "faster-whisper", "ctranslate2",
                     "yt-dlp", "google-genai", "python-dotenv")
    packages = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    result = {"python": sys.version.split()[0], "packages": packages,
              "ffmpeg": bool(shutil.which("ffmpeg")), "ffprobe": bool(shutil.which("ffprobe")),
              "cuda_available": torch.cuda.is_available(), "compiled_cuda": torch.version.cuda,
              "gemini_api_key_present": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
              "translation_provider": "gemini",
              "gemini_model": args.gemini_model,
              "model_access": "not checked; --check-gemini makes a live translation request",
              "voice_mode": "stock English voice; no voice cloning"}
    args.work_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.work_dir / "environment.json", result)
    print(json.dumps(result, indent=2))
    if args.check_gemini:
        translated = translate_text("नमस्ते। आज हम इंटरनेट के बारे में बात करेंगे।", "Hindi",
                                    model=args.gemini_model)
        print(f"[gemini] API translation succeeded: {translated}", flush=True)
    return 0 if result["ffmpeg"] and result["ffprobe"] else 1


def main(argv: list[str] | None = None) -> int:
    environment_parser = argparse.ArgumentParser(add_help=False)
    environment_parser.add_argument("--env-file", type=Path)
    environment_args, _ = environment_parser.parse_known_args(argv)
    if environment_args.env_file and not environment_args.env_file.is_file():
        print(f"[error] Environment file does not exist: {environment_args.env_file}", file=sys.stderr)
        return 1
    load_environment(environment_args.env_file)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Dub multilingual YouTube videos into English.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("languages", help="List language tags and ASR coverage")
    check = commands.add_parser("doctor", help="Inspect dependencies and optionally check Gemini")
    model_options(check)
    check.add_argument("--check-gemini", action="store_true", help="Make a small live Gemini translation request")
    dub = commands.add_parser("dub", help="Download, transcribe, translate, synthesize and remix")
    model_options(dub)
    dub.add_argument("source", help="YouTube video URL, or a local video for repeatable testing")
    dub.add_argument("--language", help="ISO code (hi/bn/ta/de/fr/...) or FLORES tag; omit for detection")
    dub.add_argument("--whisper-model", default="large-v3")
    dub.add_argument("--asr-backend", choices=["faster-whisper", "whisper"], default="faster-whisper")
    dub.add_argument("--asr-threads", type=positive, default=8)
    dub.add_argument("--voice", default="en-IN-PrabhatNeural")
    dub.add_argument("--rate", default="+0%", help="TTS rate, e.g. --rate=+10%%")
    dub.add_argument("--height", type=positive, default=720)
    dub.add_argument("--chunk-seconds", type=positive, default=300)
    dub.add_argument("--max-speed", type=float, default=1.35)
    dub.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "languages":
            from whisper.tokenizer import LANGUAGES
            print(json.dumps([
                {"iso": code, "tag": tag, "asr_supported": code in LANGUAGES,
                 "output_language": "English", "mode": "passthrough" if code == "en" else "dub",
                 "quality_note": "Model coverage; individual languages require real-sample evaluation"}
                for code, tag in ISO_TO_FLORES.items()], indent=2))
            return 0
        if args.command == "doctor":
            return doctor(args)
        if not 1 <= args.max_speed <= 2:
            parser.error("--max-speed must be between 1 and 2")
        settings = Settings(
            language=args.language, whisper_model=args.whisper_model, device=args.device,
            asr_backend=args.asr_backend, asr_threads=args.asr_threads,
            gemini_model=args.gemini_model, voice=args.voice, rate=args.rate,
            height=args.height, chunk_seconds=args.chunk_seconds, max_speed=args.max_speed)
        run_pipeline(args.source, args.work_dir.resolve(), settings, args.output)
        return 0
    except KeyboardInterrupt:
        print("\n[interrupted] Checkpoints saved; repeat the command to resume.", file=sys.stderr)
        return 130
    except (DubbingError, ImportError, OSError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
