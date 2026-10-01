"""Generate a labeled Hindi control fixture and run real dubbing; not a submission video."""

import argparse
import asyncio
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from avd.data import write_json
from avd.media import run
from avd.pipeline import Settings, run_pipeline
from avd.config import load_environment

TEXT = "नमस्ते। आज हम इंटरनेट के बारे में बात करेंगे। इंटरनेट हमें दुनिया भर के लोगों से जोड़ता है।"


def main() -> None:
    load_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("output"))
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--asr-backend", choices=["faster-whisper", "whisper"], default="faster-whisper")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    root = args.work_dir.resolve()
    directory = root / "hindi-control"
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "synthetic_hindi.mp4"
    if not source.exists():
        import edge_tts
        audio = directory / "source.mp3"
        asyncio.run(edge_tts.Communicate(TEXT, voice="hi-IN-MadhurNeural").save(str(audio)))
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             "color=c=blue:s=640x360:r=25", "-i", str(audio), "-c:v", "libx264",
             "-c:a", "aac", "-shortest", str(source)])
    begin = time.perf_counter()
    output = run_pipeline(str(source), root, Settings(language="hi", whisper_model=args.whisper_model,
                                                    asr_backend=args.asr_backend))
    write_json(directory / "verification.json", {
        "kind": "synthetic Hindi control fixture; not a YouTube/assignment submission",
        "source_text": TEXT, "source": str(source), "output": str(output),
        "processing_seconds": time.perf_counter() - begin,
        "whisper_model": args.whisper_model,
        "human_listening_review": False,
    })


if __name__ == "__main__":
    main()
