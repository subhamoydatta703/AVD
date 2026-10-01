"""Verify English passthrough with a labeled synthetic English control."""

import argparse
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from avd.data import write_json
from avd.media import run
from avd.pipeline import Settings, run_pipeline

TEXT = ("Hello. Today we are testing this video dubbing system. This video is already in English, "
        "so its original voice and picture should stay the same.")


def digest(path: Path, stream: str) -> str:
    return run(["ffmpeg", "-v", "error", "-i", str(path), "-map", stream,
                "-c", "copy", "-f", "hash", "-hash", "sha256", "-"]).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=["en", "auto"], default="auto")
    args = parser.parse_args()
    directory = Path("output/english-control").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "synthetic_english.mp4"
    if not source.exists():
        import edge_tts
        speech = directory / "source.mp3"
        asyncio.run(edge_tts.Communicate(TEXT, voice="en-IN-PrabhatNeural").save(str(speech)))
        run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
             "color=c=blue:s=640x360:r=25", "-i", str(speech), "-c:v", "libx264",
             "-c:a", "aac", "-shortest", str(source)])
    output = run_pipeline(str(source), Path("output").resolve(),
                          Settings(language="en" if args.language == "en" else None))
    checks = {stream: digest(source, stream) == digest(output, stream)
              for stream in ("0:v:0", "0:a:0")}
    write_json(directory / f"verification_{args.language}.json", {
        "kind": "synthetic English control; not a submission video",
        "source": str(source), "output": str(output),
        "routing": args.language, "source_text": TEXT,
        "identical_video_packets": checks["0:v:0"],
        "identical_audio_packets": checks["0:a:0"],
    })
    if not all(checks.values()):
        raise RuntimeError("English passthrough changed the source audio or video packets.")
    print("[verify] English source audio and video packets are identical.", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
