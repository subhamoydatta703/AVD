"""Test a cached ASR model on short passages from the selected real videos."""

import argparse
from pathlib import Path
import sys
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from avd.data import read_json, write_json
from avd.errors import DubbingError
from avd.languages import validate_script
from avd.recognition import Recognizer

SOURCES = {
    "hi": Path("output/sources/Ph6p7VUmk74.wav"),
    "bn": Path("output/sources/Cta7Jtl2mtw.wav"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--backend", choices=["faster-whisper", "whisper"], default="faster-whisper")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    recognizer = Recognizer(args.model, args.backend, "cpu", args.threads, Path("output/models"))
    recognizer.load()
    directory = Path("output/benchmarks/asr-probes") / f"{args.backend}-{args.model}-beam5"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for language, source in SOURCES.items():
        for start in (0, 150):
            clip = directory / f"{language}_{start}.wav"
            with wave.open(str(source), "rb") as audio:
                audio.setpos(start * audio.getframerate())
                with wave.open(str(clip), "wb") as output:
                    output.setparams(audio.getparams())
                    output.writeframes(audio.readframes(30 * audio.getframerate()))
            print(f"[ASR probe] {args.model}: {language} at {start}s", flush=True)
            began = time.perf_counter()
            result = recognizer.transcribe(clip, language)
            elapsed = time.perf_counter() - began
            write_json(clip.with_suffix(".json"), result)
            failures = []
            for segment in result["segments"]:
                try:
                    validate_script(segment["text"], language)
                except DubbingError:
                    failures.append(segment)
            record = {"model": args.model, "backend": args.backend, "beam_size": 5,
                      "language": language, "offset": start,
                      "elapsed_seconds": elapsed, "text": result["text"],
                      "script_failures": failures, "meaning_accuracy_reviewed": False}
            records.append(record)
            write_json(directory / "report.json", records)
            print(f"[ASR probe] {elapsed:.1f}s: {result['text']}", flush=True)
    recognizer.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
