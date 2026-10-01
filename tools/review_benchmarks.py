"""Collect technical evidence and text samples for a completed dubbing job.

This report does not establish translation accuracy or listening quality.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from avd.data import read_json, write_json
from avd.media import probe


def video_hash(path: Path) -> str:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:V:0",
         "-c", "copy", "-f", "hash", "-hash", "sha256", "-"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def review(directory: Path) -> dict:
    report = read_json(directory / "report.json")
    source = Path(read_json(directory / "source.json")["path"])
    result = {
        "job": str(directory.resolve()),
        "status": report["status"],
        "source": str(source),
        "processing_seconds": report.get("total_processing_seconds"),
        "stage_seconds": report["stages"],
        "human_listening_review": "pending",
        "translation_meaning_review": "pending",
        "voice_mode": report["voice_mode"],
    }
    translated_path = directory / "translation.json"
    if translated_path.exists():
        segments = read_json(translated_path)["segments"]
        result["translated_segments"] = len(segments)
        fractions = (0, 0.2, 0.4, 0.6, 0.8, 1)
        indices = sorted({round(f * (len(segments) - 1)) for f in fractions}) if segments else []
        result["text_samples"] = [segments[i] for i in indices]
    alignment_path = directory / "alignment.json"
    if alignment_path.exists():
        alignment = read_json(alignment_path)
        result["alignment_segments"] = len(alignment)
        result["highest_required_tempos"] = sorted(
            alignment, key=lambda row: row["required_speed"], reverse=True,
        )[:10]
    output = Path(report["output"]) if report.get("output") else None
    if report["status"] == "complete" and output and output.is_file():
        original, dubbed = probe(source), probe(output)
        source_hash, output_hash = video_hash(source), video_hash(output)
        result.update(
            output=str(output),
            source_duration_seconds=original.duration,
            output_duration_seconds=dubbed.duration,
            duration_difference_seconds=abs(original.duration - dubbed.duration),
            video_packet_hash_source=source_hash,
            video_packet_hash_output=output_hash,
            video_packets_identical=source_hash == output_hash,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    arguments = parser.parse_args()
    evidence = review(arguments.job)
    destination = arguments.job / "technical_review.json"
    write_json(destination, evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    main()
