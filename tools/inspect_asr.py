"""Print rejected script passages and save a compact ASR diagnostic report."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from avd.data import read_json, write_json
from avd.errors import DubbingError
from avd.languages import validate_script


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--language", required=True)
    args = parser.parse_args()
    payload = read_json(args.raw)
    rejected = []
    for segment in payload["segments"]:
        try:
            validate_script(segment["text"], args.language)
        except DubbingError:
            rejected.append(segment)
    evidence = {
        "raw_file": str(args.raw.resolve()),
        "language": args.language,
        "segment_count": len(payload["segments"]),
        "rejected_script_passages": rejected,
        "text": payload.get("text", ""),
        "meaning_accuracy_reviewed": False,
    }
    write_json(args.raw.with_name(args.raw.stem + "_review.json"), evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
