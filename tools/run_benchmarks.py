"""Run user-selected videos sequentially, keeping logs and failure records."""

from contextlib import redirect_stderr, redirect_stdout
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from avd.data import read_json, write_json
from avd.pipeline import Settings, run_pipeline
from avd.config import DEFAULT_GEMINI_MODEL, load_environment


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, value):
        self.terminal.write(value)
        self.logfile.write(value)
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def main() -> int:
    load_environment()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--reuse-downloads", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--gemini-model", default=os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL)
    args = parser.parse_args()
    if args.threads is not None:
        if args.threads < 1:
            parser.error("--threads must be positive")
    root = Path("output").resolve()
    directory = root / "benchmarks"
    sources = read_json(directory / "sources.json")
    previous_path = directory / "runs.json"
    if previous_path.exists():
        history_path = directory / "runs_history.json"
        history = read_json(history_path) if history_path.exists() else []
        history.append(read_json(previous_path))
        write_json(history_path, history)
    records = []
    for source in sources:
        language = source["language"]
        output = directory / f"{language}_english.mkv"
        began = time.perf_counter()
        record = {**source, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "status": "running", "voice_mode": "single stock English voice",
                  "whisper_model": args.whisper_model}
        records.append(record)
        write_json(directory / "runs.json", records)
        with (directory / f"{language}.log").open("a", encoding="utf-8") as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                try:
                    print(f"[benchmark] {language}: {source['title']}", flush=True)
                    input_source = source["url"]
                    if args.reuse_downloads:
                        retained = root / "sources" / f"{source['id']}.mkv"
                        if retained.is_file():
                            input_source = str(retained)
                            record["retained_source"] = input_source
                        else:
                            for cached_record in (root / "jobs").glob("*/source.json"):
                                cached_report = read_json(cached_record.parent / "report.json")
                                cached_source = Path(read_json(cached_record)["path"])
                                if (cached_report["identity"]["source"] == source["url"]
                                        and cached_source.is_file()):
                                    input_source = str(cached_source)
                                    record["retained_source"] = input_source
                                    break
                    result = run_pipeline(input_source, root,
                                          Settings(language=language, whisper_model=args.whisper_model,
                                                   asr_threads=args.threads, gemini_model=args.gemini_model), output)
                    record.update(status="complete", output=str(result))
                except KeyboardInterrupt:
                    record.update(status="interrupted")
                    raise
                except Exception as exc:
                    record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    print(f"[benchmark] {language} failed: {exc}", file=sys.stderr, flush=True)
                finally:
                    record["attempt_seconds"] = time.perf_counter() - began
                    write_json(directory / "runs.json", records)
    print(json.dumps(records, ensure_ascii=False, indent=2), flush=True)
    return 0 if all(record["status"] == "complete" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
