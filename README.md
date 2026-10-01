# Automated Video Dubbing System

Turn a YouTube video into an English-dubbed video using a Python CLI. The program downloads the video, transcribes its speech, translates the text with Gemini, generates English speech with Edge TTS, and replaces the audio with FFmpeg. It prints progress and saves checkpoints and processing-time reports.

The downloaded video stream is copied without re-encoding. The English audio uses a stock voice; original-speaker voice cloning and emotion matching are not implemented.

## Setup

Run commands from the project root. Install Python 3.12+, `uv`, and FFmpeg/FFprobe on PATH:

```powershell
uv sync --locked
uv run avd doctor
```

On a fresh clone, create a **`.env` file** from the template. Keep an existing configured `.env`:

```powershell
Copy-Item .env.example .env
```

Set your Gemini API key in `.env`:

```dotenv
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.8-flash
```

`.env` is ignored by Git; `.env.example` is the shareable blank template. The CLI loads the file in the current directory. `--env-file path/to/.env` selects another file. Process environment variables take precedence. Keys are never included in job settings or reports.

`doctor` checks the local environment without calling Gemini. To make a small live translation request:

```powershell
uv run avd doctor --check-gemini
```

Get a key from [Google AI Studio](https://aistudio.google.com/apikey). Translation sends source text to Google. Gemini requires internet access, model access and sufficient API quota; usage may incur charges. Select a model available to your account with `GEMINI_MODEL` or `--gemini-model`. See the [model catalog](https://ai.google.dev/gemini-api/docs/models) and [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).

## Usage

```powershell
uv run avd dub "https://www.youtube.com/watch?v=VIDEO_ID" --language hi
uv run avd dub "URL" --language bn --output output/bengali_english.mkv
uv run avd dub "URL" --language de
uv run avd dub "URL" --language fr
uv run avd dub "assets/local_video.mp4" --language ta
```

Omit `--language` for detection from samples near the beginning, middle and end. For a recording with one known language, specify it when detection is uncertain. Mixed-language speech needs manual review; setting one language does not implement passage-level language switching. English input preserves its original English audio and video without translation or synthesis; it needs no API key. `uv run avd languages` lists supported routing tags and Whisper coverage.

```powershell
uv run avd dub "URL" --language hi --whisper-model large-v3 --asr-backend faster-whisper
uv run avd dub "URL" --language hi --gemini-model gemini-3.8-flash
uv run avd dub "URL" --language hi --voice en-IN-NeerjaNeural --rate=+5%
uv run avd dub "URL" --language hi --device cuda --chunk-seconds 300
uv run avd dub "URL" --language hi --env-file config/.env --work-dir output/custom
```

Defaults: faster-whisper `large-v3`, CPU, five-minute ASR chunks, Gemini translation, and `en-IN-PrabhatNeural` English speech. First-run Whisper model downloads can be substantial. Downloads default to at most 720p (`--height` changes this). MKV is the default; MP4 is available when compatible with the video codec. Preserved visuals means no re-encoding of the selected downloaded video stream.

`--device cuda` requires compatible GPU hardware and dependencies. The verified local environment uses CPU. Use `uv run avd dub --help` for all options.

## Architecture

| File in `src/avd/` | Responsibility |
| --- | --- |
| `cli.py`, `config.py` | CLI commands, settings and `.env` loading |
| `download.py`, `media.py` | yt-dlp download, audio extraction, FFmpeg output and media checks |
| `routing.py`, `languages.py` | Source-language detection, tags and script checks |
| `recognition.py`, `transcription.py` | ASR backends and timestamped transcription chunks |
| `gemini_translation.py` | One function that translates text using the Google Gen AI SDK |
| `synthesis.py`, `timeline.py` | Cached Edge TTS speech and timestamp alignment |
| `pipeline.py`, `data.py` | Stage orchestration, checkpoints and processing reports |

Gemini is the only translation provider. The retired IndicTrans2 adapter, fastText detector and compatibility wrappers have been removed. Whisper and faster-whisper dependencies remain in use.

## Translation, timing and resume

Gemini translation is one function in `src/avd/gemini_translation.py`. It takes text, source language, target language and an optional model, calls `client.models.generate_content`, and returns the translated text:

```python
from avd.gemini_translation import translate_text

english = translate_text("Bonjour le monde!", "French", "English")
```

The pipeline calls this function once per speech segment and saves each completed translation. Timestamps remain in the pipeline; Gemini returns text only. Repeat the same command to resume from the last saved segment. Changing the source or processing settings creates a separate job. Empty responses and API errors stop processing; repeat the command after resolving the cause. Review the translations against the original; a successful API response does not certify accuracy.

Each job saves these artifacts under `output/jobs/<id>/`:

- `source.json` and downloaded video; local inputs remain at their original path.
- `source.wav`, mono 16 kHz transcription audio.
- `transcription/chunk_*.json`, diagnostic raw ASR results and `transcript.json`.
- `translation.json`, English text, source IDs/times and provider/model.
- `speech/*.wav`, cached speech; `alignment.json`, timing and tempo changes.
- `english.wav` and `english.mkv`, assembled audio and final video.
- `report.json`, stage/attempt times, configuration, status and errors.

Repeat the command to resume a job. Editing English text in `translation.json` invalidates the corresponding synthesized clips. If a crash leaves `.running`, confirm the process stopped before removing that specific lock file. Source videos are never overwritten.

Speech starts at its source timestamp. Silence before the next segment can absorb longer phrasing. Pitch-preserving tempo changes default to a maximum of 1.35×. Excessive compression or overlap causes an error instead of cutting sentences. Review/shorten the affected English wording and resume; `--max-speed` changes the limit.

Reports accumulate failed/resumed processing attempts. Model downloads and setup affect first runs; record setup time separately when comparing performance. API usage is available in Google AI Studio.

## Verification

```powershell
uv run python -m unittest discover -s tests -v
uv build --wheel
```

Tests check behavior using mocked model/API responses and synthetic media, including language parameters, empty responses, API errors, routing, alignment and video preservation. They do not measure live translation, recognition or listening quality.

Run the short smoke checks:

```powershell
uv run python tools/hindi_smoke.py
uv run python tools/english_smoke.py --language en
uv run python tools/english_smoke.py --language auto
```

The Hindi check uses real ASR, Gemini translation and Edge speech synthesis. It needs internet access and Gemini quota. English checks preserve the source audio/video; automatic routing also runs language detection. These commands reuse existing fixtures and matching job checkpoints.

Latest recorded verification, from the cleanup run on 2026-10-01:

- All 19 automated tests passed.
- The short Hindi dub completed, preserving the source video packets.
- Repeating the Hindi job reused the transcript, translation and synthesized speech.
- Explicit and automatic English routing preserved both audio and video packets.
- The wheel build and CLI checks passed.

Evidence is saved locally in `output/cleanup_verification.json`. The Hindi test output is `output/jobs/724c3e36543c2031/english.mkv`. These are synthetic controls; human listening review and full-video quality validation remain pending. Generated output files are not part of a fresh clone.

## Preserved source videos

The five older jobs were removed after the current checks passed. Their downloaded source videos and extracted audio were preserved in `output/sources/`, with matching SHA-256 hashes recorded in `output/sources/preservation.json`.

| Language | Local source video | Duration |
| --- | --- | --- |
| Hindi | `output/sources/Ph6p7VUmk74.mkv` | 30:53 |
| Bengali | `output/sources/Cta7Jtl2mtw.mkv` | 1:03:36 |

Dub either preserved source directly:

```powershell
uv run avd dub "output/sources/Ph6p7VUmk74.mkv" --language hi
uv run avd dub "output/sources/Cta7Jtl2mtw.mkv" --language bn
```

The Bengali source includes a Hindi opening passage, so forcing Bengali throughout requires careful review. Completed full-length dubs are still outstanding.

To process both benchmark sources sequentially using the preserved downloads:

```powershell
uv run python tools/run_benchmarks.py --reuse-downloads
```

This runs the full videos and may use substantial time and Gemini quota. Old job checkpoints were deleted; the preserved videos avoid another download, but their transcription and translation must run again. New jobs resume normally.

## Limitations

- Edge TTS uses a stock English voice; original-speaker cloning and guaranteed emotion matching are not implemented.
- One source language and one English voice are used per job; diarization and automatic passage-level code-switch routing are not implemented.
- Whisper does not support every listed Indian language. Unsupported languages are rejected before transcription. Gemini cannot reliably repair an inaccurate source transcript.
- Script validation catches some obvious ASR failures, not every hallucination. Speech at chunk boundaries needs review.
- Original audio is replaced completely, including background music and effects.
- Fluent translations may need manual wording adjustment to fit their speaking slots.

## Assignment submission

The assignment requires **one 30-minute video and one two-hour video**. For each retain the source, reviewed English dub and actual processing time. Short videos are development checks, not substitutes. Inspect meaning, names/numbers, naturalness and timing throughout each video.

Record a two-minute walkthrough with before/after playback, architecture, Gemini choice/API dependency, segment timing, resume behavior and measured processing time. Long-video outputs and the walkthrough still require actual runs and human review; the program does not email submissions automatically.

The retained Bengali video is about one hour, so a two-hour source is still needed. Email the source videos, dubbed outputs, processing times and walkthrough to `careers@idealabsdigital.com` when complete.
