"""English stock-voice synthesis, cached per segment with bounded retries."""

import asyncio
import hashlib
import json
from pathlib import Path

from .data import Segment, read_json, write_json
from .errors import DubbingError
from .media import fit_audio, pcm_duration


async def synthesize(segments: list[Segment], directory: Path, voice: str,
                     rate: str = "+0%") -> list[Path]:
    import edge_tts
    import aiohttp

    directory.mkdir(parents=True, exist_ok=True)
    cache_path = directory / "cache.json"
    cache = read_json(cache_path) if cache_path.exists() else {}
    paths = []
    for index, segment in enumerate(segments):
        if not segment.english.strip():
            raise DubbingError(f"Segment {segment.id} has no English translation.")
        path = directory / f"{segment.id:06d}.wav"
        signature = hashlib.sha256(json.dumps(
            [segment.english, voice, rate, "edge-tts", 24000], ensure_ascii=False).encode()).hexdigest()
        print(f"[synthesize] Segment {index + 1}/{len(segments)}", flush=True)
        if path.exists() and cache.get(str(segment.id)) == signature:
            pcm_duration(path, 24000)
        else:
            compressed = directory / f"{segment.id:06d}.part.mp3"
            temporary = path.with_suffix(".part.wav")
            for attempt in range(3):
                try:
                    speech = edge_tts.Communicate(segment.english, voice=voice, rate=rate)
                    await asyncio.wait_for(speech.save(str(compressed)), timeout=120)
                    break
                except (OSError, TimeoutError, aiohttp.ClientError, edge_tts.exceptions.EdgeTTSException) as exc:
                    if attempt == 2:
                        raise DubbingError(f"TTS failed for segment {segment.id}: {exc}") from exc
                    print(f"[synthesize] Retrying segment {segment.id}", flush=True)
                    await asyncio.sleep(2 ** attempt)
            fit_audio(compressed, temporary, 1.0)
            pcm_duration(temporary, 24000)
            temporary.replace(path)
            compressed.unlink(missing_ok=True)
            cache[str(segment.id)] = signature
            write_json(cache_path, cache)
        paths.append(path)
    return paths
