"""Behavioral checks without model downloads or network access."""

import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import wave

from avd.data import Segment, load_segments, read_json, save_segments
from avd.download import download_video
from avd.errors import DubbingError
from avd.languages import flores_tag, iso_code, validate_script
from avd.media import mux, probe, run
from avd.pipeline import Settings, run_pipeline
from avd.timeline import RATE, assemble
from avd.transcription import transcribe
from avd.routing import decide_language

ROOT = Path(__file__).resolve().parents[1]


def tone(path: Path, seconds: float) -> None:
    samples = b"".join(struct.pack("<h", round(5000 * math.sin(2 * math.pi * 440 * n / RATE)))
                       for n in range(round(seconds * RATE)))
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(RATE)
        audio.writeframes(samples)


class CoreTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "output" / "tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_invalid_timestamps_fail(self):
        for start, end in [(float("nan"), 1), (0, float("inf")), (-1, 2), (1, 1)]:
            with self.assertRaises(DubbingError):
                Segment(0, start, end, "speech")

    def test_transcript_preserves_unicode_and_times(self):
        segments = [Segment(0, 1.25, 2.5, "नमस्ते", "Hello")]
        path = self.directory / "transcript.json"
        save_segments(path, segments, language="hin_Deva")
        self.assertEqual(load_segments(path), segments)

    def test_correct_language_tags_and_script_overrides(self):
        self.assertEqual(flores_tag("ne"), "npi_Deva")
        self.assertEqual(flores_tag("kok"), "gom_Deva")
        self.assertEqual(iso_code("mni_Mtei"), "mni")
        self.assertEqual(flores_tag("fr"), "fra_Latn")
        self.assertEqual(flores_tag("de"), "deu_Latn")
        with self.assertRaises(DubbingError):
            flores_tag("unknown")

    def test_rejects_wrong_script_but_allows_english_terms(self):
        validate_script("यह Internet के बारे में है।", "hi")
        with self.assertRaises(DubbingError):
            validate_script("یہ اردو عبارت ہے", "hi")
        validate_script("یہ اردو عبارت ہے", "ur")

    def test_rejected_asr_preserves_raw_output_and_rechecks_it_on_resume(self):
        source = self.directory / "source.wav"
        tone(source, 1)
        directory = self.directory / "asr"
        raw = {"language": "hi", "segments": [
            {"start": 0, "end": 0.5, "text": "یہ اردو عبارت ہے"},
        ]}
        model = MagicMock()
        model.transcribe.return_value = raw
        with patch("whisper.load_model", return_value=model) as loader:
            for _ in range(2):
                with self.assertRaisesRegex(DubbingError, "Chunk 1, at 0.00s"):
                    transcribe(source, directory, language="hi", model_name="small", backend="whisper")
        loader.assert_called_once()
        self.assertEqual(read_json(directory / "raw_00000.json"), raw)
        self.assertFalse((directory / "chunk_00000.json").exists())

    def test_download_uses_postprocessed_path_without_second_extraction(self):
        final = self.directory / "id.final.mkv"
        final.write_bytes(b"media fixture")
        fake = MagicMock()

        def construct(options):
            def extract(*args, **kwargs):
                options["post_hooks"][0](str(final))
                return {"id": "id", "ext": "webm", "filepath": str(self.directory / "wrong.webm")}
            fake.extract_info.side_effect = extract
            fake.prepare_filename.return_value = str(self.directory / "wrong.webm")
            return fake

        fake.__enter__.return_value = fake
        with patch("yt_dlp.YoutubeDL", side_effect=construct):
            self.assertEqual(download_video("https://youtube.com/watch?v=id", self.directory), final.resolve())
        fake.extract_info.assert_called_once()

    def test_rejects_non_youtube_hosts_before_network(self):
        with self.assertRaises(DubbingError):
            download_video("https://youtube.com.evil.example/watch?v=id", self.directory)

    def test_timeline_preserves_gaps_duration_and_video_packets(self):
        first, second = self.directory / "first.wav", self.directory / "second.wav"
        tone(first, 0.4)
        tone(second, 0.4)
        segments = [Segment(0, 0.5, 1.0, "first", "First"), Segment(1, 2, 2.5, "second", "Second")]
        track = self.directory / "dub.wav"
        assemble(segments, [first, second], 3, track)
        with wave.open(str(track), "rb") as audio:
            self.assertEqual(audio.getnframes(), RATE * 3)
            self.assertEqual(audio.readframes(RATE // 2), bytes(RATE))
            self.assertNotEqual(audio.readframes(100), bytes(200))
            audio.setpos(RATE)
            self.assertEqual(audio.readframes(RATE // 2), bytes(RATE))
        source = self.directory / "source with spaces.mp4"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             "color=c=blue:s=320x180:r=10:d=3", "-f", "lavfi", "-i", "sine=frequency=220:duration=3",
             "-c:v", "libx264", "-c:a", "aac", str(source)])
        destination = self.directory / "dubbed.mkv"
        mux(source, track, destination)
        self.assertLess(abs(probe(destination).duration - 3), 0.3)
        def digest(path):
            return run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
                        "-c:v", "copy", "-f", "hash", "-hash", "sha256", "-"]).strip()
        self.assertEqual(digest(source), digest(destination))

    def test_timeline_refuses_excessive_compression_without_publishing_output(self):
        clip = self.directory / "long.wav"
        tone(clip, 2)
        destination = self.directory / "dub.wav"
        with self.assertRaisesRegex(DubbingError, "has not been truncated"):
            assemble([Segment(0, 0, 0.5, "source", "English")], [clip], 1, destination)
        self.assertFalse(destination.exists())

    def test_overlapping_source_speech_is_rejected(self):
        clip = self.directory / "clip.wav"
        tone(clip, 0.2)
        with self.assertRaises(DubbingError):
            assemble([Segment(0, 0, 1, "one"), Segment(1, 0.5, 1.5, "two")],
                     [clip, clip], 2, self.directory / "dub.wav")

    def test_resume_after_synthesis_failure_reuses_transcription_and_records_attempts(self):
        import json
        source = self.directory / "source.mp4"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             "color=s=320x180:r=10:d=2", "-f", "lavfi", "-i", "sine=duration=2",
             "-c:v", "libx264", "-c:a", "aac", str(source)])
        calls = 0

        async def synthesis(segments, directory, voice, rate):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise DubbingError("simulated network interruption")
            clip = directory / "000000.wav"
            tone(clip, 0.3)
            return [clip]

        root = self.directory / "work"

        output = self.directory / "english.mkv"
        translated = [Segment(0, 0.5, 1, "नमस्ते", "Hello")]
        with patch("avd.pipeline.transcribe", return_value=(translated, "hin_Deva")) as asr:
            with patch("avd.pipeline.synthesize", side_effect=synthesis), patch(
                    "avd.pipeline.translate_text", return_value="Hello") as translator, patch(
                    "avd.pipeline.gemini_api_key", return_value="test-key"):
                with self.assertRaises(DubbingError):
                    run_pipeline(str(source), root, Settings(language="hi"), output)
                reports = list((root / "jobs").glob("*/report.json"))
                self.assertEqual(json.loads(reports[0].read_text())["status"], "failed")
                self.assertEqual(run_pipeline(str(source), root, Settings(language="hi"), output), output)
        asr.assert_called_once()
        translator.assert_called_once()
        report = json.loads(reports[0].read_text())
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["attempts"]), 2)
        self.assertIn("synthesize", report["attempts"][0]["stages"])
        self.assertFalse(list((root / "jobs").glob("*/.running")))

    def test_english_input_copies_both_tracks_without_translation_or_synthesis(self):
        source = self.directory / "english.mp4"
        run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
             "color=c=blue:s=320x180:r=10:d=2", "-f", "lavfi", "-i", "sine=duration=2",
             "-c:v", "libx264", "-c:a", "aac", str(source)])
        for language in ("en", None):
            with self.subTest(language=language):
                root = self.directory / ("explicit" if language else "automatic")
                output = root / "english.mkv"
                with patch("avd.pipeline.detect_source_language", return_value="eng_Latn") as detector, patch(
                        "avd.pipeline.transcribe") as asr, patch(
                        "avd.pipeline.translate_text") as translator, patch(
                        "avd.pipeline.gemini_api_key") as key, patch(
                        "avd.pipeline.synthesize") as tts:
                    self.assertEqual(run_pipeline(str(source), root, Settings(language=language), output), output)
                asr.assert_not_called()
                translator.assert_not_called()
                key.assert_not_called()
                tts.assert_not_called()
                self.assertEqual(detector.call_count, 0 if language else 1)
                for stream in ("0:v:0", "0:a:0"):
                    def digest(path):
                        return run(["ffmpeg", "-v", "error", "-i", str(path), "-map", stream,
                                    "-c", "copy", "-f", "hash", "-hash", "sha256", "-"]).strip()
                    self.assertEqual(digest(source), digest(output))
                report = read_json(next((root / "jobs").glob("*/report.json")))
                self.assertEqual(report["mode"], "english_passthrough")

    def test_language_routing_rejects_uncertain_silent_or_mixed_samples(self):
        def sample(language, probability=0.95, speech=True):
            return {"language": language, "probability": probability, "speech_present": speech}
        for language, tag in (("en", "eng_Latn"), ("hi", "hin_Deva"), ("bn", "ben_Beng"),
                              ("ta", "tam_Taml"), ("te", "tel_Telu"), ("mr", "mar_Deva"),
                              ("fr", "fra_Latn"), ("de", "deu_Latn")):
            self.assertEqual(decide_language([sample(language)]), tag)
        for samples in ([sample("en", 0.5)], [sample("en", speech=False)],
                        [sample("en"), sample("hi")], [sample("unknown")]):
            with self.assertRaises(DubbingError):
                decide_language(samples)

    def test_indian_inputs_are_translated_before_english_synthesis(self):
        source = self.directory / "indian.mp4"
        run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
             "color=c=blue:s=320x180:r=10:d=2", "-f", "lavfi", "-i", "sine=duration=2",
             "-c:v", "libx264", "-c:a", "aac", str(source)])
        for language, tag, text in (("hi", "hin_Deva", "नमस्ते"), ("bn", "ben_Beng", "নমস্কার")):
            with self.subTest(language=language):
                root = self.directory / language

                source_segments = [Segment(0, 0.5, 1, text)]
                translated = [Segment(0, 0.5, 1, text, "Hello")]
                async def synthesis(segments, directory, voice, rate):
                    self.assertEqual(segments, translated)
                    clip = directory / "000000.wav"
                    tone(clip, 0.3)
                    return [clip]
                with patch("avd.pipeline.transcribe", return_value=(source_segments, tag)), patch(
                        "avd.pipeline.translate_text", return_value="Hello") as translator, patch(
                        "avd.pipeline.gemini_api_key", return_value="test-key"), patch(
                        "avd.pipeline.synthesize", side_effect=synthesis):
                    run_pipeline(str(source), root, Settings(language=language))
                translator.assert_called_once_with(text, tag, model=Settings().gemini_model)


if __name__ == "__main__":
    unittest.main()
