"""ASR backends with explicit decoding and bounded, local model caching."""

from dataclasses import asdict
import gc
from pathlib import Path
import wave

from .errors import DubbingError

LARGE_V3_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"


def pcm_samples(audio: Path):
    """Feed our verified PCM directly, avoiding a second media decoder."""
    import numpy as np

    with wave.open(str(audio), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
            raise DubbingError("ASR requires mono 16 kHz, 16-bit PCM from the FFmpeg extraction stage.")
        return np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768


class Recognizer:
    def __init__(self, name: str, backend: str, device: str, threads: int,
                 model_root: Path):
        if backend not in {"faster-whisper", "whisper"} or threads < 1:
            raise DubbingError("Choose a supported ASR backend and positive CPU thread count.")
        self.name, self.backend, self.device = name, backend, device
        self.threads, self.model_root = threads, model_root
        self.model = None

    def load(self):
        if self.model is not None:
            return self.model
        print(f"[ASR] Loading {self.backend}/{self.name} on {self.device}", flush=True)
        try:
            if self.backend == "faster-whisper":
                import ctranslate2
                from faster_whisper import WhisperModel

                types = ctranslate2.get_supported_compute_types(self.device)
                preferred = "int8" if self.device == "cpu" else "float16"
                compute = preferred if preferred in types else "float32"
                cached = self.model_root / "faster-whisper-large-v3"
                local = cached if self.name == "large-v3" and (cached / "model.bin").is_file() else None
                self.model = WhisperModel(
                    str(local) if local else self.name, device=self.device,
                    compute_type=compute, cpu_threads=self.threads,
                    download_root=str(self.model_root / "hub"),
                    revision=LARGE_V3_REVISION if self.name == "large-v3" else None,
                )
                print(f"[ASR] Compute type: {compute}", flush=True)
            else:
                import torch
                import whisper

                if self.device == "cuda" and not torch.cuda.is_available():
                    raise DubbingError("CUDA requested but unavailable in this PyTorch runtime.")
                torch.set_num_threads(self.threads)
                self.model = whisper.load_model(self.name, device=self.device)
        except DubbingError:
            raise
        except Exception as exc:
            raise DubbingError(f"ASR could not load {self.backend}/{self.name}: {exc}") from exc
        return self.model

    def transcribe(self, audio: Path, language: str | None) -> dict:
        model = self.load()
        try:
            if self.backend == "faster-whisper":
                segments, info = model.transcribe(
                    pcm_samples(audio), language=language, task="transcribe",
                    beam_size=5, best_of=5, temperature=(0.0, 0.2),
                    condition_on_previous_text=False, word_timestamps=True,
                    vad_filter=True, vad_parameters={"min_silence_duration_ms": 300},
                    log_progress=True,
                )
                rows = [asdict(segment) for segment in segments]
                return {"language": info.language, "language_probability": info.language_probability,
                        "text": " ".join(row["text"].strip() for row in rows), "segments": rows}
            return model.transcribe(
                str(audio), language=language, task="transcribe", fp16=self.device == "cuda",
                verbose=False, word_timestamps=True, condition_on_previous_text=False,
                beam_size=5, best_of=5, temperature=(0.0, 0.2),
            )
        except Exception as exc:
            raise DubbingError(f"ASR inference failed for {audio.name}: {exc}") from exc

    def detect(self, audio: Path) -> tuple[str, float, bool]:
        model = self.load()
        if self.backend == "faster-whisper":
            # Getting info performs detection; no decoding is needed from the lazy generator.
            _, info = model.transcribe(pcm_samples(audio), task="transcribe", vad_filter=True)
            return info.language, info.language_probability, info.duration_after_vad >= 1
        import whisper

        samples = whisper.pad_or_trim(whisper.load_audio(str(audio)))
        mel = whisper.log_mel_spectrogram(samples, n_mels=model.dims.n_mels).to(model.device)
        _, probabilities = model.detect_language(mel)
        language = max(probabilities, key=probabilities.get)
        # The legacy backend has no VAD evidence, so automatic passthrough is conservative.
        return language, probabilities[language], False

    def close(self):
        self.model = None
        gc.collect()
        if self.backend == "whisper" and self.device == "cuda":
            import torch
            torch.cuda.empty_cache()
