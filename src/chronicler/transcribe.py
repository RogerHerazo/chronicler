"""Local speech-to-text with faster-whisper."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import soundfile as sf

from chronicler.audio.source import SAMPLE_RATE

log = logging.getLogger(__name__)

# (model, approx download size) in order of preference per hardware tier.
GPU_DEFAULT_MODEL = "large-v3-turbo"
CPU_DEFAULT_MODEL = "small"
MODEL_SIZES_MB = {
    "tiny": 75,
    "base": 145,
    "small": 485,
    "medium": 1530,
    "large-v3-turbo": 1620,
    "large-v3": 3090,
}


@dataclass(frozen=True)
class Segment:
    start: float  # seconds from the start of the session
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DevicePlan:
    device: str
    compute_type: str
    model: str


def cuda_device_count() -> int:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def gpu_usable() -> bool:
    from chronicler import cuda

    return cuda_device_count() > 0 and not cuda.missing_libraries()


def plan_device(model: str = "auto", device: str = "auto") -> DevicePlan:
    if device == "auto":
        device = "cuda" if gpu_usable() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    if model == "auto":
        model = GPU_DEFAULT_MODEL if device == "cuda" else CPU_DEFAULT_MODEL
    return DevicePlan(device, compute_type, model)


def model_is_cached(model: str) -> bool:
    try:
        from faster_whisper.utils import download_model

        download_model(model, local_files_only=True)
        return True
    except Exception:
        return False


class Transcriber:
    """Loads the Whisper model once and transcribes chunks one at a time."""

    def __init__(self, plan: DevicePlan, language: str | None = None):
        self.plan = plan
        self.language = language or None
        self._model: Any = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel

            if self.plan.device == "cuda":
                from chronicler import cuda

                cuda.preload()

            log.info(
                "loading whisper %s on %s (%s)",
                self.plan.model,
                self.plan.device,
                self.plan.compute_type,
            )
            self._model = WhisperModel(
                self.plan.model, device=self.plan.device, compute_type=self.plan.compute_type
            )

    def transcribe(self, audio: Path | Any, offset: float = 0.0) -> list[Segment]:
        """Transcribe a 16 kHz mono WAV written by the recorder, or a float32 array."""
        if isinstance(audio, Path):
            # Read it ourselves: faster-whisper's own decoder is pinned to an older PyAV API.
            audio, rate = sf.read(audio, dtype="float32", always_2d=False)
            if rate != SAMPLE_RATE:
                raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {rate} Hz.")
        self.load()
        with self._lock:
            segments, _info = self._model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            return [
                Segment(round(offset + s.start, 2), round(offset + s.end, 2), s.text.strip())
                for s in segments
                if s.text.strip()
            ]

    def benchmark(self, audio: Any) -> float:
        """Real-time factor (processing seconds per audio second), VAD disabled."""
        self.load()
        start = time.perf_counter()
        with self._lock:
            segments, _ = self._model.transcribe(audio, beam_size=5, vad_filter=False)
            list(segments)  # segments are lazy; force decoding
        return (time.perf_counter() - start) / (len(audio) / SAMPLE_RATE)


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def segments_to_text(segments: list[dict[str, Any]] | list[Segment]) -> str:
    lines = []
    for s in segments:
        d = s.to_dict() if isinstance(s, Segment) else s
        lines.append(f"[{format_timestamp(d['start'])}] {d['text']}")
    return "\n".join(lines)
