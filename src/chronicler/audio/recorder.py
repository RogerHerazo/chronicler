"""Turns a continuous audio source into chunk WAV files plus a full-session FLAC.

Chunks are cut at the quietest moment near the target length so a word is
never split between two transcriptions.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from chronicler.audio.source import SAMPLE_RATE, AudioSource

log = logging.getLogger(__name__)

CUT_WINDOW_SECONDS = 20.0
CUT_FRAME_SECONDS = 0.25
MIN_TAIL_SECONDS = 2.0


@dataclass(frozen=True)
class AudioChunk:
    index: int
    start_s: float  # offset from the start of the session
    end_s: float
    path: Path


def choose_cut(
    samples: np.ndarray,
    target: int,
    window: int,
    frame: int = int(SAMPLE_RATE * CUT_FRAME_SECONDS),
) -> int:
    """Index near `target` (within ±window) at the centre of the quietest frame."""
    lo = max(frame, target - window)
    hi = min(samples.size - frame, target + window)
    if hi <= lo:
        return min(target, samples.size)
    region = samples[lo:hi]
    n_frames = region.size // frame
    if n_frames == 0:
        return target
    energy = np.square(region[: n_frames * frame].reshape(n_frames, frame), dtype=np.float64)
    rms = energy.mean(axis=1)
    # Prefer the frame closest to the target among equally quiet ones.
    centres = lo + np.arange(n_frames) * frame + frame // 2
    best = np.lexsort((np.abs(centres - target), np.round(rms, 12)))[0]
    return int(centres[best])


class Recorder:
    def __init__(
        self,
        source: AudioSource,
        session_dir: Path,
        chunk_seconds: float,
        on_chunk: Callable[[AudioChunk], None],
        *,
        first_index: int = 0,
        start_offset: float = 0.0,
    ):
        self.source = source
        self.session_dir = session_dir
        self.chunk_samples = max(int(chunk_seconds * SAMPLE_RATE), SAMPLE_RATE)
        self.window = min(int(CUT_WINDOW_SECONDS * SAMPLE_RATE), self.chunk_samples // 4)
        self.on_chunk = on_chunk
        self.next_index = first_index
        # Session offset of the first buffered sample, and of the newest one.
        self.chunk_start_samples = int(start_offset * SAMPLE_RATE)
        self.total_samples = self.chunk_start_samples
        self.error: str | None = None
        self.finished = threading.Event()
        self._blocks: list[np.ndarray] = []
        self._buffered = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)

    @property
    def elapsed_seconds(self) -> float:
        return self.total_samples / SAMPLE_RATE

    @property
    def buffered_seconds(self) -> float:
        return self._buffered / SAMPLE_RATE

    def start(self) -> None:
        """Start consuming. The caller starts the source first, so device errors
        surface before anything is written."""
        (self.session_dir / "chunks").mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop capturing and flush the last partial chunk. Blocks until flushed."""
        self._stop.set()
        self.source.stop()
        self._thread.join()

    def _audio_path(self) -> Path:
        n = 1
        while (path := self.session_dir / f"recording_{n:02d}.flac").exists():
            n += 1
        return path

    def _run(self) -> None:
        try:
            with sf.SoundFile(
                self._audio_path(),
                "w",
                samplerate=SAMPLE_RATE,
                channels=1,
                format="FLAC",
                subtype="PCM_16",
            ) as full:
                while not self._stop.is_set():
                    block = self.source.read()
                    if block is None:
                        break
                    if block.size == 0:
                        continue
                    full.write(block)
                    self._blocks.append(block)
                    self._buffered += block.size
                    self.total_samples += block.size
                    if self._buffered >= self.chunk_samples + self.window:
                        self._emit(final=False)
                self._emit(final=True)
        except Exception as e:
            log.exception("recorder failed")
            self.error = str(e) or type(e).__name__
            try:
                self._emit(final=True)
            except Exception:
                log.exception("could not flush the last chunk")
        finally:
            self.finished.set()

    def _emit(self, *, final: bool) -> None:
        if not self._blocks:
            return
        samples = np.concatenate(self._blocks)
        if final:
            if samples.size < MIN_TAIL_SECONDS * SAMPLE_RATE:
                self._blocks, self._buffered = [], 0
                self.chunk_start_samples += samples.size
                return
            cut = samples.size
        else:
            cut = choose_cut(samples, self.chunk_samples, self.window)
        head, tail = samples[:cut], samples[cut:]
        start = self.chunk_start_samples / SAMPLE_RATE
        path = self.session_dir / "chunks" / f"chunk_{self.next_index:03d}.wav"
        sf.write(path, head, SAMPLE_RATE, subtype="PCM_16")
        chunk = AudioChunk(self.next_index, start, start + head.size / SAMPLE_RATE, path)
        self.next_index += 1
        self.chunk_start_samples += head.size
        self._blocks = [tail] if tail.size else []
        self._buffered = tail.size
        self.on_chunk(chunk)
