"""Audio sources that yield 16 kHz mono float32 blocks.

`LiveSource` mixes a loopback device (what the computer plays) with a mic.
`FileReplaySource` streams an existing recording, optionally in real time, so
the whole pipeline can be exercised without a sound card.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import soxr

from chronicler.audio import devices

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
BLOCK_SECONDS = 0.1
SILENCE_DBFS = -90.0


def dbfs(block: np.ndarray) -> float:
    if block.size == 0:
        return SILENCE_DBFS
    rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
    return max(SILENCE_DBFS, 20.0 * np.log10(rms)) if rms > 0 else SILENCE_DBFS


class AudioSource(Protocol):
    def start(self) -> None: ...

    def read(self) -> np.ndarray | None:
        """Block until the next block of samples is ready. None means end of stream."""
        ...

    def stop(self) -> None: ...

    def levels(self) -> dict[str, float]: ...

    def errors(self) -> dict[str, str]: ...


class _DeviceStream:
    """Captures one device on a background thread into a 16 kHz mono buffer."""

    MAX_BACKLOG_SECONDS = 1.0
    OPEN_TIMEOUT_SECONDS = 5.0

    def __init__(self, label: str, kind: Literal["loopback", "mic"], device_id: str | None):
        self.label = label
        self.kind: Literal["loopback", "mic"] = kind
        self.device_id = device_id
        self.level = SILENCE_DBFS
        self.error: str | None = None
        self.backend: str | None = None
        self._opened = threading.Event()
        self._buf: deque[np.ndarray] = deque()
        self._buffered = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"capture-{label}", daemon=True)

    def start(self) -> None:
        """Start capturing. Raises if the device cannot be opened."""
        self._thread.start()
        if not self._opened.wait(self.OPEN_TIMEOUT_SECONDS):
            raise devices.AudioUnavailableError(f"Timed out opening the {self.label} device.")
        if self.error:
            raise devices.AudioUnavailableError(self.error)

    def stop(self) -> None:
        self._stop.set()
        # A WASAPI loopback stream can block while nothing is playing, so don't
        # wait forever; the thread is a daemon.
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        # Open on this thread: WASAPI objects belong to the thread that made them.
        try:
            capture = devices.open_capture(self.kind, self.device_id, blocksize=4800)
        except Exception as e:
            log.warning("could not open %s: %s", self.label, e)
            self.error = str(e) or type(e).__name__
            self._opened.set()
            return
        self.backend = capture.backend
        self._opened.set()
        resampler = soxr.ResampleStream(capture.samplerate, SAMPLE_RATE, 1, dtype="float32")
        frames = int(capture.samplerate * BLOCK_SECONDS)
        try:
            while not self._stop.is_set():
                data = capture.read(frames)
                mono = data.mean(axis=1) if data.ndim == 2 else data
                out = resampler.resample_chunk(np.ascontiguousarray(mono, dtype=np.float32))
                self.level = dbfs(out)
                self._push(out)
        except Exception as e:
            log.exception("capture failed on %s", self.label)
            self.error = str(e) or type(e).__name__
        finally:
            try:
                capture.close()
            except Exception:
                log.debug("error closing %s", self.label, exc_info=True)

    def _push(self, samples: np.ndarray) -> None:
        with self._lock:
            self._buf.append(samples)
            self._buffered += samples.size
            # Device clocks drift against the wall clock; drop the oldest audio
            # rather than letting latency grow without bound.
            limit = int(SAMPLE_RATE * self.MAX_BACKLOG_SECONDS)
            while self._buffered > limit and self._buf:
                dropped = self._buf.popleft()
                self._buffered -= dropped.size

    def take(self, n: int) -> np.ndarray:
        """Return exactly n samples, zero-padded if the device delivered fewer."""
        out = np.zeros(n, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < n and self._buf:
                head = self._buf[0]
                need = n - filled
                if head.size <= need:
                    out[filled : filled + head.size] = head
                    filled += head.size
                    self._buf.popleft()
                else:
                    out[filled:n] = head[:need]
                    self._buf[0] = head[need:]
                    filled = n
            self._buffered -= filled
        return out


class LiveSource:
    """Mixes loopback + mic, paced by the wall clock.

    Pacing by the clock (rather than by whichever device delivers first) keeps
    the timeline correct when a device goes quiet. WASAPI loopback, for example,
    delivers nothing at all while no sound is playing.
    """

    def __init__(self, loopback_id: str | None, mic_id: str | None, mic_enabled: bool = True):
        self._streams: list[_DeviceStream] = [_DeviceStream("system", "loopback", loopback_id)]
        if mic_enabled:
            self._streams.append(_DeviceStream("mic", "mic", mic_id))
        self._t0 = 0.0
        self._emitted = 0
        self._stopped = threading.Event()

    def start(self) -> None:
        """Open every device. If one fails, close the others and raise."""
        started: list[_DeviceStream] = []
        try:
            for s in self._streams:
                s.start()
                started.append(s)
        except Exception:
            for s in started:
                s.stop()
            raise
        self._t0 = time.monotonic()
        self._emitted = 0

    def read(self) -> np.ndarray | None:
        if self._stopped.is_set():
            return None
        target_time = self._t0 + (self._emitted + SAMPLE_RATE * BLOCK_SECONDS) / SAMPLE_RATE
        delay = target_time - time.monotonic()
        if delay > 0 and self._stopped.wait(delay):
            return None
        due = int((time.monotonic() - self._t0) * SAMPLE_RATE) - self._emitted
        if due <= 0:
            return np.zeros(0, dtype=np.float32)
        mixed = np.zeros(due, dtype=np.float32)
        for s in self._streams:
            mixed += s.take(due)
        np.clip(mixed, -1.0, 1.0, out=mixed)
        self._emitted += due
        return mixed

    def stop(self) -> None:
        self._stopped.set()
        for s in self._streams:
            s.stop()

    def levels(self) -> dict[str, float]:
        return {s.label: s.level for s in self._streams}

    def errors(self) -> dict[str, str]:
        return {s.label: s.error for s in self._streams if s.error}

    def backends(self) -> dict[str, str | None]:
        return {s.label: s.backend for s in self._streams}


class FileReplaySource:
    """Streams an audio file as if it were being recorded live.

    `speed` is a playback multiplier: 1 = real time, 10 = ten times faster,
    0 = as fast as decoding allows.
    """

    def __init__(self, path: str | Path, speed: float = 1.0, start_at: float = 0.0):
        self.path = Path(path)
        self.speed = speed
        self.start_at = start_at
        self._level = SILENCE_DBFS
        self._stopped = threading.Event()
        self._blocks: Any = None
        self._t0 = 0.0
        self._emitted = 0
        self._pending = np.zeros(0, dtype=np.float32)

    def start(self) -> None:
        self._blocks = self._decode()
        self._t0 = time.monotonic()
        self._emitted = 0

    def _decode(self) -> Any:
        import av  # bundled with faster-whisper

        skip = int(self.start_at * SAMPLE_RATE)
        with av.open(str(self.path)) as container:
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    samples = out.to_ndarray().reshape(-1).astype(np.float32, copy=False)
                    if skip:
                        cut = min(skip, samples.size)
                        samples = samples[cut:]
                        skip -= cut
                    if samples.size:
                        yield samples
            for out in resampler.resample(None):
                yield out.to_ndarray().reshape(-1).astype(np.float32, copy=False)

    def read(self) -> np.ndarray | None:
        if self._stopped.is_set() or self._blocks is None:
            return None
        n = int(SAMPLE_RATE * BLOCK_SECONDS)
        while self._pending.size < n:
            nxt = next(self._blocks, None)
            if nxt is None:
                break
            self._pending = np.concatenate([self._pending, nxt])
        if self._pending.size == 0:
            return None
        block, self._pending = self._pending[:n], self._pending[n:]
        if self.speed > 0:
            due = self._t0 + (self._emitted + block.size) / SAMPLE_RATE / self.speed
            delay = due - time.monotonic()
            if delay > 0 and self._stopped.wait(delay):
                return None
        self._emitted += block.size
        self._level = dbfs(block)
        return block

    def stop(self) -> None:
        self._stopped.set()

    def levels(self) -> dict[str, float]:
        return {"file": self._level}

    def errors(self) -> dict[str, str]:
        return {}
