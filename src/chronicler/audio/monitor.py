"""Live input-level monitoring for the Settings page, outside of a session."""

from __future__ import annotations

import threading
import time

from chronicler.audio.source import LiveSource

MAX_SECONDS = 60.0


class LevelMonitor:
    """Opens the inputs and keeps reading so their meters stay live.

    Stops by itself after MAX_SECONDS so a forgotten browser tab never keeps
    the microphone open.
    """

    def __init__(self, source: LiveSource):
        self.source = source
        self.started_at = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="level-monitor", daemon=True)

    def start(self) -> None:
        self.source.start()  # raises if a device cannot be opened
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set() and self.remaining > 0:
                if self.source.read() is None:
                    break
        finally:
            self.source.stop()

    @property
    def remaining(self) -> float:
        return max(0.0, MAX_SECONDS - (time.monotonic() - self.started_at))

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def snapshot(self) -> dict[str, object]:
        return {
            "running": self.running,
            "remaining": round(self.remaining),
            "levels": self.source.levels(),
            "clipping": self.source.clipping(),
            "errors": self.source.errors(),
        }
