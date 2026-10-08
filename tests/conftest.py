from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chronicler.audio.source import BLOCK_SECONDS, SAMPLE_RATE
from chronicler.config import Settings
from chronicler.llm.base import (
    ChunkAnalysis,
    ChunkContext,
    EntityMention,
    ProviderStatus,
    SessionContext,
    SessionSummary,
    ThreadUpdate,
)
from chronicler.transcribe import Segment


class ArraySource:
    """An AudioSource that plays a numpy array as fast as possible."""

    def __init__(self, samples: np.ndarray):
        self.samples = samples.astype(np.float32)
        self.pos = 0
        self.stopped = False

    def start(self) -> None:
        self.pos = 0

    def read(self) -> np.ndarray | None:
        if self.stopped or self.pos >= self.samples.size:
            return None
        n = int(SAMPLE_RATE * BLOCK_SECONDS)
        block = self.samples[self.pos : self.pos + n]
        self.pos += n
        return block

    def stop(self) -> None:
        self.stopped = True

    def levels(self) -> dict[str, float]:
        return {"file": -20.0}

    def errors(self) -> dict[str, str]:
        return {}

    def clipping(self) -> dict[str, bool]:
        return {}


def speechy(seconds: float, quiet_at: list[float] | None = None) -> np.ndarray:
    """Loud noise with 1-second silent gaps at the given times."""
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(int(seconds * SAMPLE_RATE)) * 0.2).astype(np.float32)
    for t in quiet_at or []:
        a = int(t * SAMPLE_RATE)
        audio[a : a + SAMPLE_RATE] = 0.0
    return audio


class FakeTranscriber:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, float]] = []

    def transcribe(self, audio: Path, offset: float = 0.0) -> list[Segment]:
        self.calls.append((audio, offset))
        n = len(self.calls)
        return [Segment(offset + 1.0, offset + 4.0, f"Line {n}: we meet Strahd at Barovia.")]


class FakeProvider:
    name = "fake"
    model = "fake-1"

    def __init__(self, fail_times: int = 0) -> None:
        self.contexts: list[ChunkContext] = []
        self.fail_times = fail_times

    def analyze_chunk(self, ctx: ChunkContext) -> ChunkAnalysis:
        from chronicler.llm.base import LLMError

        self.contexts.append(ctx)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise LLMError("temporary outage", retryable=True)
        n = len(self.contexts)
        return ChunkAnalysis(
            beats=[f"Beat {n}"],
            recap=f"Recap after part {ctx.chunk_index + 1}.",
            entities=[
                EntityMention(name="Strahd", kind="npc", known=n > 1, note=f"Strahd note {n}"),
                EntityMention(name="Barovia", kind="place", known=False, note="A gloomy village."),
            ],
            threads=[
                ThreadUpdate(
                    title="Who sent the letter?",
                    change="opened" if n == 1 else "advanced",
                    note=f"Thread note {n}",
                )
            ],
        )

    def summarize_session(self, ctx: SessionContext) -> SessionSummary:
        from chronicler.llm.base import Bullet, KeyEvent

        return SessionSummary(
            title="Into the Mists",
            summary="The party entered Barovia.",
            key_events=[
                KeyEvent(title="Arrival", bullets=[Bullet(lead="The gates", detail="Closed.")])
            ],
            open_questions=[Bullet(lead="The letter", detail="Who wrote it?")],
        )

    def check(self) -> ProviderStatus:
        return ProviderStatus(True, "fake ok")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", export_dir=tmp_path / "exports")
