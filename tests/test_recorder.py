from __future__ import annotations

import itertools

import numpy as np
import soundfile as sf

from chronicler.audio.recorder import AudioChunk, Recorder, choose_cut
from chronicler.audio.source import SAMPLE_RATE, dbfs
from tests.conftest import ArraySource, speechy

SR = SAMPLE_RATE


def test_choose_cut_prefers_silence_near_target() -> None:
    audio = speechy(120, quiet_at=[52.0, 66.0])
    cut = choose_cut(audio, target=60 * SR, window=10 * SR)
    # Two silent gaps are within the window; both are quiet, pick the closer one.
    assert 66 * SR <= cut <= 67 * SR or 52 * SR <= cut <= 53 * SR
    assert abs(cut - 60 * SR) <= 10 * SR


def test_choose_cut_falls_back_to_target_region_when_no_silence() -> None:
    audio = speechy(120)
    cut = choose_cut(audio, target=60 * SR, window=10 * SR)
    assert 50 * SR <= cut <= 70 * SR


def test_choose_cut_handles_short_buffers() -> None:
    audio = speechy(1)
    assert choose_cut(audio, target=10 * SR, window=SR) == audio.size


def test_recorder_emits_contiguous_chunks(tmp_path) -> None:
    audio = speechy(100, quiet_at=[29.0, 61.0])
    chunks: list[AudioChunk] = []
    rec = Recorder(ArraySource(audio), tmp_path, chunk_seconds=30, on_chunk=chunks.append)
    rec.start()
    rec.finished.wait(10)

    assert len(chunks) >= 3
    assert chunks[0].start_s == 0
    for a, b in itertools.pairwise(chunks):
        assert abs(a.end_s - b.start_s) < 1e-6
    assert abs(chunks[-1].end_s - 100) < 0.2
    # Cut points land inside the silent gaps.
    assert 29 <= chunks[0].end_s <= 30
    # Chunk files contain exactly the audio between their boundaries.
    total = sum(sf.info(c.path).frames for c in chunks)
    assert total == audio.size
    # The full-session FLAC holds everything.
    flac = tmp_path / "recording_01.flac"
    assert sf.info(flac).frames == audio.size


def test_recorder_continues_numbering_and_offsets(tmp_path) -> None:
    chunks: list[AudioChunk] = []
    rec = Recorder(
        ArraySource(speechy(12)),
        tmp_path,
        chunk_seconds=60,
        on_chunk=chunks.append,
        first_index=4,
        start_offset=600.0,
    )
    rec.start()
    rec.finished.wait(10)
    assert [c.index for c in chunks] == [4]
    assert chunks[0].start_s == 600.0
    assert chunks[0].path.name == "chunk_004.wav"
    assert (tmp_path / "recording_01.flac").exists()


def test_recorder_drops_tiny_tail(tmp_path) -> None:
    chunks: list[AudioChunk] = []
    rec = Recorder(ArraySource(speechy(1.0)), tmp_path, chunk_seconds=60, on_chunk=chunks.append)
    rec.start()
    rec.finished.wait(10)
    assert chunks == []


def test_dbfs() -> None:
    assert dbfs(np.zeros(100, dtype=np.float32)) == -90.0
    assert abs(dbfs(np.ones(100, dtype=np.float32)) - 0.0) < 1e-6
