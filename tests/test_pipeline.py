# pyright: reportOptionalMemberAccess=false, reportArgumentType=false
from __future__ import annotations

import asyncio
from pathlib import Path

from chronicler.campaign.store import Store
from chronicler.config import Settings
from chronicler.export import session_export_dir
from chronicler.pipeline import Pipeline
from tests.conftest import ArraySource, FakeProvider, FakeTranscriber, speechy


def make_pipeline(settings: Settings, provider: FakeProvider, transcriber: FakeTranscriber):
    store = Store(settings.db_path)
    pipeline = Pipeline(
        store=store,
        transcriber_factory=lambda: transcriber,  # type: ignore[arg-type,return-value]
        provider_factory=lambda: provider,
        export_root=lambda: settings.resolved_export_dir,
        sessions_root=lambda: settings.sessions_dir,
        notes_language=lambda: "English",
    )
    return store, pipeline


async def record(pipeline: Pipeline, campaign_id: int, seconds: float, **kw):
    session = await pipeline.start_recording(campaign_id, ArraySource(speechy(seconds)), 20, **kw)
    rec = pipeline.recording
    assert rec is not None
    await asyncio.to_thread(rec.recorder.finished.wait, 10)
    await asyncio.sleep(0.05)  # let the recorder watcher run
    await pipeline.wait_idle()
    return session


async def test_full_session_flow(settings: Settings) -> None:
    provider, transcriber = FakeProvider(), FakeTranscriber()
    store, pipeline = make_pipeline(settings, provider, transcriber)
    await pipeline.start()
    events = pipeline.bus.subscribe()
    campaign = store.create_campaign("Curse of Strahd")

    session = await record(pipeline, campaign.id, 65)

    chunks = store.list_chunks(session.id)
    assert [c.status for c in chunks] == ["done"] * len(chunks)
    assert len(chunks) == 3
    assert store.get_session(session.id).status == "stopped"
    # Transcripts carry absolute session timestamps.
    assert [off for _, off in transcriber.calls] == [c.start_s for c in chunks]
    # Rolling recap from the last chunk, and context passed forward.
    assert store.get_session(session.id).recap == "Recap after part 3."
    assert provider.contexts[1].previous_recap == "Recap after part 1."
    assert ("Strahd", "npc", []) in provider.contexts[1].known_entities
    assert provider.contexts[1].open_threads == ["Who sent the letter?"]

    # Entities are deduplicated across chunks and flagged as new this session.
    entities = store.session_entities(session.id)
    assert sorted(e.name for e, _, _ in entities) == ["Barovia", "Strahd"]
    assert all(is_new for _, _, is_new in entities)
    assert all(e.status == "suggested" for e, _, _ in entities)
    [(_thread, thread_events)] = store.session_threads(session.id)
    assert [k for k, _ in thread_events] == ["opened", "advanced", "advanced"]

    out = session_export_dir(settings.resolved_export_dir, store.get_session(session.id))
    live = (out / "live_notes.md").read_text(encoding="utf-8")
    assert "Recap after part 3." in live
    assert "| Strahd | NPC |" in live
    assert "**Who sent the letter?**" in live
    transcript = (out / "transcript.txt").read_text(encoding="utf-8")
    assert transcript.count("we meet Strahd") == 3

    await pipeline.finish_session(session.id)
    assert store.get_session(session.id).status == "finished"
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert summary.startswith("# D&D Session Summary: Into the Mists")

    seen = set()
    while not events.empty():
        seen.add(events.get_nowait()["event"])
    assert {"session", "chunk", "session_updated", "summary"} <= seen
    await pipeline.shutdown()


async def test_failed_analysis_keeps_transcript_and_retries(settings: Settings) -> None:
    provider, transcriber = FakeProvider(fail_times=1), FakeTranscriber()
    store, pipeline = make_pipeline(settings, provider, transcriber)
    await pipeline.start()
    campaign = store.create_campaign("C")
    session = await record(pipeline, campaign.id, 15)

    [chunk] = store.list_chunks(session.id)
    assert chunk.status == "failed"
    assert chunk.error == "temporary outage"
    assert chunk.transcript_json is not None

    await pipeline.retry_chunk(chunk.id)
    await pipeline.wait_idle()
    assert store.get_chunk(chunk.id).status == "done"
    assert len(transcriber.calls) == 1  # transcription was not repeated
    await pipeline.shutdown()


async def test_reanalysis_does_not_duplicate_tracker_rows(settings: Settings) -> None:
    provider, transcriber = FakeProvider(), FakeTranscriber()
    store, pipeline = make_pipeline(settings, provider, transcriber)
    await pipeline.start()
    campaign = store.create_campaign("C")
    session = await record(pipeline, campaign.id, 15)
    [chunk] = store.list_chunks(session.id)
    analysis = provider.analyze_chunk(pipeline._chunk_context(store.get_session(session.id), chunk))
    pipeline.apply_analysis(store.get_session(session.id), chunk, analysis)
    [(_, events)] = store.session_threads(session.id)
    assert len(events) == 1
    assert all(len(notes) == 1 for _, notes, _ in store.session_entities(session.id))
    await pipeline.shutdown()


async def test_continue_session_and_resume_after_restart(settings: Settings) -> None:
    provider, transcriber = FakeProvider(), FakeTranscriber()
    store, pipeline = make_pipeline(settings, provider, transcriber)
    await pipeline.start()
    campaign = store.create_campaign("C")
    session = await record(pipeline, campaign.id, 25)
    first = store.list_chunks(session.id)
    await record(pipeline, campaign.id, 15, continue_session_id=session.id)
    chunks = store.list_chunks(session.id)
    assert [c.idx for c in chunks] == list(range(len(chunks)))
    assert chunks[len(first)].start_s == first[-1].end_s
    recordings = await asyncio.to_thread(lambda: list(Path(session.dir).glob("recording_*")))
    assert len(recordings) == 2
    await pipeline.shutdown()

    # Simulate a crash: a chunk left mid-flight is picked up on the next start.
    store.set_chunk_status(chunks[-1].id, "analyzing")
    store.set_session_status(session.id, "recording")
    store2, pipeline2 = make_pipeline(settings, FakeProvider(), FakeTranscriber())
    await pipeline2.start()
    await pipeline2.wait_idle()
    assert store2.get_session(session.id).status == "stopped"
    assert store2.get_chunk(chunks[-1].id).status == "done"
    await pipeline2.shutdown()


class BrokenSource(ArraySource):
    def start(self) -> None:
        from chronicler.audio.devices import AudioUnavailableError

        raise AudioUnavailableError("Could not open the microphone 'USB Mic'.")


async def test_device_failure_creates_no_session(settings: Settings) -> None:
    import pytest

    from chronicler.audio.devices import AudioUnavailableError

    store, pipeline = make_pipeline(settings, FakeProvider(), FakeTranscriber())
    await pipeline.start()
    campaign = store.create_campaign("C")
    with pytest.raises(AudioUnavailableError):
        await pipeline.start_recording(campaign.id, BrokenSource(speechy(5)), 20)
    assert store.list_sessions(campaign.id) == []
    assert pipeline.recording is None
    await pipeline.shutdown()
