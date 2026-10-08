"""Session orchestration: recording -> transcription -> analysis -> tracker + exports.

One asyncio worker processes chunks strictly in order, because each analysis
builds on the rolling recap of the chunk before it. Blocking work (Whisper,
LLM calls, the recorder thread) runs off the event loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chronicler.audio.recorder import AudioChunk, Recorder
from chronicler.audio.source import AudioSource
from chronicler.campaign.notes import NotesBundle, load_notes
from chronicler.campaign.store import Chunk, Session, Store
from chronicler.export import export_session
from chronicler.llm.base import (
    CampaignContext,
    ChunkAnalysis,
    ChunkContext,
    KnownEntity,
    LLMError,
    LLMProvider,
    SessionContext,
)
from chronicler.transcribe import Segment, Transcriber, segments_to_text

log = logging.getLogger(__name__)


class EventBus:
    """Fan-out of pipeline events to SSE subscribers."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(q)

    def publish(self, event: str, **data: Any) -> None:
        """Safe to call from any thread."""
        payload = {"event": event, **data}
        loop = self._loop
        if loop is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._deliver(payload)
        else:
            with contextlib.suppress(RuntimeError):  # loop closed during shutdown
                loop.call_soon_threadsafe(self._deliver, payload)

    def _deliver(self, payload: dict[str, Any]) -> None:
        for q in list(self._subscribers):
            with contextlib.suppress(asyncio.QueueFull):  # a stalled browser tab
                q.put_nowait(payload)


@dataclass
class ActiveRecording:
    session_id: int
    campaign_id: int
    recorder: Recorder
    source: AudioSource


@dataclass
class Pipeline:
    store: Store
    transcriber_factory: Callable[[], Transcriber]
    provider_factory: Callable[[], LLMProvider]
    export_root: Callable[[], Path]
    sessions_root: Callable[[], Path]
    notes_language: Callable[[], str]
    bus: EventBus = field(default_factory=EventBus)

    _queue: asyncio.Queue[int] = field(init=False)
    _worker: asyncio.Task[None] | None = field(init=False, default=None)
    _transcriber: Transcriber | None = field(init=False, default=None)
    _notes: dict[int, NotesBundle] = field(init=False, default_factory=dict)
    recording: ActiveRecording | None = field(init=False, default=None)
    current_chunk: tuple[int, str] | None = field(init=False, default=None)  # (chunk id, stage)
    _watcher: asyncio.Task[None] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._queue = asyncio.Queue()

    # --- lifecycle -------------------------------------------------------------

    async def start(self) -> None:
        self.bus.bind(asyncio.get_running_loop())
        for sid in self.store.mark_interrupted_sessions():
            log.warning("session %s was interrupted; marked as stopped", sid)
        for chunk in self.store.pending_chunks():
            self._queue.put_nowait(chunk.id)
        self._worker = asyncio.create_task(self._work(), name="chunk-worker")

    async def shutdown(self) -> None:
        if self.recording:
            await self.stop_recording()
        if self._worker:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker

    def transcriber(self) -> Transcriber:
        if self._transcriber is None:
            self._transcriber = self.transcriber_factory()
        return self._transcriber

    def reset_transcriber(self) -> None:
        """Called when transcription settings change."""
        self._transcriber = None

    # --- recording -------------------------------------------------------------

    async def start_recording(
        self,
        campaign_id: int,
        source: AudioSource,
        chunk_seconds: float,
        *,
        continue_session_id: int | None = None,
    ) -> Session:
        if self.recording:
            raise RuntimeError("A session is already recording.")
        if continue_session_id is not None and self.store.get_session(continue_session_id) is None:
            raise ValueError("Unknown session.")
        # Open the audio devices before touching the database, so a device that
        # cannot be opened leaves no half-started session behind.
        await asyncio.to_thread(source.start)
        if continue_session_id is not None:
            session = self.store.get_session(continue_session_id)
            assert session is not None
            chunks = self.store.list_chunks(session.id)
            first_index = chunks[-1].idx + 1 if chunks else 0
            offset = chunks[-1].end_s if chunks else 0.0
            self.store.set_session_status(session.id, "recording")
        else:
            session = self.store.create_session(campaign_id, self.sessions_root())
            first_index, offset = 0, 0.0

        loop = asyncio.get_running_loop()

        def on_chunk(chunk: AudioChunk) -> None:  # recorder thread
            loop.call_soon_threadsafe(self._enqueue_new_chunk, session.id, chunk)

        recorder = Recorder(
            source,
            session.path,
            chunk_seconds,
            on_chunk,
            first_index=first_index,
            start_offset=offset,
        )
        recorder.start()
        self.recording = ActiveRecording(session.id, campaign_id, recorder, source)
        self.bus.publish("session", session_id=session.id, status="recording")
        self._watcher = asyncio.create_task(self._watch_recorder(recorder, session.id))
        return session

    async def _watch_recorder(self, recorder: Recorder, session_id: int) -> None:
        """Notice when a source ends on its own (replay finished, device error)."""
        await asyncio.to_thread(recorder.finished.wait)
        if self.recording and self.recording.recorder is recorder:
            self.recording = None
            self.store.set_session_status(session_id, "stopped")
            self.bus.publish(
                "session", session_id=session_id, status="stopped", error=recorder.error
            )

    async def stop_recording(self) -> None:
        rec = self.recording
        if rec is None:
            return
        self.recording = None
        await asyncio.to_thread(rec.recorder.stop)
        self.store.set_session_status(rec.session_id, "stopped")
        self.bus.publish("session", session_id=rec.session_id, status="stopped")

    def _enqueue_new_chunk(self, session_id: int, chunk: AudioChunk) -> None:
        row = self.store.add_chunk(
            session_id, chunk.index, chunk.start_s, chunk.end_s, str(chunk.path)
        )
        self._queue.put_nowait(row.id)
        self.bus.publish("chunk", session_id=session_id, chunk_id=row.id, status="queued")

    async def retry_chunk(self, chunk_id: int) -> None:
        chunk = self.store.get_chunk(chunk_id)
        if chunk is None or chunk.status != "failed":
            return
        self.store.set_chunk_status(chunk_id, "queued")
        self._queue.put_nowait(chunk_id)
        self.bus.publish("chunk", session_id=chunk.session_id, chunk_id=chunk_id, status="queued")

    async def wait_idle(self) -> None:
        """Wait until every queued chunk has been processed (used by replay and tests)."""
        await self._queue.join()

    # --- worker ------------------------------------------------------------------

    async def _work(self) -> None:
        while True:
            chunk_id = await self._queue.get()
            try:
                await self._process(chunk_id)
            except Exception:
                log.exception("unexpected error processing chunk %s", chunk_id)
            finally:
                self.current_chunk = None
                self._queue.task_done()

    def _set_stage(self, chunk: Chunk, status: str, error: str | None = None) -> None:
        self.store.set_chunk_status(chunk.id, status, error)
        self.current_chunk = (chunk.id, status) if status in ("transcribing", "analyzing") else None
        self.bus.publish(
            "chunk", session_id=chunk.session_id, chunk_id=chunk.id, status=status, error=error
        )

    async def _process(self, chunk_id: int) -> None:
        chunk = self.store.get_chunk(chunk_id)
        if chunk is None or chunk.status == "done":
            return
        session = self.store.get_session(chunk.session_id)
        assert session is not None

        if chunk.transcript_json is None:
            self._set_stage(chunk, "transcribing")
            try:
                segments: list[Segment] = await asyncio.to_thread(
                    self.transcriber().transcribe, Path(chunk.audio_path), chunk.start_s
                )
            except Exception as e:
                log.exception("transcription failed for chunk %s", chunk.id)
                self._set_stage(chunk, "failed", f"Transcription failed: {e}")
                self._export(session.id)
                return
            self.store.set_chunk_transcript(chunk.id, [s.to_dict() for s in segments])
            chunk = self.store.get_chunk(chunk.id)
            assert chunk is not None
            self._export(session.id)

        self._set_stage(chunk, "analyzing")
        try:
            ctx = self._chunk_context(session, chunk)
            provider = self.provider_factory()
            analysis = await asyncio.to_thread(provider.analyze_chunk, ctx)
        except LLMError as e:
            self._set_stage(chunk, "failed", str(e))
            self._export(session.id)
            return
        except Exception as e:
            log.exception("analysis failed for chunk %s", chunk.id)
            self._set_stage(chunk, "failed", f"Analysis failed: {e}")
            self._export(session.id)
            return

        self.apply_analysis(session, chunk, analysis)
        self._set_stage(chunk, "done")
        self._export(session.id)

    # --- context + tracker ---------------------------------------------------------

    def notes_for(self, campaign_id: int, *, reload: bool = False) -> NotesBundle:
        if reload or campaign_id not in self._notes:
            campaign = self.store.get_campaign(campaign_id)
            self._notes[campaign_id] = load_notes(campaign.notes_dir if campaign else None)
        return self._notes[campaign_id]

    def campaign_context(self, campaign_id: int) -> CampaignContext:
        campaign = self.store.get_campaign(campaign_id)
        assert campaign is not None
        return CampaignContext(
            campaign_name=campaign.name,
            notes_text=self.notes_for(campaign_id).text,
            notes_language=self.notes_language(),
        )

    def _known_entities(self, campaign_id: int) -> list[KnownEntity]:
        return [
            (e.name, e.kind, e.aliases)
            for e in self.store.list_entities(campaign_id, ("confirmed", "suggested"))
        ]

    def _chunk_context(self, session: Session, chunk: Chunk) -> ChunkContext:
        return ChunkContext(
            campaign=self.campaign_context(session.campaign_id),
            chunk_index=chunk.idx,
            start_s=chunk.start_s,
            end_s=chunk.end_s,
            transcript=segments_to_text(chunk.transcript),
            previous_recap=session.recap,
            known_entities=self._known_entities(session.campaign_id),
            open_threads=[
                t.title
                for t in self.store.list_threads(
                    session.campaign_id, ("suggested", "confirmed"), ("open",)
                )
            ],
        )

    def apply_analysis(self, session: Session, chunk: Chunk, analysis: ChunkAnalysis) -> None:
        store = self.store
        store.clear_chunk_effects(chunk.id)
        store.set_chunk_analysis(chunk.id, analysis.model_dump())

        for mention in analysis.entities:
            entity = store.find_entity(session.campaign_id, mention.name)
            if entity is None:
                # Something from the GM's notes is canon that predates this
                # session: it is known and needs no confirmation.
                entity = store.add_entity(
                    session.campaign_id,
                    mention.name,
                    mention.kind,
                    notes=mention.note,
                    status="confirmed" if mention.in_notes else "suggested",
                    first_session_id=None if mention.in_notes else session.id,
                )
            store.add_mention(entity.id, session.id, chunk.id, mention.note)

        for update in analysis.threads:
            thread = store.find_thread(session.campaign_id, update.title)
            if thread is None:
                thread = store.add_thread(
                    session.campaign_id, update.title, first_session_id=session.id
                )
            store.add_thread_event(thread.id, session.id, chunk.id, update.change, update.note)
            new_state = "resolved" if update.change == "resolved" else "open"
            if thread.state != new_state:
                store.update_thread(thread.id, state=new_state)

        # A retried older chunk must not roll the recap back.
        latest_done = max(
            (c.idx for c in store.list_chunks(session.id) if c.status == "done"), default=-1
        )
        if chunk.idx >= latest_done:
            store.set_session_recap(session.id, analysis.recap)

    def _export(self, session_id: int) -> None:
        session = self.store.get_session(session_id)
        if session is None:
            return
        campaign = self.store.get_campaign(session.campaign_id)
        try:
            export_session(
                self.store, session, self.export_root(), campaign.name if campaign else "Campaign"
            )
        except OSError:
            log.exception("could not write exports for session %s", session_id)
            self.bus.publish("warning", message="Could not write the export files.")
        self.bus.publish("session_updated", session_id=session_id)

    # --- finishing -----------------------------------------------------------------

    async def finish_session(self, session_id: int) -> None:
        if self.recording and self.recording.session_id == session_id:
            await self.stop_recording()
        await self.wait_idle()
        session = self.store.get_session(session_id)
        if session is None:
            raise ValueError("Unknown session.")
        chunks = self.store.list_chunks(session_id)
        transcript = "\n".join(
            segments_to_text(c.transcript) for c in chunks if c.transcript_json is not None
        )
        if not transcript.strip():
            raise LLMError("There is no transcript to summarize yet.")
        beats = [b for c in chunks if c.analysis for b in c.analysis.get("beats", [])]
        ctx = SessionContext(
            campaign=self.campaign_context(session.campaign_id),
            transcript=transcript,
            recap=session.recap,
            beats=beats,
            known_entities=self._known_entities(session.campaign_id),
        )
        self.bus.publish("summary", session_id=session_id, status="running")
        try:
            provider = self.provider_factory()
            summary = await asyncio.to_thread(provider.summarize_session, ctx)
        except Exception as e:
            self.bus.publish("summary", session_id=session_id, status="failed", error=str(e))
            raise
        self.store.set_session_summary(session_id, summary.title, summary.model_dump_json())
        self.store.set_session_status(session_id, "finished")
        self._export(session_id)
        self.bus.publish("summary", session_id=session_id, status="done")
