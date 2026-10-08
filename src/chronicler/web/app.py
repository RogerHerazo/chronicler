"""FastAPI app: a local web UI bound to 127.0.0.1."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import subprocess
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt
from sse_starlette.sse import EventSourceResponse

from chronicler import __version__
from chronicler.audio import devices
from chronicler.audio.source import LiveSource
from chronicler.campaign.store import Session
from chronicler.config import get_secret, secret_source, set_secret
from chronicler.doctor import build_checks, run_check
from chronicler.export import render_summary, session_export_dir
from chronicler.llm.base import LLMError, SessionSummary
from chronicler.transcribe import MODEL_SIZES_MB, format_timestamp, plan_device
from chronicler.web.state import AppState

log = logging.getLogger(__name__)

HERE = Path(__file__).parent
WHISPER_MODELS = ["auto", "tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"]
CLAUDE_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5"]
LANGUAGES = [
    ("", "Auto-detect"),
    ("en", "English"),
    ("es", "Spanish"),
    ("pt", "Portuguese"),
    ("fr", "French"),
    ("de", "German"),
    ("it", "Italian"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("ja", "Japanese"),
]

_md = MarkdownIt("commonmark", {"html": False}).enable("table")


def _local_dt(value: str | None, fmt: str = "%a %d %b %Y, %H:%M") -> str:
    if not value:
        return ""
    return datetime.fromisoformat(value).astimezone().strftime(fmt)


def minutes(value: float) -> str:
    return "1 minute" if value == 1 else f"{value:g} minutes"


def meter_pct(db: float) -> int:
    """Map -60..0 dBFS to a 0..100 meter width."""
    return max(0, min(100, round((db + 60) / 60 * 100)))


def create_app(
    state: AppState,
    on_startup: Callable[[AppState], Awaitable[None]] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await state.pipeline.start()
        if on_startup:
            await on_startup(state)
        yield
        await state.pipeline.shutdown()

    app = FastAPI(title="Chronicler", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.filters["ts"] = format_timestamp
    templates.env.filters["md"] = lambda text: _md.render(text or "")
    templates.env.filters["dt"] = _local_dt
    templates.env.filters["meter_pct"] = meter_pct
    templates.env.filters["minutes"] = minutes
    templates.env.globals["version"] = __version__

    def render(request: Request, name: str, **ctx: Any) -> HTMLResponse:
        campaign = state.active_campaign()
        base = {
            "state": state,
            "settings": state.settings,
            "campaign": campaign,
            "campaigns": state.store.list_campaigns(),
            "recording": state.pipeline.recording,
            "blocking": state.blocking_failures,
        }
        return templates.TemplateResponse(request, name, {**base, **ctx})

    def redirect(url: str) -> RedirectResponse:
        return RedirectResponse(url, status_code=303)

    def get_session_or_404(sid: int) -> Session:
        session = state.store.get_session(sid)
        if session is None:
            raise HTTPException(404, "Session not found")
        return session

    # --- doctor ------------------------------------------------------------------

    def checks() -> list[Any]:
        campaign = state.active_campaign()
        return build_checks(state.settings, campaign.notes_dir if campaign else None)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> Response:
        if not state.doctor_results:
            return redirect("/doctor")
        return redirect("/live")

    @app.get("/doctor", response_class=HTMLResponse)
    async def doctor_page(request: Request) -> Response:
        state.doctor_results.clear()
        return render(request, "doctor.html", checks=checks())

    @app.post("/doctor/check/{check_id}", response_class=HTMLResponse)
    async def doctor_check(request: Request, check_id: str) -> Response:
        check = next((c for c in checks() if c.id == check_id), None)
        if check is None:
            raise HTTPException(404)
        if check.id == "speed":
            # The speed test loads a Whisper model onto the GPU; running it next to
            # the GPU check would make both slower and less reliable.
            for _ in range(600):
                if {"gpu", "model"} <= state.doctor_results.keys():
                    break
                await asyncio.sleep(0.1)
        result = await asyncio.to_thread(run_check, check)
        state.doctor_results[check.id] = result
        response = render(request, "_check_row.html", check=check, result=result)
        response.headers["HX-Trigger"] = "doctor-updated"
        return response

    @app.get("/doctor/summary", response_class=HTMLResponse)
    async def doctor_summary(request: Request) -> Response:
        return render(request, "_doctor_summary.html", total=len(checks()))

    @app.post("/doctor/download-model", response_class=HTMLResponse)
    async def download_model(request: Request) -> Response:
        plan = plan_device(state.settings.whisper_model, state.settings.whisper_device)

        def download() -> None:
            from faster_whisper.utils import download_model as fw_download

            fw_download(plan.model)

        await asyncio.to_thread(download)
        check = next(c for c in checks() if c.id == "model")
        result = await asyncio.to_thread(run_check, check)
        state.doctor_results[check.id] = result
        response = render(request, "_check_row.html", check=check, result=result)
        response.headers["HX-Trigger"] = "doctor-updated"
        return response

    # --- campaigns -------------------------------------------------------------------

    @app.post("/campaigns")
    async def create_campaign(
        name: Annotated[str, Form()], notes_dir: Annotated[str, Form()] = ""
    ) -> Response:
        if not name.strip():
            raise HTTPException(400, "A campaign needs a name.")
        campaign = state.store.create_campaign(name, notes_dir.strip() or None)
        state.settings.active_campaign_id = campaign.id
        state.save()
        return redirect("/live")

    @app.post("/campaigns/select")
    async def select_campaign(campaign_id: Annotated[int, Form()]) -> Response:
        if state.pipeline.recording:
            raise HTTPException(409, "Stop the recording before switching campaigns.")
        if state.store.get_campaign(campaign_id) is None:
            raise HTTPException(404)
        state.settings.active_campaign_id = campaign_id
        state.save()
        return redirect("/live")

    @app.post("/campaigns/{campaign_id}")
    async def update_campaign(
        campaign_id: int,
        name: Annotated[str, Form()],
        notes_dir: Annotated[str, Form()] = "",
    ) -> Response:
        if state.store.get_campaign(campaign_id) is None:
            raise HTTPException(404)
        state.store.update_campaign(campaign_id, name=name, notes_dir=notes_dir.strip() or None)
        state.pipeline.notes_for(campaign_id, reload=True)
        return redirect("/campaign")

    # --- live session ------------------------------------------------------------------

    def live_context(session: Session | None) -> dict[str, Any]:
        if session is None:
            return {"session": None}
        store = state.store
        chunks = store.list_chunks(session.id)
        entities = store.session_entities(session.id)
        threads = store.session_threads(session.id)
        transcript_tail: list[dict[str, Any]] = []
        for c in reversed(chunks):
            if c.transcript_json is not None:
                transcript_tail = c.transcript[-12:]
                break
        return {
            "session": session,
            "chunks": chunks,
            "new_entities": [(e, n) for e, n, is_new in entities if is_new],
            "known_entities": [(e, n) for e, n, is_new in entities if not is_new],
            "open_threads": [(t, ev) for t, ev in threads if t.state == "open"],
            "resolved_threads": [(t, ev) for t, ev in threads if t.state == "resolved"],
            "transcript_tail": transcript_tail,
            "current_chunk": state.pipeline.current_chunk,
            "chunk_minutes": state.settings.chunk_minutes,
        }

    def current_session() -> Session | None:
        rec = state.pipeline.recording
        if rec:
            return state.store.get_session(rec.session_id)
        campaign = state.active_campaign()
        if campaign is None:
            return None
        sessions = state.store.list_sessions(campaign.id)
        return sessions[0] if sessions else None

    @app.get("/live", response_class=HTMLResponse)
    async def live_page(request: Request) -> Response:
        if state.active_campaign() is None:
            return render(request, "setup.html")
        return render(request, "live.html", **live_context(current_session()))

    @app.get("/live/panel", response_class=HTMLResponse)
    async def live_panel(request: Request) -> Response:
        return render(request, "_live_panel.html", **live_context(current_session()))

    @app.get("/live/controls", response_class=HTMLResponse)
    async def live_controls(request: Request) -> Response:
        return render(request, "_live_controls.html", session=current_session())

    @app.post("/live/start")
    async def live_start(continue_session_id: Annotated[int | None, Form()] = None) -> Response:
        campaign = state.active_campaign()
        if campaign is None:
            raise HTTPException(400, "Create a campaign first.")
        if state.blocking_failures:
            raise HTTPException(409, "Fix the failing health checks first.")
        s = state.settings
        try:
            source = await asyncio.to_thread(
                LiveSource, s.loopback_device, s.mic_device, s.mic_enabled
            )
        except Exception as e:
            raise HTTPException(500, f"Could not open the audio devices: {e}") from e
        try:
            await state.pipeline.start_recording(
                campaign.id,
                source,
                s.chunk_minutes * 60,
                continue_session_id=continue_session_id,
            )
        except devices.AudioUnavailableError as e:
            raise HTTPException(409, f"{e} Check the devices in Settings.") from e
        return redirect("/live")

    @app.post("/live/stop")
    async def live_stop() -> Response:
        await state.pipeline.stop_recording()
        return redirect("/live")

    @app.post("/chunks/{chunk_id}/retry")
    async def retry_chunk(chunk_id: int) -> Response:
        await state.pipeline.retry_chunk(chunk_id)
        return Response(status_code=204)

    @app.get("/api/status")
    async def api_status() -> JSONResponse:
        rec = state.pipeline.recording
        if rec is None:
            return JSONResponse({"recording": False})
        chunk_seconds = state.settings.chunk_minutes * 60
        return JSONResponse(
            {
                "recording": True,
                "session_id": rec.session_id,
                "elapsed": rec.recorder.elapsed_seconds,
                "buffered": rec.recorder.buffered_seconds,
                "chunk_seconds": chunk_seconds,
                "levels": rec.source.levels(),
                "errors": rec.source.errors(),
            }
        )

    @app.get("/events")
    async def events(request: Request) -> EventSourceResponse:
        queue = state.pipeline.bus.subscribe()

        async def stream() -> AsyncIterator[dict[str, str]]:
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15)
                    except TimeoutError:
                        continue
                    yield {"event": event["event"], "data": json.dumps(event)}
            finally:
                state.pipeline.bus.unsubscribe(queue)

        return EventSourceResponse(stream(), ping=15)

    # --- campaign tracker --------------------------------------------------------------

    @app.get("/campaign", response_class=HTMLResponse)
    async def campaign_page(request: Request, show: str = "active") -> Response:
        campaign = state.active_campaign()
        if campaign is None:
            return redirect("/live")
        statuses = ("dismissed",) if show == "dismissed" else ("suggested", "confirmed")
        entities = state.store.list_entities(campaign.id, statuses)
        threads = state.store.list_threads(campaign.id, statuses)
        notes = state.pipeline.notes_for(campaign.id)
        return render(
            request,
            "campaign.html",
            entities=entities,
            threads=threads,
            show=show,
            notes=notes,
            suggested_count=sum(e.status == "suggested" for e in entities)
            + sum(t.status == "suggested" for t in threads),
        )

    @app.post("/entities/add")
    async def add_entity(
        name: Annotated[str, Form()], kind: Annotated[str, Form()] = "npc"
    ) -> Response:
        campaign = state.active_campaign()
        if campaign is None or not name.strip():
            raise HTTPException(400)
        if state.store.find_entity(campaign.id, name) is None:
            state.store.add_entity(campaign.id, name, kind, status="confirmed")
        return redirect("/campaign")

    @app.post("/entities/{entity_id}")
    async def entity_action(
        entity_id: int,
        action: Annotated[str, Form()],
        name: Annotated[str, Form()] = "",
        kind: Annotated[str, Form()] = "",
        notes: Annotated[str, Form()] = "",
        merge_into: Annotated[int | None, Form()] = None,
    ) -> Response:
        store = state.store
        if store.get_entity(entity_id) is None:
            raise HTTPException(404)
        if action == "confirm":
            store.update_entity(entity_id, status="confirmed")
        elif action == "dismiss":
            store.update_entity(entity_id, status="dismissed")
        elif action == "restore":
            store.update_entity(entity_id, status="suggested")
        elif action == "save":
            store.update_entity(
                entity_id, name=name or None, kind=kind or None, notes=notes, status="confirmed"
            )
        elif action == "merge" and merge_into:
            store.merge_entities(entity_id, merge_into)
        else:
            raise HTTPException(400, "Unknown action")
        return redirect("/campaign")

    @app.post("/threads/{thread_id}")
    async def thread_action(
        thread_id: int, action: Annotated[str, Form()], title: Annotated[str, Form()] = ""
    ) -> Response:
        store = state.store
        if store.get_thread(thread_id) is None:
            raise HTTPException(404)
        updates: dict[str, dict[str, str]] = {
            "confirm": {"status": "confirmed"},
            "dismiss": {"status": "dismissed"},
            "restore": {"status": "suggested"},
            "resolve": {"state": "resolved"},
            "reopen": {"state": "open"},
        }
        if action == "save" and title.strip():
            store.update_thread(thread_id, title=title, status="confirmed")
        elif action in updates:
            store.update_thread(thread_id, **updates[action])
        else:
            raise HTTPException(400, "Unknown action")
        return redirect("/campaign#threads")

    # --- sessions ------------------------------------------------------------------------

    @app.get("/sessions", response_class=HTMLResponse)
    async def sessions_page(request: Request) -> Response:
        campaign = state.active_campaign()
        if campaign is None:
            return redirect("/live")
        sessions = state.store.list_sessions(campaign.id)
        counts = {s.id: len(state.store.list_chunks(s.id)) for s in sessions}
        return render(request, "sessions.html", sessions=sessions, counts=counts)

    @app.get("/sessions/{sid}", response_class=HTMLResponse)
    async def session_page(request: Request, sid: int, tab: str = "summary") -> Response:
        session = get_session_or_404(sid)
        summary_md = ""
        if session.summary_json:
            summary_md = render_summary(SessionSummary.model_validate_json(session.summary_json))
        export_dir = session_export_dir(state.settings.resolved_export_dir, session)
        live_notes = export_dir / "live_notes.md"
        live_notes_md = await asyncio.to_thread(
            lambda: live_notes.read_text(encoding="utf-8") if live_notes.exists() else ""
        )
        return render(
            request,
            "session.html",
            session=session,
            tab=tab,
            summary_md=summary_md,
            live_notes_md=live_notes_md,
            chunks=state.store.list_chunks(sid),
            export_dir=export_dir,
        )

    @app.post("/sessions/{sid}/finish")
    async def finish_session(request: Request, sid: int) -> Response:
        get_session_or_404(sid)
        try:
            await state.pipeline.finish_session(sid)
        except LLMError as e:
            return render(request, "_error.html", message=str(e))
        response = Response(status_code=204)
        response.headers["HX-Redirect"] = f"/sessions/{sid}"
        return response

    @app.post("/sessions/{sid}/open-folder")
    async def open_folder(sid: int) -> Response:
        session = get_session_or_404(sid)
        folder = session_export_dir(state.settings.resolved_export_dir, session)
        folder.mkdir(parents=True, exist_ok=True)
        _open_in_file_manager(folder)
        return Response(status_code=204)

    # --- settings ------------------------------------------------------------------------

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request, saved: bool = False) -> Response:
        try:
            device_list = await asyncio.to_thread(devices.list_devices)
            device_error = ""
        except devices.AudioUnavailableError as e:
            device_list, device_error = [], str(e)
        return render(
            request,
            "settings.html",
            saved=saved,
            loopbacks=[d for d in device_list if d.kind == "loopback"],
            mics=[d for d in device_list if d.kind == "mic"],
            device_error=device_error,
            whisper_models=WHISPER_MODELS,
            model_sizes=MODEL_SIZES_MB,
            claude_models=CLAUDE_MODELS,
            languages=LANGUAGES,
            key_source=secret_source("anthropic_api_key"),
            plan=plan_device(state.settings.whisper_model, state.settings.whisper_device),
        )

    @app.post("/settings")
    async def save_settings_route(
        request: Request,
        provider: Annotated[str, Form()],
        claude_model: Annotated[str, Form()],
        ollama_url: Annotated[str, Form()],
        ollama_model: Annotated[str, Form()],
        notes_language: Annotated[str, Form()],
        whisper_model: Annotated[str, Form()],
        whisper_device: Annotated[str, Form()],
        audio_language: Annotated[str, Form()] = "",
        chunk_minutes: Annotated[float, Form()] = 15,
        loopback_device: Annotated[str, Form()] = "",
        mic_device: Annotated[str, Form()] = "",
        mic_enabled: Annotated[bool, Form()] = False,
        export_dir: Annotated[str, Form()] = "",
        anthropic_api_key: Annotated[str, Form()] = "",
    ) -> Response:
        s = state.settings
        before = (s.whisper_model, s.whisper_device, s.audio_language)
        state.settings = type(s).model_validate(
            {
                **s.model_dump(),
                "provider": provider,
                "claude_model": claude_model.strip() or s.claude_model,
                "ollama_url": ollama_url.strip() or s.ollama_url,
                "ollama_model": ollama_model.strip() or s.ollama_model,
                "notes_language": notes_language.strip() or "English",
                "whisper_model": whisper_model,
                "whisper_device": whisper_device,
                "audio_language": audio_language or None,
                "chunk_minutes": min(max(chunk_minutes, 1.0), 60.0),
                "loopback_device": loopback_device or None,
                "mic_device": mic_device or None,
                "mic_enabled": mic_enabled,
                "export_dir": export_dir.strip() or None,
            }
        )
        state.save()
        if anthropic_api_key.strip():
            try:
                set_secret("anthropic_api_key", anthropic_api_key.strip())
            except Exception as e:
                return render(
                    request,
                    "_error.html",
                    message=f"Settings saved, but the API key could not be stored in the OS "
                    f"keychain ({e}). Set the ANTHROPIC_API_KEY environment variable instead.",
                )
        new = state.settings
        if before != (new.whisper_model, new.whisper_device, new.audio_language):
            state.pipeline.reset_transcriber()
        # Health results depend on settings; re-run them on the next doctor visit.
        state.doctor_results.clear()
        return redirect("/settings?saved=1")

    @app.post("/settings/test-audio", response_class=HTMLResponse)
    async def test_audio(request: Request) -> Response:
        if state.pipeline.recording:
            return render(request, "_audio_test.html", error="Stop the recording first.")
        s = state.settings
        try:
            peaks = await asyncio.to_thread(
                _measure_levels, s.loopback_device, s.mic_device, s.mic_enabled
            )
        except Exception as e:
            return render(request, "_audio_test.html", error=str(e))
        return render(request, "_audio_test.html", peaks=peaks)

    @app.get("/api/has-key")
    async def has_key() -> JSONResponse:
        return JSONResponse({"has_key": bool(get_secret("anthropic_api_key"))})

    return app


def _measure_levels(
    loopback: str | None, mic: str | None, mic_enabled: bool, seconds: float = 3.0
) -> dict[str, float]:
    source = LiveSource(loopback, mic, mic_enabled)
    source.start()
    peaks: dict[str, float] = {}
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            source.read()
            for label, level in source.levels().items():
                peaks[label] = max(peaks.get(label, -90.0), level)
        errors = source.errors()
        if errors:
            raise RuntimeError("; ".join(f"{k}: {v}" for k, v in errors.items()))
    finally:
        source.stop()
    return peaks


def _open_in_file_manager(path: Path) -> None:
    system = platform.system()
    if system == "Windows":
        os.startfile(path)  # type: ignore[attr-defined]
    elif system == "Darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
