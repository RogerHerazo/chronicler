# pyright: reportOptionalMemberAccess=false, reportArgumentType=false
from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from chronicler.audio import devices
from chronicler.audio.devices import DeviceInfo
from chronicler.config import Settings
from chronicler.web.app import create_app, meter_pct, minutes
from chronicler.web.state import AppState
from tests.conftest import FakeProvider, FakeTranscriber


@dataclass
class Harness:
    client: TestClient
    state: AppState


@pytest.fixture
def h(settings: Settings, tmp_path, monkeypatch):
    monkeypatch.setattr(
        devices,
        "list_devices",
        lambda: [
            DeviceInfo("s1", "Speakers", "loopback", True),
            DeviceInfo("m1", "USB Mic", "mic", True),
        ],
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    state = AppState(settings, tmp_path / "config.toml")
    state.pipeline.transcriber_factory = FakeTranscriber  # type: ignore[assignment]
    state.pipeline.provider_factory = FakeProvider
    with TestClient(create_app(state)) as c:
        yield Harness(c, state)


def test_first_run_flow(h: Harness) -> None:
    r = h.client.get("/", follow_redirects=False)
    assert r.headers["location"] == "/doctor"
    assert "Health check" in h.client.get("/doctor").text
    page = h.client.get("/live").text
    assert "Create your campaign" in page

    r = h.client.post("/campaigns", data={"name": "Curse of Strahd", "notes_dir": ""})
    assert r.status_code == 200
    assert "Ready when you are" in r.text
    assert h.state.settings.active_campaign_id is not None
    assert "Curse of Strahd" in h.client.get("/campaign").text
    assert "No sessions yet" in h.client.get("/sessions").text


def test_doctor_check_rows(h: Harness) -> None:
    r = h.client.post("/doctor/check/python")
    assert r.status_code == 200
    assert 'class="check check-ok"' in r.text
    assert r.headers["HX-Trigger"] == "doctor-updated"
    assert h.client.post("/doctor/check/nope").status_code == 404
    assert "Running checks" in h.client.get("/doctor/summary").text


def test_settings_page_and_save(h: Harness) -> None:
    page = h.client.get("/settings").text
    assert "Speakers (default)" in page and "USB Mic (default)" in page
    r = h.client.post(
        "/settings",
        data={
            "provider": "ollama",
            "claude_model": "claude-opus-5-5",
            "ollama_url": "http://localhost:11434",
            "ollama_model": "qwen3:14b",
            "notes_language": "English",
            "whisper_model": "small",
            "whisper_device": "cpu",
            "audio_language": "es",
            "chunk_minutes": "10",
            "loopback_device": "s1",
            "mic_device": "",
            "export_dir": "",
        },
    )
    assert r.status_code == 200 and "Settings saved" in r.text
    s = h.state.settings
    assert (s.provider, s.chunk_minutes, s.audio_language, s.mic_enabled) == (
        "ollama",
        10,
        "es",
        False,
    )
    assert h.state.config_file.exists()


def test_start_blocked_by_failing_checks(h: Harness) -> None:
    from chronicler.doctor import CheckResult

    h.client.post("/campaigns", data={"name": "C"})
    h.state.doctor_results["audio"] = CheckResult("audio", "Audio", "fail", "no")
    r = h.client.post("/live/start")
    assert r.status_code == 409
    assert "Recording is disabled" in h.client.get("/live").text


def test_tracker_actions(h: Harness) -> None:
    h.client.post("/campaigns", data={"name": "C"})
    store = h.state.store
    cid = h.state.settings.active_campaign_id
    a = store.add_entity(cid, "Strahd", "npc", status="confirmed")
    b = store.add_entity(cid, "Strad", "npc")
    t = store.add_thread(cid, "The letter")

    page = h.client.get("/campaign").text
    assert "unconfirmed" in page and "Strad" in page
    h.client.post(f"/entities/{b.id}", data={"action": "merge", "merge_into": a.id})
    assert store.get_entity(a.id).aliases == ["Strad"]
    h.client.post(f"/threads/{t.id}", data={"action": "resolve"})
    assert store.get_thread(t.id).state == "resolved"
    h.client.post(f"/entities/{a.id}", data={"action": "dismiss"})
    assert "Strahd" in h.client.get("/campaign?show=dismissed").text
    h.client.post("/entities/add", data={"name": "Ireena", "kind": "npc"})
    assert store.find_entity(cid, "ireena").status == "confirmed"


def test_session_pages(h: Harness) -> None:
    h.client.post("/campaigns", data={"name": "C"})
    state = h.state
    session = state.store.create_session(
        state.settings.active_campaign_id, state.settings.sessions_dir
    )
    state.store.set_session_status(session.id, "stopped")
    for tab in ("summary", "live", "transcript"):
        r = h.client.get(f"/sessions/{session.id}?tab={tab}")
        assert r.status_code == 200
    r = h.client.post(f"/sessions/{session.id}/finish")
    assert "no transcript to summarize" in r.text
    assert h.client.get("/sessions/999").status_code == 404
    assert h.client.get("/live/panel").status_code == 200
    assert h.client.get("/api/status").json() == {"recording": False}


def test_meter_pct() -> None:
    assert meter_pct(-90) == 0
    assert meter_pct(0) == 100
    assert meter_pct(-30) == 50


def test_minutes() -> None:
    assert minutes(1) == "1 minute"
    assert minutes(15.0) == "15 minutes"
    assert minutes(7.5) == "7.5 minutes"
