from __future__ import annotations

from chronicler import doctor
from chronicler.audio import devices
from chronicler.audio.devices import DeviceInfo
from chronicler.config import Settings


def by_id(settings: Settings, notes_dir=None):
    return {c.id: c for c in doctor.build_checks(settings, notes_dir)}


def test_python_and_packages_pass(settings: Settings) -> None:
    checks = by_id(settings)
    assert doctor.run_check(checks["python"]).status == "ok"
    assert doctor.run_check(checks["packages"]).status == "ok"


def test_wsl_is_reported_as_blocking(settings: Settings, monkeypatch) -> None:
    monkeypatch.setattr(devices, "is_wsl", lambda: True)
    result = doctor.run_check(by_id(settings)["audio"])
    assert result.status == "fail" and result.blocking
    assert "Windows" in result.fix


def test_audio_without_loopback(settings: Settings, monkeypatch) -> None:
    monkeypatch.setattr(devices, "is_wsl", lambda: False)
    monkeypatch.setattr(devices, "list_devices", lambda: [DeviceInfo("m1", "USB Mic", "mic", True)])
    result = doctor.run_check(by_id(settings)["audio"])
    assert result.status == "fail"
    assert "loopback" in result.detail.lower()


def test_audio_ok_with_both_devices(settings: Settings, monkeypatch) -> None:
    monkeypatch.setattr(devices, "is_wsl", lambda: False)
    monkeypatch.setattr(
        devices,
        "list_devices",
        lambda: [
            DeviceInfo("s1", "Speakers", "loopback", True),
            DeviceInfo("m1", "USB Mic", "mic", True),
        ],
    )
    result = doctor.run_check(by_id(settings)["audio"])
    assert result.status == "ok"
    assert "Speakers" in result.detail and "USB Mic" in result.detail


def test_storage_and_notes(settings: Settings, tmp_path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "npcs.md").write_text("# Strahd\nThe vampire lord.", encoding="utf-8")
    (notes / ".obsidian").mkdir()
    (notes / ".obsidian" / "x.md").write_text("ignored", encoding="utf-8")
    checks = by_id(settings, str(notes))
    assert doctor.run_check(checks["storage"]).status == "ok"
    result = doctor.run_check(checks["notes"])
    assert result.status == "ok" and result.detail.startswith("1 file")
    missing = doctor.run_check(by_id(settings, str(tmp_path / "nope"))["notes"])
    assert missing.status == "fail"


def test_llm_check_uses_provider(settings: Settings, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("chronicler.config.get_secret", lambda name: None)
    monkeypatch.setattr("chronicler.llm.get_secret", lambda name: None)
    result = doctor.run_check(by_id(settings)["llm"])
    assert result.status == "fail" and "API key" in result.detail


def test_crashing_check_is_reported() -> None:
    def boom():
        raise RuntimeError("kaboom")

    result = doctor.run_check(doctor.Check("x", "X", boom))
    assert result.status == "fail" and "kaboom" in result.detail
