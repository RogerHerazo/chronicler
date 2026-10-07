from __future__ import annotations

from pathlib import Path

from chronicler import config
from chronicler.config import Settings, load_settings, save_settings


def test_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    s = Settings(data_dir=tmp_path / "d", chunk_minutes=10, audio_language="es", mic_enabled=False)
    save_settings(s, path)
    loaded = load_settings(path)
    assert loaded == s
    assert "api_key" not in path.read_text()


def test_missing_file_gives_defaults(tmp_path: Path) -> None:
    assert load_settings(tmp_path / "nope.toml") == Settings()


def test_home_is_expanded() -> None:
    s = Settings(export_dir="~/notes")  # type: ignore[arg-type]
    assert s.export_dir is not None and "~" not in str(s.export_dir)


def test_env_secret_wins(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    assert config.get_secret("anthropic_api_key") == "sk-env"
    assert config.secret_source("anthropic_api_key") == "env"
