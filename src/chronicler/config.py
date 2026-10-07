"""User settings, stored as TOML in the platform config directory.

Secrets (API keys) never go into the TOML file. They live in the OS keychain
via `keyring`, with environment variables taking precedence.
"""

from __future__ import annotations

import contextlib
import os
import tomllib
from pathlib import Path
from typing import Literal

import tomli_w
from platformdirs import user_config_dir, user_data_dir
from pydantic import BaseModel, Field, field_validator

APP_NAME = "chronicler"
KEYRING_SERVICE = "chronicler"

Provider = Literal["claude", "ollama"]


def config_path() -> Path:
    override = os.environ.get("CHRONICLER_CONFIG")
    if override:
        return Path(override)
    return Path(user_config_dir(APP_NAME, appauthor=False)) / "config.toml"


def default_data_dir() -> Path:
    return Path(user_data_dir(APP_NAME, appauthor=False))


class Settings(BaseModel):
    # Where the database, session audio and chunk WAVs live.
    data_dir: Path = Field(default_factory=default_data_dir)
    # Where Markdown/text exports go. Empty means `<data_dir>/exports`.
    export_dir: Path | None = None

    active_campaign_id: int | None = None

    # Audio
    chunk_minutes: float = 15.0
    loopback_device: str | None = None  # device id; None = default speaker
    mic_device: str | None = None  # device id; None = default microphone
    mic_enabled: bool = True

    # Transcription
    whisper_model: str = "auto"
    whisper_device: Literal["auto", "cuda", "cpu"] = "auto"
    audio_language: str | None = None  # ISO code like "es"; None = auto-detect

    # Analysis
    provider: Provider = "claude"
    claude_model: str = "claude-opus-5-5"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:14b"
    notes_language: str = "English"

    @field_validator("data_dir", "export_dir")
    @classmethod
    def _expand_user(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value else value

    @property
    def resolved_export_dir(self) -> Path:
        return self.export_dir or (self.data_dir / "exports")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "chronicler.db"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"


def load_settings(path: Path | None = None) -> Settings:
    path = path or config_path()
    if not path.exists():
        return Settings()
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return Settings.model_validate(raw)


def save_settings(settings: Settings, path: Path | None = None) -> None:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = settings.model_dump(mode="json", exclude_none=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(tomli_w.dumps(data), encoding="utf-8")
    tmp.replace(path)


# --- Secrets -----------------------------------------------------------------

SECRET_ENV = {"anthropic_api_key": "ANTHROPIC_API_KEY"}


def get_secret(name: str) -> str | None:
    env = SECRET_ENV.get(name)
    if env and os.environ.get(env):
        return os.environ[env]
    try:
        import keyring

        return keyring.get_password(KEYRING_SERVICE, name)
    except Exception:
        return None


def set_secret(name: str, value: str) -> None:
    """Store a secret in the OS keychain. Raises if no keychain is available."""
    import keyring
    from keyring.errors import PasswordDeleteError

    if value:
        keyring.set_password(KEYRING_SERVICE, name, value)
    else:
        with contextlib.suppress(PasswordDeleteError):
            keyring.delete_password(KEYRING_SERVICE, name)


def secret_source(name: str) -> Literal["env", "keychain", "missing"]:
    env = SECRET_ENV.get(name)
    if env and os.environ.get(env):
        return "env"
    try:
        import keyring

        if keyring.get_password(KEYRING_SERVICE, name):
            return "keychain"
    except Exception:
        pass
    return "missing"
