"""Shared application state for the web server and the CLI."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from chronicler.audio.monitor import LevelMonitor
from chronicler.campaign.store import Campaign, Store
from chronicler.config import Settings, save_settings
from chronicler.doctor import CheckResult
from chronicler.llm import make_provider
from chronicler.llm.base import LLMProvider
from chronicler.pipeline import Pipeline
from chronicler.transcribe import Transcriber, plan_device


@dataclass
class AppState:
    settings: Settings
    config_file: Path | None = None
    store: Store = field(init=False)
    pipeline: Pipeline = field(init=False)
    doctor_results: dict[str, CheckResult] = field(default_factory=dict)
    model_download: asyncio.Task[None] | None = None
    monitor: LevelMonitor | None = None

    def __post_init__(self) -> None:
        self.store = Store(self.settings.db_path)
        self.pipeline = Pipeline(
            store=self.store,
            transcriber_factory=self._make_transcriber,
            provider_factory=self._make_provider,
            export_root=lambda: self.settings.resolved_export_dir,
            sessions_root=lambda: self.settings.sessions_dir,
            notes_language=lambda: self.settings.notes_language,
        )

    def _make_transcriber(self) -> Transcriber:
        plan = plan_device(self.settings.whisper_model, self.settings.whisper_device)
        return Transcriber(plan, self.settings.audio_language)

    def _make_provider(self) -> LLMProvider:
        return make_provider(self.settings)

    def save(self) -> None:
        save_settings(self.settings, self.config_file)

    def active_campaign(self) -> Campaign | None:
        cid = self.settings.active_campaign_id
        campaign = self.store.get_campaign(cid) if cid is not None else None
        if campaign is None:
            campaigns = self.store.list_campaigns()
            campaign = campaigns[-1] if campaigns else None
            if campaign is not None:
                self.settings.active_campaign_id = campaign.id
        return campaign

    @property
    def blocking_failures(self) -> list[CheckResult]:
        return [r for r in self.doctor_results.values() if r.status == "fail" and r.blocking]
