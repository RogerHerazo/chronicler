"""Provider-neutral analysis types and the LLMProvider interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, Field

EntityKind = Literal["npc", "place", "item", "faction", "other"]
# (canonical name, kind, aliases)
KnownEntity = tuple[str, str, list[str]]


class EntityMention(BaseModel):
    name: str = Field(description="Canonical spelling. Reuse the known name exactly if it matches.")
    kind: EntityKind
    in_notes: bool = Field(
        description=(
            "True only if this entity appears in the campaign notes provided by the game "
            "master. False if it is new, or only came up earlier in this session."
        )
    )
    note: str = Field(description="One sentence: what this entity did or what was learned.")


class ThreadUpdate(BaseModel):
    title: str = Field(
        description="Short label for the thread. Reuse the exact title of a known open thread."
    )
    change: Literal["opened", "advanced", "resolved"]
    note: str = Field(description="One sentence on what changed.")


class ChunkAnalysis(BaseModel):
    beats: list[str] = Field(description="2-6 short bullets: what happened in this part.")
    recap: str = Field(
        description="Updated recap of the whole session so far, one to three short paragraphs."
    )
    entities: list[EntityMention]
    threads: list[ThreadUpdate]


class Bullet(BaseModel):
    lead: str = Field(description="Bold 2-5 word lead-in.")
    detail: str


class KeyEvent(BaseModel):
    title: str
    bullets: list[Bullet]


class SessionSummary(BaseModel):
    title: str = Field(description="Short evocative title for the session, never 'Session N'.")
    summary: str = Field(description="One paragraph, 3-6 sentences.")
    key_events: list[KeyEvent] = Field(description="3-5 chronological scenes.")
    open_questions: list[Bullet] = Field(description="3-5 forward-looking questions.")


@dataclass
class CampaignContext:
    """Stable per-session context. Kept identical across calls so it caches well."""

    campaign_name: str
    notes_text: str = ""
    notes_language: str = "English"


@dataclass
class ChunkContext:
    campaign: CampaignContext
    chunk_index: int
    start_s: float
    end_s: float
    transcript: str
    previous_recap: str = ""
    known_entities: list[KnownEntity] = field(default_factory=list)
    open_threads: list[str] = field(default_factory=list)


@dataclass
class SessionContext:
    campaign: CampaignContext
    transcript: str
    recap: str
    beats: list[str] = field(default_factory=list)
    known_entities: list[KnownEntity] = field(default_factory=list)


class LLMError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class ProviderStatus:
    ok: bool
    detail: str
    fix_hint: str = ""


class LLMProvider(Protocol):
    name: str
    model: str

    def analyze_chunk(self, ctx: ChunkContext) -> ChunkAnalysis: ...

    def summarize_session(self, ctx: SessionContext) -> SessionSummary: ...

    def check(self) -> ProviderStatus: ...
