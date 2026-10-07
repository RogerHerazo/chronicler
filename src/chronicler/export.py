"""Markdown / text exports of a session."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

from chronicler.campaign.store import Session, Store
from chronicler.llm.base import SessionSummary
from chronicler.transcribe import format_timestamp, segments_to_text

KIND_LABELS = {
    "npc": "NPC",
    "place": "Place",
    "item": "Item",
    "faction": "Faction",
    "other": "Other",
}


def write_atomic(path: Path, text: str) -> None:
    """Write via a temp file + rename so editors never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def session_export_dir(export_root: Path, session: Session) -> Path:
    return export_root / session.path.name


def _cell(text: str) -> str:
    return " ".join(text.replace("|", "\\|").split())


def _session_date(session: Session) -> str:
    return datetime.fromisoformat(session.started_at).astimezone().strftime("%Y-%m-%d")


def render_live_notes(store: Store, session: Session, campaign_name: str) -> str:
    chunks = store.list_chunks(session.id)
    done = [c for c in chunks if c.status == "done"]
    lines = [f"# {campaign_name} - live notes, {_session_date(session)}", ""]
    if done:
        last = done[-1]
        stamp = datetime.now().strftime("%H:%M")
        lines += [
            f"_Updated {stamp}, after part {last.idx + 1} "
            f"(up to {format_timestamp(last.end_s)} into the session)._",
            "",
        ]
    else:
        lines += ["_Waiting for the first part to be analyzed._", ""]

    lines += ["## Recap", "", session.recap.strip() or "_Nothing yet._", ""]

    threads = store.session_threads(session.id)
    open_threads = [(t, ev) for t, ev in threads if t.state == "open"]
    resolved = [(t, ev) for t, ev in threads if t.state == "resolved"]
    lines += ["## Open threads", ""]
    if open_threads:
        lines += [f"- **{t.title}**: {ev[-1][1]}" for t, ev in open_threads]
    else:
        lines.append("_None yet._")
    lines.append("")
    if resolved:
        lines += ["## Resolved this session", ""]
        lines += [f"- **{t.title}**: {ev[-1][1]}" for t, ev in resolved]
        lines.append("")

    entities = store.session_entities(session.id)
    for heading, wanted_new in (("New this session", True), ("Known", False)):
        rows = [(e, notes) for e, notes, is_new in entities if is_new == wanted_new]
        lines += [f"## {heading}", ""]
        if not rows:
            lines += ["_None yet._", ""]
            continue
        lines += ["| Name | Kind | What happened |", "| --- | --- | --- |"]
        for e, notes in rows:
            kind = KIND_LABELS.get(e.kind, e.kind)
            lines.append(f"| {_cell(e.name)} | {kind} | {_cell(' '.join(notes))} |")
        lines.append("")

    if done:
        lines += ["## Timeline", ""]
        for c in reversed(done):
            analysis = c.analysis or {}
            lines += [
                f"### Part {c.idx + 1} ({format_timestamp(c.start_s)} - "
                f"{format_timestamp(c.end_s)})",
                "",
            ]
            lines += [f"- {b}" for b in analysis.get("beats", [])] or ["_No beats._"]
            lines.append("")
    failed = [c for c in chunks if c.status == "failed"]
    if failed:
        lines += ["## Problems", ""]
        lines += [f"- Part {c.idx + 1}: {c.error}" for c in failed]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_transcript(store: Store, session: Session) -> str:
    parts = []
    for c in store.list_chunks(session.id):
        if c.transcript_json is not None:
            parts.append(segments_to_text(c.transcript))
    return "\n".join(p for p in parts if p).rstrip() + "\n"


def render_summary(summary: SessionSummary) -> str:
    lines = [f"# D&D Session Summary: {summary.title}", "", "## Summary", "", summary.summary]
    lines += ["", "---", "", "## Key Events", ""]
    for i, event in enumerate(summary.key_events, start=1):
        lines += [f"### {i}. {event.title}"]
        lines += [f"* **{b.lead}:** {b.detail}" for b in event.bullets]
        lines.append("")
    lines += ["---", "", "## Open Questions"]
    lines += [f"* **{q.lead}:** {q.detail}" for q in summary.open_questions]
    return "\n".join(lines).rstrip() + "\n"


def export_session(
    store: Store,
    session: Session,
    export_root: Path,
    campaign_name: str,
) -> Path:
    out = session_export_dir(export_root, session)
    write_atomic(out / "live_notes.md", render_live_notes(store, session, campaign_name))
    write_atomic(out / "transcript.txt", render_transcript(store, session))
    if session.summary_json:
        write_atomic(
            out / "summary.md",
            render_summary(SessionSummary.model_validate_json(session.summary_json)),
        )
    return out
