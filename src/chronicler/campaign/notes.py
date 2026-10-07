"""Load an optional folder of campaign notes (Markdown / text) as LLM context."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

NOTE_SUFFIXES = {".md", ".markdown", ".txt"}
# Roughly 60k tokens. Large enough for a solid world bible, small enough to keep
# per-chunk latency and cost reasonable (the block is prompt-cached anyway).
DEFAULT_TOKEN_BUDGET = 60_000


def estimate_tokens(text: str) -> int:
    # ~4 characters per token is a good average for English and Spanish prose.
    return (len(text) + 3) // 4


@dataclass
class NotesBundle:
    root: Path | None
    text: str = ""
    files: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # files dropped to fit the budget
    tokens: int = 0
    error: str | None = None

    @property
    def truncated(self) -> bool:
        return bool(self.skipped)


def load_notes(root: str | Path | None, budget: int = DEFAULT_TOKEN_BUDGET) -> NotesBundle:
    if not root:
        return NotesBundle(root=None)
    path = Path(root).expanduser()
    if not path.is_dir():
        return NotesBundle(root=path, error=f"Folder not found: {path}")

    files = sorted(
        p
        for p in path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in NOTE_SUFFIXES
        and not any(part.startswith(".") for part in p.relative_to(path).parts)
    )
    bundle = NotesBundle(root=path)
    parts: list[str] = []
    for f in files:
        rel = f.relative_to(path).as_posix()
        try:
            content = f.read_text(encoding="utf-8", errors="replace").strip()
        except OSError as e:
            bundle.skipped.append(f"{rel} ({e})")
            continue
        if not content:
            continue
        block = f"### File: {rel}\n\n{content}\n"
        cost = estimate_tokens(block)
        if bundle.tokens + cost > budget:
            bundle.skipped.append(rel)
            continue
        parts.append(block)
        bundle.files.append(rel)
        bundle.tokens += cost
    bundle.text = "\n".join(parts)
    return bundle
