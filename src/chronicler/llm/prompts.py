"""Prompt text shared by every provider."""

from __future__ import annotations

from chronicler.llm.base import CampaignContext, ChunkContext, KnownEntity, SessionContext
from chronicler.transcribe import format_timestamp

SYSTEM_INSTRUCTIONS = """\
You are the scribe for a tabletop role-playing game (usually D&D) that is being \
played right now. You receive the session's transcript in parts, roughly every \
fifteen minutes, and keep the game master and players oriented while they play.

About the transcripts:
- They come from automatic speech recognition of the whole table, with no \
speaker labels. In-character speech, out-of-character chatter, rules talk and \
jokes are all mixed together.
- Names and in-world words are often misheard or spelled phonetically. Map them \
to the canonical spellings from the campaign notes and the known-entity list \
whenever the match is reasonable. If a name is genuinely new, keep the most \
plausible spelling.
- Ignore real-world meta discussion (snacks, scheduling, rules lookups) unless it \
changed what happened in the game.

Writing rules:
- Write everything in {notes_language}, even when the table speaks another language.
- Third person, past tense, narrative and concise. Never invent events that are \
not supported by the transcript; when something is unclear, say so briefly.
- Entities are the people, creatures, places, items and factions of the game \
world that actually matter in the transcript. Do not list player-character \
mechanics, dice or rules terms as entities.
- Threads are unresolved questions, promises, mysteries, hooks and goals. Reuse \
the exact title of a known open thread when the same thread continues.
"""


def system_prompt(campaign: CampaignContext) -> str:
    parts = [SYSTEM_INSTRUCTIONS.format(notes_language=campaign.notes_language)]
    parts.append(f"Campaign: {campaign.campaign_name}")
    if campaign.notes_text:
        parts.append(
            "Campaign notes provided by the game master (canon for spelling and "
            "continuity):\n\n<campaign_notes>\n" + campaign.notes_text + "\n</campaign_notes>"
        )
    return "\n\n".join(parts)


def _known_entities_block(entities: list[KnownEntity]) -> str:
    if not entities:
        return "(none yet)"
    lines = []
    for name, kind, aliases in entities:
        alias = f" (also heard as: {', '.join(aliases)})" if aliases else ""
        lines.append(f"- {name} [{kind}]{alias}")
    return "\n".join(lines)


def chunk_user_prompt(ctx: ChunkContext) -> str:
    threads = "\n".join(f"- {t}" for t in ctx.open_threads) or "(none yet)"
    recap = ctx.previous_recap or "(this is the first part of the session)"
    return f"""\
Known entities from earlier sessions and earlier parts of this one:
{_known_entities_block(ctx.known_entities)}

Known open threads:
{threads}

Recap of the session so far:
{recap}

New transcript, part {ctx.chunk_index + 1} \
({format_timestamp(ctx.start_s)} to {format_timestamp(ctx.end_s)}):
<transcript>
{ctx.transcript or "(no speech detected)"}
</transcript>

Analyze this new part. The recap you return replaces the previous one, so it \
must cover the whole session so far, not only this part. List only entities and \
thread changes that appear in this new part."""


def summary_user_prompt(ctx: SessionContext) -> str:
    beats = "\n".join(f"- {b}" for b in ctx.beats) or "(none)"
    return f"""\
The session has ended. Write the final session summary.

Known entities (canonical spellings):
{_known_entities_block(ctx.known_entities)}

Live recap written during play:
{ctx.recap or "(none)"}

Beats noted during play:
{beats}

Full transcript:
<transcript>
{ctx.transcript}
</transcript>

Guidelines: the title is a short evocative phrase capturing the session's arc. \
The summary is one paragraph that works as a standalone recap for someone who \
missed the session: starting situation, major turns, ending or cliffhanger. Key \
events are 3-5 distinct scenes in strict chronological order, each with a short \
descriptive heading and 1-4 bullets; each bullet has a bold 2-5 word lead-in and \
1-3 sentences of detail. Open questions are 3-5 things a player would wonder \
about after the session: motives, fates, secrets, unresolved threats."""
