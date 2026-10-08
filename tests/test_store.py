# pyright: reportOptionalMemberAccess=false, reportArgumentType=false
from __future__ import annotations

from chronicler.campaign.store import MIGRATIONS, Store


def make(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    campaign = store.create_campaign("Curse of Strahd", None)
    session = store.create_session(campaign.id, tmp_path / "sessions")
    return store, campaign, session


def test_migrations_are_idempotent(tmp_path) -> None:
    Store(tmp_path / "db.sqlite").close()
    store = Store(tmp_path / "db.sqlite")
    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)


def test_find_entity_matches_aliases_case_insensitively(tmp_path) -> None:
    store, campaign, _ = make(tmp_path)
    e = store.add_entity(campaign.id, "Ireena Kolyana", "npc", aliases=["Irina"])
    assert store.find_entity(campaign.id, "  ireena   KOLYANA ").id == e.id
    assert store.find_entity(campaign.id, "irina").id == e.id
    store.update_entity(e.id, status="dismissed")
    assert store.find_entity(campaign.id, "Irina") is None


def test_merge_moves_mentions_and_adds_aliases(tmp_path) -> None:
    store, campaign, session = make(tmp_path)
    chunk = store.add_chunk(session.id, 0, 0, 60, "x.wav")
    good = store.add_entity(campaign.id, "Strahd", "npc", status="confirmed")
    typo = store.add_entity(campaign.id, "Strad", "npc", aliases=["Estrad"], notes="vampire")
    store.add_mention(typo.id, session.id, chunk.id, "appeared")
    store.merge_entities(typo.id, good.id)

    merged = store.get_entity(good.id)
    assert merged.aliases == ["Strad", "Estrad"]
    assert "vampire" in merged.notes
    assert store.get_entity(typo.id) is None
    [(entity, notes, _is_new)] = store.session_entities(session.id)
    assert entity.id == good.id and notes == ["appeared"]


def test_session_entities_flags_new_ones(tmp_path) -> None:
    store, campaign, session = make(tmp_path)
    chunk = store.add_chunk(session.id, 0, 0, 60, "x.wav")
    old = store.add_entity(campaign.id, "Barovia", "place", first_session_id=None)
    new = store.add_entity(campaign.id, "Ismark", "npc", first_session_id=session.id)
    store.add_mention(old.id, session.id, chunk.id, "a")
    store.add_mention(new.id, session.id, chunk.id, "b")
    flags = {e.name: is_new for e, _, is_new in store.session_entities(session.id)}
    assert flags == {"Barovia": False, "Ismark": True}


def test_clear_chunk_effects(tmp_path) -> None:
    store, campaign, session = make(tmp_path)
    chunk = store.add_chunk(session.id, 0, 0, 60, "x.wav")
    e = store.add_entity(campaign.id, "Ismark", "npc")
    t = store.add_thread(campaign.id, "The burgomaster's funeral")
    store.add_mention(e.id, session.id, chunk.id, "a")
    store.add_thread_event(t.id, session.id, chunk.id, "opened", "b")
    store.clear_chunk_effects(chunk.id)
    assert store.session_entities(session.id) == []
    assert store.session_threads(session.id) == []


def test_interrupted_and_pending(tmp_path) -> None:
    store, _, session = make(tmp_path)
    store.add_chunk(session.id, 0, 0, 60, "a.wav")
    c2 = store.add_chunk(session.id, 1, 60, 120, "b.wav")
    store.set_chunk_status(c2.id, "done")
    assert store.mark_interrupted_sessions() == [session.id]
    assert store.get_session(session.id).status == "stopped"
    assert [c.idx for c in store.pending_chunks()] == [0]


def test_session_dir_is_created(tmp_path) -> None:
    _, _, session = make(tmp_path)
    assert session.path.is_dir()
    assert session.path.name.endswith(f"_session-{session.id}")


def test_migration_2_reclassifies_entities_from_notes(tmp_path) -> None:
    """Version-1 databases tracked notes entities as new; the first mention's
    `known` flag tells which ones came from the notes."""
    import json
    import sqlite3

    path = tmp_path / "v1.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(f"BEGIN; {MIGRATIONS[0]}; PRAGMA user_version = 1; COMMIT;")
    conn.executescript(
        """
        INSERT INTO campaigns VALUES (1, 'Valkara', NULL, 'x');
        INSERT INTO sessions VALUES (1, 1, NULL, 'x', NULL, 'stopped', '/tmp', '', NULL);
        INSERT INTO entities VALUES (1, 1, 'Solene Brey', 'npc', '[]', '', 'suggested', 1, 'x');
        INSERT INTO entities VALUES (2, 1, 'Golden thread knot', 'other', '[]', '',
                                     'suggested', 1, 'x');
        INSERT INTO entities VALUES (3, 1, 'Mickey', 'npc', '[]', '', 'dismissed', 1, 'x');
        """
    )
    first = {
        "entities": [
            {"name": "solene brey", "kind": "npc", "known": True, "note": ""},
            {"name": "Golden thread knot", "kind": "other", "known": False, "note": ""},
            {"name": "Mickey", "kind": "npc", "known": True, "note": ""},
        ]
    }
    # Later chunks call the knot "known" only because it was already tracked.
    later = {
        "entities": [{"name": "Golden thread knot", "kind": "other", "known": True, "note": ""}]
    }
    for chunk_id, analysis in ((1, first), (2, later)):
        conn.execute(
            "INSERT INTO chunks (id, session_id, idx, start_s, end_s, audio_path, status, "
            "analysis_json) VALUES (?, 1, ?, 0, 1, 'a.wav', 'done', ?)",
            (chunk_id, chunk_id - 1, json.dumps(analysis)),
        )
    conn.executemany(
        "INSERT INTO mentions (entity_id, session_id, chunk_id, note) VALUES (?, 1, ?, '')",
        [(1, 1), (2, 1), (3, 1), (2, 2)],
    )
    conn.commit()
    conn.close()

    store = Store(path)
    solene, knot, mickey = (store.get_entity(i) for i in (1, 2, 3))
    assert (solene.status, solene.first_session_id) == ("confirmed", None)
    assert (knot.status, knot.first_session_id) == ("suggested", 1)
    assert mickey.status == "dismissed"  # the user's decision is kept
