"""Regression tests for the bug/stub sweep: timeline recency, affect totals
and the hourly condition-check phase."""

import json
from datetime import datetime, timedelta, timezone


from colonymind.events.journal import replay_events
from colonymind.contacts.affect import AffectStore


# --- journal: newest_first drops the OLD end of an over-cap window -----------

def _write_event(d, seq, recorded_at, etype="test.event"):
    (d / f"{seq:08d}.ulid{seq}.json").write_text(json.dumps({
        "type": etype, "recordedAt": recorded_at, "data": {"n": seq}}))


def test_replay_newest_first_keeps_recent(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_EVENT_JOURNAL_DIR", str(tmp_path))
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(10):
        _write_event(tmp_path, i, (base + timedelta(minutes=i)).isoformat())

    old = replay_events("2000-01-01T00:00:00+00:00", limit=4)
    assert [e["seq"] for e in old["events"]] == [0, 1, 2, 3]      # oldest-first walk
    assert old["hasMore"] is True

    new = replay_events("2000-01-01T00:00:00+00:00", limit=4, newest_first=True)
    assert [e["seq"] for e in new["events"]] == [9, 8, 7, 6]      # the RECENT end
    assert new["hasMore"] is True

    all_new = replay_events("2000-01-01T00:00:00+00:00", limit=50, newest_first=True)
    assert len(all_new["events"]) == 10
    assert all_new["hasMore"] is False


def test_replay_has_more_boundary_and_types(tmp_path, monkeypatch):
    """cap == N-1: the single unprocessed boundary file must count toward
    hasMore (remaining[1:] used to drop it); and hasMore must honor the
    types filter."""
    monkeypatch.setenv("COLONYMIND_EVENT_JOURNAL_DIR", str(tmp_path))
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(5):
        _write_event(tmp_path, i, (base + timedelta(minutes=i)).isoformat(),
                     etype="a.type" if i < 4 else "b.type")

    r = replay_events("2000-01-01T00:00:00+00:00", limit=4)
    assert len(r["events"]) == 4
    assert r["hasMore"] is True          # exactly one unread file remains

    # types filter: 4 matching 'a.type' events, cap at 4 — the only remaining
    # file is 'b.type', so hasMore must be False
    r2 = replay_events("2000-01-01T00:00:00+00:00", limit=4, types=["a.type"])
    assert len(r2["events"]) == 4
    assert r2["hasMore"] is False


# --- affect: paginated total is the real count, not the page size ------------

def test_affect_count_events(tmp_path):
    store = AffectStore(db_path=tmp_path / "affect.db")
    for i in range(7):
        store.create_event(contact_id="c1", valence=0.1, source="test")
    store.create_event(contact_id="c2", valence=0.1, source="other")
    page = store.list_events(contact_id="c1", limit=3)
    assert len(page) == 3
    assert store.count_events(contact_id="c1") == 7
    assert store.count_events() == 8
    assert store.count_events(contact_id="c1", source="test") == 7


# --- autonomy: hourly condition-check phase exists and dedups ----------------

