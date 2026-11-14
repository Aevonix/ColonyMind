"""All store cancellation paths settle an owner cadence group atomically."""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from colonymind.commitments.store import CommitmentStore

NOW = datetime(2026, 11, 2, 12, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", "owner")
    monkeypatch.setattr("colonymind.commitments.store.now_utc", lambda: NOW)
    return CommitmentStore(tmp_path / "commitments.db")


def cadence(store, *, person="owner", recipient="me", minutes=1440, topic="the garden", made=None):
    row = store.create(person_id=person, description="Requested check-in", metadata={
        "kind": "cadence", "recipient": recipient, "cadence_minutes": minutes, "topic": topic})
    if made is not None:
        with store._connect() as conn:
            conn.execute("UPDATE commitments SET made_at=? WHERE id=?", (made.isoformat(), row["id"]))
    return row


@pytest.mark.parametrize("outcome", ["wont_do", "obsolete", "cancelled"])
def test_cancellation_settles_pending_and_overdue_siblings_with_provenance(store, outcome):
    first = cadence(store, made=NOW - timedelta(weeks=3))
    second = cadence(store, topic="the workshop", made=NOW - timedelta(weeks=3, minutes=-2))
    store.update(first["id"], status="overdue")
    weekly = cadence(store, minutes=7 * 1440)
    contact = cadence(store, recipient="contact")
    other_person = cadence(store, person="contact")
    unrelated = store.create(person_id="owner", description="Prepare the report")
    if outcome == "cancelled":
        store.update(second["id"], status="cancelled")
    else:
        store.resolve(second["id"], outcome=outcome, operation_id="cancel-cadence")
        store.resolve(second["id"], outcome=outcome, operation_id="cancel-cadence")
    sibling = store.get(first["id"])
    assert sibling["status"] == "cancelled"
    assert sibling["metadata"]["topic"] == "the garden"
    resolution = sibling["metadata"]["resolution"]
    assert resolution["outcome"] == "obsolete" and resolution["at"]
    assert resolution["cancelled_by_commitment_id"] == second["id"]
    for row in (weekly, contact, other_person, unrelated):
        assert store.get(row["id"])["status"] == "pending"


@pytest.mark.parametrize("outcome", ["done", "invalid", "duplicate"])
def test_resolving_a_duplicate_or_completed_row_keeps_the_cadence(store, outcome):
    first, second = cadence(store), cadence(store)
    store.resolve(second["id"], outcome=outcome)
    assert store.get(first["id"])["status"] == "pending"


@pytest.mark.parametrize("update", [False, True])
def test_sibling_failure_rolls_back_the_whole_cancellation(store, update, monkeypatch):
    first = cadence(store, made=NOW - timedelta(weeks=3))
    second = cadence(store, made=NOW - timedelta(weeks=1))
    events = []
    monkeypatch.setattr("colonymind.events.broadcaster.emit", lambda *args: events.append(args))
    settle = store._resolve_owner_cadence_siblings

    def failed(*args, **kwargs):
        settle(*args, **kwargs)
        raise sqlite3.OperationalError("sibling write failed")

    monkeypatch.setattr(store, "_resolve_owner_cadence_siblings", failed)
    with pytest.raises(sqlite3.OperationalError, match="sibling write failed"):
        if update:
            store.update(second["id"], status="cancelled")
        else:
            store.resolve(second["id"], outcome="wont_do", operation_id="cancel-cadence")
    assert store.get(first["id"])["status"] == store.get(second["id"])["status"] == "pending"
    assert store.get_resolution_operation(second["id"]) is None
    assert events == []


def test_contact_cadence_cancellation_does_not_cancel_its_siblings(store):
    first, second = cadence(store, person="contact"), cadence(store, person="contact")
    store.resolve(second["id"], outcome="wont_do")
    assert store.get(first["id"])["status"] == "pending"


def test_resolving_an_already_cancelled_row_leaves_a_newer_request_running(store):
    old = cadence(store)
    store.update(old["id"], status="cancelled")
    newer = cadence(store, topic="the workshop")
    try:
        store.resolve(old["id"], outcome="obsolete")
    except Exception as error:
        assert type(error).__name__ == "CommitmentResolutionConflict"
    assert store.get(newer["id"])["status"] == "pending"


@pytest.mark.parametrize("cancel_first", [False, True])
@pytest.mark.parametrize("outcome", ["wont_do", "obsolete", "cancelled"])
def test_cancelling_either_old_sibling_stops_both(store, cancel_first, outcome):
    first = cadence(store, made=NOW - timedelta(weeks=4))
    second = cadence(store, made=NOW - timedelta(weeks=1))
    selected = first if cancel_first else second
    if outcome == "cancelled":
        store.update(selected["id"], status="cancelled")
    else:
        store.resolve(selected["id"], outcome=outcome)
    assert store.get(first["id"])["status"] == store.get(second["id"])["status"] == "cancelled"


@pytest.mark.parametrize("age, cancelled", [(599, False), (600, False), (601, True)])
@pytest.mark.parametrize("outcome", ["wont_do", "obsolete", "cancelled"])
def test_cascade_spares_only_recent_replacements(store, age, cancelled, outcome):
    old = cadence(store, made=NOW - timedelta(weeks=3))
    replacement = cadence(store, made=NOW - timedelta(seconds=age))
    if outcome == "cancelled":
        store.update(old["id"], status="cancelled")
    else:
        store.resolve(old["id"], outcome=outcome)
    assert store.get(replacement["id"])["status"] == ("cancelled" if cancelled else "pending")


def test_cascade_decodes_only_candidate_cadences(store, monkeypatch):
    old = cadence(store, made=NOW - timedelta(weeks=3))
    sibling = cadence(store, made=NOW - timedelta(weeks=1))
    recent = cadence(store)
    weekly = cadence(store, minutes=7 * 1440, made=NOW - timedelta(weeks=1))
    contact = cadence(store, recipient="contact", made=NOW - timedelta(weeks=1))
    wrong_type = cadence(store, minutes=1440.0, made=NOW - timedelta(weeks=1))
    with store._connect() as conn:
        conn.executemany(
            "INSERT INTO commitments (id,person_id,description,made_at,status,metadata) VALUES (?,?,?,?,?,?)",
            [(f"unrelated-{index}", "owner", "Prepare report", NOW.isoformat(), "pending", '{}')
             for index in range(1000)])
    decoded = []
    row_to_dict = store._row_to_dict

    def observed(row):
        decoded.append(row["id"])
        return row_to_dict(row)

    monkeypatch.setattr(store, "_row_to_dict", observed)
    store.resolve(old["id"], outcome="obsolete")
    assert set(decoded) == {old["id"], sibling["id"], contact["id"], wrong_type["id"]}
    assert store.get(sibling["id"])["status"] == "cancelled"
    for row in (recent, weekly, contact, wrong_type):
        assert store.get(row["id"])["status"] == "pending"


def test_sibling_resolution_events_follow_commit_once_with_provenance(store, monkeypatch):
    old = cadence(store, made=NOW - timedelta(weeks=3))
    siblings = [cadence(store, made=NOW - timedelta(weeks=1)) for _ in range(2)]
    events = []

    def emitted(kind, payload):
        assert all(store.get(row["id"])["status"] == "cancelled" for row in [old, *siblings])
        events.append((kind, payload))

    monkeypatch.setattr("colonymind.events.broadcaster.emit", emitted)
    for _ in range(2):
        store.resolve(old["id"], outcome="wont_do", resolved_by="conversation", operation_id="cancel")
    assert len(events) == 3
    assert events[0] == ("commitment.cancelled", {
        "commitment_id": old["id"], "person_id": "owner", "outcome": "wont_do", "resolved_by": "conversation"})
    assert {payload["commitment_id"] for _, payload in events[1:]} == {row["id"] for row in siblings}
    for kind, payload in events[1:]:
        assert kind == "commitment.cancelled"
        assert payload["person_id"] == "owner" and payload["outcome"] == "obsolete"
        assert payload["resolved_by"] == "conversation"
        assert payload["cancelled_by_commitment_id"] == old["id"]


def test_sibling_event_failure_is_visible_and_does_not_stop_other_events(store, monkeypatch, caplog):
    from colonymind.health_counters import snapshot

    old = cadence(store, made=NOW - timedelta(weeks=3))
    siblings = [cadence(store, made=NOW - timedelta(weeks=1)) for _ in range(2)]
    events = []
    before = snapshot().get("writes_dropped.commitment_resolution_event", 0)

    def emitted(kind, payload):
        events.append(payload)
        if len(events) == 2:
            error = sqlite3.OperationalError("private content")
            error.sqlite_errorname = "SQLITE_BUSY"
            raise error

    monkeypatch.setattr("colonymind.events.broadcaster.emit", emitted)
    store.resolve(old["id"], outcome="obsolete")
    assert len(events) == 3
    assert all(store.get(row["id"])["status"] == "cancelled" for row in siblings)
    assert snapshot()["writes_dropped.commitment_resolution_event"] == before + 1
    assert "OperationalError" in caplog.text and "SQLITE_BUSY" in caplog.text
    assert "private content" not in caplog.text


def test_status_cancel_emits_each_sibling_cancellation_after_commit(store, monkeypatch):
    old = cadence(store, made=NOW - timedelta(weeks=3))
    sibling = cadence(store, made=NOW - timedelta(weeks=1))
    events = []

    def emitted(kind, payload):
        assert store.get(sibling["id"])["status"] == "cancelled"
        events.append((kind, payload))

    monkeypatch.setattr("colonymind.events.broadcaster.emit", emitted)
    store.update(old["id"], status="cancelled")
    assert [(kind, payload["commitment_id"], payload["cancelled_by_commitment_id"]) for kind, payload in events] == [
        ("commitment.cancelled", sibling["id"], old["id"])]


def test_a_row_with_unreadable_metadata_does_not_break_the_cascade(store):
    old = cadence(store, made=NOW - timedelta(weeks=3))
    sibling = cadence(store, made=NOW - timedelta(weeks=1))
    broken = store.create(person_id="owner", description="Prepare the report")
    with sqlite3.connect(store._db_path) as conn:
        conn.execute("UPDATE commitments SET metadata='not json' WHERE id=?", (broken["id"],))
    store.resolve(old["id"], outcome="wont_do")
    assert store.get(sibling["id"])["status"] == "cancelled"
