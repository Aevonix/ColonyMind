"""Owner query vectors have one derived, bounded, erasable durable home."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import sqlite3

import numpy as np
import pytest

from colonymind.plasticity.query_log import QueryLog
from colonymind.turns.erasure_hooks import notify_source_erasure


class Clock:
    def __init__(self):
        self.now = datetime(2025, 1, 1, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


def runtime(*, loaded=True, enabled=True):
    result = SimpleNamespace(loaded=loaded, enabled=enabled)
    result.find = lambda implementation: "recall-learner" if result.loaded and implementation == "query_adapter" else None
    return result


@pytest.mark.parametrize("active", [None, runtime(loaded=False), runtime(enabled=False)])
def test_no_loaded_learner_creates_no_store(tmp_path, active):
    log = QueryLog(tmp_path / "derived" / "queries.db", runtime=active)
    assert not log.record("turn-1", [0.4, 0.7], audience="owner")
    assert log.get("turn-1") is None
    assert log.turn_ids() == ()
    assert log.prune() == 0
    assert log.erase_sources(["turn-1"]) == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("audience", ["contact", "guest", "", None, "unknown"])
def test_only_owner_recalls_are_persisted(tmp_path, audience):
    log = QueryLog(tmp_path / "queries.db", runtime=runtime())
    assert not log.record("turn-1", [0.4, 0.7], audience=audience)
    assert not log.path.exists()


def test_loaded_state_is_checked_at_each_write(tmp_path):
    active = runtime(loaded=False)
    log = QueryLog(tmp_path / "queries.db", runtime=active)
    assert not log.record("turn-1", [0.4, 0.7], audience="owner")
    active.loaded = True
    assert log.record("turn-1", [0.4, 0.7], audience="owner")
    active.loaded = False
    assert not log.record("turn-2", [0.6, 0.9], audience="owner")
    assert log.turn_ids() == ("turn-1",)


def test_float32_round_trip_owner_scope_and_no_extra_durable_copy(tmp_path):
    log = QueryLog(tmp_path / "queries.db", runtime=runtime())
    vector = np.array([0.132467891, -0.467891235, 0.989712345], dtype=np.float64)
    expected = vector.astype("<f4")
    assert log.record("turn-1", vector, audience="owner", source_ids=["passage-1"])
    restored = log.get("turn-1")
    assert restored.dtype == np.dtype("float32")
    np.testing.assert_array_equal(restored, expected)
    restored[0] = 88
    np.testing.assert_array_equal(log.get("turn-1"), expected)
    assert log.get("turn-1", audience="guest") is None
    assert log.get("turn-1", audience="contact") is None
    with sqlite3.connect(log.path) as conn:
        row = conn.execute("SELECT audience, dimension, vector FROM query_vectors").fetchone()
    assert row == ("owner", 3, expected.tobytes())
    assert log.path.stat().st_mode & 0o777 == 0o600
    containing = [path for path in tmp_path.rglob("*") if path.is_file() and expected.tobytes() in path.read_bytes()]
    assert containing == [log.path]
    assert not list(tmp_path.glob("*-journal"))
    assert not list(tmp_path.glob("*-wal"))


def test_retention_deletes_expired_bytes_and_keeps_younger_recalls(tmp_path):
    clock = Clock()
    log = QueryLog(tmp_path / "queries.db", runtime=runtime(), clock=clock)
    expired = np.array([0.19237466, -0.48647388, 0.84988337], dtype="<f4")
    log.record("turn-old", expired, audience="owner", source_ids=["old-source"])
    clock.now += timedelta(days=1)
    log.record("turn-new", [0.1, 0.8, 0.4], audience="owner")
    clock.now += timedelta(days=29)
    assert log.prune() == 1
    assert log.get("turn-old") is None
    assert log.turn_ids() == ("turn-new",)
    assert log.refs_resting_on(["old-source"]) == ()
    assert expired.tobytes() not in log.path.read_bytes()


def test_read_prunes_and_repeat_turn_does_not_extend_retention(tmp_path):
    clock = Clock()
    log = QueryLog(tmp_path / "queries.db", runtime=runtime(), clock=clock)
    log.record("turn-1", [0.4, 0.7], audience="owner", source_ids=["before"])
    clock.now += timedelta(days=29)
    log.record("turn-1", [0.5, 0.8], audience="owner", source_ids=["after"])
    assert log.turn_ids() == ("turn-1",)
    assert log.refs_resting_on(["before"]) == ()
    assert log.refs_resting_on(["after"]) == ("turn-1",)
    clock.now += timedelta(days=1)
    assert log.get("turn-1") is None
    assert log.turn_ids() == ()


def test_erasure_maps_turns_and_sources_before_removing_every_copy(tmp_path):
    log = QueryLog(tmp_path / "queries.db", runtime=runtime())
    erased = np.array([0.16232894, -0.95488238, 0.37813287], dtype="<f4")
    log.record("turn-a", erased, audience="owner", source_ids=["source-a", "shared"])
    log.record("turn-b", [0.2, 0.3, 0.4], audience="owner", source_ids=["source-b", "shared"])
    log.record("turn-c", [0.5, 0.6, 0.7], audience="owner", source_ids=["source-c"])
    assert log.refs_resting_on(["turn-a", "shared", "shared"]) == ("turn-a", "turn-b")
    assert log.erase_sources(["source-a", "source-a"]) == 1
    assert log.get("turn-a") is None
    assert erased.tobytes() not in log.path.read_bytes()
    assert b"turn-a" not in log.path.read_bytes()
    assert b"source-a" not in log.path.read_bytes()
    assert log.erase_sources(["turn-b"]) == 1
    assert log.erase_sources(["turn-b"]) == 0
    assert log.turn_ids() == ("turn-c",)


def test_optional_source_hook_is_idempotent_and_can_be_closed(tmp_path):
    log = QueryLog(tmp_path / "queries.db", runtime=runtime())
    log.record("turn-1", [0.4, 0.7], audience="owner")
    notify_source_erasure(["turn-1"])
    assert log.get("turn-1") is not None
    log.bind_erasure()
    log.bind_erasure()
    try:
        notify_source_erasure(["turn-1"])
        assert log.get("turn-1") is None
    finally:
        log.close()
        log.close()
    log.record("turn-2", [0.4, 0.7], audience="owner")
    notify_source_erasure(["turn-2"])
    assert log.get("turn-2") is not None


@pytest.mark.parametrize("vector", [[], [[0.1, 0.2]], [float("nan")], [float("inf")]])
def test_invalid_vectors_are_rejected_before_creating_a_store(tmp_path, vector):
    log = QueryLog(tmp_path / "queries.db", runtime=runtime())
    with pytest.raises(ValueError, match="vector"):
        log.record("turn-1", vector, audience="owner")
    assert not log.path.exists()


def test_invalid_turn_id_is_rejected_before_creating_a_store(tmp_path):
    log = QueryLog(tmp_path / "queries.db", runtime=runtime())
    with pytest.raises(ValueError, match="turn"):
        log.record("", [0.4, 0.7], audience="owner")
    assert not log.path.exists()


def test_committed_source_erasure_removes_query_vector(tmp_path, monkeypatch):
    from colonymind.turns import TurnIdempotencyLedger

    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    ledger.record_source("turn-1", contact_id="owner-contact", session_id="session-1",
                         messages=[{"role": "user", "content": "Recall the earlier plan."}])
    log = QueryLog(tmp_path / "queries.db", runtime=runtime(), bind_erasure=True)
    try:
        log.record("turn-1", [0.4, 0.7], audience="owner")
        from colonymind.memory import curated
        before = log.path.read_bytes()
        curated.forget_sources(ledger, 'owner-contact', source_ids=['turn-1'])
        assert log.get("turn-1") is None
        assert log.turn_ids() == ()
        assert log.path.read_bytes() == before
        assert not log.record('turn-1', [0.4, 0.7], audience='owner')
    finally:
        log.close()
