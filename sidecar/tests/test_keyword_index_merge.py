"""The keyword indexes' stepwise segment merge (``TurnIdempotencyLedger.merge_keyword_indexes``) and its task."""
from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing

import pytest

from colonymind.turns import keywords
from colonymind.turns.idempotency import TurnIdempotencyLedger, keyword_index_maintenance


def _ledger(tmp_path, count=120):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    for index in range(count):
        ledger.record_source(f"t{index}", contact_id="owner", session_id=f"s{index}",
                             messages=[{"role": "user", "content": f"we talked about the harbor trip number {index}"}],
                             occurred_at="2026-05-01T10:00:00+00:00", derive_claims=False)
    return ledger


def _blocks(ledger, table):
    with closing(sqlite3.connect(ledger.db_path)) as conn:
        return conn.execute(f"SELECT count(*) FROM {table}_data").fetchone()[0]


def _hits(ledger):
    return sorted((row["turn_id"], row.get("content")) for row in ledger.search_sources(
        "harbor trip", contact_id="owner", session_id="s0", limit=500))


def test_maintenance_survives_a_failed_pass(tmp_path):
    calls, sleeps = [], []

    class Ledger:
        def merge_keyword_indexes(self, stop=None):
            calls.append(1)
            if len(calls) == 1:
                raise sqlite3.OperationalError("database is locked")
            return {"turn_source_search": 3}

    async def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 3:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(keyword_index_maintenance(Ledger(), sleep=sleep, first_delay=5, interval=7))
    assert sleeps == [5, 7, 7] and len(calls) == 2


def test_cancelling_the_task_stops_a_running_merge(tmp_path):
    import threading
    entered, seen = threading.Event(), {}

    class Ledger:
        def merge_keyword_indexes(self, stop):
            entered.set()
            seen["stopped"] = stop.wait(5)
            return {}

    async def main():
        task = asyncio.create_task(keyword_index_maintenance(lambda: Ledger(), first_delay=0, interval=60))
        while not entered.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.to_thread(lambda: None)

    asyncio.run(main())
    assert seen["stopped"] is True


def test_an_unreadable_ledger_is_a_failed_pass_not_a_failed_start():
    sleeps = []

    def ledger():
        raise sqlite3.OperationalError("unable to open database file")

    async def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(keyword_index_maintenance(ledger, sleep=sleep, first_delay=1, interval=2))
    assert sleeps == [1, 2]

from colonymind.memory import curated

def test_merge_shrinks_the_indexes_and_keeps_every_hit(tmp_path):
    ledger = _ledger(tmp_path)
    curated.forget_sources(ledger, "owner", source_ids=[f"t{index}" for index in range(0, 120, 2)])
    hits, before = _hits(ledger), {table: _blocks(ledger, table) for table in ("turn_source_search", keywords.TABLE)}
    pauses = []
    steps = ledger.merge_keyword_indexes(pages=4, pause=0.01, sleep=pauses.append)
    assert hits and _hits(ledger) == hits
    assert all(_blocks(ledger, table) < before[table] for table in before)
    assert all(count >= 2 for count in steps.values()) and pauses and set(pauses) == {0.01}
    assert ledger.merge_keyword_indexes(pages=4, sleep=pauses.append) == {"turn_source_search": 1, keywords.TABLE: 1}


def test_merge_stops_at_max_steps(tmp_path):
    ledger = _ledger(tmp_path)
    curated.forget_sources(ledger, "owner", source_ids=[f"t{index}" for index in range(60)])
    assert ledger.merge_keyword_indexes(pages=1, max_steps=1, sleep=lambda _: None) == {
        "turn_source_search": 1, keywords.TABLE: 1}


def test_merge_keeps_every_stemmed_hit_and_the_exact_survivors(tmp_path):
    ledger = _ledger(tmp_path)
    curated.forget_sources(ledger, "owner", source_ids=[f"t{index}" for index in range(0, 120, 2)])
    def stems():
        return [(row["turn_id"], row.get("content")) for row in ledger.search_stems(
            "talking harbors", contact_id="owner", session_id="s0", limit=500)]
    def survivors():
        with closing(sqlite3.connect(ledger.db_path)) as conn:
            return sorted(row[0] for row in conn.execute("SELECT DISTINCT turn_id FROM turn_source_search"))
    stemmed, written, ranked = stems(), survivors(), [row["turn_id"] for row in ledger.search_sources(
        "harbor trip", contact_id="owner", session_id="s0", limit=500)]
    ledger.merge_keyword_indexes(pages=4, sleep=lambda _: None)
    assert stemmed and stems() == stemmed
    assert written == sorted(f"t{index}" for index in range(120)) and survivors() == written
    assert [row["turn_id"] for row in ledger.search_sources(
        "harbor trip", contact_id="owner", session_id="s0", limit=500)] == ranked


def test_a_set_stop_ends_the_merge_between_steps(tmp_path):
    import threading
    ledger = _ledger(tmp_path)
    curated.forget_sources(ledger, "owner", source_ids=[f"t{index}" for index in range(60)])
    stop = threading.Event()
    steps = ledger.merge_keyword_indexes(pages=1, sleep=lambda _: stop.set(), stop=stop)
    assert steps == {"turn_source_search": 1, keywords.TABLE: 0}
