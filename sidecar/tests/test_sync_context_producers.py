"""Synchronous context reads obey the response deadline without worker writes."""
import asyncio
from contextlib import closing
import sqlite3
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from colonymind.api import viewer
from colonymind.api.routers import executions, host, social_state
from colonymind.api.schemas.host import ContextAssembleRequest, ContextSection
from colonymind.contacts.comms import CommsLog
from colonymind.commitments.store import CommitmentStore
from colonymind.initiatives.temporal_followup import TemporalFollowups
from colonymind.initiatives.store import InitiativeStore
from colonymind.memory import blocks, producers, turn_budget
from colonymind.turns import TurnIdempotencyLedger, get_turn_idempotency_ledger


def request():
    return ContextAssembleRequest.model_validate({
        "identity": {"host_id": "fixture"},
        "context": {"contact_id": "owner", "session_id": "session"},
        "incoming_message": {"role": "user", "content": ""},
        "include_initiatives": True,
    })


@pytest.fixture
def assembly(monkeypatch, tmp_path):
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", "owner")
    monkeypatch.setenv("COLONYMIND_CONTEXT_STATUS", "off")
    monkeypatch.setenv(turn_budget.WAIT_ENV, "300")
    blocks.reset_settings()
    for name in ("_initiative_store", "_commitment_store", "_contacts_store",
                 "_preference_learner", "_comms_log", "_telemetry"):
        monkeypatch.setattr(host, name, None)
    monkeypatch.setattr(host, "_mind", lambda: None)
    monkeypatch.setattr(viewer, "authorized_viewer", lambda *a, **kw: ("owner", True))

    async def queue(view, **kwargs):
        return view

    monkeypatch.setattr(executions, "with_queue_work", queue)
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    pool = producers.ProducerPool(workers=2)
    monkeypatch.setattr(producers, "POOL", pool)
    yield ledger, pool
    pool.executor.shutdown(wait=True)
    blocks.reset_settings()


async def test_blocking_sync_producer_cannot_hold_the_response(assembly, monkeypatch):
    release, entered = threading.Event(), threading.Event()

    def blocked(**kwargs):
        entered.set()
        release.wait(2)
        return [SimpleNamespace(type="note", description="late", priority=1)]

    monkeypatch.setattr(host, "_initiative_store", SimpleNamespace(list=blocked))
    try:
        started = time.monotonic()
        response = await host.context_assemble(request())
        elapsed = time.monotonic() - started
        assert entered.is_set() and elapsed < .3
        assert response.sections == [] and response.budget["timed_out"]
        # The timed-out worker still owns its slot.
        assert assembly[1].slots.acquire(blocking=False)
        try:
            assert not assembly[1].slots.acquire(blocking=False)
        finally:
            assembly[1].slots.release()
    finally:
        release.set()


async def test_comms_section_uses_worker_local_read_only_connection(assembly, monkeypatch, tmp_path):
    ledger, _ = assembly
    log = CommsLog(str(tmp_path / "communications.db"), source_ledger=ledger)
    log.log("owner", channel="chat", direction="out", summary="fixture")
    monkeypatch.setattr(host, "_comms_log", log)

    class Contacts:
        async def get(self, person):
            return SimpleNamespace(display_name="Contact", interaction_count=1,
                                   timezone="UTC", last_interaction_at=None)

    monkeypatch.setattr(host, "_contacts_store", Contacts())
    loop_thread = threading.get_ident()
    reads = []
    original = log.last_per_channel

    def checked(person):
        reads.append(threading.get_ident())
        assert log._conn.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            log._conn.execute("DELETE FROM communications")
        return original(person)

    monkeypatch.setattr(log, "last_per_channel", checked)
    try:
        response = await host.context_assemble(request())
        section = next(s for s in response.sections if s.id == "colonymind-comms-landscape")
        assert "Recorded channels: Other" in section.body
        assert "Last recorded outgoing message: Other" in section.body
        assert reads and all(thread != loop_thread for thread in reads)
        assert log._conn.execute("SELECT count(*) FROM communications").fetchone()[0] == 1
    finally:
        log._conn.close()


async def test_full_pool_skips_counts_and_does_not_admit_more_work(assembly, monkeypatch):
    _, pool = assembly
    release, entered = threading.Event(), threading.Event()
    started = []

    def blocked():
        started.append(threading.get_ident())
        if len(started) == 2:
            entered.set()
        release.wait(2)

    budget = turn_budget.TurnBudget(wait_s=.1)
    with turn_budget.turn(budget):
        tasks = [asyncio.create_task(producers.read(blocked)) for _ in range(2)]
        assert await asyncio.to_thread(entered.wait, .5)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    # Cancellation must not admit a new producer while the workers still run.
    calls = []
    monkeypatch.setattr(host, "_initiative_store", SimpleNamespace(list=lambda **kw: calls.append(kw)))
    try:
        response = await host.context_assemble(request())
        assert response.budget["sync_pool_skipped"] >= 1
        assert not response.budget.get("timed_out") and not calls and len(started) == 2
    finally:
        release.set()
    def drained():
        assert pool.slots.acquire(timeout=.5)
        assert pool.slots.acquire(timeout=.5)
        pool.slots.release()
        pool.slots.release()
    await asyncio.to_thread(drained)
    with turn_budget.turn(turn_budget.TurnBudget(wait_s=1)):
        # Capacity is reusable after the actual workers finish.
        assert await producers.read(lambda: "ready") == "ready"


async def test_sql_deadline_interrupts_and_closes_abandoned_read(assembly):
    ledger, _ = assembly
    closed, interrupted = threading.Event(), threading.Event()
    connections = []

    def scan():
        conn = ledger._connect()
        connections.append(conn)
        try:
            conn.execute("""WITH RECURSIVE numbers(n) AS (
                SELECT 1 UNION ALL SELECT n+1 FROM numbers WHERE n<1000000000)
                SELECT sum(n) FROM numbers""").fetchone()
        except sqlite3.OperationalError as error:
            assert "interrupted" in str(error)
            interrupted.set()
            raise
        finally:
            conn.close()
            closed.set()

    with turn_budget.turn(turn_budget.TurnBudget(wait_s=.05)):
        with pytest.raises((TimeoutError, sqlite3.OperationalError)):
            await producers.read(scan)
    assert await asyncio.to_thread(closed.wait, .5)
    assert interrupted.is_set()


async def test_abandoned_render_never_records_a_use(assembly, monkeypatch):
    release, entered = threading.Event(), threading.Event()
    writes = []

    def blocked():
        entered.set()
        release.wait(2)
        return "late text"

    async def sections(*args, **kwargs):
        text = await producers.read(blocked)
        turn_budget.defer_write(writes.append, threading.get_ident())
        return [ContextSection(id="x", body=text, priority=1)]

    monkeypatch.setattr(host, "_assemble_sections", sections)
    try:
        response = await host.context_assemble(request())
        assert entered.is_set() and response.budget["timed_out"] and not writes
    finally:
        release.set()
        assembly[1].executor.shutdown(wait=True)
    assert not writes


async def test_accepted_render_writes_only_on_event_loop_after_read(assembly, monkeypatch):
    writes, reads = [], []
    loop_thread = threading.get_ident()

    async def sections(*args, **kwargs):
        await producers.read(lambda: reads.append(threading.get_ident()))
        turn_budget.defer_write(writes.append, threading.get_ident())
        assert not writes
        return [ContextSection(id="x", body="in time", priority=1)]

    monkeypatch.setattr(host, "_assemble_sections", sections)
    response = await host.context_assemble(request())
    await asyncio.gather(*turn_budget._PENDING)
    assert response.sections and writes == [loop_thread]
    assert reads and reads[0] != loop_thread


async def test_bound_initiative_connection_does_not_allow_worker_writes(assembly, tmp_path):
    store = InitiativeStore(tmp_path)
    writer = store._db

    def read():
        assert store._db is not writer
        assert store._db.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            store._db.execute("DELETE FROM initiatives")
        return store.list(status=["pending"], limit=10)

    try:
        with turn_budget.turn(turn_budget.TurnBudget(wait_s=1)):
            assert await producers.read(read) == []
        assert store._db is writer
    finally:
        writer.close()


async def test_abandoned_database_read_cannot_write_when_it_resumes(assembly):
    ledger, pool = assembly
    ledger.record_source("retained", contact_id="owner", session_id="session",
                         messages=[{"role": "user", "content": "retained words"}])
    entered, release = threading.Event(), threading.Event()
    denied = []

    def blocked():
        conn = ledger._connect()
        assert conn.execute("SELECT count(*) FROM turn_sources").fetchone()[0] == 1
        entered.set()
        release.wait(2)
        try:
            conn.execute("DELETE FROM turn_sources")
        except sqlite3.Error:
            denied.append(True)
        # The pool owns closing this connection, including after cancellation.

    try:
        with turn_budget.turn(turn_budget.TurnBudget(wait_s=.05)):
            with pytest.raises(TimeoutError):
                await producers.read(blocked)
        assert entered.is_set()
    finally:
        release.set()
        pool.executor.shutdown(wait=True)
    assert denied == [True]
    with closing(ledger._connect()) as conn:
        assert conn.execute("SELECT count(*) FROM turn_sources").fetchone()[0] == 1


async def test_read_facade_does_not_initialize_or_poison_the_writer_cache(assembly, tmp_path):
    state = tmp_path / "new-state"
    with turn_budget.turn(turn_budget.TurnBudget(wait_s=1)):
        reader = await producers.read(get_turn_idempotency_ledger, state)
    assert not state.exists()
    writer = get_turn_idempotency_ledger(state)
    assert reader is not writer
    writer.record_source("first", contact_id="owner", session_id="session",
                         messages=[{"role": "user", "content": "first words"}])
    assert writer.source_references(["first"], contact_id="owner", session_id="session")


async def test_worker_write_proposals_are_discarded_with_an_abandoned_read(assembly):
    writes = []
    release = threading.Event()

    def blocked():
        producers.after_read(writes.append, "proposed")
        release.wait(2)

    try:
        with turn_budget.turn(turn_budget.TurnBudget(wait_s=.05)) as budget:
            with pytest.raises(TimeoutError):
                await producers.read(blocked)
            assert not budget.writes
    finally:
        release.set()
        assembly[1].executor.shutdown(wait=True)
    assert not writes


@pytest.mark.parametrize("accepted", [True, False])
async def test_waiting_receipt_snapshot_writes_only_for_an_accepted_response(
        assembly, monkeypatch, tmp_path, accepted):
    ledger, _ = assembly
    ledger.record_source("task:one", contact_id="owner", session_id="session",
                         messages=[{"role": "user", "content": "Obtain the task result"}])
    version = ledger.source_references(["task:one"], contact_id="owner", session_id="session")[0]
    store = CommitmentStore(tmp_path / "commitments.db")
    parent = store.create(person_id="owner", description="Obtain the task result")
    waits = TemporalFollowups(store)
    waits.expect_reply(wait_id="wait-one", commitment_id=parent['id'], work_id="task-one",
        contact_id="recipient", outbound_ref="message:out", source_refs=["task:one"],
        source_versions={"task:one": version['source_version']}, source_session_id="session",
        expected_after_seconds=60, expires_at=time.time() + 86400)
    log = CommsLog(str(tmp_path / "communications.db"), source_ledger=ledger)
    log.log_receipt(event_id="delivery", contact_id="recipient", channel="chat", direction="out",
        external_ref="provider:out", receipt_ref="receipt:sent", outbound_ref="message:out",
        occurred_at=datetime.now(timezone.utc).isoformat(), status="sent")
    monkeypatch.setattr(host, "_commitment_store", store)
    monkeypatch.setattr(host, "_comms_log", log)
    try:
        with turn_budget.turn(turn_budget.TurnBudget(wait_s=1)) as budget:
            text = await producers.read(social_state.waiting_context, "owner")
            assert "state=not_due" in text
            assert waits.get("wait-one")['dispatch_receipt_ref'] is None
            assert budget.writes
            if not accepted:
                budget.started -= 60
            writes = turn_budget.take_writes([]) if budget.assembly_left() > 0 else []
        await turn_budget.finish_writes(writes)
        assert waits.get("wait-one")['dispatch_receipt_ref'] == ("receipt:sent" if accepted else None)
    finally:
        log._conn.close()
