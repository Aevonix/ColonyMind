"""B7 review regressions: response commitment, safe skips and bounded reads."""
import asyncio
from dataclasses import asdict
import json
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks

from onekey import RequestAuthority
from colonymind import vector
from colonymind.api.routers import executions, host, social_state
from colonymind.api.schemas.host import ContextSection
from colonymind.commitments.store import CommitmentStore
from colonymind.contacts.comms import CommsLog
from colonymind.initiatives.store import InitiativeStore
from colonymind.initiatives.temporal_followup import TemporalFollowups
from colonymind.memory import producers, search, turn_budget
from colonymind.mind import status
from colonymind.plasticity.learners.query_adapter import RecallRequest
from colonymind.plasticity.revisions import Pinned
from colonymind.turns import source_vectors
from test_plasticity_query_adapter_driver import runtime, promote  # noqa: F401
from test_recall_query_adapter_live import Pipeline, Store
from test_source_vectors import setup, drain
from test_sync_context_producers import assembly, request  # noqa: F401

_QUEUE_WORK = executions.with_queue_work


async def flush():
    if turn_budget._PENDING:
        await asyncio.gather(*turn_budget._PENDING)


@pytest.mark.parametrize('recipient', [False, True])
@pytest.mark.parametrize('fails', [False, True])
async def test_served_callback_leaves_loop_and_has_stable_failure_site(assembly, monkeypatch, caplog,
                                                                    recipient, fails):
    from functools import wraps
    from colonymind.health_counters import snapshot

    monkeypatch.setenv(turn_budget.WAIT_ENV, '2000')
    selected = [{'id': 'fixture'}]
    async def collect(*args, **kwargs):
        return SimpleNamespace(semantic={}, curated_recall=None)
    async def select(*args, **kwargs):
        return SimpleNamespace(content='Recalled fixture', selected=selected, source_refs=[])
    async def fit(sections, recalled, **kwargs):
        return sections, recalled.packet.selected
    monkeypatch.setattr(search, 'collect_sources', collect)
    monkeypatch.setattr(search, 'select_memory', select)
    monkeypatch.setattr(host, '_memory_within_allowance', fit)
    calls = []
    loop_thread = threading.get_ident()
    @wraps(host._learning_served)
    def served(ledger, query, body, rows):
        calls.append((threading.get_ident(), rows))
        if fails:
            raise ValueError('private served payload')
    monkeypatch.setattr(host, '_learning_served', served)
    before = snapshot()
    body = request()
    body.incoming_message.content = 'fixture query'
    response = (await host.assemble_packet('owner', query='fixture query') if recipient
                else await host.context_assemble(body))
    assert response if recipient else any(section.id == 'colonymind-memory' for section in response.sections)
    assert calls == []
    await flush()
    assert len(calls) == 1 and calls[0][1] == selected
    assert calls[0][0] != loop_thread
    after = snapshot()
    for counter in ('context_writes_dropped', 'writes_dropped._learning_served'):
        assert after.get(counter, 0) == before.get(counter, 0) + int(fails)
    assert after.get('writes_dropped.<lambda>', 0) == before.get('writes_dropped.<lambda>', 0)
    if fails:
        assert '_learning_served' in caplog.text and 'ValueError' in caplog.text
        assert 'private served payload' not in caplog.text


@pytest.mark.parametrize("recipient", [False, True])
async def test_blocking_write_runs_after_response_outside_turn(assembly, monkeypatch, recipient):
    monkeypatch.setenv(turn_budget.WAIT_ENV, "100")
    calls = []
    loop_thread = threading.get_ident()

    def write():
        calls.append((threading.get_ident(), turn_budget.current()))
        time.sleep(.15)

    async def sections(*args, **kwargs):
        turn_budget.defer_write(write)
        return [ContextSection(id="fixture", body="accepted", priority=1)]

    monkeypatch.setattr(host, "_assemble_sections", sections)
    started = time.monotonic()
    response = (await host.assemble_packet("owner") if recipient
                else await host.context_assemble(request()))
    assert time.monotonic() - started < .1
    assert response if recipient else response.sections
    assert calls == []
    await flush()
    assert calls == [(loop_thread, None)]


async def test_response_expired_after_proposing_write_discards_it(assembly, monkeypatch):
    calls = []

    async def sections(*args, **kwargs):
        turn_budget.defer_write(calls.append, "discarded")
        await asyncio.sleep(1)

    monkeypatch.setattr(host, "_assemble_sections", sections)
    response = await host.context_assemble(request())
    assert response.budget["timed_out"] and response.sections == []
    await flush()
    assert calls == []


async def test_http_write_task_starts_only_from_the_post_response_hook(assembly, monkeypatch):
    sent, calls = [], []
    background = BackgroundTasks()

    def write():
        assert sent == ["response bytes"]
        assert turn_budget.current() is None
        calls.append("write")

    async def sections(*args, **kwargs):
        turn_budget.defer_write(write)
        return [ContextSection(id="fixture", body="accepted", priority=1)]

    monkeypatch.setattr(host, "_assemble_sections", sections)
    response = await host.context_assemble(request(), background_tasks=background)
    assert response.sections and calls == []
    await asyncio.sleep(0)
    assert calls == []
    sent.append("response bytes")
    await background()
    await flush()
    assert calls == ["write"]


async def test_real_owner_assembly_has_capacity_for_work_and_recall(monkeypatch, tmp_path):
    # All section producers and work readers are production implementations.
    # Only the embedding endpoint is supplied by the local deterministic fixture.
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", "owner")
    monkeypatch.setenv(turn_budget.WAIT_ENV, "2000")
    monkeypatch.setenv("COLONYMIND_CONTEXT_STATUS", "off")
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    for name in ("_contacts_store", "_preference_learner", "_comms_log", "_telemetry",
                 "_context_recall_selector", "_reranker"):
        monkeypatch.setattr(host, name, None)
    monkeypatch.setattr(host, "_mind", lambda: None)
    ledger, store, pipeline, projection = await setup(tmp_path)
    ledger.record_source("source", contact_id="owner", session_id="session",
                         messages=[{"role": "user", "content": "The office labeler is ready."}])
    await drain(projection)
    initiatives = InitiativeStore(tmp_path)
    commitments = CommitmentStore(tmp_path / "commitments.db")
    commitments.create(person_id="owner", description="Inspect the office labeler")
    TemporalFollowups(commitments)
    pool = producers.ProducerPool()
    monkeypatch.setattr(producers, "POOL", pool)
    monkeypatch.setattr(host, "_initiative_store", initiatives)
    monkeypatch.setattr(host, "_commitment_store", commitments)
    monkeypatch.setattr(vector, "_store", store)
    monkeypatch.setattr(vector, "_pipeline", pipeline)
    authority = RequestAuthority(principal_id="fixture", credential_id="fixture",
        scopes=frozenset({"context:read"}), viewer_person_id="owner",
        person_ids=frozenset({"owner"}), audiences=frozenset({"viewer"}), authenticated=True)
    body = request()
    body.incoming_message.content = "office labeler"
    try:
        response = await host.context_assemble(body, SimpleNamespace(
            state=SimpleNamespace(colonymind_authority=authority)))
        assert not response.budget.get("timed_out")
        assert response.budget["sync_pool_skipped"] == 0
        assert response.budget["sync_read_failed"] == 0
        assert any(section.id == "colonymind-memory" for section in response.sections)
        assert response.source_erasure_watermark == 0
        await flush()
    finally:
        pool.executor.shutdown(wait=True)
        initiatives.close()


async def test_healthy_concurrent_reads_do_not_count_as_abandoned_capacity(assembly):
    pool = assembly[1]
    with turn_budget.turn(turn_budget.TurnBudget(wait_s=1)) as budget:
        results = await asyncio.gather(*(pool.read(lambda: (time.sleep(.01), "read")[1])
                                         for _ in range(8)))
        assert results == ["read"] * 8 and budget.sync_pool_skipped == 0


def skip():
    turn_budget.current().sync_pool_skipped += 1
    raise producers.Skipped("fixture exhausted pool")


@pytest.mark.parametrize("site", ["host", "recall"])
async def test_erasure_watermark_skip_fails_closed(assembly, monkeypatch, site):
    original = producers.read

    async def read(call, *args, **kwargs):
        if getattr(call, "__name__", "") == "erasure_watermark" or (
                "erasure_watermark" in getattr(call, "__code__", SimpleNamespace(co_names=())).co_names):
            return skip()
        return await original(call, *args, **kwargs)

    monkeypatch.setattr(producers, "read", read)
    if site == "host":
        response = await host.context_assemble(request())
        assert response.sections == [] and response.source_erasure_watermark is None
        assert response.budget["unavailable"] and response.budget["sync_pool_skipped"] == 1
    else:
        with turn_budget.turn(turn_budget.TurnBudget(wait_s=1)) as budget:
            with pytest.raises(producers.Skipped):
                await search.collect_sources(assembly[0], query="words", contact_id="owner", session_id="session")
            assert budget.sync_pool_skipped == 1


async def test_skipped_generation_never_opens_unchecked_table(assembly, monkeypatch):
    store, pipeline = Store(), Pipeline()
    opened = []

    async def table(*args, **kwargs):
        opened.append(kwargs)

    monkeypatch.setattr(store, "_table", table)
    original = producers.read

    async def read(call, *args, **kwargs):
        if call is store.catalog.read_generation:
            assert "default" not in kwargs
            return skip()
        return await original(call, *args, **kwargs)

    monkeypatch.setattr(producers, "read", read)
    with turn_budget.turn(turn_budget.TurnBudget(wait_s=1)) as budget:
        with pytest.raises(producers.Skipped):
            await source_vectors.SourceVectors(None, store, pipeline).search(
                "words", contact_id="owner", session_id="session")
        assert budget.sync_pool_skipped == 1 and opened == []


async def test_skipped_reservations_render_unknown(assembly, monkeypatch, tmp_path):
    store = CommitmentStore(tmp_path / "commitments.db")
    store.create(person_id="owner", description="Inspect the selected documents")
    monkeypatch.setattr(host, "_commitment_store", store)
    original = producers.read

    async def read(call, *args, **kwargs):
        if "for_commitments" in getattr(call, "__code__", SimpleNamespace(co_names=())).co_names:
            assert "default" not in kwargs
            return skip()
        return await original(call, *args, **kwargs)

    monkeypatch.setattr(producers, "read", read)
    response = await host.context_assemble(request())
    section = next(s for s in response.sections if s.id == "colonymind-commitments")
    assert "work=unknown" in section.body and "work=unclaimed" not in section.body
    assert response.budget["sync_pool_skipped"] == 1
    await flush()


def owner_search(runtime, monkeypatch):
    driver = runtime.loaded["recall-test"].driver
    driver.document_identity = asdict(Pipeline.index_identity.document_identity)
    monkeypatch.setattr(source_vectors, "hydrate", lambda *a, **kw: {
        "turn_id": "source", "role": "user", "scope": "person", "content": "source text"})
    return source_vectors.SourceVectors(None, Store(), Pipeline()), driver


async def test_adversarial_skip_prevents_learned_ranking(assembly, runtime, monkeypatch):
    projection, driver = owner_search(runtime, monkeypatch)
    pinned = promote(runtime)
    state = dict(pinned.state)
    state["shown_U"] = state["shown_U"].copy()
    state["shown_V"] = state["shown_V"].copy()
    state["shown_U"][0, 0] = state["shown_V"][0, 0] = .01
    original_pin = runtime.pin
    monkeypatch.setattr(runtime, "pin", lambda *a, **kw: Pinned(pinned.closure, state)
                        if original_pin(*a, **kw).closure.id == pinned.closure.id else original_pin(*a, **kw))
    # This test isolates ranking's mandatory non-owner guard from cutoff validation.
    monkeypatch.setattr(driver, "cutoff", lambda pinned: None)
    table = SimpleNamespace()
    table.query = lambda: table
    table.where = lambda *args: table
    table.select = lambda *args: table

    async def arrow():
        return SimpleNamespace(to_pylist=lambda: [{"id": "chunk", "vector": [.6, .8, 0., 0.]}])

    table.to_arrow = arrow
    original = producers.read
    ranked = []
    monkeypatch.setattr(driver.learner, "rank_scores", lambda *a, **kw: ranked.append(kw))

    async def read(call, *args, **kwargs):
        if getattr(call, "__name__", "") == "<lambda>" and not args:
            assert "default" not in kwargs
            return skip()
        return await original(call, *args, **kwargs)

    monkeypatch.setattr(producers, "read", read)
    with turn_budget.turn(turn_budget.TurnBudget(wait_s=1)) as budget:
        with pytest.raises(producers.Skipped):
            await projection._owner_search([.6, .8, 0., 0.],
                RecallRequest.create(runtime, owner=True, turn_id="skipped-guard"),
                {"limit": 10}, table, contact_id="owner", session_id="session")
        assert budget.sync_pool_skipped == 1 and not ranked and not budget.writes


async def test_vector_catalog_and_hit_eligibility_run_on_workers(assembly, monkeypatch, tmp_path):
    ledger, store, pipeline, projection = await setup(tmp_path)
    ledger.record_source("source", contact_id="owner", session_id="session",
                         messages=[{"role": "user", "content": "The office labeler is ready."}])
    await drain(projection)
    loop_thread = threading.get_ident()
    reads = []
    for name in ("read_generation", "deleted", "source_erased"):
        original = getattr(store.catalog, name)

        def checked(*args, _name=name, _original=original, **kwargs):
            reads.append(_name)
            assert threading.get_ident() != loop_thread and producers.active()
            return _original(*args, **kwargs)

        monkeypatch.setattr(store.catalog, name, checked)
    with turn_budget.turn(turn_budget.TurnBudget(wait_s=1)):
        hits = await store.search(vector.Collection.CONVERSATIONS, pipeline.vector("office"))
    assert hits and set(reads) == {"read_generation", "deleted", "source_erased"}


async def test_runtime_lock_cannot_hold_the_response(assembly, runtime, monkeypatch):
    projection, driver = owner_search(runtime, monkeypatch)
    recall_request = RecallRequest.create(runtime, owner=True, turn_id="locked-read")
    entered, release = threading.Event(), threading.Event()

    def holder():
        with runtime.lock:
            entered.set()
            release.wait(2)

    thread = threading.Thread(target=holder)
    thread.start()
    assert await asyncio.to_thread(entered.wait, .5)
    try:
        started = time.monotonic()
        with turn_budget.turn(turn_budget.TurnBudget(wait_s=.05)) as budget:
            with pytest.raises(TimeoutError):
                await projection.search("words", contact_id="owner", session_id="session",
                    recall_request=recall_request)
            assert not budget.writes
        assert time.monotonic() - started < .1
    finally:
        release.set()
        thread.join(.5)
    assert driver.queries.get("locked-read") is None


@pytest.mark.parametrize("cursor", [False, True])
async def test_many_short_lookups_release_abandoned_slot_at_deadline(assembly, cursor):
    ledger, pool = assembly
    stopped, entered = threading.Event(), threading.Event()

    def lookups():
        try:
            conn = ledger._connect()
            reader = conn.cursor() if cursor else conn
            entered.set()
            while True:
                reader.execute("SELECT 1").fetchone()
        finally:
            stopped.set()

    started = time.monotonic()
    with turn_budget.turn(turn_budget.TurnBudget(wait_s=.08)):
        task = asyncio.create_task(producers.read(lookups))
        assert await asyncio.to_thread(entered.wait, .5)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert await asyncio.to_thread(stopped.wait, .15)
    assert time.monotonic() - started < .2
    assert pool.slots.acquire(timeout=.1) and pool.slots.acquire(timeout=.1)
    pool.slots.release()
    pool.slots.release()


@pytest.mark.parametrize("operation", ["execute", "executemany", "cursor", "cursor_execute", "cursor_executemany"])
async def test_bound_connections_check_every_statement_entry(assembly, tmp_path, operation):
    store = InitiativeStore(tmp_path)

    def read():
        conn = store._db
        cursor = conn.cursor()
        producers._READ.get().abandoned.set()
        with pytest.raises(sqlite3.OperationalError, match="deadline"):
            if operation == "execute":
                store._db.execute("SELECT 1")
            elif operation == "executemany":
                conn.executemany("SELECT 1", [()])
            elif operation == "cursor":
                conn.cursor()
            elif operation == "cursor_execute":
                cursor.execute("SELECT 1")
            else:
                cursor.executemany("SELECT 1", [()])

    try:
        with turn_budget.turn(turn_budget.TurnBudget(wait_s=1)):
            await producers.read(read)
    finally:
        store.close()


async def test_read_failures_count_even_if_handled_and_warn_per_error(assembly, tmp_path, caplog):
    ledger, _ = assembly

    def failed(kind):
        if kind == "cantopen":
            producers.connection(tmp_path / "absent.db")
        elif kind == "memory":
            producers.connection(":memory:")
        else:
            conn = ledger._connect()
            producers._READ.get().abandoned.set()
            try:
                conn.execute("SELECT 1")
            except sqlite3.Error:
                return "handled"

    with turn_budget.turn(turn_budget.TurnBudget(wait_s=1)) as budget:
        await asyncio.gather(*(producers.read(failed, kind) for kind in
                               ("cantopen", "memory", "interrupted")), return_exceptions=True)
        assert budget.sync_read_failed == 3
    assert len([r for r in caplog.records if "context worker read failed" in r.message]) == 3


async def test_status_stall_guard_is_per_store(assembly, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    mind = SimpleNamespace()

    def blocked():
        calls.append("blocked")
        entered.set()
        release.wait(2)
        return "old value"

    monkeypatch.setattr(status, "_plan", lambda *args: [
        ("Autonomy", lambda: "Autonomy: standard.", "", "head"),
        ("Capture", blocked, "capture-fixture", "degraded"),
        ("Breaker", lambda: "Breaker: clear.", "mind-fixture", "posture")])
    try:
        with turn_budget.turn(turn_budget.TurnBudget(wait_s=1)):
            first = await status.collect(mind, deadline=.05)
            second = await status.collect(mind, deadline=.05)
        assert entered.is_set() and calls == ["blocked"]
        assert "Capture: unknown." in first and "Capture: unknown." in second
        assert "Breaker: clear." in first and "Breaker: clear." in second
    finally:
        release.set()
        assembly[1].executor.shutdown(wait=True)


def waiting_store(tmp_path):
    store = CommitmentStore(tmp_path / "commitments.db")
    parent = store.create(person_id="owner", description="Read the task result")
    waits = TemporalFollowups(store)
    waits.expect_reply(wait_id="wait", commitment_id=parent["id"], work_id="task",
        contact_id="recipient", outbound_ref="message:out", source_refs=["source"],
        source_versions={"source": "version-one"}, source_session_id="session",
        expected_after_seconds=60, expires_at=time.time() + 86400)
    return store, waits


async def test_waiting_snapshot_invalidation_uses_overlay_revision(assembly, tmp_path):
    store, waits = waiting_store(tmp_path)
    before = waits.get("wait")["revision"]

    def read():
        snapshot = social_state._waiting_reader(store)
        snapshot.defer("wait", until=time.time() + 3600, evidence_ref="evidence:defer")
        snapshot.invalidate_sources(["source"], evidence_ref="source:erased")
        return snapshot.get("wait")

    with turn_budget.turn(turn_budget.TurnBudget(wait_s=1)):
        row = await producers.read(read)
        writes = turn_budget.take_writes([])
    assert row["state"] == "cancelled" and row["revision"] == before + 2
    await turn_budget.finish_writes(writes)
    assert waits.get("wait")["state"] == "cancelled"
    assert waits.get("wait")["revision"] == before + 2


async def test_waiting_snapshot_refresh_returns_saved_revision(assembly, tmp_path):
    store, waits = waiting_store(tmp_path)
    before = waits.get("wait")["revision"]

    def read():
        snapshot = social_state._waiting_reader(store)
        row = snapshot.get("wait")
        with snapshot.transaction() as db:
            return snapshot._refresh(db, row, row["expires_at"] + 1)

    with turn_budget.turn(turn_budget.TurnBudget(wait_s=1)):
        row = await producers.read(read)
        writes = turn_budget.take_writes([])
    assert row["state"] == "expired" and row["revision"] == before + 1
    await turn_budget.finish_writes(writes)
    assert waits.get("wait")["revision"] == row["revision"]


async def test_skipped_work_readers_report_unavailable(assembly, monkeypatch):
    async def read(*args, **kwargs):
        return skip()

    monkeypatch.setattr(producers, "read", read)
    # Use the production aggregator, removed by the shared small-pool fixture.
    aggregator = _QUEUE_WORK
    with turn_budget.turn(turn_budget.TurnBudget(wait_s=1)) as budget:
        view = await aggregator({"items": []}, owner=True)
        assert budget.sync_pool_skipped == 4
    for name in ("native_cron", "native_kanban", "reported_worker", "native_processes"):
        assert view["work_sources"][name]["status"] == "unavailable"


async def test_comms_lineage_without_turn_id_does_not_hide_other_rows(assembly, tmp_path):
    log = CommsLog(str(tmp_path / "communications.db"), source_ledger=assembly[0])
    log.log("owner", channel="chat", direction="out", summary="retained")
    log.log("owner", channel="other", direction="out", summary="unproven")
    log._conn.execute("UPDATE communications SET source_lineage_json=? WHERE channel='other'",
                      (json.dumps({"session_id": "session"}),))
    log._conn.commit()
    try:
        with turn_budget.turn(turn_budget.TurnBudget(wait_s=1)):
            rows = await producers.read(log.last_per_channel, "owner")
        assert rows["chat"]["summary"] == "retained" and "other" not in rows
    finally:
        log._conn.close()


async def test_deferred_comparator_rechecks_previous_inside_lock(assembly, runtime, monkeypatch):
    projection, driver = owner_search(runtime, monkeypatch)
    with turn_budget.turn(turn_budget.TurnBudget(wait_s=1)):
        await projection.search("words", contact_id="owner", session_id="session",
            recall_request=RecallRequest.create(runtime, owner=True, turn_id="changed-reference"))
        writes = turn_budget.take_writes([])
    binding = runtime.loaded["recall-test"]
    monkeypatch.setattr(binding.controller, "previous", lambda closure: SimpleNamespace(id="changed"))
    await turn_budget.finish_writes(writes)
    assert not binding.comparator.records()
    assert driver.queries.get("changed-reference") is None


async def test_vector_producer_failure_discards_its_earlier_record(assembly, runtime, monkeypatch):
    projection, driver = owner_search(runtime, monkeypatch)
    original = source_vectors.hydrate
    calls = []

    def hydrate(*args, **kwargs):
        calls.append(1)
        # Both predictors and their rechecks succeeded; fail the final source read.
        if len(calls) == 7:
            raise ValueError("canonical source unavailable")
        return original(*args, **kwargs)

    monkeypatch.setattr(source_vectors, "hydrate", hydrate)
    with turn_budget.turn(turn_budget.TurnBudget(wait_s=1)) as budget:
        with pytest.raises(ValueError, match="canonical source"):
            await projection.search("words", contact_id="owner", session_id="session",
                recall_request=RecallRequest.create(runtime, owner=True, turn_id="failed-recall"))
        assert not budget.writes
    assert not runtime.loaded["recall-test"].comparator.records()
    assert driver.queries.get("failed-recall") is None


@pytest.mark.parametrize("outcome", ["failed", "held"])
async def test_host_recall_records_and_trace_require_a_served_section(assembly, runtime, monkeypatch, outcome):
    projection, driver = owner_search(runtime, monkeypatch)
    ledger = assembly[0]
    ledger.record_source("source", contact_id="owner", session_id="session",
                         messages=[{"role": "user", "content": "source text"}])
    hit = ledger.search_sources("source text", contact_id="owner", session_id="session")[0]
    monkeypatch.setattr(source_vectors, "hydrate", lambda *a, **kw: hit)
    monkeypatch.setattr(vector, "_store", projection.store)
    monkeypatch.setattr(vector, "_pipeline", projection.pipeline)
    monkeypatch.setattr(host, "_mind", lambda: SimpleNamespace(plasticity=runtime))
    traces = []
    monkeypatch.setattr(driver, "record_trace", lambda *args: traces.append(args))
    body = request()
    body.incoming_message.content = "source text"
    body.context.turn_id = "recall-turn"
    if outcome == "failed":
        async def failing(*args, **kwargs):
            raise ValueError("selection failed after vector read")
        monkeypatch.setattr(search, "select_memory", failing)
    else:
        body.inline_chars = 1
    response = await host.context_assemble(body)
    assert not response.budget.get("timed_out")
    assert not any(section.id == "colonymind-memory" for section in response.sections)
    await flush()
    assert not runtime.loaded["recall-test"].comparator.records() and traces == []
    assert driver.queries.get("recall-turn") is None
