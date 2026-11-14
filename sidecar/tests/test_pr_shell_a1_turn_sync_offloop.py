"""A turn sync does its blocking storage steps in a worker thread.

While one sync is inside such a step the event loop answers other requests and the timeout of context assembly
can fire; the steps of one sync run in the order they always did; the steps of two syncs stay apart; and a
request that is cancelled during a step leaves no reservation open.
"""
from __future__ import annotations

import asyncio
from contextlib import closing
import contextvars
import sqlite3
import threading
from types import SimpleNamespace

import anyio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from colonymind import vector
from colonymind.api.credentials import CredentialTable
from colonymind.api.errors import install_exception_handlers
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import host, transport_ingress_api
from colonymind.contacts.comms import CommsLog
from colonymind.events import journal
from colonymind.memory import blocks, curated, turn_budget
from colonymind.secrets import seal
from colonymind.turns import TurnIdempotencyLedger
from colonymind.util import session_safety
from test_observed_conversations import CONVERSATION, KEY, OWNER, Contacts, body as observed_body

# How long a held step waits for its release: a run in which the loop is stopped ends by itself.
BOUND = 10.0
# Answers from memory alone: no store, no file, no thread.
CHEAP = "/v1/host/memory/embed/async/none"
NOT_ANSWERED = "the loop answered only after the held storage step had ended"

# The storage steps of one keyed turn (user and assistant message, a summary, no contacts store), stretch by
# stretch: between two stretches the turn sync awaits, and only there may another turn sync's steps run.
STRETCHES = (
    ("seal.redact", "ledger.is_source_erased", "erasure_waits", "ledger.reserve"),
    ("seal.redact",),
    ("ledger.is_source_erased", "ledger.retained_messages", "ledger.record_source", "ledger.is_projection_erased",
     "curated.is_forgotten", "seal.commit", "session.save_last_user_message_at", "journal.append_event",
     "comms.source_input", "comms.log", "comms.log", "ledger.complete", "transport.reconcile_source"),
)

_SYNC = contextvars.ContextVar("pr_shell_a1_sync", default=None)


def _payload(turn_id="turn-001", *, session="session-1", contact="contact-1", channel="test:thread-1",
             text="Remember alpha"):
    return {
        "identity": {"host_id": "test-host", "instance_id": "instance-a"},
        "context": {"session_id": session, "contact_id": contact, "channel_id": channel, "turn_id": turn_id},
        "topics": ["alpha"],
        "summary": "Asked to remember: " + text,
        "user_message": {"role": "user", "content": text},
        "assistant_message": {"role": "assistant", "content": "I will."},
    }


@pytest.fixture
def stores(monkeypatch, tmp_path):
    """Binds the turn sync's stores (source ledger, journal, communications log) to a directory of the test."""
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    blocks.reset_settings()
    for name in ("_contacts_store", "_telemetry", "_preference_learner"):
        monkeypatch.setattr(host, name, None)
    bound = []

    def bind(name="state"):
        directory = tmp_path / name
        directory.mkdir()
        monkeypatch.setenv("COLONYMIND_STATE_DIR", str(directory))
        ledger = TurnIdempotencyLedger(directory / "turn-idempotency.db")
        comms = CommsLog(str(directory / "comms.db"), source_ledger=ledger)
        monkeypatch.setattr(host, "_comms_log", comms)
        bound.append(SimpleNamespace(ledger=ledger, comms=comms))
        return bound[-1]

    yield bind
    for state in bound:
        state.comms._conn.close()
    blocks.reset_settings()


@pytest.fixture
def app() -> FastAPI:
    application = FastAPI()
    application.include_router(host.router)
    application.include_router(host.v2_router)
    return application


@pytest.fixture
def observed_app(monkeypatch, tmp_path):
    """The app of the observed conversation tests: the owner's credential in front of both routers."""
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_OWNER_PERSON_ID", OWNER)
    monkeypatch.setattr(vector, "get_store", lambda: None)
    monkeypatch.setattr(vector, "get_pipeline", lambda: None)
    for name in ("_telemetry", "_reranker", "_context_recall_selector", "_comms_log", "_affect_store",
                 "_commitment_store"):
        monkeypatch.setattr(host, name, None)
    monkeypatch.setattr(host, "_contacts_store", Contacts({"c-dana"}))
    application = FastAPI()
    install_exception_handlers(application)
    application.include_router(host.router)
    application.include_router(host.v2_router)
    application.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=CredentialTable.from_config({
        "api-key": {"owner": ["companion"], "contacts": "all"}}))
    from colonymind.turns import projection_backlog
    monkeypatch.setattr(projection_backlog, "configured_per_hour", lambda: float("inf"))
    return application


def _client(application, **kwargs):
    return AsyncClient(transport=ASGITransport(app=application), base_url="http://test", **kwargs)


class Held:
    """One storage step held open: its first call waits for ``release``, for at most ``BOUND`` seconds."""

    def __init__(self, monkeypatch, owner, name):
        self.started, self.release, self.waits = threading.Event(), threading.Event(), []
        step = getattr(owner, name)

        def held(*args, **kwargs):
            if not self.started.is_set():
                self.started.set()
                # False when nothing released the step before the bound: the loop was not there to do it.
                self.waits.append(self.release.wait(BOUND))
            return step(*args, **kwargs)

        monkeypatch.setattr(owner, name, held)


async def _answered_while_held(held, client):
    """Whether the cheap route answers while the turn sync is inside the held step; the step is then released."""
    assert await asyncio.to_thread(held.started.wait, BOUND), "the turn sync never reached the held step"
    cheap = await asyncio.wait_for(client.get(CHEAP), BOUND)
    in_time = not held.waits
    held.release.set()
    assert cheap.status_code == 200 and cheap.json() == {"task_id": "none", "status": "running"}
    return in_time


@pytest.mark.asyncio
async def test_the_loop_answers_while_a_v1_turn_sync_is_held_in_its_storage_work(app, stores, monkeypatch):
    stores()
    held = Held(monkeypatch, TurnIdempotencyLedger, "reserve")
    async with _client(app) as client:
        sync = asyncio.create_task(client.post("/v1/host/turns/sync", json=_payload()))
        answered = await _answered_while_held(held, client)
        response = await asyncio.wait_for(sync, BOUND)
    assert response.status_code == 200 and response.json()["source_recorded"] is True
    assert answered and held.waits == [True], NOT_ANSWERED


@pytest.mark.asyncio
async def test_the_loop_answers_while_a_v2_turn_sync_is_held_in_its_storage_work(app, stores, monkeypatch):
    stores()
    held = Held(monkeypatch, TurnIdempotencyLedger, "record_source")
    async with _client(app) as client:
        sync = asyncio.create_task(client.put("/v2/host/turns/turn-001", json=_payload()))
        answered = await _answered_while_held(held, client)
        response = await asyncio.wait_for(sync, BOUND)
    assert (response.status_code, response.headers["Idempotency-Status"]) == (201, "created")
    assert answered and held.waits == [True], NOT_ANSWERED


@pytest.mark.asyncio
async def test_the_loop_answers_while_an_observed_conversation_sync_is_held_in_its_storage_work(
        observed_app, monkeypatch):
    held = Held(monkeypatch, TurnIdempotencyLedger, "reserve")
    async with _client(observed_app, headers={"Authorization": "Bearer " + KEY}) as client:
        sync = asyncio.create_task(client.put(f"/v2/host/conversations/observed/{CONVERSATION}",
                                              json=observed_body()))
        answered = await _answered_while_held(held, client)
        response = await asyncio.wait_for(sync, BOUND)
    assert (response.status_code, response.headers["Idempotency-Status"]) == (201, "created")
    assert answered and held.waits == [True], NOT_ANSWERED


@pytest.mark.asyncio
async def test_the_assembly_timeout_fires_while_a_turn_sync_is_held_in_its_storage_work(app, stores, monkeypatch):
    stores()
    assembling = asyncio.Event()

    async def never(body, request):
        assembling.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(host, "_context_assemble", never)
    held = Held(monkeypatch, TurnIdempotencyLedger, "reserve")
    async with _client(app) as client:
        assembly = asyncio.create_task(client.post("/v1/host/context/assemble", json={
            "identity": {"host_id": "test-host"}, "context": {"contact_id": "contact-1", "session_id": "session-1"},
            "incoming_message": {"role": "user", "content": "alpha"}, "wait_ms": 300}))
        await asyncio.wait_for(assembling.wait(), BOUND)
        # The host waits 300 ms for its context; the turn sync's step is held far longer than that.
        sync = asyncio.create_task(client.post("/v1/host/turns/sync", json=_payload()))
        assert await asyncio.to_thread(held.started.wait, BOUND), "the turn sync never reached the held step"
        answer = await asyncio.wait_for(assembly, BOUND)
        fired_while_held = not held.waits
        held.release.set()
        assert (await asyncio.wait_for(sync, BOUND)).status_code == 200
        await asyncio.gather(*tuple(turn_budget._PENDING))
    assert answer.status_code == 200 and answer.json()["sections"] == []
    assert answer.json()["budget"]["timed_out"] is True and answer.json()["budget"]["wait_ms"] == 300
    assert fired_while_held and held.waits == [True], \
        "the assembly timeout fired only after the held storage step had ended"


class Steps:
    """The storage steps of turn syncs as they ran: (sync, step, "begins" or "ends") for each outermost call."""

    WRAPPED = (
        ("seal.redact", seal, "redact_sensitive_turn"),
        ("seal.commit", seal.TurnSeal, "commit"),
        ("erasure_waits", host, "_erasure_waits"),
        ("ledger.is_source_erased", TurnIdempotencyLedger, "is_source_erased"),
        ("ledger.resolve_input_dependencies", TurnIdempotencyLedger, "resolve_input_dependencies"),
        ("ledger.reserve", TurnIdempotencyLedger, "reserve"),
        ("ledger.retained_messages", TurnIdempotencyLedger, "retained_messages"),
        ("ledger.record_source", TurnIdempotencyLedger, "record_source"),
        ("ledger.is_projection_erased", TurnIdempotencyLedger, "is_projection_erased"),
        ("ledger.complete", TurnIdempotencyLedger, "complete"),
        ("ledger.mark_ambiguous", TurnIdempotencyLedger, "mark_ambiguous"),
        ("ledger.release", TurnIdempotencyLedger, "release"),
        ("curated.is_forgotten", curated, "is_forgotten"),
        ("session.save_last_user_message_at", session_safety, "save_last_user_message_at"),
        ("journal.append_event", journal, "append_event"),
        ("comms.source_input", CommsLog, "source_input"),
        ("comms.log", CommsLog, "log"),
        ("transport.reconcile_source", transport_ingress_api, "reconcile_source"),
    )

    def __init__(self, monkeypatch):
        self.ran, self._depth = [], threading.local()
        for name, owner, attribute in self.WRAPPED:
            monkeypatch.setattr(owner, attribute, self._recorded(name, getattr(owner, attribute)))

    def _recorded(self, name, call):
        def step(*args, **kwargs):
            sync, depth = _SYNC.get(), getattr(self._depth, "value", 0)
            outermost = sync is not None and depth == 0
            if outermost:
                self.ran.append((sync, name, "begins"))
            self._depth.value = depth + 1
            try:
                return call(*args, **kwargs)
            finally:
                self._depth.value = depth
                if outermost:
                    self.ran.append((sync, name, "ends"))
        return step

    def of(self, sync):
        return [name for tag, name, edge in self.ran if tag == sync and edge == "begins"]

    def stretches(self, sync):
        """Where each stretch of ``sync`` begins and ends in the record of all steps."""
        places = [place for place, (tag, _, _) in enumerate(self.ran) if tag == sync]
        spans, done = [], 0
        for stretch in STRETCHES:
            spans.append((places[2 * done], places[2 * (done + len(stretch)) - 1]))
            done += len(stretch)
        return spans


async def _sync(client, turn_id, **changes):
    """One keyed turn sync whose storage steps are recorded under its turn id."""
    _SYNC.set(turn_id)
    try:
        return await client.put(f"/v2/host/turns/{turn_id}", json=_payload(turn_id, **changes))
    finally:
        _SYNC.set(None)


def _stored(state):
    """What turn syncs left in the stores, without the clock's values and without an order between syncs."""
    with closing(sqlite3.connect(state.ledger.db_path)) as conn:
        ingested = conn.execute("SELECT turn_id, content_sha256, state, response_json, error FROM turn_ingestion "
                                "ORDER BY turn_id").fetchall()
        sources = conn.execute("SELECT turn_id, content_sha256, contact_id, session_id, scope, messages_json "
                               "FROM turn_sources ORDER BY turn_id").fetchall()
    said = state.comms._conn.execute("SELECT contact_id, channel, direction, summary, session_id "
                                     "FROM communications ORDER BY session_id, summary, direction").fetchall()
    turns = sorted(repr((event.get("type"), event.get("data")))
                   for event in journal.replay_events(types=["conversation.turn"])["events"])
    return ingested, sources, said, turns


def _answers(responses):
    return [(response.status_code, response.headers["Idempotency-Status"], response.json()) for response in responses]


@pytest.mark.asyncio
async def test_one_turn_sync_runs_its_storage_steps_in_the_order_they_always_ran(app, stores, monkeypatch):
    state = stores()
    steps = Steps(monkeypatch)
    async with _client(app) as client:
        response = await _sync(client, "turn-order")
    assert (response.status_code, response.headers["Idempotency-Status"]) == (201, "created")
    assert steps.of("turn-order") == [step for stretch in STRETCHES for step in stretch]
    assert all(edge == ("begins", "ends")[place % 2] for place, (_, _, edge) in enumerate(steps.ran))
    assert state.ledger.get("turn-order")["state"] == "completed"


@pytest.mark.parametrize("sessions", ["same", "different"])
@pytest.mark.asyncio
async def test_two_turn_syncs_sent_together_keep_their_steps_apart(app, stores, monkeypatch, sessions):
    other = {"text": "Remember beta"}
    if sessions == "different":
        other.update(session="session-2", contact="contact-2", channel="test:thread-2")
    apart = stores("apart")
    async with _client(app) as client:
        one_by_one = [await _sync(client, "turn-a"), await _sync(client, "turn-b", **other)]
    expected = _stored(apart), _answers(one_by_one)
    assert [row[2] for row in expected[0][0]] == ["completed", "completed"] and len(expected[0][2]) == 4

    together = stores("together")
    steps = Steps(monkeypatch)
    async with _client(app) as client:
        both = await asyncio.wait_for(
            asyncio.gather(_sync(client, "turn-a"), _sync(client, "turn-b", **other)), BOUND)
    for sync in ("turn-a", "turn-b"):
        assert steps.of(sync) == [step for stretch in STRETCHES for step in stretch]
        for begins, ends in steps.stretches(sync):
            inside = steps.ran[begins:ends + 1]
            assert all(tag == sync for tag, _, _ in inside), f"a step of the other turn sync ran inside: {inside}"
    assert (_stored(together), _answers(both)) == expected


async def _cancelled_while_held(app, monkeypatch, held, how):
    """Cancel one keyed turn sync while it is inside the held step, release the step, and wait for the end."""
    async with _client(app) as client:
        if how == "task":
            # As a server cancels the task of a request.
            sync = asyncio.create_task(client.put("/v2/host/turns/turn-gone", json=_payload("turn-gone")))
            assert await asyncio.to_thread(held.started.wait, BOUND), "the turn sync never reached the held step"
            sync.cancel()
            held.release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(sync, BOUND)
            return
        # As a middleware cancels the scope of a request whose client went away.
        with anyio.CancelScope() as scope:
            async with anyio.create_task_group() as group:
                group.start_soon(lambda: client.put("/v2/host/turns/turn-gone", json=_payload("turn-gone")))
                with anyio.CancelScope(shield=True):
                    assert await asyncio.to_thread(held.started.wait, BOUND), "the held step was never reached"
                scope.cancel()
                held.release.set()
        assert scope.cancel_called


@pytest.mark.parametrize("how", ["task", "scope"])
@pytest.mark.asyncio
async def test_a_turn_sync_cancelled_in_its_reserving_step_marks_the_reservation_at_its_next_await(
        app, stores, monkeypatch, how):
    """The step runs to its end in its thread; the cancellation lands where the sync next waits, as before."""
    state = stores()
    channel = host._ensure_channel_id

    async def waiting(*args, **kwargs):
        await asyncio.sleep(0)
        return await channel(*args, **kwargs)

    monkeypatch.setattr(host, "_ensure_channel_id", waiting)
    steps = Steps(monkeypatch)
    held = Held(monkeypatch, TurnIdempotencyLedger, "reserve")
    token = _SYNC.set("turn-gone")
    try:
        await _cancelled_while_held(app, monkeypatch, held, how)
    finally:
        _SYNC.reset(token)
    assert held.waits == [True]
    assert steps.of("turn-gone") == [*STRETCHES[0], *STRETCHES[1], "ledger.mark_ambiguous"]
    assert state.ledger.get("turn-gone")["state"] == "ambiguous"


@pytest.mark.parametrize("how", ["task", "scope"])
@pytest.mark.asyncio
async def test_a_turn_sync_cancelled_in_its_last_stretch_completes_its_reservation(app, stores, monkeypatch, how):
    """No await follows in this stretch: the steps run to the end, as they did when nothing could interrupt them."""
    state = stores()
    steps = Steps(monkeypatch)
    held = Held(monkeypatch, TurnIdempotencyLedger, "record_source")
    token = _SYNC.set("turn-gone")
    try:
        await _cancelled_while_held(app, monkeypatch, held, how)
    finally:
        _SYNC.reset(token)
    assert held.waits == [True]
    assert steps.of("turn-gone") == [step for stretch in STRETCHES for step in stretch]
    assert state.ledger.get("turn-gone")["state"] == "completed"
    assert [row[2] for row in _stored(state)[2]] == ["in", "out"]
