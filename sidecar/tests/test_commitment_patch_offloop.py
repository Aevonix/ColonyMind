"""Commitment PATCH stores run in workers and deliver events on the subscriber loop."""

import asyncio
import threading
from datetime import timedelta

import pytest

from colonymind.api.routers import host
from colonymind.events import broadcaster, journal
from colonymind.events.stream import EventSubscriberBuffer
from test_commitment_cadence_cancel import NOW, cadence, store  # noqa: F401
from test_commitment_resolution import _client


@pytest.mark.parametrize("outcome", ["cancelled", "wont_do", "obsolete"])
async def test_patch_cascade_sql_runs_offloop_and_events_reach_loop(store, monkeypatch, outcome):
    first = cadence(store, made=NOW - timedelta(days=2))
    second = cadence(store, made=NOW - timedelta(days=1))
    loop_thread = threading.get_ident()
    queries, publications, deliveries = [], [], []
    connect = store._connect

    def traced():
        db = connect()

        def trace(sql):
            if sql.startswith("SELECT * FROM commitments WHERE person_id="):
                queries.append(("SELECT", threading.get_ident()))
            elif sql.startswith("UPDATE commitments SET status='cancelled',metadata="):
                queries.append(("UPDATE", threading.get_ident()))

        db.set_trace_callback(trace)
        return db

    monkeypatch.setattr(store, "_connect", traced)
    subscriber = EventSubscriberBuffer(4, loop=asyncio.get_running_loop())
    publish_on_loop = subscriber._publish_on_loop

    def delivered(event):
        deliveries.append(threading.get_ident())
        publish_on_loop(event)

    append = journal.append_event_record

    def published(event_type, payload, **kwargs):
        publications.append((payload["commitment_id"], threading.get_ident()))
        return append(event_type, payload, **kwargs)

    monkeypatch.setattr(subscriber, "_publish_on_loop", delivered)
    monkeypatch.setattr(host, "_event_subscribers", [subscriber])
    monkeypatch.setattr(broadcaster, "_broadcast_fn", host.broadcast_event)
    monkeypatch.setattr(broadcaster, "_subscribers", {})
    monkeypatch.setattr(journal, "append_event_record", published)
    body = {"status": "cancelled"} if outcome == "cancelled" else {"outcome": outcome}
    async with _client(store) as client:
        response = await client.patch("/v1/host/commitments/" + second["id"], json=body)
    assert response.status_code == 200
    assert response.json()["id"] == second["id"] and response.json()["status"] == "cancelled"
    assert store.get(first["id"])["status"] == "cancelled"
    assert [kind for kind, _ in queries] == ["SELECT", "UPDATE"]
    assert all(thread != loop_thread for _, thread in queries)
    events = [await asyncio.wait_for(subscriber.get(), 1) for _ in range(2)]
    assert subscriber.queue.empty()
    assert deliveries == [loop_thread, loop_thread]
    assert {event["payload"]["commitment_id"] for event in events} == {first["id"], second["id"]}
    assert all(event["type"] == "commitment.cancelled" for event in events)
    assert events[0]["seq"] < events[1]["seq"]
    assert len(publications) == 2
    for ident, thread in publications:
        assert (thread == loop_thread) == (outcome == "cancelled" and ident == second["id"])


@pytest.mark.parametrize("body", [{"status": "cancelled"}, {"outcome": "wont_do"}, {"outcome": "obsolete"}])
async def test_patch_missing_commitment_still_returns_404(store, body):
    async with _client(store) as client:
        response = await client.patch("/v1/host/commitments/missing", json=body)
    assert response.status_code == 404
    assert response.json() == {"detail": "Commitment not found"}


@pytest.mark.parametrize("method, body", [("update", {"status": "cancelled"}), ("resolve", {"outcome": "obsolete"})])
async def test_patch_worker_value_error_still_returns_422(store, monkeypatch, method, body):
    threads = []

    def rejected(*args, **kwargs):
        threads.append(threading.get_ident())
        raise ValueError("invalid commitment change")

    monkeypatch.setattr(store, method, rejected)
    async with _client(store) as client:
        response = await client.patch("/v1/host/commitments/rejected", json=body)
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid commitment change"}
    assert len(threads) == 1 and threads[0] != threading.get_ident()
