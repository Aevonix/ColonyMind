"""Route handlers that never awaited and are safe next to other threads run in the framework's thread pool.

Each one is sent through the app in process: the call into its store runs on another thread than the event
loop's, and the answer is the one the handler always gave.
"""
from __future__ import annotations

from importlib.resources import files
import threading
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from colonymind import vector
from colonymind.api.auth import KEY_PRINCIPAL, key_authority
from colonymind.api.routers import host
from colonymind.beliefs.source_projection import SourceClaimProjection
from colonymind.commitments.store import CommitmentStore
from colonymind.feedback.store import TypeFeedbackStore
from colonymind.mind import ideas_presentation
from colonymind.turns import TurnIdempotencyLedger
from colonymind.turns.media import SourceMedia

ON_THE_LOOP = "the handler called its store on the event loop's thread"
PERSON = "person"
REPORT = "Machine-authored report. The archive digest matched its receipt. Verified at 09:14."
NOTE = "The digest comparison is supported. The verification time was not measured."
PROMISE = "Send the build recap"
ICON = "packet-money-v1.svg"


@pytest.fixture
def world(monkeypatch, tmp_path):
    """The stores behind the handlers, in a directory of the test, and an app whose requests carry the key."""
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(vector, "get_store", lambda: None)
    monkeypatch.setattr(vector, "get_pipeline", lambda: None)
    commitments = CommitmentStore(db_path=tmp_path / "commitments.db")
    feedback = TypeFeedbackStore(db_path=str(tmp_path / "feedback.db"))
    monkeypatch.setattr(host, "_commitment_store", commitments)
    monkeypatch.setattr(host, "_feedback_store", feedback)
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("report", contact_id=PERSON, session_id="work",
                         messages=[{"role": "assistant", "content": REPORT}])
    application = FastAPI()

    @application.middleware("http")
    async def _key(request, call_next):
        request.state.colonymind_authority = key_authority()
        return await call_next(request)

    application.include_router(host.router)
    yield SimpleNamespace(app=application, commitments=commitments, feedback=feedback, ledger=ledger)
    feedback._conn.close()


class Watched:
    """Records the thread of every call of one function that a handler uses."""

    def __init__(self, monkeypatch, owner, name):
        self.threads = []
        call = getattr(owner, name)

        def watched(*args, **kwargs):
            self.threads.append(threading.get_ident())
            return call(*args, **kwargs)

        monkeypatch.setattr(owner, name, watched)


def _case(watch, method, path, status, answer, **request):
    """One handler: the store call to watch, the request, and ``answer(response)`` as (seen, expected)."""
    return SimpleNamespace(watch=watch, request=(method, path), kwargs=request, status=status, answer=answer)


def _json(expected):
    return lambda response: (response.json(), expected)


def _commitment(stored):
    return {"id": stored["id"], "person_id": PERSON, "description": PROMISE, "made_at": stored["made_at"],
            "due_at": None, "fulfilled_at": None, "status": "pending", "source_type": "manual",
            "source_context": None, "priority": 50, "metadata": None, "deduped": False}


# The commitment store: one lock, a connection for each call. A write that also emits an event, a read, and
# a refused delete (the delete, then the read that words the refusal).

def _create_commitment(world):
    def answer(response):
        return response.json(), _commitment(world.commitments.get(response.json()["id"]))

    return _case((world.commitments, "create"), "POST", "/v1/host/commitments", 201, answer,
                 json={"person_id": PERSON, "description": PROMISE})


def _get_commitment(world):
    made = world.commitments.create(person_id=PERSON, description=PROMISE)
    return _case((world.commitments, "get"), "GET", "/v1/host/commitments/" + made["id"], 200,
                 _json(_commitment(made)))


def _delete_commitment(world):
    made = world.commitments.create(person_id=PERSON, description=PROMISE)
    return _case((world.commitments, "delete"), "DELETE", "/v1/host/commitments/" + made["id"], 409,
                 _json({"detail": "Cannot delete commitment in 'pending' state. Cancel it first."}))


# The source ledger: a connection for each call. A write in one transaction, three reads, and a media read.

def _source_ref(world):
    return world.ledger.source_references(["report"], contact_id=PERSON, session_id="later")[0]


def _append_source_annotation(world):
    ref = _source_ref(world)
    note = {"contact_id": PERSON, "session_id": "later", "annotation_id": "audit-1", **ref,
            "excerpt": "Verified at 09:14.", "correction": NOTE}

    def answer(response):
        # The same note again is the ledger's own account of the one that was stored.
        again = world.ledger.append_source_annotation(**note, author_principal=KEY_PRINCIPAL)
        return response.json(), {**again, "accepted": True, "created": True, "target": ref}

    return _case((TurnIdempotencyLedger, "append_source_annotation"), "POST",
                 "/v1/host/memory/sources/annotations", 200, answer, json=note)


FEED = {"contact_id": PERSON, "head": 0, "through": 0, "events": [], "complete": True}


def _source_erasure_feed(world):
    return _case((TurnIdempotencyLedger, "erasure_feed"), "GET", "/v1/host/memory/sources/erasures", 200,
                 _json(FEED), params={"contact_id": PERSON})


def _source_freshness_feed(world):
    check = {"contact_id": PERSON, "session_id": "later", "source_refs": [_source_ref(world)]}
    return _case((host, "_source_freshness_feed"), "POST", "/v1/host/memory/sources/erasures", 200,
                 _json({**FEED, "sources_current": True}), json=check)


def _source_claim_projection_status(world):
    semantic = {"index_state": "unverified_or_unavailable", "active_generation": None, "source_turns": 1,
                "projected_turns": 0, "pending_turns": 1, "failed_turns": 0}
    return _case((SourceClaimProjection, "status"), "GET", "/v1/host/memory/sources/claims/status", 200,
                 _json({"sources": [], "media": [], "semantic": semantic}), params={"contact_id": PERSON})


def _read_source_asset(world):
    return _case((SourceMedia, "read"), "GET", "/v1/host/memory/sources/assets/" + "0" * 64, 404,
                 _json({"detail": "unknown source asset"}), params={"contact_id": PERSON, "session_id": "work"})


# The feedback store: one connection for all threads, every use of it under the store's lock.

def _get_type_feedback(world):
    world.feedback.record("checkin", "preferred", source="initiative-1")
    [row] = world.feedback.snapshot()
    assert {**row, "updated_at": None} == {"itype": "checkin", "multiplier": 1.12, "actioned": 0, "dismissed": 0,
                                           "ignored": 0, "other": 1, "updated_at": None}
    return _case((world.feedback, "snapshot"), "GET", "/v1/host/feedback", 200,
                 _json({"available": True, "types": [row]}))


# A file of the package.

def _idea_media(world):
    icon = files("colonymind.mind").joinpath("idea_assets", ICON).read_bytes()
    headers = {"content-type": "image/svg+xml", "cache-control": "private, max-age=86400",
               "x-content-type-options": "nosniff",
               "content-security-policy": "default-src 'none'; style-src 'none'; sandbox"}

    def answer(response):
        return (response.content, {name: response.headers[name] for name in headers}), (icon, headers)

    return _case((ideas_presentation, "media"), "GET", "/v1/host/initiatives/media/" + ICON, 200, answer)


CASES = {case.__name__[1:]: case for case in (
    _create_commitment, _get_commitment, _delete_commitment, _append_source_annotation, _source_erasure_feed,
    _source_freshness_feed, _source_claim_projection_status, _read_source_asset, _get_type_feedback,
    _idea_media)}


async def _send(world, case):
    async with AsyncClient(transport=ASGITransport(app=world.app), base_url="http://test") as client:
        return await client.request(*case.request, **case.kwargs)


@pytest.mark.parametrize("name", sorted(CASES))
async def test_handler_calls_its_store_off_the_loop_thread(world, monkeypatch, name):
    case = CASES[name](world)
    watched = Watched(monkeypatch, *case.watch)
    response = await _send(world, case)
    assert response.status_code == case.status, response.text
    assert watched.threads, "the handler never reached its store"
    assert threading.get_ident() not in watched.threads, ON_THE_LOOP


@pytest.mark.parametrize("name", sorted(CASES))
async def test_handler_answers_as_before(world, name):
    case = CASES[name](world)
    response = await _send(world, case)
    assert response.status_code == case.status, response.text
    seen, expected = case.answer(response)
    assert seen == expected
