"""The search tool's own time (docs/RECALL-HYBRID.md, "Recall and time"): ``colonymind_memory_search`` takes
``window`` ({start, end}) and ``as_of`` as the sidecar's search does, so a question in a language the sidecar's date
parser does not read, or one whose span the model already knows, is not held to English date words.

The schema says what both are within the per-request budget. The handler sends them as the sidecar reads them
(timezone-aware instants, in UTC), and names a value the sidecar would refuse back to the model in the form it can
correct, without sending it. Against the sidecar's real search a window finds the span's messages, which share no
word with the question, and ``as_of`` moves what "last week" means.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from conftest import API_KEY, OWNER, build_home, probe, refused
from test_tools_commands import TOOL_CODE


def test_the_search_schema_documents_window_and_as_of_within_the_budget():
    from colonymind_hermes.tools import SEARCH_SCHEMA
    from colonymind_memory.provider import _COLONYMIND_TOOL_SCHEMAS
    from test_overhead_budget import PLUGIN_SCHEMAS, TOOL_BUDGET_CHARS, rendered

    properties = SEARCH_SCHEMA["parameters"]["properties"]
    assert properties["window"] == {"type": "object"} and properties["as_of"] == {"type": "string"}
    assert SEARCH_SCHEMA["parameters"]["required"] == ["query"]
    text = SEARCH_SCHEMA["description"]
    assert "window {start, end}: the span asked about" in text
    assert "as_of: the time relative dates count from" in text and "ISO with offset" in text
    total = sum(len(rendered(schema)) for schema in [*PLUGIN_SCHEMAS, *_COLONYMIND_TOOL_SCHEMAS])
    assert total <= TOOL_BUDGET_CHARS, total


WEEK = {"start": "2026-03-02T00:00:00+01:00", "end": "2026-03-09T00:00:00+01:00"}


def test_a_window_and_as_of_reach_the_sidecar_as_it_reads_them(home, sidecar):
    result = probe(TOOL_CODE + '''
o = owner()
emit(timed=call("colonymind_memory_search", {"query": "de quoi avons-nous parlé ?", "window": %r,
                                            "as_of": "2026-03-10T09:30:00Z"}, o),
     plain=call("colonymind_memory_search", {"query": "plans"}, o),
     blank=call("colonymind_memory_search", {"query": "plans", "window": None, "as_of": ""}, o))
''' % WEEK, home)
    assert result["timed"]["content"] == f"excerpt for {OWNER}"
    timed, plain, blank = sidecar.calls("/v1/host/memory/search", "POST")
    assert timed["json"]["window"] == {"start": "2026-03-01T23:00:00+00:00", "end": "2026-03-08T23:00:00+00:00"}
    assert timed["json"]["as_of"] == "2026-03-10T09:30:00+00:00"
    assert timed["json"]["person_id"] == OWNER and timed["json"]["query"] == "de quoi avons-nous parlé ?"
    # Asked without a time, the search is the one it was: no time field at all, not an empty one.
    assert not {"window", "as_of"} & set(plain["json"]) and not {"window", "as_of"} & set(blank["json"])


@pytest.mark.requires_sidecar
def test_the_sidecars_own_schema_reads_the_same_instants():
    """What the handler sends is what the sidecar's request schema accepts, with the instants the model meant."""
    from colonymind.api.schemas import host as schemas
    from colonymind_hermes.tools import search_time

    fields, problem = search_time({"window": WEEK, "as_of": "2026-03-10T10:30:00+01:00"})
    assert problem is None
    request = schemas.MemorySearchRequest.model_validate({
        "identity": {"host_id": "hermes"}, "person_id": OWNER, "query": "x", **fields})
    assert request.window.start == datetime(2026, 3, 1, 23, tzinfo=timezone.utc)
    assert request.window.end == datetime(2026, 3, 8, 23, tzinfo=timezone.utc)
    assert request.as_of == datetime(2026, 3, 10, 9, 30, tzinfo=timezone.utc)


REFUSED = {
    "local dates": {"window": {"start": "2026-03-02", "end": "2026-03-09"}},     # dates with no offset to place them
    "no offset": {"window": {"start": "2026-03-02T00:00:00", "end": "2026-03-09T00:00:00"}},
    "no end": {"window": {"start": "2026-03-02T00:00:00+00:00"}},
    "a list": {"window": ["2026-03-02T00:00:00Z", "2026-03-09T00:00:00Z"]},
    "words": {"window": {"start": "last week", "end": "now"}},
    "a number": {"window": {"start": 1772409600, "end": 1773014400}},
    "before any instant": {"window": {"start": "0001-01-01T00:00:00+05:00", "end": "2026-03-09T00:00:00Z"}},
    "inverted": {"window": {"start": "2026-03-09T00:00:00Z", "end": "2026-03-02T00:00:00Z"}},
    "empty": {"window": {"start": "2026-03-02T00:00:00Z", "end": "2026-03-02T01:00:00+01:00"}},
    "as_of without offset": {"as_of": "2026-03-10T09:30:00"},
    "as_of a number": {"as_of": 1773135000},
}


def test_a_time_the_sidecar_would_refuse_is_named_back_and_never_sent(home, sidecar):
    """A window or reference time the sidecar would answer with 422 is an argument the model can correct: the answer
    names the valid form (never a final answer, which would end the search), and nothing is sent."""
    result = probe(TOOL_CODE + '''
o = owner()
emit(**{name: call("colonymind_memory_search", {"query": "plans", **args}, o) for name, args in %r.items()})
''' % REFUSED, home)
    for name, answer in result.items():
        assert set(answer) == {"error"}, (name, answer)
        field = "as_of" if name.startswith("as_of") else "window"
        assert refused(answer).startswith(field), (name, answer)
    for name in ("local dates", "no offset", "no end", "a list", "words", "a number", "before any instant"):
        assert "{start, end}" in result[name]["error"] and "UTC offset" in result[name]["error"], name
    for name in ("inverted", "empty"):
        assert "ends after it starts" in result[name]["error"] and "exclusive" in result[name]["error"], name
    assert "UTC offset" in result["as_of without offset"]["error"]
    assert sidecar.calls("/v1/host/memory/search", "POST") == []


def test_a_search_that_cannot_run_this_turn_is_final_whatever_its_window(home, sidecar):
    """A turn with no resolved participant cannot be searched with any window: one final answer, never a correction
    the model would spend another call on only to be told the search cannot run."""
    result = probe(TOOL_CODE + '''
u = guest("unbound-1", sender="", platform="telegram")
emit(search=call("colonymind_memory_search", {"query": "plans", "window": {"start": "2026-03-02"}}, u))
''', home)
    assert "no resolved participant" in refused(result["search"]) and result["search"]["retry"] is False
    assert sidecar.calls("/v1/host/memory/search", "POST") == []


# --- Against the sidecar's real search ------------------------------------------------------------------------------

@pytest.fixture
def search_sidecar(tmp_path, monkeypatch):
    """The sidecar's real host routes on a fresh ledger behind the one key: recall with its lanes, the date-window lane
    and the search's own time. No vector index and no reranker, so only words and time find anything."""
    from fastapi import FastAPI
    import uvicorn
    from colonymind import vector
    from colonymind.api.middleware import ApiKeyMiddleware
    from colonymind.api.routers import host
    from colonymind.turns import TurnIdempotencyLedger

    state = tmp_path / "sidecar"
    state.mkdir()
    for name, value in (("COLONYMIND_STATE_DIR", str(state)), ("COLONYMIND_OWNER_CONTACT_ID", OWNER),
                        ("COLONYMIND_RECALL_RERANK", "off")):
        monkeypatch.setenv(name, value)
    for name in ("_contacts_store", "_telemetry", "_reranker", "_context_recall_selector", "_comms_log"):
        monkeypatch.setattr(host, name, None)
    monkeypatch.setattr(vector, "get_store", lambda: None)
    monkeypatch.setattr(vector, "get_pipeline", lambda: None)
    app = FastAPI()
    app.include_router(host.router)
    app.add_middleware(ApiKeyMiddleware, api_key=API_KEY)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, lifespan="off", access_log=False,
                                           log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    started = time.monotonic()
    while not server.started:
        assert thread.is_alive() and time.monotonic() - started < 10, "the sidecar did not start"
        time.sleep(0.02)
    ledger = TurnIdempotencyLedger(state / "turn-idempotency.db")

    def record(turn_id: str, content: str, at: datetime) -> None:
        ledger.record_source(turn_id, contact_id=OWNER, session_id="earlier", scope="person",
                             occurred_at=at.isoformat(), messages=[{"role": "user", "content": content}],
                             derive_claims=False)

    try:
        yield SimpleNamespace(port=port, url=f"http://127.0.0.1:{port}", record=record)
    finally:
        server.should_exit = True
        thread.join(10)


def iso(at: datetime) -> str:
    return at.astimezone(timezone.utc).isoformat()


@pytest.mark.requires_sidecar
def test_a_window_finds_the_spans_messages_and_as_of_moves_last_week(tmp_path, search_sidecar):
    """What the owner sees: "what did we talk about" shares no word with the week's messages, so only the window
    finds them; "last week" counted from ``as_of`` is the week before it, not the week before today."""
    now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    # A Thursday noon at least three weeks back: in any zone within 12 hours of UTC it is a Wednesday, Thursday or
    # Friday, so the week before it holds the day seven days earlier.
    as_of = now - timedelta(days=21 + (now.weekday() - 3) % 7)
    kiln = as_of - timedelta(days=7)
    search_sidecar.record("kiln", "The kiln reached cone six before lunch.", kiln)
    search_sidecar.record("ferry", "The ferry was late again.", now - timedelta(days=90))
    window = {"start": iso(kiln - timedelta(days=1)), "end": iso(kiln + timedelta(days=1))}
    home = build_home(tmp_path, search_sidecar.port, search_sidecar.url)
    result = probe(TOOL_CODE + '''
u = guest("cli-1", sender="", platform="cli")
emit(windowed=call("colonymind_memory_search", {"query": "What did we talk about?", "window": %r}, u),
     words=call("colonymind_memory_search", {"query": "What did we talk about?"}, u),
     then=call("colonymind_memory_search", {"query": "What did we talk about last week?", "as_of": %r}, u),
     today=call("colonymind_memory_search", {"query": "What did we talk about last week?"}, u))
''' % (window, iso(as_of)), home)
    assert "cone six" in result["windowed"]["content"] and "ferry" not in result["windowed"]["content"]
    assert "cone six" in result["then"]["content"] and "ferry" not in result["then"]["content"]
    for name in ("words", "today"):      # the same question without the time finds nothing, and says so once
        assert "nothing retained matches" in refused(result[name]), (name, result[name])
