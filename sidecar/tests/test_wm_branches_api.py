"""Branch routes bind owner access to the credential and the session sender."""
import asyncio
import json
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from onekey import KEY
from colonymind.api.credentials import CredentialTable, secret_digest
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import branches, host


GUEST_KEY = "guest-branch-fixture-key-0123456789"
HANDLE = "br-012345abcdef"
OWNER_SENDER = {"platform": "chat", "user_id": "owner-handle"}


class Contacts:
    async def resolve_messaging_handle(self, platform, user_id):
        known = {"owner-handle": "owner", "sam-handle": "sam"}
        return SimpleNamespace(contact_id=known[user_id]) if user_id in known else None


class Store:
    enabled = True
    withdrawn = False
    summary_only = False

    def __init__(self):
        self.calls = []

    def capture(self, **kwargs):
        self.calls.append(("capture", kwargs))
        return {"handle": HANDLE, "audience": "owner"}

    def list_branches(self, **kwargs):
        self.calls.append(("list", kwargs))
        return [{"handle": HANDLE, "brief": "Read the worker material."}]

    def session_branch(self, **kwargs):
        self.calls.append(("session", kwargs))
        return {"handle": HANDLE}

    async def consult(self, **kwargs):
        self.calls.append(("consult", kwargs))
        return {"answer": "Sam reports to Robin.", "handles": [HANDLE]}

    def reaction(self, **kwargs):
        self.calls.append(("react", kwargs))
        return {"labelled": True}


@pytest.fixture
def setup(monkeypatch):
    store = Store()
    monkeypatch.setattr(branches, "_store", lambda: store)
    monkeypatch.setattr(host, "_contacts_store", Contacts())
    monkeypatch.setattr(host, "_llm_router", object())
    app = FastAPI()
    app.include_router(branches.router)
    app.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=CredentialTable.from_config({
        "api-key": {"owner": ["chat", "cli"], "contacts": "all"},
        "guest": {"sha256": secret_digest(GUEST_KEY), "owner": [], "contacts": "all"},
    }))
    return app, store


async def request(app, path="", *, body=None, params=None, key=KEY):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test",
                           headers={"Authorization": "Bearer " + key}) as client:
        if body is None:
            return await client.get("/v1/memory/branches" + path, params=params)
        return await client.post("/v1/memory/branches" + path, json=body)


def body(**values):
    return {"requester": "owner", "sender": OWNER_SENDER, **values}


@pytest.mark.asyncio
async def test_owner_capture_list_session_and_consult_use_resolved_identity(setup):
    app, store = setup
    response = await request(app, body=body(parent_session="parent", child_session="child",
        messages=[{"role": "tool", "content": "Sam reports to Robin."}], brief="Read the material.",
        summary="The worker finished."))
    assert response.status_code == 200, response.text
    assert response.json()["branch"]["handle"] == HANDLE
    response = await request(app, params={"session_id": "parent", "requester": "owner", **OWNER_SENDER})
    assert response.status_code == 200, response.text
    assert response.json()["branches"][0]["handle"] == HANDLE
    response = await request(app, "/session", body=body(session_id="parent"))
    assert response.status_code == 200, response.text
    response = await request(app, "/consult", body=body(
        session_id="parent", handles=[HANDLE], question="Who supervises Sam?"))
    assert response.status_code == 200, response.text
    assert response.json()["answer"] == "Sam reports to Robin."
    assert [name for name, _ in store.calls] == ["capture", "list", "session", "consult"]
    assert all(values["requester"] == "owner" for _, values in store.calls)
    assert store.calls[-1][1]["router"] is host._llm_router


@pytest.mark.asyncio
@pytest.mark.parametrize("sender,requester,key", [
    ({"platform": "chat", "user_id": "sam-handle"}, "owner", KEY),
    ({"platform": "chat", "user_id": "unknown-handle"}, "owner", KEY),
    ({"platform": "chat", "user_id": "owner-handle"}, "sam", KEY),
    ({"platform": "ungranted", "user_id": "owner-handle"}, "owner", KEY),
    ({"platform": "chat", "user_id": "owner-handle"}, "owner", GUEST_KEY),
    ({"platform": "chat", "user_id": ""}, "owner", KEY),
])
async def test_claimed_owner_never_overrides_sender_or_credential(setup, sender, requester, key):
    app, store = setup
    common = {"requester": requester, "sender": sender}
    responses = [
        await request(app, params={"requester": requester, "session_id": "parent", **sender}, key=key),
        await request(app, "/consult", body={**common, "session_id": "parent", "handles": [HANDLE],
                                              "question": "Read private material."}, key=key),
        await request(app, body={**common, "parent_session": "parent", "child_session": "child",
                                "messages": [{"role": "assistant", "content": "private"}]}, key=key),
        await request(app, "/session", body={**common, "session_id": "parent"}, key=key),
    ]
    assert [response.status_code for response in responses] == [403] * len(responses)
    assert store.calls == []


@pytest.mark.asyncio
async def test_cli_requires_its_credential_owner_platform(setup):
    app, store = setup
    response = await request(app, params={"session_id": "parent", "requester": "owner",
                                          "platform": "cli", "user_id": ""})
    assert response.status_code == 200, response.text
    assert store.calls[0][0] == "list"


@pytest.mark.asyncio
async def test_empty_session_is_only_a_tool_availability_probe(setup):
    app, store = setup
    response = await request(app, params={"session_id": "", "requester": "owner", **OWNER_SENDER})
    assert response.status_code == 200, response.text
    assert response.json() == {"branches": [], "enabled": True, "available": True, "tool_available": True}
    store.withdrawn = True
    response = await request(app, params={"session_id": "", "requester": "owner", **OWNER_SENDER})
    assert response.json()["tool_available"] is False
    assert store.calls == []


@pytest.mark.asyncio
async def test_faculty_off_and_summary_arm_refuse_consult_before_model_call(setup, monkeypatch, tmp_path):
    from colonymind.memory.branches import BranchStore
    from colonymind.turns.idempotency import TurnIdempotencyLedger
    app, _ = setup
    store = BranchStore(TurnIdempotencyLedger(tmp_path / "turn-idempotency.db"), key=b"fixture-branch-key")
    monkeypatch.setattr(branches, "_store", lambda: store)
    store.enabled = False
    response = await request(app, params={"session_id": "parent", "requester": "owner", **OWNER_SENDER})
    assert response.status_code == 200
    assert response.json()["branches"] == []
    response = await request(app, body=body(parent_session="parent", child_session="child",
                                           messages=[{"role": "assistant", "content": "result"}]))
    assert response.status_code == 409
    store.enabled = True
    store.summary_only = True
    response = await request(app, "/consult", body=body(
        session_id="parent", handles=[HANDLE], question="Read the material."))
    assert response.status_code == 409
    assert store.report()["refusals"] == {"disabled": 1, "summary_only": 1}


@pytest.mark.asyncio
async def test_untrusted_fields_and_invalid_handles_are_refused_before_store(setup):
    app, store = setup
    response = await request(app, "/consult", body=body(
        session_id="parent", handles=[HANDLE], question="Read the material.", tools=[{"name": "run"}]))
    assert response.status_code == 422
    response = await request(app, "/consult", body=body(
        session_id="parent", handles=["not-a-branch"], question="Read the material."))
    assert response.status_code == 422
    assert store.calls == []


@pytest.mark.asyncio
async def test_owner_reaction_passes_raw_turn_to_existing_reader(setup):
    app, store = setup
    response = await request(app, "/reaction", body=body(
        session_id="parent", text="That was useful.", turn_id="owner-turn"))
    assert response.status_code == 200, response.text
    assert store.calls[-1] == ("react", {"requester": "owner", "session_id": "parent",
                                      "text": "That was useful.", "turn_id": "owner-turn",
                                      "reference_text": None})


@pytest.mark.asyncio
async def test_explicit_reference_reaction_is_separate_owner_evidence(setup):
    app, store = setup
    response = await request(app, "/reaction", body=body(session_id="parent", text="Not useful.",
        turn_id="owner-turn", reference_text="The summary answer was useful."))
    assert response.status_code == 200, response.text
    assert store.calls[-1][1]["reference_text"] == "The summary answer was useful."


def test_consult_is_a_configurable_router_task():
    from colonymind.router.functions import TASK_ROLES
    assert TASK_ROLES["consult"] == "reasoning"


@pytest.mark.asyncio
async def test_real_store_capture_consult_refusals_and_reaction(setup, monkeypatch, tmp_path):
    from colonymind.memory.branches import BranchStore
    from colonymind.turns.idempotency import TurnIdempotencyLedger
    app, _ = setup
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    store = BranchStore(ledger, key=b"fixture-branch-key", hourly_budget=1)
    monkeypatch.setattr(branches, "_store", lambda: store)
    monkeypatch.setattr(store, "sampled", lambda ref: False)
    calls = []

    async def complete(messages, **kwargs):
        calls.append((messages, kwargs))
        return SimpleNamespace(content="Sam reports to Robin.")

    monkeypatch.setattr(host, "_llm_router", SimpleNamespace(complete=complete))
    response = await request(app, body=body(parent_session="parent", child_session="child",
        messages=[{"role": "tool", "content": "Sam reports to Robin."}], brief="Read the material."))
    assert response.status_code == 200, response.text
    handle = response.json()["branch"]["handle"]
    response = await request(app, params={"session_id": "parent", "requester": "owner", **OWNER_SENDER})
    assert response.status_code == 200, response.text
    assert response.json()["branches"][0]["handle"] == handle
    consult = body(session_id="parent", handles=[handle], question="Who supervises Sam?")
    response = await request(app, "/consult", body=consult)
    assert response.status_code == 200, response.text
    assert calls[0][1] == {"context": {"task": "consult"}, "tools": []}
    assert "Sam reports to Robin." in calls[0][0][0]["content"]
    response = await request(app, "/consult", body=consult)
    assert response.status_code == 429, response.text
    assert response.json()["detail"] == {"code": "hourly_budget", "handles": [handle]}
    response = await request(app, "/reaction", body=body(session_id="parent", text="Useful.", turn_id="next"))
    assert response.status_code == 200, response.text
    assert response.json()["labelled"] is False
    response = await request(app, "/consult", body={**consult, "handles": ["br-ffffffffffff"]})
    assert response.status_code == 404, response.text
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_retention_runs_from_router_lifespan_even_when_faculty_is_off(setup, monkeypatch):
    _, store = setup
    store.enabled = False
    pruned = asyncio.Event()
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(store, "prune", lambda: loop.call_soon_threadsafe(pruned.set), raising=False)

    async def in_process(operation):
        return operation()

    # Executor shutdown cannot wake its loop in the restricted test environment.
    monkeypatch.setattr(branches.asyncio, "to_thread", in_process)
    async with branches._lifespan(None):
        await asyncio.wait_for(pruned.wait(), timeout=2)


def test_route_store_factory_reads_current_instance_settings(monkeypatch, tmp_path):
    from colonymind.config import load_config
    import colonymind.config as config_module
    config = load_config(tmp_path)
    monkeypatch.setattr(config_module, "load_config", lambda: config)
    store = branches._store()
    assert store.enabled is True
    assert store.retention_days == 14
    assert store.hourly_budget == 20
    assert store.token_cap == 48000


@pytest.mark.asyncio
async def test_structured_next_owner_turn_labels_both_real_answers(setup, monkeypatch, tmp_path):
    from colonymind.memory.branches import BranchStore
    from colonymind.turns.idempotency import TurnIdempotencyLedger
    app, _ = setup
    store = BranchStore(TurnIdempotencyLedger(tmp_path / "turn-idempotency.db"), key=b"fixture-branch-key")
    monkeypatch.setattr(branches, "_store", lambda: store)
    monkeypatch.setattr(store, "sampled", lambda ref: True)
    answers = iter(["Sam reports to nobody.", "Sam reports to Robin."])

    async def complete(messages, **kwargs):
        return SimpleNamespace(content=next(answers))

    monkeypatch.setattr(host, "_llm_router", SimpleNamespace(complete=complete))
    response = await request(app, body=body(parent_session="parent", child_session="child",
        messages=[{"role": "tool", "content": "Sam reports to Robin."}], brief="Read the material.",
        summary="Sam reports to Robin."))
    handle = response.json()["branch"]["handle"]
    response = await request(app, "/consult", body=body(
        session_id="parent", handles=[handle], question="Who supervises Sam?"))
    assert response.status_code == 200, response.text
    assert response.json()["reference_answer"] == "Sam reports to Robin."
    response = await request(app, "/reaction", body=body(session_id="parent", turn_id="next",
        text=json.dumps({"answer": "Not helpful.", "summary": "Keep going."})))
    assert response.status_code == 200, response.text
    assert response.json()["labelled"] is True
    assert store.report()["comparator"]["labelled"] == 1
    assert store.report()["comparator"]["losses"] == 1
