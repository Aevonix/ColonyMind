"""Offline vertical slice: synthetic Google originals, fake model and Hands queue."""
import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.credentials import CredentialTable, default_credentials, secret_digest
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import host, mind as mind_routes
from colonymind.initiatives.store import InitiativeStore
from colonymind.mind.concerns import MindState, open_mind_db
from colonymind.mind.ideas import Ideas, MARKER
from colonymind.mind.ideas_copy import RETRY_PROMPT, named
from colonymind.secrets.vault import Vault
from colonymind.turns.idempotency import TurnIdempotencyLedger

NOW = datetime(2026, 10, 17, 14, tzinfo=timezone.utc)
ACCOUNT = {"server": "google", "account": "synthetic-account", "calendars": ["primary"]}
KEY = "offline-test-owner-key"


def mailbox():
    messages = [
        {"id": "m-invoice", "threadId": "t-order", "subject": "Invoice INV-204 / order PO-88",
         "snippet": "Invoice INV-204 is $480; order PO-88 lists $450. Please confirm by 2026-10-21."},
        {"id": "m-renewal", "threadId": "t-renewal", "subject": "Workspace renewal on 2026-10-22",
         "snippet": "The annual renewal is $120 on 2026-10-22; cancellation closes 2026-10-20."},
        {"id": "m-form", "threadId": "t-form", "subject": "Consent form due 2026-10-20",
         "snippet": "Please return the consent form before 2026-10-20. The attachment needs a signature."},
    ]
    return [{"server": "google", "account": ACCOUNT["account"], "tool": "gmail_get",
             "result": {"account": ACCOUNT["account"], "data": m}} for m in messages] + [
        {"server": "google", "account": ACCOUNT["account"], "tool": "calendar_events", "calendar_id": "primary",
         "result": {"account": ACCOUNT["account"], "data": {"items": [
             {"id": "e-review", "summary": "Project review 2026-10-19", "description": "Bring the invoice comparison.",
              "start": {"dateTime": "2026-10-19T15:00:00Z"}}]}}}]


class Model:
    def __init__(self):
        self.calls = 0
        self.rewrite = lambda value: value
        self.retry = lambda offers: []  # a copy-only retry that offers nothing new

    async def complete(self, messages, **kwargs):
        self.calls += 1
        self.messages = deepcopy(messages)
        assert kwargs["context"].get("allow_fallback", True) is True
        if messages[0]["content"] == named(RETRY_PROMPT):
            return SimpleNamespace(content=json.dumps({"copies": self.retry(json.loads(messages[-1]["content"]))}),
                                   usage={}, cost_usd=0.0, model_id="test")
        sources = json.loads(messages[-1]["content"])
        offers = []
        for s in sources:
            value = s["value"]
            subject = value.get("subject", value.get("summary", value.get("description")))
            if not subject:
                continue
            if s["tool"] != "calendar_events" and not any(word in subject.lower() for word in ("invoice", "renewal", "form")):
                continue
            family = "renewal" if "renewal" in subject else "document" if "form" in subject else "calendar" if s["tool"] == "calendar_events" else "packet"
            offer = {"family": family, "sourceIds": [s["key"]], "anchor": subject,
                     "facts": [{"sourceId": s["key"], "quote": value.get("snippet", value.get("description"))}]}
            if s["tool"] == "commitment":
                title, summary, fit = {
                    "packet": ("Check the invoice documents", "I’ll compare the invoice details and flag gaps for review.",
                               "I see an open invoice review to resolve before your next decision."),
                    "document": ("Check the consent requirements", "I’ll list the consent form requirements and missing information.",
                                 "I’ll clarify what you need to complete the consent form."),
                    "renewal": ("Review the renewal decision", "I’ll outline the renewal terms and questions for review.",
                                "I’ll clarify the terms while you can still decide about the renewal."),
                }[family]
                offer["content"] = {"title": title, "summary": summary, "fitReason": fit,
                                    "detailDescription": "I’ll check the recorded requirements and flag gaps for review.",
                                    "buildSummary": "I’ll return a checklist and open questions for your review."}
            offers.append(offer)
        return SimpleNamespace(content=json.dumps({"offers": self.rewrite(offers)}), usage={"total_tokens": 800},
                               cost_usd=0.001, model_id="offline-fake")


class Queue:
    def __init__(self):
        self.calls = []
        self.tasks = {}
        self.state = "accepted"
        self.lose_reply = False

    async def enqueue(self, definition, brief, execution, request):
        self.calls.append((definition, brief, execution))
        self.tasks.setdefault(execution, {"taskId": "t-offline-1"})
        if self.lose_reply:
            self.lose_reply = False
            raise RuntimeError("lost reply after durable queue acceptance")
        return self.tasks[execution]

    async def status(self, task):
        return {"status": self.state, **({"result": {"packet": "INV-204: $30 discrepancy; reply draft remains unsent.",
                                                       "branch": "private-result-1"}} if self.state == "done" else {})}


@pytest.fixture
def env(tmp_path, monkeypatch):
    clock = [NOW]
    store = InitiativeStore(tmp_path)
    state = MindState(open_mind_db(tmp_path / "mind.db"), clock=lambda: clock[0])
    model, queue = Model(), Queue()
    mind = SimpleNamespace(store=store, mind_state=state, config={"ideas": {"accounts": [ACCOUNT], "definition_id": "d-review"}},
                           owner_id="synthetic-owner", state_dir=tmp_path, ledger=TurnIdempotencyLedger(tmp_path / "sources.db"),
                           clock=lambda: clock[0], tz=timezone.utc, router=model, commitments=None, enabled=True)
    ideas = Ideas(mind, vault=Vault(tmp_path, key_store="file"), queue=queue)
    mind.ideas = store._ideas = ideas
    monkeypatch.setattr(host, "_initiative_store", store)
    def record_feedback(key, outcome, *, source):
        row = store.get(source)
        if (key == "grounded_offer:duty" and outcome == "dismissed" and row is not None
                and row.status == "cancelled" and (row.context.get("idea") or {}).get("generator") == MARKER):
            return
        pytest.fail("unexpected taste write")
    monkeypatch.setattr(host, "_feedback_store", SimpleNamespace(record=record_feedback))
    monkeypatch.setattr(mind_routes, "_mind", mind)
    yield SimpleNamespace(ideas=ideas, mind=mind, store=store, model=model, queue=queue, clock=clock, tmp=tmp_path)
    store.close()
    state.conn.close()


async def generate(env, observations=None):
    env.ideas.ingest(mailbox() if observations is None else observations)
    await env.ideas.refresh(env.clock[0], env.ideas.meta().get("generation", 0))
    return env.ideas.catalog()


def app():
    value = FastAPI()
    value.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=CredentialTable.from_config(default_credentials()))
    value.include_router(host.router)
    value.include_router(mind_routes.router)
    return value


@pytest.mark.asyncio
async def test_grounded_contract_details_legacy_exclusion_no_calls_on_reads(env):
    legacy = env.store.create(type="TASK_SUGGESTION", description="Generic legacy work")
    rows = await generate(env)
    assert len(rows) == 4
    assert env.model.calls == 1
    async with AsyncClient(transport=ASGITransport(app=app()), base_url="http://test",
                           headers={"Authorization": "Bearer " + KEY}) as client:
        for endpoint in ("/v1/host/initiatives/ideas?contract=2", "/v1/host/initiatives?contract=2"):
            response = await client.get(endpoint)
            assert response.status_code == 200, response.text
            data = response.json()
            assert len(response.content) < 1024 * 1024
            assert len(data["sections"]) == 2 and data["sections"][0]["presentation"]["layout"] == "hero"
            cards = [card for section in data["sections"] for card in section["cards"]]
            assert legacy.id not in {c["id"] for c in cards}
            for card in cards:
                assert set(card["content"]) == {"title", "summary", "detailDescription", "fitReason", "buildSummary"}
                assert not card["content"]["title"].startswith("I can") and len(card["content"]["title"]) <= 40
                assert card["kind"] in {"packet", "renewal", "document", "calendar"}
                detail = await client.get(f"/v1/host/initiatives/{card['id']}?contract=2")
                assert detail.json() == {"card": card}
                assert "sourceRefs" not in detail.json() and "private" not in detail.json()
        states = await client.get("/v1/host/initiatives/ideas/state", params={"ids": legacy.id})
        assert states.json()["states"][0]["exists"] is False
    assert env.model.calls == 1
    assert env.store.get(legacy.id).status == "pending" and env.store.get_history(legacy.id) == []
    # Private copy is absent from generic initiative storage and source text.
    raw = env.store._db.execute("SELECT description,context FROM initiatives WHERE type='grounded_offer'").fetchall()
    assert "INV-204" not in str([tuple(r) for r in raw])


@pytest.mark.asyncio
@pytest.mark.parametrize("corrupt", ["quote", "source", "title", "family"])
async def test_rejects_hallucinated_fact_source_anchor_and_capability(env, corrupt):
    def rewrite(offers):
        o = offers[0]
        if corrupt == "quote": o["facts"][0]["quote"] = "The $999 invoice was paid on 2026-10-01"
        if corrupt == "source": o["sourceIds"] = ["invented-source"]
        if corrupt == "title": o["anchor"] = "A fabricated recipient and date"
        if corrupt == "family": o["family"] = "send_payment"
        return [o]
    env.model.rewrite = rewrite
    assert await generate(env) == []


@pytest.mark.asyncio
async def test_thin_evidence_returns_one_and_expiry_never_refreshes_on_reads(env):
    rows = await generate(env, mailbox()[:1])
    assert len(rows) == 1
    expiry = rows[0].expires_at
    assert env.ideas.catalog()[0].expires_at == expiry
    env.clock[0] += timedelta(hours=24)
    assert env.ideas.catalog() == []
    assert env.store.idea_states([rows[0].id])[0]["exists"] is False
    await env.ideas.tick(env.clock[0])
    assert env.store.get(rows[0].id).status == "expired"
    assert not any(h.action == "llm_dismissed" for h in env.store.get_history(rows[0].id))
    await env.ideas.close()


@pytest.mark.asyncio
async def test_dedupe_and_dismissal_suppression_without_taste(env):
    rows = await generate(env, mailbox()[:1])
    assert await generate(env, mailbox()[:1]) == rows
    await host.respond_to_initiative(rows[0].id, action="dismissed", details=None, request=None,
                                    mode=None, item_ids=None, itemIds=None, origin=None)
    assert env.ideas.catalog() == []
    assert env.ideas.meta().get("pending_at") is not None
    assert (await generate(env, mailbox()[:1])) == []
    assert len(env.ideas.rows()) == 1


@pytest.mark.asyncio
async def test_build_runs_once_after_lost_reply_restart_and_maps_real_result(env):
    row = (await generate(env, mailbox()[:1]))[0]
    env.queue.lose_reply = True
    with pytest.raises(RuntimeError, match="lost reply"):
        await env.ideas.build(row, "selectedItems", ["draft"], None)
    restarted = Ideas(env.mind, vault=env.ideas.vault, queue=env.queue)
    env.store._ideas = restarted
    a, b = await asyncio.gather(restarted.build(row, "selectedItems", ["draft"], None),
                                restarted.build(row, "full", None, None))
    assert a == b and a["buildState"] == "accepted" and a["buildStatus"] == "in_progress" and a["taskId"] == "t-offline-1"
    assert len(env.queue.tasks) == 1 and len(env.queue.calls) == 2
    assert env.queue.calls[0] == env.queue.calls[1]
    brief = json.loads(env.queue.calls[0][1])
    assert brief["selectedItems"] == ["draft", "evidence"]
    assert "No send, spend" in brief["effects"]
    assert all(s["account"] == ACCOUNT["account"] for s in brief["sourceRefs"])
    env.queue.state = "working"
    await restarted.reconcile()
    assert restarted.card(env.store.get(row.id))["buildState"] == "working"
    env.queue.state = "done"
    await restarted.reconcile()
    card = restarted.card(env.store.get(row.id))
    assert card["buildStatus"] == "complete" and card["buildState"] == "done" and "$30 discrepancy" in card["result"]["packet"]
    assert env.store.get(row.id).result is None


@pytest.mark.asyncio
async def test_build_http_and_invalid_selections_and_missing_queue(env):
    row = (await generate(env, mailbox()[:1]))[0]
    for ids in (None, ["unknown"], ["draft", "draft"]):
        with pytest.raises(HTTPException) as e:
            await env.ideas.build(row, "selectedItems", ids, None)
        assert e.value.status_code == 422
    async with AsyncClient(transport=ASGITransport(app=app()), base_url="http://test",
                           headers={"Authorization": "Bearer " + KEY}) as client:
        response = await client.post(f"/v1/host/initiatives/{row.id}/respond",
                                     json={"action": "approved", "mode": "selectedItems", "itemIds": ["draft"]})
        assert response.status_code == 200, response.text
        assert response.json()["taskId"] == "t-offline-1"
    env.queue.state = "failed"
    await env.ideas.reconcile()
    assert env.ideas.card(env.store.get(row.id))["buildStatus"] == "failed"


@pytest.mark.asyncio
async def test_missing_definition_never_returns_fake_acceptance(env):
    from colonymind.mind.ideas_build import HandsQueue
    row = (await generate(env, mailbox()[:1]))[0]
    env.ideas.queue = HandsQueue()
    env.ideas.config["definition_id"] = None
    with pytest.raises(HTTPException) as error:
        await env.ideas.build(row, "full", None, None)
    assert error.value.status_code == 409 and error.value.detail == "idea_review_definition_required"
    assert not env.store.get(row.id).context["idea"].get("execution")
    assert "buildStatus" not in env.ideas.card(env.store.get(row.id))


@pytest.mark.asyncio
async def test_provenance_changes_during_inference_withhold_the_whole_offer(env):
    original = env.model.complete
    async def race(*args, **kwargs):
        result = await original(*args, **kwargs)
        correction = mailbox()[:1]
        correction[0]["result"]["data"]["snippet"] = "The invoice is resolved; no action is needed."
        env.ideas.ingest(correction)
        return result
    env.model.complete = race
    assert await generate(env, mailbox()[:1]) == []
    assert env.ideas.meta().get("pending_at") is not None


@pytest.mark.asyncio
async def test_new_thread_reply_revalidates_old_offer_and_resolution_prevents_republication(env):
    row = (await generate(env, mailbox()[:1]))[0]
    reply = deepcopy(mailbox()[:1])
    reply[0]["result"]["data"].update(id="m-resolved", snippet="Invoice INV-204 is resolved; no review is needed.")
    env.ideas.ingest(reply)
    assert not env.ideas.valid(row)
    await env.ideas.refresh(env.clock[0], env.ideas.meta()["generation"])
    assert env.ideas.catalog() == []


@pytest.mark.asyncio
async def test_more_urgent_new_offer_supersedes_weak_offer_without_dismissal(env):
    def rewrite(offers):
        for offer in offers:
            offer["content"] = {
                "title": "Review the invoice", "summary": "I'll check the invoice documents and flag gaps for review.",
                "detailDescription": "I'll prepare the comparison" +
                                     (" before 2026-10-18." if "2026-10-18" in offer["anchor"] else " for review."),
                "fitReason": "I'll clarify the invoice details before you decide how to reply.",
                "buildSummary": "I'll return a cited comparison and open questions.",
            }
        return offers
    env.model.rewrite = rewrite
    batch = []
    for i in range(6):
        obs = deepcopy(mailbox()[0])
        obs["result"]["data"] = {"id": f"weak-{i}", "threadId": f"weak-{i}", "subject": f"Invoice packet {i}",
                                 "snippet": "Please check the attached document."}
        batch.append(obs)
    await generate(env, batch)
    assert len(env.ideas.catalog()) == 6
    urgent = deepcopy(mailbox()[:1])
    urgent[0]["result"]["data"].update(id="m-urgent", threadId="t-urgent", subject="Invoice review by 2026-10-18",
                                        snippet="Please check the invoice by 2026-10-18.")
    await generate(env, urgent)
    assert len(env.ideas.catalog()) == 6
    assert any("2026-10-18" in env.ideas.card(r)["content"]["detailDescription"] for r in env.ideas.catalog())
    assert len(env.ideas.rows()) == 7
    assert any(h.action == "idea_superseded" for r in env.ideas.rows() for h in env.store.get_history(r.id))
    assert all(h.action != "llm_dismissed" for r in env.ideas.rows() for h in env.store.get_history(r.id))


@pytest.mark.asyncio
async def test_six_card_and_utf8_relay_bounds(env):
    env.model.rewrite = lambda offers: [{**offer, "content": {
        "title": "Check the invoice", "summary": "I'll compare the invoice details and flag gaps for review.",
        "detailDescription": "I'll flag unclear terms in the invoice. 待确认的问题会列在清单里。",
        "fitReason": "I'll clarify the invoice details before you decide how to reply.",
        "buildSummary": "I'll return a cited comparison and open questions.",
    }} for offer in offers]
    batch = []
    for i in range(10):
        obs = deepcopy(mailbox()[0])
        obs["result"]["data"] = {"id": f"wide-{i}", "threadId": f"wide-thread-{i}",
                                 "subject": f"Invoice {i} " + "界" * 100, "snippet": "界" * 500}
        batch.append(obs)
    rows = await generate(env, batch)
    assert len(rows) == 6
    assert len([r for r in env.ideas.rows() if r.status == "proposed"]) == 6
    value = host._ideas_catalog_v2(1000, None)
    assert len(json.dumps(value, ensure_ascii=True).encode()) < 1_048_576
    assert sum(len(s["cards"]) for s in value["sections"]) == 6
    # Long quoted evidence belongs in the private payload/originals, so a
    # Unicode-rich card still fits the existing 8 KB Hands task-brief contract.
    result = await env.ideas.build(rows[0], "full", None, None)
    assert result["taskId"] == "t-offline-1"
    assert len(env.queue.calls[0][1]) <= 8000


@pytest.mark.asyncio
async def test_source_deadline_and_commitment_resolution(env):
    env.clock[0] = NOW.replace(day=20, hour=23)
    data = mailbox()[:1]
    rows = await generate(env, data)
    assert rows[0].expires_at == datetime(2026, 10, 21, tzinfo=timezone.utc)
    env.clock[0] += timedelta(hours=2)
    assert not env.ideas.valid(rows[0])
    env.clock[0] = NOW
    env.mind.commitments = SimpleNamespace(list=lambda **kwargs: {"commitments": [
        {"id": "c-review", "description": "Review Invoice INV-204", "status": "open"}]})
    env.model.rewrite = lambda offers: [{**offers[0], "sourceIds": [*offers[0]["sourceIds"], "commitment:c-review"]}]
    # A new action family avoids the already generated packet's dedupe key.
    env.model.rewrite = lambda offers: [{**offers[0], "family": "document",
        "sourceIds": [*offers[0]["sourceIds"], "commitment:c-review"], "content": {
            "title": "Check the invoice requirements", "summary": "I'll list missing invoice information before you reply.",
            "detailDescription": "I'll check invoice requirements before 2026-10-21.",
            "fitReason": "I'll clarify the invoice details before you decide how to reply.",
            "buildSummary": "I'll return a requirements checklist and open questions.",
        }}]
    row = (await generate(env, data))[0]
    env.mind.commitments = SimpleNamespace(list=lambda **kwargs: {"commitments": []})
    assert not env.ideas.valid(row)


def test_account_mismatch_rejected_before_publication(env):
    batch = mailbox()[:1]
    batch[0]["result"]["account"] = "unselected-account"
    with pytest.raises(ValueError, match="account_mismatch"):
        env.ideas.ingest(batch)
    assert env.ideas.catalog() == []


@pytest.mark.asyncio
async def test_model_credential_cannot_start_work_or_read_private_cards(env):
    row = (await generate(env, mailbox()[:1]))[0]
    credentials = default_credentials()
    credentials["model"] = {"owner": ["sms"], "contacts": "all", "access": ["read", "write", "manage"],
                            "secrets": [{"sha256": secret_digest("offline-model-key")}]}
    value = FastAPI()
    value.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=CredentialTable.from_config(credentials))
    value.include_router(host.router)
    async with AsyncClient(transport=ASGITransport(app=value), base_url="http://test",
                           headers={"Authorization": "Bearer offline-model-key"}) as client:
        for endpoint in ("/v1/host/initiatives/ideas?contract=2", f"/v1/host/initiatives/{row.id}?contract=2"):
            assert (await client.get(endpoint)).status_code == 403
        response = await client.post(f"/v1/host/initiatives/{row.id}/respond", json={"action": "approved", "mode": "full"})
        assert response.status_code == 403
        assert (await client.post(f"/v1/host/initiatives/{row.id}/execute", json={"mode": "full"})).status_code == 403
    assert env.queue.tasks == {}


@pytest.mark.asyncio
async def test_generation_execution_and_expiry_never_write_engagement_or_feedback(env):
    row = (await generate(env, mailbox()[:1]))[0]
    await env.ideas.build(row, "full", None, None)
    env.queue.state = "done"
    await env.ideas.reconcile()
    env.clock[0] += timedelta(days=2)
    await env.ideas.tick(env.clock[0], schedule=False)
    assert env.store._db.execute("SELECT count(*) FROM initiative_engagement").fetchone()[0] == 0
    assert all(h.action not in {"feedback_up", "llm_dismissed", "llm_approved", "llm_actioned"}
               for h in env.store.get_history(row.id))
    # The execution result remains available privately as historical evidence;
    # recommendation freshness alone does not rewrite its real task status.
    saved = env.store.get(row.id)
    assert env.ideas.valid(saved, accepted=True)
    assert env.ideas.card(saved)["buildStatus"] == "complete"


@pytest.mark.asyncio
async def test_hands_seam_uses_definition_task_interface_and_true_attempt_state(monkeypatch):
    import sys
    from colonymind.hands import controller
    from colonymind.mind.ideas_build import HandsQueue
    calls = []
    task = {"id": "t-seam", "state": "open", "intent_version": 1, "dispatch_state": "queued",
            "status": "running", "status_detail": {"attempt_state": None, "hand_id": None}}
    hand = {"id": "h-seam", "intent_version": 1, "branch": "private-packet-branch", "evidence": {"level": "E2"},
            "decisions": {"result_text": "Cited comparison and unsent draft"}}
    class Controller:
        async def queue_definition_task(self, ident, body, key, *, request):
            calls.append((ident, body.brief, key, request))
            return task
        async def task_view(self, ident): return task
        async def get_hand(self, ident): return hand
    async def get(): return Controller()
    monkeypatch.setattr(controller.Controller, "queue_definition_task", Controller.queue_definition_task, raising=False)
    monkeypatch.setattr(controller, "get_controller", get)
    monkeypatch.setitem(sys.modules, "colonymind.hands.definitions", SimpleNamespace(TaskBrief=lambda **kw: SimpleNamespace(**kw)))
    queue = HandsQueue()
    assert await queue.enqueue("d-seam", '{"sourceReads":[{"server":"google"}]}', "stable-execution", "trusted-request") == {"taskId": "t-seam"}
    assert calls == [("d-seam", '{"sourceReads":[{"server":"google"}]}', "idea:stable-execution", "trusted-request")]
    assert (await queue.status("t-seam"))["status"] == "accepted"
    task.update(dispatch_state=None, status_detail={"attempt_state": "running", "hand_id": "h-seam"})
    assert (await queue.status("t-seam"))["status"] == "working"
    task.update(state="done", status_detail={"attempt_state": "done", "hand_id": "h-seam"})
    assert (await queue.status("t-seam"))["result"]["packet"] == "Cited comparison and unsent draft"
    task["intent_version"] = 2
    assert (await queue.status("t-seam"))["status"] != "done"


def test_body_collector_uses_existing_native_reads_exact_accounts_and_bounds(env):
    path = Path(__file__).resolve().parents[2] / "plugins/hermes-plugin/ideas.py"
    spec = importlib.util.spec_from_file_location("offline_google_ideas", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls, posted = [], []
    class Response:
        is_success = True
        def json(self): return {"due": True, "accounts": [ACCOUNT]}
    class Client:
        def get(self, *args, **kwargs): return Response()
        def post(self, *args, **kwargs):
            posted.append(kwargs["json"])
            return Response()
    def native(server, tool, args, timeout):
        calls.append((server, tool, args))
        assert timeout == 3 and args["account"] == ACCOUNT["account"]
        if tool == "gmail_search": data = {"messages": [{"id": f"m-{i}"} for i in range(30)]}
        elif tool == "gmail_get": data = {"id": args["message_id"], "snippet": "Original content"}
        else: data = {"items": []}
        return {"ok": True, "result": {"account": ACCOUNT["account"], "data": data}}
    collector = module.GoogleIdeas(Client(), native, clock=lambda: NOW)
    assert collector.collect()
    assert len(calls) == 10 and len(posted[0]["observations"]) == 9
    assert {c[1] for c in calls} <= module.READS
    assert calls[0][2]["query"] == "newer_than:14d"
    assert calls[-1][2]["calendar_id"] == "primary"
    with pytest.raises(ValueError, match="read_only"):
        collector.read("google", ACCOUNT["account"], "gmail_send", {})


@pytest.mark.asyncio
async def test_one_tick_owned_debounced_batch_and_off_switch(env):
    env.ideas.ingest(mailbox())
    await env.ideas.tick(env.clock[0])
    assert env.ideas.job is None
    env.clock[0] += timedelta(seconds=61)
    await env.ideas.tick(env.clock[0])
    first = env.ideas.job
    await env.ideas.tick(env.clock[0])
    assert env.ideas.job is first
    await first
    assert env.model.calls == 1
    await env.ideas.tick(env.clock[0])
    assert env.ideas.job is None
    env.ideas.trigger()
    env.clock[0] += timedelta(minutes=2)
    env.mind.enabled = False
    await env.ideas.tick(env.clock[0])
    assert env.ideas.job is None


@pytest.mark.asyncio
async def test_simulated_day_calls_per_refresh_no_fixed_daily_cap(env):
    """80 arriving messages + 8 event revisions in 20 activity bursts, 200 reads.

    Five owner refresh triggers (3 Build, 2 dismiss) share the same debounce.
    This measures synthesis only, separate from Hands execution inference.
    """
    start = env.clock[0]
    for burst in range(20):
        env.clock[0] = start + timedelta(minutes=25 * burst)
        batch = []
        for j in range(4):
            obs = deepcopy(mailbox()[0])
            obs["result"]["data"] = {"id": f"mail-{burst}-{j}", "threadId": f"notice-{burst}-{j}",
                                     "subject": f"Service notice {burst}-{j}",
                                     "snippet": "Automated service update; no owner action requested."}
            batch.append(obs)
        if burst == 0:
            batch[:3] = mailbox()[:3]  # three actionable threads among 77 routine messages
        if burst < 8:
            event = mailbox()[-1]
            event["result"]["data"]["items"][0]["description"] += f" Revision {burst}."
            batch.append(event)
        env.ideas.ingest(batch)
        env.clock[0] += timedelta(seconds=61)
        await env.ideas.tick(env.clock[0])
        await env.ideas.job
        for _ in range(10):
            env.ideas.catalog()
    for _ in range(5):
        env.ideas.trigger()
        env.clock[0] += timedelta(seconds=61)
        await env.ideas.tick(env.clock[0])
        await env.ideas.job
    meta = env.ideas.meta()
    # More than SPLIT_GROUPS source groups: every refresh is two synthesis calls of disjoint groups.
    assert env.model.calls == 50
    assert meta["calls_per_day"]["2026-10-17"] == 50
    assert meta["last_refresh"]["model_calls"] == 2

from colonymind.memory import curated

@pytest.mark.asyncio
async def test_revalidation_on_correction_erasure_and_event_cancellation(env):
    rows = await generate(env)
    invoice = next(r for r in rows if "INV-204" in env.ideas.card(r)["content"]["title"])
    changed = mailbox()[:1]
    changed[0]["result"]["data"]["snippet"] = "Invoice INV-204 is resolved; no review is needed."
    env.ideas.ingest(changed)
    assert not env.ideas.valid(invoice)
    with pytest.raises(HTTPException, match="409"):
        await env.ideas.build(invoice, "full", None, None)
    event = next(r for r in rows if env.ideas.card(r)["kind"] == "calendar")
    cancellation = mailbox()[-1:]
    cancellation[0]["result"]["data"]["items"][0]["status"] = "cancelled"
    env.ideas.ingest(cancellation)
    assert not env.ideas.valid(event)
    renewal = next(r for r in rows if env.ideas.card(r)["kind"] == "renewal")
    source = env.ideas.open(renewal.context["idea"]["private"])["sources"][0]
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(env.mind.ledger)
    _source_forget_receipt = curated.forget_sources(env.mind.ledger, env.mind.owner_id, source_ids=[source['ref']['source_id']])
    assert historical_bytes(env.mind.ledger) == _native_before_forget
    assert all(curated.is_forgotten(env.mind.ledger, sid) for sid in _source_forget_receipt['source_ids'])
    assert not env.ideas.valid(renewal)
