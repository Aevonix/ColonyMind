"""Offline commitment-only Ideas through the existing tick and owner API."""
from datetime import datetime, timedelta
import importlib.util
import json
import logging
from pathlib import Path
import re

from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.routers import host
from colonymind.commitments import store as commitment_store
from colonymind.hands import controller, definitions
from colonymind.mind import ideas as ideas_module
from colonymind.mind.ideas_build import HandsQueue
from colonymind.mind.ideas_copy import RETRY_PROMPT, fallback_copy, named, validate_copy
from test_grounded_ideas import ACCOUNT, KEY, app, env, mailbox  # noqa: F401
from test_hands_controller import c  # noqa: F401


@pytest.fixture
def commitments(env, monkeypatch):
    monkeypatch.setattr(commitment_store, "now_utc", lambda: env.clock[0])
    env.mind.commitments = commitment_store.CommitmentStore(env.tmp / "commitments.db")
    env.ideas.config["accounts"] = []
    return env.mind.commitments


def add_open(env, commitments):
    pending = commitments.create(env.mind.owner_id, "Review invoice documents",
                                 due_at=(env.clock[0] + timedelta(hours=12)).isoformat())
    overdue = commitments.create(env.mind.owner_id, "Prepare consent form",
                                 due_at=(env.clock[0] - timedelta(days=2)).isoformat(), allow_overdue=True)
    return pending, overdue


async def debounced_refresh(env):
    await env.ideas.tick(env.clock[0])
    assert env.ideas.job is None
    env.clock[0] += timedelta(seconds=59)
    await env.ideas.tick(env.clock[0])
    assert env.ideas.job is None
    env.clock[0] += timedelta(seconds=2)
    await env.ideas.tick(env.clock[0])
    job = env.ideas.job
    assert job is not None
    await env.ideas.tick(env.clock[0])
    assert env.ideas.job is job
    await job
    await env.ideas.tick(env.clock[0])
    assert env.ideas.job is None
    return env.ideas.catalog()


@pytest.mark.asyncio
async def test_no_accounts_batches_owner_open_commitments_and_never_models_on_reads(env, commitments):
    pending, overdue = add_open(env, commitments)
    commitments.create("another-owner", "Review private invoice")
    terminal = commitments.create(env.mind.owner_id, "Review completed invoice")
    commitments.update(terminal["id"], status="fulfilled")
    legacy = env.store.create(type="TASK_SUGGESTION", description="Legacy work")
    rows = await debounced_refresh(env)
    assert len(rows) == 2 and env.model.calls == 1
    inputs = json.loads(env.model.messages[-1]["content"])
    assert {s["key"] for s in inputs} == {"commitment:" + c["id"] for c in (pending, overdue)}
    assert {s["tool"] for s in inputs} == {"commitment"}
    for row in rows:
        payload = env.ideas.open(row.context["idea"]["private"])
        source = payload["sources"][0]
        assert source["tool"] == "commitment" and source["account"] == "owner"
        card = env.ideas.card(row)
        validate_copy(card["content"])
        assert card["content"]["fitReason"].startswith(("I see ", "I’ll "))
        assert_clean_card(card, next(c["description"] for c in (pending, overdue) if c["id"] == source["provider_id"]))
        assert "email" not in str(card["content"]).lower()
        expected = (env.clock[0] + timedelta(hours=24) if source["provider_id"] == overdue["id"]
                    else datetime.fromisoformat(pending["due_at"]))
        assert row.expires_at == expected
    async with AsyncClient(transport=ASGITransport(app=app()), base_url="http://test",
                           headers={"Authorization": "Bearer " + KEY}) as client:
        for endpoint in ("/v1/host/initiatives/ideas?contract=2", "/v1/host/initiatives?contract=2"):
            response = await client.get(endpoint)
            assert response.status_code == 200
            cards = [c for s in response.json()["sections"] for c in s["cards"]]
            assert {c["id"] for c in cards} == {r.id for r in rows}
        for row in rows:
            response = await client.get(f"/v1/host/initiatives/{row.id}?contract=2")
            assert response.status_code == 200 and response.json()["card"]["id"] == row.id
    assert env.model.calls == 1
    assert env.store.get_history(legacy.id) == []
    assert env.store._db.execute("SELECT count(*) FROM initiative_engagement").fetchone()[0] == 0
    # Another refresh is one batch and preserves the same action identities.
    await env.ideas.refresh(env.clock[0], env.ideas.meta()["generation"])
    assert env.model.calls == 2 and {r.id for r in env.ideas.catalog()} == {r.id for r in rows}
    await host.respond_to_initiative(rows[0].id, action="dismissed", details=None, request=None,
                                    mode=None, item_ids=None, itemIds=None, origin=None)
    await env.ideas.refresh(env.clock[0], env.ideas.meta()["generation"])
    assert env.model.calls == 3 and len(env.ideas.catalog()) == 1 and len(env.ideas.rows()) == 2
    assert env.store._db.execute("SELECT count(*) FROM initiative_engagement").fetchone()[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("has_store", [False, True])
async def test_no_accounts_no_commitments_empty_without_model_calls(env, commitments, has_store):
    if not has_store:
        env.mind.commitments = None
        env.ideas.trigger()
    assert await debounced_refresh(env) == []
    assert env.model.calls == 0
    assert env.ideas.meta()["last_refresh"]["model_calls"] == 0
    assert env.ideas.catalog() == []


@pytest.mark.asyncio
async def test_commitment_build_keeps_missing_definition_409(env, commitments):
    add_open(env, commitments)
    row = (await debounced_refresh(env))[0]
    env.ideas.config.pop("definition_id")
    async with AsyncClient(transport=ASGITransport(app=app()), base_url="http://test",
                           headers={"Authorization": "Bearer " + KEY}) as client:
        response = await client.post(f"/v1/host/initiatives/{row.id}/respond",
                                     json={"action": "approved", "mode": "full"})
    assert response.status_code == 409
    assert response.json()["detail"] == "idea_review_definition_required"
    assert env.queue.calls == [] and env.model.calls == 1
    assert not env.store.get(row.id).context["idea"].get("execution")


@pytest.mark.asyncio
async def test_commitment_build_queues_a_plain_task_without_source_reads(env, commitments, c, monkeypatch):
    async def get_controller():
        return c
    monkeypatch.setattr(controller, "get_controller", get_controller)
    definition = await c.save_definition(definitions.CreateDefinition(name="Idea tasks",
        tools=["colonymind_memory_search"]), "idea-definition")
    env.ideas.config["definition_id"] = definition["id"]
    env.ideas.queue = HandsQueue()
    add_open(env, commitments)
    row = (await debounced_refresh(env))[0]
    async with AsyncClient(transport=ASGITransport(app=app()), base_url="http://test",
                           headers={"Authorization": "Bearer " + KEY}) as client:
        response = await client.post(f"/v1/host/initiatives/{row.id}/respond",
                                     json={"action": "approved", "mode": "full"})
    assert response.status_code == 200, response.text
    task_id = env.store.get(row.id).context["idea"]["execution"]["taskId"]
    assert response.json()["taskId"] == task_id
    assert json.loads((await c.one("SELECT brief FROM definition_task_queue WHERE task_id=?",
                                   (task_id,)))["brief"])["sourceReads"] == []
    assert (await c.one("SELECT kind FROM tasks WHERE id=?", (task_id,)))["kind"] == "deep"
    assert await c.task_definition(task_id) == definition


@pytest.mark.asyncio
async def test_commitment_revisions_resolution_and_expiry_revalidate(env, commitments):
    pending, overdue = add_open(env, commitments)
    rows = await debounced_refresh(env)
    by_source = {env.ideas.open(r.context["idea"]["private"])["sources"][0]["provider_id"]: r for r in rows}
    commitments.update(pending["id"], description="Review revised invoice documents")
    assert not env.ideas.valid(by_source[pending["id"]])
    with pytest.raises(HTTPException) as error:
        await env.ideas.build(by_source[pending["id"]], "full", None, None)
    assert error.value.status_code == 409 and error.value.detail == "idea_sources_stale"
    commitments.update(overdue["id"], status="fulfilled")
    assert env.ideas.catalog() == []
    await debounced_refresh(env)
    assert len(env.ideas.catalog()) == 1 and env.model.calls == 2
    row = env.ideas.catalog()[0]
    env.clock[0] = row.expires_at
    assert env.ideas.catalog() == [] and env.model.calls == 2
    await env.ideas.tick(env.clock[0], schedule=False)
    assert env.store.get(row.id).status == "expired"


@pytest.mark.asyncio
async def test_commitment_changed_during_synthesis_cannot_publish(env, commitments):
    pending, _ = add_open(env, commitments)
    original = env.model.complete
    async def race(*args, **kwargs):
        result = await original(*args, **kwargs)
        commitments.update(pending["id"], description="Review corrected invoice documents")
        return result
    env.model.complete = race
    rows = await debounced_refresh(env)
    assert len(rows) == 1 and env.model.calls == 1
    assert env.ideas.open(rows[0].context["idea"]["private"])["sources"][0]["provider_id"] != pending["id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("corrupt", ["quote", "source", "anchor", "family", "copy", "missing_copy"])
async def test_commitment_offers_keep_source_and_copy_validation(env, commitments, corrupt):
    add_open(env, commitments)
    def rewrite(offers):
        offer = offers[0]
        if corrupt == "quote": offer["facts"][0]["quote"] = "An invented invoice amount of $999"
        if corrupt == "source": offer["sourceIds"] = ["commitment:invented"]
        if corrupt == "anchor": offer["anchor"] = "An invented invoice description"
        if corrupt == "family": offer["family"] = "send_payment"
        if corrupt == "copy": offer["content"] = {"title": "Broken copy"}
        if corrupt == "missing_copy": offer.pop("content")
        return [offer]
    env.model.rewrite = rewrite
    rows = await debounced_refresh(env)
    # Only verified evidence with bad copy earns the two copy-only retries.
    assert env.model.calls == (3 if corrupt in {"copy", "missing_copy"} else 1)
    assert rows == []


@pytest.mark.asyncio
async def test_adding_accounts_collects_google_and_preserves_commitment_cards(env, commitments):
    add_open(env, commitments)
    rows = await debounced_refresh(env)
    env.ideas.config["accounts"] = [ACCOUNT]
    assert env.ideas.collection() == {"due": True, "accounts": [ACCOUNT]}
    env.ideas.ingest(mailbox())
    result = await debounced_refresh(env)
    assert len(result) == 6 and env.model.calls == 3  # 1 call, then 6 groups split into 2
    assert {r.id for r in rows} <= {r.id for r in result}


def test_no_accounts_skips_google_collection(env, commitments):
    path = Path(__file__).resolve().parents[2] / "plugins/hermes-plugin/ideas.py"
    spec = importlib.util.spec_from_file_location("offline_commitment_collector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class Response:
        is_success = True
        def json(self): return env.ideas.collection()
    class Client:
        def get(self, *args, **kwargs): return Response()
        def post(self, *args, **kwargs): pytest.fail("unexpected Google source ingestion")
    def native(*args, **kwargs): pytest.fail("unexpected Google connector call")
    assert env.ideas.collection() == {"due": False, "accounts": []}
    assert not module.GoogleIdeas(Client(), native).collect()


@pytest.mark.asyncio
async def test_refresh_uses_role_deadline_and_retries_a_failed_synthesis(env, commitments, monkeypatch):
    add_open(env, commitments)
    waits = []
    real_wait_for = ideas_module.asyncio.wait_for

    async def wait_for(awaitable, timeout):
        waits.append(timeout)
        return await real_wait_for(awaitable, timeout)

    monkeypatch.setattr(ideas_module.asyncio, "wait_for", wait_for)
    env.model.function_deadline_seconds = lambda context: 300.0
    complete = env.model.complete

    async def slow(messages, **kwargs):
        env.model.calls += 1
        raise TimeoutError

    env.model.complete = slow
    assert await debounced_refresh(env) == []
    assert waits == [305.0] and env.model.calls == 1
    assert env.ideas.meta()["last_refresh"]["failure"] == "TimeoutError"
    # Unchanged inputs fire no new trigger; the failure itself re-arms one.
    env.clock[0] += timedelta(minutes=14)
    await env.ideas.tick(env.clock[0])
    assert env.ideas.job is None
    env.model.complete = complete
    env.clock[0] += timedelta(minutes=1)
    await env.ideas.tick(env.clock[0])
    assert env.ideas.job is None and "pending_at" in env.ideas.meta()
    env.clock[0] += timedelta(seconds=61)
    await env.ideas.tick(env.clock[0])
    await env.ideas.job
    await env.ideas.tick(env.clock[0])
    assert len(env.ideas.catalog()) == 2 and env.model.calls == 2
    assert env.ideas.meta()["last_refresh"]["failure"] is None and "pending_at" not in env.ideas.meta()
    # A long role deadline is used as configured, with the existing grace.
    env.model.function_deadline_seconds = lambda context: 900
    await env.ideas.refresh(env.clock[0], env.ideas.meta()["generation"])
    assert waits[-1] == 905.0


# Synthetic owner work, including punctuation that is escaped in model inputs.
REALISTIC_COMMITMENTS = [
    ("packet", "Verify current ownership/status of 2 domains: atlas.example and beacon.example, "
     "including registrar access and billing responsibility.",
     "Review domain ownership", "I’ll compare registrar access and billing status for 2 domains."),
    ("packet", "File the approval path defect: the gateway returned 403 on /review/approve after confirmation, "
     "so capture the failing step for review.",
     "Outline the approval defect", "I’ll organize the 403 response and failing approval step into a cited comparison."),
    ("document", "Pause the outreach job or add a hard approval check: review the 3/5 retries and write a checklist "
     "before any further outreach.",
     "Prepare an outreach checklist", "I’ll list the approval gaps and 3/5 retries for your review."),
    ("renewal", "Review the renewal terms: compare the $120 charge and cancellation date, and prepare an unsent reply "
     "about the outstanding questions.",
     "Review the renewal decision", "I’ll check the $120 charge and list questions about cancellation."),
    ("document", "Prepare the consent form checklist: review Dr. Lee’s request for 2 signatures and record the missing "
     "information for an unsent follow-up.",
     "Check the consent requirements", "I’ll gather Dr. Lee’s requested information and check the 2 signatures."),
    ("calendar", "Prepare the meeting brief: compare the 2 rollout options, list open questions, and keep the proposed "
     "agenda ready for review.",
     "Organize the rollout brief", "I’ll compare the 2 options and list questions for the meeting."),
]
COMMITMENT_COPY = {
    "title": "Review domain ownership",
    "summary": "I’ll compare registrar access and billing status for 2 domains.",
    "detailDescription": "I’ll check the recorded requirements and flag gaps for your review.",
    "fitReason": "The recorded work calls for a bounded comparison.",
    "buildSummary": "I’ll return a cited comparison and open questions for your review.",
}


def assert_clean_card(card, description):
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
    def runs(value):
        words = re.findall(r"[^\W_]+", value.casefold())
        return {tuple(words[i:i + 6]) for i in range(len(words) - 5)}
    for value in strings(card):
        assert "i can help with" not in value.casefold()
        assert "I'll review the details you recorded and prepare the selected deliverables." not in value.replace("’", "'")
        assert not runs(value) & runs(description)


def recorded_offer(source, content=COMMITMENT_COPY, family="packet"):
    return {"family": family, "sourceIds": [source["key"]], "anchor": source["value"]["description"],
            "facts": [{"sourceId": source["key"], "quote": source["value"]["description"][:500]}],
            "content": dict(content)}


@pytest.mark.asyncio
async def test_realistic_model_commitment_copy_survives_refresh_without_fallback(env, commitments, caplog):
    due = (env.clock[0] + timedelta(days=2)).isoformat()
    cases = {}
    for family, description, title, summary in REALISTIC_COMMITMENTS:
        record = commitments.create(env.mind.owner_id, description, due_at=due)
        cases[record["id"]] = (family, title, summary)
    supplied = {}

    def rewrite(_):
        offers = []
        for source in json.loads(env.model.messages[-1]["content"]):
            family, title, summary = cases[source["value"]["id"]]
            content = {**COMMITMENT_COPY, "title": title, "summary": summary,
                       "detailDescription": "I’ll prepare the selected deliverables before " + due[:10] + ".",
                       "fitReason": "Your commitment records this open work for review.",
                       "buildSummary": "I’ll return the selected deliverables and open questions for owner review."}
            supplied[source["key"]] = content
            offers.append(recorded_offer(source, content, family))
        return offers

    env.model.rewrite = rewrite
    with caplog.at_level(logging.INFO, logger="colonymind.mind.ideas"):
        rows = await debounced_refresh(env)
    assert len(rows) == 6 and env.model.calls == 2  # 6 commitments split into 2 calls
    assert env.ideas.meta()["last_refresh"]["failure"] is None
    for row in rows:
        payload = env.ideas.open(row.context["idea"]["private"])
        assert env.ideas.card(row)["content"] == supplied[payload["sources"][0]["key"]]
        description = next(c[1] for c in REALISTIC_COMMITMENTS if c[2] == payload["content"]["title"])
        assert_clean_card(env.ideas.card(row), description)
    assert [r.getMessage() for r in caplog.records if r.name == "colonymind.mind.ideas"] == [
        "Ideas copy counts: supplied=6 rejected=0 fallback=0"]


@pytest.mark.parametrize("field,value", [
    ("fitReason", "Your commitment records this open review."),
    ("buildSummary", "I’ll return a cited comparison for owner review."),
    ("detailDescription", "I’ll prepare the comparison before 2026-10-19."),
])
def test_commitment_language_and_recorded_due_date_are_valid_copy(env, commitments, field, value):
    commitments.create(env.mind.owner_id, REALISTIC_COMMITMENTS[0][1],
                       due_at=(env.clock[0] + timedelta(days=2)).isoformat())
    sources = env.ideas.evidence(env.clock[0])
    offer = recorded_offer(sources[0], {**COMMITMENT_COPY, field: value})
    counts = dict(supplied=0, rejected=0, fallback=0)
    # Isolate each copy rule as well as the actual admission path.
    if field == "detailDescription":
        with pytest.raises(ValueError, match="copy_unsupported_fact"):
            validate_copy(offer["content"], sources, evidence=offer["anchor"])
    evidence = offer["anchor"] + " 2026-10-19"
    assert validate_copy(offer["content"], sources, evidence=evidence) == offer["content"]
    payload = env.ideas.validate_offer(offer, sources, env.clock[0], copy_counts=counts)
    assert payload["content"] == offer["content"]
    assert counts == dict(supplied=1, rejected=0, fallback=0)


def test_long_commitment_anchor_uses_exact_decoded_description(env, commitments):
    description = (REALISTIC_COMMITMENTS[1][1] + ' Include the "confirm/review" step and Dr. Lee’s notes: '
                   'keep the report bounded to the recorded failure and prepare an unsent draft for review.')
    assert len(description) > 160
    commitments.create(env.mind.owner_id, description)
    sources = env.ideas.evidence(env.clock[0])
    # The JSON wire representation contains escapes; quotes must match decoded text.
    assert "\\u2019" in sources[0]["text"]
    content = {**COMMITMENT_COPY, "title": "Outline the approval defect",
               "summary": "I’ll organize the 403 response and failing approval step for review."}
    offer = recorded_offer(sources[0], content)
    published = env.ideas.publish([offer], sources, env.clock[0])
    assert len(published) == 1
    assert env.ideas.card(env.store.get(published[0]))["content"] == content
    for field in ("anchor", "quote"):
        invalid = recorded_offer(sources[0], content)
        if field == "anchor":
            invalid["anchor"] = description[:100]
        else:
            invalid["facts"][0]["quote"] = json.dumps(description)[1:-1]
        with pytest.raises(ValueError, match="unsupported_anchor" if field == "anchor" else "unsupported_fact"):
            env.ideas.validate_offer(invalid, sources, env.clock[0])


@pytest.mark.parametrize("field,value,reason", [
    ("summary", "I’ll check the 9 domains.", "copy_unsupported_fact"),
    ("summary", "I’ll check the 77 records.", "copy_unsupported_fact"),
    ("detailDescription", "I’ll prepare the comparison before 2026-10-31.", "copy_unsupported_fact"),
    ("fitReason", "The commitment is {provider_id}.", "copy_internal_identifier"),
    ("fitReason", "The revision is {revision}.", "copy_internal_identifier"),
    ("summary", "I’ll review domain ownership.", "copy_title_echo"),
    ("summary", "I’ll gather the details. I’ll list the gaps.", "copy_summary_one_sentence"),
    ("summary", "I’ll gather the details..", "copy_doubled_punctuation"),
    ("title", "Verify current ownership/status of 2", "copy_verbatim_source"),
    ("summary", "I’ll check current ownership/status of 2 domains.", "copy_verbatim_source"),
    ("fitReason", "I see current ownership/status of 2 domains to check.", "copy_verbatim_source"),
])
def test_commitment_unsafe_copy_is_rejected_without_publishing(env, commitments, field, value, reason):
    commitments.create(env.mind.owner_id, REALISTIC_COMMITMENTS[0][1], metadata={"unquoted_count": 77},
                       due_at=(env.clock[0] + timedelta(days=2)).isoformat())
    sources = env.ideas.evidence(env.clock[0])
    content = {**COMMITMENT_COPY, field: value.format(**sources[0])}
    offer = recorded_offer(sources[0], content)
    with pytest.raises(ValueError, match=reason):
        validate_copy(content, sources, evidence=offer["anchor"] + " 2026-10-19", source_texts=[offer["anchor"]])
    with pytest.raises(ValueError, match="^copy_unavailable$"):
        env.ideas.validate_offer(offer, sources, env.clock[0])
    assert env.ideas.publish([offer], sources, env.clock[0]) == []
    assert env.ideas.rows() == []


@pytest.mark.parametrize("bad", ["account", "tool", "deadline", "long_anchor"])
def test_google_copy_rules_remain_strict_alongside_commitments(env, commitments, bad):
    env.ideas.config["accounts"] = [ACCOUNT]
    observations = mailbox()[-1:]
    event = observations[0]["result"]["data"]["items"][0]
    event["summary"] = "Project review" + (" requested preparation" * 10 if bad == "long_anchor" else "")
    description = event["summary"] if bad == "long_anchor" else REALISTIC_COMMITMENTS[0][1]
    commitments.create(env.mind.owner_id, description, due_at=(env.clock[0] + timedelta(days=2)).isoformat())
    env.ideas.ingest(observations)
    sources = env.ideas.evidence(env.clock[0])
    google = next(s for s in sources if s["tool"] == "calendar_events")
    recorded = next(s for s in sources if s["tool"] == "commitment")
    content = {**COMMITMENT_COPY, "summary": "I’ll gather the requested material and list open questions."}
    if bad in {"account", "tool"}:
        content["fitReason"] = "The entry is from " + google[bad] + "."
    if bad == "deadline":
        content["detailDescription"] = "I’ll prepare the brief before 2026-10-19."
    offer = {"family": "calendar", "sourceIds": [google["key"], recorded["key"]],
             "anchor": event["summary"], "facts": [{"sourceId": google["key"], "quote": event["description"]}],
             "content": content}
    counts = dict(supplied=0, rejected=0, fallback=0)
    if bad == "long_anchor":
        with pytest.raises(ValueError, match="unsupported_anchor"):
            env.ideas.validate_offer(offer, sources, env.clock[0])
    else:
        payload = env.ideas.validate_offer(offer, sources, env.clock[0], copy_counts=counts)
        assert payload["content"] != content
        assert counts == dict(supplied=1, rejected=1, fallback=1)


@pytest.mark.parametrize("family,description", [
    ("packet", "Verify current ownership/status of 2 domains"),
    ("packet", REALISTIC_COMMITMENTS[0][1]),
    ("packet", REALISTIC_COMMITMENTS[1][1]),
    ("document", REALISTIC_COMMITMENTS[2][1]),
    ("renewal", REALISTIC_COMMITMENTS[3][1]),
    ("calendar", REALISTIC_COMMITMENTS[5][1]),
    ("document", "Review the required approval and"),
    ("packet", "Review " + "x" * 80),
    ("packet", "Review access — confirm the owner’s status – record gaps"),
])
def test_missing_commitment_copy_never_falls_back_to_source_text(env, commitments, family, description):
    commitments.create(env.mind.owner_id, description)
    sources = env.ideas.evidence(env.clock[0])
    offer = recorded_offer(sources[0], family=family)
    offer.pop("content")
    with pytest.raises(ValueError, match="^copy_unavailable$"):
        env.ideas.validate_offer(offer, sources, env.clock[0])
    with pytest.raises(ValueError, match="^copy_unavailable$"):
        fallback_copy(family, description, [description], sources)
    assert env.ideas.publish([offer], sources, env.clock[0]) == []
    assert env.ideas.rows() == []


CLEAN_EXAMPLES = [
    ("Pause the outreach job or add a hard text-only/no-fabrication constraint before it fires at 14:54 "
     "(third mention here, active work begun, HIGHEST priority)",
     "Pause the outreach job", "I'll outline a text-only safeguard before the next outreach run.",
     "I'll help prevent invented claims from reaching people while the outreach is active."),
    ("Review the two roof quotes and choose a roofer after checking costs, materials and exclusions.",
     "Compare the two roof quotes", "I'll check the roofing costs, materials and exclusions side by side.",
     "I'll make the tradeoffs clear so you can choose a roofer with fewer unanswered questions."),
    ("Review vendor invoice INV-204 against purchase order PO-88 and flag any mismatch before replying.",
     "Compare INV-204 and PO-88", "I'll check the invoice amounts and terms against the purchase order.",
     "I'll clarify any discrepancies before you decide how to reply to the vendor."),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("description,title,summary,fit", CLEAN_EXAMPLES)
async def test_clean_model_copy_publishes_unchanged_without_raw_commitment(env, commitments, description, title, summary, fit):
    commitments.create(env.mind.owner_id, description)
    content = {**COMMITMENT_COPY, "title": title, "summary": summary, "fitReason": fit}
    env.model.rewrite = lambda _: [recorded_offer(s, content, "document")
                                  for s in json.loads(env.model.messages[-1]["content"])]
    row, = await debounced_refresh(env)
    card = env.ideas.card(row)
    assert card["content"] == content
    assert_clean_card(card, description)
    assert all(note not in json.dumps(card) for note in ("14:54", "third mention", "active work begun", "HIGHEST"))


@pytest.mark.asyncio
@pytest.mark.parametrize("old_copy", [None, 1])
async def test_copy_retirement_reoffers_same_commitments_once_and_preserves_other_statuses(env, commitments, monkeypatch, old_copy):
    pending, overdue = add_open(env, commitments)
    rows = await debounced_refresh(env)
    # Construct pre-version rows with the same private evidence and original dedup keys.
    originals = []
    for row in rows:
        idea = dict(row.context["idea"])
        idea.pop("copy")
        if old_copy is not None:
            idea["copy"] = old_copy
        env.store.update(row.id, context={"idea": idea}, dedup_key=row.dedup_key.rsplit(":c", 1)[0])
        originals.append(env.store.get(row.id))
    protected = []
    for status in ("accepted", "building", "done", "failed", "cancelled", "proposed"):
        record = commitments.create(env.mind.owner_id, "Review invoice documents for " + status)
        source = next(s for s in env.ideas.evidence(env.clock[0]) if s["provider_id"] == record["id"])
        row_id, = env.ideas.publish([recorded_offer(source, {**COMMITMENT_COPY, "title": "Check the invoice documents",
            "summary": "I'll compare the invoice details and flag gaps for review."})], [source], env.clock[0])
        row = env.store.get(row_id)
        idea = dict(row.context["idea"])
        if status != "proposed":
            idea.pop("copy")
            idea["execution"] = {"status": status}
        env.store.update(row.id, status=status, context={"idea": idea},
                         dedup_key=row.dedup_key if status == "proposed" else row.dedup_key.rsplit(":c", 1)[0])
        protected.append(env.store.get(row.id))
    histories = {r.id: env.store.get_history(r.id) for r in originals + protected}
    triggers = []
    trigger = env.ideas.trigger
    def tracked(now=None):
        triggers.append(now)
        trigger(now)
    monkeypatch.setattr(env.ideas, "trigger", tracked)
    await env.ideas.tick(env.clock[0])
    assert triggers == [env.clock[0]]
    for row in originals:
        assert env.store.get(row.id).status == "expired"
        assert [h.action for h in env.store.get_history(row.id)] == [h.action for h in histories[row.id]] + ["idea_copy_retired"]
    await env.ideas.tick(env.clock[0])
    assert len(triggers) == 1
    env.clock[0] += timedelta(seconds=61)
    await env.ideas.tick(env.clock[0])
    await env.ideas.job
    await env.ideas.tick(env.clock[0])
    replacements = [r for r in env.ideas.rows() if r.status == "proposed" and r.id not in {p.id for p in protected}]
    assert len(replacements) == 2 and env.model.calls == 3  # 1 call, then 8 groups split into 2
    assert {env.ideas.open(r.context["idea"]["private"])["sources"][0]["provider_id"] for r in replacements} == {
        pending["id"], overdue["id"]}
    for row in replacements:
        original = next(r for r in originals if r.context["idea"]["action_key"] == row.context["idea"]["action_key"])
        assert row.context["idea"]["copy"] == ideas_module.COPY_VERSION == 2
        assert row.id != original.id and row.dedup_key == original.dedup_key + ":c2"
    for row in protected:
        assert env.store.get(row.id) == row
        assert env.store.get_history(row.id) == histories[row.id]
    assert len(triggers) == 1
    # Cancelled work permanently suppresses another offer, including a later refresh.
    await env.ideas.refresh(env.clock[0], env.ideas.meta()["generation"])
    assert len(env.ideas.rows()) == len(originals) + len(protected) + len(replacements)
    assert all(len(env.store.get_history(r.id)) == len(histories[r.id]) + 1 for r in originals)


def broken_first_pass(env, which=None):
    """Keep the model's good copy aside and send rejected copy for the chosen anchors (all by default)."""
    good = {}
    def rewrite(offers):
        for offer in offers:
            if which is None or offer["anchor"] in which:
                good[offer["anchor"]] = offer["content"]
                offer["content"] = {"title": "Broken copy"}
        return offers
    env.model.rewrite = rewrite
    return good


@pytest.mark.asyncio
async def test_rejected_copy_is_recovered_by_one_copy_only_retry(env, commitments):
    add_open(env, commitments)
    good = broken_first_pass(env)
    asked = []
    def retry(offers):
        asked.extend(o["rejected"] for o in offers)
        return [{"index": o["index"], "content": good[o["anchor"]]} for o in offers]
    env.model.retry = retry
    rows = await debounced_refresh(env)
    assert len(rows) == 2 and env.model.calls == 2 and asked == ["invalid_copy_fields"] * 2
    assert sorted(env.ideas.open(r.context["idea"]["private"])["content"]["title"] for r in rows) == \
        sorted(c["title"] for c in good.values())
    last = env.ideas.meta()["last_refresh"]
    assert (last["model_calls"], last["copy_retries"], last["copy_recovered"], last["copy_dropped"],
            last["copy_retry_errors"], last["failure"]) == (2, 1, 2, 0, 0, None)


@pytest.mark.asyncio
async def test_copy_rejected_three_times_publishes_nothing_and_records_counts(env, commitments):
    add_open(env, commitments)
    broken_first_pass(env)
    env.model.retry = lambda offers: [{"index": o["index"], "content": {"title": "Still broken"}} for o in offers]
    assert await debounced_refresh(env) == [] and env.model.calls == 3
    last = env.ideas.meta()["last_refresh"]
    assert (last["model_calls"], last["copy_retries"], last["copy_recovered"], last["copy_dropped"]) == (3, 2, 0, 2)
    assert env.ideas.meta()["calls_per_day"][env.clock[0].date().isoformat()] == 3


@pytest.mark.asyncio
async def test_mixed_batch_keeps_good_cards_and_retries_only_the_rejected_one(env, commitments):
    add_open(env, commitments)
    good = broken_first_pass(env, which={"Prepare consent form"})
    seen = []
    def retry(offers):
        seen.extend(o["anchor"] for o in offers)
        return [{"index": o["index"], "content": good[o["anchor"]]} for o in offers]
    env.model.retry = retry
    rows = await debounced_refresh(env)
    assert len(rows) == 2 and seen == ["Prepare consent form"] and env.model.calls == 2


@pytest.mark.asyncio
async def test_retry_error_keeps_first_pass_cards_and_does_not_fail_the_refresh(env, commitments):
    add_open(env, commitments)
    broken_first_pass(env, which={"Prepare consent form"})
    original = env.model.complete
    async def complete(messages, **kwargs):
        if messages[0]["content"] == named(RETRY_PROMPT):
            env.model.calls += 1
            raise RuntimeError("model unavailable")
        return await original(messages, **kwargs)
    env.model.complete = complete
    rows = await debounced_refresh(env)
    assert len(rows) == 1 and env.model.calls == 2
    last = env.ideas.meta()["last_refresh"]
    assert last["failure"] is None and last["copy_retry_errors"] == 1 and last["copy_dropped"] == 1
    assert last["published"] == 1


COMMITMENT_SOURCES = [{"tool": "commitment", "key": "commitment:c-1", "account": "owner", "action_id": "c-1",
                       "revision": "r1", "ref": {"source_id": "c-1"}}]
CLEAN = {"title": "Shortlist Florida homes", "summary": "I’ll pull current listings and compare price, size and area.",
         "detailDescription": "I’ll gather current listings, compare them side by side and flag what needs a visit.",
         "fitReason": "You asked me to look into this, and a shortlist makes the next step easy.",
         "buildSummary": "I’ll return a short comparison of current listings."}


@pytest.mark.parametrize("field,value,reason", [
    ("fitReason", "Research Florida luxury properties.", "copy_verbatim_source"),
    ("title", "Research Florida luxury properties", "copy_verbatim_source"),
    ("fitReason", "I'll help because this is the third mention here and HIGHEST priority.", "copy_internal_note"),
    ("summary", "I’ll start now since this is top priority for you.", "copy_internal_note"),
])
def test_short_commitment_quotes_and_owner_notes_are_rejected(field, value, reason):
    with pytest.raises(ValueError, match=reason):
        validate_copy({**CLEAN, field: value}, COMMITMENT_SOURCES,
                      source_texts=["Research Florida luxury properties"])


def test_ordinary_paraphrase_of_a_short_commitment_is_accepted():
    assert validate_copy(CLEAN, COMMITMENT_SOURCES, source_texts=["Research Florida luxury properties"]) == CLEAN


@pytest.mark.asyncio
async def test_vault_failure_during_a_retry_supersede_keeps_every_published_card(env, commitments):
    for n in range(6):
        commitments.create(env.mind.owner_id, f"Review invoice batch {n}",
                           due_at=(env.clock[0] + timedelta(days=5)).isoformat())
    commitments.create(env.mind.owner_id, "Review urgent invoice packet",
                       due_at=(env.clock[0] + timedelta(hours=2)).isoformat())
    good = broken_first_pass(env, which={"Review urgent invoice packet"})
    env.model.retry = lambda offers: [{"index": o["index"], "content": good[o["anchor"]]} for o in offers]
    seal = env.ideas.seal
    def failing_seal(value):
        if value["quotes"][0]["quote"] == "Review urgent invoice packet":
            raise OSError("vault unavailable")
        return seal(value)
    env.ideas.seal = failing_seal
    rows = await debounced_refresh(env)
    assert len(rows) == 6 and len(env.ideas.catalog()) == 6
    last = env.ideas.meta()["last_refresh"]
    assert last["failure"] is None and last["copy_retry_errors"] == 1 and last["published"] == 6


def test_synthesis_splits_only_beyond_four_groups_and_keeps_threads_together(env):
    thread = [{"tool": "gmail_get", "account": "a", "server": "google", "action_id": "t-1", "key": f"m-{n}"}
              for n in range(3)]
    singles = [{"tool": "commitment", "account": "owner", "action_id": f"c-{n}", "key": f"c-{n}"} for n in range(5)]
    assert env.ideas.chunks(thread + singles[:3]) == [thread + singles[:3]]
    parts = env.ideas.chunks(thread + singles)
    assert len(parts) == 2 and sorted(s["key"] for p in parts for s in p) == sorted(s["key"] for s in thread + singles)
    assert any(all(s in part for s in thread) for part in parts)  # a thread is never split across calls


@pytest.mark.asyncio
async def test_a_failed_synthesis_call_keeps_the_other_calls_cards(env, commitments):
    for n in range(6):
        commitments.create(env.mind.owner_id, f"Review invoice batch {n}",
                           due_at=(env.clock[0] + timedelta(days=3)).isoformat())
    original, prompts = env.model.complete, []
    async def complete(messages, **kwargs):
        prompts.append(messages[0]["content"])
        if len(prompts) == 2:
            env.model.calls += 1
            raise TimeoutError("lane busy")
        return await original(messages, **kwargs)
    env.model.complete = complete
    rows = await debounced_refresh(env)
    assert len(rows) == 3 and env.model.calls == 2
    assert all("Select up to three" in p for p in prompts)
    last = env.ideas.meta()["last_refresh"]
    assert last["failure"] == "partial" and last["synthesis_failures"] == 1 and last["published"] == 3


@pytest.mark.asyncio
async def test_an_offer_citing_another_calls_source_is_dropped(env, commitments):
    for n in range(6):
        commitments.create(env.mind.owner_id, f"Review invoice batch {n}",
                           due_at=(env.clock[0] + timedelta(days=3)).isoformat())
    original, calls = env.model.complete, []
    async def complete(messages, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise TimeoutError("lane busy")
        # The first call answers as if it had seen every source, including the second call's.
        everything = json.dumps([{k: s[k] for k in ("key", "tool", "account", "value")}
                                 for s in env.ideas.evidence(env.clock[0])])
        return await original([messages[0], {**messages[-1], "content": everything}], **kwargs)
    env.model.complete = complete
    rows = await debounced_refresh(env)
    first_call = {s["key"] for s in env.ideas.chunks(env.ideas.evidence(env.clock[0]))[0]}
    assert len(rows) == 3
    assert all(env.ideas.open(r.context["idea"]["private"])["sources"][0]["key"] in first_call for r in rows)


@pytest.mark.asyncio
async def test_failed_refresh_cooldown_runs_from_when_it_finished(env, commitments, monkeypatch):
    add_open(env, commitments)
    await debounced_refresh(env)
    meta = env.ideas.meta()
    started, finished = env.clock[0] - timedelta(minutes=16), env.clock[0]
    meta["last_refresh"].update(at=started.isoformat(), finished_at=finished.isoformat(), failure="partial")
    env.ideas.save_meta(meta)
    triggers = []
    monkeypatch.setattr(env.ideas, "trigger", lambda now=None: triggers.append(now))
    await env.ideas.tick(finished + timedelta(seconds=61), schedule=True)
    assert triggers == []  # 16 minutes after the start, but only a minute after it finished
    await env.ideas.tick(finished + ideas_module.RETRY_AFTER, schedule=True)
    assert len(triggers) == 1


def wrap_replies(env, wrap):
    original = env.model.complete
    async def complete(messages, **kwargs):
        reply = await original(messages, **kwargs)
        reply.content = wrap(reply.content)
        return reply
    env.model.complete = complete


@pytest.mark.asyncio
@pytest.mark.parametrize("wrap", [
    lambda c: "```json\n" + c + "\n```",
    lambda c: "Here are the offers, as asked:\n" + c + "\nThat is all.",
    lambda c: "<think>budget reached</think>\n" + c,
])
async def test_answer_wrapped_in_fence_or_prose_still_publishes(env, commitments, wrap):
    add_open(env, commitments)
    wrap_replies(env, wrap)
    rows = await debounced_refresh(env)
    last = env.ideas.meta()["last_refresh"]
    assert len(rows) == 2 and last["failure"] is None and last["unparsed_replies"] == []


@pytest.mark.asyncio
async def test_wrapped_copy_retry_reply_is_recovered(env, commitments):
    add_open(env, commitments)
    good = broken_first_pass(env)
    env.model.retry = lambda offers: [{"index": o["index"], "content": good[o["anchor"]]} for o in offers]
    original = env.model.complete
    async def complete(messages, **kwargs):
        reply = await original(messages, **kwargs)
        if messages[0]["content"] == named(RETRY_PROMPT):
            reply.content = "```json\n" + reply.content + "\n```"
        return reply
    env.model.complete = complete
    rows = await debounced_refresh(env)
    last = env.ideas.meta()["last_refresh"]
    assert len(rows) == 2 and last["copy_recovered"] == 2 and last["copy_retry_errors"] == 0


@pytest.mark.asyncio
async def test_reply_without_json_records_its_shape_never_its_text(env, commitments):
    add_open(env, commitments)
    prose = "Let me think about the Florida invoice {draft"
    wrap_replies(env, lambda c: prose)
    assert await debounced_refresh(env) == []
    last = env.ideas.meta()["last_refresh"]
    assert last["failure"] == "AnswerUnavailable" and last["synthesis_failures"] == 1
    assert last["unparsed_replies"] == [{"chars": len(prose), "first": "text", "fence": False, "think_tag": False}]
    assert "Florida" not in json.dumps(env.ideas.meta())


@pytest.mark.parametrize("text,expected", [
    ('{"offers": []}', {"offers": []}),
    ('Sure. {"offers": [1]} Done.', {"offers": [1]}),
    ('```json\n{"offers": [2]}\n```', {"offers": [2]}),
    ('Here it is:\n```json\n{"offers": [3]}\n```\nThanks.', {"offers": [3]}),
])
def test_answer_reads_a_whole_fenced_or_prose_set_object(text, expected):
    assert ideas_module.answer(text, "offers") == expected


@pytest.mark.parametrize("text", ["", "   ", "[1, 2]", "no braces", '{"offers": [1', '{"other": 1}',
                                  '[{"offers":[]}]', '{"draft":{"offers":[]}', 'Example {"offers":[]} Final {"offers":[',
                                  'x {"a": 1} y {"offers": [1]}', '```\n{"offers": [0]}\n``` then ```json\n{"offers": [3]}\n```',
                                  'Example:\n```json\n{"offers":[1]}\n```\nFinal: {"offers":[', '[```{"offers":[1]}```'])
def test_answer_without_an_object_raises_with_shape(text):
    with pytest.raises(ideas_module.AnswerUnavailable) as raised:
        ideas_module.answer(text, "offers")
    assert set(raised.value.shape) == {"chars", "first", "fence", "think_tag"}


@pytest.mark.asyncio
async def test_unparsed_copy_retry_reply_records_its_shape(env, commitments):
    add_open(env, commitments)
    broken_first_pass(env)
    original = env.model.complete
    async def complete(messages, **kwargs):
        reply = await original(messages, **kwargs)
        if messages[0]["content"] == named(RETRY_PROMPT):
            reply.content = "I could not rewrite these."
        return reply
    env.model.complete = complete
    assert await debounced_refresh(env) == []
    last = env.ideas.meta()["last_refresh"]
    assert last["copy_retry_errors"] == 1 and last["copy_dropped"] == 2
    assert last["unparsed_replies"] == [{"chars": 26, "first": "text", "fence": False, "think_tag": False}]


@pytest.mark.parametrize("text", ['{"k":' * 8000, "```" + " " * 39996 + "x", "{" + "```" * 13000 + "}"])
def test_answer_is_linear_on_a_large_malformed_reply(text):
    import time
    started = time.monotonic()
    with pytest.raises(ideas_module.AnswerUnavailable):
        ideas_module.answer(text, "offers")
    assert time.monotonic() - started < 0.2


@pytest.mark.parametrize("text", ['<think>Example: {"offers":[1]}</think>\nNo offers.', '<think>{"offers":[1]}',
                                  '<think>a</think>{"offers":[1]}<think>b'])
def test_answer_inside_thinking_is_refused(text):
    with pytest.raises(ideas_module.AnswerUnavailable):
        ideas_module.answer(text, "offers")


def test_answer_after_thinking_is_read():
    assert ideas_module.answer('<think>{"offers":[0]}</think>\n{"offers":[1]}', "offers") == {"offers": [1]}
    assert ideas_module.answer('stray {"offers":[0]}</think>\n```json\n{"offers":[2]}\n```', "offers") == {"offers": [2]}
