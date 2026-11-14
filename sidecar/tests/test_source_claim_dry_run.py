"""``POST /v1/host/claims/dry-run``: what a message would become, with nothing written.

The owner (a credential that manages the instance) names a person and supplies text. The dry run runs extraction
with that person's priors and key list, every validation step, the review, and the commit's own checks in a
transaction it rolls back, and returns the accepted and rejected proposals with their reasons and the tokens the
calls reported. ``prompt_only`` returns the prompts extraction would send, with no model call.
"""
from contextlib import closing
from secrets import token_hex

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from onekey import KEY
from colonymind.api.credentials import CredentialTable, secret_digest
from colonymind.api.errors import install_exception_handlers
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import claims as claims_routes
from colonymind.api.routers import host
from colonymind.beliefs.source_claims import SYSTEM
from colonymind.beliefs.source_projection import SourceClaimProjection
from colonymind.turns import get_turn_idempotency_ledger
from colonymind.turns.source_vectors import SourceVectors
from colonymind.vector.indexes import IndexCatalog
from colonymind.vector.store import VectorStore
from test_source_claim_priors import Pipeline, drain
from test_source_claim_projection import claim

GUEST = token_hex(24)
USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
OFFICE, TEA = "My office is in River.", "My tea preference is green tea."
TEXT = ("Correction: my office is in Lake, not River. My tea preference is green tea. "
        "My bike is at the station. I like Nowhere music.")


class Scripted:
    """Extraction returns the scripted proposals for a message; review keeps all but a value it is told to refuse.

    ``choose`` fills a proposal's prior with the offered record of that predicate, as a model would."""

    def __init__(self, outputs, refuse=()):
        self.outputs, self.refuse, self.calls = outputs, set(refuse), []

    def tier_config(self, tier):
        return SimpleNamespace(base_url="http://127.0.0.1:8080/v1")

    async def complete(self, messages, **kwargs):
        payload = json.loads(messages[-1]["content"])
        task = kwargs.get("context", {}).get("task")
        self.calls.append((task, payload))
        if task == "source_claim_review":
            body = {str(row["index"]): {"keep": row["claim"]["value"] not in self.refuse,
                                        "reason": "Scripted review of this fixture."}
                    for row in payload["proposals"]}
            return SimpleNamespace(content=json.dumps(body), model_id="fixture-model", usage=USAGE)
        proposals = []
        for proposal in self.outputs[payload["message"]]:
            proposal = dict(proposal)
            wanted = proposal.pop("choose", None)
            if wanted:
                proposal["prior_claim_id"] = next(row["id"] for row in payload["prior_assertions"]
                                                  if row["predicate"] == wanted)
            proposals.append(proposal)
        return SimpleNamespace(content=json.dumps(proposals), model_id="fixture-model", usage=USAGE)


def proposals_for_text():
    return [claim(TEXT, "Lake", operation="correct") | {"choose": "office location"},
            claim(TEXT, "green tea", predicate="tea_preference"),
            claim(TEXT, "the station", predicate="bike_location"),
            claim(TEXT, "purple", predicate="bike_colour"),           # not in the text: not grounded
            claim(TEXT, "Nowhere music", predicate="music_preference")]  # refused by review


def app_with_guest():
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(host.router)
    app.include_router(claims_routes.router)
    app.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=CredentialTable.from_config({
        "api-key": {"owner": ["whatsapp"], "contacts": "all"},
        "relay": {"sha256": secret_digest(GUEST), "owner": [], "contacts": "all"}}))
    return app


async def post(app, body, *, key=KEY, base_url="http://test"):
    headers = {"Authorization": "Bearer " + key} if key else {}
    async with AsyncClient(transport=ASGITransport(app=app), base_url=base_url, headers=headers) as client:
        return await client.post("/v1/host/claims/dry-run", json=body)


@pytest.fixture
def state(monkeypatch, tmp_path):
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_LEARNING", "on")
    for name in ("_contacts_store", "_telemetry", "_reranker", "_context_recall_selector", "_comms_log"):
        monkeypatch.setattr(host, name, None)
    return tmp_path


async def seed(model, *, contact="contact-a", texts=(OFFICE, TEA), start=0):
    ledger = get_turn_idempotency_ledger(Path(host.get_state_dir()))
    projection = SourceClaimProjection(ledger)
    for index, text in enumerate(texts, start):
        ledger.record_source(f"{contact}-{index}", contact_id=contact, session_id=f"s{index}",
                             occurred_at=f"2026-03-0{index + 1}T09:00:00+00:00",
                             messages=[{"role": "user", "content": text}])
        assert await projection.process_one(model)
    return ledger, projection


def counts(ledger):
    with closing(ledger._connect()) as conn:
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        return {table: conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0] for table in tables}


def files(root):
    return sorted(str(path.relative_to(root)) for path in Path(root).rglob("*")
                  if path.is_file() and not path.name.endswith(("-wal", "-shm", "-journal")))


def body(**changes):
    return {"person_id": "contact-a", "session_id": "later", "text": TEXT,
            "occurred_at": "2026-03-10T09:00:00+00:00", "timezone": "UTC", **changes}


@pytest.mark.asyncio
async def test_a_dry_run_writes_nothing_in_any_store(state, monkeypatch):
    import colonymind.vector as vector
    from colonymind.vector import Collection
    model = Scripted({OFFICE: [claim(OFFICE, "River")], TEA: [claim(TEA, "green tea", predicate="tea_preference")],
                      TEXT: proposals_for_text()}, refuse={"Nowhere music"})
    ledger, _ = await seed(model)
    index, pipeline = VectorStore(str(state / "lancedb"), identity=Pipeline.index_identity,
                                  catalog=IndexCatalog(ledger)), Pipeline()
    await index.connect(3)
    await index.ensure_collections(3)
    await drain(SourceVectors(ledger, index, pipeline))
    monkeypatch.setattr(vector, "_store", index)
    monkeypatch.setattr(vector, "_pipeline", pipeline)
    monkeypatch.setattr(host, "_llm_router", model)

    async def vector_rows():
        return await (await index._table(Collection.CONVERSATIONS)).count_rows()
    before, listing, projected = counts(ledger), files(state), await vector_rows()
    assert projected
    for changes in ({}, {"prompt_only": True}):
        response = await post(app_with_guest(), body(**changes))
        assert response.status_code == 200, response.text
        assert counts(ledger) == before
        assert files(state) == listing
        assert await vector_rows() == projected


@pytest.mark.asyncio
async def test_dry_run_verdicts_equal_a_real_commit_of_the_same_input(state, monkeypatch):
    model = Scripted({OFFICE: [claim(OFFICE, "River")], TEA: [claim(TEA, "green tea", predicate="tea_preference")],
                      TEXT: proposals_for_text()}, refuse={"Nowhere music"})
    ledger, projection = await seed(model)
    monkeypatch.setattr(host, "_llm_router", model)
    model.calls.clear()
    response = await post(app_with_guest(), body())
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "complete" and result["writes"] == "none"
    assert result["extraction_version"] == "source-claims-v19"
    assert result["usage"] == {"calls": 2, **{key: 2 * value for key, value in USAGE.items()}}
    river = next(row["id"] for row in result["offered_priors"] if row["value"] == "River")
    tea = next(row["id"] for row in result["offered_priors"] if row["value"] == "green tea")
    accepted = {row["claim"]["value"]: row for row in result["accepted"]}
    assert set(accepted) == {"Lake", "green tea", "the station"}
    assert accepted["Lake"]["effect"] == "new" and accepted["Lake"]["retracts"] == [river]
    assert accepted["green tea"]["effect"] == "restates" and accepted["green tea"]["claim_id"] == tea
    assert accepted["the station"]["effect"] == "new" and not accepted["the station"]["retracts"]
    rejected = {(row["stage"], row["reason"]) for row in result["rejected"]}
    assert rejected == {("validation", "value_not_grounded"), ("review", "Scripted review of this fixture.")}

    # The same message committed for real, under the same scripted model.
    ledger.record_source("real", contact_id="contact-a", session_id="later", occurred_at="2026-03-10T09:00:00+00:00",
                         messages=[{"role": "user", "content": TEXT}])
    assert await projection.process_one(model)
    with closing(ledger._connect()) as conn:
        rows = {row["id"]: dict(row) | json.loads(row["data_json"]) for row in conn.execute("SELECT * FROM source_claims")}
        job = conn.execute("SELECT diagnostics_json FROM source_claim_jobs WHERE turn_id='real'").fetchone()
    new = {row["value"]: row for row in rows.values() if row["turn_id"] == "real"}
    assert set(new) - {"green tea"} == {"Lake", "the station"}
    assert rows[river]["retracted_by"] == new["Lake"]["id"]
    assert rows[tea]["witnesses"] == 2
    diagnostics = json.loads(job[0])
    assert diagnostics["rejection_counts"] == {"value_not_grounded": 1}
    assert diagnostics["review_rejected_count"] == 1


@pytest.mark.asyncio
async def test_a_guest_or_a_keyless_caller_is_refused(state, monkeypatch):
    model = Scripted({TEXT: proposals_for_text()})
    monkeypatch.setattr(host, "_llm_router", model)
    refused = await post(app_with_guest(), body(), key=GUEST)
    assert refused.status_code == 403
    dev = FastAPI()
    install_exception_handlers(dev)
    dev.include_router(claims_routes.router)
    dev.add_middleware(ApiKeyMiddleware, api_key=None)
    keyless = await post(dev, body(), key=None, base_url="http://localhost")
    assert keyless.status_code == 403
    # Mounted without the middleware (an in-process caller that did not authenticate): still refused.
    bare = FastAPI()
    install_exception_handlers(bare)
    bare.include_router(claims_routes.router)
    unauthenticated = await post(bare, body(), key=None)
    assert unauthenticated.status_code == 403
    assert unauthenticated.json()["detail"]["code"] == "dry_run_not_granted"
    assert model.calls == []


@pytest.mark.asyncio
async def test_the_dry_run_reads_only_the_named_persons_records(state, monkeypatch):
    model = Scripted({OFFICE: [claim(OFFICE, "River")], TEA: [claim(TEA, "green tea", predicate="tea_preference")],
                      "My office is in Harbour.": [claim("My office is in Harbour.", "Harbour")],
                      "My locker word is blue.": [claim("My locker word is blue.", "blue", predicate="locker_word")],
                      TEXT: proposals_for_text()}, refuse={"Nowhere music"})
    await seed(model)
    await seed(model, contact="contact-b", texts=("My office is in Harbour.", "My locker word is blue."), start=2)
    monkeypatch.setattr(host, "_llm_router", model)
    response = await post(app_with_guest(), body(prompt_only=True))
    assert response.status_code == 200, response.text
    result = response.json()
    assert {row["value"] for row in result["offered_priors"]} == {"River", "green tea"}
    assert {key["predicate"] for key in result["known_keys"]} == {"office location", "tea preference"}
    assert "Harbour" not in response.text and "locker" not in response.text
    missing = await post(app_with_guest(), {k: v for k, v in body().items() if k != "person_id"})
    assert missing.status_code == 422


@pytest.mark.asyncio
async def test_prompt_only_returns_the_extraction_prompts_with_no_model_call(state, monkeypatch):
    model = Scripted({OFFICE: [claim(OFFICE, "River")], TEA: [claim(TEA, "green tea", predicate="tea_preference")]})
    await seed(model)
    model.calls.clear()
    monkeypatch.setattr(host, "_llm_router", model)
    response = await post(app_with_guest(), body(prompt_only=True))
    assert response.status_code == 200, response.text
    result = response.json()
    assert model.calls == []
    assert result["status"] == "prompt_only" and result["writes"] == "none"
    system, user = result["prompts"]
    assert system == {"role": "system", "content": SYSTEM}
    payload = json.loads(user["content"])
    assert user["role"] == "user" and payload["message"] == TEXT
    assert {row["value"] for row in payload["prior_assertions"]} == {"River", "green tea"}
    assert payload["known_keys"] == result["known_keys"]
    assert result["response_schema"]["name"] == "source_claims"
    # With no model configured, the prompt form still answers; a full dry run cannot.
    monkeypatch.setattr(host, "_llm_router", None)
    assert (await post(app_with_guest(), body(prompt_only=True))).status_code == 200
    unavailable = await post(app_with_guest(), body())
    assert unavailable.status_code == 503


def test_the_route_is_registered_with_the_app(monkeypatch):
    monkeypatch.setenv("COLONYMIND_API_KEY", token_hex(24))
    from colonymind.server import create_app
    assert "post" in create_app().openapi()["paths"]["/v1/host/claims/dry-run"]


# -- a secret in the supplied text ----------------------------------------------------------------------------

PASSWORD = token_hex(12)                     # Generated synthetic detector fixture.
SECRET_TEXT = "Correction: my office is in Lake, not River. The wifi password is " + PASSWORD


class Recording(Pipeline):
    """The fixture embedding, keeping every text it is asked to embed as a query."""

    def __init__(self):
        super().__init__()
        self.queries = []

    async def embed_query(self, text):
        self.queries.append(text)
        return await super().embed_query(text)


def vault_rows(state):
    with closing(__import__("sqlite3").connect(state / "secrets-vault.db")) as conn:
        return conn.execute("SELECT ref,scope_contact,scope_session,source_turn,uses FROM sealed ORDER BY ref").fetchall()


@pytest.mark.asyncio
async def test_a_secret_in_the_supplied_text_is_sealed_before_any_model_sees_it(state, monkeypatch):
    import colonymind.vector as vector
    from colonymind.secrets.detect import REFERENCE
    seen = "The password for gmail is Tr0ub4dor&3"    # a value sealed earlier: the vault exists, with its key
    model = Scripted({OFFICE: [claim(OFFICE, "River")], TEA: [claim(TEA, "green tea", predicate="tea_preference")]})
    model.outputs[SECRET_TEXT] = []
    ledger, projection = await seed(model)
    ledger.record_source("contact-a-9", contact_id="contact-a", session_id="s9",
                         occurred_at="2026-03-09T09:00:00+00:00", messages=[{"role": "user", "content": seen}])
    index, pipeline = VectorStore(str(state / "lancedb"), identity=Pipeline.index_identity,
                                  catalog=IndexCatalog(ledger)), Recording()
    await index.connect(3)
    await index.ensure_collections(3)
    await drain(SourceVectors(ledger, index, pipeline))
    monkeypatch.setattr(vector, "_store", index)
    monkeypatch.setattr(vector, "_pipeline", pipeline)
    extracted = []

    async def extraction(messages, **kwargs):
        payload = json.loads(messages[-1]["content"])
        extracted.append(messages)
        model.outputs.setdefault(payload["message"], [])
        return await Scripted.complete(model, messages, **kwargs)
    monkeypatch.setattr(model, "complete", extraction)
    monkeypatch.setattr(host, "_llm_router", model)
    before, listing, sealed = counts(ledger), files(state), vault_rows(state)
    assert sealed
    for changes in ({"prompt_only": True}, {}):
        pipeline.queries.clear()
        response = await post(app_with_guest(), body(text=SECRET_TEXT, **changes))
        assert response.status_code == 200, response.text
        assert PASSWORD not in response.text
        # The semantic lane's query embedding and the extraction prompt both carry a reference, never the value.
        assert pipeline.queries and not any(PASSWORD in query for query in pipeline.queries)
        assert all(REFERENCE.search(query) for query in pipeline.queries)
        prompts = response.json()["prompts"] if changes else extracted[-1]
        message = json.loads(prompts[-1]["content"])["message"]
        assert PASSWORD not in message and REFERENCE.search(message)
        assert counts(ledger) == before and files(state) == listing and vault_rows(state) == sealed
    assert extracted and not any(PASSWORD in json.dumps(call) for call in extracted)


@pytest.mark.asyncio
async def test_with_no_vault_yet_a_dry_run_withholds_the_secret_and_makes_no_vault(state, monkeypatch):
    from colonymind.secrets.detect import WITHHELD
    model = Scripted({OFFICE: [claim(OFFICE, "River")], TEA: [claim(TEA, "green tea", predicate="tea_preference")]})
    ledger, _ = await seed(model)
    monkeypatch.setattr(host, "_llm_router", model)
    listing = files(state)
    assert not [name for name in listing if name.startswith("secrets-vault")]
    response = await post(app_with_guest(), body(text=SECRET_TEXT, prompt_only=True))
    assert response.status_code == 200, response.text
    assert PASSWORD not in response.text
    assert WITHHELD in json.loads(response.json()["prompts"][-1]["content"])["message"]
    assert files(state) == listing


# -- an erased message -------------------------------------------------------------------------------------------


from colonymind.memory import curated
@pytest.mark.asyncio
async def test_a_dry_run_of_an_erased_message_is_refused_as_ingestion_refuses_it(state, monkeypatch):
    from colonymind.turns.idempotency import SourceErased
    said = 'My locker is at the north gate. The wifi password is ' + PASSWORD
    model = Scripted({OFFICE: [claim(OFFICE, 'River')], TEA: [claim(TEA, 'green tea', predicate='tea_preference')], said: []})
    ledger, projection = await seed(model, texts=(OFFICE, TEA, said))
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, 'contact-a', source_ids=['contact-a-0', 'contact-a-2'])
    assert historical_bytes(ledger) == _native_before_withdrawal
    for text, session in ((OFFICE, 's0'), (said, 's2')):
        assert ledger.record_source('replay-' + session, contact_id='contact-a', session_id=session, messages=[{'role': 'user', 'content': text}])
        assert curated.is_forgotten(ledger, 'replay-' + session)
    monkeypatch.setattr(host, '_llm_router', model)
    model.calls.clear()
    before, listing = (counts(ledger), files(state))
    for text, session in ((OFFICE, 's0'), (said, 's2')):
        for changes in ({}, {'prompt_only': True}):
            response = await post(app_with_guest(), body(text=text, session_id=session, **changes))
            assert response.status_code == 200, response.text
            result = response.json()
            assert result['status'] == 'source_erased' and result['writes'] == 'none', result
            assert not {'accepted', 'prompts', 'offered_priors'} & set(result)
    assert model.calls == []
    assert counts(ledger) == before and files(state) == listing
    other = await post(app_with_guest(), body(text=OFFICE, session_id='elsewhere'))
    assert other.status_code == 200 and other.json()['status'] == 'complete'
    ref, = [row[0] for row in vault_rows(state)]
    assert ledger.vault.forget(ref)
    assert ledger.record_source('replay-forgotten', contact_id='contact-a', session_id='s2', messages=[{'role': 'user', 'content': said}])
    assert curated.is_forgotten(ledger, 'replay-forgotten')
    sealed = vault_rows(state)
    model.calls.clear()
    for changes in ({}, {'prompt_only': True}):
        response = await post(app_with_guest(), body(text=said, session_id='s2', **changes))
        assert response.status_code == 200 and response.json()['status'] == 'source_erased', response.text
    assert model.calls == [] and vault_rows(state) == sealed

@pytest.mark.asyncio
async def test_when_the_vault_key_cannot_be_read_an_erasure_check_answers_503_as_ingestion_does(state, monkeypatch):
    from colonymind.secrets import vault as vaults
    said = 'My locker is at the north gate. The wifi password is ' + PASSWORD
    model = Scripted({OFFICE: [claim(OFFICE, 'River')], TEA: [claim(TEA, 'green tea', predicate='tea_preference')], said: []})
    ledger, _ = await seed(model, texts=(OFFICE, TEA, said))
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, 'contact-a', source_ids=['contact-a-2'])
    assert historical_bytes(ledger) == _native_before_withdrawal
    (state / 'secrets-vault.key').chmod(420)
    monkeypatch.setattr(vaults, '_VAULTS', {})
    monkeypatch.setattr(host, '_llm_router', model)
    model.calls.clear()
    response = await post(app_with_guest(), body(text=said, session_id='s2'))
    assert response.status_code == 503, response.text
    assert response.json()['detail']['code'] == 'secrets_vault_unavailable'
    assert PASSWORD not in response.text and model.calls == []
