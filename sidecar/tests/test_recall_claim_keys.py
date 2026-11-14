"""Claim keys that locate evidence (docs/SOURCE-CLAIMS.md, Recall and time; docs/RECALL-HYBRID.md, Keyword lanes).

Each claim's subject, predicate and value are indexed in a derived full-text table, keyed to the message that states
it. A question in the key's words ("who is my employer" for ``employer: Acme`` from "I just started at Acme") finds
the message through the ``claim_key`` lane, and recall expands it as any source hit: what is shown is the grounded
assertion bundle, never the key. The rows follow the claim's lineage, whoever changes it.

Everything runs the real ledger, projection and recall with a controlled extractor.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from colonymind.api.errors import install_exception_handlers
from colonymind.api.routers import host
from colonymind.beliefs.source_projection import SourceClaimProjection
from colonymind.beliefs.source_time import MemoryTimeQuery
from colonymind.memory import learning, search
from colonymind.memory import trace as recall_trace
from colonymind.memory.search import collect_sources, select_memory
from colonymind.memory.selection import RecallSelector, spread_across_sessions
from colonymind.turns import keywords
from colonymind.turns.idempotency import TurnIdempotencyLedger, source_message_hash
from test_source_claim_projection import Model, claim
from test_source_episode_memory import episode

SCOPE = {"contact_id": "owner", "session_id": "later"}
STARTED = "I just started at Acme."
QUESTION = "Who is my employer?"


def projection_at(tmp_path):
    return SourceClaimProjection(TurnIdempotencyLedger(tmp_path / "turn-idempotency.db"))


async def said(projection, turn, text, value, *, contact="owner", session=None, extra=(), model=None, **fields):
    """``text`` said by ``contact`` in its own source (with ``extra`` messages after it), and its claim formed."""
    projection.ledger.record_source(turn, contact_id=contact, session_id=session or "s-" + turn,
                                    messages=[{"role": "user", "content": text}, *extra],
                                    occurred_at="2026-05-01T10:00:00+00:00")
    assert await projection.process_one(model or Model({text: claim(text, value, **fields)}))


def key_rows(ledger):
    """The claim key index: each row's claim, source, message and indexed text."""
    from colonymind.turns import claim_keys
    with closing(ledger._connect()) as conn:
        return sorted(tuple(row) for row in conn.execute(
            f"SELECT k.claim_id, k.turn_id, k.message_hash, f.content FROM {claim_keys.ROWS} k "
            f"JOIN {claim_keys.TABLE} f ON f.rowid = k.id"))


def key_counts(ledger):
    from colonymind.turns import claim_keys
    with closing(ledger._connect()) as conn:
        return tuple(conn.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
                     for name in (claim_keys.TABLE, claim_keys.ROWS, claim_keys.PENDING))


def claims(ledger):
    with closing(ledger._connect()) as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM source_claims")]


# --- 1. A question in the key's words finds the bundle --------------------------------------------------------------

async def test_a_question_in_the_predicates_words_finds_the_claim_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    projection = projection_at(tmp_path)
    await said(projection, "start", STARTED, "Acme", predicate="employer")
    ledger = projection.ledger
    # No word of the question is in the message: neither keyword lane can reach it.
    assert ledger.search_sources(QUESTION, **SCOPE) == [] and ledger.search_stems(QUESTION, **SCOPE) == []
    hit, = ledger.search_claim_keys(QUESTION, **SCOPE)
    assert hit["turn_id"] == "start" and hit["role"] == "user" and hit["content"] == STARTED
    assert hit["source_message_hash"] == source_message_hash("s-start", {"role": "user", "content": STARTED})
    assert (hit["contact_id"], hit["session_id"], hit["scope"]) == ("owner", "s-start", "person")

    tracer = recall_trace.RecallTrace()
    collected = await collect_sources(ledger, query=QUESTION, **SCOPE, trace=tracer)
    assert collected.hits == [hit] and collected.lexical_hits == [hit]
    packet = await select_memory(collected, query=QUESTION, selector=RecallSelector(), timezone_name="UTC",
                                 trace=tracer)
    # The key only locates: the packet holds the grounded assertion bundle, not the key and not a loose quotation.
    row, = packet.selected
    card = json.loads(row["content"])
    assert row["content_format"] == "source_assertions_v1" and card["predicate"] == "employer"
    assert [(a["value"], a["quote"]) for a in card["assertions"]] == [("Acme", STARTED)]
    assert STARTED in packet.content and "Acme" in packet.content
    assert tracer._lane_hits["claim_key"] == 1
    traced, = [line for line in tracer.report()["rows"] if line["candidate"]]
    assert traced["lanes"] == {"claim_key": 1} and traced["exit"] is None


def test_the_claim_key_lane_is_registered_where_the_lanes_are_fused():
    lane = search.LANES[3]
    assert [lane.name for lane in search.LANES] == ["lexical", "vector", "stemmed", "claim_key", "window"]
    assert lane.keyword and not lane.optional and not lane.media


async def test_underscored_predicates_and_stemmed_forms_find_the_key(tmp_path):
    projection = projection_at(tmp_path)
    text = "The team moved us to the harbour annex."
    await said(projection, "annex", text, "the harbour annex", subject="team", predicate="current_workplace")
    ledger = projection.ledger
    for question in ("Which workplaces did I mention?", "what is the current workplace"):
        assert [hit["turn_id"] for hit in ledger.search_claim_keys(question, **SCOPE)] == ["annex"]
    (_, _, _, indexed), = key_rows(ledger)
    assert "_" not in indexed and "current workplace" in indexed


async def test_a_cjk_key_is_found_by_its_two_character_pieces(tmp_path):
    projection = projection_at(tmp_path)
    text = "我在东京工作"
    await said(projection, "cjk", text, "东京", subject="我", predicate="工作地点")
    assert [hit["turn_id"] for hit in projection.ledger.search_claim_keys("你的工作地点在哪里", **SCOPE)] == ["cjk"]
    (_, _, _, indexed), = key_rows(projection.ledger)
    assert indexed == keywords.cjk_bigrams("我 工作地点 东京")


# --- 2. Erasure leaves no key rows ----------------------------------------------------------------------------------


async def test_a_claim_deleted_by_any_writer_takes_its_rows(tmp_path):
    """The rows follow the claim in SQL: a claim deleted by any path (the recursive erasure of the claims built on an
    erased one, a writer this module does not know) takes its rows."""
    projection = projection_at(tmp_path)
    await said(projection, "start", STARTED, "Acme", predicate="employer")
    ledger = projection.ledger
    with closing(ledger._connect()) as conn, conn:
        conn.execute("DELETE FROM source_claims")
    assert key_counts(ledger) == (0, 0, 0) and ledger.search_claim_keys(QUESTION, **SCOPE) == []


# --- 3. Rejected claims get no rows ---------------------------------------------------------------------------------

class Refusing(Model):
    """Extracts as ``Model`` does; its review keeps nothing."""
    async def complete(self, messages, **kwargs):
        if kwargs.get("context", {}).get("task") == "source_claim_review":
            payload = json.loads(messages[-1]["content"])
            self.calls.append((payload, kwargs))
            from types import SimpleNamespace
            return SimpleNamespace(content=json.dumps({str(row["index"]): {
                "keep": False, "reason": "Controlled review refuses this fixture."}
                for row in payload["proposals"]}), model_id=self.model)
        return await super().complete(messages, **kwargs)


@pytest.mark.parametrize("refusal", ["evidence-not-in-source", "review"])
async def test_rejected_claims_get_no_key_rows(tmp_path, refusal):
    projection = projection_at(tmp_path)
    if refusal == "review":
        model = Refusing({STARTED: claim(STARTED, "Acme", predicate="employer")})
    else:
        model = Model({STARTED: claim("I started at Initech.", "Initech", predicate="employer")})
    await said(projection, "start", STARTED, "Acme", model=model)
    assert claims(projection.ledger) == []
    assert key_counts(projection.ledger) == (0, 0, 0)
    assert projection.ledger.search_claim_keys(QUESTION, **SCOPE) == []


async def test_a_dry_run_leaves_no_key_rows(tmp_path):
    projection = projection_at(tmp_path)
    await said(projection, "office", "My office is in River.", "River")
    before = key_rows(projection.ledger)
    result = await projection.rehearse(Model({STARTED: claim(STARTED, "Acme", predicate="employer")}),
                                       contact_id="owner", session_id="s-dry", text=STARTED,
                                       occurred_at="2026-05-02T10:00:00+00:00")
    assert result["status"] == "complete" and result["accepted"]
    assert key_rows(projection.ledger) == before and key_counts(projection.ledger)[2] == 0


# --- 4. Scope ---------------------------------------------------------------------------------------------------------

async def test_the_key_lane_keeps_the_persons_scope(tmp_path):
    projection = projection_at(tmp_path)
    await said(projection, "mine", STARTED, "Acme", predicate="employer")
    theirs = "I just started at Globex."
    await said(projection, "theirs", theirs, "Globex", contact="sam", predicate="employer")
    ledger = projection.ledger
    assert [hit["turn_id"] for hit in ledger.search_claim_keys(QUESTION, **SCOPE)] == ["mine"]
    assert [hit["turn_id"] for hit in ledger.search_claim_keys(QUESTION, contact_id="sam", session_id="x")] == [
        "theirs"]
    assert ledger.search_claim_keys(QUESTION, contact_id="", session_id="x") == []
    # A source whose attribution was invalidated is out of every viewer's scope.
    with closing(ledger._connect()) as conn, conn:
        conn.execute("INSERT INTO source_attribution_invalidations VALUES ('mine', 'mine', 'op')")
    assert ledger.search_claim_keys(QUESTION, **SCOPE) == []


async def test_a_corrected_value_is_reached_through_the_old_values_key(tmp_path, monkeypatch):
    """A retracted claim's key may stay: it locates, and expansion picks the value current at the question's time."""
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    projection = projection_at(tmp_path)
    old, new = "My office is in River.", "Correction: My office is in Lake, not River."
    await said(projection, "old", old, "River", predicate="workplace")
    projection.ledger.record_source("new", contact_id="owner", session_id="s-new",
                                    messages=[{"role": "user", "content": new}],
                                    occurred_at="2026-05-02T10:00:00+00:00")
    assert await projection.process_one(Model({new: claim(new, "Lake", predicate="workplace",
                                                          operation="correct", match_prior=True)}))
    hits = [hit for hit in projection.ledger.search_claim_keys("which workplace", **SCOPE) if hit["turn_id"] == "old"]
    assert hits
    _, rows = projection.prepare_context([], hits, **SCOPE, time_query=MemoryTimeQuery())
    row, = rows
    assert [a["value"] for a in json.loads(row["content"])["assertions"]] == ["Lake"]


# --- The index: built at commit, rebuilt the same, built on upgrade -------------------------------------------------

async def test_a_stale_key_row_never_spends_the_result_budget(tmp_path):
    from colonymind.turns import claim_keys
    projection = projection_at(tmp_path)
    await said(projection, "start", STARTED, "Acme", predicate="employer")
    with closing(projection.ledger._connect()) as conn, conn:
        # An indexed text that is not the key of the claim at its place (it cannot happen through the writers; the
        # check is what keeps it out if it ever did).
        conn.execute(f"UPDATE {claim_keys.TABLE} SET content='lighthouse keeper'")
    assert projection.ledger.search_claim_keys("lighthouse", **SCOPE) == []


async def test_a_rebuild_is_idempotent_and_matches_what_the_commit_wrote(tmp_path, monkeypatch):
    monkeypatch.setenv(learning.FLAG, "1")
    projection = projection_at(tmp_path)
    await said(projection, "first", STARTED, "Acme", predicate="employer")
    await said(projection, "again", STARTED, "Acme", predicate="employer")
    await said(projection, "office", "My office is in River.", "River")
    ledger = projection.ledger
    written = key_rows(ledger)
    assert len(written) == 3
    assert ledger.rebuild_claim_keys() == 2                                  # two claims, three locations
    assert key_rows(ledger) == written
    assert ledger.rebuild_claim_keys() == 2
    assert key_rows(ledger) == written and key_counts(ledger) == (3, 3, 0)


def downgrade(path):
    """The store as 1.11 left it: no claim key index, no triggers, no queue."""
    from colonymind.turns import claim_keys
    with closing(sqlite3.connect(path)) as conn, conn:
        for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE ?",
                                    (claim_keys.TABLE + "%",)).fetchall():
            conn.execute(f'DROP TRIGGER "{name}"')
        for name in (claim_keys.TABLE, claim_keys.ROWS, claim_keys.PENDING):
            conn.execute(f'DROP TABLE "{name}"')


async def test_the_upgrade_builds_the_key_index_for_an_existing_store(tmp_path, caplog):
    projection = projection_at(tmp_path)
    await said(projection, "start", STARTED, "Acme", predicate="employer")
    await said(projection, "office", "My office is in River.", "River")
    written = key_rows(projection.ledger)
    downgrade(tmp_path / "turn-idempotency.db")
    with caplog.at_level("INFO", logger="colonymind.turns.idempotency"):
        upgraded = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    assert [row[1:] for row in key_rows(upgraded)] == [row[1:] for row in written]
    assert [hit["turn_id"] for hit in upgraded.search_claim_keys(QUESTION, **SCOPE)] == ["start"]
    assert any("claim key index" in record.getMessage() and "2 claims" in record.getMessage()
               for record in caplog.records)


async def test_claims_written_by_an_older_version_are_indexed_when_the_ledger_opens(tmp_path):
    projection = projection_at(tmp_path)
    await said(projection, "office", "My office is in River.", "River")
    ledger = projection.ledger
    with closing(sqlite3.connect(ledger.db_path)) as conn, conn:
        # What 1.11 writes: a source and its claim, no key rows (the queue trigger is in the store's schema).
        conn.execute("INSERT INTO turn_sources(turn_id, content_sha256, contact_id, session_id, scope, messages_json)"
                     " VALUES ('older', ?, 'owner', 's-older', 'person', ?)",
                     ("0" * 64, json.dumps([{"role": "user", "content": STARTED}])))
        data = {"subject": "I", "predicate": "employer", "value": "Acme", "evidence": STARTED,
                "span_start": 0, "span_end": len(STARTED)}
        conn.execute("INSERT INTO source_claims(id, turn_id, message_hash, subject_key, predicate, value_key, "
                     "data_json) VALUES ('claim:older', 'older', ?, 'speaker', 'employer', 'acme', ?)",
                     (source_message_hash("s-older", {"role": "user", "content": STARTED}), json.dumps(data)))
    assert key_counts(ledger)[2] == 1 and ledger.search_claim_keys(QUESTION, **SCOPE) == []
    reopened = TurnIdempotencyLedger(ledger.db_path)
    assert key_counts(reopened)[2] == 0
    assert [hit["turn_id"] for hit in reopened.search_claim_keys(QUESTION, **SCOPE)] == ["older"]


async def test_an_episode_is_not_given_key_rows(tmp_path):
    """An episode's subject and predicate are fixed words and its value is the message itself, which the keyword
    lanes already search: its rows would only match every episode on "reported" or "episode"."""
    projection = projection_at(tmp_path)
    text = "Yesterday we repaired the gate hinge together."
    await said(projection, "gate", text, text, model=Model({text: episode(text)}))
    assert [row["subject_key"].startswith("episode:") for row in claims(projection.ledger)] == [True]
    assert key_counts(projection.ledger) == (0, 0, 0)


# --- A value sealed or released: the keys are located again at once --------------------------------------------------

TOKEN = "hf_" + "AbCdEfGhIjKlMnOpQrStUvWxYz012345"     # synthetic, assembled at run time


class _EveryMessage(dict):
    """The extractor's answer for any message: the employer claim quoting the first sentence."""

    def __getitem__(self, message):
        return claim(STARTED, "Acme", predicate="employer")


@pytest.mark.parametrize("rewrite", ["sealed by the migration", "released"])
async def test_a_claim_is_located_at_once_after_its_message_is_sealed_or_released(tmp_path, monkeypatch, rewrite):
    """Sealing a value or releasing it changes the words of the message it is in, so every identity the message had
    is re-pointed, and the claims it states move with it (their rows are forgotten and queued). The rewrite indexes
    the queue when it is done: a claim the value never touched is found by its key at once, not after a restart."""
    from colonymind.secrets.config import SecretsSettings, set_settings
    from colonymind.secrets.migrate import migrate, release
    from colonymind.secrets.vault import Vault, reset_vaults
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    reset_vaults()
    try:
        if rewrite == "sealed by the migration":
            set_settings(SecretsSettings(capture_enabled=False))      # stored as said, sealed later
        projection = projection_at(tmp_path)
        ledger = projection.ledger
        await said(projection, "start", f"{STARTED} My token is {TOKEN}.", "Acme", model=Model(_EveryMessage()))
        set_settings(None)
        assert [hit["turn_id"] for hit in ledger.search_claim_keys(QUESTION, **SCOPE)] == ["start"]
        before = key_rows(ledger)
        if rewrite == "sealed by the migration":
            migrate(tmp_path, owner_id="owner", write_report=False)
        else:
            vault = Vault(tmp_path)
            ref, = [entry["ref"] for entry in vault.entries()]
            release(ref, tmp_path, vault=vault)
        with closing(ledger._connect()) as conn:
            message, = json.loads(conn.execute("SELECT messages_json FROM turn_sources WHERE turn_id='start'")
                                  .fetchone()[0])
        assert (TOKEN in message["content"]) == (rewrite == "released")
        after = key_rows(ledger)
        assert [row[1:2] + row[3:] for row in after] == [row[1:2] + row[3:] for row in before]
        assert after[0][2] == source_message_hash("s-start", message) != before[0][2]
        assert key_counts(ledger)[2] == 0                           # nothing left queued
        hit, = ledger.search_claim_keys(QUESTION, **SCOPE)
        assert (hit["turn_id"], hit["content"]) == ("start", STARTED)
        collected = await collect_sources(ledger, query=QUESTION, **SCOPE)
        packet = await select_memory(collected, query=QUESTION, selector=RecallSelector(), timezone_name="UTC")
        assert STARTED in packet.content and "Acme" in packet.content
    finally:
        set_settings(None)
        reset_vaults()


# --- What the person sees: context assembly recalls a claim through its key ------------------------------------------

@pytest.fixture
def source_app(monkeypatch, tmp_path):
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    for name in ("_contacts_store", "_telemetry", "_reranker", "_context_recall_selector", "_comms_log"):
        monkeypatch.setattr(host, name, None)
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(host.router)
    app.include_router(host.v2_router)
    return app


async def test_context_assembly_recalls_a_claim_only_its_key_finds(source_app, tmp_path):
    async with AsyncClient(transport=ASGITransport(app=source_app), base_url="http://test") as client:
        response = await client.put("/v2/host/turns/turn-said", json={
            "identity": {"host_id": "test-host"},
            "context": {"session_id": "session-a", "contact_id": "contact-a", "channel_id": "test:thread-a",
                        "turn_id": "turn-said"},
            "user_message": {"role": "user", "content": STARTED},
            "assistant_message": {"role": "assistant", "content": "Noted."}})
        assert response.status_code == 201, response.text
        projection = SourceClaimProjection(TurnIdempotencyLedger(tmp_path / "turn-idempotency.db"))
        assert await projection.process_one(Model({STARTED: claim(STARTED, "Acme", predicate="employer")}))
        response = await client.post("/v1/host/context/assemble", json={
            "identity": {"host_id": "test-host"},
            "context": {"contact_id": "contact-a", "session_id": "session-b"},
            "incoming_message": {"role": "user", "content": QUESTION}})
        assert response.status_code == 200, response.text
        memory = "\n".join(section["body"] for section in response.json()["sections"]
                           if section["id"] == "colonymind-memory")
        assert STARTED in memory and "employer" in memory and "Noted." not in memory


# --- Carried from the second wave: an assertion bundle names its anchor source's session ----------------------------

async def test_an_assertion_bundle_carries_its_anchor_sources_session(tmp_path):
    """The reranker's batch is spread across sessions (``selection.spread_across_sessions``). A projected assertion
    bundle is grouped by its key, not by a session, so it names the session of the source it is anchored in: two
    properties stated in one conversation count as that conversation, not as two."""
    projection = projection_at(tmp_path)
    said_in = {"office": ("My office is in River.", "River", "office_location", "conversation-1"),
               "desk": ("My desk is by the window.", "by the window", "desk_location", "conversation-1"),
               "car": ("My car is blue.", "blue", "car_colour", "conversation-2")}
    for turn, (text, value, predicate, session) in said_in.items():
        await said(projection, turn, text, value, predicate=predicate, session=session)
    ledger = projection.ledger
    hits = [hit for turn in said_in for hit in ledger.search_sources(said_in[turn][0], **SCOPE) if hit["turn_id"] == turn]
    _, rows = projection.prepare_context([], hits, **SCOPE, time_query=MemoryTimeQuery())
    bundles = [row for row in rows if row.get("content_format") == "source_assertions_v1"]
    assert {row["history_anchor"]["source_id"]: row["session_id"] for row in bundles} == {
        "office": "conversation-1", "desk": "conversation-1", "car": "conversation-2"}
    chosen = spread_across_sessions(bundles, 2)
    assert {row["history_anchor"]["source_id"] for row in chosen} == {"office", "car"}

from colonymind.memory import curated

async def test_erasing_a_source_leaves_no_key_rows(tmp_path):
    projection = projection_at(tmp_path)
    await said(projection, "start", STARTED, "Acme", predicate="employer")
    await said(projection, "office", "My office is in River.", "River")
    ledger = projection.ledger
    assert {row[1] for row in key_rows(ledger)} == {"start", "office"}
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(ledger)
    _source_forget_receipt = curated.forget_sources(ledger, 'owner', source_ids=['start'])
    assert historical_bytes(ledger) == _native_before_forget
    assert all(curated.is_forgotten(ledger, sid) for sid in _source_forget_receipt['source_ids'])


    assert ledger.search_claim_keys(QUESTION, **SCOPE) == []


async def test_erasing_one_message_keeps_the_keys_of_the_others(tmp_path):
    projection = projection_at(tmp_path)
    erased, kept = ('I just started at Acme.', 'My office is in River.')
    projection.ledger.record_source('both', contact_id='owner', session_id='s-both', messages=[{'role': 'user', 'content': erased}, {'role': 'user', 'content': kept}])
    assert await projection.process_one(Model({erased: claim(erased, 'Acme', predicate='employer'), kept: claim(kept, 'River')}))
    ledger = projection.ledger
    assert len(key_rows(ledger)) == 2
    ledger.record_source('copy', contact_id='owner', session_id='s-both', derive_claims=False, messages=[{'role': 'user', 'content': erased}])
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, 'owner', source_ids=['copy'])
    assert historical_bytes(ledger) == _native_before_withdrawal
    assert len(claims(ledger)) == 2
    (_, turn, digest, indexed), = [row for row in key_rows(ledger) if row[2] == source_message_hash('s-both', {'role': 'user', 'content': kept})]
    assert turn == 'both' and digest == source_message_hash('s-both', {'role': 'user', 'content': kept})
    assert 'Acme' not in indexed and ledger.search_claim_keys(QUESTION, **SCOPE) == []
    assert [hit['content'] for hit in ledger.search_claim_keys('office location', **SCOPE)] == [kept]

async def test_a_restated_claim_is_located_at_each_witness_and_erased_with_each(tmp_path, monkeypatch):
    monkeypatch.setenv(learning.FLAG, '1')
    projection = projection_at(tmp_path)
    await said(projection, 'first', STARTED, 'Acme', predicate='employer')
    await said(projection, 'again', STARTED, 'Acme', predicate='employer')
    ledger = projection.ledger
    row, = claims(ledger)
    assert row['witnesses'] == 2 and row['turn_id'] == 'again'
    assert {(key[0], key[1]) for key in key_rows(ledger)} == {(row['id'], 'first'), (row['id'], 'again')}
    assert {hit['turn_id'] for hit in ledger.search_claim_keys(QUESTION, **SCOPE)} == {'first', 'again'}
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, 'owner', source_ids=['again'])
    assert historical_bytes(ledger) == _native_before_withdrawal
    assert {(key[0], key[1]) for key in key_rows(ledger)} == {(row['id'], 'first'), (row['id'], 'again')}
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, 'owner', source_ids=['first'])
    assert historical_bytes(ledger) == _native_before_withdrawal
    assert len(key_rows(ledger)) == 2
    assert ledger.search_claim_keys(QUESTION, **SCOPE) == []

async def test_an_erased_correction_takes_its_key_rows_and_revives_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv('COLONYMIND_RECALL_RERANK', 'off')
    projection = projection_at(tmp_path)
    old, new = ('My office is in River.', 'Correction: My office is in Lake, not River.')
    await said(projection, 'old', old, 'River', predicate='workplace')
    projection.ledger.record_source('new', contact_id='owner', session_id='s-new', messages=[{'role': 'user', 'content': new}], occurred_at='2026-05-02T10:00:00+00:00')
    assert await projection.process_one(Model({new: claim(new, 'Lake', predicate='workplace', operation='correct', match_prior=True)}))
    ledger = projection.ledger
    assert {row[1] for row in key_rows(ledger)} == {'old', 'new'}
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, 'owner', source_ids=['new'])
    assert historical_bytes(ledger) == _native_before_withdrawal
    assert {row[1] for row in key_rows(ledger)} == {'old', 'new'}
    assert any(('Lake' in row[3] for row in key_rows(ledger)))
    hits = ledger.search_claim_keys('which workplace', **SCOPE)
    assert [hit['turn_id'] for hit in hits] == ['old']
    _, rows = projection.prepare_context([], hits, **SCOPE, time_query=MemoryTimeQuery())
    assert not any(('River' in str(row.get('content')) for row in rows))
