"""A recall says why it kept or left out each candidate (docs/RECALL-HYBRID.md, "Tracing a recall").

``trace: true`` on memory search (and on context assembly, for a credential that manages the instance) returns, for
each candidate the recall collected and ranked, its rank in each retrieval lane, its place in the fused order,
whether the reranker read it and what it scored, and one exit reason from a closed set (none when it is in the
packet). The trace carries ids, ranks, scores and closed-set values only; it is limited to the caller's scope, drops
erased ids, changes nothing in the packet and costs nothing when it is off. ``colonymind recall --explain`` prints it
in plain words.
"""
from __future__ import annotations

import argparse
import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from onekey import KEY
from colonymind import cli as colonymind_cli
from colonymind import recall_cli
from colonymind.api.credentials import CredentialTable, secret_digest
from colonymind.api.errors import install_exception_handlers
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import host
from colonymind.api.schemas.host import RecallTraceReport
from colonymind.config import DEFAULTS, save_config
from colonymind.memory import trace as recall_trace
from colonymind.memory import turn_budget
from colonymind.memory.recall import pack_memory_context, provider_calibration_metadata, source_candidates
from colonymind.memory.search import CollectedSources, collect_sources, select_memory
from colonymind.memory.rerank_gate import gate_for
from colonymind.memory.selection import RecallSelector
from colonymind.memory.turn_budget import TurnBudget
from colonymind.turns import TurnIdempotencyLedger
from recall_route import ANSWER, OWNER, recall_app
from test_recall_checkpoint_copies import KEYS, LONG_INPUT, SESSION, _AnswerReranker, _calibrated, checkpoint
from test_recall_owner_first import NOW, PROBE, REPLY, SAID, _Claims
from test_turn_source_evidence import source_app  # noqa: F401 (fixture)

QUERY = "hydrofoil departure"
GUEST_SECRET = "guest-integration-secret-0123456789"
DISTRACTORS = {f"d{n}" for n in range(7)}
LATER = "Later: the spare bike keys stay in the garage drawer."


async def post(app, path, body, *, key=KEY, base_url="http://test"):
    headers = {"Authorization": "Bearer " + key} if key else {}
    async with AsyncClient(transport=ASGITransport(app=app), base_url=base_url, headers=headers) as client:
        return await client.post(path, json=body)


def search_body(**changes):
    return {"identity": {"host_id": "fixture"}, "person_id": OWNER, "session_id": "session-b", "query": QUERY,
            **changes}


def assemble_body(**changes):
    return {"identity": {"host_id": "fixture"}, "context": {"contact_id": OWNER, "session_id": "session-b"},
            "incoming_message": {"role": "user", "content": QUERY}, **changes}


async def traced_search(app, **changes):
    response = await post(app, "/v1/host/memory/search", search_body(trace=True, **changes))
    assert response.status_code == 200, response.text
    return response.json()


def memory_section(response):
    return next(section for section in response.json()["sections"] if section["id"] == "colonymind-memory")


# --- 1. The packet is the same with the trace on and off ---------------------------------------------------------

async def test_the_packet_is_the_same_with_the_trace_on_and_off(monkeypatch, tmp_path):
    setup = await recall_app(monkeypatch, tmp_path, initiatives=0)
    plain = await post(setup.app, "/v1/host/memory/search", search_body())
    traced = await post(setup.app, "/v1/host/memory/search", search_body(trace=True))
    assert plain.status_code == traced.status_code == 200, traced.text
    assert ANSWER in plain.json()["content"]                      # not vacuous: the answer is recalled
    assert "trace" not in plain.json()                            # off: the response is what it was
    assert {key: value for key, value in traced.json().items() if key != "trace"} == plain.json()
    assert traced.json()["trace"]["rows"]

    off = await post(setup.app, "/v1/host/context/assemble", assemble_body())
    on = await post(setup.app, "/v1/host/context/assemble", assemble_body(trace=True))
    assert off.status_code == on.status_code == 200, on.text
    assert "trace" not in off.json()
    assert ANSWER in memory_section(off)["body"] and memory_section(on) == memory_section(off)
    assert on.json()["trace"]["section"] == "in_context" and on.json()["trace"]["rows"]


# --- 2. The trace's selected ids are the packet's ------------------------------------------------------------------

async def test_the_traces_selected_ids_are_the_packets_and_each_candidate_has_its_stages(monkeypatch, tmp_path):
    setup = await recall_app(monkeypatch, tmp_path, initiatives=0)
    selector = RecallSelector(setup.reranker.rerank,
                              calibration_metadata=lambda: provider_calibration_metadata(setup.reranker))
    tracer = recall_trace.RecallTrace()
    collected = await collect_sources(setup.ledger, query=QUERY, contact_id=OWNER, session_id="session-b",
                                      vector_store=setup.store, embedding_pipeline=setup.embedder, trace=tracer)
    packet = await select_memory(collected, query=QUERY, selector=selector, timezone_name="UTC", trace=tracer)
    report = tracer.report()
    assert packet.selected and report["selected"] == [row["id"] for row in packet.selected]
    in_packet = {row["candidate"] for row in report["rows"] if row["exit"] is None and row["shown_by"] is None}
    assert in_packet == set(report["selected"])
    assert report["lanes"] == {"lexical": 8, "vector": 8, "vector_media": 0, "stemmed": 8, "claim_key": 0,
                               "lexical_media": 0}
    assert report["rerank"]["outcome"] == "scored" and report["rerank"]["submitted"] == 8
    assert report["calibration"] == {"status": "configuration_verified_weights_unverified", "cutoff": 0.8}
    assert report["packet"] == {"limit": 5, "max_chars": 4000, "count": 1}

    answer, = [row for row in report["rows"] if row["turn_id"] == "answer"]
    assert answer["exit"] is None and answer["form"] == "quotation" and answer["submitted"]
    assert answer["rerank_score"] == pytest.approx(0.99)
    assert set(answer["lanes"]) == {"lexical", "vector", "stemmed"} and answer["fused_rank"] >= 1
    others = [row for row in report["rows"] if row["turn_id"] != "answer"]
    assert {row["turn_id"] for row in others} == DISTRACTORS
    assert all(row["exit"] == "below_cutoff" and row["rerank_score"] == pytest.approx(0.1) for row in others)

    # One message's way through the recall, for a harness attributing a miss to a stage.
    assert tracer.exit_for("answer") is None and tracer.exit_for("d0") == "below_cutoff"
    assert tracer.exit_for("never-recorded") == "no_lane"

    # The route reports the same trace.
    body = await traced_search(setup.app)
    assert len(body["trace"]["selected"]) == body["count"] == 1
    assert {row["turn_id"]: row["exit"] for row in body["trace"]["rows"]} == {
        row["turn_id"]: row["exit"] for row in report["rows"]}


async def test_the_submission_bound_and_an_unverified_cutoff_are_reported_as_they_were(monkeypatch, tmp_path):
    setup = await recall_app(monkeypatch, tmp_path, initiatives=0)
    bounded = (await traced_search(setup.app, limit=1))["trace"]
    assert bounded["rerank"]["submission_bound"] == 4 and bounded["rerank"]["submitted"] == 4
    for row in bounded["rows"]:
        if row["fused_rank"] > 4:
            assert not row["submitted"] and row["rerank_score"] is None
            assert row["exit"] == "below_submission_bound"
        else:
            assert row["submitted"] and row["exit"] in (None, "below_cutoff")
            assert (row["exit"] is None) == (row["rerank_score"] >= 0.8)

    # A cutoff whose calibration does not match is not applied: no candidate is labelled below it.
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_CALIBRATION", "0" * 64)
    body = await traced_search(setup.app)
    unverified = body["trace"]
    assert unverified["calibration"] == {"status": "mismatch", "cutoff": None}
    assert body["count"] == 5 and ANSWER in body["content"]
    assert not any(row["exit"] == "below_cutoff" for row in unverified["rows"])
    assert sorted(str(row["exit"]) for row in unverified["rows"]) == ["None"] * 5 + ["over_packet_limit"] * 3

    # Reranking off: nothing was submitted or scored, and the packet's limit decided.
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    off = (await traced_search(setup.app, limit=2))["trace"]
    assert off["rerank"] == {"mode": "off", "outcome": "off", "missed": None, "late": False,
                             "submission_bound": None, "submitted": 0}
    assert not any(row["submitted"] or row["rerank_score"] is not None for row in off["rows"])
    assert sorted(str(row["exit"]) for row in off["rows"]) == ["None"] * 2 + ["over_packet_limit"] * 6


async def _incomplete(query, documents, top_k=10):
    return [{"index": 0, "score": 0.9}]                   # one score for two documents


async def _failing(query, documents, top_k=10):
    raise RuntimeError("the reranker failed")


async def _slow(query, documents, top_k=10):
    await asyncio.sleep(0.5)
    return [{"index": i, "score": 0.9} for i in range(len(documents))]


async def _traced_selection(directory, rerank_fn):
    """Two candidates and a packet of one: the reranker is asked to read both."""
    directory.mkdir()
    ledger = TurnIdempotencyLedger(directory / "turn-idempotency.db")
    for turn in ("a", "b"):
        ledger.record_source(turn, contact_id="p", session_id="s",
                             messages=[{"role": "user", "content": f"Harbour timetable note {turn}."}])
    candidates = source_candidates(ledger.search_sources("harbour timetable", contact_id="p", session_id="s"))
    tracer = recall_trace.RecallTrace()
    tracer.recall(CollectedSources(ledger, "p", "s", None), limit=1, max_chars=4000)
    selected, content = await RecallSelector(rerank_fn).select_context(
        "harbour timetable", [], candidates, limit=1, max_chars=4000, trace=tracer)
    tracer.packet(selected, content)
    await gate_for(rerank_fn).idle()
    report = tracer.report()
    assert len(candidates) == len(report["rows"]) == 2 and selected
    return report, recall_cli.explain({"count": len(selected), "trace": report}, person="p")


@pytest.mark.parametrize("rerank_fn, reason", [(_incomplete, "invalid"), (_failing, "error"), (_slow, "timeout")])
async def test_a_rerank_that_missed_after_it_was_sent_says_what_it_was_sent(tmp_path, monkeypatch, rerank_fn,
                                                                           reason):
    # The call started: the reranker was sent both candidates, whatever came of it.
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "on")
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_TIMEOUT_MS", "50")
    report, lines = await _traced_selection(tmp_path / reason, rerank_fn)
    assert report["rerank"] == {"mode": "on", "outcome": "missed", "missed": reason, "late": False,
                                "submission_bound": 4, "submitted": 2}
    assert all(row["submitted"] and row["rerank_score"] is None for row in report["rows"])
    assert f"missed ({reason}) after it was sent 2 candidates" in lines[2]
    assert all(f"sent to the reranker, which missed ({reason})" in line for line in lines[3:])


async def test_a_rerank_that_missed_before_it_was_sent_says_nothing_was(tmp_path, monkeypatch):
    # A busy reranker and a turn with no time left for it: no call started, nothing was sent.
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "on")
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_TIMEOUT_MS", "50")
    assert gate_for(_slow).start(lambda: asyncio.sleep(0.3)) is not None       # an earlier call holds the reranker
    busy = await _traced_selection(tmp_path / "busy", _slow)
    with turn_budget.turn(TurnBudget(wait_s=0.0)):
        budget = await _traced_selection(tmp_path / "budget", _slow)
    for reason, (report, lines) in (("busy", busy), ("budget", budget)):
        assert report["rerank"] == {"mode": "on", "outcome": "missed", "missed": reason, "late": False,
                                    "submission_bound": None, "submitted": 0}
        assert not any(row["submitted"] for row in report["rows"])
        assert f"missed ({reason}) before any candidate was sent" in lines[2]
        assert all(f"not reranked (the reranker missed: {reason})" in line for line in lines[3:])


def test_the_packer_says_whether_the_budget_or_the_limit_left_a_candidate_out(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    texts = {"a": "Alpha note.", "b": "Bravo " * 300, "c": "Charlie note.", "d": "Delta note."}
    for turn, text in texts.items():
        ledger.record_source(turn, contact_id="p", session_id="s", messages=[{"role": "user", "content": text}])
    hits = {hit["turn_id"]: hit for hit in ledger.search_sources(
        "alpha bravo charlie delta", contact_id="p", session_id="s", limit=10)}
    ordered = source_candidates([hits[turn] for turn in "abcd"])
    ordered[1]["atomic_evidence"] = True        # indivisible: shown whole or not at all
    tracer = recall_trace.RecallTrace()
    tracer.recall(CollectedSources(ledger, "p", "s", None), limit=2, max_chars=1200)
    tracer.fused(ordered)
    selected, content = pack_memory_context(ordered, limit=2, max_chars=1200, trace=tracer)
    tracer.packet(selected, content)
    exits = {row["turn_id"]: row["exit"] for row in tracer.report()["rows"]}
    assert exits == {"a": None, "b": "over_char_budget", "c": None, "d": "over_packet_limit"}


def test_a_lane_hit_without_a_message_hash_is_traced_by_its_source(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("t", contact_id="p", session_id="s", messages=[{"role": "user", "content": "Alpha note."}])
    hit = ledger.search_sources("alpha", contact_id="p", session_id="s", limit=10)[0]
    tracer = recall_trace.RecallTrace()
    tracer.lane("lexical", [hit])
    tracer.lane("vector", [{"turn_id": "t"}])            # a lane that knows the source, not the message
    tracer.recall(CollectedSources(ledger, "p", "s", None), limit=5, max_chars=4000)
    tracer.packet([], "")
    rows = tracer.report()["rows"]
    assert {(*row["lanes"].items(), row["message_hash"], row["exit"]) for row in rows} == {
        (("lexical", 1), hit["source_message_hash"], "no_lane"), (("vector", 1), None, "no_lane")}
    assert tracer.exit_for("t") == "no_lane" and tracer.exit_for("t", hit["source_message_hash"]) == "no_lane"


def test_two_images_of_one_message_keep_their_own_lane_ranks(tmp_path):
    # Two images attached to one message are two candidates, each found by the image lanes on its own description:
    # an image lane's rank belongs to the image it found, not to every image of that message.
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("t", contact_id="p", session_id="s",
                         messages=[{"role": "user", "content": "Three harbour photos."}])
    digest = ledger.search_sources("harbour", contact_id="p", session_id="s")[0]["source_message_hash"]

    def image(asset):
        # As the image keyword lane returns one (``SourceMedia.search``); the vector lane's also has ``turn_id``.
        return {"id": "media:" + asset * 64, "kind": "media_description", "asset_id": "sha256:" + asset * 64,
                "source_uri": "turn:t", "source_turn_id": "t", "source_message_hash": digest, "role": "user",
                "epistemic_state": "derived_unverified", "content": f"Photo {asset} of the harbour at dusk."}

    first, second, third = image("a"), image("b"), image("c")
    tracer = recall_trace.RecallTrace()
    tracer.lane("vector_media", [dict(second, turn_id="t"), dict(first, turn_id="t")], media=True)
    tracer.lane("lexical_media", [first, second, third], media=True)   # the third is then left out before ranking
    tracer.recall(CollectedSources(ledger, "p", "s", None), limit=5, max_chars=4000)
    tracer.fused([first, second])
    selected, content = pack_memory_context([first, second], limit=5, max_chars=4000, trace=tracer)
    tracer.packet(selected, content)
    rows = tracer.report()["rows"]
    ranks = {row["candidate"]: row["lanes"] for row in rows}
    assert ranks[first["id"]] == {"vector_media": 2, "lexical_media": 1}
    assert ranks[second["id"]] == {"vector_media": 1, "lexical_media": 2}
    # The image no candidate carried to ranking is a row of its own, with its own rank.
    left, = [row for row in rows if row["candidate"] is None]
    assert (left["turn_id"], left["message_hash"], left["lanes"], left["exit"]) == (
        "t", digest, {"lexical_media": 3}, "no_lane")
    assert tracer.exit_for("t", digest) is None                        # the message's images are in the packet


async def test_a_message_carried_by_a_pair_or_by_claims_is_traced_through_what_carries_it(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("owner-1:t-1", contact_id="owner", session_id=SESSION, derive_claims=False,
                         occurred_at="2026-09-25T12:00:04+00:00",
                         messages=[{"role": "user", "content": SAID}, {"role": "assistant", "content": REPLY}])
    for n in range(2):
        checkpoint(ledger, [{"role": "user", "content": SAID}, {"role": "assistant", "content": REPLY},
                            {"role": "user", "content": f"Later message {n}."}])
    tracer = recall_trace.RecallTrace()
    collected = await collect_sources(ledger, query=PROBE, contact_id="owner", session_id=SESSION, trace=tracer)
    packet = await select_memory(collected, query=PROBE, selector=RecallSelector(), timezone_name="UTC", now=NOW,
                                 trace=tracer)
    report = tracer.report()
    pair, = [row for row in report["rows"] if row["form"] == "conversation"]
    assert pair["exit"] is None and pair["candidate"] in report["selected"]
    assert [row["kind"] for row in packet.selected].count("conversation_pair") == 1
    # Every copy of the person's words and the reply is in the packet through the pair, whichever turn holds it.
    for hit in ledger.search_sources(PROBE, contact_id="owner", session_id=SESSION, limit=10):
        if hit["content"] in (SAID, REPLY):
            assert tracer.exit_for(hit["turn_id"], hit["source_message_hash"]) is None, hit["turn_id"]

    # With claims, the person's words come as the complete message recalled with its claims, and the reply its
    # input cannot stand beside is left out before ranking.
    (tmp_path / "claims").mkdir()
    claimed = TurnIdempotencyLedger(tmp_path / "claims" / "turn-idempotency.db")
    claimed.record_source("owner-1:t-1", contact_id="owner", session_id="s-1", occurred_at="2026-09-25T12:00:04+00:00",
                          messages=[{"role": "user", "content": SAID}, {"role": "assistant", "content": REPLY}])
    from colonymind.beliefs.source_projection import SourceClaimProjection
    assert await SourceClaimProjection(claimed).process_one(_Claims())
    tracer = recall_trace.RecallTrace()
    collected = await collect_sources(claimed, query=PROBE, contact_id="owner", session_id="s-2", trace=tracer)
    packet = await select_memory(collected, query=PROBE, selector=RecallSelector(), timezone_name="UTC", now=NOW,
                                 trace=tracer)
    assert REPLY not in packet.content
    hits = claimed.search_sources(PROBE, contact_id="owner", session_id="s-2", limit=10)
    said, = [hit for hit in hits if hit["role"] == "user"]
    reply, = [hit for hit in hits if hit["role"] == "assistant"]
    assert tracer.exit_for(said["turn_id"], said["source_message_hash"]) is None
    assert tracer.exit_for(reply["turn_id"], reply["source_message_hash"]) == "no_lane"
    rows = tracer.report()["rows"]
    assert any(row["message_hash"] == said["source_message_hash"] and row["exit"] is None for row in rows)
    assert any(row["message_hash"] == reply["source_message_hash"] and row["exit"] == "no_lane"
               and row["candidate"] is None for row in rows)


async def test_a_candidate_the_recheck_removes_is_labelled_so(source_app, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    source_app.add_middleware(ApiKeyMiddleware, api_key=KEY)
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("report", contact_id="person", session_id="work",
                         messages=[{"role": "assistant", "content": "The archive digest matched its receipt."}])

    class Annotating(RecallSelector):
        async def select_context(self, *args, **kwargs):
            selected, text = await super().select_context(*args, **kwargs)
            ref, = ledger.source_references(["report"], contact_id="person", session_id="later")
            ledger.append_source_annotation(contact_id="person", session_id="later", annotation_id="audit", **ref,
                                            excerpt="matched its receipt.", correction="Only the size matched.",
                                            author_principal="person")
            return selected, text

    monkeypatch.setattr(host, "_context_recall_selector", (host._reranker, Annotating()))
    body = await post(source_app, "/v1/host/memory/search", {
        "identity": {"host_id": "fixture"}, "person_id": "person", "session_id": "later",
        "query": "archive digest", "trace": True})
    assert body.status_code == 200, body.text
    assert body.json()["count"] == 0
    row, = [row for row in body.json()["trace"]["rows"] if row["turn_id"] == "report"]
    assert row["exit"] == "removed_by_recheck"


# --- 3. No content ----------------------------------------------------------------------------------------------

def _free_strings(node, defs, path="trace"):
    """Paths of the string values the schema leaves free: no enum, constant or pattern."""
    if "$ref" in node:
        yield from _free_strings(defs[node["$ref"].rsplit("/", 1)[-1]], defs, path)
        return
    for branch in node.get("anyOf", ()):
        yield from _free_strings(branch, defs, path)
    if node.get("type") == "string" and not {"enum", "const", "pattern"} & node.keys():
        yield path
    if "enum" in node or "const" in node:
        assert all(value is None or isinstance(value, str) and " " not in value
                   for value in node.get("enum", [node.get("const")])), path
    for name, child in node.get("properties", {}).items():
        yield from _free_strings(child, defs, f"{path}.{name}")
    if isinstance(node.get("items"), dict):
        yield from _free_strings(node["items"], defs, path + "[]")
    if isinstance(node.get("additionalProperties"), dict):
        yield from _free_strings(node["additionalProperties"], defs, path + "{}")
    if isinstance(node.get("propertyNames"), dict):
        yield from _free_strings(node["propertyNames"], defs, path + "{key}")


def test_the_trace_schema_has_no_room_for_text():
    schema = RecallTraceReport.model_json_schema()
    # A turn id is the host's own identifier; everything else is a digest, a lane name or a closed-set value.
    assert list(_free_strings(schema, schema.get("$defs", {}))) == ["trace.rows[].turn_id"]
    names = json.dumps(schema).lower()
    assert not any(field in names for field in ('"content"', '"text"', '"quote"', '"query"', '"ranking_text"'))


async def test_a_trace_carries_no_words_and_nothing_outside_the_callers_scope(source_app, tmp_path,  # noqa: F811
                                                                             monkeypatch):
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    source_app.add_middleware(ApiKeyMiddleware, api_key=KEY)
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("report", contact_id="person", session_id="work", messages=[
        {"role": "assistant", "content": "The hydrofoil departure is Friday at nine."}])
    ledger.record_source("private-other", contact_id="other", session_id="work", messages=[
        {"role": "user", "content": "The hydrofoil cabinet has the private amber label."}])
    ledger.record_source("scoped", contact_id="person", session_id="original", scope="session", messages=[
        {"role": "user", "content": "The hydrofoil session secret is cobalt."}])
    response = await post(source_app, "/v1/host/memory/search", {
        "identity": {"host_id": "fixture"}, "person_id": "person", "session_id": "later",
        "query": "hydrofoil departure", "trace": True})
    assert response.status_code == 200, response.text
    trace = response.json()["trace"]
    assert "Friday at nine" in response.json()["content"]
    assert {row["turn_id"] for row in trace["rows"]} == {"report"}
    text = json.dumps(trace).lower()
    for word in ("hydrofoil", "departure", "friday", "nine", "amber", "cobalt", "private-other", "scoped"):
        assert word not in text


# --- 4. Erased ids are absent -------------------------------------------------------------------------------------


# --- 5. Off, the collector is never used --------------------------------------------------------------------------

def _recording_methods():
    return sorted(name for name, value in vars(recall_trace.RecallTrace).items()
                  if callable(value) and not name.startswith("_") and name not in ("report", "exit_for"))


def test_the_no_op_collector_has_every_recording_method_and_no_state():
    assert not recall_trace.NULL_TRACE and recall_trace.RecallTrace()
    assert recall_trace.NullTrace.__slots__ == ()
    for name in _recording_methods():
        assert callable(getattr(recall_trace.NullTrace, name)), name


async def test_with_the_trace_off_the_collector_is_never_used(monkeypatch, tmp_path):
    setup = await recall_app(monkeypatch, tmp_path, initiatives=0)

    def refuse(*args, **kwargs):
        raise AssertionError("the trace collector was used with the trace off")

    for name in [*_recording_methods(), "report", "exit_for", "__init__"]:
        monkeypatch.setattr(recall_trace.RecallTrace, name, refuse)
    search = await post(setup.app, "/v1/host/memory/search", search_body())
    assemble = await post(setup.app, "/v1/host/context/assemble", assemble_body())
    assert search.status_code == assemble.status_code == 200
    assert ANSWER in search.json()["content"] and ANSWER in memory_section(assemble)["body"]
    tracer = recall_trace.NULL_TRACE
    collected = await collect_sources(setup.ledger, query=QUERY, contact_id=OWNER, session_id="session-b",
                                      vector_store=setup.store, embedding_pipeline=setup.embedder, trace=tracer)
    packet = await select_memory(collected, query=QUERY, selector=RecallSelector(), timezone_name="UTC",
                                 trace=tracer)
    assert packet.content


# --- 6. Who may ask for a trace ------------------------------------------------------------------------------------

def guest_app():
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(host.router)
    app.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=CredentialTable.from_config({
        "api-key": {"owner": ["whatsapp"], "contacts": "all"},
        # Reads and writes as anyone the sidecar resolves except the owner; manages nothing.
        "relay": {"sha256": secret_digest(GUEST_SECRET), "owner": [], "contacts": "all"},
    }))
    return app


async def test_a_trace_is_the_owners_to_ask_for_or_the_callers_own(monkeypatch, tmp_path):
    setup = await recall_app(monkeypatch, tmp_path, initiatives=0)
    setup.ledger.record_source("guest-note", contact_id="guest-1", session_id="g", messages=[
        {"role": "user", "content": "The hydrofoil guest pass is blue, valid Friday at nine."}])
    app = guest_app()
    guest = {"person_id": "guest-1", "session_id": "g"}
    plain = await post(app, "/v1/host/memory/search", search_body(**guest), key=GUEST_SECRET)
    assert plain.status_code == 200 and "blue" in plain.json()["content"]      # it may search that person
    refused = await post(app, "/v1/host/memory/search", search_body(trace=True, **guest), key=GUEST_SECRET)
    assert refused.status_code == 403 and refused.json()["detail"]["code"] == "trace_not_granted"
    assert "blue" not in refused.text and "guest-note" not in refused.text
    own = await post(app, "/v1/host/memory/search", search_body(trace=True, person_id="credential:relay"),
                     key=GUEST_SECRET)
    assert own.status_code == 200 and "trace" in own.json()
    owner = await post(app, "/v1/host/memory/search", search_body(trace=True, **guest))
    assert owner.status_code == 200 and [row["turn_id"] for row in owner.json()["trace"]["rows"]] == ["guest-note"]

    context = {"contact_id": "guest-1", "session_id": "g"}
    refused = await post(app, "/v1/host/context/assemble", assemble_body(context=context, trace=True),
                         key=GUEST_SECRET)
    assert refused.status_code == 403 and refused.json()["detail"]["code"] == "trace_not_granted"
    untraced = await post(app, "/v1/host/context/assemble", assemble_body(context=context), key=GUEST_SECRET)
    assert untraced.status_code == 200 and "trace" not in untraced.json()
    managed = await post(app, "/v1/host/context/assemble", assemble_body(context=context, trace=True))
    assert managed.status_code == 200 and managed.json()["trace"]["rows"]

    dev = FastAPI()
    install_exception_handlers(dev)
    dev.include_router(host.router)
    dev.add_middleware(ApiKeyMiddleware, api_key=None)
    keyless = await post(dev, "/v1/host/context/assemble", assemble_body(context=context, trace=True), key=None,
                         base_url="http://localhost")
    assert keyless.status_code == 403 and keyless.json()["detail"]["code"] == "trace_not_granted"


# --- 7. colonymind recall --explain -------------------------------------------------------------------------------

def cli_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default=None)
    recall_cli.add_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


def test_the_main_cli_registers_recall():
    source = open(colonymind_cli.__file__, encoding="utf-8").read()
    assert "add_recall_parser(sub)" in source and 'args.command == "recall"' in source
    args = cli_args(["recall", "what is due", "--as", "p-01"])
    assert args.question == "what is due" and args.person == "p-01" and args.explain is False
    with pytest.raises(SystemExit):
        cli_args(["recall", "what is due"])                       # whose memory is always named


async def test_recall_explain_prints_a_stage_line_per_candidate(monkeypatch, tmp_path, capsys):
    setup = await recall_app(monkeypatch, tmp_path / "state", initiatives=0)
    response = await traced_search(setup.app)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("COLONYMIND_HOME", str(home))
    monkeypatch.delenv("COLONYMIND_API_KEY", raising=False)
    save_config(dict(DEFAULTS), home)
    (home / "api.key").write_text("k\n")
    requests = []

    def handler(request):
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json=response)

    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real(transport=httpx.MockTransport(handler)))

    assert recall_cli.run(cli_args(["recall", QUERY, "--as", OWNER, "--session", "session-b", "--explain"])) == 0
    path, sent = requests[-1]
    assert path == "/v1/host/memory/search"
    assert sent["trace"] is True and sent["person_id"] == OWNER and sent["session_id"] == "session-b"
    assert sent["query"] == QUERY
    out = capsys.readouterr().out
    stages = [line.strip() for line in out.splitlines() if line.strip()[:1].isdigit()]
    assert len(stages) == len(response["trace"]["rows"]) == 8
    answer, = [line for line in stages if " answer " in line]
    assert "found by keyword (rank" in answer and ") and vector (rank" in answer and "; fused " in answer
    assert "reranked 0.99 (calibrated cutoff 0.80, passed)" in answer and answer.endswith("in the packet")
    assert all("below the calibrated cutoff" in line for line in stages if line != answer)

    assert recall_cli.run(cli_args(["recall", QUERY, "--as", OWNER])) == 0
    assert not requests[-1][1].get("trace")                         # the packet only: no trace asked for
    assert ANSWER in capsys.readouterr().out

from colonymind.memory import curated
async def test_a_source_erased_during_the_recall_leaves_no_id_in_the_trace(monkeypatch, tmp_path):
    setup = await recall_app(monkeypatch, tmp_path, initiatives=0)
    original = TurnIdempotencyLedger.search_sources

    def search_then_erase(self, *args, **kwargs):
        hits = original(self, *args, **kwargs)
        if any(hit["turn_id"] == "answer" for hit in hits):
            curated.forget_sources(self, OWNER, source_ids=['answer'])
        return hits

    monkeypatch.setattr(TurnIdempotencyLedger, "search_sources", search_then_erase)
    body = await traced_search(setup.app)
    trace = body["trace"]
    assert ANSWER not in body["content"]
    assert "answer" not in json.dumps(trace)
    erased = [row for row in trace["rows"] if row["exit"] == "erased"]
    assert erased, "the lexical lane found the erased source: the trace says one candidate went, and no more"
    for row in erased:
        assert row["turn_id"] is None and row["message_hash"] is None and row["candidate"] is None
        assert row["lanes"] == {} and row["shown_by"] is None
    assert {row["turn_id"] for row in trace["rows"] if row["turn_id"]} <= DISTRACTORS


@pytest.mark.parametrize("shown", [False, True])
async def test_a_message_erased_from_a_checkpoint_that_keeps_other_messages_leaves_no_id(tmp_path, monkeypatch,
                                                                                        shown):
    # Erasing a message takes its copies out of the checkpoints that hold it; a checkpoint that holds other messages
    # stays. A copy recalled from it (in the packet itself, or shown there by the pair of its original) is erased as
    # its original is, and the trace keeps no id of it: not its message hash, not its candidate id, not in
    # ``selected`` and not as a ``shown_by``. The checkpoint's other message keeps its ids.
    class Reranker(_AnswerReranker):
        async def rerank(self, query, documents, top_k):
            if not shown:            # only the copy of the reply clears the cutoff (the pair shows none of it)
                return await super().rerank(query, documents, top_k)
            # The pair clears it first and shows the copies whole.
            return [{"index": i, "score": .95 if text.startswith("User input:") else .9}
                    for i, text in enumerate(documents)]

    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("owner-1:t-1", contact_id="owner", session_id=SESSION, derive_claims=False,
                         occurred_at="2026-09-25T12:00:04+00:00",
                         messages=[{"role": "user", "content": LONG_INPUT}, {"role": "assistant", "content": KEYS}])
    held = checkpoint(ledger, [{"role": "user", "content": LONG_INPUT}, {"role": "assistant", "content": KEYS},
                               {"role": "user", "content": LATER}])
    query = "Where did we say the spare keys go?"
    later, = [hit for hit in ledger.search_sources(query, contact_id="owner", session_id=SESSION, limit=10)
              if hit["content"] == LATER]
    tracer = recall_trace.RecallTrace()
    collected = await collect_sources(ledger, query=query, contact_id="owner", session_id=SESSION, trace=tracer)
    packet = await select_memory(collected, query=query, selector=_calibrated(monkeypatch, Reranker("blue tin")),
                                 timezone_name="UTC", now=NOW, trace=tracer)
    before = tracer.report()
    # Not vacuous: a copy from the checkpoint is in the packet, by itself or through the pair.
    copies = [row for row in before["rows"]
              if row["turn_id"] == held and row["message_hash"] != later["source_message_hash"]]
    in_packet = [row for row in copies if row["exit"] is None]
    assert in_packet and all(bool(row["shown_by"]) is shown for row in in_packet)
    kept,= [row for row in before["rows"] if row["message_hash"] == later["source_message_hash"]]
    erased = {value for row in before["rows"] if row is not kept
              for value in (row["message_hash"], row["candidate"], row["shown_by"]) if value}
    erased.update(key for key in before["selected"] if key != kept["candidate"])
    assert len(erased) >= 3

    curated.forget_sources(ledger, 'owner', source_ids=['owner-1:t-1'])
    assert ledger.source_references([held], contact_id="owner", session_id=SESSION)    # it keeps the later message
    report = tracer.report()
    text = json.dumps(report)
    assert [value for value in erased if value in text] == []
    after = {row["message_hash"]: row for row in report["rows"]}
    assert after[kept["message_hash"]] == kept                                         # untouched
    assert report["selected"] == [key for key in before["selected"] if key == kept["candidate"]]
    for row in report["rows"]:
        if row is not after[kept["message_hash"]]:
            assert row["exit"] == "erased" and row["turn_id"] is None and row["candidate"] is None
            assert row["shown_by"] is None and row["lanes"] == {}
    # One message's way: an erased copy says so, whichever source is asked about; the kept message does not.
    for row in copies:
        assert tracer.exit_for(held, row["message_hash"]) == "erased"
        assert tracer.exit_for("owner-1:t-1", row["message_hash"]) == "erased"
    assert tracer.exit_for(held, kept["message_hash"]) == kept["exit"]
    assert packet.selected                                                             # the packet was not traced away
