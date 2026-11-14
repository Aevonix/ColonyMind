"""A subject named in an earlier message of the same conversation (docs/SOURCE-CLAIMS.md, extraction version 19).

"My sister moved last month." followed by "She lives in Denver now." formed no usable claim: a claim's subject had to
occur in its own quotation. The extraction call is now offered the conversation's previous user messages, and a
claim may take its subject from an exact span of one of them (``subject_basis``). The span grounds who or what the
subject is and nothing else: the value, the time and the operation still come from the current quotation, the review
checks the reference, the commit checks the span again, and recall ranks the claim by its own quotation.

Everything runs the real ledger, projection, commit and recall with a scripted extractor and review.
"""
from contextlib import closing
from datetime import datetime
import json
from types import SimpleNamespace

import pytest

from colonymind.beliefs import source_claims
from colonymind.beliefs.source_claims import (EXTRACTION_VERSION, REVIEW_SYSTEM, SYSTEM, extraction_request,
                                             validated_claims)
from colonymind.beliefs.source_projection import SourceClaimProjection
from colonymind.beliefs.source_time import interpret_time_query
from colonymind.memory import recall
from colonymind.turns import TurnIdempotencyLedger
from colonymind.turns.idempotency import canonical_turn_digest, source_message_hash

OWNER, CHAT = "owner", "chat"
MOVED = "My sister moved last month."
LIVES = "She lives in Denver now."
BOTH = "My sister and my brother visited on Sunday."
PILOT = "He flies planes for a living."
CORRECT = "Correction: she lives in Dallas, not Denver."
AUSTIN = "My sister moved to Austin in June."


def at(minute, hour=10):
    return f"2026-05-04T{hour:02d}:{minute:02d}:00+00:00"


def fact(text, subject, predicate, value, **fields):
    return {"subject": subject, "predicate": predicate, "value": value, "evidence": text, "operation": "assert",
            "prior_claim_id": None, "memory_kind": "personal_context",
            "recall_reason": "Use it when this person's family comes up.",
            "valid_from_text": None, "valid_to_text": None, "event_at_text": None, **fields}


class Scripted:
    """Extraction returns the scripted proposals of its message. A proposal's ``basis`` is ``(earlier text, words)``:
    its ``subject_basis`` names the offered earlier message with that text and those words (an id the call was not
    offered when none has that text). ``link`` names a predicate: the prior is the offered record of it. Review keeps
    every proposal ``refuse`` does not refuse. Every call is kept with its task, system prompt and payload."""

    def __init__(self, outputs=None, *, refuse=None):
        self.outputs, self.refuse, self.calls = outputs or {}, refuse, []

    def tier_config(self, tier):
        return SimpleNamespace(base_url="http://127.0.0.1:8080/v1")

    def call(self, task, text=None):
        return next(row for row in self.calls if row.task == task and (text is None or row.payload["message"] == text))

    async def complete(self, messages, **kwargs):
        payload = json.loads(messages[-1]["content"])
        task = kwargs.get("context", {}).get("task")
        self.calls.append(SimpleNamespace(task=task, system=messages[0]["content"], payload=payload, kwargs=kwargs))
        if task == "source_claim_review":
            return SimpleNamespace(model_id="fixture-model", content=json.dumps({str(row["index"]): {
                "keep": not (self.refuse and self.refuse(row["claim"])),
                "reason": "Scripted review of the reference and the value."} for row in payload["proposals"]}))
        proposals = []
        for proposal in self.outputs.get(payload["message"], []):
            proposal = dict(proposal)
            basis = proposal.pop("basis", None)
            if basis is not None:
                text, words = basis
                offered = next((row["id"] for row in payload.get("earlier_messages", []) if row["text"] == text),
                               "earlier_9")
                proposal["subject_basis"] = {"message_id": offered, "quote": words}
            link = proposal.pop("link", None)
            if link:
                proposal["prior_claim_id"] = next(row["id"] for row in payload["prior_assertions"]
                                                  if row["predicate"] == link)
            proposals.append(proposal)
        return SimpleNamespace(model_id="fixture-model", content=json.dumps({"claims": proposals}))


def say(ledger, turn, text, *, minute=0, hour=10, session=CHAT, contact=OWNER, reply=None):
    messages = [{"role": "user", "content": text}] + ([{"role": "assistant", "content": reply}] if reply else [])
    ledger.record_source(turn, contact_id=contact, session_id=session, occurred_at=at(minute, hour),
                         messages=messages)


async def extract(ledger, model):
    projection = SourceClaimProjection(ledger)
    while await projection.process_one(model):
        pass
    return projection


def stored(ledger):
    with closing(ledger._connect()) as conn:
        return {row["turn_id"]: json.loads(row["data_json"]) | dict(row)
                for row in conn.execute("SELECT * FROM source_claims")}


def readable(ledger, contact=OWNER):
    projection = SourceClaimProjection(ledger)
    with closing(ledger._connect()) as conn:
        return projection._rows(conn, contact, "")


def diagnostics(ledger, turn):
    with closing(ledger._connect()) as conn:
        return json.loads(conn.execute("SELECT diagnostics_json FROM source_claim_jobs WHERE turn_id=?",
                                       (turn,)).fetchone()[0])


def bundles(ledger, query, contact=OWNER):
    projection = SourceClaimProjection(ledger)
    hits = ledger.search_sources(query, contact_id=contact, session_id="later", limit=10)
    _, rows = projection.prepare_context([], hits, contact_id=contact, session_id="later", time_query=
                                         interpret_time_query(query, now=datetime.fromisoformat(at(30)),
                                                              timezone_name="UTC"))
    return [row for row in rows if row.get("atomic_evidence")]


def hash_of(text, session=CHAT):
    return source_message_hash(session, {"role": "user", "content": text})


async def sister(tmp_path, **outputs):
    """The conversation "My sister moved last month." / "She lives in Denver now.", extracted."""
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    model = Scripted({LIVES: [fact(LIVES, "My sister", "location", "Denver", basis=(MOVED, "My sister"))],
                      **outputs})
    say(ledger, "moved", MOVED, minute=0)
    say(ledger, "lives", LIVES, minute=2)
    await extract(ledger, model)
    return ledger, model


# ------------------------------------------------------------------ the claim and its basis

@pytest.mark.asyncio
async def test_the_sister_example_forms_a_claim_whose_basis_is_the_exact_span(tmp_path):
    ledger, model = await sister(tmp_path)
    call = model.call("source_claim_extraction", LIVES)
    assert call.payload["earlier_messages"] == [{"id": "earlier_1", "reported_at": at(0), "text": MOVED}]
    assert call.system == SYSTEM + source_claims.SUBJECT_BASIS_RULES
    first = model.call("source_claim_extraction", MOVED)
    assert "earlier_messages" not in first.payload and first.system == SYSTEM   # the conversation's first message

    claim = stored(ledger)["lives"]
    assert (claim["subject"], claim["subject_key"], claim["predicate"], claim["value"], claim["evidence"]) == (
        "My sister", "my sister", "location", "Denver", LIVES)
    assert claim["subject_basis_message"] == {"turn_id": "moved", "message_hash": hash_of(MOVED), "span_start": 0,
                                              "span_end": 9, "evidence": "My sister", "occurrence": 0}
    assert claim["extraction_version"] == EXTRACTION_VERSION == "source-claims-v19"
    assert claim["admission_review"]["basis"] == "model_judgment_unverified"

    row, = readable(ledger)
    basis = row["subject_basis"]
    assert (basis["evidence"], basis["turn_id"], basis["message_hash"]) == ("My sister", "moved", hash_of(MOVED))
    assert basis["disposition"] == "subject_identity_only"
    assert basis["value_use"] == "not_evidence_for_current_value"
    assert basis["basis"] == "earlier_message" and basis["span_start"] == 0 and basis["span_end"] == 9
    with closing(ledger._connect()) as conn:
        messages = json.loads(conn.execute("SELECT messages_json FROM turn_sources WHERE turn_id='moved'")
                              .fetchone()[0])
    assert basis["source_version"] == canonical_turn_digest(messages)

    bundle, = bundles(ledger, "Denver")
    assert set(bundle["source_turn_ids"]) == {"lives", "moved"}
    assert bundle["_source_message_hashes"]["moved"] == [hash_of(MOVED)]
    card = json.loads(bundle["content"])
    assert card["subject"] == "My sister" and card["assertions"][0]["subject_basis"]["evidence"] == "My sister"


@pytest.mark.asyncio
async def test_the_basis_is_the_subject_s_own_words_within_the_quote(tmp_path):
    ledger, _ = await sister(tmp_path, **{
        LIVES: [fact(LIVES, "my sister", "location", "Denver", basis=(MOVED, "My sister moved"))]})
    claim = stored(ledger)["lives"]
    assert claim["subject_basis_message"]["evidence"] == "My sister"
    assert (claim["subject_basis_message"]["span_start"], claim["subject_basis_message"]["span_end"]) == (0, 9)


@pytest.mark.asyncio
async def test_a_subject_in_the_current_quotation_needs_no_basis(tmp_path):
    text = "My sister lives in Denver now."
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    say(ledger, "moved", MOVED, minute=0)
    say(ledger, "lives", text, minute=2)
    model = Scripted({text: [fact(text, "My sister", "location", "Denver", basis=(MOVED, "My sister"))]})
    await extract(ledger, model)
    claim = stored(ledger)["lives"]
    assert "subject_basis_message" not in claim and claim["subject"] == "My sister"
    review = model.call("source_claim_review")
    assert review.system == REVIEW_SYSTEM and "earlier_messages" not in review.payload


# ------------------------------------------------------------------ refused bases

@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["another_conversation", "another_person", "said_later"])
async def test_a_message_of_another_conversation_person_or_later_time_is_never_a_basis(where, tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    if where == "another_conversation":
        say(ledger, "moved", MOVED, minute=0, session="another")
    elif where == "another_person":
        say(ledger, "moved", MOVED, minute=0, contact="someone")
    else:
        say(ledger, "moved", MOVED, minute=5)
    say(ledger, "lives", LIVES, minute=2)
    model = Scripted({LIVES: [fact(LIVES, "My sister", "location", "Denver", basis=(MOVED, "My sister"))]})
    await extract(ledger, model)
    call = model.call("source_claim_extraction", LIVES)
    assert "earlier_messages" not in call.payload and call.system == SYSTEM
    assert "lives" not in stored(ledger)
    assert diagnostics(ledger, "lives")["rejection_counts"] == {"subject_basis_unknown": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("subject, words, reason", [
    ("My sister", "my sister", "subject_basis_not_exact"),      # the message says "My sister"
    ("My sister", "My sistr", "subject_basis_not_exact"),
    ("My sister", "My sister left", "subject_basis_not_exact"),  # not a span of the message
    ("My brother", "My sister", "subject_basis_mismatch"),     # the words do not name the subject
])
async def test_a_basis_span_that_is_not_exact_is_refused(subject, words, reason, tmp_path):
    ledger, _ = await sister(tmp_path, **{
        LIVES: [fact(LIVES, subject, "location", "Denver", basis=(MOVED, words))]})
    assert "lives" not in stored(ledger)
    assert diagnostics(ledger, "lives")["rejection_counts"] == {reason: 1}


@pytest.mark.asyncio
async def test_the_speaker_is_never_named_by_an_earlier_message(tmp_path):
    ledger, _ = await sister(tmp_path, **{
        LIVES: [fact(LIVES, "I", "location", "Denver", basis=(MOVED, "My"))]})
    assert "lives" not in stored(ledger)
    assert diagnostics(ledger, "lives")["rejection_counts"] == {"subject_not_grounded": 1}


@pytest.mark.asyncio
async def test_an_earlier_message_supplies_no_value(tmp_path):
    text = "She loves it there."
    ledger, _ = await sister(tmp_path)
    say(ledger, "loves", text, minute=4)
    await extract(ledger, Scripted({text: [fact(text, "My sister", "mood", "last month",
                                                basis=(MOVED, "My sister"))]}))
    assert "loves" not in stored(ledger)
    assert diagnostics(ledger, "loves")["rejection_counts"] == {"value_not_grounded": 1}


def test_the_commit_refuses_a_basis_in_another_conversation(tmp_path):
    """A proposal whose basis names a message of another conversation, however it was made, never commits."""
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    say(ledger, "moved", MOVED, minute=0, session="another")
    say(ledger, "lives", LIVES, minute=2)
    forged = [{"id": "earlier_1", "turn_id": "moved", "message_hash": hash_of(MOVED, "another"),
               "reported_at": at(0), "text": MOVED, "start": 0, "content": MOVED}]
    proposal = fact(LIVES, "My sister", "location", "Denver")
    proposal["subject_basis"] = {"message_id": "earlier_1", "quote": "My sister"}
    claims = validated_claims(json.dumps({"claims": [proposal]}), message=LIVES, prior=[], observed_at=at(2),
                              earlier=forged)
    assert claims[0]["subject_basis_message"]["turn_id"] == "moved"
    claims = [dict(claims[0], admission_review={"version": "source-claim-review-v1",
                                                "basis": "model_judgment_unverified", "reason": "forged"})]
    projection = SourceClaimProjection(ledger)
    source = {"turn_id": "lives", "session_id": CHAT}
    assert projection.commit(source, {"role": "user", "content": LIVES}, claims, model="fixture-model") == 0
    assert "lives" not in stored(ledger)
    with closing(ledger._connect()) as conn:
        reasons = [row[0] for row in conn.execute("SELECT reason FROM learning_log WHERE action='rejected'")]
    assert reasons == ["subject basis no longer current"]


# ------------------------------------------------------------------ the review

@pytest.mark.asyncio
@pytest.mark.parametrize("subject, kept", [("My sister", False), ("my brother", True)])
async def test_the_review_sees_the_earlier_messages_and_rejects_a_mismatched_antecedent(subject, kept, tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    say(ledger, "both", BOTH, minute=0)
    say(ledger, "pilot", PILOT, minute=1)
    words = "My sister" if subject == "My sister" else "my brother"
    model = Scripted({PILOT: [fact(PILOT, subject, "occupation", "flies planes", basis=(BOTH, words))]},
                     refuse=lambda claim: claim.get("subject_basis_message", {}).get("evidence") == "My sister")
    await extract(ledger, model)
    review = model.call("source_claim_review")
    assert review.system == REVIEW_SYSTEM + source_claims.REVIEW_SUBJECT_BASIS
    assert review.payload["earlier_messages"] == [{"id": "earlier_1", "reported_at": at(0), "text": BOTH}]
    shown = review.payload["proposals"][0]["claim"]["subject_basis_message"]
    assert shown["message_id"] == "earlier_1" and shown["evidence"] == words
    assert "known_keys" not in review.payload
    assert ("pilot" in stored(ledger)) is kept
    counts = diagnostics(ledger, "pilot")
    assert (counts["review_kept_count"], counts["review_rejected_count"]) == ((1, 0) if kept else (0, 1))
    if kept:
        assert "message_id" not in stored(ledger)["pilot"]["subject_basis_message"]


MOTHER = "Where does your mother live?"


@pytest.mark.asyncio
async def test_the_calls_see_what_the_assistant_said_between_and_never_take_a_subject_from_it(tmp_path):
    """"My sister moved last month." / the assistant: "Where does your mother live?" / "She lives in Denver now.": the
    pronoun is the mother, whom only the assistant named. The extraction and the review are shown the assistant's
    messages that followed each offered earlier message (``assistant_after``), in order, so they can tell the
    answer is about someone else; those words are context only: no id names them, and a basis quoting them is
    refused."""
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    say(ledger, "moved", MOVED, minute=0, reply=MOTHER)
    say(ledger, "lives", LIVES, minute=2)
    projection = SourceClaimProjection(ledger)
    with closing(ledger._connect()) as conn:
        source = dict(conn.execute("SELECT * FROM turn_sources WHERE turn_id='lives'").fetchone())
    current = {"role": "user", "content": LIVES}
    offered, = projection.earlier_messages(source, current)
    assert (offered["id"], offered["text"], offered["assistant_after"]) == ("earlier_1", MOVED, [MOTHER])
    request = extraction_request(source, current, [], earlier=[offered])
    shown, = json.loads(request["messages"][1]["content"])["earlier_messages"]
    assert shown == {"id": "earlier_1", "reported_at": at(0), "text": MOVED, "assistant_after": [MOTHER]}
    assert "assistant_after" in request["messages"][0]["content"]
    assert MOTHER not in json.dumps(request["response_schema"])        # nothing the assistant said can be named

    # The assistant's words are never a basis: a quote from them is no span of any offered message.
    for words in ("your mother", MOTHER):
        proposal = fact(LIVES, "your mother", "location", "Denver")
        proposal["subject_basis"] = {"message_id": "earlier_1", "quote": words}
        rejected = []
        assert validated_claims(json.dumps([proposal]), message=LIVES, prior=[], observed_at=at(2),
                                earlier=[offered], rejections=rejected) == []
        assert [row["reason"] for row in rejected] == ["subject_basis_not_exact"], rejected

    # A proposal naming the sister reaches the review with what the assistant asked between.
    model = Scripted({LIVES: [fact(LIVES, "My sister", "location", "Denver", basis=(MOVED, "My sister"))]})
    await extract(ledger, model)
    review = model.call("source_claim_review")
    assert review.system == REVIEW_SYSTEM + source_claims.REVIEW_SUBJECT_BASIS
    assert "assistant_after" in source_claims.REVIEW_SUBJECT_BASIS
    assert review.payload["earlier_messages"] == [
        {"id": "earlier_1", "reported_at": at(0), "text": MOVED, "assistant_after": [MOTHER]}]


@pytest.mark.asyncio
async def test_what_the_assistant_said_is_shown_in_order_bounded_and_only_after_an_offered_message(tmp_path):
    """The assistant's messages are attached to the user message they followed, oldest first, the last few of them
    by their end; after a message that is not offered (an annotated one) they are not shown, since they answer what
    is hidden; a tool call with no words is no message."""
    from colonymind.beliefs.source_projection import EARLIER_CHARS, EARLIER_MESSAGES
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    long_reply = "y" * 1500 + " Who is she?"
    ledger.record_source("first", contact_id=OWNER, session_id=CHAT, occurred_at=at(0), messages=[
        {"role": "user", "content": MOVED}, {"role": "assistant", "content": None, "tool_calls": [{"id": "c"}]},
        {"role": "tool", "content": "{}"}, *({"role": "assistant", "content": f"Reply {n}."} for n in range(6)),
        {"role": "assistant", "content": long_reply}])
    say(ledger, "hidden", "My aunt called.", minute=1, reply="How is your aunt?")
    say(ledger, "now", LIVES, minute=2)
    from colonymind.turns.source_annotations import append
    append(ledger, contact_id=OWNER, session_id="notes", annotation_id="note", source_id="hidden",
           source_version=canonical_turn_digest(messages(ledger, "hidden")), excerpt="My aunt called.",
           correction="That was my cousin.", author_principal="owner-principal")
    projection = SourceClaimProjection(ledger)
    with closing(ledger._connect()) as conn:
        source = dict(conn.execute("SELECT * FROM turn_sources WHERE turn_id='now'").fetchone())
    offered, = projection.earlier_messages(source, {"role": "user", "content": LIVES})
    after = offered["assistant_after"]
    assert len(after) == EARLIER_MESSAGES and after[:-1] == ["Reply 3.", "Reply 4.", "Reply 5."][-(EARLIER_MESSAGES - 1):]
    assert after[-1] == long_reply[-EARLIER_CHARS:]
    assert "How is your aunt?" not in json.dumps(offered)


# ------------------------------------------------------------------ recall: the ranking text and the format

@pytest.mark.asyncio
async def test_the_ranking_text_and_the_candidate_format_are_unchanged(tmp_path):
    assert recall.CANDIDATE_FORMAT == "grounded-quotation-bundles-v2-corrections-first-rerank-cut-1500-2000"
    ledger, _ = await sister(tmp_path)
    bundle, = bundles(ledger, "Denver")
    assert bundle["ranking_text"] == LIVES
    assert recall._ranking_window(bundle) == LIVES and "sister" not in recall._ranking_window(bundle)


# ------------------------------------------------------------------ the request

def test_without_earlier_messages_the_request_is_unchanged_and_with_them_it_asks_for_a_basis():
    source = {"turn_id": "t", "contact_id": OWNER, "session_id": CHAT, "occurred_at": at(2)}
    message = {"role": "user", "content": LIVES}
    plain = extraction_request(source, message, [])
    assert plain["messages"][0]["content"] == SYSTEM
    assert "subject_basis" not in json.dumps(plain["response_schema"])
    assert "earlier_messages" not in json.loads(plain["messages"][1]["content"])
    earlier = [{"id": "earlier_1", "turn_id": "moved", "message_hash": hash_of(MOVED), "reported_at": at(0),
                "text": MOVED, "start": 0, "content": MOVED}]
    asked = extraction_request(source, message, [], earlier=earlier)
    assert asked["messages"][0]["content"] == SYSTEM + source_claims.SUBJECT_BASIS_RULES
    assert json.loads(asked["messages"][1]["content"])["earlier_messages"] == [
        {"id": "earlier_1", "reported_at": at(0), "text": MOVED}]
    assert "earlier_messages" not in asked["review_payload"]
    for branch in asked["response_schema"]["schema"]["properties"]["claims"]["items"]["anyOf"]:
        kind = branch["properties"].get("representation", {}).get("const")
        if kind in {"assertion", "procedure", "preference"}:
            assert "subject_basis" in branch["required"]
            basis = next(option for option in branch["properties"]["subject_basis"]["anyOf"]
                         if option.get("type") == "object")
            assert basis["properties"]["message_id"]["enum"] == ["earlier_1"]
        else:
            assert "subject_basis" not in branch["properties"]
    assert "subject_basis" in source_claims.SUBJECT_BASIS_RULES and "earlier_messages" not in SYSTEM


@pytest.mark.asyncio
async def test_the_offered_earlier_messages_are_the_conversation_s_last_few_user_messages(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    say(ledger, "hours-ago", "My aunt called this morning.", minute=0, hour=2)
    for minute in range(6):
        say(ledger, f"m{minute}", f"Message number {minute}.", minute=minute, reply="Noted.")
    say(ledger, "other-person", "My cousin is here.", minute=6, contact="someone")
    say(ledger, "other-chat", "My uncle is here.", minute=6, session="another")
    long = "x" * 1500 + " My sister moved."
    say(ledger, "long", long, minute=7)
    current = {"role": "user", "content": LIVES}
    ledger.record_source("now", contact_id=OWNER, session_id=CHAT, occurred_at=at(8),
                         messages=[{"role": "user", "content": "Guess what."}, {"role": "assistant", "content": "?"},
                                   current])
    projection = SourceClaimProjection(ledger)
    with closing(ledger._connect()) as conn:
        source = dict(conn.execute("SELECT * FROM turn_sources WHERE turn_id='now'").fetchone())
    offered = projection.earlier_messages(source, current)
    assert [row["id"] for row in offered] == ["earlier_1", "earlier_2", "earlier_3", "earlier_4"]
    assert [row["text"] for row in offered[:2]] == ["Message number 4.", "Message number 5."]
    assert offered[2]["turn_id"] == "long" and offered[2]["text"] == long[-1000:] and offered[2]["start"] == 517
    assert offered[3]["turn_id"] == "now" and offered[3]["text"] == "Guess what."
    # A conversation's messages from hours before are not offered (the window).
    say(ledger, "quiet-early", "My aunt called this morning.", minute=0, hour=2, session="quiet")
    quiet = {"turn_id": "quiet-now", "contact_id": OWNER, "session_id": "quiet", "scope": "person",
             "occurred_at": at(0), "ingested_at": None}
    assert projection.earlier_messages(quiet, current) == []
    request = extraction_request(source, current, [], earlier=offered)
    shown = json.loads(request["messages"][1]["content"])["earlier_messages"]
    assert shown[2] == {"id": "earlier_3", "reported_at": at(7), "text": long[-1000:], "excerpt": "end"}

    proposal = fact(LIVES, "My sister", "location", "Denver")
    proposal["subject_basis"] = {"message_id": "earlier_3", "quote": "My sister"}
    claim, = validated_claims(json.dumps([proposal]), message=LIVES, prior=[], observed_at=at(8), earlier=offered)
    basis = claim["subject_basis_message"]
    assert long[basis["span_start"]:basis["span_end"]] == "My sister" and basis["span_start"] == 1501


# ------------------------------------------------------------------ erasure, attribution, annotation

def messages(ledger, turn):
    with closing(ledger._connect()) as conn:
        return json.loads(conn.execute("SELECT messages_json FROM turn_sources WHERE turn_id=?", (turn,)).fetchone()[0])


def remove(ledger, action):
    if action == 'erase':
        from colonymind.memory import curated
        curated.forget_sources(ledger, OWNER, source_ids=['moved'])
        return
    if action not in {'reattribute', 'annotate'}:
        raise ValueError('unknown_source_fixture_action')
    if action == 'reattribute':
        from colonymind.turns.source_attribution import correct
        correct(ledger, operation_id='attribution', performed_by='operator', old_contact_id=OWNER, contact_id='someone', source_ids=['moved'], evidence_refs=['operator:correction'])
    else:
        from colonymind.turns.source_annotations import append
        with closing(ledger._connect()) as conn:
            messages = json.loads(conn.execute("SELECT messages_json FROM turn_sources WHERE turn_id='moved'").fetchone()[0])
        append(ledger, contact_id=OWNER, session_id='notes', annotation_id='note', source_id='moved', source_version=canonical_turn_digest(messages), excerpt=MOVED, correction='That was my cousin, not my sister.', author_principal='owner-principal')


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["reattribute", "annotate"])
async def test_losing_the_earlier_message_takes_the_claim_named_from_it(action, tmp_path):
    ledger, _ = await sister(tmp_path)
    assert readable(ledger)
    remove(ledger, action)
    assert not readable(ledger) and not bundles(ledger, "Denver")
    assert ("lives" in stored(ledger)) is (action == "annotate")   # an annotation hides, it does not erase
    assert any(hit["turn_id"] == "lives" for hit in ledger.search_sources("Denver", contact_id=OWNER,
                                                                           session_id="later"))


@pytest.mark.asyncio
@pytest.mark.parametrize("rewrite", ["release", "seal"])
async def test_a_secret_released_or_sealed_before_the_words_keeps_the_basis(rewrite, tmp_path):
    """Releasing a sealed value, or sealing one stored before capture sealed, rewrites the earlier message in place
    (its hash re-pointed everywhere) and moves the words after it. The claim it named, the one a correction retired
    and the correction still read, each basis locating the same words (here the second "My sister") in the message
    as it now is."""
    from colonymind.secrets.config import SecretsSettings, set_settings
    from colonymind.secrets.migrate import migrate, release
    from colonymind.secrets.vault import Vault, reset_vaults
    key = "hf_" + "AbCdEfGhIjKlMnOpQrStUvWxYz012345"  # secret-scan: synthetic
    reset_vaults()
    set_settings(SecretsSettings(capture_enabled=rewrite == "release"))   # "seal": stored as it arrived
    try:
        ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
        say(ledger, "moved", f"My sister said my token is {key}. My sister moved last month.", minute=0)
        before = messages(ledger, "moved")[0]["content"]
        say(ledger, "lives", LIVES, minute=2)
        await extract(ledger, Scripted({LIVES: [fact(LIVES, "My sister", "location", "Denver",
                                                     basis=(before, "My sister moved"))]}))
        say(ledger, "correct", CORRECT, minute=0, hour=20, session="later-chat")
        await extract(ledger, Scripted({CORRECT: [fact(CORRECT, "My sister", "location", "Dallas",
                                                       operation="correct", link="location")]}))
        assert sorted(row["value"] for row in readable(ledger)) == ["Dallas", "Denver"]
        set_settings(None)
        if rewrite == "release":
            ref, = [entry["ref"] for entry in Vault(tmp_path).entries()]
            release(ref, tmp_path)
        else:
            migrate(tmp_path, owner_id=OWNER)
        after, = messages(ledger, "moved")
        assert after["content"] != before and after["content"].rindex("My sister") != before.rindex("My sister")
        rows = readable(ledger)
        assert sorted((row["value"], bool(row["retracted_by"])) for row in rows) == [("Dallas", False),
                                                                                   ("Denver", True)]
        for row in rows:
            basis = row["subject_basis"]
            assert basis["message_hash"] == source_message_hash(CHAT, after)
            assert (basis["span_start"], basis["span_end"]) == (after["content"].rindex("My sister"),
                                                                after["content"].rindex("My sister") + 9)
        bundle, = bundles(ledger, "Dallas")
        assert json.loads(bundle["content"])["assertions"][0]["subject_basis"]["evidence"] == "My sister"
    finally:
        set_settings(None)
        reset_vaults()




@pytest.mark.asyncio
async def test_a_claim_named_from_an_earlier_message_is_offered_by_that_message_s_words(tmp_path):
    ledger, _ = await sister(tmp_path)
    say(ledger, "austin", AUSTIN, minute=0, hour=20, session="later-chat")
    model = Scripted({AUSTIN: []})
    await extract(ledger, model)
    offered = model.call("source_claim_extraction", AUSTIN).payload["prior_assertions"]
    assert [row["id"] for row in offered] == [stored(ledger)["lives"]["id"]]


@pytest.mark.asyncio
async def test_a_claim_named_from_an_earlier_message_never_folds_into_another(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_LEARNING", "on")
    ledger, _ = await sister(tmp_path)
    text = "My sister lives in Denver now."
    say(ledger, "literal", text, minute=5)
    await extract(ledger, Scripted({text: [fact(text, "My sister", "location", "Denver")]}))
    found = stored(ledger)
    assert found["lives"]["id"] != found["literal"]["id"] and found["lives"]["witnesses"] == 1


@pytest.mark.asyncio
async def test_a_correction_of_the_same_value_retires_the_claim_named_earlier_too(tmp_path, monkeypatch):
    """With the learning rules, a correction retires every live row of the corrected value, not only the one the
    extractor was shown: the claim a pronoun formed holds that value as much as the literal one does."""
    monkeypatch.setenv("COLONYMIND_LEARNING", "on")
    ledger, _ = await sister(tmp_path)
    text = "My sister lives in Denver now."
    say(ledger, "literal", text, minute=5)
    await extract(ledger, Scripted({text: [fact(text, "My sister", "location", "Denver")]}))
    fix = "Correction: my sister lives in Dallas, not Denver."
    say(ledger, "fix", fix, minute=0, hour=20, session="later-chat")
    await extract(ledger, Scripted({fix: [fact(fix, "my sister", "location", "Dallas", operation="correct",
                                                link="location")]}))
    found = stored(ledger)
    assert found["literal"]["retracted_by"] == found["lives"]["retracted_by"] == found["fix"]["id"]
    assert [row["value"] for row in readable(ledger) if not row["retracted_by"]] == ["Dallas"]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["reattribute", "annotate"])
async def test_words_a_correction_carries_from_a_claim_named_earlier_depend_on_that_message(action, tmp_path):
    """A partial correction that names the subject itself still carries the corrected claim's words ("Ford"), and
    those words are about the subject the earlier message named. Losing that message takes the correction, and the
    view formed on it, as it takes the claim itself; while it stands, the correction cites it."""
    from colonymind.self_model.judgments import SelfJudgments
    from test_self_judgments import proposal
    visited, bought = MOVED, "She bought a red Ford."
    fix = "Correction: My sister bought a blue one."
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    say(ledger, "moved", visited, minute=0)
    say(ledger, "bought", bought, minute=2)
    event = {"memory_kind": "substantive_event"}
    await extract(ledger, Scripted({bought: [fact(bought, "My sister", "car", "red Ford",
                                                  basis=(visited, "My sister"), **event)]}))
    say(ledger, "fix", fix, minute=0, hour=20, session="later-chat")
    await extract(ledger, Scripted({fix: [fact(fix, "My sister", "car", "blue Ford", operation="correct",
                                               link="car", **event)]}))
    found = stored(ledger)
    assert "subject_basis_message" not in found["fix"]
    assert [(part["text"].strip(), part.get("source_claim_id")) for part in found["fix"]["value_parts"]] == [
        ("blue", None), ("Ford", found["bought"]["id"])]
    row, = [row for row in readable(ledger) if not row["retracted_by"]]
    assert row["value"] == "blue Ford"
    assert row["value_basis"][0]["subject_basis"]["turn_id"] == "moved"
    bundle, = bundles(ledger, "Correction blue")
    assert "moved" in bundle["source_turn_ids"] and bundle["_source_message_hashes"]["moved"] == [hash_of(visited)]
    state = SelfJudgments(ledger, owner_id=OWNER)
    premise, = state.admitted_premises("fix")
    assert state.form(proposal([premise], topic="cars", source_ref="turn:fix",
                               session_id="later-chat")).disposition == "formed"
    with closing(ledger._connect()) as conn:
        dependencies = json.loads(conn.execute("SELECT dependency_json FROM self_judgment_revisions").fetchone()[0])
    assert {row["turn_id"] for row in dependencies} >= {"fix", "bought", "moved"}
    assert state.revisions()
    remove(ledger, action)
    assert readable(ledger) == [] and bundles(ledger, "Correction blue") == []
    assert ("fix" in stored(ledger)) is (action == "annotate")
    assert state.admitted_premises("fix") == [] and state.revisions() == []


@pytest.mark.asyncio
async def test_words_carried_from_a_claim_whose_subject_a_prior_claim_named_depend_on_that_claim_too(tmp_path):
    """The same holds for the other inherited subject: words carried from a correction that took its subject from a
    prior claim ("she" after "My sister moved last month.") are about that claim's subject, and go with it."""
    blue = "Correction: she moved into a blue house."
    dallas = "Correction: My sister moved into a house, same color, in Dallas."
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    say(ledger, "moved", MOVED, minute=0)
    await extract(ledger, Scripted({MOVED: [fact(MOVED, "My sister", "home", "moved last month")]}))
    say(ledger, "blue", blue, minute=0, hour=12, session="second")
    await extract(ledger, Scripted({blue: [fact(blue, "My sister", "home", "a blue house", operation="correct",
                                                link="home")]}))
    say(ledger, "dallas", dallas, minute=0, hour=14, session="third")
    await extract(ledger, Scripted({dallas: [fact(dallas, "My sister", "home", "a blue house in Dallas",
                                                  operation="correct", link="home")]}))
    found = stored(ledger)
    assert found["blue"]["subject_basis_claim_id"] == found["moved"]["id"]
    assert [part.get("source_claim_id") for part in found["dallas"]["value_parts"]] == [found["blue"]["id"], None]
    row, = [row for row in readable(ledger) if row["turn_id"] == "dallas"]
    assert row["value_basis"][0]["subject_basis"]["turn_id"] == "moved"
    remove(ledger, "annotate")
    assert [row["turn_id"] for row in readable(ledger) if row["turn_id"] != "moved"] == []


# ------------------------------------------------------------------ the dry run and re-extraction

@pytest.mark.asyncio
async def test_the_dry_run_offers_the_earlier_messages_and_writes_nothing(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    say(ledger, "moved", MOVED, minute=0)
    await extract(ledger, Scripted())
    with closing(ledger._connect()) as conn:
        before = conn.execute("SELECT count(*) FROM turn_sources").fetchone()[0], conn.execute(
            "SELECT count(*) FROM source_claims").fetchone()[0]
    model = Scripted({LIVES: [fact(LIVES, "My sister", "location", "Denver", basis=(MOVED, "My sister"))]})
    projection = SourceClaimProjection(ledger)
    result = await projection.rehearse(model, contact_id=OWNER, session_id=CHAT, text=LIVES, occurred_at=at(2))
    assert result["offered_earlier_messages"] == [{"id": "earlier_1", "turn_id": "moved", "reported_at": at(0),
                                                   "text": MOVED}]
    accepted, = result["accepted"]
    assert accepted["effect"] == "new" and accepted["claim"]["subject_basis_message"]["evidence"] == "My sister"
    prompts = await projection.rehearse(model, contact_id=OWNER, session_id=CHAT, text=LIVES, occurred_at=at(2),
                                        prompt_only=True)
    assert prompts["prompts"][0]["content"] == SYSTEM + source_claims.SUBJECT_BASIS_RULES
    assert json.loads(prompts["prompts"][1]["content"])["earlier_messages"][0]["text"] == MOVED
    with closing(ledger._connect()) as conn:
        assert (conn.execute("SELECT count(*) FROM turn_sources").fetchone()[0], conn.execute(
            "SELECT count(*) FROM source_claims").fetchone()[0]) == before


@pytest.mark.asyncio
async def test_re_extraction_offers_a_message_the_earlier_messages_of_its_conversation(tmp_path):
    from colonymind.beliefs import source_projection
    from colonymind.beliefs.reextract import Reextraction
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    say(ledger, "moved", MOVED, minute=0)
    say(ledger, "lives", LIVES, minute=2)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(source_projection, "EXTRACTION_VERSION", "source-claims-v18")
        await extract(ledger, Scripted({LIVES: [fact(LIVES, "She", "location", "Denver")]}))
    assert stored(ledger)["lives"]["subject"] == "She"
    model = Scripted({LIVES: [fact(LIVES, "My sister", "location", "Denver", basis=(MOVED, "My sister"))]})
    worker = Reextraction(SourceClaimProjection(ledger))
    worker.start(person=OWNER)
    for _ in range(20):
        if not await worker.process_one(model):
            break
    assert model.call("source_claim_extraction", LIVES).payload["earlier_messages"][0]["text"] == MOVED
    claim = stored(ledger)["lives"]
    assert claim["subject"] == "My sister" and claim["subject_basis_message"]["turn_id"] == "moved"


# ------------------------------------------------------------------ opinion premises

@pytest.mark.asyncio
async def test_a_premise_named_from_an_earlier_message_depends_on_that_message(tmp_path):
    from colonymind.self_model.judgments import SelfJudgments
    from test_self_judgments import proposal
    text = "She lost a week of work when her laptop died without a backup."
    ledger, _ = await sister(tmp_path)
    say(ledger, "lost", text, minute=5)
    await extract(ledger, Scripted({text: [fact(text, "My sister", "incident", "lost a week of work",
                                                memory_kind="substantive_event", basis=(MOVED, "My sister"))]}))
    state = SelfJudgments(ledger, owner_id=OWNER)
    premise, = state.admitted_premises("lost")
    result = state.form(proposal([premise], topic="backups", source_ref="turn:lost", session_id=CHAT))
    assert result.disposition == "formed"
    with closing(ledger._connect()) as conn:
        dependencies = json.loads(conn.execute("SELECT dependency_json FROM self_judgment_revisions").fetchone()[0])
    assert {row["turn_id"] for row in dependencies} >= {"lost", "moved"}
    remove(ledger, "annotate")
    assert state.admitted_premises("lost") == []


@pytest.mark.asyncio
async def test_a_correction_of_a_claim_named_earlier_keeps_its_basis_and_goes_with_it(tmp_path):
    ledger, _ = await sister(tmp_path)
    say(ledger, "correct", CORRECT, minute=0, hour=20, session="later-chat")
    model = Scripted({CORRECT: [fact(CORRECT, "My sister", "location", "Dallas", operation="correct",
                                     link="location")]})
    await extract(ledger, model)
    offered = model.call("source_claim_extraction", CORRECT).payload["prior_assertions"]
    assert offered[0]["subject_basis"]["evidence"] == "My sister"
    found = stored(ledger)
    assert found["correct"]["subject_basis_message"] == found["lives"]["subject_basis_message"]
    assert found["correct"]["prior_claim_id"] == found["lives"]["id"]
    assert found["lives"]["retracted_by"] == found["correct"]["id"]
    assert [row["value"] for row in readable(ledger) if not row["retracted_by"]] == ["Dallas"]
    remove(ledger, "erase")
    assert set(stored(ledger)) == {'lives', 'correct'}
    assert readable(ledger) == []
