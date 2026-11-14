"""OC4 provenance against the durable controller and canonical source ledger."""

from contextlib import closing
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from colonymind.hands.provenance import Provenance, hand_lesson_allowed, lesson_allowed, recalled_hand_marker, relay_marker
from colonymind.beliefs.source_projection import SourceClaimProjection
from colonymind.mind.lessons import Lessons
from colonymind.mind.outcomes import Autobiography
from colonymind.turns.idempotency import TurnIdempotencyLedger, source_message_hash
from colonymind.memory.curated import current_candidates
from test_hands_controller import c, finish, running
from test_hands_provenance_capture import captured


OWNER = "owner"
RECEIPT = {"scope": {"kind": "tests", "reference": "a" * 40,
                      "dirty_tree_sha256": "b" * 64, "command": "pytest tests/test_artifact.py",
                      "details": {"runner": "hands-checks-v1", "requirement_passed": True}},
           "passed": True, "exit_code": 0, "output_hash": "c" * 64}
FIELDS = {"signature": "artifact-check", "kind": "strategy", "title": "Check the final artifact",
          "when_to_use": "an artifact changes", "content": "Run the planned check on its final version."}


@pytest.fixture
def ledger(tmp_path):
    instance = TurnIdempotencyLedger(tmp_path / "turns.db")
    instance.record_source("owner-turn", contact_id=OWNER, session_id="parent", messages=[
        {"role": "user", "content": "Check the final artifact for Sam."}], derive_claims=False)
    return instance


def sources(ledger):
    with closing(ledger._connect()) as db:
        return {row["turn_id"]: json.loads(row["messages_json"])
                for row in db.execute("SELECT turn_id,messages_json FROM turn_sources")}


def provenance(c, ledger):
    biography = Autobiography(ledger, owner_id=OWNER, clock=c.clock)
    lessons = Lessons(ledger=ledger, store=None, owner_id=OWNER, autobiography=biography, clock=c.clock)
    return Provenance(c, ledger, owner_id=OWNER, lessons=lessons)


async def result_at(c, level="E3", key="plan-1"):
    p = await running(c, key=key)
    c.evidence_evaluator = lambda *args: {
        "level": level, "scope": deepcopy(RECEIPT["scope"])}
    await finish(c, p)
    async def receipts():
        await c.update("hands", "id", p["hand_id"], check=[deepcopy(RECEIPT)])
        return {}
    await c.write("receipts:" + key, "/test/receipts", {}, receipts)
    return p


async def test_only_e3_accepted_result_writes_one_entry_with_embedded_receipts(c, ledger):
    service = provenance(c, ledger)
    low = await result_at(c, "E2")
    assert await service.record_accepted(low["hand_id"], "The artifact opens.") is None
    high = await result_at(c, key="second")
    identifier = await service.record_accepted(high["hand_id"], "The planned artifact check passed.")
    assert identifier and identifier.startswith("hand:")
    assert await service.record_accepted(high["hand_id"], "A replay uses no new words.") == identifier
    entries = {key: messages for key, messages in sources(ledger).items() if key.startswith("hand:")}
    assert list(entries) == [identifier]
    metadata = entries[identifier][0]["metadata"]
    assert metadata["origin"] == "hand" and metadata["hand_id"] == high["hand_id"]
    assert metadata["evidence"]["level"] == "E3"
    assert metadata["receipts"] == [RECEIPT]
    row = await c.one("SELECT * FROM derivations WHERE derived_id=?", (identifier,))
    assert row["derived_kind"] == "autobiography" and row["state"] == "live"


async def test_e0_relay_keeps_marker_and_cannot_become_lesson(c, ledger):
    service = provenance(c, ledger)
    p = await result_at(c, "E0")
    refs = await service.record_relay("relay-turn", [p["hand_id"]])
    ledger.record_source("relay-turn", contact_id=OWNER, session_id="parent", messages=[
        {"role": "assistant", "content": "Robin found that the artifact works.",
         "metadata": {"hand_refs": refs}}], derive_claims=True)
    marker = relay_marker(refs, controller_path=c.store.path, derived_id="relay-turn")
    assert "own analysis, not checked" in marker and "E0" in marker
    assert not hand_lesson_allowed(ledger, owner_id=OWNER, lineage=["relay-turn"])
    assert await service.admit_lesson(p["hand_id"], FIELDS) is None
    class NeverExtract:
        async def complete(self, *args, **kwargs):
            pytest.fail("Assistant relays must never reach claim extraction")
    projection = SourceClaimProjection(ledger)
    while await projection.process_one(NeverExtract()):
        pass
    with closing(ledger._connect()) as db:
        assert not db.execute("SELECT 1 FROM source_claims WHERE turn_id='relay-turn'").fetchone()


async def test_expiry_keeps_entry_and_embedded_support_but_marks_relays(c, ledger):
    service = provenance(c, ledger)
    p = await result_at(c)
    identifier = await service.record_accepted(p["hand_id"], "The artifact passed the planned check.")
    refs = await service.record_relay("relay-turn", [p["hand_id"]])
    await service.expire_branch("branch-1")
    assert identifier in sources(ledger)
    assert "evidence expired" in relay_marker(refs, controller_path=c.store.path, derived_id="relay-turn")
    rows = await c.all("SELECT derived_kind,state FROM derivations")
    assert {row["derived_kind"]: row["state"] for row in rows} == {
        "autobiography": "live", "relay": "downgraded"}
    assert hand_lesson_allowed(ledger, owner_id=OWNER, lineage=[identifier])
    message = sources(ledger)[identifier][0]
    recalled = {"kind": "source_quote", "role": "assistant", "source_turn_id": identifier,
                "source_message_hash": source_message_hash("parent", message),
                "contact_id": OWNER, "session_id": "later"}
    marker = recalled_hand_marker(recalled, ledger_path=ledger.db_path, controller_path=c.store.path)
    assert "requirement tested" in marker and "pytest tests/test_artifact.py" in marker
    assert "expired" not in marker


async def test_erasure_retracts_entry_lesson_and_relay_and_does_not_recreate(c, ledger):
    service = provenance(c, ledger)
    p = await result_at(c)
    identifier = await service.record_accepted(p["hand_id"], "The artifact passed the planned check.")
    lesson = await service.admit_lesson(p["hand_id"], FIELDS)
    assert lesson is not None
    assert await service.admit_lesson(p["hand_id"], FIELDS) == lesson
    refs = await service.record_relay("relay-turn", [p["hand_id"]])
    ledger.record_source("relay-turn", contact_id=OWNER, session_id="parent", messages=[
        {"role": "assistant", "content": "The artifact passed.", "metadata": {"hand_refs": refs}}],
        derive_claims=False)
    await service.erase_parent("parent")
    assert identifier in sources(ledger)
    assert not current_candidates(ledger, [{'turn_id': identifier}], contact_id=OWNER)
    assert service.lessons.get(lesson.id) is None
    assert "relay-turn" in sources(ledger)
    assert not current_candidates(ledger, [{'turn_id': 'relay-turn'}], contact_id=OWNER)
    assert all(row["state"] == "retracted" for row in await c.all("SELECT * FROM derivations"))
    assert await service.record_accepted(p["hand_id"], "A retry.") is None


async def test_owner_dispute_retains_entry_but_invalidates_lessons(c, ledger):
    service = provenance(c, ledger)
    p = await result_at(c)
    identifier = await service.record_accepted(p["hand_id"], "The artifact passed the planned check.")
    lesson = await service.admit_lesson(p["hand_id"], FIELDS)
    ledger.record_source("owner-reversal", contact_id=OWNER, session_id="parent", messages=[
        {"role": "user", "content": "That result was wrong. The artifact fails for Sam."}], derive_claims=False)
    await service.dispute(p["hand_id"], "owner-reversal")
    assert identifier in sources(ledger)
    assert service.lessons.get(lesson.id).status == "retired"
    assert not hand_lesson_allowed(ledger, owner_id=OWNER, lineage=[identifier])
    assert await service.admit_lesson(p["hand_id"], FIELDS) is None
    hand = await c.get_hand(p["hand_id"])
    assert hand["decisions"]["disputed"] is True
    with closing(ledger._connect()) as db:
        assert db.execute("SELECT 1 FROM source_annotations WHERE target_source_id=?", (identifier,)).fetchone()


async def test_dispute_without_lesson_service_still_retires_stored_lessons(c, ledger):
    service = provenance(c, ledger)
    p = await result_at(c)
    await service.record_accepted(p["hand_id"], "The artifact passed the planned check.")
    lesson = await service.admit_lesson(p["hand_id"], FIELDS)
    ledger.record_source("owner-reversal", contact_id=OWNER, session_id="parent", messages=[
        {"role": "user", "content": "That result is wrong for Sam."}], derive_claims=False)
    await Provenance(c, ledger, owner_id=OWNER).dispute(p["hand_id"], "owner-reversal")
    assert service.lessons.get(lesson.id).status == "retired"


async def test_mixed_lineage_never_upgrades_an_unverified_relay(c, ledger):
    service = provenance(c, ledger)
    high = await result_at(c)
    entry = await service.record_accepted(high["hand_id"], "The planned check passed.")
    low = await result_at(c, "E0", key="second")
    refs = await service.record_relay("relay-turn", [low["hand_id"]])
    ledger.record_source("relay-turn", contact_id=OWNER, session_id="parent", messages=[
        {"role": "assistant", "content": "Robin thinks a different artifact works.",
         "metadata": {"hand_refs": refs}}], derive_claims=False)
    assert not hand_lesson_allowed(ledger, owner_id=OWNER, lineage=[entry, "relay-turn"])
    assert hand_lesson_allowed(ledger, owner_id=OWNER, lineage=["owner-turn"])


def test_self_report_and_claimed_failure_cannot_authorize_lessons():
    assert not lesson_allowed({"level": "E0"})
    assert not lesson_allowed({"level": "E0"}, observed_failure=True)
    assert lesson_allowed({"level": "E3", "receipts": [RECEIPT]})
    assert lesson_allowed({"level": "E1", "receipts": [{**RECEIPT, "passed": False,
                                                          "exit_code": 1}]}, observed_failure=True)


async def test_non_owner_dispute_does_not_change_evidence(c, ledger):
    service = provenance(c, ledger)
    p = await result_at(c)
    await service.record_accepted(p["hand_id"], "The artifact passed the planned check.")
    ledger.record_source("guest-reversal", contact_id="sam", session_id="guest", messages=[
        {"role": "user", "content": "That result was wrong."}], derive_claims=False)
    with pytest.raises(ValueError, match="owner"):
        await service.dispute(p["hand_id"], "guest-reversal")
    assert not (await c.get_hand(p["hand_id"]))["decisions"].get("disputed")


async def test_capture_recall_and_lessons_share_expiry_and_erasure(c, captured):
    from colonymind.memory.recall import render_memory_context, source_candidates

    module, capture, outbox, settings, ledger = captured
    service = provenance(c, ledger)

    class Client:
        def post(self, path, *, json, timeout):
            ledger.record_source(json["context"]["turn_id"], contact_id=OWNER, session_id="parent",
                                 messages=[json["user_message"], json["assistant_message"]])
            return SimpleNamespace(status_code=200, is_success=True, json=lambda: {})

    async def relay(plan, turn, word):
        hand = await c.get_hand(plan["hand_id"])
        refs = await service.record_relay(turn, [hand["id"]])
        line = f"{hand['name']} (deep, text): done; {hand['evidence']['level']}"
        capture.record_hand_refs(session_id="parent", turn_id=turn,
            digest={"text": "Your hands:\n- " + line, "items": [{
                "hand_id": hand["id"], "text": line, "evidence": hand["evidence"]}]})
        capture.post_llm_call(session_id="parent", turn_id=turn, user_message=f"Check the {word} valve.",
                              assistant_response=f"The {word} valve is ready.")
        payload = next(row["payload"] for row in outbox.rows() if row["turn_id"] == turn)
        assert payload["hand_refs"] == refs
        assert module.deliver(Client(), capture.sessions, settings, payload)

    def recalled(word):
        hits = source_candidates(ledger.search_sources(word, contact_id=OWNER, session_id="later"))
        hits = current_candidates(ledger, hits, contact_id=OWNER)
        return render_memory_context([row for row in hits if row["role"] == "assistant"])

    low = await result_at(c, "E0")
    await relay(low, "relay-low", "silver")
    assert "own analysis, not checked" in recalled("silver")
    assert service.lessons.admit(FIELDS, verified="check", origin="night", status="active",
        evidence=["relay-low"], lineage=["relay-low"]) is None

    high = await result_at(c, key="second")
    entry = await service.record_accepted(high["hand_id"], "The planned valve check passed.")
    await relay(high, "relay-high", "copper")
    assert "requirement tested" in recalled("copper")
    lesson = await service.admit_lesson(high["hand_id"], FIELDS)
    assert lesson is not None

    await service.expire_branch("branch-1")
    assert "evidence expired" in recalled("copper")
    assert entry in sources(ledger) and service.lessons.get(lesson.id) is not None

    await service.erase_parent("parent")
    assert entry in sources(ledger) and service.lessons.get(lesson.id) is None
    assert not recalled("copper")


async def scoped_result(c, level, scope):
    p = await running(c)
    c.evidence_evaluator = lambda *args: {"level": level, "scope": deepcopy(scope)}
    await finish(c, p)
    assert not (await c.get_hand(p["hand_id"]))["check"]
    return p


async def test_independent_review_without_runner_check_embeds_bound_review_receipt(c, ledger):
    service = provenance(c, ledger)
    scope = {"kind": "deep", "reference": "branch-1", "sha256": "a" * 64,
             "producer_families": ["family-a"], "reviewer_family": "family-b", "verdict": "pass"}
    p = await scoped_result(c, "E4", scope)
    entry = await service.record_accepted(p["hand_id"], "The independent review passed.")
    assert entry is not None
    receipt = sources(ledger)[entry][0]["metadata"]["receipts"][0]
    assert receipt["kind"] == "independent_review"
    assert receipt["scope"] == scope and receipt["passed"] is True
    assert lesson_allowed({"level": "E4", "scope": scope})
    assert await service.record_accepted(p["hand_id"], "A replay.") == entry
    await service.expire_branch("branch-1")
    assert await service.admit_lesson(p["hand_id"], FIELDS) is not None


@pytest.mark.parametrize("changes", [
    {"reviewer_family": "family-a"}, {"verdict": "unclear"}, {"sha256": "missing"},
    {"reference": "earlier-branch"}, {"producer_families": ["family-b"]},
])
async def test_review_only_result_requires_independence_and_final_binding(c, ledger, changes):
    service = provenance(c, ledger)
    scope = {"kind": "deep", "reference": "branch-1", "sha256": "a" * 64,
             "producer_families": ["family-a"], "reviewer_family": "family-b", "verdict": "pass", **changes}
    p = await scoped_result(c, "E4", scope)
    assert await service.record_accepted(p["hand_id"], "The review passed.") is None
    assert await service.admit_lesson(p["hand_id"], FIELDS) is None


@pytest.mark.parametrize("author,role", [("sam", "user"), (OWNER, "assistant")])
async def test_owner_only_receipt_refuses_non_owner_or_assistant_sources(c, ledger, author, role):
    service = provenance(c, ledger)
    ledger.record_source("claimed-acceptance", contact_id=author, session_id="parent", messages=[
        {"role": role, "content": "I accept the result."}], derive_claims=False)
    scope = {"kind": "deep", "reference": "branch-1", "sha256": "a" * 64,
             "details": {"owner_answer_ref": "claimed-acceptance"}}
    p = await scoped_result(c, "E5", scope)
    assert await service.record_accepted(p["hand_id"], "The owner accepted it.") is None
    assert await service.admit_lesson(p["hand_id"], FIELDS) is None

from colonymind.memory import curated

async def test_erased_source_cannot_be_recreated_by_replayed_result(c, ledger):
    service = provenance(c, ledger)
    p = await result_at(c)
    entry = await service.record_accepted(p["hand_id"], "The planned check passed.")
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(ledger)
    _source_forget_receipt = curated.forget_sources(ledger, OWNER, source_ids=[entry])
    assert historical_bytes(ledger) == _native_before_forget
    assert all(curated.is_forgotten(ledger, sid) for sid in _source_forget_receipt['source_ids'])
    assert await service.record_accepted(p["hand_id"], "The planned check passed.") is None


async def test_owner_acceptance_without_runner_check_embeds_attributed_owner_receipt(c, ledger):
    service = provenance(c, ledger)
    ledger.record_source("owner-accepted", contact_id=OWNER, session_id="later", messages=[
        {"role": "user", "content": "I accept the result for Sam."}], derive_claims=False)
    scope = {"kind": "deep", "reference": "branch-1", "sha256": "a" * 64,
             "details": {"owner_answer_ref": "owner-accepted"}}
    p = await scoped_result(c, "E5", scope)
    entry = await service.record_accepted(p["hand_id"], "The owner accepted the result.")
    assert entry is not None
    receipt = sources(ledger)[entry][0]["metadata"]["receipts"][0]
    assert receipt["kind"] == "owner_acceptance"
    assert receipt["source_id"] == "owner-accepted" and receipt["contact_id"] == OWNER
    assert receipt["owner_messages"][0]["content"] == "I accept the result for Sam."
    assert len(receipt["source_version"]) == 64
    await service.expire_branch("branch-1")
    assert await service.admit_lesson(p["hand_id"], FIELDS) is not None
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(ledger)
    _source_forget_receipt = curated.forget_sources(ledger, OWNER, source_ids=['owner-accepted'])
    assert historical_bytes(ledger) == _native_before_forget
    assert all(curated.is_forgotten(ledger, sid) for sid in _source_forget_receipt['source_ids'])
