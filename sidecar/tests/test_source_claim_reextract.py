"""Re-extraction: history's claims formed again under the current extraction version (docs/SOURCE-CLAIMS.md,
Re-extraction).

Sources are never backfilled, so a better extractor (priors by meaning, the key list, vague dates as periods) helps
only new messages. Re-extraction queues the attributed person-scope sources whose claims an older version formed and
extracts them again, one contact at a time in occurrence order, through the same validation, review and commit. An
old claim is retired (``reextracted_by``) only in the transaction that commits the new ones, and what the owner and
the store decided about the old claims carries forward: a retraction, a supersession, an owner's answer, an owner's
correction, an owner's decision on a held change. Nothing retracted comes back.

Everything runs the real ledger, projection and commit with a scripted extractor.
"""
from contextlib import closing
from datetime import datetime, timezone
import json
import sqlite3
from types import SimpleNamespace

import pytest

from colonymind.beliefs import reextract, source_projection
from colonymind.beliefs.reextract import Reextraction
from colonymind.beliefs.source_claims import EXTRACTION_VERSION, norm_value
from colonymind.beliefs.source_projection import SourceClaimProjection
from colonymind.memory import learning, live_recall
from colonymind.mind.consolidate import Consolidation
from colonymind.turns import TurnIdempotencyLedger
from test_source_claim_projection import claim as fact
from test_source_quoted_preferences import TEXT as HANDOFF, preference

OWNER = "owner"
OLD = "source-claims-v17"
USAGE = {"prompt_tokens": 900, "completion_tokens": 100, "total_tokens": 1000}

RIVER = "My office is in River."
LAKE = "Correction: my office is in Lake, not River."
TEA = "My tea preference is green tea."
BIKE = "My bike is at the station."
MOVED = "We relocated to Munich, and I work there now."


class Scripted:
    """The extraction call returns the scripted proposals of its message (``outputs[text]``, a list). A proposal's
    ``link`` names a predicate: its prior is the offered record of that predicate, as a model would choose it.
    Review keeps every proposal. ``fail`` lists messages whose extraction raises, ``invalid`` those it answers with
    something that is not JSON. Every call is recorded with its task and payload."""

    def __init__(self, outputs, *, fail=(), invalid=()):
        self.outputs, self.fail, self.invalid, self.calls = outputs, set(fail), set(invalid), []

    def tier_config(self, tier):
        return SimpleNamespace(base_url="http://127.0.0.1:8080/v1")

    def extracted(self):
        return [payload["message"] for task, payload in self.calls if task == "source_claim_extraction"]

    def payload(self, text):
        return next(payload for task, payload in self.calls
                    if task == "source_claim_extraction" and payload["message"] == text)

    async def complete(self, messages, **kwargs):
        payload = json.loads(messages[-1]["content"])
        task = kwargs.get("context", {}).get("task")
        self.calls.append((task, payload))
        if task == "source_claim_review":
            return SimpleNamespace(model_id="fixture-model", usage=USAGE, content=json.dumps({
                str(row["index"]): {"keep": True, "reason": "Scripted review keeps this fixture."}
                for row in payload["proposals"]}))
        text = payload["message"]
        if text in self.fail:
            raise ConnectionError("model endpoint unavailable")
        if text in self.invalid:
            return SimpleNamespace(model_id="fixture-model", usage=USAGE, content="no claims here")
        proposals = []
        for proposal in self.outputs.get(text, []):
            proposal = dict(proposal)
            link = proposal.pop("link", None)
            if link:
                wanted = norm_value(link.replace("_", " "))
                proposal["prior_claim_id"] = next((row["id"] for row in payload["prior_assertions"]
                                                   if row["predicate"] == wanted), None)
            proposals.append(proposal)
        return SimpleNamespace(model_id="fixture-model", usage=USAGE, content=json.dumps({"claims": proposals}))


def at(hour):
    return f"2026-03-01T{hour:02d}:00:00+00:00"


def ledger_at(tmp_path):
    return TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")


async def history(ledger, turns, outputs, *, version=OLD, contact=OWNER):
    """Each ``(turn, text, hour)`` recorded as the contact's message and extracted under ``version``."""
    projection = SourceClaimProjection(ledger)
    model = Scripted(outputs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(source_projection, "EXTRACTION_VERSION", version)
        for turn, text, hour in turns:
            ledger.record_source(turn, contact_id=contact, session_id="session-" + turn, occurred_at=at(hour),
                                 messages=[{"role": "user", "content": text}])
            assert await projection.process_one(model)
    return model


async def drain(worker, model, rounds=100):
    for _ in range(rounds):
        if not await worker.process_one(model):
            return
    raise AssertionError("re-extraction did not settle")


async def reextract_all(ledger, outputs, *, person=None, limit=None, model=None):
    model = model or Scripted(outputs)
    worker = Reextraction(SourceClaimProjection(ledger))
    started = worker.start(person=person, limit=limit)
    await drain(worker, model)
    return SimpleNamespace(model=model, worker=worker, started=started)


def claims(ledger):
    with closing(ledger._connect()) as conn:
        # The columns hold the lifecycle (a supersession closes ``valid_to`` there): they win over the claim's data.
        return {row["id"]: json.loads(row["data_json"]) | dict(row) for row in conn.execute("SELECT * FROM source_claims")}


def by_turn(ledger):
    found = {}
    for row in claims(ledger).values():
        found.setdefault(row["turn_id"], []).append(row)
    return found


def live(ledger, contact=OWNER):
    projection = SourceClaimProjection(ledger)
    with closing(ledger._connect()) as conn:
        rows = projection._rows(conn, contact, "")
    return sorted(row["value"] for row in rows if not row["superseded_by"] and not row["retracted_by"])


def jobs(ledger):
    with closing(ledger._connect()) as conn:
        return {row["turn_id"]: dict(row) for row in conn.execute("SELECT * FROM source_reextract_jobs")}


def reports(ledger):
    with closing(ledger._connect()) as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM source_reextract_reports ORDER BY id")]


def retired(ledger):
    with closing(ledger._connect()) as conn:
        return {row["claim_id"]: dict(row) for row in conn.execute("SELECT * FROM source_claim_reextractions")}


OFFICE = {RIVER: [fact(RIVER, "River")],
          LAKE: [fact(LAKE, "Lake", operation="correct", link="office_location")],
          TEA: [fact(TEA, "green tea", predicate="tea_preference")]}


# -- 1. idempotent -------------------------------------------------------------------------------------------------

async def test_running_it_twice_is_idempotent_and_an_unchanged_reading_keeps_its_claim(tmp_path):
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", RIVER, 9), ("t2", TEA, 10), ("t3", LAKE, 11)], OFFICE)
    before = claims(ledger)
    assert {row["extraction_version"] for row in before.values()} == {OLD}
    first = await reextract_all(ledger, OFFICE)
    assert first.started["queued"] == 3 and first.model.extracted() == [RIVER, TEA, LAKE]
    after = claims(ledger)
    # The same reading of the same message is the same claim: its id, links and lifecycle stay.
    assert set(after) == set(before)
    assert {row["extraction_version"] for row in after.values()} == {EXTRACTION_VERSION}
    for ident, row in before.items():
        assert {key: after[ident][key] for key in ("retracted_by", "superseded_by", "valid_from", "valid_to",
                                                    "operation", "prior_claim_id")} == {
            key: row[key] for key in ("retracted_by", "superseded_by", "valid_from", "valid_to", "operation",
                                      "prior_claim_id")}
    assert live(ledger) == ["Lake", "green tea"]
    with closing(ledger._connect()) as conn:
        assert {row[0] for row in conn.execute("SELECT extraction_version FROM source_claim_jobs")} == {
            EXTRACTION_VERSION}
    assert retired(ledger) == {}
    assert {row["status"] for row in jobs(ledger).values()} == {"complete"}

    second = await reextract_all(ledger, OFFICE)
    assert second.started["queued"] == 0 and second.model.calls == []
    assert claims(ledger) == after


async def test_the_current_version_links_a_change_the_older_one_missed(tmp_path):
    """What re-extraction is for: the older version formed the relocation as a second office, the current one
    offers the prior and the change supersedes it, in the order the messages occurred."""
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", RIVER, 9), ("t2", MOVED, 10)],
                  {RIVER: [fact(RIVER, "River")], MOVED: [fact(MOVED, "Munich")]})
    assert live(ledger) == ["Munich", "River"]
    await reextract_all(ledger, {RIVER: [fact(RIVER, "River")],
                                 MOVED: [fact(MOVED, "Munich", operation="change", link="office_location")]})
    assert live(ledger) == ["Munich"]
    rows = by_turn(ledger)
    [river], [munich] = rows["t1"], rows["t2"]
    assert river["superseded_by"] == munich["id"] and river["valid_to"] == at(10)


async def test_a_vague_date_read_again_becomes_its_period_on_the_same_claim(tmp_path):
    trip = "I visited Lisbon in 2015."
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", trip, 9)], {trip: [fact(trip, "Lisbon", predicate="visited_city")]})
    [before] = claims(ledger).values()
    assert before["event_time"]["status"] != "resolved"
    await reextract_all(ledger, {trip: [fact(trip, "Lisbon", predicate="visited_city", event_at_text="in 2015")]})
    [after] = claims(ledger).values()
    assert after["id"] == before["id"]
    assert (after["event_time"]["precision"], after["event_time"]["period"]) == ("period", "year")
    assert after["event_time"]["start"] == "2015-01-01T00:00:00+00:00"


# -- 2. never revived ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["link missed", "key drift", "correction reads nothing"])
async def test_a_retracted_value_is_never_revived_and_the_owners_correction_carries_forward(tmp_path, case):
    """The owner's correction retracted River. Read again, the correction may lose its link, both messages may be
    filed under another key, or the correction may form no claim at all: River stays retracted in every case."""
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", RIVER, 9), ("t2", LAKE, 10)], OFFICE)
    [river] = by_turn(ledger)["t1"]
    [lake] = by_turn(ledger)["t2"]
    assert river["retracted_by"] == lake["id"]
    outputs = {"link missed": {RIVER: [fact(RIVER, "River")], LAKE: [fact(LAKE, "Lake")]},
               "key drift": {RIVER: [fact(RIVER, "River", predicate="workplace_city")],
                             LAKE: [fact(LAKE, "Lake", predicate="workplace_city")]},
               "correction reads nothing": {RIVER: [fact(RIVER, "River")], LAKE: []}}[case]
    await reextract_all(ledger, outputs)
    assert live(ledger) == ([] if case == "correction reads nothing" else ["Lake"])
    rows = by_turn(ledger)
    [river_now] = rows["t1"]
    assert river_now["retracted_by"]
    if case == "key drift":
        [lake_now] = rows["t2"]
        assert river_now["predicate"] == lake_now["predicate"] == "workplace city"
        assert river_now["retracted_by"] == lake_now["id"]          # the link follows the correction's new claim
        assert retired(ledger)[river["id"]]["reextracted_by"] == river_now["id"]
        assert retired(ledger)[lake["id"]]["reextracted_by"] == lake_now["id"]
    if case == "link missed":
        [lake_now] = rows["t2"]
        assert (lake_now["operation"], lake_now["prior_claim_id"]) == ("correct", river["id"])


@pytest.mark.parametrize("order", ["said before the correction", "said again after it"])
async def test_a_value_read_anew_from_an_earlier_message_keeps_the_later_correction(tmp_path, order):
    """The older version read nothing in the middle message; the current one reads River there. Said before the
    owner corrected River, it is retracted with it; said again after the correction, it is a new statement."""
    middle = "I cycle to River every morning for work."
    turns = [("t1", RIVER, 9), ("t2", middle, 10), ("t3", LAKE, 11)] if order.endswith("correction") else [
        ("t1", RIVER, 9), ("t3", LAKE, 10), ("t2", middle, 11)]
    ledger = ledger_at(tmp_path)
    await history(ledger, turns, {**OFFICE, middle: []})
    await reextract_all(ledger, {**OFFICE, middle: [fact(middle, "River")]})
    [again] = by_turn(ledger)["t2"]
    [lake] = by_turn(ledger)["t3"]
    if order.endswith("correction"):
        assert again["retracted_by"] == lake["id"] and live(ledger) == ["Lake"]
    else:
        assert not again["retracted_by"] and live(ledger) == ["Lake", "River"]


@pytest.mark.parametrize("case", ["another subject", "two candidates"])
async def test_a_correction_never_moves_to_a_claim_about_something_else(tmp_path, case):
    """The owner corrected where their office is. Read again, the message may yield only a claim about someone
    else with the same value, or two claims of theirs that could each have been the office: the correction moves
    to neither (a successor needs the same subject, and only one candidate)."""
    text = {"another subject": "My office is in River, and Alex works in River too.",
            "two candidates": "My office and my studio are both in River."}[case]
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", text, 9), ("t2", LAKE, 10)], {text: [fact(text, "River")], LAKE: OFFICE[LAKE]})
    [river] = by_turn(ledger)["t1"]
    assert river["retracted_by"] and live(ledger) == ["Lake"]
    again = {"another subject": [fact(text, "River", subject="Alex", predicate="work_city")],
             "two candidates": [fact(text, "River", predicate="studio_city"),
                                fact(text, "River", predicate="workplace_city")]}[case]
    await reextract_all(ledger, {text: again, LAKE: OFFICE[LAKE]})
    rows = {row["predicate"]: row for row in by_turn(ledger)["t1"]}
    kept = rows["work city" if case == "another subject" else "studio city"]
    assert not kept["retracted_by"] and not kept["superseded_by"]


def recalled(ledger, query):
    """What recall shows for ``query``: the claim bundles, and the quotations no claim covers."""
    from colonymind.beliefs.source_time import interpret_time_query
    hits = ledger.search_sources(query, contact_id=OWNER, session_id="session-recall", limit=10)
    _, rows = SourceClaimProjection(ledger).prepare_context(
        [], hits, contact_id=OWNER, session_id="session-recall",
        time_query=interpret_time_query(query, now=datetime(2026, 3, 20, tzinfo=timezone.utc)))
    return [row["content"] for row in rows if not row.get("atomic_evidence")]


@pytest.mark.parametrize("said", ["once", "twice"])
async def test_a_corrected_claim_the_new_reading_lacks_stays_corrected_in_recall(tmp_path, monkeypatch, said):
    """The owner corrected River to Lake. Read again, the message that said River (once, or a second time, which
    the learning faculty keeps as a witness of the same claim) yields no claim: the corrected claim stays as it
    was, so recall never offers the River quotation as if nothing had corrected it."""
    monkeypatch.setenv(learning.FLAG, "1")
    again = "As I said, my office is in River."
    turns = [("t1", RIVER, 9)] + ([("t2", again, 10)] if said == "twice" else []) + [("t3", LAKE, 11)]
    ledger = ledger_at(tmp_path)
    await history(ledger, turns, {**OFFICE, again: [fact(again, "River")]})
    before = claims(ledger)
    assert not [text for text in recalled(ledger, "office River") if "River" in text]
    await reextract_all(ledger, {RIVER: [], again: [], LAKE: OFFICE[LAKE]})
    assert not [text for text in recalled(ledger, "office River") if "River" in text]
    river = [row for row in claims(ledger).values() if row["value"] == "River"]
    assert [row["id"] for row in river] == [row["id"] for row in before.values() if row["value"] == "River"]
    assert all(row["retracted_by"] for row in river) and live(ledger) == ["Lake"]


async def test_a_value_read_anew_stays_corrected_when_the_corrected_claim_changed_key(tmp_path):
    """The older version read River as the office and nothing in the middle message; the current one files River
    under another key, reads River in the middle message under the old key, and the correction under the new one.
    The middle message's River was said before the correction: it is retracted with it, though the claim that
    held the old key has left the store."""
    middle = "I cycle to River every morning for work."
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", RIVER, 9), ("t2", middle, 10), ("t3", LAKE, 11)], {**OFFICE, middle: []})
    await reextract_all(ledger, {RIVER: [fact(RIVER, "River", predicate="workplace_city")],
                                 middle: [fact(middle, "River")],
                                 LAKE: [fact(LAKE, "Lake", predicate="workplace_city", operation="correct",
                                             link="workplace_city")]})
    [again] = by_turn(ledger)["t2"]
    [lake] = by_turn(ledger)["t3"]
    assert again["retracted_by"] == lake["id"] and live(ledger) == ["Lake"]


# -- 3. the owner's answers and decisions carry forward -------------------------------------------------------------

ACME, GLOBEX = "I work at Acme.", "My employer is Globex now."


def owner_answered(night, first, second, answer, *, ref, now):
    """What an earlier build wrote when the owner answered its question about two statements (the engine asks none
    now; a store still holds these records): ``first`` keeps the first and retracts the second by ``ref``, ``both``
    records the pair as compatible; each change is in the learning log as the owner's answer."""
    drop = [second] if answer == "first" else []
    with night._conn() as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        dropped = [*drop, *night._retract(conn, OWNER, drop, [first, second], ref, now)]
        for action, idents in (("declined", dropped), ("ratified", [c for c in (first, second) if c not in drop])):
            for ident in idents:
                learning.log(conn, store="claim", action=action, stage="owner_answer",
                             reason=f"the owner answered {answer}", item_id=ident, contact_id=OWNER, refs=[ref])
    if answer == "both":
        night._record_relation({"id": first}, {"id": second}, "adds", contact_id=OWNER, by=ref, now=now)


@pytest.mark.parametrize("case", ["the same key, now linked", "another key"])
async def test_an_owners_answer_carries_forward(tmp_path, case):
    """The owner answered an earlier build's contradiction question: keep Acme, drop Globex. Read again, Globex may
    come back as a change of Acme, or under another key: Acme stays, Globex stays dropped, and the refused link is
    reported."""
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", ACME, 9), ("t2", GLOBEX, 10)],
                  {ACME: [fact(ACME, "Acme", predicate="employer")],
                   GLOBEX: [fact(GLOBEX, "Globex", predicate="employer")]})
    [acme] = by_turn(ledger)["t1"]
    [globex] = by_turn(ledger)["t2"]
    night = Consolidation(store=None, ledger=ledger, concerns=None, mind_state=None, contacts=None, router=None,
                          autobiography=None, owner_id=OWNER, budgets=None, tokens_allowed=None, faculties={})
    owner_answered(night, acme["id"], globex["id"], "first", ref="intention:ask-1",
                   now=datetime(2026, 3, 2, tzinfo=timezone.utc))
    assert live(ledger) == ["Acme"]
    again = {"the same key, now linked": [fact(GLOBEX, "Globex", predicate="employer", operation="change",
                                               link="employer")],
             "another key": [fact(GLOBEX, "Globex", predicate="works_for")]}[case]
    await reextract_all(ledger, {ACME: [fact(ACME, "Acme", predicate="employer")], GLOBEX: again})
    assert live(ledger) == ["Acme"]
    [acme_now] = by_turn(ledger)["t1"]
    [globex_now] = by_turn(ledger)["t2"]
    assert acme_now["id"] == acme["id"] and not acme_now["superseded_by"] and acme_now["valid_to"] is None
    assert globex_now["retracted_by"] == "intention:ask-1"
    if case == "the same key, now linked":
        assert [(row["kind"], row["claim_id"]) for row in reports(ledger)] == [("owner_kept", acme["id"])]


async def test_an_owners_answer_that_both_hold_follows_a_claim_that_changed_id(tmp_path):
    """"Both hold" is kept in the mind's state as the pair's decided relation, so the pair is never asked again.
    When the second statement is read again under another key its id changes: the relation follows it."""
    from colonymind.mind.concerns import Concerns
    from colonymind.mind.consolidate import RELATION_KEY, _pair_id
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", ACME, 9), ("t2", GLOBEX, 10)],
                  {ACME: [fact(ACME, "Acme", predicate="employer")],
                   GLOBEX: [fact(GLOBEX, "Globex", predicate="employer")]})
    [acme] = by_turn(ledger)["t1"]
    [globex] = by_turn(ledger)["t2"]
    state = Concerns(tmp_path / "mind.db").state
    night = Consolidation(store=None, ledger=ledger, concerns=None, mind_state=state, contacts=None, router=None,
                          autobiography=None, owner_id=OWNER, budgets=None, tokens_allowed=None, faculties={})
    owner_answered(night, acme["id"], globex["id"], "both", ref="intention:ask-2",
                   now=datetime(2026, 3, 2, tzinfo=timezone.utc))
    assert night.relation(acme["id"], globex["id"]) == "adds"
    model = Scripted({ACME: [fact(ACME, "Acme", predicate="employer")],
                      GLOBEX: [fact(GLOBEX, "Globex", predicate="works_for")]})
    worker = Reextraction(SourceClaimProjection(ledger), relations=lambda: state)
    worker.start()
    await drain(worker, model)
    [globex_now] = by_turn(ledger)["t2"]
    assert globex_now["id"] != globex["id"] and live(ledger) == ["Acme", "Globex"]
    assert night.relation(acme["id"], globex_now["id"]) == "adds"
    assert state.get(RELATION_KEY + _pair_id(acme["id"], globex_now["id"]))["causes"][2] == "intention:ask-2"


@pytest.mark.parametrize("drift", [False, True])
async def test_the_owners_decision_on_a_held_change_carries_forward(tmp_path, monkeypatch, drift):
    """A preference change the owner's words did not name was held, and the owner declined it (both hold). Read
    again, under the same key or another, the change is not put to the owner a second time."""
    monkeypatch.setenv(learning.FLAG, "1")
    mislinked = "Correction: I prefer quiet rooms."
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", HANDOFF, 9), ("t2", mislinked, 10)],
                  {HANDOFF: [preference()], mislinked: [preference(mislinked, operation="correct")
                                                       | {"link": "handoff format"}]})
    projection = SourceClaimProjection(ledger)
    with closing(ledger._connect()) as conn:
        [held] = learning.pending(conn, contact_id=OWNER)
    assert projection.decide(held["id"], ratify=False, reason="both hold")["status"] == "declined"
    key = "handoff style" if drift else "handoff format"
    first, second = preference(), preference(mislinked, operation="correct")
    first["predicate"] = second["predicate"] = key
    await reextract_all(ledger, {HANDOFF: [first], mislinked: [second | {"link": key}]})
    with closing(ledger._connect()) as conn:
        assert learning.pending(conn, contact_id=OWNER) == []
        decided = learning.pending(conn, contact_id=OWNER, status="declined")
    assert decided and all(row["decided_by"] == "owner" for row in decided)
    assert {row["turn_id"] for row in projection.preferences(OWNER)} == {"t1", "t2"}


@pytest.mark.parametrize("drift", [False, True])
async def test_a_value_said_three_times_stays_one_claim_with_its_witnesses(tmp_path, monkeypatch, drift):
    """With the learning faculty on, a restatement is a witness of the live claim, not a row. Read again, the three
    messages still witness one claim: under its own id when the reading is the same, else under one new claim that
    carries the old one's correction."""
    monkeypatch.setenv(learning.FLAG, "1")
    texts = ["My office is in River.", "As I said, my office is in River.", "My office is still in River."]
    ledger = ledger_at(tmp_path)
    await history(ledger, [(f"t{n}", text, 9 + n) for n, text in enumerate(texts)] + [("t9", LAKE, 18)],
                  {**{text: [fact(text, "River")] for text in texts}, LAKE: OFFICE[LAKE]})
    [river] = [row for row in claims(ledger).values() if row["value"] == "River"]
    assert river["witnesses"] == 3 and river["retracted_by"]
    predicate = "workplace_city" if drift else "office_location"
    await reextract_all(ledger, {**{text: [fact(text, "River", predicate=predicate)] for text in texts},
                                 LAKE: [fact(LAKE, "Lake", predicate=predicate, operation="correct", link=predicate)]})
    [again] = [row for row in claims(ledger).values() if row["value"] == "River"]
    assert again["witnesses"] == 3 and again["retracted_by"] and live(ledger) == ["Lake"]
    assert (again["id"] == river["id"]) is not drift
    with closing(ledger._connect()) as conn:
        edges = [json.loads(row[0]) for row in conn.execute("SELECT data_json FROM source_claim_witnesses")]
    assert {edge["extraction_version"] for edge in edges} == {EXTRACTION_VERSION}
    assert {row["extraction_version"] for row in claims(ledger).values()} == {EXTRACTION_VERSION}


@pytest.mark.parametrize("dated", [False, True])
async def test_a_corrected_episode_keeps_its_report_and_its_correction(tmp_path, dated):
    """An episode is its exact report; a correction of it names the report as its subject. Read again, with or
    without the report's date, the report stays the retracted original and the correction the current revision."""
    from test_source_episode_memory import episode
    report = ("On 2026-02-24, the pressure sensor reset twice during a four-hour bench run, both times after the "
              "pump started. The cause was not established.")
    correction = "Correction: it reset only once, not twice. The first entry was copied twice."
    outputs = {report: [episode(report)],
               correction: [episode(correction) | {"operation": "correct", "link": "reported episode"}]}
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", report, 9), ("t2", correction, 10)], outputs)
    [first] = by_turn(ledger)["t1"]
    [second] = by_turn(ledger)["t2"]
    assert first["retracted_by"] == second["id"] and second["subject_basis_claim_id"] == first["id"]
    if dated:
        outputs = {**outputs, report: [episode(report) | {"event_at_text": "2026-02-24"}]}
    await reextract_all(ledger, outputs)
    [first_now] = by_turn(ledger)["t1"]
    [second_now] = by_turn(ledger)["t2"]
    assert (first_now["id"], second_now["id"]) == (first["id"], second["id"])
    assert first_now["retracted_by"] == second["id"] and second_now["subject_basis_claim_id"] == first["id"]
    assert live(ledger) == [correction] and reports(ledger) == []
    assert (first_now["event_time"]["status"] == "resolved") is dated


# -- 4. erased sources ---------------------------------------------------------------------------------------------


KIT = "The loaner kit needs four glove pairs and two nylon brushes."
KIT_FIX = ("Correction: the loaner kit now needs six glove pairs instead of four. "
           "The brush count stays the same.")


# -- 5. failure leaves the old claims -------------------------------------------------------------------------------

@pytest.mark.parametrize("failure", ["the model is down", "the model answers nonsense"])
async def test_old_claims_survive_a_failed_reextraction(tmp_path, failure):
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", RIVER, 9), ("t2", TEA, 10)], OFFICE)
    before = claims(ledger)
    worker = Reextraction(SourceClaimProjection(ledger))
    worker.start()
    model = Scripted(OFFICE, **({"fail": [RIVER]} if failure == "the model is down" else {"invalid": [RIVER]}))
    for attempt in range(1, reextract.MAX_ATTEMPTS + 1):
        assert await worker.process_one(model)
        assert claims(ledger) == before
        job = jobs(ledger)["t1"]
        assert job["attempts"] == attempt and job["error"]
        if attempt < reextract.MAX_ATTEMPTS:
            assert job["status"] == "pending"
            # The contact's later message waits for it: occurrence order holds.
            assert not await worker.process_one(model)
            with closing(ledger._connect()) as conn, conn:
                conn.execute("UPDATE source_reextract_jobs SET next_attempt=0")
    assert jobs(ledger)["t1"]["status"] == "failed"
    await drain(worker, model)
    assert jobs(ledger)["t2"]["status"] == "complete"
    assert claims(ledger)[next(i for i, row in before.items() if row["turn_id"] == "t1")] == before[
        next(i for i, row in before.items() if row["turn_id"] == "t1")]


async def test_a_crash_inside_the_commit_retires_nothing(tmp_path, monkeypatch):
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", RIVER, 9)], OFFICE)
    before = claims(ledger)
    worker = Reextraction(SourceClaimProjection(ledger))
    worker.start()

    def crash(*args, **kwargs):
        raise RuntimeError("the process died mid-commit")

    monkeypatch.setattr(SourceClaimProjection, "_commit", crash)
    assert await worker.process_one(Scripted({RIVER: [fact(RIVER, "River", predicate="workplace_city")]}))
    assert claims(ledger) == before and retired(ledger) == {}
    assert jobs(ledger)["t1"]["status"] == "pending"


async def test_a_reading_the_commit_rejects_leaves_the_old_claims(tmp_path):
    """The new reading takes its subject from the counted cupboard's claim, and the owner retracts that claim while
    the message is being read: the commit rejects the reading (its subject basis is no longer current). The message
    keeps its earlier claim, which stood on its own, and the report says why."""
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", COUNT, 9), ("t2", RECOUNT, 10)],
                  {COUNT: [fact(COUNT, "14", subject="loading cupboard", predicate="count")],
                   RECOUNT: [fact(RECOUNT, "17", subject="cupboard", predicate="count")]})
    [count] = by_turn(ledger)["t1"]
    assert live(ledger) == ["14", "17"]
    before = by_turn(ledger)["t2"]

    class Answered(Scripted):
        async def complete(self, messages, **kwargs):
            if json.loads(messages[-1]["content"]).get("message") == RECOUNT:
                with closing(ledger._connect()) as conn, conn:
                    conn.execute("UPDATE source_claims SET retracted_by='intention:ask-3' WHERE id=?", (count["id"],))
            return await super().complete(messages, **kwargs)

    model = Answered({COUNT: [fact(COUNT, "14", subject="loading cupboard", predicate="count")],
                      RECOUNT: [fact(RECOUNT, "17", subject="loading cupboard", predicate="count",
                                     operation="correct", link="count")]})
    await reextract_all(ledger, {}, model=model)
    assert by_turn(ledger)["t2"] == before and live(ledger) == ["17"]
    assert [(row["kind"], row["turn_id"]) for row in reports(ledger)] == [("commit_rejected", "t2")]
    # The message still holds a claim an older version formed: the source is not marked current (the next run
    # queues it again), and its job says it kept its earlier claims.
    assert jobs(ledger)["t2"]["status"] == "kept"
    with closing(ledger._connect()) as conn:
        assert dict(conn.execute("SELECT turn_id,extraction_version FROM source_claim_jobs")) == {
            "t1": EXTRACTION_VERSION, "t2": OLD}


async def test_a_kept_message_is_queued_again_and_only_it_is_read_again(tmp_path):
    """A source of two messages: the first's new reading commits, the second's is rejected (its basis was retracted
    while it was read), so the second keeps its earlier claim. The next run queues the source again (the plan
    counts that one message) and reads only the message that was kept; its reading now commits and the source is
    current."""
    ledger = ledger_at(tmp_path)
    count = {COUNT: [fact(COUNT, "14", subject="loading cupboard", predicate="count")]}
    await history(ledger, [("t1", COUNT, 9)], count)
    ledger.record_source("t2", contact_id=OWNER, session_id="session-t2", occurred_at=at(10), messages=[
        {"role": "user", "content": BIKE}, {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": RECOUNT}])
    first_reading = {**count, BIKE: [fact(BIKE, "the station", predicate="bike_location")],
                     RECOUNT: [fact(RECOUNT, "17", subject="cupboard", predicate="count")]}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(source_projection, "EXTRACTION_VERSION", OLD)
        assert await SourceClaimProjection(ledger).process_one(Scripted(first_reading))
    [counted] = by_turn(ledger)["t1"]

    class Answered(Scripted):
        async def complete(self, messages, **kwargs):
            if json.loads(messages[-1]["content"]).get("message") == RECOUNT:
                with closing(ledger._connect()) as conn, conn:
                    conn.execute("UPDATE source_claims SET retracted_by='intention:ask-3' WHERE id=?",
                                 (counted["id"],))
            return await super().complete(messages, **kwargs)

    await reextract_all(ledger, {}, model=Answered({
        **first_reading, RECOUNT: [fact(RECOUNT, "17", subject="loading cupboard", predicate="count",
                                        operation="correct", link="count")]}))
    assert jobs(ledger)["t2"]["status"] == "kept"
    assert {row["extraction_version"] for row in by_turn(ledger)["t2"]} == {EXTRACTION_VERSION, OLD}
    worker = Reextraction(SourceClaimProjection(ledger))
    plan = worker.plan()
    assert (plan["sources"], plan["messages"], plan["claims"]) == (1, 1, 1)
    again = await reextract_all(ledger, first_reading)
    assert again.started["queued"] == 1
    assert again.model.extracted() == [RECOUNT]
    assert jobs(ledger)["t2"]["status"] == "complete"
    assert {row["extraction_version"] for row in claims(ledger).values()} == {EXTRACTION_VERSION}
    with closing(ledger._connect()) as conn:
        assert {row[0] for row in conn.execute("SELECT extraction_version FROM source_claim_jobs")} == {
            EXTRACTION_VERSION}
    assert worker.plan()["sources"] == 0


# -- 6. resumable --------------------------------------------------------------------------------------------------

async def test_a_resumed_job_continues_where_it_stopped(tmp_path):
    ledger = ledger_at(tmp_path)
    ledger.record_source("t1", contact_id=OWNER, session_id="session-t1", occurred_at=at(9), messages=[
        {"role": "user", "content": RIVER}, {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": TEA}])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(source_projection, "EXTRACTION_VERSION", OLD)
        assert await SourceClaimProjection(ledger).process_one(Scripted(OFFICE))
    await history(ledger, [("t2", BIKE, 10)], {BIKE: [fact(BIKE, "the station", predicate="bike_location")]})
    outputs = {**OFFICE, BIKE: [fact(BIKE, "the station", predicate="bike_location")]}
    first = Reextraction(SourceClaimProjection(ledger))
    first.start()
    stopped = Scripted(outputs, fail=[TEA])
    assert await first.process_one(stopped)
    assert stopped.extracted() == [RIVER, TEA]
    job = jobs(ledger)["t1"]
    assert job["status"] == "pending" and len(json.loads(job["done_json"])) == 1
    with closing(ledger._connect()) as conn, conn:
        conn.execute("UPDATE source_reextract_jobs SET next_attempt=0")
    # A restarted sidecar: a new worker, a model that answers. The first message is not read again.
    resumed = Scripted(outputs)
    await drain(Reextraction(SourceClaimProjection(ledger)), resumed)
    assert resumed.extracted() == [TEA, BIKE]
    assert {row["status"] for row in jobs(ledger).values()} == {"complete"}
    assert {row["extraction_version"] for row in claims(ledger).values()} == {EXTRACTION_VERSION}


async def test_a_job_resumed_under_a_newer_version_reads_its_done_messages_again(tmp_path):
    """A run stopped after the first message of a source; the extractor was upgraded before it resumed. The message
    done under the version before is read again under the current one, so the source is never marked current
    with claims an older version formed."""
    ledger = ledger_at(tmp_path)
    ledger.record_source("t1", contact_id=OWNER, session_id="session-t1", occurred_at=at(9), messages=[
        {"role": "user", "content": RIVER}, {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": TEA}])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(source_projection, "EXTRACTION_VERSION", OLD)
        assert await SourceClaimProjection(ledger).process_one(Scripted(OFFICE))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(source_projection, "EXTRACTION_VERSION", "source-claims-between")
        worker = Reextraction(SourceClaimProjection(ledger))
        worker.start()
        stopped = Scripted(OFFICE, fail=[TEA])
        assert await worker.process_one(stopped)
        assert stopped.extracted() == [RIVER, TEA]
        assert len(json.loads(jobs(ledger)["t1"]["done_json"])) == 1
    with closing(ledger._connect()) as conn, conn:
        conn.execute("UPDATE source_reextract_jobs SET next_attempt=0")
    assert Reextraction(SourceClaimProjection(ledger)).status()["remaining_messages"] == 2
    resumed = Scripted(OFFICE)
    await drain(Reextraction(SourceClaimProjection(ledger)), resumed)
    assert resumed.extracted() == [RIVER, TEA]
    assert {row["extraction_version"] for row in claims(ledger).values()} == {EXTRACTION_VERSION}
    with closing(ledger._connect()) as conn:
        assert conn.execute("SELECT extraction_version FROM source_claim_jobs").fetchone()[0] == EXTRACTION_VERSION


async def test_a_queue_an_earlier_build_created_gains_the_checkpoint_version(tmp_path):
    """A ledger whose queue an earlier build created has no ``done_version`` column (a ``CREATE TABLE IF NOT EXISTS``
    never adds one): opening it adds the column. A checkpoint written there names no version, so it counts for
    nothing: the message it lists is read again under the current version, and the run completes."""
    ledger = ledger_at(tmp_path)
    ledger.record_source("t1", contact_id=OWNER, session_id="session-t1", occurred_at=at(9), messages=[
        {"role": "user", "content": RIVER}, {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": TEA}])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(source_projection, "EXTRACTION_VERSION", OLD)
        assert await SourceClaimProjection(ledger).process_one(Scripted(OFFICE))
    Reextraction(SourceClaimProjection(ledger)).start()
    from colonymind.turns.idempotency import source_message_hash
    with closing(ledger._connect()) as conn, conn:
        conn.execute("ALTER TABLE source_reextract_jobs DROP COLUMN done_version")
        conn.execute("UPDATE source_reextract_jobs SET done_json=?",
                     (json.dumps([source_message_hash("session-t1", {"role": "user", "content": RIVER})]),))
    reopened = ledger_at(tmp_path)
    with closing(reopened._connect()) as conn:
        assert "done_version" in {row[1] for row in conn.execute("PRAGMA table_info(source_reextract_jobs)")}
    assert Reextraction(SourceClaimProjection(reopened)).status()["remaining_messages"] == 2
    model = Scripted(OFFICE)
    await drain(Reextraction(SourceClaimProjection(reopened)), model)
    assert model.extracted() == [RIVER, TEA]
    assert jobs(reopened)["t1"]["status"] == "complete"
    assert {row["extraction_version"] for row in claims(reopened).values()} == {EXTRACTION_VERSION}


async def test_the_command_reads_and_starts_on_a_queue_an_earlier_build_created(tmp_path, monkeypatch, capsys):
    """``colonymind memory reextract`` opens the ledger itself, without the sidecar's start-up that adds the
    checkpoint version: on a queue an earlier build created, ``--status`` and ``--dry-run`` (read-only) read the
    missing version as none, and a start plans, then adds the column when it initializes the queue."""
    import argparse
    from colonymind.beliefs import reextract_cli
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", RIVER, 9), ("t2", TEA, 10)], OFFICE)
    Reextraction(SourceClaimProjection(ledger)).start(limit=1)
    with closing(ledger._connect()) as conn, conn:
        conn.execute("ALTER TABLE source_reextract_jobs DROP COLUMN done_version")

    def command(*argv):
        parser = argparse.ArgumentParser(prog="colonymind")
        reextract_cli.add_parser(parser.add_subparsers(dest="command"))
        code = reextract_cli.run(parser.parse_args(["memory", "reextract", *argv]))
        return code, capsys.readouterr()
    database = tmp_path / "turn-idempotency.db"
    raw = database.read_bytes()
    code, out = command("--status", "--json")
    assert code == 0, out.err
    assert json.loads(out.out)["remaining_messages"] == 1
    code, out = command("--all", "--dry-run", "--json")
    assert code == 0, out.err
    assert (json.loads(out.out)["sources"], json.loads(out.out)["queued"]) == (1, 1)
    assert database.read_bytes() == raw
    code, out = command("--all")
    assert code == 0, out.err
    assert "Queued 1 source" in out.out
    with closing(sqlite3.connect(database)) as conn:
        assert "done_version" in {row[1] for row in conn.execute("PRAGMA table_info(source_reextract_jobs)")}


async def test_a_job_whose_worker_died_is_taken_up_again(tmp_path):
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", RIVER, 9)], OFFICE)
    worker = Reextraction(SourceClaimProjection(ledger))
    worker.start()
    with closing(ledger._connect()) as conn, conn:
        conn.execute("UPDATE source_reextract_jobs SET status='running',lease_until=?,lease_token='gone'", (1.0,))
    model = Scripted(OFFICE)
    await drain(Reextraction(SourceClaimProjection(ledger)), model)
    assert model.extracted() == [RIVER] and jobs(ledger)["t1"]["status"] == "complete"


# -- 7. the plan and its estimate (the CLI's dry run is in test_source_claim_reextract_cli.py) -------------------

async def test_the_plan_counts_and_estimates_and_writes_nothing(tmp_path):
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", RIVER, 9), ("t2", TEA, 10), ("t3", LAKE, 11)], OFFICE)
    await history(ledger, [("s1", BIKE, 12)], {BIKE: [fact(BIKE, "the station", predicate="bike_location")]},
                  contact="sam")
    worker = Reextraction(SourceClaimProjection(ledger))
    plan = worker.plan()
    assert (plan["people"], plan["sources"], plan["messages"], plan["claims"]) == (2, 4, 4, 4)
    assert plan["from_versions"] == {OLD: 4} and plan["version"] == EXTRACTION_VERSION
    assert plan["rate"] is None and plan["estimate"]["seconds"] is None      # nothing measured yet
    assert plan["estimate"]["calls"] == 8                                     # one extraction and one review each
    assert worker.plan(person="sam")["sources"] == 1
    assert worker.plan(limit=2)["sources"] == 2
    assert jobs(ledger) == {}
    # A small run measures the rate; the plan of what is left uses it.
    worker.start(limit=1)
    await drain(worker, Scripted(OFFICE))
    plan = worker.plan()
    assert plan["sources"] == 3 and plan["rate"]["messages"] == 1
    assert plan["rate"]["tokens_per_message"] == 2000                       # extraction and review, 1000 each
    assert plan["estimate"]["tokens"] == 6000 and plan["estimate"]["seconds"] is not None


async def test_the_plan_describes_the_sources_a_start_would_queue(tmp_path):
    """With the first source already queued, a plan with a limit names the next one, which is what a start with
    the same limit queues; the queued one is counted apart."""
    ledger = ledger_at(tmp_path)
    ledger.record_source("t2", contact_id=OWNER, session_id="session-t2", occurred_at=at(10), messages=[
        {"role": "user", "content": TEA}, {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": BIKE}, {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": MOVED}])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(source_projection, "EXTRACTION_VERSION", OLD)
        assert await SourceClaimProjection(ledger).process_one(Scripted({}))
    await history(ledger, [("t1", RIVER, 9)], OFFICE)
    worker = Reextraction(SourceClaimProjection(ledger))
    assert worker.start(limit=1)["queued"] == 1 and set(jobs(ledger)) == {"t1"}
    plan = worker.plan(limit=1)
    assert (plan["sources"], plan["messages"], plan["queued"]) == (1, 3, 1)
    assert worker.start(limit=1)["queued"] == 1 and set(jobs(ledger)) == {"t1", "t2"}
    plan = worker.plan()
    assert (plan["sources"], plan["messages"], plan["queued"]) == (0, 0, 2)


# -- 8. imported history -------------------------------------------------------------------------------------------

async def test_imported_history_and_checkpoints_are_not_queued(tmp_path):
    ledger = ledger_at(tmp_path)
    ledger.record_source("imported", contact_id=OWNER, session_id="history", occurred_at=at(8),
                         messages=[{"role": "user", "content": RIVER}], derive_claims=False)
    ledger.record_source("checkpoint", contact_id=OWNER, session_id="session-c", scope="session",
                         messages=[{"role": "user", "content": TEA}])
    await history(ledger, [("t1", BIKE, 9)], {BIKE: [fact(BIKE, "the station", predicate="bike_location")]})
    worker = Reextraction(SourceClaimProjection(ledger))
    assert worker.plan()["sources"] == 1
    model = Scripted({BIKE: [fact(BIKE, "the station", predicate="bike_location")]})
    assert worker.start()["queued"] == 1
    await drain(worker, model)
    assert model.extracted() == [BIKE] and set(jobs(ledger)) == {"t1"}


# -- 9. occurrence order -------------------------------------------------------------------------------------------

async def test_with_the_queue_shuffled_a_correction_is_re_extracted_after_the_message_it_corrects(tmp_path):
    ledger = ledger_at(tmp_path)
    # Turn ids sort the other way round from when the messages were said.
    await history(ledger, [("z-river", RIVER, 9), ("m-tea", TEA, 10), ("a-lake", LAKE, 11)],
                  {**OFFICE, LAKE: [fact(LAKE, "Lake")]})
    assert live(ledger) == ["Lake", "River", "green tea"]            # the older version missed the correction
    worker = Reextraction(SourceClaimProjection(ledger))
    worker.start()
    with closing(ledger._connect()) as conn, conn:
        queued = [dict(row) for row in conn.execute("SELECT * FROM source_reextract_jobs ORDER BY rowid")]
        conn.execute("DELETE FROM source_reextract_jobs")
        for row in reversed(queued):
            conn.execute(f"INSERT INTO source_reextract_jobs({','.join(row)}) VALUES ({','.join('?' for _ in row)})",
                         list(row.values()))
        # And the ledger received them in the other order too.
        stamps = [row[0] for row in conn.execute("SELECT ingested_at FROM turn_sources ORDER BY ingested_at")]
        for turn, stamp in zip(["a-lake", "m-tea", "z-river"], stamps):
            conn.execute("UPDATE turn_sources SET ingested_at=? WHERE turn_id=?", (stamp, turn))
    model = Scripted({RIVER: [fact(RIVER, "River", predicate="workplace_city")], TEA: OFFICE[TEA],
                      LAKE: [fact(LAKE, "Lake", predicate="workplace_city", operation="correct",
                                  link="workplace_city")]})
    await drain(worker, model)
    assert model.extracted() == [RIVER, TEA, LAKE]
    # The correction saw the corrected message as the current version reads it, and retracted it.
    assert [row["predicate"] for row in model.payload(LAKE)["prior_assertions"]
            if row["value"] == "River"] == ["workplace city"]
    assert live(ledger) == ["Lake", "green tea"]


async def test_a_later_message_of_the_same_source_is_offered_the_earlier_ones_records(tmp_path):
    """One source holds River and, two messages later, its correction; the older version missed the correction.
    Read again, the correction is offered River as it was the first time (the source's earlier message, read
    again just before it), links it and retracts it. A later message's records are still never offered."""
    ledger = ledger_at(tmp_path)
    ledger.record_source("t1", contact_id=OWNER, session_id="session-t1", occurred_at=at(9), messages=[
        {"role": "user", "content": RIVER}, {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": LAKE}])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(source_projection, "EXTRACTION_VERSION", OLD)
        first = Scripted({**OFFICE, LAKE: [fact(LAKE, "Lake")]})
        assert await SourceClaimProjection(ledger).process_one(first)
    assert "River" in {row["value"] for row in first.payload(LAKE)["prior_assertions"]}
    assert live(ledger) == ["Lake", "River"]
    run = await reextract_all(ledger, OFFICE)
    assert "River" in {row["value"] for row in run.model.payload(LAKE)["prior_assertions"]}
    assert "Lake" not in {row["value"] for row in run.model.payload(RIVER)["prior_assertions"]}
    assert live(ledger) == ["Lake"]


async def test_an_earlier_message_is_never_offered_a_later_messages_records(tmp_path):
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", RIVER, 9), ("t2", MOVED, 10)],
                  {RIVER: [fact(RIVER, "River")], MOVED: [fact(MOVED, "Munich", predicate="home_city")]})
    run = await reextract_all(ledger, {RIVER: [fact(RIVER, "River")],
                                       MOVED: [fact(MOVED, "Munich", predicate="home_city")]})
    first = run.model.payload(RIVER)
    assert "Munich" not in {row["value"] for row in first["prior_assertions"]}
    assert "home city" not in {key["predicate"] for key in first.get("known_keys", [])}
    # Its own earlier reading's key is offered, so a property keeps its name.
    assert {"subject": "I", "predicate": "office location"} in first["known_keys"]
    assert "River" in {row["value"] for row in run.model.payload(MOVED)["prior_assertions"]}


# -- dependents, yielding and the lane -----------------------------------------------------------------------------

COUNT, RECOUNT = "The loading cupboard contains 14 units.", "Correction to my cupboard count: 17 units, not 14."


@pytest.mark.parametrize("drift", [False, True])
async def test_a_claim_grounded_in_an_older_claim_keeps_its_basis(tmp_path, drift):
    """The correction names no subject of its own: its subject comes from the counted cupboard's claim. When the
    count reads the same, the correction keeps its basis; when the count is filed under another key the basis
    cannot follow, so that message keeps its older claims and the report says why."""
    cupboard = {"subject": "loading cupboard", "predicate": "count"}
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", COUNT, 9), ("t2", RECOUNT, 10)],
                  {COUNT: [fact(COUNT, "14", **cupboard)],
                   RECOUNT: [fact(RECOUNT, "17", **cupboard, operation="correct", link="count")]})
    [count] = by_turn(ledger)["t1"]
    [recount] = by_turn(ledger)["t2"]
    assert recount["subject_basis_claim_id"] == count["id"]
    again = fact(COUNT, "14", **(cupboard | {"predicate": "unit_count"} if drift else cupboard))
    await reextract_all(ledger, {COUNT: [again],
                                 RECOUNT: [fact(RECOUNT, "17", **cupboard, operation="correct", link="count")]})
    assert live(ledger) == ["17"]
    rows = by_turn(ledger)
    assert rows["t1"][0]["id"] == count["id"]
    assert rows["t2"][0]["subject_basis_claim_id"] == count["id"]
    if drift:
        assert rows["t1"][0]["extraction_version"] == OLD
        assert [row["kind"] for row in reports(ledger)] == ["kept_for_dependents"]
    else:
        assert rows["t1"][0]["extraction_version"] == EXTRACTION_VERSION


async def test_it_yields_to_live_recall_and_to_a_new_turns_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(reextract, "YIELD_SECONDS", 0.05)
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", RIVER, 9)], OFFICE)
    worker = Reextraction(SourceClaimProjection(ledger))
    worker.start()
    model = Scripted(OFFICE)
    with live_recall.LIVE.running():
        assert not await worker.process_one(model)
    assert model.calls == []
    ledger.record_source("new", contact_id=OWNER, session_id="session-new", occurred_at=at(12),
                         messages=[{"role": "user", "content": TEA}])
    assert not await worker.process_one(model)                      # the new turn's extraction goes first
    assert await SourceClaimProjection(ledger).process_one(Scripted(OFFICE))
    assert await worker.process_one(model)
    assert model.extracted() == [RIVER]


async def test_it_does_not_commit_while_a_new_turn_of_the_person_is_being_extracted(tmp_path):
    """A new turn's extraction that began after this message was read may hold one of its claims as a prior: the
    message's commit waits for it (no attempt spent), so the new turn never links to a retired claim."""
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", RIVER, 9)], OFFICE)
    worker = Reextraction(SourceClaimProjection(ledger))
    worker.start()
    before = claims(ledger)
    ledger.record_source("new", contact_id=OWNER, session_id="session-new", occurred_at=at(12),
                         messages=[{"role": "user", "content": LAKE}])
    with closing(ledger._connect()) as conn, conn:
        conn.execute("UPDATE source_claim_jobs SET next_attempt=? WHERE turn_id='new'", (4e9,))

    class Arriving(Scripted):
        async def complete(self, messages, **kwargs):
            # The new turn's job is taken while this message is being read.
            with closing(ledger._connect()) as conn, conn:
                conn.execute("UPDATE source_claim_jobs SET status='running',lease_until=? WHERE turn_id='new'",
                             (4e9,))
            return await super().complete(messages, **kwargs)

    model = Arriving({RIVER: [fact(RIVER, "River", predicate="workplace_city")]})
    assert await worker.process_one(model)
    assert model.extracted() == [RIVER] and claims(ledger) == before
    job = jobs(ledger)["t1"]
    assert (job["status"], job["attempts"], job["error"]) == ("pending", 0, "live_extraction")


async def test_status_reports_progress_and_cancel_stops_what_waits(tmp_path):
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", RIVER, 9), ("t2", TEA, 10), ("t3", LAKE, 11)], OFFICE)
    worker = Reextraction(SourceClaimProjection(ledger))
    run = worker.start()
    assert await worker.process_one(Scripted(OFFICE))
    state = worker.status()
    [row] = state["runs"]
    assert row["id"] == run["run_id"] and row["status"] == "running"
    assert row["jobs"] == {"complete": 1, "pending": 2}
    assert state["rate"]["messages"] == 1 and state["remaining_messages"] == 2
    assert worker.cancel()["cancelled"] == 2
    assert not await worker.process_one(Scripted(OFFICE))
    [row] = worker.status()["runs"]
    assert row["status"] == "cancelled" and row["jobs"] == {"complete": 1, "cancelled": 2}
    assert {row["extraction_version"] for row in claims(ledger).values() if row["turn_id"] != "t1"} == {OLD}


async def test_the_source_worker_runs_it_on_its_own_lane(tmp_path, monkeypatch):
    import asyncio
    from colonymind.self_model.appraisals import AppraisalStore
    from colonymind.self_model.judgments import SelfJudgments
    from colonymind.commitments.extract import CommitmentExtractor
    from colonymind.turns import media

    async def no_work(*_args, **_kwargs):
        return False

    for kind in (SelfJudgments, AppraisalStore, media.SourceMedia, CommitmentExtractor):
        monkeypatch.setattr(kind, "process_one", no_work)
    monkeypatch.setattr(media.SourceMedia, "recover_unowned_files", lambda _self: None)
    ledger = ledger_at(tmp_path)
    await history(ledger, [("t1", RIVER, 9)], OFFICE)
    Reextraction(SourceClaimProjection(ledger)).start()
    model = Scripted(OFFICE)
    task = asyncio.create_task(source_projection.run_source_claim_worker(ledger, lambda: model, vectors=False))
    try:
        for _ in range(200):
            if jobs(ledger)["t1"]["status"] == "complete":
                break
            await asyncio.sleep(0.02)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert jobs(ledger)["t1"]["status"] == "complete" and model.extracted() == [RIVER]


from colonymind.memory import curated
async def test_erased_sources_are_skipped(tmp_path):
    ledger = ledger_at(tmp_path)
    await history(ledger, [('t1', RIVER, 9), ('t2', TEA, 10), ('t3', BIKE, 11)], {**OFFICE, BIKE: [fact(BIKE, 'the station', predicate='bike_location')]})
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, OWNER, source_ids=['t2'])
    assert historical_bytes(ledger) == _native_before_withdrawal
    worker = Reextraction(SourceClaimProjection(ledger))
    assert worker.plan()['sources'] == 2
    assert worker.start()['queued'] == 2
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, OWNER, source_ids=['t3'])
    assert historical_bytes(ledger) == _native_before_withdrawal
    model = Scripted({**OFFICE, BIKE: [fact(BIKE, 'the station', predicate='bike_location')]})
    await drain(worker, model)
    assert model.extracted() == [RIVER]
    assert set(by_turn(ledger)) == {'t1', 't2', 't3'}
    assert live(ledger) == ['River']

async def test_erasing_a_message_removes_its_retired_claims(tmp_path):
    ledger = ledger_at(tmp_path)
    await history(ledger, [('t1', RIVER, 9)], OFFICE)
    [river] = by_turn(ledger)['t1']
    await reextract_all(ledger, {RIVER: [fact(RIVER, 'River', predicate='workplace_city')]})
    assert river['id'] in retired(ledger)
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, OWNER, source_ids=['t1'])
    assert historical_bytes(ledger) == _native_before_withdrawal
    assert retired(ledger) and claims(ledger) and jobs(ledger)
    assert live(ledger) == []

@pytest.mark.parametrize('root', ['live', 'retired'])
async def test_erasing_a_message_removes_the_retired_claims_that_carried_its_words(tmp_path, root):
    """A partial correction carries its unchanged words ("two nylon brushes") from the claim it corrects. When a
    re-extraction retires the correction, the retired copy still quotes those words: erasing the message they came
    from removes it, whether that message's own claim is live or was itself retired by a later re-extraction."""
    kit = {'subject': 'loaner kit', 'predicate': 'needs'}
    first = fact(KIT, 'four glove pairs and two nylon brushes', **kit)
    ledger = ledger_at(tmp_path)
    await history(ledger, [('t1', KIT, 9), ('t2', KIT_FIX, 10)], {KIT: [first], KIT_FIX: [fact(KIT_FIX, 'six glove pairs and two nylon brushes', **kit, operation='correct', link='needs')]})
    [fix] = by_turn(ledger)['t2']
    assert fix['value_parts']
    await reextract_all(ledger, {KIT: [first], KIT_FIX: [fact(KIT_FIX, 'six glove pairs', **kit, operation='correct', link='needs')]})
    assert fix['id'] in retired(ledger)
    if root == 'retired':
        [kit_claim] = by_turn(ledger)['t1']
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(source_projection, 'EXTRACTION_VERSION', 'source-claims-later')
            await reextract_all(ledger, {KIT: [first | {'evidence': KIT[4:-1]}]}, limit=1)
        assert kit_claim['id'] in retired(ledger)
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, OWNER, source_ids=['t1'])
    assert historical_bytes(ledger) == _native_before_withdrawal
    with closing(ledger._connect()) as conn:
        kept = [row[0] for row in conn.execute('SELECT data_json FROM source_claim_reextractions')]
        live_rows = [row[0] for row in conn.execute('SELECT data_json FROM source_claims')]
    assert any(('nylon' in text for text in kept + live_rows))
    assert live(ledger) == ['six glove pairs']
