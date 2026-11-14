"""The learning rules on M9 lessons (``mind.faculties.learning``): a verified restatement is a witness on the
lesson it repeats (no second lesson, no supersession that resets its tally); the ambiguous band goes to one
labelling call; refused operations are logged; a lesson no verified use has won for ``COLD_DAYS`` is cold
(kept, listed, not served) until a verified restatement brings it back. With the faculty off, the M9 flow
is unchanged. The real ledger, store and mind run the night, as in ``test_mind_lessons_night``.
"""
from __future__ import annotations

from lesson_support import vectors_settled

from contextlib import closing

import pytest

from colonymind.memory import learning
from colonymind.mind import lessons as lessons_module
from test_mind_consolidate import OWNER
from test_mind_lessons_night import RULE_QUOTE, VERDICT, add, label, make, night, ruled, training_day
from test_mind_lessons import embedder  # noqa: F401  (pytest fixture: lesson selection is by meaning)

LEARNING = {"quiet_hours": "", "faculties": {"lessons": True, "consolidation": False, "learning": True}}
PARAPHRASE = "Channel letter, then the order's final two digits, then the contact's two digits."
VERDICT_2 = VERDICT.replace("train-01.json", "train-02.json")


def teach(content):
    """Each night's answer: the day's verdict, and one lesson citing it with ``content``."""
    def answer(prompt):
        needle = "Verdict on train-02.json" if "train-02" in prompt else "Verdict on train-01.json"
        verdicts = [{"turn": label(prompt, needle), "work_was": "wrong", "quote": RULE_QUOTE}]
        return {"verdicts": verdicts, "ops": [add([label(prompt, needle)], quote=RULE_QUOTE, content=content.pop(0))]}
    return answer


def log_rows(fx, **kwargs):
    with closing(fx.ledger._connect()) as conn:
        return learning.entries(conn, **kwargs)


def revise(fields):
    """Night one teaches the rule; night two supersedes the current lesson with ``fields`` changed."""
    holder = []

    def answer(prompt):
        needle = "Verdict on train-02.json" if "train-02" in prompt else "Verdict on train-01.json"
        op = add([label(prompt, needle)], quote=RULE_QUOTE)
        if "train-02" in prompt:
            [current] = holder[0].mind.lessons.all()
            op.update(op="supersede", lesson_id=current.id, **fields)
        return {"verdicts": [{"turn": label(prompt, needle), "work_was": "wrong", "quote": RULE_QUOTE}], "ops": [op]}
    return answer, holder


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [
    "Do not use the channel letter, then the order's last two digits, then the contact's two digits.",
    "Channel letter, then the order's last three digits, then the contact's two digits."])
async def test_a_negated_or_renumbered_supersession_replaces_the_lesson(tmp_path, monkeypatch, content):
    """Lexically the same words (the scorer drops "not" and numbers alike): still the owner's correction."""
    answer, holder = revise({"content": content})
    fx = make(tmp_path, monkeypatch, answer, config=LEARNING)
    holder.append(fx)
    training_day(fx)
    await night(fx)
    [first] = fx.mind.lessons.all()
    training_day(fx, 2, verdict=VERDICT_2)
    result = await night(fx)
    assert "lessons_witnessed" not in result["counts"] and result["counts"].get("lessons_superseded") == 1
    [current] = fx.mind.lessons.all()
    assert current.content == content and current.supersedes == first.id
    fx.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("on", [False, True])
async def test_a_supersession_that_narrows_the_situation_replaces_the_lesson(tmp_path, monkeypatch, on):
    """The same advice for a narrower situation is another lesson: the owner's narrowing replaces the
    broader one (it is neither a witness nor the unchanged lesson under the same id)."""
    narrowed = "an order code is asked for a retail order"
    answer, holder = revise({"when_to_use": narrowed})
    config = LEARNING if on else {"quiet_hours": "",
                                  "faculties": {"lessons": True, "consolidation": False, "learning": False}}
    fx = make(tmp_path, monkeypatch, answer, config=config)
    holder.append(fx)
    training_day(fx)
    await night(fx)
    [first] = fx.mind.lessons.all()
    training_day(fx, 2, verdict=VERDICT_2)
    result = await night(fx)
    assert "lessons_witnessed" not in result["counts"] and result["counts"].get("lessons_superseded") == 1
    [current] = fx.mind.lessons.all()
    assert current.when_to_use == narrowed and current.content == first.content and current.supersedes == first.id
    assert fx.mind.lessons.get(first.id).status == "superseded"
    fx.store.close()


def revise_from(first, second, *, op="supersede"):
    """Night one teaches a lesson with ``first``; night two ``op``s it with ``second`` changed."""
    holder = []

    def answer(prompt):
        needle = "Verdict on train-02.json" if "train-02" in prompt else "Verdict on train-01.json"
        row = add([label(prompt, needle)], quote=RULE_QUOTE, **first)
        if "train-02" in prompt:
            row.update(second)
            if op == "supersede":
                [current] = holder[0].mind.lessons.all()
                row.update(op="supersede", lesson_id=current.id)
        return {"verdicts": [{"turn": label(prompt, needle), "work_was": "wrong", "quote": RULE_QUOTE}], "ops": [row]}
    return answer, holder


# The same words, another instruction: swapped recipients, swapped quantities, a situation narrowed from
# "or" to "and". Every similarity score reads each pair as one lesson.
REFORMED = [
    ({"content": "Send the weekly report to finance, not sales."},
     {"content": "Send the weekly report to sales, not finance."}),
    ({"content": "Print 3 copies for the office and 5 copies for the warehouse."},
     {"content": "Print 5 copies for the office and 3 copies for the warehouse."}),
    ({"when_to_use": "handling reports marked public or approved"},
     {"when_to_use": "handling reports marked public and approved"})]


@pytest.mark.asyncio
@pytest.mark.parametrize("first,second", REFORMED)
@pytest.mark.parametrize("op,model_says", [("supersede", None), ("supersede", True), ("add", None)])
async def test_the_same_words_in_another_form_replace_the_lesson(tmp_path, monkeypatch, first, second, op,
                                                                 model_says):
    """Neither a witness nor the unchanged lesson: the new form replaces the old. An explicit supersession
    is not folded into its own target even by a labeller that calls the two the same."""
    answer, holder = revise_from(first, second, op=op)
    fx = make(tmp_path, monkeypatch, answer, config=LEARNING)
    holder.append(fx)
    if model_says is not None:
        fx.router.answers[learning.LABEL_TASK] = lambda messages, context: {"pairs": [
            {"id": pair["id"], "same": model_says} for pair in __import__("json").loads(messages[-1]["content"])["pairs"]]}

    async def alike(text, others):
        return [0.99 for _ in others]
    fx.mind.lessons.scorer = alike
    training_day(fx)
    await night(fx)
    [first_lesson] = fx.mind.lessons.all()
    training_day(fx, 2, verdict=VERDICT_2)
    result = await night(fx)
    assert "lessons_witnessed" not in result["counts"] and result["counts"].get("lessons_superseded") == 1
    [current] = fx.mind.lessons.all()
    for field, value in second.items():
        assert getattr(current, field) == value
    assert current.supersedes == first_lesson.id and fx.mind.lessons.get(first_lesson.id).status == "superseded"
    fx.store.close()


def test_the_reflector_gate_does_not_fold_another_situation(tmp_path, monkeypatch):
    fx = make(tmp_path, monkeypatch, config=LEARNING)
    lessons = fx.mind.lessons
    fields = {"signature": "research:lookup", "kind": "pitfall", "title": "Lookups time out",
              "when_to_use": "a lookup service is slow", "content": "Do not retry a timed-out lookup more than once."}
    lessons.admit(fields, verified="none", origin="reflector", status="candidate", evidence=["intention:1"], lineage=[])
    for changed in ({"when_to_use": "a lookup service is slow at night"}, {"signature": "research:archive"}):
        assert lessons._gate_sync({**fields, **changed}).verdict == "novel"
    fx.store.close()


@pytest.mark.asyncio
async def test_the_ambiguous_band_takes_one_labelling_call(tmp_path, monkeypatch):
    contents = [add([])["content"], "Use the channel letter and the order digits for any code."]
    fx = make(tmp_path, monkeypatch, teach(contents), config=LEARNING)
    labels = []

    def label_task(messages, context):
        import json
        pairs = json.loads(messages[-1]["content"])["pairs"]
        labels.append(pairs)
        return {"pairs": [{"id": pair["id"], "same": True} for pair in pairs]}
    fx.router.answers[learning.LABEL_TASK] = label_task

    async def middling(text, others):
        return [0.85 for _ in others]           # inside the embedding bands' ambiguous middle
    fx.mind.lessons.scorer = middling
    training_day(fx)
    await night(fx)
    training_day(fx, 2, verdict=VERDICT_2)
    result = await night(fx)
    assert len(labels) == 1 and len(labels[0]) == 1
    assert result["counts"].get("lessons_witnessed") == 1 and len(fx.mind.lessons.all()) == 1
    assert [row["decision"]["step"] for row in log_rows(fx, action="labelled")] == ["model"]
    fx.store.close()


@pytest.mark.asyncio
async def test_refused_lesson_operations_are_logged_with_ids_only(tmp_path, monkeypatch):
    def answer(prompt):
        verdict = label(prompt, "Verdict on train-01.json")
        return {"verdicts": ruled(prompt), "ops": [add([verdict]), add(["t9"], quote=RULE_QUOTE)]}
    fx = make(tmp_path, monkeypatch, answer)          # the log is mandatory: the faculty is off here
    training_day(fx)
    await night(fx)
    rows = log_rows(fx, store="lesson", action="rejected")
    assert {row["reason"] for row in rows} == {"an owner citation without the owner's exact words",
                                              "cites something outside the packet"}
    assert ["turn:day-01-verdict"] in [row["refs"] for row in rows]
    assert not any("channel" in str(row).lower() for row in rows)
    fx.store.close()


@pytest.mark.asyncio
async def test_an_unwon_lesson_cools_is_not_served_and_a_verified_restatement_revives_it(tmp_path, monkeypatch,
                                                                                          embedder):
    from datetime import timedelta
    contents = [add([])["content"], add([])["content"]]
    fx = make(tmp_path, monkeypatch, teach(contents), config=LEARNING)
    training_day(fx)
    await night(fx)
    [lesson] = fx.mind.lessons.all()
    query = "an order code is asked for: what is the order code channel letter"
    await vectors_settled(fx.mind.lessons)
    assert (await fx.mind.lessons.turn_lesson(query, session_id="s", record=False))[1] == [lesson.id]
    fx.shift(days=lessons_module.COLD_DAYS + 1)
    changed = fx.mind.lessons.review()
    assert changed["cooled"] == [lesson.id]
    [cold] = fx.mind.lessons.all()
    assert cold.tier == "cold" and cold.status == "active"                       # kept and listed
    assert await fx.mind.lessons.turn_lesson(query, session_id="s", record=False) == ("", [])
    fx.mind.lessons.learning = False                                                # the flag off: served again
    await fx.mind.lessons.fill_vectors()                                            # its vector made again
    assert (await fx.mind.lessons.turn_lesson(query, session_id="s", record=False))[1] == [lesson.id]
    fx.mind.lessons.learning = True
    training_day(fx, 2, verdict=VERDICT_2)                                          # the owner says it again
    await night(fx)
    [warm] = fx.mind.lessons.all()
    assert warm.id == lesson.id and warm.tier == "warm" and warm.witnesses == 2
    assert [row["item_id"] for row in log_rows(fx, action="revived")] == [lesson.id]
    fx.shift(days=1)
    assert fx.mind.lessons.review()["cooled"] == []
    fx.store.close()


@pytest.mark.asyncio
async def test_a_reflector_repeat_is_a_witness_that_does_not_revive(tmp_path, monkeypatch):
    fx = make(tmp_path, monkeypatch, config=LEARNING)
    lessons = fx.mind.lessons
    fields = {"signature": "research:lookup", "kind": "pitfall", "title": "Lookups time out",
              "when_to_use": "a lookup service is slow", "content": "Do not retry a timed-out lookup more than once."}
    first = lessons.admit(fields, verified="none", origin="reflector", status="candidate", evidence=["intention:1"],
                          lineage=[])
    decision = lessons._gate_sync({**fields, "content": "Do not retry a timed out lookup more than once!"})
    assert decision.verdict == "repeat" and decision.match == first.id
    assert lessons._gate_sync({**fields, "content": "Ask the owner which supplier to use."}).verdict == "novel"
    fx.store.close()


def _claim(fx, ident, value, representation=None):
    import json
    data = {"value": value, "evidence": value, **({"representation": representation} if representation else {})}
    with closing(fx.ledger._connect()) as conn, conn:
        conn.execute("INSERT INTO source_claims(id,turn_id,message_hash,subject_key,predicate,value_key,data_json) "
                     "VALUES (?,?,?,?,?,?,?)", (ident, "t", "h", "s", "p", value.casefold(), json.dumps(data)))


@pytest.mark.asyncio
@pytest.mark.parametrize("reply,next_message,expected", [
    ("The loading cupboard count is 14.", "Right, that is correct, thanks.", "useful"),
    ("I am not sure what the count is.", "Right, that is correct, thanks.", "unused"),
    ("The loading cupboard count is 14.", "Wrong: it is 17.", "wrong"),
    ("The loading cupboard count is 14.", "Also book the van for Friday.", "unjudged")])
async def test_served_claims_are_reinforced_only_by_useful_recall(tmp_path, monkeypatch, reply, next_message, expected):
    """The night scores each served claim by the owner's next message: used in the reply and judged right
    reinforces it; unused, wrong or no verdict closes the use without reinforcing anything."""
    from types import SimpleNamespace
    from colonymind.api.routers.host import _learning_served
    monkeypatch.setenv(learning.FLAG, "1")
    fx = make(tmp_path, monkeypatch, config=LEARNING)
    _claim(fx, "claim:count", "14")
    question = "What is the loading cupboard count?"
    body = SimpleNamespace(context=SimpleNamespace(session_id="sess", metadata={}))
    _learning_served(fx.ledger, question, body, [{"history_anchor": {"claim_id": "claim:count"}}])
    fx.turn("s-1", OWNER, "sess", question, reply)
    fx.shift(minutes=2)
    fx.turn("s-2", OWNER, "sess", next_message, "Noted.")
    lessons = fx.mind.lessons
    packet = lessons._packet(fx.now)
    verdict_label = label(packet["text"], next_message[:20])
    judged = {} if expected == "unjudged" else {verdict_label: "wrong" if expected == "wrong" else "right"}
    night_counts = SimpleNamespace(count=lambda *args, **kwargs: None)
    lessons._score_recall(judged, packet, night_counts, fx.now)
    with closing(fx.ledger._connect()) as conn:
        [use] = [dict(row) for row in conn.execute("SELECT * FROM learning_uses")]
        items = [dict(row) for row in conn.execute("SELECT * FROM learning_items")]
    assert use["result"] == expected
    assert bool(items and items[0]["useful"] == 1) is (expected == "useful")
    fx.store.close()


@pytest.mark.asyncio
async def test_a_repeated_question_is_a_new_use_and_a_retried_turn_scores_once(tmp_path, monkeypatch):
    """Asked again in the same session, the same question serving the same claim is another use: a first
    ``wrong`` does not keep the later, right one from reinforcing the claim. Two records of one turn (a
    retried context call) map to the same owner message and are scored once."""
    from types import SimpleNamespace
    monkeypatch.setenv(learning.FLAG, "1")
    fx = make(tmp_path, monkeypatch, config=LEARNING)
    _claim(fx, "claim:count", "14")
    question, lessons = "What is the loading cupboard count?", fx.mind.lessons
    counts = SimpleNamespace(count=lambda *args, **kwargs: None)
    key = lessons_module.served_key(question)

    def serve(occurrence):
        with closing(fx.ledger._connect()) as conn, conn:
            return learning.record_served(conn, session_id="sess", served=key, store="claim", item_ids=["claim:count"],
                                          occurrence=occurrence, now=fx.now.timestamp())

    def score(*verdicts):
        packet = lessons._packet(fx.now)
        judged = {label(packet["text"], text[:20]): verdict for text, verdict in verdicts}
        lessons._score_recall(judged, packet, counts, fx.now)

    assert serve("turn-1") == 1
    fx.turn("s-1", OWNER, "sess", question, "I am not sure what the count is.")
    fx.shift(minutes=2)
    fx.turn("s-2", OWNER, "sess", "Wrong: it is written on the door.", "Noted.")
    score(("Wrong: it is written on the door.", "wrong"))
    fx.shift(minutes=30)
    assert serve("turn-2") == 1                                     # the same question again: a new use
    assert serve("") == 1                                            # the same turn recorded twice, no turn id
    fx.turn("s-3", OWNER, "sess", question, "The loading cupboard count is 14.")
    fx.shift(minutes=2)
    fx.turn("s-4", OWNER, "sess", "Right, that is correct, thanks.", "Noted.")
    score(("Wrong: it is written on the door.", "wrong"), ("Right, that is correct, thanks.", "right"))
    with closing(fx.ledger._connect()) as conn:
        results = sorted(row["result"] for row in conn.execute("SELECT result FROM learning_uses"))
        [item] = [dict(row) for row in conn.execute("SELECT * FROM learning_items")]
    assert results == ["duplicate", "useful", "wrong"] and item["useful"] == 1
    fx.store.close()


def test_the_host_names_each_use_by_its_turn(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from colonymind.api.routers.host import _learning_served
    from colonymind.turns import TurnIdempotencyLedger
    ledger = TurnIdempotencyLedger(tmp_path / "sources.db")
    monkeypatch.setenv(learning.FLAG, "1")
    selected = [{"history_anchor": {"claim_id": "claim:a"}}]

    def serve(turn_id=None):
        context = SimpleNamespace(session_id="s", metadata={}, **({"turn_id": turn_id} if turn_id else {}))
        _learning_served(ledger, "q", SimpleNamespace(context=context), selected)
        with closing(ledger._connect()) as conn:
            return conn.execute("SELECT count(*) FROM learning_uses").fetchone()[0]
    assert (serve("t1"), serve("t1"), serve("t2"), serve(), serve()) == (1, 1, 2, 3, 4)


def test_the_use_log_skips_recipient_packets_worker_runs_and_the_faculty_off(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from colonymind.api.routers.host import _learning_served
    from colonymind.turns import TurnIdempotencyLedger
    ledger = TurnIdempotencyLedger(tmp_path / "sources.db")
    selected = [{"history_anchor": {"claim_id": "claim:a"}}]

    def served():
        with closing(ledger._connect()) as conn:
            return conn.execute("SELECT count(*) FROM learning_uses").fetchone()[0]
    monkeypatch.setenv(learning.FLAG, "0")
    _learning_served(ledger, "q", SimpleNamespace(context=SimpleNamespace(session_id="s", metadata={})), selected)
    monkeypatch.setenv(learning.FLAG, "1")
    _learning_served(ledger, "q", SimpleNamespace(context=SimpleNamespace(session_id="mind:p-1", metadata={})), selected)
    _learning_served(ledger, "q", SimpleNamespace(context=SimpleNamespace(session_id="s", metadata={"kanban_task": "t"})),
                     selected)
    assert served() == 0
    _learning_served(ledger, "q", SimpleNamespace(context=SimpleNamespace(session_id="s", metadata={})), selected)
    assert served() == 1


# -- the digest's cold tier --------------------------------------------------------------------------

def _digest_fixture(tmp_path, monkeypatch):
    from test_mind_consolidate import Fixture
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    return Fixture(tmp_path, config={"faculties": {"learning": True}})


@pytest.mark.asyncio
async def test_a_digest_whose_every_claim_went_cold_is_cleared(tmp_path, monkeypatch):
    from test_mind_consolidate import CONTACT
    fx = _digest_fixture(tmp_path, monkeypatch)
    office = await fx.fact("turn-a", CONTACT, "telegram-1", "My office is room 4.", "room 4")
    await fx.mind.consolidate()
    record = fx.contacts.records[CONTACT]
    assert record.digest_sources == [office] and "room 4" in record.digest
    fx.shift(days=learning.COLD_AFTER_DAYS + 5)
    fx.interaction(CONTACT)                                   # still a recent contact
    await fx.mind.consolidate()
    assert not record.digest and record.digest_sources == []
    fx.store.close()


@pytest.mark.asyncio
async def test_a_rebuilt_digest_does_not_carry_a_cold_claim_over_from_the_previous_one(tmp_path, monkeypatch):
    from test_mind_consolidate import CONTACT
    fx = _digest_fixture(tmp_path, monkeypatch)
    office = await fx.fact("turn-a", CONTACT, "telegram-1", "My office is room 4.", "room 4")
    desk = await fx.fact("turn-b", CONTACT, "telegram-1", "My desk is desk 9.", "desk 9", predicate="desk_location")
    await fx.mind.consolidate()
    record = fx.contacts.records[CONTACT]
    assert sorted(record.digest_sources) == sorted([office, desk]) and "room 4" in record.digest
    fx.shift(days=learning.COLD_AFTER_DAYS + 5)
    with closing(fx.ledger._connect()) as conn, conn:        # the desk was recalled, used and judged right
        learning.score_use(conn, {"session_id": "s", "served": "k", "store": "claim", "item_id": desk}, "useful",
                           now=fx.now.timestamp())
    fx.interaction(CONTACT)
    await fx.mind.consolidate()
    assert record.digest_sources == [desk] and "room 4" not in record.digest
    fx.store.close()


from colonymind.memory import curated
@pytest.mark.asyncio
@pytest.mark.parametrize('on', [False, True])
async def test_a_verified_paraphrase_is_a_witness_not_a_supersession(tmp_path, monkeypatch, on):
    contents = [add([])['content'], PARAPHRASE]
    config = LEARNING if on else {'quiet_hours': '', 'faculties': {'lessons': True, 'consolidation': False, 'learning': False}}
    fx = make(tmp_path, monkeypatch, teach(contents), config=config)
    training_day(fx)
    await night(fx)
    [first] = fx.mind.lessons.all()
    training_day(fx, 2, verdict=VERDICT_2)
    result = await night(fx)
    current = fx.mind.lessons.all()
    if not on:
        assert result['counts'].get('lessons_superseded') == 1
        assert [lesson.id for lesson in current] != [first.id]
        fx.store.close()
        return
    assert result['counts'].get('lessons_witnessed') == 1 and 'lessons_superseded' not in result['counts']
    [lesson] = current
    assert lesson.id == first.id and lesson.witnesses == 2 and (lesson.status == 'active')
    assert lesson.evidence == ['turn:day-01-verdict', 'turn:day-02-verdict']
    [row] = log_rows(fx, store='lesson', action='witnessed')
    assert row['item_id'] == first.id and row['decision']['step'] in {'hash', 'similarity'}
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(fx.ledger)
    curated.forget_sources(fx.ledger, OWNER, source_ids=['day-02-verdict'])
    assert historical_bytes(fx.ledger) == _native_before_withdrawal
    [lesson] = fx.mind.lessons.all()
    assert lesson.witnesses == 1 and lesson.evidence == ['turn:day-01-verdict']
    fx.store.close()
