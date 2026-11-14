"""New reference loaders read the caller's state without putting its words in the decision log."""

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import timedelta

import pytest

from colonymind.instinct import REGISTRY, outcomes
from colonymind.instinct.log import Store
from colonymind.instinct.tiers import Instinct
from test_mind_consolidate import CONTACT, Fixture, OWNER


POINTS = ("interest_settled", "mind_wish_scope", "ask_answer", "claim_relation", "percept_attend")
ROW_KEYS = {"ref", "point", "version", "state", "state_hash", "label", "teacher", "push_away",
            "source_kind", "audience", "at"}


@dataclass
class Case:
    fx: object
    engine: object
    row: dict
    turn: str
    words: tuple
    change: object

    @property
    def sources(self):
        return outcomes.Sources(self.fx.state, owner_id=OWNER)

    def render(self, row=None):
        return outcomes.rerender(self.row if row is None else row, self.sources, self.engine.store.secret())

    def rows(self):
        return list(outcomes.replay_rows(self.engine.store, self.sources, point=self.row["point"]))


def rewrite(fx, turn, transform):
    with closing(fx.ledger._connect()) as conn, conn:
        messages = json.loads(conn.execute("SELECT messages_json FROM turn_sources WHERE turn_id=?",
                                          (turn,)).fetchone()[0])
        transform(messages)
        conn.execute("UPDATE turn_sources SET messages_json=? WHERE turn_id=?", (json.dumps(messages), turn))


@pytest.fixture(params=POINTS)
async def case(request, tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    point = request.param
    fx = Fixture(tmp_path, config={"quiet_hours": "", "faculties": {"outreach": False,
                                                                  "contradictions_by_meaning": False}})
    engine = Instinct(store=Store(tmp_path / "instinct.db"), clock=lambda: fx.now)
    fx.mind.instinct = engine
    turn = "t-answer"

    if point == "interest_settled":
        from colonymind.commitments.extract import CommitmentExtractor
        from colonymind.commitments.store import CommitmentStore
        topic, words = "tidal energy", "The tidal energy question is answered now."
        fx.mind.add_interest(topic)
        fx.turn(turn, OWNER, "owner-session", words, "Understood.")
        commitments = CommitmentStore(tmp_path / outcomes.COMMITMENTS_DB)
        extractor = CommitmentExtractor(fx.ledger, lambda: commitments, interests=lambda: fx.mind)
        extractor.instinct = engine
        await extractor._decided_settlements([topic], [], person_id=OWNER, text=words, turn_id=turn)
        change = lambda: fx.mind.mind_state.set("interest:tidal-energy", text="tidal systems")
        markers = (topic, words)
    elif point == "mind_wish_scope":
        from colonymind.mind.consolidate import Night
        from colonymind.mind.wishes import Capability, Kind, effects_for
        capability = Capability(Kind.DATA_SOURCE, "research:tide-tables", "tide tables")
        wish = {"capability": capability, "words": {"name": "archive parser", "why": "fetches fail"},
                "effects": effects_for(needs_code=False, skills=False), "needs_code": False,
                "failures": [], "lessons": [], "entry": {"failures": []}, "words_check": "within"}
        night = Night(local_date=fx.now.date().isoformat(), started_at=fx.now.isoformat(), note_id="note-1")
        fx.mind.wishes.instinct = engine

        async def scope(*args, **kwargs):
            return {"scope": "within"}

        await fx.mind.wishes.scope(night, wish, call=scope)
        fx.mind.wishes.raise_wish(wish, night, fx.now)
        concern = fx.mind.concerns.open()[0]

        def change():
            with fx.mind.mind_state.conn as conn:
                detail = json.loads(conn.execute("SELECT detail_json FROM concerns WHERE id=?",
                                                (concern.id,)).fetchone()[0])
                detail["extra"]["wish"]["words"]["name"] = "different parser"
                conn.execute("UPDATE concerns SET detail_json=? WHERE id=?", (json.dumps(detail), concern.id))

        turn, markers = "", ("archive parser", "fetches fail", capability.name)
    elif point == "ask_answer":
        from test_ask_answers_binding import asked, deliver, evidence, sent
        from test_instinct_ask_answer import live, reader
        engine = live(fx, reader("unrelated"), store=engine.store)
        ask = asked(fx)
        notice = deliver(fx, ask, message_id="out-1")
        fx.shift(seconds=30)
        words = "yes, maybe"
        await fx.mind.read_answer(evidence(fx, words, message_id="in-1", ledger=[sent(notice)], turn_id=turn))
        fx.turn(turn, OWNER, "owner-session", words, "Understood.")
        change = lambda: rewrite(fx, turn, lambda messages: messages[0].update(content="no, maybe"))
        markers = (words, ask.description)
    elif point == "claim_relation":
        from test_instinct_points import pair
        fx.mind.consolidation.instinct = engine
        await pair(fx)
        await fx.mind.consolidate()
        turn = "turn-a"

        def change():
            with closing(fx.ledger._connect()) as conn, conn:
                data = json.loads(conn.execute("SELECT data_json FROM source_claims WHERE turn_id=?",
                                               (turn,)).fetchone()[0])
                data["evidence"] = "My office is room 9."
                conn.execute("UPDATE source_claims SET data_json=? WHERE turn_id=?", (json.dumps(data), turn))

        markers = ("My office is room 4.", "My office is room 7.", "your office location")
    else:
        from test_mind_observations_sources import clip, observation, post
        image, digest = clip()
        item = observation(fx, kind="percept", image=image, frame_hash=digest, device_principal="device:unit")
        await post(fx, items=[item])
        [decision] = engine.store.rows(point=point)
        turn = decision["refs"]["observation"]

        def change():
            rewrite(fx, turn, lambda messages: messages[0]["content"][0].update(summary="Something else changed"))

        markers = (item["summary"], item["subject"], item["feature"])

    await engine.drain()
    [row] = engine.store.rows(point=point)
    outcomes.label(engine.store, row["id"], next(iter(REGISTRY[point].labels)), now=fx.now)
    value = Case(fx, engine, row, turn, markers, change)
    yield value
    fx.store.close()


def test_fresh_caller_decision_matches_and_replays_exact_row_keys(case):
    assert case.render() == "matched"
    [row] = case.rows()
    assert set(row) == ROW_KEYS
    assert row["ref"] == case.row["id"] and row["audience"] == "owner"
    assert row["teacher"] == "owner_label" and row["source_kind"] == "explicit"
    assert row["state_hash"] == case.row["state_hash"]
    assert not row["push_away"]


def test_changed_record_cannot_teach(case):
    case.change()
    assert case.render() == "changed"
    assert case.rows() == []


def test_old_rows_keep_no_loader_and_explain_the_missing_references(case):
    row = {**case.row, "refs": {"turn": case.turn} if case.turn else {}}
    assert case.render(row) == "no_loader"
    assert outcomes.no_loader_reason(row["point"], row=row) == "before_references"
    assert outcomes.NO_LOADER_REASONS["before_references"] == "logged before its references"
    case.engine.store.write(row)
    assert case.rows() == []
    assert outcomes.replay_share(case.engine.store, case.sources)[row["point"]] == {"no_loader": 1}


def test_each_required_reference_distinguishes_old_rows_from_missing_records(case):
    required = {
        "interest_settled": ("turn", "interest"),
        "mind_wish_scope": ("wish",),
        "ask_answer": ("ask", "message", "name"),
        "claim_relation": ("claims", "turns", "messages", "about", "days", "times", "topic_kind"),
        "percept_attend": ("observation",),
    }
    assert outcomes.no_loader_reason(case.row["point"], row=case.row) is None
    for key in required[case.row["point"]]:
        row = {**case.row, "refs": {name: value for name, value in case.row["refs"].items() if name != key}}
        assert case.render(row) == "no_loader", key
        assert outcomes.no_loader_reason(row["point"], row=row) == "before_references"


def test_decision_log_contains_only_references_and_keyed_hashes(case):
    logged = json.dumps(case.engine.store.rows(), ensure_ascii=False)
    for words in case.words:
        assert words not in logged
    if case.row["point"] == "interest_settled":
        assert case.row["refs"]["interest"].startswith("h:")
    elif case.row["point"] == "mind_wish_scope":
        assert case.row["refs"]["wish"]
    elif case.row["point"] == "ask_answer":
        assert case.row["refs"]["message"] and case.row["refs"]["name"]
    elif case.row["point"] == "claim_relation":
        assert len(case.row["refs"]["turns"]) == 2


def test_external_and_owner_gate_loaders_are_still_unavailable():
    assert outcomes.NO_LOADER["owner_send"] == "redacted"
    for point in ("cron_supplier_reply", "inbound_triage", "tool_screen"):
        assert point not in outcomes.LOADERS and outcomes.no_loader_reason(point) == "external"


@pytest.mark.parametrize("case", ["ask_answer"], indirect=True)
def test_answer_without_any_capture_link_is_missing(case):
    row = {**case.row, "refs": {key: value for key, value in case.row["refs"].items() if key != "turn"}}
    with case.fx.mind.answer_turns.conn as conn:
        conn.execute("UPDATE answer_turns SET turn_id=NULL")
    assert case.render(row) == "missing"


@pytest.mark.parametrize("case", ["interest_settled", "ask_answer", "percept_attend"], indirect=True)
@pytest.mark.parametrize("field, value", [("contact_id", "guest-1"), ("scope", "session")])
def test_owner_audience_is_checked_against_the_source_record(case, field, value):
    with closing(case.fx.ledger._connect()) as conn, conn:
        conn.execute(f"UPDATE turn_sources SET {field}=? WHERE turn_id=?", (value, case.turn))
    assert case.render() == "missing"
    assert case.rows() == []


@pytest.mark.parametrize("case", ["percept_attend"], indirect=True)
def test_expired_percept_replays_from_retained_event_metadata(case):
    from colonymind.mind.observations import PREFIX
    case.fx.shift(days=2)
    with case.fx.mind.mind_state.conn as conn:
        for row in conn.execute("SELECT key,text FROM mind_state WHERE key LIKE ?", (PREFIX + "%",)).fetchall():
            data = json.loads(row["text"])
            data["observations"] = []
            conn.execute("UPDATE mind_state SET text=? WHERE key=?", (json.dumps(data), row["key"]))
    assert case.render() == "matched"


@pytest.mark.parametrize("case", ["claim_relation"], indirect=True)
def test_changed_claim_timestamp_cannot_recover_the_old_local_day(case):
    with closing(case.fx.ledger._connect()) as conn, conn:
        conn.execute("UPDATE turn_sources SET occurred_at=? WHERE turn_id=?",
                     ((case.fx.now + timedelta(days=1)).isoformat(), "turn-b"))
    assert case.render() == "changed"
    assert case.rows() == []


@pytest.mark.parametrize("case", ["mind_wish_scope"], indirect=True)
async def test_wish_record_replays_after_the_tick_forms_its_ask(case):
    await case.fx.mind.tick(force=True)
    [ask] = case.fx.mind.wishes.rows()
    assert case.render() == "matched"
    with closing(case.engine.store._connect()) as conn:
        before = list(conn.iterdump())
    assert case.rows()
    with closing(case.engine.store._connect()) as conn:
        assert list(conn.iterdump()) == before
    with closing(case.fx.store._connect()) as conn, conn:
        conn.execute("DELETE FROM initiatives WHERE id=?", (ask.id,))
    assert case.render() == "missing"


async def test_named_message_ask_keeps_its_sentence_after_settlement(tmp_path, monkeypatch):
    from test_ask_answers_binding import asked, deliver, evidence, sent
    from test_instinct_ask_answer import live, reader

    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fx = Fixture(tmp_path)
    try:
        fx.contacts.records[CONTACT].display_name = "Sam"
        with closing(sqlite3.connect(tmp_path / "contacts.db")) as conn, conn:
            conn.execute("CREATE TABLE contacts (contact_id TEXT, display_name TEXT, given_name TEXT, deleted_at TEXT)")
            conn.execute("INSERT INTO contacts VALUES (?, 'Sam', NULL, NULL)", (CONTACT,))
        engine = live(fx, reader("decline"))
        ask = asked(fx, kind="message", type_="reply", context={"text": "The archive is ready."})
        fx.store.update(ask.id, entity_id=CONTACT)
        ask = fx.store.get(ask.id)
        notice = deliver(fx, ask, message_id="out-1")
        fx.shift(seconds=30)
        await fx.mind.read_answer(evidence(fx, "yes, maybe", message_id="in-1", ledger=[sent(notice)]))
        fx.turn("t-in-1", OWNER, "owner-session", "yes, maybe", "Understood.")
        await engine.drain()
        [row] = engine.store.rows(point="ask_answer")
        assert fx.store.get(ask.id).status == "cancelled"
        sources = outcomes.Sources(tmp_path, owner_id=OWNER)
        assert outcomes.rerender(row, sources, engine.store.secret()) == "matched"
        assert "Sam" not in json.dumps(engine.store.rows())
        with closing(sqlite3.connect(tmp_path / "contacts.db")) as conn, conn:
            conn.execute("UPDATE contacts SET display_name='Robin' WHERE contact_id=?", (CONTACT,))
        assert outcomes.rerender(row, sources, engine.store.secret()) == "changed"
    finally:
        fx.store.close()


@pytest.mark.parametrize("action", ["annotate", "withhold"])
async def test_claim_loader_requires_the_sources_that_ground_an_inherited_subject(tmp_path, action):
    from test_source_claim_earlier_subjects import remove, sister

    ledger, _ = await sister(tmp_path)
    with closing(ledger._connect()) as conn:
        claim = dict(conn.execute("SELECT * FROM source_claims WHERE turn_id='lives'").fetchone())
    reading = outcomes._Reading(outcomes.Sources(tmp_path, owner_id="owner"), b"fixture-key")

    def load():
        return outcomes._load_claim_witness(reading, claim["id"], "lives", claim["message_hash"])

    assert load() is not None
    if action == "annotate":
        remove(ledger, action)
    else:
        with closing(ledger._connect()) as conn, conn:
            conn.execute("INSERT INTO source_projection_erasures VALUES ('moved', 'withheld-source')")
    assert load() is None


@pytest.mark.parametrize("hidden", ['withheld'])
def test_unavailable_source_is_missing_and_cannot_teach(case, hidden):
    if not case.turn:
        # Wishes reference their ask record, not a captured turn.
        with case.fx.mind.mind_state.conn as conn:
            conn.execute("DELETE FROM concerns")
    else:
        with closing(case.fx.ledger._connect()) as conn, conn:
            conn.execute('INSERT INTO source_projection_erasures (turn_id, source_turn_id) VALUES (?, ?)',
                         (case.turn, 'withheld-source'))
    assert case.render() == "missing"
    assert case.rows() == []

from colonymind.memory import curated

@pytest.mark.parametrize("case", ["ask_answer"], indirect=True)
def test_answer_message_key_locates_a_turn_captured_after_the_decision(case):
    row = {**case.row, "refs": {key: value for key, value in case.row["refs"].items() if key != "turn"}}
    assert case.render(row) == "matched"
    with closing(case.fx.ledger._connect()) as conn, conn:
        conn.execute("DELETE FROM turn_sources WHERE turn_id=?", (case.turn,))
    assert case.render(row) == "missing"
    case.fx.turn(case.turn, OWNER, "owner-session", "yes, maybe", "Understood.")
    assert case.render(row) == "matched"
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(case.fx.ledger)
    _source_forget_receipt = curated.forget_sources(case.fx.ledger, OWNER, source_ids=[case.turn])
    assert historical_bytes(case.fx.ledger) == _native_before_forget
    assert all(curated.is_forgotten(case.fx.ledger, sid) for sid in _source_forget_receipt['source_ids'])
    assert case.render(row) == "missing"


@pytest.mark.parametrize("case", ["claim_relation"], indirect=True)
async def test_claim_restatement_does_not_move_the_decisions_pinned_witness(case, monkeypatch):
    from colonymind.memory import learning
    monkeypatch.setenv(learning.FLAG, "1")
    await case.fx.fact("turn-c", OWNER, "owner-session", "My office is room 7.", "room 7",
                       at=case.fx.now + timedelta(days=1))
    with closing(case.fx.ledger._connect()) as conn:
        anchor = conn.execute("SELECT turn_id FROM source_claims WHERE id=?",
                              (case.row["refs"]["claims"][1],)).fetchone()[0]
    assert anchor == "turn-c"
    assert case.render() == "matched"
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(case.fx.ledger)
    _source_forget_receipt = curated.forget_sources(case.fx.ledger, OWNER, source_ids=['turn-b'])
    assert historical_bytes(case.fx.ledger) == _native_before_forget
    assert all(curated.is_forgotten(case.fx.ledger, sid) for sid in _source_forget_receipt['source_ids'])
    assert case.render() == "missing"
    assert case.rows() == []
