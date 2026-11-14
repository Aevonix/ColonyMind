"""Whose answer teaches the mind (docs/MIND.md, Asks): the verdicts an ask leaves.

An ask's verdict (what the ranker's feedback and the agent's own feelings learn from) comes only from the owner's
own answer in their own turn: a reaction, a quoted reply, their next message, a list, or the ask's code typed in
their own words (the earlier chat path). An answer given anywhere else (the CLI, a credential, a script at the
console) settles the ask as asked, and teaches nothing. A lapse (silence until the ask expires) is no verdict
either. The upgrade clears once the verdicts earlier releases took from an operator's answer, takes their
feedback back, and the feelings recompute without them; an open ask's last notice from before this release is
recorded as a list binding, so a quoted reply to it still binds whatever the notice format becomes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest

from colonymind.initiatives.store import InitiativeStore
from colonymind.mind import asks_report
from colonymind.mind import tick as tick_module
from colonymind.mind.affect import CAP, DISMISSAL, HALF_LIVES, Affect
from colonymind.mind.concerns import MindState, open_mind_db
from test_mind_loop import OWNER, Fixture, pinned_clock  # noqa: F401  (the pinned wall clock)
from owner_answer import owner_answers

PLATFORM = "telegram"
CHAT = "owner-chat"
TYPE_KEY = "research:curiosity"


@pytest.fixture
def fx(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fixture = Fixture(tmp_path)
    yield fixture
    fixture.store.close()


def asked(fx, code, *, kind="task"):
    row, _ = fx.store.create_intention(
        kind=kind, type="research", title=f"Re-run the archive export ({code})", drive="curiosity", cls="self",
        decision="ask", decision_reason="suggest: self -> ask", status="asked", dedup_key=f"test:{code}",
        ask_code=code, context={"notice": True}, hermes_kind="none", created_at=fx.now,
        expires_at=fx.now + timedelta(days=3))
    return row


def deliver(fx, row, *, message_id):
    """The ask's own message, delivered to the owner's chat now, as the body reports it."""
    notice = fx.mind.outbox.notice(type="ask_show", title="ask", text="the ask in words", dedup_key=f"t:{message_id}",
                                   extra={"asks": [row.id], "list": [row.id]})
    fx.mind.outbox.sending(notice.id, target=f"{PLATFORM}:{CHAT}")
    return fx.mind.outbox.sent(notice.id, result="sent", hermes_ref=message_id)


def evidence(fx, text="", *, message_id, ledger=(), **extra):
    return {"platform": PLATFORM, "contact_id": OWNER, "chat_id": CHAT, "chat_type": "dm", "turn_id": f"t-{message_id}",
            "message_id": message_id, "arrived_at": fx.now.isoformat(), "text": text,
            "ledger": None if ledger is None else list(ledger), **extra}


def history(fx, row, action):
    return [item for item in fx.store.get_history(row.id, limit=100) if item.action == action]


def dismissed_level(fx) -> float:
    return float((fx.mind.mind_state.get("affect.dismissed") or {}).get("level") or 0.0)


def settled_before(fx, row, *, by, yes=False):
    """An ask as an earlier release left it once answered: the settling history row naming who answered (no record
    of the path yet), the verdict taken from it, and its feedback."""
    if yes:
        fx.store.transition(row.id, "approved", action="queued", at=fx.now, verdict="actioned",
                            details={"by": by, "code": row.ask_code})
        fx.feedback.record(f"{row.type}:{row.drive}", "actioned", source=row.id)
        return
    fx.store.transition(row.id, "cancelled", action="outcome_denied", at=fx.now,
                        details={"by": by, "verified": "none", "check": None}, outcome="denied", verified="none",
                        verdict="dismissed", cancelled_at=fx.now, cancelled_by=by, cancelled_reason="denied",
                        result=f"the owner said no ({by})")
    fx.feedback.record(f"{row.type}:{row.drive}", "dismissed", source=row.id)


# -- live: only the owner's own answer teaches ----------------------------------------------------------------------

@pytest.mark.parametrize("by", ["cli", "api", "operator", "owner", "owner-turn:forged", "owner-turn:telegram:m-1"])
async def test_an_answer_at_the_console_settles_the_ask_and_teaches_nothing(fx, by):
    """The CLI, a credential or a script answering at the console (whatever ``by`` it names: the route's default is
    "owner", and a caller may name the label an answer read from the owner's own message carries) settles the ask as
    it was answered, and leaves no verdict: no feedback, no dismissal felt. The console path the answer took decides,
    never the label."""
    row = asked(fx, "K7F")
    await fx.mind.answer("K7F", yes=False, by=by)
    settled = fx.store.get(row.id)
    assert (settled.status, settled.outcome) == ("cancelled", "denied")
    settled_row, = history(fx, row, "outcome_denied")
    assert settled_row.details["answer"]["path"] == "console"
    assert settled.verdict is None
    assert fx.feedback.multiplier(TYPE_KEY) == 1.0
    fx.mind.feelings.update(fx.now)
    assert dismissed_level(fx) == 0.0 and fx.mind.feelings.view().dismissals == 0


async def test_the_owners_own_no_teaches(fx):
    row = asked(fx, "K7F")
    own = deliver(fx, row, message_id="m-1")
    fx.shift(minutes=2)
    result = await fx.mind.read_answer(evidence(fx, "no thanks", message_id="in-1",
                                                ledger=[{"at": own.completed_at.isoformat(), "outbox_id": own.id}]))
    assert [item["id"] for item in result["settled"]] == [row.id]
    assert fx.store.get(row.id).verdict == "dismissed"
    assert fx.feedback.multiplier(TYPE_KEY) == pytest.approx(0.85)
    fx.mind.feelings.update(fx.now)
    assert fx.mind.feelings.view().dismissals == 1 and dismissed_level(fx) == pytest.approx(DISMISSAL, abs=0.01)
    # The earlier chat path (the ask's code in the owner's own words, read by the sidecar) is their own answer too.
    typed = asked(fx, "Q2Z")
    await owner_answers(fx.mind, "Q2Z", "no")
    assert fx.store.get(typed.id).verdict == "dismissed"


async def test_a_lapse_is_no_verdict(fx):
    """Silence until an ask expires is no answer: nothing is done, and nothing is learned or felt. An answer that
    arrives after the lapse approves nothing and teaches nothing either."""
    silent, late = asked(fx, "K7F"), asked(fx, "Q2Z")
    fx.shift(days=3, minutes=1)
    answered = await owner_answers(fx.mind, "Q2Z")
    assert answered.outcome == "expired"
    await fx.mind.tick(force=True)
    for row in (silent, late):
        lapsed = fx.store.get(row.id)
        assert (lapsed.status, lapsed.outcome, lapsed.verdict) == ("expired", "expired", None)
    assert fx.feedback.multiplier(TYPE_KEY) == 1.0
    fx.mind.feelings.update(fx.now)
    assert dismissed_level(fx) == 0.0 and fx.mind.feelings.view().dismissals == 0


# -- the feelings: a dismissal taken back ---------------------------------------------------------------------------

class Feelings:
    """The agent's own feelings over a real intention store and ``mind_state``, on a clock of its own."""

    def __init__(self, tmp_path):
        self.now = datetime(2030, 1, 7, 12, 0, tzinfo=timezone.utc)
        self.state = MindState(open_mind_db(tmp_path / "mind.db"), clock=lambda: self.now)
        self.store = InitiativeStore(state_dir=tmp_path)
        self.affect = Affect(self.state, store=self.store, owner_id=OWNER, clock=lambda: self.now)

    def declined(self, n: int) -> str:
        row, _ = self.store.create_intention(
            kind="task", type="research", title=f"an ask {n}", drive="curiosity", cls="self", decision="ask",
            decision_reason="x", status="asked", dedup_key=f"d-{n}", ask_code=f"K{n}F", created_at=self.now)
        self.store.transition(row.id, "cancelled", action="outcome_denied", at=self.now, outcome="denied",
                              verdict="dismissed", cancelled_at=self.now)
        return row.id

    def level(self) -> float:
        return float((self.state.get("affect.dismissed") or {}).get("level") or 0.0)

    def causes(self) -> list:
        return list((self.state.get("affect.dismissed") or {}).get("causes") or [])


def test_a_dismissal_whose_verdict_is_withdrawn_leaves_the_feeling(tmp_path):
    """The level is recomputed from the dismissals still standing, each as if applied at its own time: four at once
    reach the cap, and taking one back leaves the three that still reach it; taking them all back leaves nothing.
    A verdict set again is felt again."""
    world = Feelings(tmp_path)
    ids = [world.declined(n) for n in range(4)]
    world.affect.update(world.now)
    assert world.level() == pytest.approx(CAP)
    world.now += timedelta(hours=6)
    decay = 0.5 ** (6 * 3600 / HALF_LIVES["dismissed"])
    world.store.update(ids[0], verdict=None)
    world.affect.update(world.now)
    assert world.level() == pytest.approx(CAP * decay, abs=1e-3)
    assert not any(ids[0] in cause for cause in world.causes())
    for ident in ids[1:]:
        world.store.update(ident, verdict=None)
    world.affect.update(world.now)
    assert world.level() == 0.0 and world.causes() == [] and world.affect.view().dismissals == 0
    world.store.update(ids[0], verdict="dismissed")
    world.affect.update(world.now)
    assert world.level() == pytest.approx(DISMISSAL * decay, abs=1e-3)


def test_an_unreadable_store_takes_nothing_back(tmp_path, monkeypatch):
    """A failed read is not a withdrawn verdict: while the intentions cannot be read, the feeling stays."""
    world = Feelings(tmp_path)
    ident = world.declined(1)
    world.affect.update(world.now)
    before = world.level()
    world.store.update(ident, verdict=None)

    def unreadable(*args, **kwargs):
        raise OSError("the store is locked")
    monkeypatch.setattr(world.store, "intentions", unreadable)
    monkeypatch.setattr(world.store, "get", unreadable)
    world.affect.update(world.now)
    assert world.level() == pytest.approx(before)


@pytest.mark.parametrize("fails", ["delete", "set"])
def test_a_recompute_that_fails_is_done_again_at_the_next_update(tmp_path, monkeypatch, fails):
    """The update that sees a withdrawn dismissal fails while it recomputes the level (the state store refuses the
    write): the withdrawal is still found at the next update, which recomputes the level without it. The feeling
    never outlives its verdict because one update failed."""
    world = Feelings(tmp_path)
    kept, withdrawn = world.declined(1), world.declined(2)
    world.affect.update(world.now)
    assert world.level() == pytest.approx(2 * DISMISSAL)
    world.store.update(withdrawn, verdict=None)
    real = getattr(world.state, fails)

    def refused(key, *args, **kwargs):
        if key == "affect.dismissed":
            raise OSError("the state store refused the write")
        return real(key, *args, **kwargs)
    monkeypatch.setattr(world.state, fails, refused)
    assert world.affect.update(world.now).get("error") == "OSError"
    monkeypatch.setattr(world.state, fails, real)
    world.affect.update(world.now)
    assert world.level() == pytest.approx(DISMISSAL, abs=1e-3)
    assert not any(withdrawn in cause for cause in world.causes()) and any(kept in c for c in world.causes())
    world.store.update(kept, verdict=None)
    world.affect.update(world.now)
    assert world.level() == 0.0 and world.affect.view().dismissals == 0


# -- the upgrade --------------------------------------------------------------------------------------------------

def test_the_upgrade_clears_the_verdicts_the_console_gave_and_the_feelings_recompute(fx, caplog):
    """Three asks an operator declined at the console before this release: each kept a dismissal the owner never
    gave, lowering the ranker's weight for their kind and holding optional nudges back. The upgrade clears exactly
    those verdicts (their outcome stays), takes their feedback back, logs the ids it touched, and the next update of
    the feelings recomputes to nothing. A second run touches nothing."""
    rows = [asked(fx, code) for code in ("K7F", "Q2Z", "M3P")]
    for row in rows:
        settled_before(fx, row, by="cli")
    fx.mind.feelings.update(fx.now)
    assert dismissed_level(fx) == pytest.approx(CAP) and fx.mind.feelings.view().satiated
    assert fx.feedback.multiplier(TYPE_KEY) == pytest.approx(0.85 ** 3)
    assert asks_report.report(fx.store, since=fx.now - timedelta(days=1), now=fx.now)["operator_verdicts"] == 3
    fx.shift(hours=12)
    with caplog.at_level(logging.INFO):
        fx.restart()
    for row in rows:
        cleared = fx.store.get(row.id)
        assert (cleared.status, cleared.outcome, cleared.verdict) == ("cancelled", "denied", None)
        step, = history(fx, row, "verdict_cleared")
        assert step.details["by"] == "upgrade" and step.details["verdict"] == "dismissed"
        assert row.id in caplog.text
    assert fx.feedback.multiplier(TYPE_KEY) == 1.0
    fx.mind.feelings.update(fx.now)
    assert dismissed_level(fx) == 0.0 and not fx.mind.feelings.view().satiated
    report = asks_report.report(fx.store, since=fx.now - timedelta(days=1), now=fx.now)
    assert report["settled"]["operator"] == 3 and report["operator_verdicts"] == 0
    fx.restart()
    assert all(len(history(fx, row, "verdict_cleared")) == 1 for row in rows)


def test_the_upgrade_touches_only_answers_the_owner_did_not_give(fx):
    console_no, console_yes, api_no = asked(fx, "K7F"), asked(fx, "Q2Z"), asked(fx, "M3P")
    settled_before(fx, console_no, by="cli")
    settled_before(fx, console_yes, by="cli", yes=True)
    settled_before(fx, api_no, by="api")
    owners_no = asked(fx, "B4N")
    settled_before(fx, owners_no, by="owner")                 # the earlier chat path: the owner's own words
    owners_yes = asked(fx, "H6T")
    settled_before(fx, owners_yes, by="owner", yes=True)
    # The operator later cancelled the work the owner approved: the owner's approval still stands as theirs.
    fx.store.transition(owners_yes.id, "cancelled", action="outcome_cancelled", at=fx.now,
                        details={"by": "cli", "verified": "none", "check": None}, outcome="cancelled")
    lapsed = asked(fx, "W8R")
    fx.store.transition(lapsed.id, "expired", action="outcome_expired", at=fx.now,
                        details={"by": "mind", "verified": "none", "check": None}, outcome="expired",
                        verdict="ignored", cancelled_at=fx.now, cancelled_reason="expired")
    fx.restart()
    for row in (console_no, console_yes, api_no):
        assert fx.store.get(row.id).verdict is None and len(history(fx, row, "verdict_cleared")) == 1
    for row, verdict in ((owners_no, "dismissed"), (owners_yes, "actioned"), (lapsed, "ignored")):
        assert fx.store.get(row.id).verdict == verdict and history(fx, row, "verdict_cleared") == []
    assert fx.feedback.multiplier(TYPE_KEY) == pytest.approx(0.85 * 1.12)


def test_the_upgrade_keeps_a_rating_the_owner_gave_after_an_operator_settled_the_ask(fx):
    """An operator declined an ask, and the owner rated it afterwards: the verdict now standing is the owner's
    rating, not the operator's answer, so the upgrade keeps it and its feedback, and changes nothing on the row."""
    row = asked(fx, "K7F")
    settled_before(fx, row, by="cli")
    fx.shift(minutes=5)
    fx.mind.outcomes.rate(row.id, "not_useful", by="owner")
    fx.restart()
    assert fx.store.get(row.id).verdict == "not_useful"
    assert history(fx, row, "verdict_cleared") == []
    assert fx.feedback.multiplier(TYPE_KEY) == pytest.approx(0.85)


def test_a_rating_the_owner_gives_after_the_upgrade_survives_every_later_start(fx):
    """The upgrade cleared an operator's verdict once; the owner then rated the ask. That rating is theirs: the
    next starts keep it and its feedback, and record no second clearing."""
    row = asked(fx, "K7F")
    settled_before(fx, row, by="cli")
    fx.restart()
    assert fx.store.get(row.id).verdict is None and fx.feedback.multiplier(TYPE_KEY) == 1.0
    fx.shift(hours=1)
    fx.mind.outcomes.rate(row.id, "useful", by="owner")
    for _ in range(2):
        fx.restart()
        assert fx.store.get(row.id).verdict == "useful"
        assert fx.feedback.multiplier(TYPE_KEY) == pytest.approx(1.12)
    assert len(history(fx, row, "verdict_cleared")) == 1


def test_a_withdrawal_that_fails_is_tried_again_at_the_next_start(fx, monkeypatch, caplog):
    """The feedback store refuses the withdrawal (locked): the verdict is not cleared while its feedback still
    counts, the row stays for the next start, and that start neutralizes the feedback and clears the verdict."""
    row = asked(fx, "K7F")
    settled_before(fx, row, by="cli")
    record = fx.feedback.record

    def locked(itype, outcome, *, source):
        if outcome == "withdrawn":
            raise OSError("database is locked")
        return record(itype, outcome, source=source)
    monkeypatch.setattr(fx.feedback, "record", locked)
    with caplog.at_level(logging.WARNING):
        fx.restart()
    assert fx.store.get(row.id).verdict == "dismissed" and history(fx, row, "verdict_cleared") == []
    assert fx.feedback.multiplier(TYPE_KEY) == pytest.approx(0.85)
    assert row.id in caplog.text
    monkeypatch.setattr(fx.feedback, "record", record)
    fx.restart()
    assert fx.store.get(row.id).verdict is None and len(history(fx, row, "verdict_cleared")) == 1
    assert fx.feedback.multiplier(TYPE_KEY) == 1.0


def test_without_a_feedback_store_the_upgrade_waits_for_one(fx):
    """A start whose feedback store did not open cannot take a verdict's feedback back: the verdict stays until a
    start that has the store clears both."""
    row = asked(fx, "K7F")
    settled_before(fx, row, by="cli")
    feedback, fx.feedback = fx.feedback, None
    fx.restart()
    assert fx.store.get(row.id).verdict == "dismissed" and history(fx, row, "verdict_cleared") == []
    fx.feedback = feedback
    fx.restart()
    assert fx.store.get(row.id).verdict is None and fx.feedback.multiplier(TYPE_KEY) == 1.0


async def test_a_quoted_reply_to_an_earlier_notice_binds_as_a_list_whatever_the_notice_format_becomes(fx, monkeypatch):
    """An ask still open at the upgrade was listed, with its code, in a notice of the earlier format. The upgrade
    records that notice's delivery as a list binding on each ask it lists, so a quoted reply to it binds as a list
    even once the notice text is no longer read that way. Recorded once."""
    first, second = asked(fx, "K7F"), asked(fx, "Q2Z")
    text = "\n".join([f"{asks_report.NEEDS_OK_1_11}:", f"- K7F: {first.description}", f"- Q2Z: {second.description}",
                      "Without an answer they expire and nothing is done."])
    notice = fx.mind.outbox.notice(type="ask_notice", title="2 open ask(s)", text=text, dedup_key="ask_notice:t")
    fx.mind.outbox.sending(notice.id, target=f"{PLATFORM}:{CHAT}")
    fx.mind.outbox.sent(notice.id, result="sent", hermes_ref="m-1")
    fx.shift(minutes=5)
    monkeypatch.setattr(tick_module, "NEEDS_OK_LABEL", "Waiting for your OK", raising=False)
    fx.restart()
    for row in (first, second):
        bound, = history(fx, row, "ask_shown")
        assert bound.details["message_id"] == "m-1" and bound.details["list"] == [first.id, second.id]
        assert (bound.details["platform"], bound.details["chat"], bound.details["by"]) == (PLATFORM, CHAT, "upgrade")
    fx.shift(minutes=1)
    result = await fx.mind.read_answer(evidence(fx, "both", message_id="in-1", reply_to_message_id="m-1",
                                                ledger=None))
    assert result["path"] == "list" and {item["id"] for item in result["settled"]} == {first.id, second.id}
    fx.restart()
    assert all(len(history(fx, row, "ask_shown")) == 1 for row in (first, second))
