"""The asks report (``colonymind.mind.asks_report``): how the owner's asks went over a window.

It is the instrument the approvals redesign is measured with, before and after: asks formed by type and class,
when each first reached the owner, who settled it (the owner, an operator, a lapse), how (the answer path and the
reader, where the settling row records them; 1.11 rows have neither), how fast, what an operator's answer taught
the mind, and the work held waiting. Served to the owner's view as ``GET /v1/mind/asks/stats`` and as
``colonymind mind asks-report``.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta

import httpx
import pytest
from fastapi import FastAPI

from onekey import KEY
from colonymind.api.credentials import CredentialTable, secret_digest
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import mind as mind_router
from colonymind.mind import asks_report
from colonymind.mind import cli as mind_cli
from colonymind.mind import digest as digest_text
from test_mind_loop import OWNER, Fixture, pinned_clock  # noqa: F401  (the pinned wall clock)
from owner_answer import owner_answers

HOUR = 3600.0


@pytest.fixture
def fx(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fixture = Fixture(tmp_path)
    yield fixture
    mind_router.set_mind(None)
    fixture.store.close()


def asked(fx, code, *, type_="research", kind="task", cls="self", context=None, at=None):
    row, _ = fx.store.create_intention(
        kind=kind, type=type_, title=f"The private plan {code}", drive="curiosity", cls=cls, decision="ask",
        decision_reason="suggest: self -> ask", status="asked", dedup_key=None, ask_code=code,
        context={"notice": True, **(context or {})}, hermes_kind="none", created_at=at or fx.now,
        expires_at=(at or fx.now) + timedelta(hours=72))
    return row


def text_1_11(rows):
    """The part of a 1.11 ask notice or digest that listed open asks: each line starts with the ask's code."""
    return "\n".join([f"{asks_report.NEEDS_OK_1_11}:", *(f"- {row.ask_code}: {row.description}." for row in rows)])


def notice(fx, rows):
    """A 1.11 ask notice listing ``rows`` by code, sent now."""
    sent = fx.mind.outbox.notice(type="ask_notice", title=f"{len(rows)} open ask(s)", dedup_key=f"notice:{fx.now}",
                                 text=text_1_11(rows) + "\nWithout an answer they expire and nothing is done.")
    fx.mind.outbox.sent(sent.id, result="sent")
    return sent


def report(fx, **kwargs):
    return asks_report.report(fx.store, since=fx.now - timedelta(days=7), now=fx.now, **kwargs)


def entry(value, row):
    return next(item for item in value["asks"] if item["id"] == row.id)


# -- who settled an ask -----------------------------------------------------------------------------------------------

def test_a_console_answer_is_the_operators_whatever_it_calls_itself():
    """Who settled an ask is read from the path the answering code recorded, as ``outcomes.owner_answer`` reads
    it: a ``by`` is the caller's label and can take the owner's attribution away, never give it. A console answer
    labelled ``owner`` (an in-process caller's default, a script's body) is the operator's; only a row with no
    recorded path (1.11) falls back to its label."""
    console = {"answer": {"path": "console", "by": "owner"}}
    for by in ("owner", "owner-turn:whatsapp:chat:m-1", "owner:cli"):
        assert asks_report.settler(by, "denied", console) == "operator", by
    assert asks_report.settler("owner-turn:whatsapp:chat:m-1", "denied", {"answer": {"path": "reply"}}) == "owner"
    assert asks_report.settler("cli", "denied", {"answer": {"path": "reply"}}) == "operator"
    assert asks_report.settler("owner", "denied") == "owner"                  # a 1.11 row: its label
    assert asks_report.settler("mind", "cancelled") == "mind"
    assert asks_report.settler("owner", "expired", console) == "lapse"


async def test_the_report_counts_a_console_answer_labelled_owner_as_the_operators(fx):
    row = asked(fx, "AB2")
    fx.shift(hours=1)
    fx.mind.outcomes.record(row.id, status="denied", summary="declined at the console", by="owner",
                            details={"answer": {"path": "console", "by": "owner"}}, implicit_verdict=False)
    value = report(fx)
    assert value["settled"] == {"owner": 0, "operator": 1, "lapse": 0, "mind": 0}
    assert value["owner_answer_rate"] == 0.0 and entry(value, row)["settled_by"] == "operator"


# -- the 1.11 baseline, read with what it has -----------------------------------------------------------------------

async def test_operator_answers_a_lapse_and_an_owner_yes_give_the_counts_and_the_times(fx):
    start = fx.now
    a, b, c = asked(fx, "AB2"), asked(fx, "CD3"), asked(fx, "EF4")
    lapsed = asked(fx, "GH5", type_="check_in", kind="message", cls="contact")
    owned = asked(fx, "JK6", type_="commitment_overdue", cls="owner")
    fx.shift(hours=10)
    notice(fx, [a, b, c, owned])                      # 10 h to first sight; the lapsed one is never delivered
    fx.shift(hours=1)
    await owner_answers(fx.mind, "JK6")
    for code, hours in (("AB2", 7), ("CD3", 24), ("EF4", 24)):
        fx.shift(hours=hours)
        await fx.mind.answer(code, yes=False, by="cli")
    fx.shift(hours=20)
    fx.mind._expire_asks(fx.now)
    value = report(fx)
    assert value["formed"] == 5 and value["open"] == 0
    assert value["by_type"] == {"research": 3, "check_in": 1, "commitment_overdue": 1}
    assert value["by_class"] == {"self": 3, "contact": 1, "owner": 1}
    assert value["settled"] == {"owner": 1, "operator": 3, "lapse": 1, "mind": 0}
    assert value["owner_answer_rate"] == pytest.approx(0.2)
    assert value["delivered"] == 4 and value["never_delivered"] == 1
    assert value["to_first_sight"] == {"count": 4, "median_s": 10 * HOUR, "p90_s": 10 * HOUR}
    assert value["to_answer"] == {"count": 1, "median_s": 1 * HOUR, "p90_s": 1 * HOUR}
    # The settling rows name their path: the CLI is the console, the owner's quoted reply is "reply".
    assert value["paths"] == {"console": 3, "reply": 1}
    assert value["readers"] == {"tables": 1}
    assert value["operator_verdicts"] == 0                          # an operator's no teaches nothing
    assert value["events"]["lapses"] == 1
    assert entry(value, lapsed)["first_sight_at"] is None and entry(value, lapsed)["settled_by"] == "lapse"
    assert entry(value, owned)["answer"] == "yes" and entry(value, a)["answer"] == "no"
    assert entry(value, a)["formed_at"] == start.isoformat()
    assert value["unbound_answers"] is None and value["invented_references"] is None   # not recorded yet
    text = asks_report.render(value)
    assert "5 asks formed" in text and "never reached the owner: 1" in text and "20%" in text
    assert "not recorded" in text and "JK6" not in text and "private plan" not in text  # counts, never ask text


async def test_a_code_reused_later_is_seen_only_for_the_ask_that_held_it(fx):
    first = asked(fx, "AB2")
    fx.shift(hours=1)
    await fx.mind.answer("AB2", yes=False, by="cli")
    fx.shift(hours=1)
    second = asked(fx, "AB2", context={"notice": True, "again": 1}, at=fx.now)
    fx.shift(hours=2)
    notice(fx, [second])
    value = report(fx)
    assert entry(value, first)["first_sight_at"] is None
    assert entry(value, second)["first_sight_at"] == fx.now.isoformat()


def test_a_digest_that_lists_an_ask_is_its_first_sight(fx):
    row = asked(fx, "AB2")
    fx.shift(hours=3)
    text = f"{digest_text.TITLE}\n\n{text_1_11([row])}"
    sent, _ = fx.store.create_intention(
        kind="message", type="digest", title="Daily digest", drive="upkeep", cls="owner", decision="act",
        decision_reason="owner notice", status="approved", dedup_key="digest:test", recipient=OWNER,
        context={"text": text}, hermes_kind="message", created_at=fx.now)
    fx.mind.outbox.sent(sent.id, result="sent")
    value = report(fx)
    assert entry(value, row)["first_sight_at"] == fx.now.isoformat()
    assert value["to_first_sight"]["median_s"] == 3 * HOUR


def test_an_ask_formed_before_the_window_is_not_counted(fx):
    asked(fx, "AB2", at=fx.now - timedelta(days=8))
    inside = asked(fx, "CD3", at=fx.now - timedelta(days=1))
    value = report(fx)
    assert value["formed"] == 1 and [item["id"] for item in value["asks"]] == [inside.id]
    assert value["per_day"] == pytest.approx(1 / 7)


def test_an_ask_counts_in_the_window_it_formed_in_whenever_its_intention_was_created(fx):
    """The window holds the asks formed in it: an intention deferred for a day and asked an hour ago is in a
    day's report, and an ask formed before the window is not, whatever happened to it inside."""
    deferred, _ = fx.store.create_intention(
        kind="task", type="research", title="The private plan AB2", drive="curiosity", cls="self", decision="defer",
        decision_reason="suggest: self -> defer", status="proposed", dedup_key=None, context={"notice": True},
        hermes_kind="none", created_at=fx.now - timedelta(hours=25), expires_at=fx.now + timedelta(days=2))
    formed = fx.now - timedelta(hours=1)
    fx.store.transition(deferred.id, "asked", action="asked", at=formed, decision="ask", ask_code="AB2",
                        details={"code": "AB2", "notice": True})
    before = asked(fx, "CD3", at=fx.now - timedelta(hours=30))
    fx.store.transition(before.id, "asked", action=asks_report.ASK_SHOWN, at=fx.now - timedelta(hours=2))
    value = asks_report.report(fx.store, since=fx.now - timedelta(hours=24), now=fx.now)
    assert value["formed"] == 1 and [item["id"] for item in value["asks"]] == [deferred.id]
    assert entry(value, deferred)["formed_at"] == formed.isoformat() and value["open"] == 1
    assert value["events"]["shows"] == 0


def test_an_open_question_moved_to_the_digest_is_no_approval_ask(fx):
    """A contradiction question the level would have asked goes to the digest and stays open there, answered in
    words on its own lifecycle: it is no approval ask, so it is never counted as a lapse or as never delivered."""
    from colonymind.mind.authority import Verdict
    from colonymind.mind.consolidate import QUESTION_KEY
    from colonymind.mind.questions import state_of as question_state
    task = asked(fx, "AB2")
    row, _ = fx.store.create_intention(
        kind="message", type="contradiction", title="Which is right for your office: room 4 or room 7?",
        drive="curiosity", cls="owner", decision="ask", decision_reason="suggest: owner -> ask", status="asked",
        dedup_key=f"{QUESTION_KEY}report0000000001", recipient=OWNER, ask_code="Q4R",
        context={"text": "You told me two things about your office. Which is right?", "notice": True},
        hermes_kind="none", created_at=fx.now, expires_at=fx.now + timedelta(days=3))
    fx.mind._apply_decision(row, Verdict(decision="ask", reason="suggest: owner -> ask", cls="owner"), fx.now,
                            reconsidered=False, code="Q4R")
    question = fx.store.get(row.id)
    assert question.decision == "ask" and question_state(question)["state"] == "digest"   # open, in the digest
    value = report(fx)
    assert [item["id"] for item in value["asks"]] == [task.id] and value["formed"] == 1
    assert value["settled"]["lapse"] == 0 and value["events"]["lapses"] == 0 and value["never_delivered"] == 1
    assert value["by_type"] == {"research": 1}


# -- what the answer path records from 1.12 on ------------------------------------------------------------------------

def test_the_answer_path_reader_and_events_are_read_from_the_settling_history(fx):
    row = asked(fx, "AB2")
    held = asked(fx, "CD3", cls="floor", type_="floor_effect", kind="note")
    fx.shift(minutes=5)
    shown, _ = fx.store.create_intention(
        kind="message", type="ask", title="Ask", drive="upkeep", cls="owner", decision="act",
        decision_reason="owner notice", status="approved", dedup_key="ask:AB2", recipient=OWNER,
        context={"text": "Can I re-run the export? Reply yes or no.", "asks": [row.id]}, hermes_kind="message",
        created_at=fx.now)
    fx.mind.outbox.sent(shown.id, result="sent")
    sight = fx.now
    fx.shift(minutes=1)
    fx.store.transition(row.id, "asked", action=asks_report.ASK_PAUSED, at=fx.now, details={"seconds": 600})
    fx.store.transition(row.id, "asked", action=asks_report.ASK_CLARIFIED, at=fx.now)
    fx.store.transition(row.id, "asked", action=asks_report.ASK_LATER, at=fx.now)
    fx.store.transition(held.id, "asked", action=asks_report.ASK_SHOWN, at=fx.now)
    fx.shift(minutes=9)
    fx.store.transition(row.id, "approved", action="queued", at=fx.now, details={
        "by": "owner", "code": "AB2", "answer": {"path": "reply", "reader": "tables", "by": "owner"}})
    fx.store.transition(row.id, "approved", action=asks_report.ASK_UNDONE, at=fx.now + timedelta(minutes=2))
    fx.shift(minutes=20)
    fx.store.transition(held.id, "cancelled", action="outcome_denied", at=fx.now, details={
        "by": "owner", "answer": {"path": "reaction", "reader": "tables", "by": "owner"}})
    value = report(fx)
    assert value["paths"] == {"reply": 1, "reaction": 1} and value["readers"] == {"tables": 2}
    assert value["events"] == {"shows": 1, "clarifications": 1, "later": 1, "undos": 1, "grace_cancels": 0,
                               "allows": 0, "revivals": 0, "lapses": 0, "paused": 1}
    assert value["paused"] == {"count": 1, "median_s": 600.0, "p90_s": 600.0}
    assert entry(value, row)["first_sight_at"] == sight.isoformat()
    assert value["to_answer"]["count"] == 2 and value["settled"]["owner"] == 2
    assert entry(value, held)["first_sight_at"] is not None       # shown in the chat: its first sight


def test_a_revived_ask_is_settled_by_the_answer_after_its_revival(fx):
    row = asked(fx, "AB2")
    fx.shift(hours=73)
    fx.mind._expire_asks(fx.now)
    fx.shift(hours=24)
    fx.store.transition(row.id, "asked", action=asks_report.ASK_REVIVED, at=fx.now)
    fx.shift(minutes=1)
    fx.store.transition(row.id, "approved", action="queued", at=fx.now, details={
        "by": "owner", "answer": {"path": "reply", "reader": "tables", "by": "owner"}})
    value = asks_report.report(fx.store, since=fx.now - timedelta(days=7), now=fx.now)
    assert entry(value, row)["settled_by"] == "owner" and entry(value, row)["answer"] == "yes"
    assert value["events"]["lapses"] == 1 and value["events"]["revivals"] == 1
    assert value["settled"] == {"owner": 1, "operator": 0, "lapse": 0, "mind": 0}


def test_message_level_events_are_counted_when_the_answer_path_records_them(fx):
    asked(fx, "AB2")
    events = [{"kind": asks_report.UNBOUND_ANSWER, "at": fx.now - timedelta(hours=1)},
              {"kind": asks_report.UNBOUND_ANSWER, "at": fx.now - timedelta(days=9)},
              {"kind": asks_report.INVENTED_REFERENCE, "at": fx.now - timedelta(hours=2)}]
    value = report(fx, events=events)
    assert value["unbound_answers"] == 1 and value["invented_references"] == 1


async def test_held_work_counts_the_hours_work_waited_on_the_owner(fx):
    task = asked(fx, "AB2", kind="task")
    asked(fx, "CD3", kind="note", type_="people_change", cls="owner")      # a question holds no work
    lapsing = asked(fx, "EF4", kind="task")
    fx.shift(hours=2)
    await owner_answers(fx.mind, "AB2")
    fx.shift(hours=71)
    fx.mind._expire_asks(fx.now)
    value = report(fx)
    assert value["held_work_hours"] == {"total": pytest.approx(2 + 73), "lapsed": pytest.approx(73)}
    assert entry(value, task)["settled_by"] == "owner" and entry(value, lapsing)["settled_by"] == "lapse"


def test_an_empty_window_reports_nothing_and_says_so(fx):
    value = report(fx)
    assert value["formed"] == 0 and value["owner_answer_rate"] is None
    assert value["to_first_sight"] == {"count": 0, "median_s": None, "p90_s": None}
    assert "No asks formed" in asks_report.render(value)


def test_p90_is_the_nearest_rank():
    assert asks_report.spread([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == {"count": 10, "median_s": 5.5, "p90_s": 9}
    assert asks_report.spread([4]) == {"count": 1, "median_s": 4, "p90_s": 4}


# -- the route: the owner's view only ---------------------------------------------------------------------------------

OPS_SECRET = "ops-credential-secret-0123456789"


async def _get(fx, path, *, secret, params=None):
    table = CredentialTable.from_config({"ops": {"sha256": secret_digest(OPS_SECRET), "owner": [], "contacts": [],
                                                 "access": ["manage"]}})
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=table)
    app.include_router(mind_router.router)
    mind_router.set_mind(fx.mind)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        return await client.get(path, params=params, headers={"Authorization": "Bearer " + secret})


async def test_the_stats_route_serves_the_owner_view_and_a_guest_nothing(fx):
    row = asked(fx, "AB2")
    fx.shift(hours=1)
    await fx.mind.answer("AB2", yes=False, by="cli")
    response = await _get(fx, "/v1/mind/asks/stats", secret=KEY, params={"since_hours": 24})
    assert response.status_code == 200
    value = response.json()
    assert value["formed"] == 1 and value["settled"]["operator"] == 1 and "1 ask formed" in value["text"]
    assert value["asks"][0]["id"] == row.id
    narrow = await _get(fx, "/v1/mind/asks/stats", secret=KEY, params={"since_hours": 0.5})
    assert narrow.json()["formed"] == 0
    for secret, params in ((OPS_SECRET, {}), (KEY, {"viewer": "p-02"})):
        guest = await _get(fx, "/v1/mind/asks/stats", secret=secret, params=params)
        assert guest.status_code == 200 and "asks" not in guest.json() and "formed" not in guest.json()


# -- the CLI ----------------------------------------------------------------------------------------------------------

def _parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default=None)
    sub = parser.add_subparsers(dest="command")
    mind_cli.add_parser(sub)
    return parser.parse_args(argv)


@pytest.fixture
def home(tmp_path, monkeypatch):
    from colonymind.config import DEFAULTS, save_config
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("COLONYMIND_API_KEY", raising=False)
    save_config({**DEFAULTS, "owner": {"contact_id": OWNER}}, tmp_path)
    (tmp_path / "api.key").write_text("k\n")
    return tmp_path


def test_the_cli_asks_the_sidecar_for_the_window_and_prints_text_or_json(home, monkeypatch, capsys):
    calls = []
    real = httpx.Client

    def client(**kwargs):
        def handler(request):
            calls.append((request.method, request.url.path, dict(request.url.params)))
            return httpx.Response(200, json={"formed": 2, "text": "2 asks formed in the window."})
        return real(transport=httpx.MockTransport(handler), timeout=kwargs.get("timeout", 5))
    monkeypatch.setattr(httpx, "Client", client)
    assert mind_cli.run(_parse(["mind", "asks-report", "--since", "7d"])) == 0
    assert capsys.readouterr().out.strip() == "2 asks formed in the window."
    assert calls[-1][:2] == ("GET", "/v1/mind/asks/stats") and float(calls[-1][2]["since_hours"]) == 168.0
    assert mind_cli.run(_parse(["mind", "asks-report", "--since", "24h", "--json"])) == 0
    assert json.loads(capsys.readouterr().out)["formed"] == 2 and float(calls[-1][2]["since_hours"]) == 24.0
    assert mind_cli.run(_parse(["mind", "--json", "asks-report"])) == 0
    assert json.loads(capsys.readouterr().out)["formed"] == 2 and float(calls[-1][2]["since_hours"]) == 168.0
    count = len(calls)
    assert mind_cli.run(_parse(["mind", "asks-report", "--since", "soon"])) == 1
    assert "--since takes a span" in capsys.readouterr().err and len(calls) == count
