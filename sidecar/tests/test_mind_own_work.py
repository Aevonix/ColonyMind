"""The agent's own work on the board: an idea that touches nobody else starts by itself, its run is read-only (any
other effect is held as its own ask), the done-check decides how the work settles, a task's report goes into memory,
and every task has one recorder line."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from colonymind import recorder
from colonymind.api.routers import mind as mind_router
from colonymind.mind.ideas import COPY_VERSION, MARKER, digest
from colonymind.mind.rank import Candidate
from colonymind.secrets.vault import Vault
from test_mind_loop import OWNER, Fixture, pinned_clock  # noqa: F401  (pinned_clock: autouse pytest fixture)
from owner_answer import owner_answers

QUOTE = "The notice names the 14th; the contract names the 21st."
FINDING = {"kind": "result_field", "field": "finding"}


@pytest.fixture
def fx(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fixture = Fixture(tmp_path)
    yield fixture
    mind_router.set_mind(None)
    fixture.store.close()


def idea(fx):
    """A passive idea card as ``Ideas.publish`` leaves it: a sealed payload, the owner as recipient, decision ask."""
    ideas = fx.mind.ideas
    ideas._vault = Vault(fx.state, key_store="file")
    expires = fx.now + timedelta(days=2)
    payload = {"family": "packet", "content": {"title": "Two dates disagree", "summary": "Compare them."},
               "items": [{"id": "evidence", "type": "deliverable", "selectable": False, "isSelected": True},
                         {"id": "draft", "type": "deliverable", "selectable": True, "isSelected": True}],
               "quotes": [{"sourceId": "s1", "quote": QUOTE}], "sources": [], "action_key": "packet:dates",
               "expires": expires.isoformat(), "generated_at": fx.now.isoformat()}
    row, _ = fx.store.create_intention(
        kind="task", type="grounded_offer", title="Private owner offer", drive="duty", cls="owner", decision="ask",
        decision_reason="Passive card; only an owner Build starts work", status="proposed", dedup_key="idea:dates",
        recipient=OWNER, priority=0.9, created_at=fx.now, expires_at=expires,
        context={"idea": {"generator": MARKER, "owner": OWNER, "action_key": payload["action_key"],
                          "copy": COPY_VERSION, "revision": digest(payload), "private": ideas.seal(payload)}})
    return row


async def research(fx, n, *, check=FINDING, text=None):
    fx.shift(minutes=5)
    row = await fx.mind._form(Candidate(type="research", drive="curiosity", kind="task", title=f"Research topic {n}",
                                        dedup_key=f"research:topic-{n}", salience=0.9, cost=0.1,
                                        text=text or f"Find out about topic {n}.", topic=f"topic {n}",
                                        concern=f"research topic {n}", evidence=[f"interest:topic-{n}"],
                                        success_check=check), 0.9, fx.now)
    assert row is not None and row.status == "approved", row
    return row


def line_of(row):
    """The one recorder line of an intention."""
    [line] = [line for line in recorder.recent(2000, kind="card", source="mind.task")
              if line["ref"] == f"initiative:{row.id}"]
    return line


def label_of(fx, row):
    """The label the settling history row of an intention carries."""
    [settled] = [item for item in fx.store.get_history(row.id) if item.action.startswith("outcome_")]
    return settled.details.get("label")


def memory(fx, row, event):
    with fx.ledger._connect() as conn:
        found = conn.execute("SELECT messages_json FROM turn_sources WHERE turn_id=?",
                             (f"mind:{row.id}:{event}",)).fetchone()
    return json.loads(found[0])[0] if found else None


# -- an idea that touches only the owner starts by itself -------------------------------------------------------

async def test_an_idea_that_touches_only_the_owner_starts_as_a_board_task(fx):
    row = idea(fx)
    await fx.mind.ideas.tick(fx.now, schedule=False)            # the tick that forms no work starts none
    assert fx.store.get(row.id).status == "proposed"
    await fx.mind.ideas.tick(fx.now)
    started = fx.store.get(row.id)
    assert (started.status, started.decision, started.cls) == ("approved", "act", "owner")
    assert started.decision_reason == "standard: owner -> act" and started.expires_at == row.expires_at
    assert started.description == "Compare the documents" and fx.mind.ideas.catalog() == []
    assert await fx.mind.outbox._done_line(started, {}) == "Started: Compare the documents."   # told in the digest
    [item] = [item for item in fx.mind.dispatch() if item["id"] == row.id]
    assert item["assignee"] == "colonymind-act" and item["title"] == "Compare the documents"
    assert "\nresult: <" in item["body"] and json.dumps(QUOTE) in item["body"]
    assert "Prepare an unsent reply" in item["body"] and "Two dates disagree" not in item["body"]
    fx.mind.bound(row.id, "kanban:idea")
    assert fx.mind.ideas.card(fx.store.get(row.id))["buildState"] == "working"
    done = fx.mind.report(row.id, status="done", outcome="done", final=True,
                          summary="result: the contract date stands.")
    assert (done.status, done.verified) == ("done", "check")
    assert fx.mind.ideas.card(done)["result"] == {"packet": "result: the contract date stands."}
    assert "the contract date stands" in memory(fx, done, "result")["content"]


async def test_at_suggest_an_idea_stays_a_passive_card(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fx = Fixture(tmp_path, autonomy="suggest")
    try:
        row = idea(fx)
        await fx.mind.ideas.tick(fx.now)
        kept = fx.store.get(row.id)
        assert (kept.status, kept.decision) == ("proposed", "ask")
        assert kept.decision_reason == ("Passive card (suggest: owner effects are suggested in the digest); "
                                        "only an owner Build starts work")
        assert [card.id for card in fx.mind.ideas.catalog()] == [row.id]
        assert not [item for item in fx.mind.dispatch() if item["id"] == row.id]
    finally:
        fx.store.close()


# -- a self-started run is read-only: any other effect is its own ask ---------------------------------------------

async def test_a_write_in_a_self_started_run_is_held_as_its_own_ask(fx):
    row = idea(fx)
    await fx.mind.ideas.tick(fx.now)
    fx.mind.bound(row.id, "kanban:idea")
    read = await fx.mind.guard(tool="read_file", args={"path": "notes.md"}, run="mind", task_id="kanban:idea")
    assert read["allow"] is True
    held = await fx.mind.guard(tool="mcp_calendar_create_event", args={"title": "Renewal"}, run="mind",
                               task_id="kanban:idea")
    assert held["action"] == "block" and held["allow"] is False and held["held"]
    ask = fx.store.get_by_ask_code(held["held"])
    assert (ask.type, ask.status, ask.decision, ask.cls) == ("floor_effect", "asked", "ask", "floor")
    assert ask.context["origin_task"] == row.id
    # The owner's yes makes the continuation, which is told the line the held task's done-check reads.
    await owner_answers(fx.mind, held["held"])
    [offered] = [item for item in fx.mind.dispatch() if item["id"] == ask.id]
    assert "\nresult: <" in offered["body"] and "mcp_calendar_create_event" in offered["body"]
    assert "owner_acted" in line_of(ask)["marks"]


# -- the done-check decides -----------------------------------------------------------------------------------

async def test_a_task_is_told_the_line_its_done_check_reads(fx):
    row = await research(fx, 1)
    assert row.context["plan_body"] == "Find out about topic 1."
    assert row.context["body"].startswith("Find out about topic 1.\n\n") and "\nfinding: <" in row.context["body"]
    named = await research(fx, 2, text="Find out about topic 2; report finding: <text>.")
    assert named.context["body"] == "Find out about topic 2; report finding: <text>."
    [item] = [item for item in fx.mind.dispatch() if item["id"] == row.id]
    assert item["body"] == row.context["body"]


async def test_a_done_report_whose_check_does_not_pass_settles_failed_with_a_label(fx):
    row = await research(fx, 1)
    fx.mind.bound(row.id, "kanban:1")
    missed = fx.mind.report(row.id, status="done", outcome="done", final=True, summary="nothing turned up")
    assert (missed.status, missed.outcome, missed.verified) == ("failed", "failed", "check")
    assert missed.result == "nothing turned up" and missed.failed_reason == "check failed: nothing turned up"
    assert missed.result_metadata["check"]["passed"] is False
    assert missed.result_metadata["breaker"] == {"counted": False, "reason": "check not passed"}
    assert label_of(fx, missed) == "check_not_passed"
    assert fx.mind.authority.breaker_state("internal")["failures"] == 0
    passed = await research(fx, 2)
    fx.mind.bound(passed.id, "kanban:2")
    done = fx.mind.report(passed.id, status="done", outcome="done", final=True, summary="finding: it is tidal.")
    assert (done.status, done.verified) == ("done", "check") and done.result_metadata["check"]["passed"] is True
    # The owner's word stands whatever the check reads.
    owned = await research(fx, 3)
    fx.mind.bound(owned.id, "kanban:3")
    kept = fx.mind.outcomes.record(owned.id, status="done", summary="the owner confirmed it", verified="owner",
                                   by="owner")
    assert (kept.status, kept.verified) == ("done", "owner")


async def test_a_run_stopped_at_its_runtime_limit_fails_as_elapsed(fx):
    row = await research(fx, 1)
    fx.mind.bound(row.id, "kanban:1")
    failed = fx.mind.report(row.id, status="failed", outcome="failed", final=True, error="elapsed 612s > limit 600s")
    assert failed.status == "failed" and label_of(fx, failed) == "elapsed"
    assert line_of(failed)["detail"] == "elapsed"


# -- a task's result goes into memory, and each task has one recorder line -------------------------------------

async def test_a_settled_task_writes_its_result_into_memory_and_closes_its_line(fx):
    task, _ = fx.store.create_intention(kind="task", type="upkeep", title="Tidy the notes folder", drive="upkeep",
                                        cls="internal", decision="act", decision_reason="standard: internal -> act",
                                        status="approved", dedup_key="task:tidy", context={"body": "Tidy it."},
                                        created_at=fx.now)
    opened = line_of(task)
    assert opened["result"] == "" and opened["took_ms"] is None      # raised, not settled
    fx.mind.bound(task.id, "kanban:tidy")
    done = fx.mind.report(task.id, status="done", outcome="done", final=True, summary="Merged four duplicate notes.")
    entry = memory(fx, done, "result")
    assert entry["content"] == "What my task 'Tidy the notes folder' came to (done): Merged four duplicate notes."
    assert entry["metadata"]["intention_id"] == task.id and entry["metadata"]["origin"] == "mind"
    closed = line_of(done)
    assert (closed["result"], closed["choice"], closed["chooser"], closed["detail"]) == \
        ("done", "act", "rule", "outcome_done")
    missed = await research(fx, 1)
    fx.mind.bound(missed.id, "kanban:1")
    fx.mind.report(missed.id, status="done", outcome="done", final=True, summary="nothing turned up")
    assert (line_of(missed)["result"], line_of(missed)["detail"]) == ("failed", "check_not_passed")


async def test_the_owners_no_marks_the_line_of_an_asked_task(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fx = Fixture(tmp_path)
    try:
        fx.shift(minutes=5)
        asked = await fx.mind._form(Candidate(type="upkeep", drive="upkeep", kind="task", title="Rotate the logs",
                                              dedup_key="task:rotate", salience=0.9, cost=0.1, text="Rotate them.",
                                              concern="rotate the logs", evidence=["interest:logs"],
                                              ask_owner=True), 0.9, fx.now)
        assert asked.status == "asked" and line_of(asked)["marks"] == []
        await owner_answers(fx.mind, asked.ask_code, "no")
        line = line_of(asked)
        assert line["marks"] == ["owner_declined"] and (line["result"], line["choice"]) == ("silent", "ask_owner")
    finally:
        fx.store.close()
