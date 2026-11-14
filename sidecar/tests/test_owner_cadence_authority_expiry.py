"""Owner check-in slots keep their deadline through asks and approval."""

from dataclasses import replace
from datetime import timedelta

import pytest

from owner_answer import owner_answers
from colonymind.mind import Mind
from colonymind.mind.authority import Verdict
from colonymind.mind.tick import TASK_WINDOW, ask_expiry
from test_mind_outreach_loop import H, OWNER, T0, make  # noqa: F401
from test_mind_owner_cadence import DAY, create_cadence


async def test_approved_ask_expires_on_restart_and_next_day_sends_at_its_slot(make):
    fx = make(config={"autonomy": "suggest"})
    cadence = create_cadence(fx, created=T0 - DAY)
    await fx.mind.tick(force=True)
    formed, = fx.outreach_rows()
    assert formed.status == "asked" and formed.expires_at == T0 + 3 * H
    fx.shift(H)
    approved = await owner_answers(fx.mind, formed.id)
    assert approved.status == "approved" and approved.expires_at == T0 + 3 * H

    fx.now = T0 + 11 * H
    fx.mind = Mind(config=fx.mind.config, store=fx.store, state_dir=fx.store._state_dir,
                   owner_id=OWNER, commitments=fx.commitments, feedback=fx.feedback, ledger=fx.ledger,
                   clock=lambda: fx.now, backups=False)
    fx.mind.digest_hour = 25
    await fx.tick()
    assert fx.store.get(formed.id).status == "expired"
    assert fx.sent == []
    fx.now = T0 + DAY - timedelta(seconds=1)
    await fx.mind.tick(force=True)
    assert len(fx.outreach_rows()) == 1
    fx.shift(timedelta(seconds=1))
    await fx.mind.tick(force=True)
    next_day, = [row for row in fx.outreach_rows() if row.status == "asked"]
    assert next_day.id != formed.id and next_day.source_id == cadence["id"]
    assert next_day.created_at == T0 + DAY
    await owner_answers(fx.mind, next_day.id)
    await fx.tick()
    assert [item["id"] for item in fx.sent] == [next_day.id]
    assert fx.store.get(next_day.id).completed_at == T0 + DAY


async def test_unanswered_ask_lapses_at_grace_end_and_next_day_forms(make):
    fx = make(config={"autonomy": "suggest"})
    create_cadence(fx, created=T0 - DAY)
    await fx.mind.tick(force=True)
    formed, = fx.outreach_rows()
    assert formed.status == "asked" and formed.expires_at == T0 + 3 * H
    fx.now = T0 + 3 * H
    await fx.tick()
    assert fx.store.get(formed.id).status == "expired"
    assert fx.store.get(formed.id).verdict is None
    assert fx.sent == []
    fx.now = T0 + DAY
    await fx.mind.tick(force=True)
    next_day, = [row for row in fx.outreach_rows() if row.status == "asked"]
    assert next_day.created_at == T0 + DAY
    assert next_day.expires_at == T0 + DAY + 3 * H
    await owner_answers(fx.mind, next_day.id)
    await fx.tick()
    assert [item["id"] for item in fx.sent] == [next_day.id]


@pytest.mark.parametrize("minutes", [240, 1440, 7 * 1440])
async def test_check_in_approved_within_grace_still_sends(make, minutes):
    fx = make(config={"autonomy": "suggest"})
    interval = timedelta(minutes=minutes)
    create_cadence(fx, created=T0 - interval, minutes=minutes)
    await fx.mind.tick(force=True)
    formed, = fx.outreach_rows()
    expiry = T0 + min(interval / 2, 3 * H)
    assert formed.expires_at == expiry
    fx.now = expiry - timedelta(minutes=1)
    approved = await owner_answers(fx.mind, formed.id)
    assert approved.status == "approved" and approved.expires_at == expiry
    await fx.tick()
    assert [item["id"] for item in fx.sent] == [formed.id]


@pytest.mark.parametrize("owner_cadence", [False, True])
@pytest.mark.parametrize("hours", [1, 48])
@pytest.mark.parametrize("decision", ["ask", "act", "approve"])
async def test_transitions_cap_only_owner_cadences_and_keep_shorter_expiry(
        make, monkeypatch, owner_cadence, hours, decision):
    fx = make(config={"autonomy": "suggest"})
    create_cadence(fx, created=T0 - DAY)
    await fx.mind.tick(force=True)
    formed, = fx.outreach_rows()
    row = fx.store.update(formed.id, context={**formed.context, "owner_cadence": owner_cadence})
    monkeypatch.setattr("colonymind.mind.tick.ask_expiry", lambda now, policy: now + hours * H)
    monkeypatch.setattr("colonymind.mind.tick.TASK_WINDOW", hours * H)
    if decision == "approve":
        updated = await owner_answers(fx.mind, row.id)
    else:
        updated = fx.mind._apply_decision(row, Verdict(decision=decision, reason="reconsidered", cls="internal"),
                                          fx.now, reconsidered=True)
    assert updated.status == ("asked" if decision == "ask" else "approved")
    assert updated.expires_at == T0 + (min(hours, 3) if owner_cadence else hours) * H


@pytest.mark.parametrize("decision", ["ask", "act"])
def test_transition_without_context_keeps_ordinary_expiry(make, decision):
    fx = make()
    row, _ = fx.store.create_intention(kind="task", type="research", title="Research", drive="curiosity",
        cls="internal", decision="defer", decision_reason="deferred", status="proposed", dedup_key=None,
        created_at=T0)
    updated = fx.mind._apply_decision(replace(row, context=None),
        Verdict(decision=decision, reason="reconsidered", cls="internal"), T0, reconsidered=True)
    assert updated.status == ("asked" if decision == "ask" else "approved")
    assert updated.expires_at == (ask_expiry(T0, fx.mind.policy) if decision == "ask" else T0 + TASK_WINDOW)
