"""Owner-addressed cadences use owner outreach, never the contact check-in path."""

from datetime import timedelta
from types import SimpleNamespace

import pytest

from colonymind.mind import outreach, rank
from colonymind.mind.affect import AffectView, discretionary
from colonymind.mind.authority import Authority, Policy
from colonymind.mind.drives import DriveInputs, cadence_confirm_candidate, duty, owner_cadence, social
from test_mind_outreach_loop import H, OWNER, T0, make  # noqa: F401

DAY = timedelta(days=1)


def cadence_row(*, recipient="me", status="pending", minutes=1440):
    return {"id": "cadence-1", "person_id": OWNER, "status": status, "made_at": T0.isoformat(),
            "description": "Check in with me daily about the garden", "due_at": None,
            "metadata": {"kind": "cadence", "recipient": recipient, "cadence_minutes": minutes,
                         "topic": "the garden", "obligor": "assistant"}}


def create_cadence(fx, *, created=T0, minutes=1440):
    row = fx.owner_commitment("Check in with me about the garden", created=created,
                              **cadence_row(minutes=minutes)["metadata"])
    return fx.commitments.get(row["id"])


def snapshot(fx, rows):
    inputs = outreach.OutreachInputs(now=fx.now, owner_id=OWNER)
    inputs.cadences = fx.mind._owner_cadences(rows, inputs=inputs)
    return inputs


@pytest.mark.parametrize("recipient", ["me", "owner", OWNER])
def test_owner_names_resolve_without_becoming_contact_cadences(recipient):
    row = cadence_row(recipient=recipient)
    assert owner_cadence(row, owner_id=OWNER) == (OWNER, 1440, "the garden")
    assert cadence_confirm_candidate(row, owner_id=OWNER, current=None) is None
    inputs = DriveInputs(now=T0 + DAY, owner_id=OWNER, commitments=[row],
                         contacts=[{"contact_id": OWNER, "cadence_minutes": 1440, "may_contact": "auto"}])
    assert duty(inputs)[1] == [] and social(inputs)[1] == []


def test_daily_cadence_is_one_due_owner_candidate_not_a_contact_check_in(make):
    fx = make()
    row = create_cadence(fx)
    fx.shift(DAY - timedelta(seconds=1))
    assert outreach.cadence_candidates(snapshot(fx, [row])) == []
    fx.shift(timedelta(seconds=1))
    inputs = snapshot(fx, [row])
    candidate, = outreach.cadence_candidates(inputs)
    assert candidate.type == "outreach_loop" and candidate.recipient == OWNER
    assert candidate.source_type == "commitment" and candidate.source_id == row["id"]
    assert candidate.extra["source_ref"] == f"commitment:{row['id']}"
    assert candidate.invalidates_if == f"commitment:{row['id']}:resolved"
    drives = DriveInputs(now=fx.now, owner_id=OWNER, commitments=[row], outreach=inputs)
    assert duty(drives)[1] == [candidate] and social(drives)[1] == []


def test_requested_cadence_has_owed_ranking_even_with_negative_feedback(make):
    fx = make()
    row = create_cadence(fx)
    fx.shift(DAY)
    inputs = snapshot(fx, [row])
    inputs.queued_24h = 10
    candidate, = outreach.cadence_candidates(inputs)
    view = AffectView(route={}, owner_id=OWNER, satiated=True, overloaded=True, boost=0.5)
    assert candidate.drive == "duty" and not discretionary(candidate)
    assert candidate.multiplier_keys() == [] and candidate.dedup_base is None
    assert rank.eligible([candidate], feedback=SimpleNamespace(multiplier=lambda _: 0.01), affect=view)
    assert candidate.type not in Authority.REQUESTED_TYPES
    gate = Authority(Policy(level="suggest"), fx.store, owner_id=OWNER, clock=lambda: fx.now)
    assert gate.decide(kind="message", recipient=OWNER, type=candidate.type, text=candidate.text).decision == "ask"


@pytest.mark.parametrize("status", ["cancelled", "fulfilled"])
def test_closed_cadence_has_no_candidate(make, status):
    fx = make()
    fx.shift(100 * DAY)
    assert outreach.cadence_candidates(snapshot(fx, [cadence_row(status=status)])) == []


@pytest.mark.parametrize("hold", ["quiet", "budget", "paused", "muted"])
def test_owner_cadence_obeys_outreach_holds(make, hold):
    fx = make()
    row = create_cadence(fx)
    fx.shift(DAY)
    inputs = snapshot(fx, [row])
    if hold == "quiet":
        inputs.quiet = True
    elif hold == "budget":
        inputs.budget = "outreach budget exhausted"
    elif hold == "paused":
        inputs.paused_until = fx.now + DAY
    else:
        inputs.mutes = {"the-garden": "the garden"}
    assert outreach.cadence_candidates(inputs) == []


def test_history_read_failure_is_visible_and_forms_nothing(make, monkeypatch, caplog):
    import sqlite3
    from colonymind.health_counters import snapshot as counters

    fx = make()
    row = create_cadence(fx)
    before = counters().get("context_reads_failed", 0)

    def broken(*_, **__):
        error = sqlite3.OperationalError("private content must not be logged")
        error.sqlite_errorname = "SQLITE_BUSY"
        raise error

    with monkeypatch.context() as patch:
        patch.setattr("colonymind.mind.tick.sqlite3.connect", broken)
        assert fx.mind._owner_cadences([row]) == []
    assert counters()["context_reads_failed"] == before + 1
    assert "OperationalError" in caplog.text and "SQLITE_BUSY" in caplog.text
    assert "private content" not in caplog.text


async def test_long_gap_sends_once_then_waits_a_full_cadence_without_scoring_silence(make):
    fx = make()
    row = create_cadence(fx)
    fx.shift(100 * DAY)
    assert len((await fx.tick())["formed"]) == 1
    first, = fx.outreach_rows()
    assert first.status == "sent" and first.type == "outreach_loop"
    assert fx.commitments.get(row["id"])["status"] == "pending"
    assert (await fx.tick())["formed"] == []
    fx.shift(DAY - timedelta(seconds=1))
    assert (await fx.tick())["formed"] == []
    fx.shift(timedelta(seconds=1))
    summary = await fx.tick()
    assert summary["outreach_scored"]["ignored"] == 0
    assert fx.store.get(first.id).verdict is None
    assert fx.feedback.multiplier("outreach_loop:duty") == 1.0
    assert len(fx.outreach_rows()) == 2
    assert (await fx.tick())["formed"] == []


async def test_waiting_check_in_blocks_the_next_and_cancellation_stops_delivery(make):
    fx = make()
    row = create_cadence(fx, created=T0 - DAY)
    await fx.mind.tick(force=True)  # no outbox pull yet
    waiting, = fx.outreach_rows()
    assert waiting.status == "approved"
    assert fx.mind._owner_cadences([row]) == []
    fx.commitments.resolve(row["id"], outcome="wont_do")
    assert await fx.mind.outbox_ready() == []
    fx.shift(2 * DAY)
    assert (await fx.tick())["formed"] == []


@pytest.mark.parametrize("config", [{"quiet_hours": "00:00-23:59"}, {"budgets": {"outreach_per_day": 0}}])
async def test_tick_respects_quiet_hours_and_outreach_budget(make, config):
    fx = make(config=config)
    create_cadence(fx, created=T0 - DAY)
    assert (await fx.tick())["formed"] == [] and fx.sent == []


async def test_send_spends_outreach_budget_and_owner_reply_is_feedback(make):
    fx = make(config={"budgets": {"outreach_per_day": 1, "owner_messages_per_day": 0}})
    create_cadence(fx, created=T0 - DAY)
    await fx.tick()
    sent, = fx.outreach_rows()
    assert sent.status == "sent"
    assert fx.mind.authority.budget_check(kind="message", recipient=OWNER, type="outreach_loop", now=fx.now)
    fx.shift(H)
    fx.mind._react(sent, "engaged", turn="owner-reply", how="topic", now=fx.now)
    assert fx.store.get(sent.id).verdict == "actioned"
    assert fx.feedback.multiplier("outreach_loop:duty") > 1.0


async def test_owner_cadence_does_not_require_people_and_does_not_set_owner_contact_cadence(make):
    fx = make(config={"faculties": {"people": False}})
    create_cadence(fx, created=T0 - DAY)
    await fx.tick()
    assert len(fx.outreach_rows()) == 1


async def test_owner_cadence_never_sets_the_owner_contact_cadence(make):
    from test_mind_social import FakeContacts, contact

    fx = make()
    fx.mind.contacts = FakeContacts([contact(OWNER, may_contact="auto")])
    create_cadence(fx, created=T0 - DAY)
    await fx.tick()
    assert fx.mind.contacts.cadences == []
    assert fx.mind.contacts.records[OWNER]["cadence_minutes"] is None
    assert len(fx.outreach_rows()) == 1


async def test_owner_gate_still_reviews_the_cadence(make):
    fx = make(config={"faculties": {"owner_gate": True}})
    create_cadence(fx, created=T0 - DAY)
    await fx.tick()
    first, = fx.outreach_rows()
    assert first.status == "sent"
    fx.shift(DAY + timedelta(seconds=1))
    await fx.tick()
    assert len(fx.outreach_rows()) == 2
    assert all(row.status == "sent" for row in fx.outreach_rows())
    assert len(fx.mind.owner_gate.ledger.recent(since=T0)) >= 2


async def test_cadence_longer_than_outreach_history_still_uses_the_previous_send(make):
    fx = make()
    row = create_cadence(fx, created=T0 - 60 * DAY, minutes=60 * 1440)
    await fx.tick()
    first, = fx.outreach_rows()
    fx.shift(40 * DAY)
    assert fx.mind._outreach_sent(fx.now) == []
    assert outreach.cadence_candidates(snapshot(fx, [row])) == []
    fx.shift(20 * DAY)
    await fx.tick()
    assert len(fx.outreach_rows()) == 2
    assert first.id in {item.id for item in fx.outreach_rows()}


async def test_equivalent_rows_share_pending_and_sent_history(make):
    fx = make()
    first = create_cadence(fx, created=T0 - 10 * DAY)
    second = create_cadence(fx, created=T0 - 5 * DAY)
    assert first["id"] != second["id"]
    await fx.mind.tick(force=True)
    assert len(fx.outreach_rows()) == 1
    assert fx.mind._owner_cadences([second, first]) == []
    await fx.tick()
    fx.shift(outreach.MIN_GAP)
    await fx.tick()
    assert len(fx.outreach_rows()) == 1
    fx.shift(DAY - outreach.MIN_GAP)
    await fx.tick()
    assert len(fx.outreach_rows()) == 2


@pytest.mark.parametrize("stated", [False, True])
@pytest.mark.parametrize("upgrade_hour", [1, 18, 23])
async def test_first_daily_send_uses_row_local_time_not_upgrade(make, stated, upgrade_hour):
    from zoneinfo import ZoneInfo

    fx = make()
    fx.mind.tz = ZoneInfo("America/New_York")
    local = T0.astimezone(fx.mind.tz).replace(hour=9)
    created = local - 30 * DAY
    metadata = cadence_row()["metadata"]
    fx.owner_commitment("Requested daily check-in", created=created if not stated else created - 3 * H,
                        due=created if stated else None, **metadata)
    fx.now = local.replace(hour=upgrade_hour)
    due = local
    assert (await fx.tick())["formed"] == []
    if upgrade_hour > 9:
        due += DAY
    fx.now = due - timedelta(seconds=1)
    assert (await fx.tick())["formed"] == []
    fx.now = due
    await fx.tick()
    assert len(fx.outreach_rows()) == 1


async def test_requested_sends_leave_discretionary_cost_and_eligibility_unchanged(make):
    fx = make()
    create_cadence(fx, created=T0 - DAY)
    for _ in range(3):
        await fx.tick()
        fx.shift(DAY)
    assert fx.mind._score_outreach(fx.now)["ignored"] == 0
    assert all(row.verdict is None for row in fx.outreach_rows())
    # Old releases may already have silence ratings; those must not count either.
    for row in fx.outreach_rows():
        fx.store.update(row.id, verdict="ignored", result_metadata={"reaction": {"class": "silence"}})
    state = (await fx.mind._gather(fx.now)).outreach
    assert len(state.sent) == 3 and all(item.requested for item in state.sent)
    assert state.queued_24h == 0
    assert outreach.ignored_streak(state) == 0
    assert outreach.interruption_cost(state) == outreach.interruption_cost(
        outreach.OutreachInputs(now=fx.now, owner_id=OWNER)) == 0
    candidate = outreach._message("outreach_finding", state, key="new-finding", topic="weather",
                                   text="A finding", why="An interest", ev={}, salience=1.0,
                                   cost=outreach.interruption_cost(state), source="intention:new",
                                   expires=12, evidence=[])
    assert rank.eligible([candidate], threshold=0.6)


async def test_requested_send_still_enforces_min_gap(make):
    fx = make()
    create_cadence(fx, created=T0 - DAY)
    await fx.tick()
    state = (await fx.mind._gather(fx.now)).outreach
    assert outreach.holds(state) is not None
    fx.shift(outreach.MIN_GAP)
    state = (await fx.mind._gather(fx.now)).outreach
    assert outreach.holds(state) is None


async def test_cadence_history_does_not_see_writer_transaction(make):
    import asyncio

    fx = make()
    row = create_cadence(fx, created=T0 - DAY)
    await fx.tick()
    sent, = fx.outreach_rows()
    fx.store._db.execute("UPDATE initiatives SET status='approved' WHERE id=?", (sent.id,))
    try:
        cadences = await asyncio.to_thread(fx.mind._owner_cadences, [row])
        assert cadences == [outreach.Cadence(row["id"], "the garden", fx.now + DAY)]
    finally:
        fx.store._db.rollback()


@pytest.mark.parametrize("marker", ["requested", "owner_cadence"])
async def test_requested_queue_count_uses_transition_time_even_after_failure(make, marker):
    fx = make()
    create_cadence(fx, created=T0 - DAY)
    await fx.tick()
    sent, = fx.outreach_rows()
    context = dict(sent.context)
    context.pop("requested")
    context.pop("owner_cadence")
    context[marker] = True if marker == "requested" else "cadence-id"
    fx.store.update(sent.id, context=context, status="failed")
    with fx.store._db:
        fx.store._db.execute("UPDATE initiatives SET created_at=? WHERE id=?",
                             ((T0 - 40 * DAY).isoformat(), sent.id))
    assert fx.store.count_transitions("queued", fx.now - DAY, kind="message", recipient=OWNER) == 1
    state = (await fx.mind._gather(fx.now)).outreach
    assert state.queued_24h == 0


def test_distinct_topics_share_a_cadence_but_intervals_stay_separate(make):
    fx = make()
    rows = [cadence_row(), cadence_row(minutes=7 * 1440), cadence_row()]
    rows[1]["id"] = "cadence-2"
    rows[2]["id"] = "cadence-3"
    rows[2]["metadata"]["topic"] = "the workshop"
    rows[2]["made_at"] = (T0 + timedelta(minutes=2)).isoformat()
    daily, weekly = sorted(fx.mind._owner_cadences(rows), key=lambda item: item.due_at)
    assert daily == outreach.Cadence("cadence-3", "the workshop", T0 + DAY + timedelta(minutes=2))
    assert weekly == outreach.Cadence("cadence-2", "the garden", T0 + 7 * DAY, 7 * 1440)


@pytest.mark.parametrize("date", [(2026, 3, 7), (2026, 10, 31)])
def test_first_daily_phase_keeps_creation_hour_across_daylight_saving(make, date):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    fx = make()
    fx.mind.tz = ZoneInfo("America/New_York")
    created = datetime(*date, 9, tzinfo=fx.mind.tz)
    row = create_cadence(fx, created=created)
    fx.now = (created + DAY).replace(hour=0)
    cadence, = snapshot(fx, [row]).cadences
    assert cadence.due_at == created + DAY


@pytest.mark.parametrize("held", [False, True])
async def test_daily_slot_does_not_drift_with_tick_delivery_delay_or_hold(make, held):
    import random
    from zoneinfo import ZoneInfo

    fx = make()
    fx.mind.tz = ZoneInfo("America/New_York")
    slot = T0.astimezone(fx.mind.tz).replace(hour=9)
    row = create_cadence(fx, created=slot - 30 * DAY)
    rng = random.Random(7)
    for day in range(30):
        due = slot + day * DAY
        delay = timedelta(seconds=rng.uniform(0, 60))
        fx.now = due + delay
        if held and day == 3:
            fx.mind.mind_state.set(outreach.PAUSE_KEY, level=1,
                                   text=(due + timedelta(minutes=90)).isoformat(), now=fx.now)
            assert (await fx.mind.tick(force=True))["formed"] == []
            fx.now = due + timedelta(minutes=90) + delay
        assert len((await fx.mind.tick(force=True))["formed"]) == 1
        fx.shift(timedelta(seconds=30))
        payload, = await fx.mind.outbox_ready()
        fx.mind.outbox.sending(payload["id"], target="capture:owner")
        fx.mind.outbox.sent(payload["id"])
        elapsed = fx.now - due
        if held and day == 3:
            elapsed -= timedelta(minutes=90)
        assert timedelta(seconds=30) <= elapsed < timedelta(seconds=90)
        next_cadence, = snapshot(fx, [row]).cadences
        assert next_cadence.due_at == due + DAY
        assert (await fx.tick())["formed"] == []
    assert len(fx.outreach_rows()) == 30


@pytest.mark.parametrize("stated", [False, True])
async def test_restart_after_first_slot_keeps_same_days_due_time(make, stated):
    from colonymind.mind import Mind
    from zoneinfo import ZoneInfo

    fx = make()
    fx.mind.tz = ZoneInfo("America/New_York")
    slot = T0.astimezone(fx.mind.tz).replace(hour=9)
    made = slot - 30 * DAY
    row = fx.owner_commitment("Requested check-in", created=made - 3 * H if stated else made,
                              due=made if stated else None, **cadence_row()["metadata"])
    row = fx.commitments.get(row["id"])
    fx.now = slot - H
    assert snapshot(fx, [row]).cadences[0].due_at == slot
    fx.now = slot + timedelta(minutes=30)
    restarted = Mind(config={"quiet_hours": ""}, store=fx.store, state_dir=fx.store._state_dir,
                     owner_id=OWNER, commitments=fx.commitments, clock=lambda: fx.now, backups=False)
    restarted.tz = fx.mind.tz
    inputs = outreach.OutreachInputs(now=fx.now, owner_id=OWNER)
    inputs.cadences = restarted._owner_cadences([row], inputs=inputs)
    assert inputs.cadences[0].due_at == slot
    assert len(outreach.cadence_candidates(inputs)) == 1


async def test_different_topics_send_once_and_cancelling_one_stops_the_group(make):
    fx = make()
    first = create_cadence(fx, created=T0 - 10 * DAY)
    second = fx.owner_commitment("Requested check-in", created=T0 - 10 * DAY + timedelta(minutes=2),
        **{**cadence_row()["metadata"], "topic": "the workshop"})
    assert (await fx.tick())["formed"] == []
    fx.now = T0 + timedelta(minutes=2)
    await fx.tick()
    sent, = fx.outreach_rows()
    assert sent.source_id == second["id"] and sent.context["topic_slug"] == "the-workshop"
    fx.shift(DAY)
    await fx.tick()
    assert len(fx.outreach_rows()) == 2
    fx.commitments.resolve(second["id"], outcome="wont_do")
    assert fx.commitments.get(first["id"])["metadata"]["resolution"]["cancelled_by_commitment_id"] == second["id"]
    fx.shift(DAY)
    assert (await fx.tick())["formed"] == []
    fx.shift(DAY)
    assert (await fx.tick())["formed"] == []
    assert len(fx.outreach_rows()) == 2


async def test_delayed_attempt_skips_missed_slots_then_returns_to_phase(make):
    fx = make()
    row = create_cadence(fx, created=T0 - DAY)
    await fx.tick()
    fx.shift(10 * DAY + timedelta(minutes=90))
    assert len((await fx.tick())["formed"]) == 1
    assert len(fx.outreach_rows()) == 2
    assert (await fx.tick())["formed"] == []
    cadence, = snapshot(fx, [row]).cadences
    assert cadence.due_at == T0 + 11 * DAY
    fx.now = cadence.due_at
    assert len((await fx.tick())["formed"]) == 1
