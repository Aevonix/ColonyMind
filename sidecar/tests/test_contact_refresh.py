"""B5: changed contacts refresh before night; unchanged contacts never rebuild."""

from contextlib import closing
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from colonymind.contacts.digest import checked_digest, not_live
from colonymind.contacts.refresh import (DEBOUNCE_SECONDS, REFRESHES_PER_HOUR, invalidate_pending,
                                       reserve_refresh)
from colonymind.mind.consolidate import TASK_DIGEST
from test_memory_identity_acceptance import person, told, world  # noqa: F401
from test_digest_grounding import section


async def prepare(fx):
    guest = await person(fx, "sms", "+15550100")
    old = await told(fx, "old", guest, "first", "My office is room 4.", "room 4")
    # Seed a generated digest, acknowledging the insertion as an existing build would.
    await invalidate_pending(fx.ledger, fx.contacts, owner_id=fx.owner_id)
    await fx.contacts.set_digest(guest, "My office is room 4.", [old], writer="night")
    other = await fx.contacts.create(display_name="Other")
    await fx.contacts.set_digest(other.contact_id, "Unchanged contact.", [], writer="template")
    return guest, old, other.contact_id


def advance(fx, seconds=DEBOUNCE_SECONDS + 2):
    fx.now = datetime.now(timezone.utc) + timedelta(seconds=seconds)


@pytest.mark.parametrize("operation", ["change", "correct"])
async def test_replacement_reaches_the_next_summary_read_without_night(world, operation):
    fx = world
    guest, old, other = await prepare(fx)
    fx.shift(minutes=1)
    date = fx.now.isoformat()
    text = (f"My office is now room 5 since {date}, instead of room 4." if operation == "change"
            else "Correction: my office is room 5, not room 4.")
    new = await told(fx, "new", guest, "second", text, "room 5", operation=operation, match_prior=True,
                     valid_from_text=date if operation == "change" else None)
    with closing(fx.ledger._connect()) as conn:
        field = "superseded_by" if operation == "change" else "retracted_by"
        assert conn.execute(f"SELECT {field} FROM source_claims WHERE id=?", (old,)).fetchone()[0] == new
    # Lineage validation stops serving the old value even before the worker runs.
    assert (await checked_digest(await fx.contacts.get(guest), store=fx.contacts, ledger=fx.ledger))[0] == ""
    writer = AsyncMock(wraps=fx.contacts.set_digest)
    fx.contacts.set_digest = writer
    advance(fx)
    assert await fx.mind._digest_upkeep(fx.now) == 1
    summary = await section(guest)
    assert "room 5" in summary and "room 4" not in summary
    assert (await fx.contacts.get(guest)).digest_sources == [new]
    assert (await fx.contacts.get(other)).digest == "Unchanged contact."
    assert writer.await_count == 1 and TASK_DIGEST not in fx.router.tasks()
    # Neither later background passes nor daily/nightly work rebuild either valid summary.
    assert await fx.mind._digest_upkeep(fx.now + timedelta(days=1)) is None
    assert await fx.mind._digests(fx.now + timedelta(days=1)) == 0
    assert await fx.mind.consolidation._digest_candidates(fx.now) == []
    assert writer.await_count == 1


async def test_deleted_fact_is_removed_with_mind_off(world):
    fx = world
    guest, old, _ = await prepare(fx)
    with closing(fx.ledger._connect()) as conn, conn:
        conn.execute("DELETE FROM source_claims WHERE id=?", (old,))
    fx.mind.policy.enabled = False
    assert not fx.mind.enabled
    advance(fx)
    assert await fx.mind._digest_upkeep(fx.now) == 1
    digest = (await fx.contacts.get(guest)).digest
    assert digest and "room 4" not in digest


async def test_quiet_period_restarts_on_each_change(world):
    fx = world
    guest, old, _ = await prepare(fx)
    await fx.contacts.mark_digest_stale(guest, changed_at=datetime.now(timezone.utc).isoformat())
    assert await fx.mind._digest_upkeep(datetime.now(timezone.utc)) is None
    with closing(fx.ledger._connect()) as conn, conn:
        conn.execute("UPDATE source_claims SET retracted_by='withdrawn' WHERE id=?", (old,))
    now = datetime.now(timezone.utc)
    assert await fx.mind._digest_upkeep(now) is None
    assert (await fx.contacts.get(guest)).digest is None
    assert await fx.mind._digest_upkeep(now + timedelta(seconds=DEBOUNCE_SECONDS + 2)) == 1
    assert await fx.mind._digest_upkeep(now + timedelta(minutes=1)) is None


async def test_refresh_budget_survives_new_connection_and_leaves_work_stale(world):
    fx = world
    guest, _, _ = await prepare(fx)
    advance(fx)
    for _ in range(REFRESHES_PER_HOUR):
        assert reserve_refresh(fx.ledger, fx.now.timestamp())
    await fx.contacts.mark_digest_stale(guest, changed_at=(fx.now - timedelta(minutes=1)).isoformat())
    assert await fx.mind._digest_upkeep(fx.now) is None
    assert (await fx.contacts.get(guest)).digest_withheld_at
    # A rebuilt Mind opens its own connections, and cannot reset the rolling budget.
    fx.restart()
    assert await fx.mind._digest_upkeep(fx.now + timedelta(minutes=1)) is None
    assert await fx.mind._digest_upkeep(fx.now + timedelta(hours=1)) == 1


async def test_failed_invalidation_keeps_its_durable_job(world, monkeypatch):
    fx = world
    guest, old, _ = await prepare(fx)
    with closing(fx.ledger._connect()) as conn, conn:
        conn.execute("UPDATE source_claims SET retracted_by='withdrawn' WHERE id=?", (old,))
    original = fx.contacts.mark_digest_stale
    monkeypatch.setattr(fx.contacts, "mark_digest_stale", AsyncMock(side_effect=RuntimeError("store unavailable")))
    with pytest.raises(RuntimeError, match="store unavailable"):
        await invalidate_pending(fx.ledger, fx.contacts, owner_id=fx.owner_id)
    with closing(fx.ledger._connect()) as conn:
        assert conn.execute("SELECT contact_id FROM contact_digest_jobs").fetchone()[0] == guest
    advance(fx)
    assert await fx.mind._digest_upkeep(fx.now) is None
    monkeypatch.setattr(fx.contacts, "mark_digest_stale", original)
    advance(fx)
    assert await fx.mind._digest_upkeep(fx.now) == 1


async def test_failed_refresh_counts_against_budget_and_keeps_the_stale_marker(world, monkeypatch):
    fx = world
    guest, _, _ = await prepare(fx)
    advance(fx)
    await fx.contacts.mark_digest_stale(guest, changed_at=(fx.now - timedelta(minutes=1)).isoformat())
    monkeypatch.setattr(fx.contacts, "set_digest", AsyncMock(side_effect=RuntimeError("write unavailable")))
    assert await fx.mind._digest_upkeep(fx.now) is None
    assert (await fx.contacts.get(guest)).digest_withheld_at
    with closing(fx.ledger._connect()) as conn:
        assert conn.execute("SELECT count(*) FROM contact_digest_refreshes").fetchone()[0] == 1


async def test_a_fact_change_during_rendering_leaves_another_rebuild_pending(world, monkeypatch):
    fx = world
    guest, old, _ = await prepare(fx)
    await fx.contacts.mark_digest_stale(guest, changed_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat())
    original = fx.contacts.set_digest

    async def concurrent_change(*args, **kwargs):
        with closing(fx.ledger._connect()) as conn, conn:
            conn.execute("UPDATE source_claims SET retracted_by='withdrawn' WHERE id=?", (old,))
        await original(*args, **kwargs)

    monkeypatch.setattr(fx.contacts, "set_digest", concurrent_change)
    advance(fx)
    await fx.mind._digest_upkeep(fx.now)
    assert (await fx.contacts.get(guest)).digest is None
    assert (await fx.contacts.get(guest)).digest_withheld_at
    monkeypatch.setattr(fx.contacts, "set_digest", original)
    assert await fx.mind._digest_upkeep(fx.now + timedelta(minutes=1)) == 1
    assert "room 4" not in (await fx.contacts.get(guest)).digest


async def test_unchanged_witness_updates_do_not_queue_refresh(world):
    fx = world
    guest, old, _ = await prepare(fx)
    with closing(fx.ledger._connect()) as conn, conn:
        conn.execute("UPDATE source_claims SET witnesses=witnesses+1 WHERE id=?", (old,))
        conn.execute("UPDATE source_claims SET valid_to=valid_to WHERE id=?", (old,))
        assert conn.execute("SELECT count(*) FROM contact_digest_jobs").fetchone()[0] == 0
    advance(fx)
    assert await fx.mind._digest_upkeep(fx.now) is None
    assert (await fx.contacts.get(guest)).digest_writer == "night"


async def test_expiration_is_withheld_and_background_refreshed(world):
    fx = world
    guest, old, _ = await prepare(fx)
    with closing(fx.ledger._connect()) as conn, conn:
        conn.execute("UPDATE source_claims SET valid_to=? WHERE id=?",
                     ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), old))
        # Isolate elapsed-time validation from the fact-update trigger.
        conn.execute("DELETE FROM contact_digest_jobs")
        assert not_live(conn, [old], contact_id=guest) == [old]
    assert await fx.mind._digest_upkeep(datetime.now(timezone.utc)) is None
    assert (await fx.contacts.get(guest)).digest_withheld_at
    advance(fx)
    assert await fx.mind._digest_upkeep(fx.now) == 1
    assert "room 4" not in (await fx.contacts.get(guest)).digest


async def test_nightly_selects_only_stale_contacts(world):
    fx = world
    guest, _, other = await prepare(fx)
    assert await fx.mind.consolidation._digest_candidates(fx.now) == []
    await fx.contacts.mark_digest_stale(guest, changed_at=fx.now.isoformat())
    candidates = await fx.mind.consolidation._digest_candidates(fx.now)
    assert [c.contact_id for c in candidates] == [guest]
    assert other not in [c.contact_id for c in candidates]


@pytest.mark.parametrize("failing_count", [60, 240])
async def test_failures_back_off_and_a_healthy_contact_refreshes_within_a_bounded_time(
        world, monkeypatch, failing_count):
    fx = world
    guest, _, other = await prepare(fx)
    advance(fx)
    started = fx.now
    failures = set()
    for index in range(failing_count):
        contact = await fx.contacts.create(display_name=f"Unavailable {index}")
        failures.add(contact.contact_id)
        await fx.contacts.mark_digest_stale(
            contact.contact_id, changed_at=(started - timedelta(minutes=2, seconds=failing_count - index)).isoformat())
    await fx.contacts.mark_digest_stale(guest, changed_at=(started - timedelta(minutes=1)).isoformat())
    attempts = []
    original = fx.contacts.set_digest

    async def write(contact_id, *args, **kwargs):
        attempts.append((contact_id, fx.now))
        if contact_id in failures:
            raise RuntimeError("write unavailable")
        await original(contact_id, *args, **kwargs)

    monkeypatch.setattr(fx.contacts, "set_digest", write)
    bound = failing_count // REFRESHES_PER_HOUR
    for hour in range(bound + 1):
        fx.now = started + timedelta(hours=hour)
        fx.restart()  # Both retry ordering and backoff must survive a rebuilt worker.
        await fx.mind._digest_upkeep(fx.now)
        if hour == 0:
            assert len(attempts) == REFRESHES_PER_HOUR and guest not in [cid for cid, _ in attempts]
            before = len(attempts)
            fx.now += timedelta(minutes=1)
            assert await fx.mind._digest_upkeep(fx.now) is None and len(attempts) == before
    healthy = await fx.contacts.get(guest)
    assert healthy.digest_withheld_at is None and "room 4" in healthy.digest
    assert [stamp for cid, stamp in attempts if cid == guest] == [started + timedelta(hours=bound)]
    assert (await fx.contacts.get(other)).digest == "Unchanged contact."
    stale = [await fx.contacts.get(cid) for cid in failures]
    assert all(contact.digest_withheld_at for contact in stale)
    first = attempts[0][0]
    with closing(fx.ledger._connect()) as conn:
        last, count, retry_at = conn.execute(
            "SELECT last_attempt, failures, retry_at FROM contact_digest_retries WHERE contact_id = ?",
            (first,)).fetchone()
    assert count == 2 and retry_at - last == 2 * 3600
    before = sum(cid == first for cid, _ in attempts)
    fx.now = datetime.fromtimestamp(retry_at - 1, timezone.utc)
    fx.restart()
    await fx.mind._digest_upkeep(fx.now)
    assert sum(cid == first for cid, _ in attempts) == before
    assert TASK_DIGEST not in fx.router.tasks()


async def test_nightly_failures_rotate_past_sixty_older_contacts(world, monkeypatch):
    fx = world
    guest, _, _ = await prepare(fx)
    advance(fx)
    failures = set()
    for index in range(60):
        contact = await fx.contacts.create(display_name=f"Unavailable {index}")
        failures.add(contact.contact_id)
        await fx.contacts.mark_digest_stale(
            contact.contact_id, changed_at=(fx.now - timedelta(minutes=2, seconds=60 - index)).isoformat())
    await fx.contacts.mark_digest_stale(guest, changed_at=(fx.now - timedelta(minutes=1)).isoformat())
    attempted = []
    original = fx.contacts.set_digest

    async def write(contact_id, *args, **kwargs):
        attempted.append(contact_id)
        if contact_id in failures:
            raise RuntimeError("write unavailable")
        await original(contact_id, *args, **kwargs)

    monkeypatch.setattr(fx.contacts, "set_digest", write)
    for _ in range(11):
        fx.restart()
        await fx.mind.consolidate()
        fx.shift(days=1)
    assert set(attempted[:60]) == failures and attempted[60] == guest
    healthy = await fx.contacts.get(guest)
    assert healthy.digest_writer == "night" and "room 4" in healthy.digest and healthy.digest_withheld_at is None
    assert TASK_DIGEST in fx.router.tasks()
