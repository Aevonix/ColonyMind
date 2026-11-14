"""A person digest shows only while every claim it cites still stands (forget really forgets).

Both digest writers record the claims a digest was built from (the template the tick renders, the text the
night generates). Before the "About this person" section or the owner's inspect shows a digest, one query by
id checks that each cited claim is still live: not erased, not from an erased source, not retracted, and still
that person's own (no attribution correction moved its source). If one is not, the digest is withheld: its text is dropped from the contact's record and it waits for a rebuild, which
reads no previous text. The forget route clears a digest its erasure reached, and the night rebuilds a withheld
digest however long ago the person last talked. Everything runs over the real contact store, ledger, claim
projection, mind and routes; only the model is scripted.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from onekey import AUTH, install_key, keyed_host_app
from colonymind.api.routers import mind as mind_router
from colonymind.api.routers import people as people_mod
from colonymind.mind.consolidate import TASK_DIGEST
from test_memory_identity_acceptance import assemble, mind_call, person, told, world  # noqa: F401  (fixture)

ROOM, TEAM = "My office is room 4.", "My team is platform."


async def guest_of(fx) -> str:
    return await person(fx, "sms", "+15550100")


async def section(guest: str):
    """The "About this person" section of the guest's own turn, or None."""
    sections, _ = await assemble(guest, "Anything new?", session="sms-later", channel="sms:+15550100")
    return sections.get("colonymind-person")


async def inspected(guest: str):
    """The owner's ``inspect`` of the guest: the digest field and the text."""
    app = FastAPI()
    app.include_router(people_mod.router)
    install_key(app)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://people", headers=AUTH) as client:
        response = await client.get(f"/v1/mind/people/{guest}")
    assert response.status_code == 200, response.text
    value = response.json()
    return value["contact"].get("digest"), value["text"]


async def host_call(method: str, path: str, **kwargs):
    async with AsyncClient(transport=ASGITransport(app=keyed_host_app()), base_url="http://host",
                           headers=AUTH) as client:
        response = await client.request(method, path, **kwargs)
    assert response.status_code == 200, response.text
    return response.json()


async def forget(guest: str, *source_ids: str):
    from colonymind.memory import curated
    from colonymind.contacts.digest import source_ledger
    return curated.forget_sources(source_ledger(), guest, source_ids=list(source_ids))


async def withheld_count() -> int:
    return (await host_call("GET", "/v1/host/health"))["recall"]["digests_withheld"]


async def two_claims(fx, guest):
    """The guest states two things, in two sessions (erasing one turn's words then reaches no copy of them in
    the other); returns their claim ids, room first."""
    room = await told(fx, "sms-1", guest, "sms-session", ROOM, "room 4")
    team = await told(fx, "sms-2", guest, "sms-session-2", TEAM, "platform", predicate="team")
    return room, team


async def generated(fx, guest):
    """The night's digest of the guest: it cites both claims."""
    night = await mind_call("POST", "/v1/mind/consolidate")
    assert night["counts"].get("digests") == 1 and night["errors"] == []
    record = await fx.real_contacts.get(guest)
    assert record.digest_writer == "night" and "room 4" in record.digest and "platform" in record.digest
    return record


def digest_prompts(fx):
    return [messages[-1]["content"] for messages, context in fx.router.calls if context["task"] == TASK_DIGEST]


# -- 1, 2: forgetting a cited source clears the digest, whichever writer wrote it -----------------------------


# -- 3: a correction that retracts a cited claim withholds the digest until the night rebuilds it --------------

async def test_a_correction_that_retracts_a_cited_claim_withholds_the_digest_until_the_night_rebuilds_it(world):
    fx = world
    guest = await guest_of(fx)
    room, _ = await two_claims(fx, guest)
    await generated(fx, guest)
    fx.shift(minutes=5)
    await told(fx, "sms-3", guest, "sms-session", "Correction: my office is room 5, not room 4.", "room 5",
               operation="correct", match_prior=True)
    with closing(fx.ledger._connect()) as conn:
        assert conn.execute("SELECT retracted_by FROM source_claims WHERE id=?", (room,)).fetchone()[0]

    assert await section(guest) is None
    digest, text = await inspected(guest)
    assert digest is None and "room 4" not in text
    assert await withheld_count() == 1

    night = await mind_call("POST", "/v1/mind/consolidate")
    assert night["counts"].get("digests") == 1
    rebuilt = await section(guest)
    assert "room 5" in rebuilt and "room 4" not in rebuilt
    assert await withheld_count() == 0


# -- 4: an unrelated erasure leaves the digest in place ---------------------------------------------------------


# -- 5: a restore that re-creates claim ids withholds the digest; it never revives it --------------------------


# -- the store: a withhold takes only the digest that was checked -----------------------------------------------

async def test_a_withhold_takes_only_the_digest_that_was_checked_and_a_rebuild_ends_it(tmp_path):
    """A reader checks the digest it read; one written again meanwhile (the night racing a forget) is left for
    its own check, never withheld on the old one's lineage."""
    from colonymind.contacts.config import ContactsConfig
    from colonymind.contacts.store import SQLiteContactStore
    store = SQLiteContactStore(ContactsConfig(sqlite_path=str(tmp_path / "contacts.db")))
    await store.connect()
    try:
        sam = (await store.create(display_name="Sam")).contact_id
        await store.set_digest(sam, "Old.", ["claim:a"], writer="night")
        await store.set_digest(sam, "New.", ["claim:b"], writer="night")        # written after "Old." was read
        assert await store.withhold_digest(sam, sources=["claim:a"]) is False
        assert (await store.get(sam)).digest == "New." and await store.count_withheld_digests() == 0
        assert await store.withhold_digest(sam, sources=["claim:b"]) is True
        held = await store.get(sam)
        assert held.digest is None and held.digest_sources == [] and held.digest_writer == "night"
        assert held.digest_withheld_at and await store.count_withheld_digests() == 1
        assert [c.contact_id for c in await store.withheld_digests()] == [sam]
        await store.set_digest(sam, "Rebuilt.", ["claim:c"], writer="night")
        rebuilt = await store.get(sam)
        assert rebuilt.digest == "Rebuilt." and rebuilt.digest_withheld_at is None
        assert await store.count_withheld_digests() == 0 and await store.withheld_digests() == []
    finally:
        await store.close()


# -- 6: a withheld digest is rebuilt however long the person has been silent -----------------------------------


# -- 7: the upgrade gives template digests lineage ------------------------------------------------------------

@pytest.mark.parametrize("mind", ["on", "switched_off", "disabled_in_config"])
async def test_the_upgrade_gives_template_digests_lineage(world, tmp_path, mind):
    """A store as 1.11 left it: a template digest marked ``["template"]`` (no lineage), and a generated one that
    cites its claims. The upgrade's store migration keeps the generated one and drops the unverifiable template
    text; the next tick renders it again from the live claims (no model call), now with its lineage, whether
    the mind is on, switched off by the owner, or disabled in the configuration (its tick then does upkeep
    only)."""
    from colonymind.init import pending_store_migrations, run_store_migrations
    fx = world
    if mind == "switched_off":
        await mind_call("POST", "/v1/mind/off", json={"reason": "owner away"})
    elif mind == "disabled_in_config":
        fx.config = {**fx.config, "enabled": False}
        fx.mind = fx.build()
        mind_router.set_mind(fx.mind)
    assert fx.mind.enabled is (mind == "on")
    guest = await guest_of(fx)
    other = (await fx.real_contacts.create(display_name="Robin", trust_tier="regular")).contact_id
    room = await told(fx, "sms-1", guest, "sms-session", ROOM, "room 4")
    desk = await told(fx, "robin-1", other, "robin-session", "My desk is 12.", "12", predicate="desk")

    store = fx.real_contacts
    db = store._require_db()
    await db.execute("DROP INDEX IF EXISTS idx_contacts_digest_withheld")
    for column in ("digest_withheld_at", "digest_writer"):
        await db.execute(f"ALTER TABLE contacts DROP COLUMN {column}")
    await db.execute("DELETE FROM schema_version WHERE version='007'")
    # The guest last talked three days ago: the daily template pass would not render them again.
    await db.execute("UPDATE contacts SET digest=?, digest_sources=?, last_interaction_at=? WHERE contact_id=?",
                     ("Friend. Notes: office_location: room 4.", '["template"]',
                      (fx.now - timedelta(days=3)).isoformat(), guest))
    await db.execute("UPDATE contacts SET digest=?, digest_sources=? WHERE contact_id=?",
                     ("Robin told me their desk is 12.", f'["{desk}"]', other))
    await db.commit()
    await store.close()

    assert pending_store_migrations(tmp_path) == ["contacts.db:007"]
    assert run_store_migrations(tmp_path) == ["contacts.db:007"]           # the upgrade's migration step
    await store.connect()
    kept = await store.get(other)
    assert kept.digest_writer == "night" and kept.digest == "Robin told me their desk is 12."
    legacy = await store.get(guest)
    assert legacy.digest is None and legacy.digest_writer == "template" and legacy.digest_withheld_at
    assert await withheld_count() == 1

    fx.router.calls.clear()
    fx.now = max(fx.now, datetime.now(timezone.utc)) + timedelta(seconds=32)
    ticked = await mind_call("POST", "/v1/mind/tick")
    assert ticked["skipped"] == (None if mind == "on" else "off")
    renewed = await store.get(guest)
    assert renewed.digest_sources == [room] and renewed.digest_withheld_at is None and "room 4" in renewed.digest
    assert TASK_DIGEST not in fx.router.tasks()                             # rendered, not generated
    assert "room 4" in await section(guest)
    assert await withheld_count() == 0


# -- 8: a rebuild after a cited claim went reads no previous digest text ----------------------------------------


# -- a forget while the night writes: nothing is stored on the erased claim ------------------------------------


# -- a cited source moved to another person: the first person's digest never shows the other's words ----------

async def moved_and_read_again(fx, guest, *, route: bool):
    """Robin is the one who said ``ROOM`` (turn ``sms-1``): an attribution correction moves that source from the
    guest to Robin, and the claim worker reads it again under Robin, which gives the claim its old id. ``route``:
    through the correction's reconciliation (``social_state.reconcile_identity_sources``), else the ledger's
    move alone. Nothing reads the guest's digest in between. Returns Robin's contact id."""
    from colonymind.api.routers import social_state
    from colonymind.beliefs.source_projection import SourceClaimProjection
    from colonymind.turns.source_attribution import correct
    from test_source_claim_projection import Model, claim
    robin = (await fx.real_contacts.create(display_name="Robin", trust_tier="regular")).contact_id
    moved = dict(operation_id="moved-1", performed_by="owner-test", evidence_refs=["source:owner-confirmation"])
    if route:
        await fx.real_contacts.add_handle(guest, "email", "desk@example.test", verified=True)
        operation = await fx.real_contacts.correct_handle_identity(
            gateway="email", address="desk@example.test", expected_contact_id=guest, contact_id=robin,
            affected_source_ids=["sms-1"], **moved)
        await social_state.reconcile_identity_sources(fx.real_contacts, fx.ledger, operation)
    else:
        correct(fx.ledger, old_contact_id=guest, contact_id=robin, source_ids=["sms-1"], **moved)
    assert await SourceClaimProjection(fx.ledger).process_one(Model({ROOM: claim(
        ROOM, "room 4", subject="I", predicate="office_location", memory_kind="personal_context")}))
    return robin


async def test_a_cited_claim_moved_to_another_person_and_extracted_again_is_not_the_first_persons(world):
    """The ledger's move alone (nothing withheld the digest at the correction): the re-extracted claim has the
    id the guest's digest cites and is live, but it is Robin's, so the guest's digest rests on it no more."""
    fx = world
    guest = await guest_of(fx)
    room, _ = await two_claims(fx, guest)
    await generated(fx, guest)
    robin = await moved_and_read_again(fx, guest, route=False)
    with closing(fx.ledger._connect()) as conn:
        whose = conn.execute("SELECT s.contact_id FROM source_claims c JOIN turn_sources s ON s.turn_id=c.turn_id "
                             "WHERE c.id=? AND c.retracted_by IS NULL", (room,)).fetchone()
    assert whose is not None and whose[0] == robin                      # the same id, live, now Robin's

    assert await section(guest) is None
    digest, text = await inspected(guest)
    assert digest is None and "room 4" not in text
    assert (await fx.real_contacts.get(guest)).digest is None


async def test_an_attribution_correction_withholds_the_digest_that_cites_a_moved_source(world):
    """Through the correction's reconciliation: the guest's digest is withheld there, before anything reads it,
    and the night rebuilds it from what the guest still said, with no previous text."""
    fx = world
    guest = await guest_of(fx)
    _, team = await two_claims(fx, guest)
    old = (await generated(fx, guest)).digest
    await moved_and_read_again(fx, guest, route=True)
    stored = await fx.real_contacts.get(guest)
    assert stored.digest is None and stored.digest_sources == [] and stored.digest_withheld_at
    assert await withheld_count() == 1
    assert await section(guest) is None

    calls = len(digest_prompts(fx))
    night = await mind_call("POST", "/v1/mind/consolidate")
    assert night["counts"].get("digests") == 2 and len(digest_prompts(fx)) == calls + 2
    prompt = next(prompt for prompt in digest_prompts(fx)[calls:] if f"Person: {guest}" in prompt)
    assert old not in prompt and "room 4" not in prompt and "Previous digest:\n(none)" in prompt
    rebuilt = await fx.real_contacts.get(guest)
    assert rebuilt.digest_sources == [team] and "room 4" not in rebuilt.digest and "platform" in rebuilt.digest
    assert await withheld_count() == 0


# -- every contact response carries only a digest that stands --------------------------------------------------

async def contact_digests(guest) -> dict:
    """The digest each host contact route returns for the guest: get, list, resolve by handle, the timezone
    write and an introduction through the guest's known handle (each serializes the stored contact)."""
    async with AsyncClient(transport=ASGITransport(app=keyed_host_app()), base_url="http://host",
                           headers=AUTH) as client:
        responses = {
            "get": await client.get(f"/v1/host/contacts/{guest}"),
            "list": await client.get("/v1/host/contacts"),
            "resolve": await client.get("/v1/host/contacts/resolve",
                                        params={"gateway": "sms", "address": "+15550100"}),
            "timezone": await client.post(f"/v1/host/contacts/{guest}/timezone", json={"timezone": "UTC"}),
            "intro": await client.post("/v1/host/contacts/intro",
                                       json={"name": "Friend", "gateway": "sms", "address": "+15550100"}),
        }
    rows = {}
    for route, response in responses.items():
        assert response.status_code in {200, 201}, (route, response.text)
        value = response.json()
        if route == "list":
            value = next(row for row in value["contacts"] if row["contact_id"] == guest)
        elif route == "intro":
            assert value["created"] is False
            value = value["contact"]
        assert value["contact_id"] == guest
        rows[route] = value
    return rows


def assert_no_digest(rows: dict, *words: str) -> None:
    for route, row in rows.items():
        assert row["digest"] is None and row["digest_sources"] == [], route
        assert not any(word in repr(row) for word in words), route


async def test_every_contact_route_shows_a_digest_only_while_its_claims_stand(world):
    fx = world
    guest = await guest_of(fx)
    room, team = await two_claims(fx, guest)
    await generated(fx, guest)
    for route, row in (await contact_digests(guest)).items():
        assert "room 4" in row["digest"] and sorted(row["digest_sources"]) == sorted([room, team]), route

    fx.shift(minutes=5)
    await told(fx, "sms-3", guest, "sms-session", "Correction: my office is room 5, not room 4.", "room 5",
               operation="correct", match_prior=True)
    # Nothing read the digest since the correction: the first contact route to serialize it checks it.
    assert_no_digest(await contact_digests(guest), "room 4", "platform")
    assert (await fx.real_contacts.get(guest)).digest is None and await withheld_count() == 1


async def test_a_contact_route_withholds_nothing_when_the_lineage_cannot_be_read(world, monkeypatch):
    """With no ledger to check the cited claims against, a digest that cites claims is not returned, and it
    stays stored (nothing shows it was wrong); a contact without a digest is served as before."""
    from colonymind.contacts import digest as digest_mod
    fx = world
    guest = await guest_of(fx)
    await two_claims(fx, guest)
    record = await generated(fx, guest)
    monkeypatch.setattr(digest_mod, "source_ledger", lambda: None)
    assert_no_digest(await contact_digests(guest), "room 4", "platform")
    kept = await fx.real_contacts.get(guest)
    assert kept.digest == record.digest and kept.digest_withheld_at is None


@pytest.mark.parametrize("how", ['retracted'])
async def test_after_a_cited_claim_goes_the_rebuild_prompt_carries_no_previous_digest(world, how):
    fx = world
    guest = await guest_of(fx)
    await two_claims(fx, guest)
    old = (await generated(fx, guest)).digest
    fx.shift(minutes=5)
    await told(fx, 'sms-3', guest, 'sms-session', 'Correction: my office is room 5, not room 4.', 'room 5', operation='correct', match_prior=True)

    calls = len(digest_prompts(fx))
    await mind_call("POST", "/v1/mind/consolidate")
    assert len(digest_prompts(fx)) == calls + 1                             # rebuilt from what stands
    prompt = digest_prompts(fx)[-1]
    assert old not in prompt and "Previous digest:\n(none)" in prompt
    if how == "erased":
        assert "room 4" not in prompt


async def test_forgetting_a_source_a_generated_digest_cites_clears_it_from_the_section_and_inspect(world):
    fx = world
    guest = await guest_of(fx)
    room, team = await two_claims(fx, guest)
    record = await generated(fx, guest)
    assert sorted(record.digest_sources) == sorted([room, team])
    assert "room 4" in await section(guest) and "room 4" in (await inspected(guest))[0]

    erased = await forget(guest, "sms-1")
    assert erased["source_erased"]
    assert erased["scope"] == "curated_source_tombstone"
    assert await section(guest) is None
    digest, text = await inspected(guest)
    assert digest is None and "room 4" not in text and "platform" not in text
    stored = await fx.real_contacts.get(guest)
    assert stored.digest is None and stored.digest_sources == [] and stored.digest_withheld_at


async def test_forgetting_a_source_a_template_digest_cites_clears_it_and_the_tick_renders_it_again(world):
    fx = world
    guest = await guest_of(fx)
    room, team = await two_claims(fx, guest)
    fx.now = max(fx.now, datetime.now(timezone.utc)) + timedelta(seconds=32)
    await mind_call("POST", "/v1/mind/tick")                       # the day's template digest
    template = await fx.real_contacts.get(guest)
    # The template records the claims whose lines it carries, and which writer it is.
    assert template.digest_writer == "template" and sorted(template.digest_sources) == sorted([room, team])
    assert "room 4" in template.digest and "room 4" in await section(guest)

    erased = await forget(guest, "sms-1")
    assert erased["source_erased"]
    assert await section(guest) is None
    digest, text = await inspected(guest)
    assert digest is None and "room 4" not in text

    fx.now = max(fx.now, datetime.now(timezone.utc)) + timedelta(seconds=32)
    await mind_call("POST", "/v1/mind/tick")                       # rendered again from what still stands
    again = await fx.real_contacts.get(guest)
    assert again.digest_writer == "template" and again.digest_sources == [team] and again.digest_withheld_at is None
    assert "platform" in again.digest and "room 4" not in again.digest
    assert "platform" in await section(guest)


async def test_an_erasure_no_cited_claim_rests_on_leaves_the_digest_in_place(world):
    fx = world
    guest = await guest_of(fx)
    await two_claims(fx, guest)
    fx.turn("sms-chat", guest, "sms-other", "Thanks, talk soon.", "Talk soon.")   # a turn no claim came from
    record = await generated(fx, guest)

    erased = await forget(guest, "sms-chat")
    assert erased["source_erased"]
    kept = await fx.real_contacts.get(guest)
    assert kept.digest == record.digest and kept.digest_sources == record.digest_sources
    assert kept.digest_withheld_at is None
    assert "room 4" in await section(guest) and "room 4" in (await inspected(guest))[0]


async def test_a_contact_silent_for_thirty_days_gets_the_rebuild(world):
    fx = world
    guest = await guest_of(fx)
    await two_claims(fx, guest)
    await generated(fx, guest)
    fx.shift(days=30)
    assert (await forget(guest, "sms-1"))["source_erased"]
    assert await section(guest) is None
    assert await withheld_count() == 1
    calls = len(digest_prompts(fx))

    night = await mind_call("POST", "/v1/mind/consolidate")
    assert night["counts"].get("digests") == 1 and len(digest_prompts(fx)) == calls + 1
    rebuilt = await fx.real_contacts.get(guest)
    assert datetime.fromisoformat(rebuilt.last_interaction_at) <= fx.now - timedelta(days=29)
    assert "platform" in rebuilt.digest and "room 4" not in rebuilt.digest and rebuilt.digest_withheld_at is None
    assert "platform" in await section(guest)
    assert await withheld_count() == 0


async def test_with_the_mind_off_a_withheld_digest_is_rendered_again_without_the_model(world):
    """The night cannot run while the mind is off: the tick's upkeep renders the withheld generated digest as a
    template from what still stands (no model call, nothing sent), instead of leaving the person without one."""
    fx = world
    guest = await guest_of(fx)
    _, team = await two_claims(fx, guest)
    await generated(fx, guest)
    await mind_call("POST", "/v1/mind/off", json={"reason": "owner away"})
    assert (await forget(guest, "sms-1"))["source_erased"]
    assert await section(guest) is None
    assert await withheld_count() == 1

    fx.router.calls.clear()
    fx.now = max(fx.now, datetime.now(timezone.utc)) + timedelta(seconds=32)
    ticked = await mind_call("POST", "/v1/mind/tick")
    assert ticked["skipped"] == "off" and ticked["digests"] == 1 and fx.router.calls == []
    again = await fx.real_contacts.get(guest)
    assert again.digest_writer == "template" and again.digest_sources == [team] and again.digest_withheld_at is None
    assert "platform" in again.digest and "room 4" not in again.digest
    assert await withheld_count() == 0


async def test_a_digest_whose_forget_cleanup_is_pending_never_leaves_through_a_contact_route(world, monkeypatch):
    """The forget route cannot withhold the digest (the contact store refuses the write): it reports
    ``pending`` and the text stays stored, but no contact route, section or inspect returns it."""
    fx = world
    guest = await guest_of(fx)
    await two_claims(fx, guest)
    await generated(fx, guest)

    async def refused(*args, **kwargs):
        raise RuntimeError("contact store unavailable")
    monkeypatch.setattr(fx.real_contacts, "withhold_digest", refused)
    assert (await forget(guest, "sms-1"))["source_erased"]
    assert "room 4" in (await fx.real_contacts.get(guest)).digest              # still stored

    assert_no_digest(await contact_digests(guest), "room 4", "platform")
    assert await section(guest) is None
    digest, text = await inspected(guest)
    assert digest is None and "room 4" not in text


from colonymind.memory import curated
async def test_a_restore_that_recreates_a_cited_claim_never_revives_its_digest(world, tmp_path):
    fx = world
    guest = await guest_of(fx)
    room, _ = await two_claims(fx, guest)
    await generated(fx, guest)
    ledger_path = fx.ledger.db_path
    snapshot = tmp_path / 'ledger-before-erasure.db'
    with closing(sqlite3.connect(ledger_path)) as source, closing(sqlite3.connect(snapshot)) as target:
        source.backup(target)
    with closing(fx.ledger._connect()) as conn:
        row = dict(conn.execute('SELECT * FROM source_claims WHERE id=?', (room,)).fetchone())
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(fx.ledger)
    curated.forget_sources(fx.ledger, guest, source_ids=['sms-1'])
    assert historical_bytes(fx.ledger) == _native_before_withdrawal
    with closing(fx.ledger._connect()) as conn, conn:
        assert dict(conn.execute('SELECT * FROM source_claims WHERE id=?', (row['id'],)).fetchone()) == row
    assert await section(guest) is None
    with closing(sqlite3.connect(snapshot)) as source, closing(sqlite3.connect(ledger_path)) as target:
        source.backup(target)
    with closing(fx.ledger._connect()) as conn:
        assert conn.execute('SELECT retracted_by FROM source_claims WHERE id=?', (room,)).fetchone()[0] is None
    assert await section(guest) is None
    digest, text = await inspected(guest)
    assert digest is None and 'room 4' not in text
    assert (await fx.real_contacts.get(guest)).digest is None

async def test_a_claim_erased_while_the_night_writes_the_digest_is_never_stored(world):
    from test_memory_identity_acceptance import digest_of_values
    fx = world
    guest = await guest_of(fx)
    await two_claims(fx, guest)

    def racing(messages, context):
        answer = digest_of_values(messages, context)
        from test_curated_memory import historical_bytes
        _native_before_withdrawal = historical_bytes(fx.ledger)
        curated.forget_sources(fx.ledger, guest, source_ids=['sms-1'])
        assert historical_bytes(fx.ledger) == _native_before_withdrawal
        return answer
    fx.router.answers[TASK_DIGEST] = racing
    night = await mind_call('POST', '/v1/mind/consolidate')
    assert night['counts'].get('digests_stale') == 1 and night['counts'].get('digests', 0) == 0
    assert (await fx.real_contacts.get(guest)).digest is None

async def test_the_night_drops_the_text_of_a_digest_whose_claim_went_even_when_it_cannot_rebuild(world):
    fx = world
    guest = await guest_of(fx)
    await two_claims(fx, guest)
    await generated(fx, guest)
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(fx.ledger)
    curated.forget_sources(fx.ledger, guest, source_ids=['sms-1'])
    assert historical_bytes(fx.ledger) == _native_before_withdrawal
    fx.router.answers[TASK_DIGEST] = {'digest': '', 'sources': []}
    night = await mind_call('POST', '/v1/mind/consolidate')
    assert await section(guest) is None
    stored = await fx.real_contacts.get(guest)
    assert 'room 4' in stored.digest
    assert await section(guest) is None
