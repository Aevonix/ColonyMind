"""The status block: the agent's posture this turn, generated from the running mind (``colonymind.mind.status``).

One render per state; owner turns only; never dropped by the per-turn selection; each line fails to
``<field>: unknown``; no section without a mind; the same bytes while nothing changed (so the memory
provider's unchanged-section line holds); the cap; no dash glyphs; the ``context.status`` switch.
"""

from __future__ import annotations

import asyncio
import inspect
import sqlite3
import threading
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from host_policy_fixtures import STATUS_REPLACEMENTS
from httpx import ASGITransport, AsyncClient

from colonymind import config
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import host
from colonymind.api.routers import mind as mind_router
from colonymind.api.schemas.host import ContextSection
from colonymind.config import ConfigError, apply_environment, load_config
from colonymind.memory import compass
from colonymind.mind import status
from test_mind_loop import CONTACT, KEY, AUTH, OWNER, Fixture, pinned_clock  # noqa: F401  (autouse clock)
from test_turn_source_evidence import source_app  # noqa: F401  (pytest fixture)

DASHES = (chr(0x2014), chr(0x2013))     # the em and en dash


STANDARD = ("Autonomy: standard (internal work and messages to the owner go ahead, contacts by their permission;"
            " the rest asks). Mind: on.")


OWNER_USED = ("Messages to the owner: today's budget is used, so reminders and heads-ups wait; the digest, its notices"
              " and owed reports do not count against it.")


def make(tmp_path, monkeypatch, *, autonomy="standard", **config_values):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    return Fixture(tmp_path, autonomy=autonomy, config=config_values)


@pytest.fixture
def fx(tmp_path, monkeypatch):
    fixture = make(tmp_path, monkeypatch)
    yield fixture
    fixture.store.close()


def lines(fx, **kwargs):
    return status.build(fx.mind, **kwargs).split("\n")


def trip_breaker(fx):
    """Three counted failures of the agent's own work: the internal class is demoted."""
    for number in range(3):
        row, _ = fx.store.create_intention(
            kind="task", type="research", title=f"research {number}", drive="curiosity", cls="internal",
            decision="act", decision_reason="test", status="approved", dedup_key=f"research:{number}",
            context={"body": f"Research topic {number}."}, hermes_kind="none", created_at=fx.now)
        fx.mind.bound(row.id, f"kanban:t{number}")
        fx.mind.outcomes.record(row.id, status="blocked", outcome="failed", final=True, hermes_ref=f"kanban:t{number}",
                                error="worker crashed", run={"outcome": "crashed", "status": "crashed", "id": number})
        fx.shift(hours=1)


# -- one render per state -----------------------------------------------------------------------------------

@pytest.mark.parametrize("level, meaning", [
    ("suggest", "internal work and updates the owner asked for go ahead; the rest waits for the owner"),
    ("standard", "internal work and messages to the owner go ahead, contacts by their permission; the rest asks"),
    ("trusted", "internal work, messages to the owner and outside actions go ahead, contacts by their permission;"
                " the floor still asks"),
])
def test_the_first_line_is_the_level_with_its_meaning_and_the_mind_switch(tmp_path, monkeypatch, level, meaning):
    fixture = make(tmp_path, monkeypatch, autonomy=level)
    assert lines(fixture)[0] == f"Autonomy: {level} ({meaning}). Mind: on."
    fixture.store.close()


@pytest.mark.parametrize("level", ["suggest", "standard", "trusted"])
def test_each_meaning_claims_exactly_what_the_decision_table_lets_go_ahead(level):
    """The meaning is held to the authority's decision table, not to a copy of itself: each class of work is
    named as going ahead exactly when the table acts on it unasked at this level, and the floor always asks."""
    from colonymind.mind.authority import decide_table

    def decide(cls, **kwargs):
        return decide_table(level=level, cls=cls, **kwargs)
    meaning = status.LEVEL_MEANINGS[level]
    owner_acts = decide("owner") == "act"
    claims = {
        "internal work": decide("internal") == "act",
        "messages to the owner": owner_acts,
        "updates the owner asked for": decide("owner", requested=True) == "act" and not owner_acts,
        "outside actions": decide("external") == "act",
        "contacts by their permission": [decide("contact", may_contact=m) for m in ("auto", "ask", "never")]
        == ["act", "ask", "drop"],
    }
    assert {phrase: phrase in meaning for phrase in claims} == claims
    assert decide("floor") == "ask" and meaning.endswith(("the rest asks", "the rest waits for the owner",
                                                          "the floor still asks"))


def test_a_default_posture_is_the_level_and_the_faculties_switched_off_and_nothing_else(fx):
    # The fixture switches consolidation off; skills ship off. affect_rules is a mode, never "off".
    assert lines(fx) == [STANDARD,
                         "Faculties off: consolidation, skills, owner gate, wishes, drift check."]


def test_the_mind_off_names_its_reason_and_the_level_it_returns_to(fx):
    fx.mind.off(reason="owner")
    assert lines(fx)[0] == "Mind: off (owner): nothing runs on its own. Autonomy when on: standard."


def test_level_off_says_nothing_runs_and_drops_the_lines_only_the_mind_acts_on(tmp_path, monkeypatch):
    fixture = make(tmp_path, monkeypatch, autonomy="off", quiet_hours="11:00-13:00",
                   budgets={"owner_messages_per_day": 0, "contact_messages_per_day": 0, "outreach_per_day": 0})
    assert lines(fixture) == ["Autonomy: off (nothing runs on its own).", "Faculties off: consolidation, skills, owner gate, wishes, drift check."]
    fixture.store.close()


QUIET = ("Quiet hours now: queued messages and reminders wait until they end, except the digest; replies go out"
         " as usual.")


def test_quiet_hours_show_only_while_they_hold(tmp_path, monkeypatch):
    fixture = make(tmp_path, monkeypatch, quiet_hours="11:00-13:00")          # the pinned clock reads 12:00
    assert QUIET in lines(fixture)
    fixture.shift(hours=2)
    assert QUIET not in lines(fixture)
    fixture.store.close()


def test_the_quiet_hours_line_is_held_to_the_outbox(tmp_path, monkeypatch):
    """What the line says waits is what the outbox holds back in quiet hours; the digest it names goes out."""
    fixture = make(tmp_path, monkeypatch, quiet_hours="11:00-13:00")
    for kind in ("commitment_reminder", "digest"):
        fixture.store.create_intention(kind="message", type=kind, title=kind, drive="duty", cls="owner",
                                       decision="act", decision_reason="test", status="approved", dedup_key=kind,
                                       recipient=OWNER, context={"text": kind}, created_at=fixture.now)
    assert QUIET in lines(fixture)
    assert [item["type"] for item in fixture.mind.outbox.ready(quiet=fixture.mind.in_quiet_hours())] == ["digest"]
    fixture.store.close()


def test_a_used_or_empty_budget_is_one_coarse_line_and_an_open_one_is_silent(tmp_path, monkeypatch):
    fixture = make(tmp_path, monkeypatch,
                   budgets={"owner_messages_per_day": 0, "contact_messages_per_day": 0, "outreach_per_day": 0})
    shown = lines(fixture)
    assert OWNER_USED in shown
    assert "Outreach to the owner: today's budget is used." in shown
    assert "Messages to contacts: none allowed today." in shown
    fixture.store.close()


def test_the_used_owner_budget_line_is_held_to_the_budget_check(tmp_path, monkeypatch):
    """With the owner budget used, what the line says waits is charged to it and what it says still goes out
    is exempt (``Authority.budget_check``), and nothing the check exempts is left unnamed."""
    fixture = make(tmp_path, monkeypatch, budgets={"owner_messages_per_day": 0})
    authority = fixture.mind.authority
    line = next(line for line in lines(fixture) if line.startswith("Messages to the owner"))
    assert line == OWNER_USED

    def held(kind):
        return authority.budget_check(kind="message", recipient=OWNER, type=kind, now=fixture.now) is not None
    waits = {"reminders": ["commitment_reminder"], "heads-ups": ["commitment_due_soon"]}
    goes = {"the digest": ["digest"], "its notices": ["ask_notice", "breaker_notice", "health_notice"],
            "owed reports": list(authority.OWED_OWNER_TYPES)}
    for phrase, kinds in waits.items():
        assert phrase in line and all(held(kind) for kind in kinds), phrase
    for phrase, kinds in goes.items():
        assert phrase in line and not any(held(kind) for kind in kinds), phrase
    assert {kind for kinds in goes.values() for kind in kinds} == set(authority.DIGEST_TYPES
                                                                       + authority.OWED_OWNER_TYPES)
    fixture.store.close()


def test_the_used_owner_budget_line_promises_no_delivery_that_quiet_hours_hold_back(tmp_path, monkeypatch):
    """The budget line says what the budget charges, never that a message goes out: whether one goes out now is
    the outbox's to say. In quiet hours with the budget used, the notices and owed reports the budget exempts
    still wait; only the digest is ready, and the two lines shown together never disagree about that."""
    fixture = make(tmp_path, monkeypatch, quiet_hours="11:00-13:00", budgets={"owner_messages_per_day": 0})
    exempt = ("digest", "ask_notice", *fixture.mind.authority.OWED_OWNER_TYPES)
    for kind in exempt:
        fixture.store.create_intention(kind="message", type=kind, title=kind, drive="duty", cls="owner",
                                       decision="act", decision_reason="test", status="approved", dedup_key=kind,
                                       recipient=OWNER, context={"text": kind}, created_at=fixture.now)
    shown = lines(fixture)
    assert QUIET in shown and OWNER_USED in shown
    assert [item["type"] for item in fixture.mind.outbox.ready(quiet=fixture.mind.in_quiet_hours())] == ["digest"]
    assert not any(claim in OWNER_USED for claim in ("go out", "goes out", "sent", "delivered", "still"))
    fixture.store.close()


def test_open_budgets_show_no_budget_line(fx):
    assert not [line for line in lines(fx) if line.startswith(("Messages to", "Outreach to"))]


def test_paused_outreach_says_so(fx):
    fx.mind.pause_outreach(None, by="owner")
    assert "Outreach to the owner: paused." in lines(fx)


def test_outreach_switched_off_is_the_faculty_line_not_a_budget_line(tmp_path, monkeypatch):
    fixture = make(tmp_path, monkeypatch, budgets={"outreach_per_day": 0}, faculties={"outreach": False})
    shown = lines(fixture)
    assert not [line for line in shown if line.startswith("Outreach to")]
    assert shown[-1] == "Faculties off: consolidation, skills, outreach, owner gate, wishes, drift check."
    fixture.store.close()


def test_a_tripped_breaker_names_the_demoted_classes(fx):
    trip_breaker(fx)
    assert fx.mind.authority.breaker_state("internal", fx.now)["tripped"] is True
    assert "Breaker tripped: internal acts become asks until it clears." in lines(fx)


def test_the_faculties_line_lists_switches_in_their_order_and_never_a_mode(tmp_path, monkeypatch):
    fixture = make(tmp_path, monkeypatch, faculties={"people": False, "affect_rules": True})
    assert lines(fixture)[-1] == "Faculties off: people, consolidation, skills, owner gate, wishes, drift check."
    fixture.store.close()


def test_the_learning_rules_are_a_mode_of_learning_never_a_faculty_switched_off(tmp_path, monkeypatch):
    """``learning`` chooses how lessons and claims are admitted (witnesses, the novelty gate, the cold tier), not
    whether the agent learns: with it off the lessons still run, so "Faculties off: learning" would tell the agent
    something false. Like ``affect_rules`` it is never listed, on or off."""
    for learning in (False, True):
        fixture = make(tmp_path / str(learning), monkeypatch, faculties={"learning": learning})
        assert "learning" not in lines(fixture)[-1]
        fixture.store.close()


def test_a_wish_in_the_models_words_is_a_mode_of_wishes_never_a_faculty_switched_off(tmp_path, monkeypatch):
    """``wishes_named`` chooses the form a wish takes (the model's own words beside the typed capability), not
    whether the agent wishes: never listed, on or off. ``wishes`` itself is a faculty and is listed while off."""
    for named in (False, True):
        fixture = make(tmp_path / str(named), monkeypatch, faculties={"wishes_named": named})
        assert "wishes named" not in lines(fixture)[-1] and "wishes" in lines(fixture)[-1]
        fixture.store.close()


def test_degradation_shows_one_fixed_line_per_problem_and_only_when_present(fx):
    ok = {"semantic_recall": lambda: False, "tick": lambda: False, "body": lambda: False, "capture": lambda: False}
    assert not [line for line in lines(fx, problems=ok) if line.startswith("Degraded")]
    bad = {name: (lambda: True) for name in ok}
    assert [line for line in lines(fx, problems=bad) if line.startswith("Degraded")] == [
        "Degraded: semantic recall unavailable, keyword search only.",
        "Degraded: the mind's tick has stalled, so reminders and your own initiative wait.",
        "Degraded: the mind's work is not being collected, so reminders and your own initiative wait.",
        "Degraded: capture is behind, so promises from recent turns may not be recorded yet.",
    ]


def test_a_problem_shows_even_with_the_mind_off(fx):
    fx.mind.off(reason="owner")
    assert "Degraded: semantic recall unavailable, keyword search only." in lines(
        fx, problems={"semantic_recall": lambda: True})


# -- fail-silent ----------------------------------------------------------------------------------------------

def test_each_line_fails_to_unknown_and_the_others_still_render(fx, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("store gone")
    monkeypatch.setattr(fx.mind.authority, "budget_check", broken)
    monkeypatch.setattr(fx.mind.authority, "breaker_state", broken)
    monkeypatch.setattr(fx.mind, "in_quiet_hours", broken)
    monkeypatch.setattr(fx.mind, "_outreach_state", broken)
    shown = lines(fx, problems={"semantic_recall": lambda: True, "capture": broken})
    assert shown == [
        STANDARD,
        "Quiet hours: unknown.",
        "Messages to the owner: unknown.",
        "Outreach to the owner: unknown.",
        "Messages to contacts: unknown.",
        "Breaker: unknown.",
        "Faculties off: consolidation, skills, owner gate, wishes, drift check.",
        "Degraded: semantic recall unavailable, keyword search only.",
        "Capture: unknown.",
    ]


def test_a_mind_that_cannot_answer_at_all_renders_unknowns_and_never_raises():
    class Broken:
        @property
        def level(self):
            raise RuntimeError("no policy")
    shown = status.build(Broken()).split("\n")
    assert shown[0] == "Autonomy: unknown."
    assert "Faculties: unknown." in shown
    assert all(line.endswith(": unknown.") for line in shown)


def test_no_mind_is_no_section():
    assert status.build(None) == ""


# -- cache-friendly: the same bytes while nothing changed, and the cap -----------------------------------------

def test_the_same_posture_renders_the_same_bytes_as_time_passes(tmp_path, monkeypatch):
    fixture = make(tmp_path, monkeypatch, quiet_hours="11:00-13:00", budgets={"contact_messages_per_day": 0})
    first = status.build(fixture.mind)
    fixture.shift(minutes=40)
    assert status.build(fixture.mind) == first
    fixture.store.close()


@pytest.mark.parametrize("reason", ["owner", "a meeting at noon", "", "   "])
def test_a_restarted_mind_renders_the_same_bytes(fx, reason):
    """The off marker keeps its time next to the reason; a restarted mind reads the reason alone, so the same
    posture is the same bytes across a restart and no time reaches the block. An empty or blank reason (the
    off route accepts one) reads back as itself, never as the marker's time."""
    fx.mind.off(reason=reason)
    before = status.build(fx.mind)
    assert before.startswith(f"Mind: off ({reason}): " if reason.strip() else "Mind: off: nothing runs")
    fx.restart()
    assert fx.mind.enabled is False and fx.mind.off_reason == reason
    assert status.build(fx.mind) == before


def test_the_cap_keeps_the_first_line_and_the_degradation_lines(tmp_path, monkeypatch):
    fixture = make(tmp_path, monkeypatch, quiet_hours="11:00-13:00",
                   budgets={"owner_messages_per_day": 0, "contact_messages_per_day": 0, "outreach_per_day": 0},
                   faculties={"people": False, "goals": False, "lessons": False})
    whole = status.build(fixture.mind, problems={"semantic_recall": lambda: True, "capture": lambda: True})
    assert len(whole) <= status.MAX_CHARS
    cut = status.build(fixture.mind, problems={"semantic_recall": lambda: True, "capture": lambda: True}, limit=300)
    shown = cut.split("\n")
    assert len(cut) <= 300
    assert shown[0].startswith("Autonomy: standard")
    assert shown[-2:] == ["Degraded: semantic recall unavailable, keyword search only.",
                          "Degraded: capture is behind, so promises from recent turns may not be recorded yet."]
    assert set(shown) < set(whole.split("\n"))
    fixture.store.close()


def test_a_long_off_reason_is_cut_to_one_short_line(fx):
    fx.mind.off(reason="because\n" + "x" * 500)
    head = lines(fx)[0]
    assert "\n" not in head and len(head) <= 160 and head.startswith("Mind: off (because x")


def test_no_dash_glyph_reaches_the_block_from_what_the_owner_typed(fx):
    """The off reason is free text from the off route; a dash in it is rewritten before it is shown."""
    fx.mind.config["status_replacements"] = STATUS_REPLACEMENTS
    em, en = DASHES
    fx.mind.off(reason=f"travelling {em} back on the {en}ish weekend{em}maybe 3{en}4 days")
    head = lines(fx)[0]
    assert not any(glyph in head for glyph in DASHES)
    assert head.startswith("Mind: off (travelling, back on the -ish weekend-maybe 3-4 days): ")


def test_no_template_string_carries_a_dash_glyph(tmp_path, monkeypatch):
    source = Path(inspect.getsourcefile(status)).read_text(encoding="utf-8")
    assert not any(glyph in source for glyph in DASHES)
    fixture = make(tmp_path, monkeypatch, quiet_hours="11:00-13:00",
                   budgets={"owner_messages_per_day": 0, "contact_messages_per_day": 0, "outreach_per_day": 0})
    trip_breaker(fixture)
    rendered = status.build(fixture.mind, problems={name: (lambda: True) for name in status.PROBLEMS})
    assert not any(glyph in rendered for glyph in DASHES)
    fixture.store.close()


# -- the route: owner turns only, pinned, from the running mind ----------------------------------------------

@pytest.fixture
def served(source_app, fx, monkeypatch):  # noqa: F811
    monkeypatch.setenv(status.SWITCH_ENV, "on")
    monkeypatch.setattr(host, "_embed_failure", None)
    monkeypatch.setattr(host, "_embedder", None)          # semantic recall not configured unless a test says so
    source_app.add_middleware(ApiKeyMiddleware, api_key=KEY)
    mind_router.set_mind(fx.mind)
    try:
        yield source_app, fx
    finally:
        mind_router.set_mind(None)


async def assemble(app, contact=OWNER, session="s-1", text="What is on today?", inline_chars=None, **context):
    guest = contact != OWNER
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/host/context/assemble", headers=AUTH, json={
            "identity": {"host_id": "fixture"}, "context": {"contact_id": contact, "session_id": session, **context},
            "incoming_message": {"role": "user", "content": text}, **({"audience": "viewer"} if guest else {}),
            **({"inline_chars": inline_chars} if inline_chars else {})})
    assert response.status_code == 200, response.text
    return response.json()["sections"]


def by_id(sections):
    return {section["id"]: section for section in sections}


async def test_an_owner_turn_carries_the_block_from_the_running_mind(served):
    app, fx = served
    section = by_id(await assemble(app))[status.SECTION_ID]
    assert section["title"] == "Your status"
    assert section["body"] == status.build(fx.mind)
    fx.mind.off(reason="owner")                      # live state: the next turn says so, nothing stored
    assert by_id(await assemble(app))[status.SECTION_ID]["body"].startswith("Mind: off (owner)")


async def test_the_block_ships_off(served, monkeypatch):
    app, _ = served
    monkeypatch.delenv(status.SWITCH_ENV)
    assert status.SECTION_ID not in by_id(await assemble(app))
    monkeypatch.setenv(status.SWITCH_ENV, "off")
    assert status.SECTION_ID not in by_id(await assemble(app))


async def test_guests_task_runs_and_recipient_packets_never_get_it(served):
    app, _ = served
    assert status.SECTION_ID not in by_id(await assemble(app, contact=CONTACT))
    assert status.SECTION_ID not in by_id(await assemble(app, metadata={"kanban_task": "t_1"}))
    assert "Your status" not in await host.assemble_packet(OWNER, query="What is on today?")


async def test_a_scheduled_run_never_gets_it(served):
    """A scheduled (cron) job runs on the owner's lane with its prompt as the message; the memory provider says
    so (``metadata.run``), and a job's run is not the owner's turn, so it carries no posture of the owner's."""
    app, fx = served
    fx.mind.off(reason="owner")
    assert status.SECTION_ID in by_id(await assemble(app))
    assert status.SECTION_ID not in by_id(await assemble(app, metadata={"run": "cron"}))


async def test_no_mind_no_section(served):
    app, _ = served
    mind_router.set_mind(None)
    assert status.SECTION_ID not in by_id(await assemble(app))


async def test_it_sits_just_above_the_mind_section(served, monkeypatch):
    app, fx = served
    monkeypatch.setattr(fx.mind, "section", lambda **kwargs: "On my mind: the quarterly report.")
    ids = [section["id"] for section in await assemble(app)]
    assert ids.index(status.SECTION_ID) + 1 == ids.index("colonymind-mind")


async def test_selection_never_drops_it_and_it_keeps_its_place(served, monkeypatch):
    app, fx = served
    monkeypatch.setattr(fx.mind, "section", lambda **kwargs: "On my mind: the quarterly report.")
    monkeypatch.delenv(compass.SELECTION_ENV, raising=False)
    monkeypatch.setenv(compass.BUDGET_ENV, "1")
    calls = []

    async def nothing_is_relevant(query, documents, top_k=10):
        calls.append(list(documents))
        return [SimpleNamespace(index=i, score=0.0) for i in range(len(documents))]
    monkeypatch.setattr(host, "_reranker", SimpleNamespace(rerank=nothing_is_relevant))
    sections = await assemble(app)
    ids = [section["id"] for section in sections]
    assert calls and "colonymind-mind" not in ids                     # selection ran and cut the Mind section
    assert by_id(sections)[status.SECTION_ID]["body"] == status.build(fx.mind)
    assert ids.index(status.SECTION_ID) == ids.index("temporal-context") + 1
    assert not any("Your status" in document for batch in calls for document in batch)   # never judged


async def test_the_shared_selector_pins_it_whole_and_in_place():
    """The block is one of the selection's own pinned sections (``compass.PINNED_SECTIONS``), so every caller of the
    shared selector keeps it whole and in its place and the judge never reads it, whatever the budget; no wrapper
    around one caller is needed for that."""
    sections = [ContextSection(id="temporal-context", title="Current Time", body="Now: 2026-09-20T12:00:00Z",
                               priority=100),
                ContextSection(id=status.SECTION_ID, title=status.TITLE, body=STANDARD, priority=status.PRIORITY),
                ContextSection(id="colonymind-mind", title="Mind", body="On my mind: the quarterly report.",
                               priority=77)]
    judged = []

    async def nothing_is_relevant(query, documents, top_k=10):
        judged.extend(documents)
        return [SimpleNamespace(index=i, score=0.0) for i in range(len(documents))]
    settings = compass.SelectionSettings(enabled=True, budget=1, timeout_s=1.0, candidates=48,
                                         due_soon=timedelta(hours=24), min_score=0.5)
    selected, _ = await compass.select_context(sections, "What is on today?", nothing_is_relevant, settings=settings)
    assert [section.id for section in selected] == ["temporal-context", status.SECTION_ID]
    assert selected[1] is sections[1]
    assert judged and not any("Autonomy" in document for document in judged)


def rendered(sections):
    """The characters the host puts inline: each section as ``## <title>`` and its body, a blank line between."""
    return (sum(len(f"## {section['title'] or section['id']}\n{section['body']}") for section in sections)
            + 2 * max(0, len(sections) - 1))


async def test_it_is_inside_the_hosts_inline_allowance(served, monkeypatch):
    """The host puts inline only what its allowance takes (a Hermes host spills a longer context to a file). The
    block is pinned like the other owner-critical sections, so it is counted in that allowance: the optional
    sections make room for it, and the whole context the host is sent fits."""
    app, fx = served
    monkeypatch.setattr(fx.mind, "section", lambda **kwargs: "On my mind: the quarterly report. " * 12)
    monkeypatch.delenv(compass.SELECTION_ENV, raising=False)
    monkeypatch.setattr(host, "_reranker", None)
    whole = await assemble(app)
    block = by_id(whole)[status.SECTION_ID]
    assert "colonymind-mind" in by_id(whole)
    allowance = rendered(whole) - len(block["body"])      # all of it but the block's own lines
    fitted = await assemble(app, session="s-2", inline_chars=allowance)
    assert by_id(fitted)[status.SECTION_ID] == block
    assert rendered(fitted) <= allowance


async def test_two_turns_without_a_change_carry_the_same_bytes(served):
    app, fx = served
    first = by_id(await assemble(app, session="s-1", text="What is on today?"))[status.SECTION_ID]
    fx.shift(minutes=5)
    second = by_id(await assemble(app, session="s-1", text="And tomorrow?"))[status.SECTION_ID]
    assert first == second


async def test_the_hosts_health_problems_reach_the_block(served, monkeypatch):
    app, fx = served
    body = lambda sections: by_id(sections)[status.SECTION_ID]["body"]  # noqa: E731
    assert "Degraded" not in body(await assemble(app))
    fx.shift(hours=1)                                  # no tick for ten intervals: the loop has stalled
    assert "Degraded: the mind's tick has stalled" in body(await assemble(app))
    await fx.mind.tick(force=True)
    fx.mind.dispatch()                                 # and the body pulls again
    assert "Degraded" not in body(await assemble(app))
    monkeypatch.setattr(host, "_embed_failure", "the embedding endpoint refused the connection")
    assert "Degraded: semantic recall unavailable, keyword search only." in body(await assemble(app))
    monkeypatch.setattr(host, "_embed_failure", None)
    fx.mind.capture = SimpleNamespace(oldest_unfinished_seconds=lambda: 2 * 3600.0)
    assert "Degraded: capture is behind" in body(await assemble(app))

    def gone():
        raise RuntimeError("ledger locked")
    fx.mind.capture = SimpleNamespace(oldest_unfinished_seconds=gone)
    assert "Capture: unknown." in body(await assemble(app)).split("\n")


async def test_a_turn_whose_semantic_recall_failed_says_so(served, monkeypatch):
    """Semantic recall that fails on this turn (after a clean start) leaves keyword recall only: the same
    degradation as an embedder that never started, shown on that turn from the turn's own retrieval."""
    from colonymind.turns import source_vectors
    app, fx = served
    body = lambda sections: by_id(sections)[status.SECTION_ID]["body"]  # noqa: E731
    down = "Degraded: semantic recall unavailable, keyword search only."
    fx.turn("t-1", OWNER, "The quarterly report is due Friday.", "Noted.")
    assert down not in body(await assemble(app, text="When is the quarterly report due?"))

    class Refusing:
        def __init__(self, *args, **kwargs):
            pass

        async def search(self, *args, **kwargs):
            raise ConnectionError("the embedding endpoint refused the connection")
    monkeypatch.setattr(source_vectors, "SourceVectors", Refusing)
    assert host._embed_failure is None
    assert down in body(await assemble(app, text="When is the quarterly report due?")).split("\n")


async def test_configured_semantic_recall_with_nothing_behind_it_says_so(served, monkeypatch):
    """A wired embedder is semantic recall configured. With no vector store behind it the turn recalls by keywords
    only, with no startup failure and no failing search (the collection says ``unavailable``): the health route
    calls that degraded, and so does the block. With no embedder at all (recall switched off by configuration)
    and with the store wired, nothing is wrong and the block is silent."""
    from colonymind import vector
    from colonymind.memory import search
    from colonymind.turns import source_vectors
    app, fx = served
    body = lambda sections: by_id(sections)[status.SECTION_ID]["body"].split("\n")  # noqa: E731
    down = "Degraded: semantic recall unavailable, keyword search only."
    question = "When is the quarterly report due?"
    fx.turn("t-1", OWNER, "The quarterly report is due Friday.", "Noted.")
    seen = []
    collect = search.collect_sources

    async def spy(*args, **kwargs):
        collected = await collect(*args, **kwargs)
        seen.append(collected.semantic)
        return collected
    monkeypatch.setattr(search, "collect_sources", spy)
    monkeypatch.setattr(vector, "_store", None)
    monkeypatch.setattr(vector, "_pipeline", None)
    monkeypatch.setattr(host, "_embedder", None)
    assert down not in body(await assemble(app, text=question)) and seen[-1] == "unavailable"

    monkeypatch.setattr(host, "_embedder", SimpleNamespace(index_identity="fixture"))
    assert host._embed_failure is None
    assert down in body(await assemble(app, text=question)) and seen[-1] == "unavailable"

    class Answering:
        def __init__(self, *args, **kwargs):
            pass

        async def search(self, *args, **kwargs):
            return [], []
    monkeypatch.setattr(source_vectors, "SourceVectors", Answering)
    monkeypatch.setattr(vector, "_store", SimpleNamespace(catalog=object()))
    monkeypatch.setattr(vector, "_pipeline", SimpleNamespace())
    assert down not in body(await assemble(app, text=question)) and seen[-1] == "ready"


async def test_a_turn_whose_recall_did_not_finish_shows_semantic_recall_as_unknown(served, monkeypatch):
    from colonymind.memory import search
    app, fx = served

    async def broken(*args, **kwargs):
        raise RuntimeError("ledger locked")
    monkeypatch.setattr(search, "collect_sources", broken)
    shown = by_id(await assemble(app, text="When is the quarterly report due?"))[status.SECTION_ID]["body"]
    assert "Semantic recall: unknown." in shown.split("\n")


async def test_a_body_that_stopped_pulling_is_a_degradation(served):
    """The tick beats before it skips a stale body's work, so a live tick says nothing about the body: while
    nothing collects the mind's work, reminders and initiative wait, and the block says so."""
    app, fx = served
    body = lambda sections: by_id(sections)[status.SECTION_ID]["body"]  # noqa: E731
    stale = "Degraded: the mind's work is not being collected, so reminders and your own initiative wait."
    fx.mind.dispatch()
    assert stale not in body(await assemble(app))
    fx.shift(minutes=10)
    assert (await fx.mind.tick())["skipped"] == "body stale"
    shown = body(await assemble(app))
    assert stale in shown.split("\n") and "tick has stalled" not in shown
    fx.mind.dispatch()
    assert stale not in body(await assemble(app))


# -- bounded: a slow store costs the turn at most the collection deadline -----------------------------------

OWNER_TURN = SimpleNamespace(context=SimpleNamespace(session_id="s-1", metadata={}))


async def section_body(**kwargs) -> str:
    """The owner's status body as the assemble gets it."""
    result = host._status_section(OWNER_TURN, **kwargs)
    return await result if inspect.isawaitable(result) else result


class Beat:
    """An event-loop heartbeat: ``longest_gap`` is the longest the loop went without running other work."""

    def __init__(self, every=0.01):
        self.every, self.beats, self.task = every, [], None

    async def _run(self):
        while True:
            self.beats.append(time.perf_counter())
            await asyncio.sleep(self.every)

    async def __aenter__(self):
        self.task = asyncio.create_task(self._run())
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *exc):
        self.task.cancel()
        self.beats.append(time.perf_counter())

    @property
    def longest_gap(self):
        return max(later - earlier for earlier, later in zip(self.beats, self.beats[1:]))


async def comes_back(field):
    """The line of ``field`` stops saying unknown within two seconds."""
    for _ in range(40):
        if f"{field}: unknown." not in (await section_body()).split("\n"):
            return
        await asyncio.sleep(0.05)
    pytest.fail(f"{field} still unknown")


async def test_a_locked_capture_store_costs_the_turn_at_most_the_deadline(served, monkeypatch):
    """The capture probe opens the capture ledger with its ten second busy timeout, longer than the host waits
    for the whole prefetch. With another connection holding the ledger in exclusive locking mode the probe
    waited the full ten seconds on the event loop. It must cost the turn at most the collection deadline: the
    line says unknown, the loop keeps running, and once the lock goes the next turn reads the queue again."""
    from colonymind.commitments.extract import CommitmentExtractor
    _app, fx = served
    monkeypatch.setattr(status, "DEADLINE_S", 0.3)
    fx.mind.capture = CommitmentExtractor(fx.ledger, lambda: fx.commitments)
    assert "Capture: unknown." not in (await section_body()).split("\n")
    holder = sqlite3.connect(str(fx.ledger.db_path), timeout=1)
    holder.execute("PRAGMA locking_mode=EXCLUSIVE")
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("CREATE TABLE IF NOT EXISTS held_by_another_connection (x)")
    holder.commit()                     # exclusive locking mode keeps the lock after the commit
    try:
        async with Beat() as beat:
            started = time.perf_counter()
            shown = (await section_body()).split("\n")
            elapsed = time.perf_counter() - started
    finally:
        holder.close()
    assert "Capture: unknown." in shown and shown[0] == STANDARD
    assert elapsed < 1.0, elapsed
    assert beat.longest_gap < 0.25, beat.longest_gap
    await comes_back("Capture")


async def test_a_locked_capture_store_leaves_the_posture_lines_read(served, monkeypatch):
    """The capture ledger and the mind's own store are separate stores: while the ledger is locked and an
    earlier turn's read of it is still stalled, the next turn still reads its budgets and breaker from the
    mind's store, and only the capture line says unknown."""
    from colonymind.commitments.extract import CommitmentExtractor
    from colonymind.memory import curated
    _app, fx = served
    monkeypatch.setattr(status, "DEADLINE_S", 0.3)
    trip_breaker(fx)
    # Outcome setup opens a persistent WAL reader. Release it before acquiring
    # the exclusive lock that this test uses to stall the capture read.
    curated._cache.cache_clear()
    fx.mind.capture = CommitmentExtractor(fx.ledger, lambda: fx.commitments)
    holder = sqlite3.connect(str(fx.ledger.db_path), timeout=1)
    holder.execute("PRAGMA locking_mode=EXCLUSIVE")
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("CREATE TABLE IF NOT EXISTS held_by_another_connection (x)")
    holder.commit()
    try:
        first = (await section_body()).split("\n")
        second = (await section_body()).split("\n")
    finally:
        holder.close()

    def unknown(shown):
        return [line for line in shown if line.endswith(": unknown.")]
    assert unknown(first) == unknown(second) == ["Capture: unknown."]
    assert "Breaker tripped: internal acts become asks until it clears." in second
    await comes_back("Capture")


LONG_SCAN = "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r LIMIT 30000000) SELECT count(*) FROM r"


async def test_a_long_read_on_the_minds_store_costs_the_whole_turn_at_most_the_deadline(served, monkeypatch):
    """A read on the mind's own store that runs long (a scan of a large history) makes that store's lines
    unknown at the deadline and costs the whole turn no more. The Mind section right after the block reads the
    same store over the same connection, so a read left running past the deadline (on any thread) would hold
    that store, and the turn, until it ends. Nothing of the deadline is left on the store afterwards."""
    app, fx = served
    monkeypatch.setattr(status, "DEADLINE_S", 0.3)
    count_transitions = fx.store.count_transitions

    def scanning(*args, **kwargs):
        fx.store._db.execute(LONG_SCAN).fetchone()
        return count_transitions(*args, **kwargs)
    monkeypatch.setattr(fx.store, "count_transitions", scanning)
    async with Beat() as beat:
        started = time.perf_counter()
        shown = by_id(await assemble(app))[status.SECTION_ID]["body"].split("\n")
        elapsed = time.perf_counter() - started
    monkeypatch.setattr(fx.store, "count_transitions", count_transitions)
    assert shown[0] == STANDARD and "Faculties off: consolidation, skills, owner gate, wishes, drift check." in shown
    assert "Messages to the owner: unknown." in shown
    assert elapsed < 2.5, elapsed     # below the read's own length; the room is for a pause of the whole process
    assert beat.longest_gap < 0.3 + 0.25, beat.longest_gap
    fx.store._db.execute(LONG_SCAN.replace("30000000", "3000000")).fetchone()     # runs to its end
    assert "Messages to the owner: unknown." not in (await section_body()).split("\n")


async def test_a_capture_store_still_locked_from_an_earlier_turn_gets_no_second_read(served, monkeypatch):
    """While an earlier turn's read of the capture ledger is still waiting past its deadline, the next turn's
    capture line is unknown at once and no second read starts, so a wedged store never holds one thread per
    turn; once the lock goes, the next turn reads it again."""
    from colonymind.commitments.extract import CommitmentExtractor
    _app, fx = served
    monkeypatch.setattr(status, "DEADLINE_S", 0.3)
    fx.mind.capture = CommitmentExtractor(fx.ledger, lambda: fx.commitments)
    reads = []
    oldest = fx.mind.capture.oldest_unfinished_seconds

    def counted():
        reads.append(1)
        return oldest()
    monkeypatch.setattr(fx.mind.capture, "oldest_unfinished_seconds", counted)
    holder = sqlite3.connect(str(fx.ledger.db_path), timeout=1)
    holder.execute("PRAGMA locking_mode=EXCLUSIVE")
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("CREATE TABLE IF NOT EXISTS held_by_another_connection (x)")
    holder.commit()
    try:
        first = (await section_body()).split("\n")
        started = time.perf_counter()
        second = (await section_body()).split("\n")
        quick = time.perf_counter() - started
        assert len(reads) == 1
    finally:
        holder.close()
    assert "Capture: unknown." in first and "Capture: unknown." in second
    assert quick < 0.2, quick
    await comes_back("Capture")
    assert len(reads) > 1


async def test_turns_at_the_same_time_each_read_the_whole_status(served):
    _app, fx = served
    bodies = await asyncio.gather(*(section_body() for _ in range(4)))
    assert bodies == [status.build(fx.mind)] * 4


# -- the switch: context.status in colonymind.yaml ------------------------------------------------------------

def test_context_status_ships_on_and_is_exported(tmp_path):
    """On unless shown harmful: the status block ships on, and the file switches it off."""
    assert config.DEFAULTS["context"]["status"] is True
    environ: dict[str, str] = {}
    apply_environment(load_config(tmp_path, environ={}), environ=environ)
    assert environ[status.SWITCH_ENV] == "on"
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump({"context": {"status": False}}))
    environ = {}
    apply_environment(load_config(tmp_path, environ={}), environ=environ)
    assert environ[status.SWITCH_ENV] == "off"
    assert status.enabled(environ) is False and status.enabled({status.SWITCH_ENV: "on"}) is True
    assert status.enabled({}) is False          # the module only reads what the install exported


def test_context_status_is_a_boolean_with_one_place_to_set_it(tmp_path):
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump({"context": {"status": "sometimes"}}))
    with pytest.raises(ConfigError, match="context.status"):
        load_config(tmp_path, environ={})
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump({"environment": {status.SWITCH_ENV: "on"}}))
    with pytest.raises(ConfigError, match="context.status"):
        load_config(tmp_path, environ={})
    assert config.RESERVED_ENVIRONMENT[status.SWITCH_ENV] == "context.status"
