"""Resident blocks (Compass 5.6 and 15.9): per-block limits, one global budget, fresh state first, no
silent cut, no growth from a timestamp bump, and a fossil guard.

The failure these tests pin: a writer that re-persists its blocks each night grows them past their
limits until the loader exceeds the host's cap, and then *every* block vanishes from the wake with no
error. Hermes replaces a memory context longer than ``hooks.output_spill.max_chars`` with a 500-character
head, a 500-character tail and a file path, so on the ColonyMind side the same failure is the context
assemble outgrowing that cap.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from test_turn_source_evidence import source_app  # noqa: F401  (pytest fixture)

from colonymind.memory import blocks
from colonymind.memory.blocks import (DEFAULT_LIMIT, HELD_TITLE, LIMITS, RANK, Block, BlockLedger, cost, fingerprint,
                                     fit, rotate)

T0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
HOST_CAP = 10_000        # Hermes' default hooks.output_spill.max_chars


def rendered(shown):
    """The provider's rendering (``_format_sections``)."""
    return "\n\n".join(f"## {block.title}\n{block.body}" for block in shown)


def lines(prefix, count, width=90):
    return "\n".join(f"{prefix} {index:03d} ".ljust(width, "x") for index in range(count))


def spill(text, directory, cap=HOST_CAP):
    """What stock Hermes does with an over-cap memory context (``tools.hook_output_spill``, when Hermes is
    installed; otherwise the same contract: a head, a tail and a path)."""
    try:
        from tools.hook_output_spill import spill_if_oversized
    except ImportError:
        spill_if_oversized = None
    if spill_if_oversized is not None:
        return spill_if_oversized(text, session_id="blocks", source="memory prefetch", config={
            "enabled": True, "max_chars": cap, "preview_head": 500, "preview_tail": 500, "directory": str(directory)})
    if len(text) <= cap:
        return text
    return f"[memory prefetch output truncated - {len(text):,} chars]\n--- head ---\n{text[:500]}\n--- tail ---\n{text[-500:]}"


# -- 1. one block over its limit ------------------------------------------------------------------------

def test_a_block_over_its_limit_keeps_whole_lines_and_says_what_it_held_back():
    body = lines("fact", 40)
    out = rotate(body, 1500, why="this block's 1500-character limit")
    assert len(out) <= 1500
    kept, marker = out.split("\n")[:-1], out.split("\n")[-1]
    assert kept and all(line in body.split("\n") for line in kept)              # whole lines, first ones
    assert kept == body.split("\n")[:len(kept)]
    held = 40 - len(kept)
    assert marker == f"[{len(body) - len(chr(10).join(kept))} characters in {held} lines held back: this block's 1500-character limit]"
    assert rotate(body, len(body), why="x") == body                              # a block that fits is untouched


def test_a_single_line_over_its_limit_is_cut_at_a_word_and_the_cut_is_said():
    body = json.dumps({"facts": [f"sensor {i} reads {i * 3} degrees" for i in range(200)]})
    out = rotate(body, 600, why="w")
    assert len(out) <= 600 and "\n" in out
    head, marker = out.split("\n")
    assert head.endswith("…") and body.startswith(head[:-1]) and marker.startswith("[") and "in 1 line held back" in marker


def test_every_producer_cap_fits_inside_its_block_limit():
    """A limit acts only on a block that outgrew what its producer promised: no normal turn changes."""
    from colonymind.contacts.digest import MAX_CHARS as DIGEST
    from colonymind.mind.opinions import CONTEXT_CHARS as STANCES
    from colonymind.mind.tick import MIND_SECTION_CHARS
    assert LIMITS["colonymind-mind"] >= MIND_SECTION_CHARS
    assert LIMITS["colonymind-person"] >= DIGEST
    assert LIMITS["colonymind-stances"] >= STANCES
    assert LIMITS["colonymind-executions"] >= 4000 and LIMITS["colonymind-memory"] >= 24000
    assert RANK[:2] == ("temporal-context", "colonymind-current-state")          # fresh state first
    assert RANK.index("colonymind-current-state") < RANK.index("colonymind-owner-preferences")
    assert RANK.index("colonymind-memory") < RANK.index("colonymind-owner-preferences")


# -- 2. the global budget -------------------------------------------------------------------------------

def turn(growth=0):
    """One owner turn's blocks: small fresh state, recall, the slow profile growing ``growth`` lines."""
    return [
        Block("temporal-context", "Current Time", "Sunday, September 1, 2026, 12:00 PM UTC."),
        Block("colonymind-memory", "Relevant Memories", lines("memory", 30)),
        Block("colonymind-owner-preferences", "How they want me to communicate", lines("pref", 5 + growth)),
        Block("colonymind-stances", "Your recorded views", lines("view", 3 + growth)),
        Block("colonymind-lessons", "What you learned", lines("lesson", 1 + growth)),
        Block("colonymind-current-state", "Current observed state and stale observations",
              json.dumps({"facts": ["deploy is green", "backup ran 02:00"], "stale": []})),
        Block("colonymind-goals", "Active Goals", lines("goal", 2 + growth)),
        Block("colonymind-initiatives", "Pending Initiatives", lines("initiative", 2 + growth)),
    ]


def test_the_status_block_and_the_corrections_are_admitted_before_the_recall():
    """Two sections the V1 line added: the status block is the agent's posture now (fresh state, beside the Mind
    section) and the corrections to earlier context are owner-critical. Unranked, a tight budget would hold them
    back before anything else; ranked, they are admitted before this turn's recall and the slow profile."""
    assert blocks.rank_of("colonymind-status") < blocks.rank_of("colonymind-memory")
    assert blocks.rank_of("colonymind-corrections") < blocks.rank_of("colonymind-memory")
    turn = [Block("temporal-context", "Current Time", "Now.", 0),
            Block("colonymind-memory", "Relevant Memories", lines("memory", 40), 1),
            Block("colonymind-stances", "Your recorded views", lines("view", 10), 2),
            Block("colonymind-status", "Your status", "Autonomy: standard. Mind: on.", 3),
            Block("colonymind-corrections", "Corrections to earlier context", lines("correction", 2), 4)]
    fitted, report = fit(turn, budget=1200)
    shown = {block.id: block.body for block in fitted}
    assert shown["colonymind-status"] == turn[3].body and shown["colonymind-corrections"] == turn[4].body
    cut = [item["id"] for item in report["rotated"] + report["held_back"]]
    assert "colonymind-stances" in cut and "colonymind-memory" in cut
    assert "colonymind-status" not in cut and "colonymind-corrections" not in cut


def test_a_turn_within_budget_is_passed_through_byte_for_byte():
    sections = turn()
    shown, report = fit(sections, budget=HOST_CAP)
    assert shown == sections and not report["rotated"] and not report["held_back"]


def test_over_budget_fresh_state_is_admitted_first_and_every_block_is_shown_or_named():
    sections = turn(growth=60)
    assert len(rendered(sections)) > HOST_CAP
    shown, report = fit(sections, budget=HOST_CAP)
    text = rendered(shown)
    assert len(text) <= HOST_CAP and report["chars_after"] <= HOST_CAP
    ids = [block.id for block in shown]
    state = next(block for block in sections if block.id == "colonymind-current-state")
    assert state in shown and sections[0] in shown                                  # fresh state whole
    for block in sections:                                                          # nothing vanishes silently
        assert block.id in ids or block.title in shown[-1].body
    for item in report["rotated"] + report["held_back"]:
        assert len(item["sha256"]) == 64                                            # what was not shown is on record
    rotated = [block for block in shown if block.body.endswith("context budget]")]
    assert rotated and all(block.id != "colonymind-current-state" for block in rotated)
    # display order is kept: admission alone follows the rank
    assert [i for i in ids if i != blocks.HELD_ID] == [b.id for b in sections if b.id in ids]


def test_under_a_tight_budget_the_slow_profile_is_held_back_before_the_current_state():
    sections = turn(growth=60)
    shown, report = fit(sections, budget=1500)
    ids = [block.id for block in shown]
    assert "colonymind-current-state" in ids and "temporal-context" in ids
    held = [item["id"] for item in report["held_back"]]
    assert held and "colonymind-current-state" not in held
    assert all(RANK.index(h) > RANK.index("colonymind-current-state") for h in held)
    assert shown[-1].title == HELD_TITLE and len(rendered(shown)) <= 1500


def test_a_section_the_provider_will_collapse_costs_one_line_and_is_never_rotated():
    """The provider sends a section its session's last turn carried as it is as one line; the budget counts
    it that way, so a later turn does not rotate or hold back what the model already has."""
    sections = turn(growth=60)
    first, _ = fit(sections, budget=HOST_CAP)
    prefs = next(block for block in first if block.id == "colonymind-owner-preferences")   # as the provider showed it
    shown, report = fit(sections, budget=HOST_CAP, shown={fingerprint(prefs.title, prefs.body)})
    assert prefs in shown
    assert "colonymind-owner-preferences" not in {item["id"] for item in report["rotated"] if item["reason"] == "budget"}
    assert report["chars_after"] < fit(sections, budget=HOST_CAP)[1]["chars_after"]
    # the clock is never collapsed, so it always counts whole
    clock = sections[0]
    _, again = fit(sections, budget=None, shown={fingerprint(clock.title, clock.body)})
    assert again["chars_after"] == fit(sections, budget=None)[1]["chars_after"]


def test_fit_invariants_hold_over_random_turns():
    """Seeded sweep: within budget, display order kept, every block shown or named, the clock and the
    current state never held back, every rotated block a prefix of its lines plus one marker."""
    import random
    rng = random.Random(20260925)
    ids = list(RANK) + ["colonymind-unknown"]
    for case in range(400):
        chosen = rng.sample(ids, rng.randint(1, len(ids)))
        if "temporal-context" not in chosen:
            chosen.insert(0, "temporal-context")
        sections = [Block(ident, f"Title {ident}", "Now." if ident == "temporal-context" else
                          "\n".join("w" * rng.randint(1, 400) for _ in range(rng.randint(1, 40))))
                    for ident in chosen]
        budget = rng.randint(1500, 20000)
        shown, report = fit(sections, budget=budget)
        assert len(rendered(shown)) <= budget, case
        by_id = {block.id: block for block in shown}
        held = shown[-1].body if shown[-1].id == blocks.HELD_ID else ""
        for block in sections:
            assert block.id in by_id or block.title in held, (case, block.id)
            if block.id in by_id and by_id[block.id].body != block.body:
                *kept, marker = by_id[block.id].body.split("\n")
                assert marker.startswith("[") and marker.endswith("]") and "held back" in marker
                assert kept == block.body.split("\n")[:len(kept)] or (len(kept) == 1 and kept[0].endswith("…"))
        assert "temporal-context" in by_id
        assert "colonymind-current-state" in by_id or "colonymind-current-state" not in chosen
        order = [block.id for block in shown if block.id != blocks.HELD_ID]
        assert order == [ident for ident in chosen if ident in by_id]


# -- 3. the Compass failure: blocks growing night after night -------------------------------------------

def persist_night(store, night):
    """The Compass persist hook: every night each persisted block gains a timestamped line."""
    for key in store:
        store[key] += f"\n{(T0 + timedelta(days=night)).isoformat()} still true: " + "y" * 60


@pytest.mark.parametrize("enabled", [False, True])
def test_blocks_growing_past_the_cap_never_make_every_block_vanish(enabled, tmp_path):
    store = {"colonymind-owner-preferences": "prefers short answers", "colonymind-stances": "views",
             "colonymind-lessons": "one lesson", "colonymind-goals": "one goal", "colonymind-initiatives": "one"}
    titles = {"colonymind-owner-preferences": "How they want me to communicate", "colonymind-stances": "Your recorded views",
              "colonymind-lessons": "What you learned", "colonymind-goals": "Active Goals",
              "colonymind-initiatives": "Pending Initiatives"}
    state = Block("colonymind-current-state", "Current observed state and stale observations", '{"facts": ["deploy is green"]}')
    vanished_on = None
    for night in range(40):
        persist_night(store, night)
        sections = [Block("temporal-context", "Current Time", f"Night {night}."), state,
                    Block("colonymind-memory", "Relevant Memories", lines("memory", 20)),
                    *[Block(key, titles[key], body) for key, body in store.items()]]
        shown = fit(sections, budget=HOST_CAP - 400)[0] if enabled else sections
        wake = spill(rendered(shown), tmp_path / "spill")
        present = [block for block in sections if f"## {block.title}" in wake or block.title in wake]
        if vanished_on is None and len(present) < len(sections):
            vanished_on = night
        if enabled:
            assert len(wake) <= HOST_CAP and state.body in wake and f"Night {night}." in wake
            assert len(present) == len(sections), night
    if enabled:
        assert vanished_on is None
    else:
        # the 1.10 loader: some night the context crosses the cap and the wake keeps only a head and a tail
        assert vanished_on is not None and vanished_on < 40


# -- 4. a timestamp bump grows nothing, and the fossil guard ---------------------------------------------

class State:
    """``MindState``'s surface the ledger uses, in memory."""

    def __init__(self):
        self.rows = {}

    def get(self, key):
        return self.rows.get(key)

    def items(self, prefix=""):
        return [row for key, row in sorted(self.rows.items()) if key.startswith(prefix)]

    def set(self, key, *, text=None, now=None, **_):
        self.rows[key] = {"key": key, "text": text, "updated_at": now.isoformat() if now else None}
        return self.rows[key]


def test_rewriting_the_same_content_never_moves_its_change_time_or_grows_it():
    ledger = BlockLedger(State(), fossil_days=7)
    first = ledger.observe("self-narrative.recent", "recent: I did a thing. [a1]", sources=["a1"], now=T0)
    for day in range(1, 30):
        again = ledger.observe("self-narrative.recent", "recent: I did a thing. [a1]", sources=["a1"], rebuilt=day % 2 == 0,
                               now=T0 + timedelta(days=day))
        assert again["changed_at"] == first["changed_at"] and again["chars"] == first["chars"]
        assert not again["stale"] and not again["fossilised"]      # its sources did not move either


def test_a_block_not_rebuilt_for_n_days_while_its_sources_moved_is_flagged_once(caplog):
    ledger = BlockLedger(State(), fossil_days=7)
    ledger.observe("digest:p-02", "They work in room 4.", sources=["c1"], now=T0)
    young = ledger.observe("digest:p-02", "They work in room 4.", sources=["c1", "c2"], now=T0 + timedelta(days=2))
    assert young["stale"] and not young["fossilised"]                             # too young to call
    assert not ledger.observe("digest:p-02", "They work in room 4.", sources=["c1", "c2"],
                              now=T0 + timedelta(days=8))["fossilised"]            # stale six days, not seven
    with caplog.at_level("WARNING", logger="colonymind.memory.blocks"):
        flagged = ledger.observe("digest:p-02", "They work in room 4.", sources=["c1", "c2"], now=T0 + timedelta(days=9))
        ledger.observe("digest:p-02", "They work in room 4.", sources=["c1", "c2"], now=T0 + timedelta(days=9, hours=12))
    assert flagged["fossilised"] and len(caplog.records) == 1 and "digest:p-02" in caplog.records[0].getMessage()
    assert [row["id"] for row in ledger.fossils(T0 + timedelta(days=9))] == ["digest:p-02"]
    # a text that only changed (a line expired) without a rebuild is still fossilised
    assert ledger.observe("digest:p-02", "They work.", sources=["c1", "c2"], now=T0 + timedelta(days=10))["fossilised"]
    # a rebuild from the moved sources clears it, even when the writer kept the same text
    cleared = ledger.observe("digest:p-02", "They work.", sources=["c1", "c2"], rebuilt=True, now=T0 + timedelta(days=11))
    assert not cleared["stale"] and not cleared["fossilised"] and not ledger.fossils(T0 + timedelta(days=40))


def test_a_digest_seen_first_rests_on_its_own_sources_not_on_todays_claims():
    ledger = BlockLedger(State(), fossil_days=7)
    seen = ledger.observe("digest:p-03", "Old digest.", sources=["c1", "c2"], built_from=["c1"], now=T0)
    assert seen["stale"] and not seen["fossilised"]
    assert ledger.observe("digest:p-03", "Old digest.", sources=["c1", "c2"], built_from=["c1"],
                          now=T0 + timedelta(days=7))["fossilised"]


# -- 5. the route, the packet and the flag ----------------------------------------------------------------

@pytest.fixture
def context_app(source_app, tmp_path, monkeypatch):  # noqa: F811
    from colonymind.api.routers import host
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    blocks.reset_settings()
    heavy = lines("view", 60)

    async def fake_sections(body, *, viewer_person_id, request=None, superseded=None):
        from colonymind.api.schemas.host import ContextSection
        return [ContextSection(id="temporal-context", title="Current Time", body="Now.", priority=100),
                ContextSection(id="colonymind-memory", title="Relevant Memories", body=lines("memory", 60), priority=90),
                ContextSection(id="colonymind-stances", title="Your recorded views", body=heavy, priority=87),
                ContextSection(id="colonymind-current-state", title="Current observed state and stale observations",
                               body='{"facts": ["deploy is green"]}', priority=76)]
    monkeypatch.setattr(host, "_assemble_sections", fake_sections)
    yield source_app, host
    blocks.reset_settings()


async def assemble(app, **extra):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/host/context/assemble", json={
            "identity": {"host_id": "fixture"}, "context": {"contact_id": "p-01", "session_id": "s-1"},
            "incoming_message": {"role": "user", "content": "status?"}, **extra})
    assert response.status_code == 200, response.text
    return response.json()


async def test_the_route_fits_the_hosts_budget_and_reports_it(context_app):
    data = await assemble(context_app[0], inline_chars=3000)
    sections = data["sections"]
    text = "\n\n".join(f"## {s['title']}\n{s['body']}" for s in sections)
    assert len(text) <= 3000
    ids = [s["id"] for s in sections]
    assert "colonymind-current-state" in ids and ids[0] == "temporal-context"
    report = data["budget"]
    assert report["budget"] == 3000 and report["chars_before"] > 3000 and report["chars_after"] <= 3000
    assert blocks.last_fit()["where"] == "context/assemble"


async def test_the_block_fit_uses_the_inline_allowance_the_host_names(context_app, monkeypatch):
    """The host's inline allowance can shorten the configured ceiling, never widen it."""
    app, host = context_app
    recall = lines("memory", 150)                  # 13,650 characters: over the default budget, inside the host's

    async def recalled(body, *, viewer_person_id, request=None, superseded=None):
        from colonymind.api.schemas.host import ContextSection
        return [ContextSection(id="temporal-context", title="Current Time", body="Now.", priority=100),
                ContextSection(id="colonymind-memory", title="Relevant Memories", body=recall, priority=90)]
    monkeypatch.setattr(host, "_assemble_sections", recalled)
    data = await assemble(app, inline_chars=20_000)
    memory = next(s for s in data["sections"] if s["id"] == "colonymind-memory")
    assert len(memory["body"]) < len(recall) and data["budget"]["budget"] == blocks.DEFAULT_BUDGET
    assert data["budget"]["rotated"]


async def test_an_allowance_the_pinned_items_outgrow_is_counted_while_the_block_fit_takes_the_context(
        context_app, monkeypatch):
    """Health's ``recall`` block counts the per-turn selection's outcome on every turn (``recall_stats``), the turn
    the block fit takes over included: an allowance the memory and the owner-critical items alone outgrow is an
    ``over_allowance`` the selection hands on to the block fit, and it is still counted."""
    from colonymind.memory import compass, recall_stats
    app, host = context_app
    monkeypatch.delenv(compass.SELECTION_ENV, raising=False)
    calls = []

    class Judge:
        async def rerank(self, query, documents, top_k=10):
            calls.append(len(documents))
            return [{"index": i, "score": 0.5} for i in range(len(documents))]
    monkeypatch.setattr(host, "_reranker", Judge())
    recall_stats.STATS.reset()
    data = await assemble(app, inline_chars=3000)
    assert data["budget"]["budget"] == 3000 and data["budget"]["chars_after"] <= 3000
    assert "colonymind-current-state" in [s["id"] for s in data["sections"]] and calls == []
    assert recall_stats.STATS.snapshot()["judge"]["over_allowance"] == 1


async def test_the_default_budget_applies_when_the_host_names_none(context_app):
    data = await assemble(context_app[0])
    assert data["budget"]["budget"] == blocks.DEFAULT_BUDGET
    stances = next(s for s in data["sections"] if s["id"] == "colonymind-stances")
    assert stances["body"].endswith("1500-character limit]") and len(stances["body"]) <= 1500


async def test_the_recipient_packet_is_never_cut_mid_text(context_app):
    packet = await context_app[1].assemble_packet("p-02", query="hello", limit_chars=1200)
    assert len(packet) <= 1200
    assert packet.rstrip().endswith("a memory search reaches them.") or "held back:" in packet
    assert "## Current observed state and stale observations" in packet


def test_the_config_validates_the_context_section():
    from colonymind.config import ConfigError, DEFAULTS, _merge, validate
    from colonymind.memory.selection import RERANK_SUBMISSION_LIMIT
    assert DEFAULTS["context"] == {"status": True, "budget_chars": 9000, "recall_items": 20, "fossil_days": 7,
                                   "limits": {}, "recall_notice_replacements": [], "recall_alarm_turns": 2}
    assert DEFAULTS["context"]["recall_items"] == RERANK_SUBMISSION_LIMIT == 20
    assert validate(_merge(DEFAULTS, {"context": {"recall_items": 30}}))["context"]["recall_items"] == 30
    good = validate(_merge(DEFAULTS, {"context": {"recall_items": 3, "limits": {"colonymind-goals": 800}}}))
    assert good["context"]["recall_items"] == 3 and good["context"]["limits"] == {"colonymind-goals": 800}
    for bad in ({"blocks": False}, {"blocks": True}, {"recall_items": 0}, {"recall_items": True}, {"recall_items": 1.5}, {"budget_chars": 100}, {"fossil_days": 0}, {"limits": {"x": 10}}, {"limits": []}):
        with pytest.raises(ConfigError):
            validate(_merge(DEFAULTS, {"context": bad}))


def test_cost_bounds_the_providers_rendering():
    shown = turn(growth=3)
    assert len(rendered(shown)) <= sum(cost(b.title, b.body) for b in shown)
    assert DEFAULT_LIMIT == 1500


# -- 6. the night's persisted blocks: the narrative and the digests ---------------------------------------

@pytest.fixture
def night(tmp_path, monkeypatch):
    from test_mind_consolidate import Fixture
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", "p-01")
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path / "home"))
    blocks.reset_settings()
    fixture = Fixture(tmp_path / "state")
    yield fixture
    fixture.store.close()
    blocks.reset_settings()


async def test_a_night_that_changes_nothing_leaves_the_narrative_as_old_as_it_is(night):
    from test_mind_consolidate import settled_task
    row = settled_task(night)
    await night.mind.consolidate()
    first = night.mind.mind_state.get("self.recent")
    text = night.mind.narrative()["text"]
    assert row.id in first["text"]
    for _ in range(3):
        night.shift(hours=24)
        night.router.answers["mind_consolidate_narrative"] = lambda messages, context: {"lines": [
            {"text": "I did the work behind " + row.id[:8] + " and learned from it", "cites": [row.id]}]}
        await night.mind.consolidate()
    again = night.mind.mind_state.get("self.recent")
    assert again["updated_at"] == first["updated_at"] and again["text"] == first["text"]
    assert night.mind.narrative()["text"] == text


async def test_an_over_limit_narrative_keeps_recent_first_and_says_what_it_held_back(night):
    from test_mind_consolidate import settled_task
    for index in range(6):
        night.mind.add_interest(f"a long standing interest in topic number {index} " + "z" * 60)
    rows = [settled_task(night, title=f"{kind} {copy}", type_=kind, dedup=f"{kind}:{copy}")
            for kind in ("research", "check", "upkeep") for copy in range(2)]
    night.router.answers["mind_consolidate_narrative"] = lambda messages, context: {"lines": [
        {"text": f"I finished {r.description} and it held up under a second look " + "w" * 40, "cites": [r.id]}
        for r in rows[:4]]}
    await night.mind.consolidate()
    narrative = night.mind.narrative()
    text = narrative["text"]
    assert len(text) <= 800 and narrative["held_back"] > 0
    held = narrative["held_back"]
    assert text.splitlines()[-1] == f"[{held} narrative line{'' if held == 1 else 's'} held back: over the 800-character limit]"
    recent = [line for line in text.splitlines() if line.startswith("recent: ")]
    assert len(recent) == len(narrative["sections"]["recent"].splitlines())       # every recent line kept
    shown = [line for line in text.splitlines() if not line.startswith("[")]
    order = ["interest", "strength", "recent", "stance"]
    assert [order.index(line.split(":")[0]) for line in shown] == sorted(order.index(line.split(":")[0]) for line in shown)


async def test_a_digest_that_keeps_failing_while_the_claims_move_is_flagged_as_fossilised(night):
    from colonymind.api.routers import mind as mind_router
    from fastapi import FastAPI
    await night.fact("t-1", "p-02", "s-2", "I work in room 4.", "room 4")
    night.contacts.touch("p-02", night.now)
    await night.mind.consolidate()
    assert night.contacts.records["p-02"].digest
    night.router.answers["mind_consolidate_digest"] = {"digest": "", "sources": []}      # rejected from now on
    night.shift(hours=24)
    await night.fact("t-2", "p-02", "s-2", "I moved to room 9.", "room 9")
    night.contacts.touch("p-02", night.now)
    await night.mind.consolidate()
    ledger = night.mind.consolidation.blocks
    assert ledger.get("digest:p-02")["stale"] and not ledger.get("digest:p-02")["fossilised"]
    for _ in range(7):
        night.shift(hours=24)
        night.contacts.touch("p-02", night.now)
        await night.mind.consolidate()
    assert ledger.get("digest:p-02")["fossilised"]
    app = FastAPI()
    app.include_router(mind_router.router)
    mind_router.set_mind(night.mind)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            data = (await client.get("/v1/mind/blocks")).json()
    finally:
        mind_router.set_mind(None)
    assert "digest:p-02" in data["fossils"] and data["settings"]["fossil_days"] == 7
    assert {"self-narrative.recent", "digest:p-02"} <= {record["id"] for record in data["blocks"]}


# -- 7. the identity block: the constitution --------------------------------------------------------------

def test_an_over_limit_constitution_holds_back_whole_values_keeps_every_boundary_and_says_so():
    from colonymind.config import CONSTITUTION_CHARS, render_constitution
    identity = {"agent": {"name": "Agent", "values": [f"value {i} ".ljust(150, "v") for i in range(12)],
                          "boundaries": [f"boundary {i} ".ljust(150, "b") for i in range(6)]}}
    text = render_constitution(identity)
    assert len(text) <= CONSTITUTION_CHARS
    assert all(f"boundary {i} " in text for i in range(6))                        # the safety floor stays whole
    kept = [i for i in range(12) if f"value {i} " in text]
    assert kept == list(range(len(kept))) and len(kept) < 12                        # values held back from the end
    assert text.endswith(f"({12 - len(kept)} more items held back: over the 1500-character limit.)")


async def test_a_narrative_the_model_stops_rebuilding_while_the_agent_acts_is_flagged(night):
    from test_mind_consolidate import settled_task
    settled_task(night, title="first", dedup="research:first")
    await night.mind.consolidate()
    assert not night.mind.narrative()["fossilised"]

    def down(messages, context):
        raise RuntimeError("the model endpoint is down")
    night.router.answers["mind_consolidate_narrative"] = down
    for day in range(8):
        night.shift(hours=24)
        settled_task(night, title=f"task {day}", dedup=f"research:{day}")        # new evidence every day
        await night.mind.consolidate()
    assert night.mind.narrative()["fossilised"] is True
    night.router.answers.pop("mind_consolidate_narrative")                         # the default answer again
    from test_mind_consolidate import cite_uuids
    night.router.answers["mind_consolidate_narrative"] = cite_uuids
    night.shift(hours=24)
    await night.mind.consolidate()
    assert night.mind.narrative()["fossilised"] is False


# -- 8. round two: what an adversarial review found ---------------------------------------------------------

class Commitments:
    """A commitment store whose person list is five long pending promises and whose overdue list holds an
    older one: the producer appends the overdue row after the list."""

    def __init__(self):
        self.pending = [{"id": f"cm-pending-{index}", "status": "pending", "person_id": "p-01",
                         "description": f"pending promise {index} " + "p" * 560} for index in range(5)]
        self.overdue = [{"id": "cm-overdue-1", "status": "overdue", "person_id": "p-01",
                         "description": "send the signed lease back " + "o" * 200}]

    def list(self, **_):
        return list(self.pending)

    def get_overdue(self):
        return list(self.overdue)


@pytest.mark.parametrize("budget", [None, 1200, 3000])
async def test_an_overdue_commitment_outlives_rotation_before_any_pending_one(source_app, monkeypatch, budget):  # noqa: F811
    from colonymind.api.routers import host
    from colonymind.api.schemas.host import ContextAssembleRequest
    blocks.reset_settings()
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", "p-01")
    monkeypatch.setattr(host, "_commitment_store", Commitments())
    body = ContextAssembleRequest.model_validate({
        "identity": {"host_id": "fixture"}, "context": {"session_id": "s-1", "contact_id": "p-01"},
        "incoming_message": {"role": "user", "content": "status?"}})
    sections = await host._assemble_sections(body, viewer_person_id="p-01")
    fitted, _ = host._fit_sections(sections, budget=budget, where="test")
    commitments = next(section.body for section in fitted if section.id == "colonymind-commitments")
    assert "cm-overdue-1" in commitments
    pending = [commitments.index(f"cm-pending-{i}") for i in range(5) if f"cm-pending-{i}" in commitments]
    assert all(commitments.index("cm-overdue-1") < at for at in pending)


def test_a_block_first_stale_after_a_long_quiet_spell_is_not_a_fossil(caplog):
    ledger = BlockLedger(State(), fossil_days=7)
    ledger.observe("digest:p-04", "They work in room 4.", sources=["c1"], built_from=["c1"], now=T0)
    with caplog.at_level("WARNING", logger="colonymind.memory.blocks"):
        moved = ledger.observe("digest:p-04", "They work in room 4.", sources=["c1", "c2"], built_from=["c1"],
                               now=T0 + timedelta(days=8))
        rebuilt = ledger.observe("digest:p-04", "They work in room 9.", sources=["c1", "c2"], built_from=["c1", "c2"],
                                 rebuilt=True, now=T0 + timedelta(days=8))
    assert moved["stale"] and not moved["fossilised"] and not caplog.records
    assert not rebuilt["stale"] and not rebuilt["fossilised"]
    # the age of a fossil counts from the first move its writer did not answer
    ledger.observe("digest:p-04", "They work in room 9.", sources=["c1", "c2", "c3"], built_from=["c1", "c2"],
                   now=T0 + timedelta(days=20))
    assert not ledger.get("digest:p-04", T0 + timedelta(days=26))["fossilised"]
    assert ledger.get("digest:p-04", T0 + timedelta(days=27))["fossilised"]


async def test_a_narrative_answer_whose_every_line_is_rejected_is_no_rebuild(night):
    from test_mind_consolidate import cite_uuids, settled_task
    settled_task(night, title="first", dedup="research:first")
    await night.mind.consolidate()
    before = night.mind.mind_state.get("self.recent")
    assert before["text"]
    ledger = night.mind.consolidation.blocks
    built = ledger.get("self-narrative.recent")["built_at"]
    night.router.answers["mind_consolidate_narrative"] = lambda messages, context: {"lines": [
        {"text": "I did something I cannot show", "cites": ["no-such-record"]}]}
    night.shift(hours=24)
    settled_task(night, title="second", dedup="research:second")
    await night.mind.consolidate()
    assert night.mind.mind_state.get("self.recent")["text"] == before["text"]        # the validated lines stay
    assert ledger.get("self-narrative.recent")["built_at"] == built
    for day in range(7):
        night.shift(hours=24)
        settled_task(night, title=f"task {day}", dedup=f"research:{day}")
        await night.mind.consolidate()
    assert ledger.get("self-narrative.recent")["built_at"] == built and night.mind.narrative()["fossilised"] is True
    night.router.answers["mind_consolidate_narrative"] = cite_uuids
    night.shift(hours=24)
    await night.mind.consolidate()
    assert night.mind.narrative()["fossilised"] is False


def test_no_fit_outgrows_a_small_budget():
    turn_blocks = [Block("temporal-context", "Current Time", "Now."),
                   Block("colonymind-memory", "Relevant Memories", lines("memory", 11)),
                   Block("colonymind-goals", "Active Goals", "g" * 1000)]
    for budget in range(0, 1600):
        for sample in ([turn_blocks[2]], turn_blocks):
            shown, report = fit(sample, budget=budget)
            assert len(rendered(shown)) <= budget and report["chars_after"] <= budget, (budget, len(sample))


async def test_the_recipient_packet_never_outgrows_a_small_limit(context_app):
    for limit in (1, 10, 40, 74, 75, 120, 300):
        packet = await context_app[1].assemble_packet("p-02", query="hello", limit_chars=limit)
        assert len(packet) <= limit, limit


async def test_the_narratives_held_back_marker_is_never_a_citation(night):
    from test_mind_consolidate import settled_task
    for index in range(6):
        night.mind.add_interest(f"a long standing interest in topic number {index} " + "z" * 60)
    rows = [settled_task(night, title=f"{kind} {copy}", type_=kind, dedup=f"{kind}:{copy}")
            for kind in ("research", "check", "upkeep") for copy in range(2)]
    night.router.answers["mind_consolidate_narrative"] = lambda messages, context: {"lines": [
        {"text": f"I finished {r.description} and it held up under a second look " + "w" * 40, "cites": [r.id]}
        for r in rows[:4]]}
    await night.mind.consolidate()
    narrative = night.mind.narrative()
    assert narrative["held_back"] > 0
    consolidation = night.mind.consolidation
    assert narrative["cites"] and all(consolidation._ref_exists(ref) for ref in narrative["cites"]), narrative["cites"]
    assert not any("held back" in ref for ref in narrative["cites"])


def _legacy(ledger, block_id, now_seen):
    """Drop ``stale_since`` as a ledger written before it existed would have left the record."""
    ledger.observe(block_id, "They work in room 4.", sources=["c1"], built_from=["c1"], now=T0)
    ledger.observe(block_id, "They work in room 4.", sources=["c1", "c2"], built_from=["c1"], now=now_seen)
    key = BlockLedger.PREFIX + block_id
    record = json.loads(ledger.state.rows[key]["text"])
    record.pop("stale_since", None)
    ledger.state.rows[key]["text"] = json.dumps(record)
    return record


@pytest.mark.parametrize("sources", [["c1", "c2"], ["c1", "c2", "c3"]])
def test_observing_a_record_written_before_stale_since_keeps_its_fossil_age(sources, caplog):
    ledger = BlockLedger(State(), fossil_days=7)
    legacy = _legacy(ledger, "digest:p-04", T0 + timedelta(days=1))
    day8 = T0 + timedelta(days=8)
    assert ledger.get("digest:p-04", day8)["fossilised"]
    with caplog.at_level("WARNING", logger="colonymind.memory.blocks"):
        seen = ledger.observe("digest:p-04", "They work in room 4.", sources=sources, built_from=["c1"],
                              rebuilt=False, now=day8)
    assert seen["fossilised"] and seen["stale_since"] == legacy["sources_changed_at"]
    assert ledger.get("digest:p-04", day8 + timedelta(days=1))["fossilised"]
    assert ledger.get("digest:p-04", day8)["built_at"] == legacy["built_at"]
    assert len(caplog.records) == 1


def test_a_record_written_before_stale_since_is_cleared_by_a_rebuild_and_restarts_from_the_next_move():
    ledger = BlockLedger(State(), fossil_days=7)
    _legacy(ledger, "digest:p-04", T0 + timedelta(days=1))
    day8 = T0 + timedelta(days=8)
    rebuilt = ledger.observe("digest:p-04", "They work in room 9.", sources=["c1", "c2"], built_from=["c1", "c2"],
                             rebuilt=True, now=day8)
    assert not rebuilt["stale"] and not rebuilt["fossilised"] and rebuilt["stale_since"] is None
    moved = ledger.observe("digest:p-04", "They work in room 9.", sources=["c1", "c2", "c3"],
                           built_from=["c1", "c2"], now=day8 + timedelta(days=1))
    assert moved["stale_since"] == (day8 + timedelta(days=1)).isoformat() and not moved["fossilised"]


def test_a_current_record_written_before_stale_since_is_stale_from_its_first_unanswered_move():
    ledger = BlockLedger(State(), fossil_days=7)
    ledger.observe("digest:p-04", "They work in room 4.", sources=["c1"], built_from=["c1"], now=T0)
    moved = ledger.observe("digest:p-04", "They work in room 4.", sources=["c1", "c2"], built_from=["c1"],
                           now=T0 + timedelta(days=20))
    assert moved["stale_since"] == (T0 + timedelta(days=20)).isoformat() and not moved["fossilised"]
