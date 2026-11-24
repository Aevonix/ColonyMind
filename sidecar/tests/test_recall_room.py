"""A turn's recalled packet takes the room the assembled context really leaves it.

Recall packs within its own budget while the other lanes are assembled (``search.turn_recall_chars``: the context
budget less a share kept for the other sections, never less than an explicit search's budget). Once the context is
assembled, the room the host's allowance leaves after every other section at its whole size is known, and the
packet is packed again within it when that is more (``host._recall_room``). "Without the change" below is the same
assembly with that room unnamed, which is the assembly as it was before it."""
from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from colonymind.api.routers import host
from colonymind.api.schemas.host import ContextSection
from colonymind.memory import blocks, compass, producers, recall, search, selection
from colonymind.memory.recall import pack_memory_context
from colonymind.memory.search import MemoryPacket
from colonymind.memory.trace import NULL_TRACE
from recall_route import OWNER, recall_app
from test_recall_packing import Passing, assemble_body, post
from test_recall_wider import history

STOCK_ALLOWANCE = 9_744          # a host cap of 10,000 less the 256 the provider keeps for its clock


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    monkeypatch.delenv("COLONYMIND_RECALL_CONTEXT_MAX_CHARS", raising=False)
    blocks.reset_settings()
    yield
    blocks.reset_settings()


@contextmanager
def without_the_change(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(host, "_recall_room", lambda *args, **kwargs: None)
        yield


# --- The packing, section by section -----------------------------------------------------------------------------

CLOCK = ContextSection(id="temporal-context", title="Current Time", body="Current UTC time: 2026-03-05", priority=100)


def quote(n, size, **extra):
    return {"id": f"quote:{n:064x}", "kind": "source_quote", "source_uri": f"turn:t-{n}", "role": "user",
            "epistemic_state": "quotation", "content": f"Note {n}: " + "harbour schedule " * (size // 17), **extra}


def other_section(items):
    body = "\n".join(f"- open item {i}: " + "the east fence needs paint " * 3 for i in range(items))
    return ContextSection(id="colonymind-relationship", title="Relationship", body=body, priority=40)


def recalled_packet(rows, allowance, limit=20):
    budget = search.turn_recall_chars(allowance)
    selected, content = pack_memory_context(rows, limit=limit, max_chars=budget)
    memory = ContextSection(id=compass.RECALLED, title="Relevant Memories", body=content, priority=90)
    packet = SimpleNamespace(packet=MemoryPacket(selected, content, [], None, {}, []),
                             offered=selection.Offered(tuple(rows), limit, budget), ledger=None, contact_id="p",
                             session_id="s", served=None)
    return memory, packet


async def fitted(rows, allowance, others, limit=20):
    memory, packet = recalled_packet(rows, allowance, limit)
    sections, shown = await host._memory_within_allowance([CLOCK, memory, *others], packet, inline_chars=allowance,
                                                          trace=NULL_TRACE)
    by_id = {section.id: section for section in sections}
    return by_id.get(compass.RECALLED), shown, [s for s in sections if s.id != compass.RECALLED], packet


def cost(section):
    return blocks.cost(section.title or section.id, section.body)


def ids(rows):
    return [str(row["id"]) for row in rows]


async def test_the_room_is_what_the_other_sections_leave_not_a_fixed_share(monkeypatch):
    rows = [quote(n, 480) for n in range(20)]
    rooms, lengths = [], []
    for allowance, items in ((STOCK_ALLOWANCE, 2), (STOCK_ALLOWANCE, 12), (STOCK_ALLOWANCE, 30), (12_000, 12),
                             (16_000, 12)):
        others = [other_section(items)]
        memory, shown, kept_others, packet = await fitted(rows, allowance, others)
        _, counted = await compass.select_context([CLOCK, *others], "", None)
        room = (allowance - counted["kept_whole"] - counted["chars_before"]
                - blocks.cost("Relevant Memories", ""))
        kept, content = recall.fit_packet(rows, limit=20, max_chars=room)
        # The packet is what the packer gives within exactly that room, and the whole context fits the allowance.
        assert memory.body == content and ids(shown) == ids(kept)
        assert cost(memory) + sum(cost(section) for section in kept_others) <= allowance
        assert kept_others == [CLOCK, *others]
        # Recall's own budget for the turn is as it was; the packet takes more than it.
        assert len(memory.body) > len(packet.packet.content) and len(memory.body) > search.turn_recall_chars(allowance)
        rooms.append(room)
        lengths.append(len(memory.body))
    # One allowance, three sizes of the other sections: three rooms. No share is fixed in advance.
    assert len(set(rooms)) == len(rooms) and rooms[0] > rooms[1] > rooms[2] and lengths[0] > lengths[2]


async def test_the_packet_that_dropped_its_last_record_now_keeps_all_five(monkeypatch):
    # A turn whose five records just exceed recall's budget at the stock allowance: the last was left out with
    # thousands of characters free beside the other sections.
    rows = [quote(0, 1900), *(quote(n, 480) for n in range(1, 4)), quote(4, 520, atomic_evidence=True)]
    others = [other_section(12)]
    with without_the_change(monkeypatch):
        before, before_shown, before_others, _ = await fitted(rows, STOCK_ALLOWANCE, others)
    after, shown, after_others, _ = await fitted(rows, STOCK_ALLOWANCE, others)
    assert ids(before_shown) == ids(rows[:4]) and len(before.body) <= 4000
    assert ids(shown) == ids(rows) and len(after.body) > 4000
    assert after.body.startswith(before.body.rsplit("\n", 1)[0])  # the records it had, as it had them, first
    assert after_others == before_others


async def test_a_wider_packing_that_would_drop_a_kept_record_leaves_the_packet_as_it_is(monkeypatch):
    # Within more room the packer would take a large record it left out, and the packet's count limit would then
    # push out one it kept: the packet stays as it was.
    rows = [quote(0, 5000, atomic_evidence=True), quote(1, 300), quote(2, 300)]
    others = [other_section(2)]
    with without_the_change(monkeypatch):
        before, before_shown, _, _ = await fitted(rows, STOCK_ALLOWANCE, others, limit=2)
    wide, _ = recall.fit_packet(rows, limit=2, max_chars=STOCK_ALLOWANCE - 1000)
    assert ids(before_shown) == ids(rows[1:]) and ids(wide) == ids(rows[:2])     # not vacuous: it would drop one
    after, shown, _, _ = await fitted(rows, STOCK_ALLOWANCE, others, limit=2)
    assert after.body == before.body and ids(shown) == ids(before_shown)


class Told:
    """A trace that keeps the name of every call it is told."""

    def __init__(self):
        self.calls = []

    def __bool__(self):
        return True

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append(name)


@pytest.mark.parametrize("failure", [producers.Skipped("pool full"), RuntimeError("read failed")])
async def test_a_wider_packet_whose_sources_cannot_be_read_leaves_the_turn_as_it_is(monkeypatch, failure):
    # The wider packet adds one read, of its sources. When it fails, by the producer pool's own refusal or any
    # other error, the turn gets exactly what it gets without the change, and the trace has been told nothing.
    rows = [quote(0, 1900), *(quote(n, 480) for n in range(1, 4)), quote(4, 520, atomic_evidence=True)]
    others = [other_section(12)]
    memory, packet = recalled_packet(rows, STOCK_ALLOWANCE)
    sections = [CLOCK, memory, *others]
    trace = Told()
    wide, _ = await host._memory_within_allowance(sections, packet, inline_chars=STOCK_ALLOWANCE, trace=trace)
    assert len(wide[1].body) > len(memory.body) and trace.calls     # not vacuous: with the read, it widens
    original = producers.read

    async def failing(call, *args, **kwargs):
        if call is recall.packet_refs:
            raise failure
        return await original(call, *args, **kwargs)

    monkeypatch.setattr(producers, "read", failing)
    trace = Told()
    result, shown = await host._memory_within_allowance(sections, packet, inline_chars=STOCK_ALLOWANCE, trace=trace)
    with without_the_change(monkeypatch):
        expected, expected_shown = await host._memory_within_allowance(sections, packet, inline_chars=STOCK_ALLOWANCE,
                                                                       trace=Told())
    assert result == expected == sections and shown is packet.packet.selected and expected_shown is shown
    assert trace.calls == []
    # Without the guard the read's error would leave the assembly, and the turn would get no context at all.


def _as_fully(after, before):
    if not after.get("excerpt_truncated"):
        return True
    return bool(before.get("excerpt_truncated")) and str(after["content"]).startswith(str(before["content"]))


async def test_every_turn_keeps_every_record_and_every_other_section(monkeypatch):
    # Allowances from tight to the default ceiling, other sections from none to large, records whole, cut and
    # atomic: the packet never loses a record or shows one less fully, and no other section changes.
    rows = [quote(n, size, **({"atomic_evidence": True} if n % 4 == 3 else {}))
            for n, size in enumerate([2200, 300, 900, 650, 120, 1500, 400, 3000, 260, 700, 480, 1100] * 2)]
    widened = 0
    for allowance in range(1_000, 16_001, 500):
        for items in (0, 3, 15, 40):
            others = [other_section(items)] if items else []
            with without_the_change(monkeypatch):
                before, before_shown, before_others, _ = await fitted(rows, allowance, others)
            after, shown, after_others, _ = await fitted(rows, allowance, others)
            assert after_others == before_others
            if before is None:
                assert after is None
                continue
            held = {str(row["id"]): row for row in shown}
            assert all(str(row["id"]) in held and _as_fully(held[str(row["id"])], row) for row in before_shown)
            assert len(after.body) >= len(before.body)
            assert cost(after) + sum(cost(section) for section in after_others) <= allowance or after == before
            widened += len(after.body) > len(before.body)
    assert widened > 0


# --- Through the route ---------------------------------------------------------------------------------------------

class JudgeFails(Passing):
    """Recall's rerank is scored; the per-turn judge's call fails, so the selection falls back."""

    async def _answer(self, query, documents):
        if any(doc.startswith("Pending Initiatives: ") for doc in documents):
            raise RuntimeError("judge unavailable")
        return await super()._answer(query, documents)


def judged(monkeypatch):
    statuses = []
    original = compass.select_context

    async def spy(sections, query, rerank_fn, **kwargs):
        result = await original(sections, query, rerank_fn, **kwargs)
        if query:
            statuses.append(result[1]["status"])
        return result

    monkeypatch.setattr(compass, "select_context", spy)
    return statuses


def rest(response):
    return [(s["id"], s["title"], s["body"], s.get("citations")) for s in response["sections"]
            if s["id"] != compass.RECALLED]


def memory_of(response):
    return next((s for s in response["sections"] if s["id"] == compass.RECALLED), None)


def records(memory):
    return memory["body"].count('"kind": "source_quote"') if memory else 0


@pytest.fixture
def still_clock(monkeypatch):
    """The Current Time block says the same on both turns of a comparison, whatever the wall clock does."""
    from colonymind.util import temporal
    monkeypatch.setattr(temporal, "describe_agent_now", lambda *args, **kwargs: "Current time: fixed.")
    monkeypatch.setattr(temporal, "describe_now", lambda *args, **kwargs: "Current time: fixed.")


async def turn(app, monkeypatch, allowance, *, change=True):
    body = assemble_body(**({"inline_chars": allowance} if allowance is not None else {}), include_initiatives=True)
    if change:
        return await post(app, "/v1/host/context/assemble", body)
    with without_the_change(monkeypatch):
        return await post(app, "/v1/host/context/assemble", body)


@pytest.mark.parametrize("mode, status", [("scored", "selected"), ("within_budget", "within_budget"),
                                          ("fallback", "fallback")])
async def test_the_other_sections_are_byte_for_byte_what_they_are_without_the_change(monkeypatch, tmp_path, mode,
                                                                                    status, still_clock):
    setup = await recall_app(monkeypatch, tmp_path, reranker=JudgeFails() if mode == "fallback" else Passing(),
                             initiatives=5)
    await history(setup)
    if mode == "within_budget":
        monkeypatch.setenv(compass.BUDGET_ENV, "100000")
    statuses = judged(monkeypatch)
    widened, seen = 0, set()
    for allowance in (2_500, 6_000, STOCK_ALLOWANCE, 14_000, None):
        statuses.clear()
        before = await turn(setup.app, monkeypatch, allowance, change=False)
        after = await turn(setup.app, monkeypatch, allowance)
        assert len(statuses) == 2 and statuses[0] == statuses[1]      # the selection did the same both times
        seen.add(statuses[0])
        assert rest(after) == rest(before)
        assert records(memory_of(after)) >= records(memory_of(before))
        assert set(json.dumps(s, sort_keys=True) for s in after["budget"]["held_back"]) <= set(
            json.dumps(s, sort_keys=True) for s in before["budget"]["held_back"])
        budget = after["budget"]["budget"]
        assert sum(len(s["title"] or s["id"]) + len(s["body"]) + 6 for s in after["sections"]) <= budget
        widened += records(memory_of(after)) > records(memory_of(before))
        if allowance == STOCK_ALLOWANCE:
            assert statuses[0] == status and records(memory_of(after)) > records(memory_of(before))
    assert status in seen and widened >= 3                  # not vacuous: the stock allowance among them


@pytest.mark.parametrize("allowance", [STOCK_ALLOWANCE, None])
async def test_a_budget_the_environment_names_widens_nothing(monkeypatch, tmp_path, allowance, still_clock):
    setup = await recall_app(monkeypatch, tmp_path, reranker=Passing(), initiatives=5)
    await history(setup)
    wide = await turn(setup.app, monkeypatch, allowance)
    monkeypatch.setenv("COLONYMIND_RECALL_CONTEXT_MAX_CHARS", "4000")
    before = await turn(setup.app, monkeypatch, allowance, change=False)
    after = await turn(setup.app, monkeypatch, allowance)
    assert after["sections"] == before["sections"] and len(memory_of(after)["body"]) <= 4000
    assert records(memory_of(wide)) > records(memory_of(after))        # not vacuous: unnamed, it widens


async def test_an_explicit_search_and_the_first_pack_are_unchanged(monkeypatch, tmp_path, still_clock):
    setup = await recall_app(monkeypatch, tmp_path, reranker=Passing(), initiatives=5)
    await history(setup)
    first = []
    original = search.select_memory

    async def spy(*args, **kwargs):
        packet = await original(*args, **kwargs)
        first.append((kwargs.get("max_chars"), ids(packet.selected), packet.content))
        return packet

    monkeypatch.setattr(search, "select_memory", spy)
    query = {"identity": {"host_id": "fixture"}, "person_id": OWNER, "session_id": "session-b",
             "query": "hydrofoil departure"}
    with without_the_change(monkeypatch):
        searched_before = await post(setup.app, "/v1/host/memory/search", query)
    searched_after = await post(setup.app, "/v1/host/memory/search", query)
    assert searched_after == searched_before and searched_after["count"] == 5
    first.clear()
    before = await turn(setup.app, monkeypatch, STOCK_ALLOWANCE, change=False)
    after = await turn(setup.app, monkeypatch, STOCK_ALLOWANCE)
    # Recall's own packing of the turn: the same budget, the same records, the same text.
    assert len(first) == 2 and first[0] == first[1] and first[0][0] == 4000
    assert records(memory_of(after)) > records(memory_of(before))
    # The trace: the first packing's limit and budget, and every candidate's reason, as without the change; a
    # candidate the wider packet shows has none (it is in the packet), and the context carries the packet it names.
    assert after["trace"]["packet"]["limit"] == before["trace"]["packet"]["limit"]
    assert after["trace"]["packet"]["max_chars"] == before["trace"]["packet"]["max_chars"] == 4000
    shown = set(after["trace"]["selected"])
    assert set(before["trace"]["selected"]) < shown
    exits = {row["candidate"]: row["exit"] for row in before["trace"]["rows"]}
    for row in after["trace"]["rows"]:
        expected = None if row["candidate"] in shown else exits[row["candidate"]]
        assert row["exit"] == expected
    assert after["trace"]["section"] == before["trace"]["section"] == "in_context"
