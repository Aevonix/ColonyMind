"""Group turns: a turn in a group chat recalls that chat's own history only and carries one line that says so. A
turn in a direct chat is never narrowed."""
import asyncio
from contextlib import closing
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from colonymind.api.errors import install_exception_handlers
from colonymind.api.routers import host
from colonymind.memory import blocks, compass, turn_budget
from colonymind.pages import store as pages
from colonymind.pages import turn
from colonymind.turns import get_turn_idempotency_ledger, group_chats
from recall_route import ANSWER, OWNER, assemble, recall_app
from test_recall_exit_counter import SESSION, counter  # noqa: F401  (the capture plugin, loaded from its file)

CREW = {"chat_type": "group", "chat": "signal:crew"}
NOTICE = ("This turn is in a group chat. Recall read this group's own history only; the rest of your memory was "
          "withheld.")
PICNIC, FERRY = "The harbour picnic is on Saturday.", "The harbour ferry leaves at seven."


@pytest.fixture
def world(monkeypatch, tmp_path):
    """Casey said one thing in a direct chat and one in the crew's group chat, whose first session is noted."""
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    for name in ("_contacts_store", "_telemetry", "_reranker", "_context_recall_selector", "_comms_log"):
        monkeypatch.setattr(host, name, None)
    from colonymind import vector
    monkeypatch.setattr(vector, "get_store", lambda: None)
    monkeypatch.setattr(vector, "get_pipeline", lambda: None)
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(host.router)
    ledger = get_turn_idempotency_ledger(tmp_path)
    for turn_id, session, said in (("t-direct", "s-direct", PICNIC), ("t-crew", "s-crew-1", FERRY)):
        ledger.record_source(turn_id, contact_id="casey", session_id=session, derive_claims=False, messages=[
            {"role": "user", "content": said}, {"role": "assistant", "content": "Noted."}])
    group_chats.note("signal:crew", "s-crew-1")
    return SimpleNamespace(app=app, ledger=ledger)


async def assembled(app, session="s-now", metadata=None, message="What about the harbour?"):
    """The sections of casey's turn, in the order sent, once the turn's accepted writes have run."""
    context = {"contact_id": "casey", "session_id": session, **({"metadata": metadata} if metadata else {})}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/host/context/assemble", json={
            "identity": {"host_id": "test-host"}, "context": context,
            "incoming_message": {"role": "user", "content": message}})
    assert response.status_code == 200, response.text
    if turn_budget._PENDING:
        await asyncio.gather(*tuple(turn_budget._PENDING))
    return {section["id"]: section["body"] for section in response.json()["sections"]}


def noted(ledger):
    with closing(ledger._read_connect()) as conn:
        return sorted(tuple(row) for row in conn.execute("SELECT chat, session_id FROM group_chat_sessions"))


async def test_a_group_turn_recalls_the_groups_own_history_and_says_so(world):
    pages.add(world.ledger, "casey", "about", "Takes the early ferry.", ["t-crew"])
    pages.add(world.ledger, "casey", "about", "Plans picnics.", ["t-direct"])
    pages.add(world.ledger, "casey", "boundary", "No calls at sea.", ["t-crew", "t-direct"])
    pages.add(world.ledger, pages.SELF, "like", "Ferry rides.", ["t-crew"])
    pages.add(world.ledger, pages.SELF, "like", "Picnics.", ["t-direct"])
    group = await assembled(world.app, "s-crew-2", CREW)
    # The clock, the one line, then what was read from the group's own history: nothing else.
    assert list(group) == ["temporal-context", "colonymind-recall-narrowed", "colonymind-memory",
                           turn.PERSON_ID, turn.SELF_ID]
    assert group["colonymind-recall-narrowed"] == NOTICE
    assert FERRY in group["colonymind-memory"] and "picnic" not in group["colonymind-memory"]
    # A page line is given when every turn it cites took place in the group.
    assert group[turn.PERSON_ID] == "- About them: Takes the early ferry."
    assert group[turn.SELF_ID] == "- You like: Ferry rides."
    # The same person's turn in a direct chat is not narrowed and carries no such line.
    for metadata in (None, {"chat_type": "dm", "chat": "signal:casey"}, {"chat_type": "private"}):
        direct = await assembled(world.app, "s-now", metadata)
        assert "colonymind-recall-narrowed" not in direct
        assert FERRY in direct["colonymind-memory"] and PICNIC in direct["colonymind-memory"]
        assert direct[turn.PERSON_ID] == ("- Your boundary with them: No calls at sea.\n"
                                          "- About them: Takes the early ferry.\n- About them: Plans picnics.")
        assert direct[turn.SELF_ID] == "- You like: Ferry rides.\n- You like: Picnics."
    assert noted(world.ledger) == [("signal:crew", "s-crew-1"), ("signal:crew", "s-crew-2")]


async def test_a_session_joins_its_groups_history_once_a_turn_there_is_accepted(world):
    # Another group has none of the crew's history, and its first turn recalls nothing.
    first = await assembled(world.app, "s-deck-1", {"chat_type": "supergroup", "chat": "telegram:-1007"})
    assert list(first) == ["temporal-context", "colonymind-recall-narrowed"]
    assert ("telegram:-1007", "s-deck-1") in noted(world.ledger)
    world.ledger.record_source("t-deck", contact_id="casey", session_id="s-deck-1", derive_claims=False, messages=[
        {"role": "user", "content": "The harbour deck is closed for repairs."}])
    # Its next session reads what was said there, and still nothing from the crew's chat or the direct one.
    later = await assembled(world.app, "s-deck-2", {"chat_type": "supergroup", "chat": "telegram:-1007"})
    assert "deck is closed" in later["colonymind-memory"]
    assert "ferry" not in later["colonymind-memory"] and "picnic" not in later["colonymind-memory"]
    # The crew's chat does not read the other group's history either.
    crew = await assembled(world.app, "s-crew-2", CREW)
    assert FERRY in crew["colonymind-memory"] and "deck" not in crew["colonymind-memory"]


async def test_a_group_kind_with_no_chat_reads_the_turns_own_session_only(world):
    own = await assembled(world.app, "s-crew-1", {"chat_type": "group"})
    assert own["colonymind-recall-narrowed"] == NOTICE
    assert FERRY in own["colonymind-memory"] and "picnic" not in own["colonymind-memory"]
    assert list(await assembled(world.app, "s-elsewhere", {"chat_type": "group"})) == [
        "temporal-context", "colonymind-recall-narrowed"]
    assert noted(world.ledger) == [("signal:crew", "s-crew-1")]   # no chat was named, so no session was noted


async def test_the_owner_alone_is_never_narrowed_and_the_owners_group_turn_is(monkeypatch, tmp_path):
    world = await recall_app(monkeypatch, tmp_path)

    def context(**metadata):
        return {"context": {"contact_id": OWNER, "session_id": "session-b",
                            "metadata": {"owner_private_turn": True, **metadata}}}

    alone, _ = await assemble(world.app, **context())
    assert ANSWER in alone["colonymind-memory"]["body"]
    assert "colonymind-recall-narrowed" not in alone and "colonymind-initiatives" in alone
    # In a group the owner's turn is narrowed like anyone's: before the chat has a history it recalls nothing.
    unknown, _ = await assemble(world.app, **context(**CREW))
    assert list(unknown) == ["temporal-context", "colonymind-recall-narrowed"]
    group_chats.note("signal:crew", "session-a")
    known, _ = await assemble(world.app, **context(**CREW))
    assert list(known) == ["temporal-context", "colonymind-recall-narrowed", "colonymind-memory"]
    assert ANSWER in known["colonymind-memory"]["body"]


def test_the_turns_group_chat_comes_from_its_chat_kind():
    for metadata in ({}, {"chat": "signal:crew"}, {"chat_type": ""}, {"chat_type": "dm", "chat": "signal:casey"},
                     {"chat_type": " Private "}, {"chat_type": "direct"}, {"chat_type": None}):
        assert group_chats.of_turn(metadata) is None
    assert group_chats.of_turn(CREW) == "signal:crew"
    assert group_chats.of_turn({"chat_type": "Channel", "chat": " slack:C42 "}) == "slack:C42"
    assert group_chats.of_turn({"chat_type": "group"}) == group_chats.of_turn({"chat_type": "group", "chat": 7}) == ""


def test_a_candidate_stays_when_every_turn_it_names_took_place_in_the_group(world):
    assert group_chats.turns_of({
        "source_turn_id": "a", "source_turn_ids": ["b"], "source_uri": "turn:c",
        "_annotation_source_refs": [{"source_id": "d"}], "_conversation_source_refs": [{"source_id": "e"}, "f"],
        "source_anchors": [{"source_id": "g"}, {}], "source_read": {"source_id": "h"}}) == set("abcdegh")
    rows = [{"source_turn_id": "t-crew"}, {"source_turn_id": "t-direct"}, {"text": "A curated entry."},
            {"source_turn_id": "t-crew", "_annotation_source_refs": [{"source_id": "t-direct"}]},
            {"source_turn_ids": ["t-crew", "t-gone"]}]
    assert group_chats.within(world.ledger, "signal:crew", "s-now", rows) == rows[:1]
    # The turn's own session counts as the group's before it is noted, and another chat's history does not.
    assert group_chats.within(world.ledger, "signal:deck", "s-direct", rows) == rows[1:2]
    assert group_chats.within(world.ledger, "", "s-now", rows) == []
    assert group_chats.within(None, "signal:crew", "s-now", rows) == []
    assert group_chats.within(world.ledger, "signal:crew", "s-now", [{"text": "Names no turn."}]) == []


def test_the_one_line_is_always_given_whole_ahead_of_the_memory():
    notice = group_chats.NOTICE
    assert notice["body"] == NOTICE and "\n" not in notice["body"]
    assert notice["id"] in blocks.ALWAYS_WHOLE and notice["id"] in compass.PINNED_SECTIONS
    assert blocks.RANK.index(notice["id"]) < blocks.RANK.index("colonymind-memory")
    assert 100 > notice["priority"] > 90   # under the clock, over the memory: the order they are sent in


def test_a_group_turn_is_left_out_of_the_recall_alarm(counter):
    capture = counter.Capture.__new__(counter.Capture)
    capture._recall_owner_handles = {}
    kinds = {"group": True, "supergroup": True, "channel": True, "": False, "dm": False, "direct": False,
             "private": False}
    for number, (kind, excluded) in enumerate(kinds.items()):
        info = SimpleNamespace(session_id=SESSION, turn_id=str(number), owner=True, host=None, platform="cli",
                               sender_id="", chat_type=kind)
        capture._begin_recall_exit(info, {})
        assert counter._turns[(SESSION, str(number))].excluded is excluded, kind
