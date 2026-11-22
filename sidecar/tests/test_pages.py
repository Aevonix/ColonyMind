"""Pages on each person and on the agent itself: the store and the nightly rewrite.

Everything runs the real source ledger with a scripted model call.
"""
import pytest

from colonymind import recorder
from colonymind.memory import curated
from colonymind.mind.consolidate import NIGHT_TASKS, STAGES, Consolidation, Night
from colonymind.pages import night as pages_night
from colonymind.pages import store
from colonymind.turns import TurnIdempotencyLedger

OWNER, GUEST = "cid-owner", "cid-guest"


@pytest.fixture
def ledger(tmp_path):
    return TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")


def said(ledger, turn, contact, text, reply=None):
    ledger.record_source(turn, contact_id=contact, session_id="session-" + contact, derive_claims=False,
                         messages=[{"role": "user", "content": text},
                                   {"role": "assistant", "content": reply or f"Noted ({turn})."}])


def tonight():
    return Night(local_date="2026-11-21", started_at="", since="", note_id="test", budget=100000)


class Model:
    """Answers each page's call with the scripted lines of its subject kind and records what it was shown."""

    def __init__(self, person=(), own=()):
        self.lines, self.shown = {pages_night.PERSON_SYSTEM: list(person), pages_night.SELF_SYSTEM: list(own)}, []

    async def __call__(self, night, *, task, system, user, schema, max_output_tokens):
        assert task == pages_night.TASK
        self.shown.append(user)
        return {"lines": self.lines[system]}


def about(text, *cites):
    return {"kind": "about", "text": text, "cites": list(cites)}


@pytest.mark.asyncio
async def test_the_night_writes_only_lines_that_cite_a_turn_it_was_shown_and_keeps_boundaries(ledger):
    said(ledger, "turn-1", GUEST, "I get up at five and I hate small talk.")
    said(ledger, "turn-2", GUEST, "Please be blunt with me.")
    boundary = store.add(ledger, GUEST, "boundary", "No money talk with them.", ["turn-2"])
    model = Model(person=[about("Early riser; dislikes small talk.", "t1"), about("Wants bluntness.", "t2", "t1"),
                          about("Owns three boats."), about("Lives abroad.", "t99")])
    night = tonight()

    assert await pages_night.rewrite_pages(ledger, night, call=model) == 2          # the guest's page and the agent's own
    page = store.lines(ledger, GUEST)
    assert [(row["kind"], row["text"], row["sources"], row["writer"]) for row in page] == [
        ("boundary", "No money talk with them.", ["turn-2"], "agent"),
        ("about", "Early riser; dislikes small talk.", ["turn-1"], "night"),
        ("about", "Wants bluntness.", ["turn-2", "turn-1"], "night")]
    assert page[0]["id"] == boundary["id"] and night.rejected_lines == 2 and night.counts["pages"] == 2
    assert "t1 | They said: I get up at five and I hate small talk." in model.shown[-1]
    line = recorder.recent(1, kind="check", source="night.pages")[0]
    assert (line["result"], line["detail"]) == ("done", "pages-2") and line["took_ms"] is not None

    # Nobody has talked since: nothing is due and the model is not called.
    calls = len(model.shown)
    assert await pages_night.rewrite_pages(ledger, tonight(), call=model) == 0 and len(model.shown) == calls
    assert [row["text"] for row in store.lines(ledger, GUEST)] == [row["text"] for row in page]


@pytest.mark.asyncio
async def test_a_line_goes_when_a_turn_it_cites_is_forgotten(ledger):
    said(ledger, "turn-1", GUEST, "My sister is ill, keep it to yourself.")
    said(ledger, "turn-2", GUEST, "I like chess.")
    await pages_night.rewrite_pages(ledger, tonight(), call=Model(
        person=[about("Their sister is ill.", "t1"), about("Likes chess.", "t2")]))
    assert len(store.lines(ledger, GUEST)) == 2

    curated.forget_sources(ledger, GUEST, source_ids=["turn-1"])

    assert [row["text"] for row in store.lines(ledger, GUEST)] == ["Likes chess."]   # not read from that moment
    night = tonight()
    await pages_night.rewrite_pages(ledger, night, call=Model())
    assert night.counts["page_lines_removed"] == 1                                   # and removed by the next night


@pytest.mark.asyncio
async def test_the_agents_own_page_holds_its_tastes_from_the_night_and_from_itself(ledger):
    said(ledger, "turn-1", OWNER, "What music do you like?", reply="I love baroque fugues; I can't stand jingles.")
    mine = store.add(ledger, store.SELF, "like", "Long walks through data.", ["turn-live"])
    model = Model(own=[{"kind": "like", "text": "Baroque fugues.", "cites": ["t1"]},
                       {"kind": "dislike", "text": "Jingles.", "cites": ["t1"]},
                       {"kind": "about", "text": "Curious about music.", "cites": ["t1"]},
                       {"kind": "like", "text": "Sailing.", "cites": []}])
    await pages_night.rewrite_pages(ledger, tonight(), call=model)

    assert "t1 | The agent said: I love baroque fugues; I can't stand jingles." in model.shown[0]
    assert "They said" not in model.shown[0]                       # the agent's own page reads only what it said
    assert store.render(store.lines(ledger, store.SELF)) == (
        "About:\n- Curious about music.\nLikes:\n- Long walks through data.\n- Baroque fugues.\nDislikes:\n- Jingles.")
    assert store.lines(ledger, store.SELF, kinds=("like",))[0]["id"] == mine["id"]
    with pytest.raises(ValueError):
        store.add(ledger, GUEST, "like", "Chess.", ["turn-1"])     # a person's page has no tastes of the agent
    assert store.add(ledger, store.SELF, "like", "Unsourced.", []) is None


def test_a_model_that_gives_no_answer_leaves_the_page_due(ledger):
    import asyncio
    said(ledger, "turn-1", GUEST, "I like chess.")

    async def silent(night, **_):
        return None

    assert asyncio.run(pages_night.rewrite_pages(ledger, tonight(), call=silent, people=True)) == 0
    assert GUEST in pages_night.due(ledger)


def test_pages_are_one_stage_of_the_nightly_pass():
    assert "pages" in NIGHT_TASKS and callable(getattr(Consolidation, STAGES["pages"]))
