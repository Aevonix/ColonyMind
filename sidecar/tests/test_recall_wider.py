"""A turn's recall is wider under the default settings: more records and more characters for the same reranker
work. Run with ``-s`` to read what the wider recall adds per turn in time."""
from __future__ import annotations

import json
import statistics

import pytest

from colonymind.memory import blocks, compass, search, selection
from colonymind.turns.source_vectors import SourceVectors
from recall_route import OWNER, assemble, recall_app
from test_recall_packing import Passing

RUNS = 9


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    monkeypatch.delenv("COLONYMIND_RECALL_CONTEXT_MAX_CHARS", raising=False)
    blocks.reset_settings()
    yield
    blocks.reset_settings()


async def history(setup, turns=40):
    """``turns`` more turns that match the fixture's query, about 400 characters each, indexed like the rest."""
    for n in range(turns):
        setup.ledger.record_source(f"note-{n}", contact_id=OWNER, session_id="session-a", messages=[
            {"role": "user", "content": f"Hydrofoil departure note {n}: "
                                        + f"the hydrofoil departure board moved to platform {n} again. " * 6}])
    projection = SourceVectors(setup.ledger, setup.store, setup.embedder)
    projection.backfill()
    for _ in range(4 * turns):
        if not await projection.process_one():
            break


def narrower(monkeypatch, tmp_path, items):
    """The settings of a narrower recall: ``items`` records within 4,000 characters of a 9,000-character context."""
    (tmp_path / "colonymind.yaml").write_text(json.dumps({"context": {"budget_chars": 9000, "recall_items": items}}))
    monkeypatch.setenv("COLONYMIND_RECALL_CONTEXT_MAX_CHARS", "4000")
    blocks.reset_settings()


async def timed(app):
    """One turn's recalled memory, and the median seconds of ``RUNS`` assemblies after a first one."""
    await assemble(app)
    runs = [await assemble(app) for _ in range(RUNS)]
    return runs[-1][0][compass.RECALLED], statistics.median(seconds for _, seconds in runs)


def records(section):
    return section["body"].count('"kind": "source_quote"')


# Without a scored rerank a turn's recall held five records; with a verified cutoff, up to context.recall_items.
@pytest.mark.parametrize("rerank, items", [("off", 5), ("on", 20)])
async def test_a_turns_recall_holds_more_records_and_more_characters(monkeypatch, tmp_path, rerank, items):
    setup = await recall_app(monkeypatch, tmp_path, reranker=Passing(), initiatives=0)
    await history(setup)
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", rerank)
    wide, wide_s = await timed(setup.app)
    sent = [call["documents"] for call in setup.reranker.calls]
    setup.reranker.calls.clear()
    most = blocks.settings()["recall_items"]
    narrower(monkeypatch, tmp_path, items)
    narrow, narrow_s = await timed(setup.app)
    assert 0 < records(narrow) <= items and len(narrow["body"]) <= 4000
    assert records(narrow) < records(wide) <= most == 20
    assert 4000 < len(wide["body"]) <= blocks.DEFAULT_BUDGET - search.REST_CHARS
    # The wider packet costs the reranker nothing more: it is sent the same documents, at most twenty a turn.
    assert [call["documents"] for call in setup.reranker.calls] == sent
    assert all(count <= selection.RERANK_SUBMISSION_LIMIT for count in sent) and bool(sent) == (rerank == "on")
    print(f"\nrecall per turn, rerank {rerank}: {records(narrow)} records and {len(narrow['body'])} characters in "
          f"{narrow_s * 1000:.1f} ms -> {records(wide)} records and {len(wide['body'])} characters in "
          f"{wide_s * 1000:.1f} ms (median of {RUNS} assemblies)")


def test_a_turns_recall_takes_what_the_context_budget_has_past_the_other_sections(monkeypatch):
    assert search.turn_recall_chars() == blocks.DEFAULT_BUDGET - search.REST_CHARS == 10_000
    assert search.turn_recall_chars(100_000) == 10_000              # a host's allowance never widens the ceiling
    assert search.turn_recall_chars(12_000) == 6_000
    # Under a small allowance recall keeps the budget an explicit search has.
    assert search.turn_recall_chars(9_744) == search.turn_recall_chars(1_200) == search.DEFAULT_RECALL_CHARS == 4000
    # A budget the environment names is the budget.
    monkeypatch.setenv("COLONYMIND_RECALL_CONTEXT_MAX_CHARS", "2500")
    assert search.turn_recall_chars() is None
