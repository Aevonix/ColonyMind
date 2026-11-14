"""One rerank call at a time: skip, never queue, and use a late answer while the turn is still being assembled.

A reranker server may finish a call its client gave up on, and the next call waits behind that work; the per-turn
judge after a missed recall then misses too. Recall and the judge now start a call only on an idle reranker, and a
call that finds it busy skips at once. Recall's call keeps the reranker until the server answers or the hold limit
passes; the judge's, which has no use for a late answer, only while the judge waits, so the next turn's recall is
never skipped behind it. An answer that arrives after
recall's deadline is still used while the turn's other lanes are being assembled, since it costs the turn nothing.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import time
from types import SimpleNamespace

import pytest

from colonymind.api.routers import host
from colonymind.api.schemas.host import ContextSection
from colonymind.memory import compass, recall_stats, rerank_gate, turn_budget
from colonymind.memory.selection import RecallSelector
from colonymind.memory.turn_budget import TurnBudget
from recall_route import ANSWER, Reranker, assemble, recall_app


class Server:
    """A reranker that serves one call at a time, like a GPU server that finishes an abandoned call first.
    A call with ``big`` in the query takes ``big_s``; any other ``small_s``."""

    def __init__(self, big_s=0.6, small_s=0.15, fail=False):
        self.big_s, self.small_s, self.fail = big_s, small_s, fail
        self.lock, self.calls = asyncio.Lock(), []

    async def rerank(self, query, documents, top_k=10):
        arrived = time.monotonic()
        async with self.lock:
            self.calls.append({"query": query, "arrived": arrived, "started": time.monotonic()})
            await asyncio.sleep(self.big_s if "big" in query else self.small_s)
            if self.fail:
                raise ConnectionError("the server went away")
            return [{"index": i, "score": 0.9 - i / 100} for i in range(len(documents))]


def candidates(n=8):
    return [{"id": f"m-{i}", "kind": "source_quote", "content": f"passage {i}", "relevance": 1 - i / 10}
            for i in range(n)]


async def timed(selector, query, deadline_ms, monkeypatch):
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_TIMEOUT_MS", str(deadline_ms))
    started = time.monotonic()
    rows = await selector.rerank(query, candidates(), 5, confidence_weighting=False, candidate_limit=20)
    return rows, time.monotonic() - started


async def judge(rerank_fn, query, timeout_s=1.0):
    """The per-turn judge (context selection) over sections that do not fit its budget; its report."""
    sections = [
        ContextSection(id="temporal-context", title="Current Time", body="Now.", priority=100),
        ContextSection(id="colonymind-stances", title="Your recorded views", priority=87,
                       body="\n".join(f"- view {n} " + "x" * 80 for n in range(8))),
    ]
    settings = compass.SelectionSettings(enabled=True, budget=200, timeout_s=timeout_s, candidates=48,
                                         due_soon=compass.timedelta(hours=24))
    _, report = await compass.select_context(sections, query, rerank_fn, settings=settings)
    return report


@pytest.fixture(autouse=True)
def rerank_on(monkeypatch):
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "on")
    monkeypatch.delenv("COLONYMIND_RECALL_RERANK_MIN_SCORE", raising=False)
    monkeypatch.delenv(rerank_gate.HOLD_ENV, raising=False)


@pytest.mark.asyncio
async def test_a_call_while_one_is_in_flight_is_skipped_not_queued(monkeypatch):
    server = Server()
    selector = RecallSelector(server.rerank)
    # The first call misses its deadline; the server goes on computing it.
    rows, elapsed = await timed(selector, "big question", 100, monkeypatch)
    assert all(row["rerank_status"] == "unavailable" for row in rows) and elapsed < 0.2
    # The next one does not queue behind that work: it skips at once and never reaches the server.
    rows, elapsed = await timed(selector, "small question", 300, monkeypatch)
    assert all(row["rerank_status"] == "unavailable" for row in rows) and elapsed < 0.05
    assert [call["query"] for call in server.calls] == ["big question"]
    # Once the server answered the abandoned call, the next call gets an idle server: its own 150 ms, not 650.
    await rerank_gate.gate_for(server.rerank).idle()
    rows, elapsed = await timed(selector, "small question", 300, monkeypatch)
    assert all(row["rerank_status"] == "scored" for row in rows)
    assert 0.14 <= elapsed < 0.25


@pytest.mark.asyncio
async def test_recall_and_the_judge_share_one_gate_per_reranker(monkeypatch):
    """While recall's call is in flight the judge skips at once, instead of waiting out its deadline behind it.
    Another reranker is not held up."""
    server = Server(big_s=0.5)
    await timed(RecallSelector(server.rerank), "big question", 50, monkeypatch)
    started = time.monotonic()
    report = await judge(server.rerank, "which view")
    assert report["status"] == "fallback" and report["reason"] == "busy"
    assert time.monotonic() - started < 0.05 and len(server.calls) == 1
    other = Server(small_s=0.01)
    report = await judge(other.rerank, "which view")
    assert report["status"] == "selected"


@pytest.mark.asyncio
async def test_a_late_answer_is_used_while_the_turn_is_still_being_assembled(monkeypatch):
    server = Server(small_s=0.3)
    budget = TurnBudget(wait_s=7.5, judge_s=1.0)
    with turn_budget.turn(budget):
        rows, elapsed = await timed(RecallSelector(server.rerank), "small question", 100, monkeypatch)
    assert all(row["rerank_status"] == "scored" for row in rows)
    assert 0.28 <= elapsed < 0.4  # it waited for the answer, not past it


@pytest.mark.asyncio
async def test_a_late_answer_is_not_waited_for_once_the_other_lanes_are_done(monkeypatch):
    server = Server(small_s=0.6)
    budget = TurnBudget(wait_s=7.5, judge_s=1.0)

    async def lanes():
        await asyncio.sleep(0.25)
        budget.lanes_assembled()
    with turn_budget.turn(budget):
        other_lanes = asyncio.create_task(lanes())
        rows, elapsed = await timed(RecallSelector(server.rerank), "small question", 100, monkeypatch)
        await other_lanes
    assert all(row["rerank_status"] == "unavailable" for row in rows)
    assert 0.24 <= elapsed < 0.35  # the lanes were done at 0.25 s; the answer (0.6 s) was not waited for
    assert rerank_gate.gate_for(server.rerank).busy()  # still held until the server answers
    await rerank_gate.gate_for(server.rerank).idle()


@pytest.mark.asyncio
async def test_on_the_route_a_late_answer_arriving_during_a_slow_lane_is_used(monkeypatch, tmp_path):
    """Recall's rerank misses its 200 ms deadline, but the turn is still reading a slow lane (the relationship
    context): the answer at 400 ms costs nothing and is used. With the lane fast, the same answer is not waited
    for and recall falls back."""
    for lane_s, scored in ((0.9, True), (0.0, False)):
        setup = await recall_app(monkeypatch, tmp_path / str(lane_s), reranker=Reranker(delay=0.4), rerank_ms=200)

        class Contacts:
            async def get(self, contact_id):
                task = asyncio.current_task()
                if task is None or task.get_name() != "colonymind-recall":
                    await asyncio.sleep(lane_s)
                return SimpleNamespace(trust_tier="owner", interaction_count=3, last_interaction_at=None,
                                       timezone=None, digest="")
        monkeypatch.setattr(host, "_contacts_store", Contacts())
        sections, elapsed = await assemble(setup.app)
        memory = sections["colonymind-memory"]["body"]
        assert (ANSWER in memory and '"rerank_status": "unavailable"' not in memory) is scored, (lane_s, memory)
        await rerank_gate.gate_for(setup.reranker.rerank).idle()


@pytest.mark.asyncio
async def test_a_cancelled_caller_leaves_the_call_holding_the_gate_until_the_server_answers(monkeypatch, caplog):
    """Cancellation safety: the caller ends at once; the server's call goes on and keeps the gate (no new call is
    piled on the server), then frees it; its failure is retrieved, never reported as an orphaned exception; the
    late-window waiter does not outlive the caller."""
    server = Server(small_s=0.3, fail=True)
    gate = rerank_gate.gate_for(server.rerank)
    budget = TurnBudget(wait_s=7.5, judge_s=1.0)
    with turn_budget.turn(budget):
        caller = asyncio.create_task(timed(RecallSelector(server.rerank), "small question", 50, monkeypatch))
        await asyncio.sleep(0.1)  # past the deadline, inside the late window
    started = time.monotonic()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert time.monotonic() - started < 0.05
    assert gate.busy()
    with caplog.at_level(logging.ERROR, logger="asyncio"):
        await gate.idle()
        await asyncio.sleep(0)
        gc.collect()
    assert not gate.busy() and "never retrieved" not in caplog.text
    pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    assert pending == []


class Surviving:
    """A server whose computation goes on when its client gives up: a client's cancellation never reaches it."""

    def __init__(self, work_s):
        self.work_s, self.running, self.most, self.jobs = work_s, 0, 0, []

    async def _compute(self):
        self.running += 1
        self.most = max(self.most, self.running)
        try:
            await asyncio.sleep(self.work_s)
        finally:
            self.running -= 1

    async def rerank(self, query, documents, top_k=10):
        job = asyncio.ensure_future(self._compute())
        self.jobs.append(job)
        await asyncio.shield(job)
        return [{"index": i, "score": 0.9} for i in range(len(documents))]


@pytest.mark.asyncio
async def test_single_flight_lasts_until_the_answer_or_the_hold_limit_whichever_comes_first(monkeypatch):
    """The gate holds a call until the server answers, and never past the hold limit: it knows when an answer
    arrives, not when a server stops working on a call its client gave up. Within the limit a second call is skipped
    and the server computes one call at a time; past it the gate is free and the next call reaches a server that is
    still computing the first, so the server computes two at once. That is the boundary of single flight."""
    monkeypatch.setenv(rerank_gate.HOLD_ENV, "300")
    server = Surviving(work_s=1.2)
    selector = RecallSelector(server.rerank)
    gate = rerank_gate.gate_for(server.rerank)
    await timed(selector, "first", 50, monkeypatch)
    rows, elapsed = await timed(selector, "second", 50, monkeypatch)
    assert all(row["rerank_status"] == "unavailable" for row in rows) and elapsed < 0.05
    assert gate.busy() and len(server.jobs) == 1 and server.most == 1
    await asyncio.sleep(0.45)  # past the hold limit
    assert not gate.busy() and server.running == 1  # freed, while the server still computes the first call
    await timed(selector, "third", 50, monkeypatch)
    assert len(server.jobs) == 2 and server.most == 2
    await asyncio.gather(*server.jobs)
    await gate.idle()


@pytest.mark.asyncio
async def test_a_judge_that_timed_out_frees_the_reranker_at_once_and_the_next_recall_reranks(monkeypatch):
    """The judge has no use for an answer after its deadline (it has no late window). Its call kept the reranker
    until the server answered, so the next quick turn's recall found it busy and skipped its rerank. The judge's call
    is now cancelled when the judge stops waiting, and the reranker is free at once: the next recall reranks."""
    server = Server(big_s=0.6, small_s=0.15)
    gate = rerank_gate.gate_for(server.rerank)
    started = time.monotonic()
    report = await judge(server.rerank, "big view", timeout_s=0.1)
    assert report["status"] == "fallback" and report["reason"] == "timeout"
    assert time.monotonic() - started < 0.3  # at its deadline, not the server's 0.6 s
    assert not gate.busy()
    rows, elapsed = await timed(RecallSelector(server.rerank), "small question", 300, monkeypatch)
    assert all(row["rerank_status"] == "scored" for row in rows)
    assert 0.14 <= elapsed < 0.25  # its own 150 ms: the server stopped the judge's call when its client did
    assert [call["query"] for call in server.calls] == ["big view", "small question"]


@pytest.mark.asyncio
async def test_a_cancelled_judge_frees_the_reranker_at_once(monkeypatch, caplog):
    """The turn is cancelled while the judge waits: the judge's call is cancelled with it, the reranker is free by
    the time the cancellation returns, the next recall reranks, and nothing is left running or unretrieved."""
    server = Server(big_s=0.6, small_s=0.15, fail=True)
    gate = rerank_gate.gate_for(server.rerank)
    caller = asyncio.create_task(judge(server.rerank, "big view", timeout_s=1.0))
    await asyncio.sleep(0.05)
    assert gate.busy()
    started = time.monotonic()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert time.monotonic() - started < 0.05
    assert not gate.busy()
    server.fail = False
    with caplog.at_level(logging.ERROR, logger="asyncio"):
        rows, elapsed = await timed(RecallSelector(server.rerank), "small question", 300, monkeypatch)
        await asyncio.sleep(0)
        gc.collect()
    assert all(row["rerank_status"] == "scored" for row in rows) and elapsed < 0.25
    assert "never retrieved" not in caplog.text
    pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    assert pending == []


@pytest.mark.asyncio
async def test_a_judge_that_timed_out_leaves_single_flight_to_a_server_that_computes_on(monkeypatch):
    """The boundary of the release, as past the hold limit: a server that goes on computing a call its client
    cancelled is reached by the next recall while it still computes the judge's call. Recall's own call that missed
    its deadline still keeps the reranker until the server answers."""
    server = Surviving(work_s=0.5)
    gate = rerank_gate.gate_for(server.rerank)
    report = await judge(server.rerank, "which view", timeout_s=0.05)
    assert report["reason"] == "timeout" and not gate.busy()
    rows, _ = await timed(RecallSelector(server.rerank), "next turn", 50, monkeypatch)
    assert all(row["rerank_status"] == "unavailable" for row in rows)
    assert len(server.jobs) == 2 and server.most == 2
    assert gate.busy()  # recall's own call keeps the reranker
    await asyncio.gather(*server.jobs)
    await gate.idle()


class JudgeHangs(Reranker):
    """Scores recall's candidates at once and never answers the judge within a test; one call at a time, and a call
    its client cancels ends there."""

    def __init__(self):
        super().__init__(capacity_one=True)

    async def _answer(self, query, documents):
        self.delay = None if any(doc.startswith("Pending Initiatives: ") for doc in documents) else 0.01
        return await super()._answer(query, documents)


@pytest.mark.asyncio
async def test_on_the_route_a_judge_that_timed_out_does_not_cost_the_next_turn_its_rerank(monkeypatch, tmp_path):
    """The release candidate's recall check: back-to-back turns whose judge misses its deadline. The next turn's
    recall found the reranker still held by that judge and skipped its rerank; now it reranks, every turn."""
    reranker = JudgeHangs()
    setup = await recall_app(monkeypatch, tmp_path, reranker=reranker, rerank_ms=1000, judge_ms=150)
    recall_stats.STATS.reset()
    for n in range(3):
        sections, _ = await assemble(setup.app, wait_ms=7500)
        memory = sections["colonymind-memory"]["body"]
        assert ANSWER in memory and '"rerank_status": "unavailable"' not in memory, (n, memory)
    assert [call["judge"] for call in reranker.calls] == [False, True] * 3
    assert not rerank_gate.gate_for(reranker.rerank).busy()
    snapshot = recall_stats.STATS.snapshot()
    assert snapshot["turns"]["scored"] == 3 and snapshot["turns"]["missed"] == 0
    assert snapshot["judge"]["fallback_by_reason"] == {"timeout": 3}
