"""A turn's recall spends only what its host waits for.

Hermes waits 8 s for the context and drops a later answer. Recall stacked a 5 s query-embedding limit (three
attempts under it), its rerank deadline and the judge's, which together could run past the host's wait. Each turn
now has one budget from the host's wait (``wait_ms``; the Hermes provider sends Hermes' wait less its margin): the
query embedding gets one attempt (1.5 s by default), recall's rerank
the lesser of its deadline and what is left less a reserve for the judge, and the judge what is left after that.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
import time
from contextlib import closing

import pytest

from colonymind.memory import compass, producers, turn_budget
from colonymind.memory.turn_budget import TurnBudget
from recall_route import ANSWER, Embedder, Reranker, assemble, recall_app, search


@pytest.fixture(autouse=True)
def defaults(monkeypatch):
    for name in (turn_budget.WAIT_ENV, turn_budget.EMBED_ENV, compass.SELECTION_ENV):
        monkeypatch.delenv(name, raising=False)


def test_the_budget_splits_the_hosts_wait():
    budget = TurnBudget(wait_s=7.5, judge_s=1.0)
    assert budget.reserve_s == 1.5
    assert budget.embed_timeout() == pytest.approx(1.5)
    assert budget.rerank_timeout(2.5) == pytest.approx(2.5, abs=0.01)
    budget.started -= 5.2  # 5.2 s spent: 7.5 - 5.2 - 1.5 = 0.8 s left for the rerank, the judge keeps its second
    assert budget.rerank_timeout(2.5) == pytest.approx(0.8, abs=0.01)
    assert budget.judge_timeout(1.0) == pytest.approx(1.0)
    budget.started -= 1.4  # 6.6 s spent, inside the reserve: recall starts nothing, the judge gets 0.4 s
    assert budget.rerank_timeout(2.5) == 0.0 and budget.embed_timeout() == 0.0
    assert budget.judge_timeout(1.0) == pytest.approx(0.4, abs=0.01)
    budget.started -= 1.0  # 7.6 s: nothing is left for anyone
    assert budget.judge_timeout(1.0) == 0.0


def test_the_worst_case_with_long_deadlines_ends_well_inside_the_wait():
    """A hung embedder, a hung reranker and a slow judge (synthetic deadlines: a 2.5 s rerank, a 1 s judge): each
    gets its whole deadline and the turn still ends inside six seconds of a 7.5 s wait."""
    budget = TurnBudget(wait_s=7.5, judge_s=1.0)
    embed = budget.embed_timeout()
    budget.started -= embed
    rerank = budget.rerank_timeout(2.5)
    budget.started -= rerank
    judge = budget.judge_timeout(1.0)
    assert (embed, round(rerank, 3), judge) == (1.5, 2.5, 1.0)
    assert embed + rerank + judge + turn_budget.RESERVE_MARGIN_MS / 1000 < 6.0


def test_a_host_names_its_wait_and_recall_comes_before_the_judge(monkeypatch):
    monkeypatch.setenv(compass.TIMEOUT_ENV, "1000")
    assert TurnBudget.for_host(None).wait_s == 7.5 and TurnBudget.for_host(None).judge_s == 1.0
    assert TurnBudget.for_host(12_000).wait_s == 7.5
    monkeypatch.setenv(turn_budget.WAIT_ENV, "5000")
    assert TurnBudget.for_host(None).wait_s == 5.0
    # A short wait: the reserve is at most a third of it, so recall still gets the larger part.
    short = TurnBudget.for_host(1500)
    assert short.reserve_s == pytest.approx(0.5) and short.recall_left() == pytest.approx(1.0, abs=0.01)
    monkeypatch.setenv(compass.SELECTION_ENV, "off")
    assert TurnBudget.for_host(None).judge_s == 0.0 and TurnBudget.for_host(None).reserve_s == 0.5
    monkeypatch.setenv(turn_budget.EMBED_ENV, "800")
    assert TurnBudget.for_host(None).embed_timeout() == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_a_hung_embedder_and_reranker_leave_the_turn_well_inside_the_hosts_wait(monkeypatch, tmp_path):
    """Measured on the route: the query embedding is abandoned after its one attempt (about 1.5 s, not 5 s),
    recall's rerank after its 2.5 s deadline, and the context still comes back in time, with the keyword
    evidence: the answer is the fourth record of the fallback's order, and the missed turn still carries it."""
    setup = await recall_app(monkeypatch, tmp_path, embedder=Embedder(query_s=None),
                             reranker=Reranker(delay=None), rerank_ms=2500, judge_ms=1000, distractors=3)
    sections, elapsed = await assemble(setup.app, wait_ms=7500)
    (query,) = setup.embedder.queries
    assert 1.4 <= query["ended"] - query["started"] <= 1.8
    assert elapsed < 6.0, elapsed
    memory = sections["colonymind-memory"]["body"]
    assert ANSWER in memory and '"rerank_status": "unavailable"' in memory  # the answer, unscored


@pytest.mark.asyncio
async def test_a_short_host_wait_is_kept(monkeypatch, tmp_path):
    """A 2 s host wait caps the embedding and starts no rerank after recall's share.

    Check the route's budget decisions: shared-runner scheduling and synchronous
    transport overhead are outside the timeout's control.
    """
    setup = await recall_app(monkeypatch, tmp_path, embedder=Embedder(query_s=None),
                             reranker=Reranker(delay=None), rerank_ms=2500, judge_ms=1000)
    allocations = []
    embed_timeout = TurnBudget.embed_timeout
    def embedding(budget):
        allowed = embed_timeout(budget)
        allocations.append((budget.wait_s, budget.judge_s, budget.reserve_s, allowed))
        return allowed
    monkeypatch.setattr(TurnBudget, 'embed_timeout', embedding)
    await assemble(setup.app, wait_ms=2000)
    (wait, judge, reserve, allowed), = allocations
    assert (wait, judge, reserve) == (2.0, 1.0, 2.0 / 3.0)
    assert 0 < allowed <= wait - reserve
    (query,) = setup.embedder.queries
    assert query['ended'] is not None  # the hung attempt was cancelled
    assert not [call for call in setup.reranker.calls if not call["judge"]]


@pytest.mark.asyncio
@pytest.mark.hardware_latency
async def test_with_time_to_spare_the_turn_is_scored_as_before(monkeypatch, tmp_path, hardware_latency):
    setup = await recall_app(monkeypatch, tmp_path, embedder=Embedder(query_s=0.05),
                             reranker=Reranker(delay=0.05), rerank_ms=2500, judge_ms=1000)
    cpu_started = time.thread_time()
    sections, elapsed = await assemble(setup.app, wait_ms=7500)
    cpu_elapsed = time.thread_time() - cpu_started
    memory = sections["colonymind-memory"]["body"]
    assert ANSWER in memory and "departure board" not in memory
    assert '"rerank_status": "unavailable"' not in memory
    hardware_latency([elapsed * 1000], [cpu_elapsed * 1000], threshold_ms=2500, statistic="max",
                     enforced=os.environ.get("GITHUB_ACTIONS") != "true")


@pytest.mark.asyncio
async def test_an_explicit_search_gets_one_embedding_attempt_too(monkeypatch, tmp_path):
    """No host waits on an explicit search, but the agent does: its query embedding is one attempt of the same
    length."""
    setup = await recall_app(monkeypatch, tmp_path, embedder=Embedder(query_s=None),
                             reranker=Reranker(delay=0.01), rerank_ms=2500)
    body, elapsed = await search(setup.app)
    (query,) = setup.embedder.queries
    assert 1.4 <= query["ended"] - query["started"] <= 1.8 and elapsed < 2.5
    assert "Friday at nine" in body["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["schema", "repeated-schema", "other", "deadline"])
async def test_worker_reprepares_a_schema_race_once_within_its_deadline(tmp_path, failure):
    """Force the surfaced SQLite error after a concurrent migration, independent of SQLite's auto-reprepare."""
    path = tmp_path / "memory.db"
    with closing(sqlite3.connect(path)) as writer, writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE evidence (body TEXT)")
        writer.execute("INSERT INTO evidence VALUES ('Room 9 now')")
    entered, migrated = threading.Event(), threading.Event()
    migration_errors, connections, attempts = [], [], []

    def migrate():
        try:
            assert entered.wait(3)
            with closing(sqlite3.connect(path)) as writer, writer:
                writer.execute("ALTER TABLE evidence ADD COLUMN revision INTEGER DEFAULT 1")
        except BaseException as error:
            migration_errors.append(error)
        finally:
            migrated.set()

    migration = threading.Thread(target=migrate)
    pool = producers.ProducerPool(workers=1)
    budget = TurnBudget(wait_s=5)

    def read_quote():
        conn = producers.connection(path)
        connections.append(conn)
        schema = conn.execute("PRAGMA schema_version").fetchone()[0]

        def execute():
            attempts.append(1)
            if len(attempts) == 1:
                entered.set()
                assert migrated.wait(3)
                assert conn.execute("PRAGMA schema_version").fetchone()[0] > schema
                if failure == "deadline":
                    conn.read.deadline = time.monotonic() - 1
            if len(attempts) == 1 or failure == "repeated-schema":
                error = sqlite3.OperationalError("forced error after schema change")
                if failure != "other":
                    error.sqlite_errorcode = sqlite3.SQLITE_SCHEMA
                    error.sqlite_errorname = "SQLITE_SCHEMA"
                raise error
            return conn.execute("SELECT body FROM evidence").fetchone()[0]

        return conn.checked(execute)

    migration.start()
    try:
        with turn_budget.turn(budget):
            if failure == "schema":
                assert await pool.read(read_quote) == "Room 9 now"
                assert budget.sync_read_failed == 0
            else:
                with pytest.raises(sqlite3.OperationalError):
                    await pool.read(read_quote)
                assert budget.sync_read_failed == 1
        assert len(attempts) == (2 if failure in {"schema", "repeated-schema"} else 1)

        def assert_closed():
            assert len(connections) == 1
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                sqlite3.Connection.execute(connections[0], "SELECT 1")

        await asyncio.wrap_future(pool.executor.submit(assert_closed))
    finally:
        entered.set()
        migration.join(3)
        pool.executor.shutdown(wait=True)
    assert not migration.is_alive() and not migration_errors
