"""Worker reads, accepted writes and private backups retain visible failure evidence."""
import asyncio
import logging
import sqlite3
import threading
from contextlib import closing
from types import SimpleNamespace

import pytest

from colonymind import health_counters
from colonymind.memory import producers, turn_budget
from test_plasticity_query_adapter_driver import runtime  # noqa: F401
from test_sync_context_producers import assembly  # noqa: F401


@pytest.mark.parametrize("stemmed", [False, True])
async def test_long_keyword_query_uses_worker_document_counts(tmp_path, monkeypatch, stemmed):
    from colonymind.turns import keywords
    path = tmp_path / "terms.db"
    terms = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mango".split()
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE VIRTUAL TABLE turn_source_search USING fts5(content)")
        conn.execute("CREATE VIRTUAL TABLE turn_source_stems USING fts5(content, tokenize='porter unicode61')")
        for table in ("turn_source_search", "turn_source_stems"):
            conn.executemany(f"INSERT INTO {table} VALUES (?)", [(terms[0],), (" ".join(terms),)])
    calls = []
    original = keywords._document_counts

    def counts(conn, query, **kwargs):
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 0
        result = original(conn, query, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(keywords, "_document_counts", counts)
    before = health_counters.snapshot().get("context_reads_failed", 0)

    def read():
        conn = producers.connection(path)
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        selected = keywords.query_terms(conn, " ".join(terms), stemmed=stemmed)
        assert selected == terms[1:]
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        return conn

    pool = producers.ProducerPool(workers=1)
    try:
        with turn_budget.turn(turn_budget.TurnBudget(wait_s=2)) as budget:
            await pool.read(read)
            assert budget.sync_read_failed == 0
        assert calls and health_counters.snapshot().get("context_reads_failed", 0) == before

        def writes_refused():
            conn = producers.connection(path, query_only=False)
            for sql in ("INSERT INTO turn_source_search VALUES ('fixture')", "CREATE TABLE forbidden (id)",
                        "CREATE INDEX forbidden_index ON turn_source_search_content(c0)"):
                with pytest.raises(sqlite3.OperationalError) as caught:
                    conn.execute(sql)
                assert caught.value.sqlite_errorname == "SQLITE_READONLY"

        with turn_budget.turn(turn_budget.TurnBudget(wait_s=2)):
            await pool.read(writes_refused)
    finally:
        pool.executor.shutdown(wait=True)


async def test_distinct_worker_errors_are_logged_once_per_turn(tmp_path, caplog):
    errors = [ValueError("private message"), sqlite3.OperationalError("private query")]
    errors[1].sqlite_errorname = "SQLITE_BUSY"
    other = sqlite3.OperationalError("private query")
    other.sqlite_errorname = "SQLITE_READONLY"
    errors.append(other)
    before = health_counters.snapshot().get("context_reads_failed", 0)
    with turn_budget.turn(turn_budget.TurnBudget(wait_s=1)) as budget:
        read = producers._Read(float("inf"), threading.Event())
        for error in errors * 2:
            read.failed(error)
        assert budget.sync_read_failed == 3
    records = [r for r in caplog.records if "context worker read failed" in r.message]
    assert len(records) == 3
    assert all(r.levelno == logging.WARNING and "private" not in r.message for r in records)
    assert "SQLITE_BUSY" in caplog.text and "SQLITE_READONLY" in caplog.text
    assert health_counters.snapshot()["context_reads_failed"] == before + 3


async def test_dropped_write_names_callable_and_counts_without_retry(caplog):
    calls = []

    def failed_write():
        calls.append(1)
        error = sqlite3.OperationalError("private statement")
        error.sqlite_errorname = "SQLITE_BUSY"
        raise error

    name = failed_write.__qualname__
    before = health_counters.snapshot()
    await turn_budget.finish_writes([(None, failed_write, (), {})])
    after = health_counters.snapshot()
    assert calls == [1]
    assert after["context_writes_dropped"] == before.get("context_writes_dropped", 0) + 1
    assert after[f"writes_dropped.{name}"] == before.get(f"writes_dropped.{name}", 0) + 1
    assert name in caplog.text and "OperationalError" in caplog.text and "SQLITE_BUSY" in caplog.text
    assert "private statement" not in caplog.text


async def test_ordinary_write_stays_on_the_loop():
    loop_thread = threading.get_ident()
    threads = []

    def write():
        threads.append(threading.get_ident())

    await turn_budget.finish_writes([(None, write, (), {})])
    assert threads == [loop_thread]


@pytest.mark.parametrize("method", ["record_trace", "record_selection"])
async def test_real_runtime_lock_writes_leave_the_loop(runtime, monkeypatch, method):
    driver = runtime.loaded["recall-test"].driver
    threads = []

    def evidence(turn_id):
        threads.append(threading.get_ident())
        return None

    monkeypatch.setattr(driver, "evidence", evidence)
    loop_thread = threading.get_ident()
    await turn_budget.finish_writes([(None, getattr(driver, method), ("turn", []), {})])
    assert len(threads) == 1 and threads[0] != loop_thread


async def test_real_learning_served_leaves_the_loop(monkeypatch):
    from colonymind.api.routers import host
    from colonymind.memory import learning

    threads = []

    def enabled():
        threads.append(threading.get_ident())
        return False

    monkeypatch.setattr(host, "_mind", lambda: None)
    monkeypatch.setattr(learning, "enabled", enabled)
    body = SimpleNamespace(context=SimpleNamespace(turn_id="turn", session_id="session"))
    loop_thread = threading.get_ident()
    await turn_budget.finish_writes([(None, host._learning_served, (None, "fixture", body, []), {})])
    assert len(threads) == 1 and threads[0] != loop_thread


async def test_real_owner_search_record_leaves_the_loop(assembly, runtime, monkeypatch):
    from colonymind.plasticity.learners.query_adapter import RecallRequest
    from test_sync_context_review import owner_search

    projection, driver = owner_search(runtime, monkeypatch)
    threads = []
    capture = driver.capture

    def captured(*args, **kwargs):
        threads.append(threading.get_ident())
        return capture(*args, **kwargs)

    monkeypatch.setattr(driver, "capture", captured)
    with turn_budget.turn(turn_budget.TurnBudget(wait_s=2)):
        await projection.search("fixture", contact_id="owner", session_id="session",
            recall_request=RecallRequest.create(runtime, owner=True, turn_id="turn"))
        writes = turn_budget.take_writes([])
    assert len(writes) == 1 and not threads
    loop_thread = threading.get_ident()
    await turn_budget.finish_writes(writes)
    assert len(threads) == 1 and threads[0] != loop_thread
    assert driver.queries.get("turn") is not None


def test_private_backup_writers_repair_permissive_modes(tmp_path):
    from colonymind.backup import _snapshot_databases, _snapshot_host_state
    from colonymind.init import backup_instance, write_hermes_config
    from colonymind.initiatives.store import InitiativeStore
    from colonymind.mind.tick import Mind

    state = tmp_path / "instance"
    state.mkdir()
    store = InitiativeStore(state)
    store.close()
    backup = state / "initiatives.db.backup"
    assert backup.stat().st_mode & 0o777 == 0o600
    backup.chmod(0o644)
    store = InitiativeStore(state)
    store.close()
    assert backup.stat().st_mode & 0o777 == 0o600

    nightly = SimpleNamespace(state_dir=state)
    Mind._backup_stores(nightly, "2026-11-02")
    upgrade = backup_instance(state)
    _snapshot_databases(state, state / "snapshot")
    source = tmp_path / "source"
    source.mkdir()
    (source / "record").write_text("synthetic")
    _snapshot_host_state([str(source)], state / "copy")
    config = tmp_path / "synthetic.yaml"
    config.write_text("enabled: true\n")
    write_hermes_config(config, {"enabled": False}, backup_dir=upgrade)
    for root in (state / "backups", state / "snapshot", state / "copy"):
        for path in (root, *root.rglob("*")):
            assert path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600), path


@pytest.mark.parametrize("encrypted", [False, True])
def test_full_backup_archive_is_private(tmp_path, encrypted):
    import tarfile
    from colonymind.backup import create_full_backup
    state = tmp_path / "instance"
    state.mkdir()
    (state / "identity.yaml").write_text("agent: synthetic\n")
    with closing(sqlite3.connect(state / "records.db")) as conn:
        conn.execute("CREATE TABLE records (id)")
    archive = create_full_backup(state, tmp_path / "nested" / "archives",
                                 passphrase=b"synthetic" if encrypted else None)
    assert archive.stat().st_mode & 0o777 == 0o600
    assert archive.parent.stat().st_mode & 0o777 == 0o700
    if not encrypted:
        with tarfile.open(archive) as tar:
            assert all(item.mode == (0o700 if item.isdir() else 0o600) for item in tar.getmembers())
