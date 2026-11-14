"""Snapshot connections close promptly and warning-free, including across threads."""
from concurrent.futures import ThreadPoolExecutor
import gc
import sqlite3
import sys
import threading
import warnings

import pytest

from colonymind.turns.idempotency import TurnIdempotencyLedger


def test_snapshot_readers_close_across_threads(tmp_path, monkeypatch):
    gc.collect()
    unraisable = []
    monkeypatch.setattr(sys, 'unraisablehook', unraisable.append)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('error', ResourceWarning)
        ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
        barrier = threading.Barrier(2)

        def read(ledger):
            with ledger.read_snapshot() as first:
                first.execute('SELECT count(*) FROM turn_sources').fetchone()
                barrier.wait(timeout=5)
            with ledger.read_snapshot() as second:
                assert second is not first
            for connection in (first, second):
                with pytest.raises(sqlite3.ProgrammingError, match='closed'):
                    connection.execute('SELECT 1')
            return first

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(read, ledger) for _ in range(2)]
            connections = [future.result(timeout=10) for future in futures]
        assert connections[0] is not connections[1]
        del ledger
        gc.collect()
        for connection in connections:
            with pytest.raises(sqlite3.ProgrammingError, match='closed'):
                connection.execute('SELECT 1')
        del connection, connections, futures, pool
        gc.collect()
    assert unraisable == []
    assert caught == []
