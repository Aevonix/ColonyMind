"""Review probes against the actual ledger; child alarms bound deadlock repros."""
from concurrent.futures import ThreadPoolExecutor
import copy
import faulthandler
import gc
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import warnings

import pytest
from colonymind.turns.idempotency import TurnIdempotencyLedger


def test_child_cleanup_does_not_wait_for_parent_snapshot(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'fork.db')
    entered, release = threading.Event(), threading.Event()
    parent_connections = []

    def read():
        with ledger.read_snapshot() as conn:
            parent_connections.append(conn)
            entered.set()
            assert release.wait(10)
            assert conn.execute('SELECT 1').fetchone()[0] == 1

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(read)
        assert entered.wait(5)
        try:
            pid = os.fork()
            if pid == 0:
                def timed_out(*_):
                    faulthandler.dump_traceback()
                    os._exit(71)
                signal.signal(signal.SIGALRM, timed_out)
                signal.alarm(2)
                try:
                    with ledger.read_snapshot() as conn:
                        assert conn is not parent_connections[0]
                        assert conn.execute('SELECT 1').fetchone()[0] == 1
                    del conn
                    print('child read used and closed a fresh connection', flush=True)
                    os._exit(0)
                except BaseException:
                    import traceback
                    traceback.print_exc()
                    os._exit(72)
            _, status = os.waitpid(pid, 0)
        finally:
            release.set()
        future.result(5)
    assert os.waitstatus_to_exitcode(status) == 0, 'child cleanup hung; alarm dumped its stack and exited 71'


def test_gc_from_another_thread(tmp_path, monkeypatch):
    gc.collect()
    unraisable = []
    monkeypatch.setattr(sys, 'unraisablehook', unraisable.append)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ResourceWarning)
        ledger = TurnIdempotencyLedger(tmp_path / 'gc-thread.db')
        with ledger.read_snapshot() as connection:
            assert connection.execute('SELECT 1').fetchone()[0] == 1
        ledger.cycle = ledger
        del ledger
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(gc.collect).result(timeout=5)
        with pytest.raises(sqlite3.ProgrammingError, match='closed'):
            connection.execute('SELECT 1')
    assert not caught and not unraisable


def test_interpreter_shutdown(tmp_path):
    script = '''
import sys
from colonymind.turns.idempotency import TurnIdempotencyLedger
ledger = TurnIdempotencyLedger(sys.argv[1])
with ledger.read_snapshot() as conn:
    assert conn.execute('SELECT 1').fetchone()[0] == 1
ledger.cycle = ledger
del conn
'''
    proc = subprocess.run([sys.executable, '-B', '-Walways::ResourceWarning', '-c', script,
                           str(tmp_path / 'shutdown.db')], capture_output=True, text=True, timeout=5)
    assert proc.returncode == 0 and proc.stderr == '', (proc.returncode, proc.stderr)


def test_same_owner_copies_nested_and_threads(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'copies.db')
    other = copy.copy(ledger)
    barrier = threading.Barrier(4)
    active, mutex = set(), threading.Lock()

    def read():
        first = previous = None
        for _ in range(30):
            with ledger.read_snapshot() as conn:
                with mutex:
                    assert id(conn) not in active
                    active.add(id(conn))
                if first is None:
                    first = conn
                assert conn is not previous
                previous = conn
                with other.read_snapshot() as inner:
                    assert inner is not conn
                    assert inner.execute('SELECT 1').fetchone()[0] == 1
                barrier.wait(timeout=5)
                with mutex:
                    active.remove(id(conn))
        return first

    with ThreadPoolExecutor(max_workers=4) as pool:
        connections = [f.result(10) for f in [pool.submit(read) for _ in range(4)]]
    assert len({id(c) for c in connections}) == 4


def test_cyclic_cleanup_from_gc_thread(tmp_path):
    script = '''
import gc, sys, threading
from colonymind.turns.idempotency import TurnIdempotencyLedger
ledger = TurnIdempotencyLedger(sys.argv[1])
with ledger.read_snapshot() as conn:
    assert conn.execute('SELECT 1').fetchone()[0] == 1
ledger.cycle = ledger
del ledger, conn
worker = threading.Thread(target=gc.collect)
worker.start()
worker.join(3)
assert not worker.is_alive()
'''
    proc = subprocess.run([sys.executable, '-B', '-Werror::ResourceWarning', '-c', script,
                           str(tmp_path / 'cyclic-gc.db')], capture_output=True, text=True, timeout=5)
    assert proc.returncode == 0 and proc.stderr == '', (proc.returncode, proc.stderr)
