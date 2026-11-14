"""End to end through the CLI: two local instances, two `colonymind twin run` processes, a
separate writer process, a re-seed, then a promote with writes in flight (docs/TWIN.md)."""

import json
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

SIDECAR = Path(__file__).parents[1]

WRITER = r'''
import os, sqlite3, sys, time
conn = sqlite3.connect(sys.argv[1], isolation_level=None)
conn.execute("PRAGMA journal_mode=WAL"); conn.execute("PRAGMA busy_timeout=5000")
conn.execute("CREATE TABLE IF NOT EXISTS c (n INTEGER PRIMARY KEY, blob BLOB)")
outbox = sqlite3.connect(sys.argv[2], isolation_level=None)
outbox.execute("PRAGMA journal_mode=WAL"); outbox.execute("PRAGMA busy_timeout=5000")
outbox.execute("CREATE TABLE IF NOT EXISTS turn_outbox (turn_id TEXT PRIMARY KEY, state TEXT)")
n, errors = 0, 0
stop = sys.argv[3]
while not os.path.exists(stop):
    try:
        conn.execute("INSERT INTO c(n, blob) VALUES (?, ?)", (n, os.urandom(200 + n % 3000)))
        outbox.execute("INSERT INTO turn_outbox VALUES (?, 'pending')", (f"turn-{n:06d}",))
        n += 1
    except sqlite3.Error:
        errors += 1
    time.sleep(0.002)
print(n, errors)
'''


def _cli(home, *args, check=None):
    result = subprocess.run([sys.executable, '-m', 'colonymind', '--instance', str(home), 'twin', *args],
                            capture_output=True, text=True, cwd=SIDECAR, timeout=120)
    if check is not None:
        assert result.returncode == check, result.stdout + result.stderr
    return result


def _status(home):
    result = _cli(home, 'status', '--json')
    return result.returncode, json.loads(result.stdout)


def _wait(predicate, timeout=30.0, what=''):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError(f'timed out waiting for {what}')


def test_two_instances_reseed_and_promote_with_writes_in_flight(tmp_path):
    primary, standby = tmp_path / 'primary', tmp_path / 'standby'
    primary.mkdir()
    standby.mkdir()
    (primary / 'instance-id').write_text('cli-twin-instance\n')
    outbox = tmp_path / 'body' / 'state' / 'colonymind-turn-outbox.sqlite3'
    outbox.parent.mkdir(parents=True)
    body = tmp_path / 'standby-body' / 'state'
    stop = tmp_path / 'stop-writer'
    _cli(standby, 'init', '--role', 'standby', '--external-dir', str(body), '--interval', '0.05', check=0)
    _cli(primary, 'init', '--role', 'primary', '--target', str(standby), '--external', str(outbox),
         '--interval', '0.05', '--max-lag', '20', check=0)
    code, report = _status(standby)
    assert code == 1 and 'runner_down' in report['alarms']
    writer = subprocess.Popen([sys.executable, '-c', WRITER, str(primary / 'counter.db'), str(outbox), str(stop)],
                              stdout=subprocess.PIPE, text=True)

    def created(path):
        try:
            return path.read_bytes()[:16] == b'SQLite format 3\x00'
        except OSError:
            return False

    # Both stores exist before the shipper's first discovery (the next one is 30 s later).
    _wait(lambda: created(primary / 'counter.db') and created(outbox), what='the writer to create its stores')
    runners = [subprocess.Popen([sys.executable, '-m', 'colonymind', '--instance', str(home), 'twin', 'run'],
                                cwd=SIDECAR, stderr=subprocess.PIPE, text=True) for home in (primary, standby)]
    try:
        _wait(lambda: _status(standby)[1].get('seq', 0) > 3, what='the standby to apply segments')
        _wait(lambda: _status(primary)[0] == 0 and _status(standby)[0] == 0, what='both sides free of alarms')
        before = _status(standby)[1]['generation']
        _cli(primary, 'reseed', '--reason', 'e2e test', check=0)
        _wait(lambda: _status(standby)[1]['generation'] != before and _status(standby)[1]['seq'] > 1,
              what='the standby to follow the re-seed')
        replica = _cli(standby, 'query', 'counter.db', 'SELECT count(*) AS n FROM c', check=0)
        assert json.loads(replica.stdout)['n'] > 0
        lags = [_status(primary)[1]['lag_seconds'] for _ in range(5)]
        started = time.monotonic()
        promoted = _cli(standby, 'promote', '--timeout', '30', check=0)
        promote_seconds = time.monotonic() - started
        assert runners[1].wait(timeout=30) == 0, 'the applier stops for the promotion'
        _wait(lambda: _status(primary)[1].get('fenced'), what='the old primary to be fenced')
    finally:
        stop.touch()
        written, writer_errors = map(int, writer.communicate(timeout=30)[0].split())
        for runner in runners:
            if runner.poll() is None:
                runner.send_signal(signal.SIGTERM)
                runner.wait(timeout=30)
    assert writer_errors == 0, 'the twin never makes a primary write fail'
    report = json.loads(promoted.stdout[:promoted.stdout.rindex('}') + 1])
    assert report['promoted']['forced'] is False
    ours = sqlite3.connect(standby / 'counter.db')
    theirs = sqlite3.connect(primary / 'counter.db')
    assert ours.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    promoted_rows = list(ours.execute('SELECT * FROM c ORDER BY n'))
    assert promoted_rows and promoted_rows == list(theirs.execute('SELECT * FROM c ORDER BY n'))[:len(promoted_rows)]
    turns = sqlite3.connect(body / outbox.name)
    assert turns.execute("SELECT count(*) FROM turn_outbox WHERE state='pending'").fetchone()[0] > 0
    assert (standby / 'instance-id').read_text() == 'cli-twin-instance\n'
    code, final = _status(standby)
    assert final['role'] == 'primary' and final['promoted']
    events = [json.loads(line)['event'] for line in (primary / '.twin' / 'events.jsonl').read_text().splitlines()]
    assert 'fenced' in events and events.count('generation') >= 2
    metrics = {'rows_written': written, 'rows_promoted': len(promoted_rows), 'rpo_rows': written - len(promoted_rows),
               'promote_seconds': round(promote_seconds, 3), 'primary_lag_samples': lags}
    (tmp_path / 'metrics.json').write_text(json.dumps(metrics))
    print('TWIN-E2E', json.dumps(metrics))


def test_init_refuses_a_target_inside_or_around_the_instance(tmp_path):
    home = tmp_path / 'primary'
    home.mkdir()
    for target in (home, home / 'twin', tmp_path):
        result = _cli(home, 'init', '--role', 'primary', '--target', str(target))
        assert result.returncode == 2 and 'another instance directory' in result.stderr
    assert _cli(home, 'status').returncode == 2          # still not configured
