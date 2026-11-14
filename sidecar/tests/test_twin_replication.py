"""Primary and standby together: two instance directories, real stores (docs/TWIN.md)."""

import json
import os
import random
import shutil
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from colonymind.contacts.comms import CommsLog
from colonymind.twin import config as twin_config
from colonymind.twin.config import TwinConfig, save
from colonymind.twin.primary import Shipper
from colonymind.twin.standby import PromoteError, Standby, connect, tree_path
from colonymind.twin.transport import RsyncTransport

TurnOutbox = pytest.importorskip('colonymind_hermes.capture').TurnOutbox


def _instance(tmp_path):
    primary, standby = tmp_path / 'primary', tmp_path / 'standby'
    primary.mkdir()
    standby.mkdir()
    (primary / 'instance-id').write_text('twin-test-instance\n')
    (primary / 'identity.yaml').write_text('agent: {name: Test}\n')
    (primary / 'colonymind.yaml').write_text('sidecar: {port: 7777}\n')
    (primary / 'backups' / '20260101T000000Z').mkdir(parents=True)
    sqlite3.connect(primary / 'backups' / '20260101T000000Z' / 'old.db').execute('CREATE TABLE t (x)')
    (primary / 'media').mkdir()
    (primary / 'media' / 'original.bin').write_bytes(os.urandom(3 << 20))
    outbox = tmp_path / 'body' / 'state' / 'colonymind-turn-outbox.sqlite3'
    outbox.parent.mkdir(parents=True)
    return primary, standby, outbox


def _configs(primary, standby, outbox, tmp_path, **overrides):
    fast = dict(push_interval=0.001, heartbeat_interval=0.05, vector_interval=0.001, files_interval=0.001,
                rediscover_interval=0.001, interval=0.01)
    cp = TwinConfig(role='primary', target=str(standby), external=[str(outbox)],
                    **{**fast, 'transfer_wait': 30.0, **overrides})
    cs = TwinConfig(role='standby', external_dir=str(tmp_path / 'standby-body' / 'state'), **fast)
    save(primary, cp)
    save(standby, cs)
    return cp, cs


def _counter(home, name='counter.db', wal=True):
    conn = sqlite3.connect(home / name, isolation_level=None, check_same_thread=False)
    if wal:
        conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA busy_timeout=5000')
    conn.execute('CREATE TABLE IF NOT EXISTS c (n INTEGER PRIMARY KEY, blob BLOB)')
    return conn


def _vectors(home, rows=3):
    import lancedb
    import pyarrow as pa
    from colonymind.vector.store import _base_schema
    db = lancedb.connect(str(home / 'lancedb'))
    schema = _base_schema(8)
    table = db.create_table('conversations', schema=schema)

    def add(start, count):
        data = {name: [None] * count for name in schema.names}
        data['id'] = [f'v{start + i}' for i in range(count)]
        data['text'] = [f'text {start + i}' for i in range(count)]
        data['vector'] = [[random.random() for _ in range(8)] for _ in range(count)]
        table.add(pa.table(data, schema=schema))
    add(0, rows)
    return table, add


def _sync(shipper, standby, rounds=4):
    for _ in range(rounds):
        shipper.tick()
        standby.cycle()


def _rows(conn, sql):
    return list(conn.execute(sql))


def test_replicas_follow_every_store_and_serve_consistent_reads(tmp_path):
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path)
    comms = CommsLog(primary / 'colonymind-comms.db')
    outbox = TurnOutbox(outbox_path)
    counter = _counter(primary)
    rollback = _counter(primary, 'rollback.db', wal=False)
    (primary / 'memory-state').mkdir()
    nested = _counter(primary / 'memory-state', 'commitments.db')
    table, add_vectors = _vectors(primary)
    shipper, standby = Shipper(primary, cp), Standby(standby_home, cs)
    for i in range(40):
        comms.log('p-1', channel='system', summary=f'e{i}')
        outbox.enqueue(f'turn-{i}', {'turn_id': f'turn-{i}', 'text': 'x' * i})
        counter.execute('INSERT INTO c(n, blob) VALUES (?, ?)', (i, os.urandom(random.randint(1, 5000))))
        rollback.execute('INSERT INTO c(n) VALUES (?)', (i,))
        nested.execute('INSERT INTO c(n) VALUES (?)', (i,))
        if i % 10 == 0:
            add_vectors(100 + i, 2)
        _sync(shipper, standby, rounds=1)
    _sync(shipper, standby)
    stores = {'colonymind-comms.db': 'SELECT id, contact_id, channel, direction, summary, ts FROM communications ORDER BY id',
              'counter.db': 'SELECT * FROM c ORDER BY n', 'rollback.db': 'SELECT * FROM c ORDER BY n',
              'memory-state/commitments.db': 'SELECT * FROM c ORDER BY n',
              '@external/colonymind-turn-outbox.sqlite3': 'SELECT turn_id, state FROM turn_outbox ORDER BY turn_id'}
    sources = {'colonymind-comms.db': sqlite3.connect(primary / 'colonymind-comms.db'), 'counter.db': counter,
               'rollback.db': rollback, 'memory-state/commitments.db': nested,
               '@external/colonymind-turn-outbox.sqlite3': sqlite3.connect(outbox_path)}
    held = connect(standby_home, 'counter.db')
    before = _rows(held, stores['counter.db'])
    for key, sql in stores.items():
        replica = connect(standby_home, key)
        assert replica.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert _rows(replica, sql) == _rows(sources[key], sql), key
    # A reader keeps its snapshot while the replica moves on; a new reader sees the new state.
    counter.execute('INSERT INTO c(n) VALUES (1000)')
    _sync(shipper, standby)
    assert _rows(held, stores['counter.db']) == before
    assert _rows(connect(standby_home, 'counter.db'), stores['counter.db'])[-1][0] == 1000
    vectors = tree_path(standby_home, 'vectors')
    import lancedb
    assert lancedb.connect(str(vectors / 'lancedb')).open_table('conversations').count_rows() == table.count_rows()
    files = tree_path(standby_home, 'files')
    assert (files / 'instance-id').read_text() == 'twin-test-instance\n'
    assert (files / 'media' / 'original.bin').read_bytes() == (primary / 'media' / 'original.bin').read_bytes()
    assert not (files / 'colonymind-comms.db').exists()        # stores travel as pages, not files
    assert (files / 'backups' / '20260101T000000Z' / 'old.db').is_file()   # backups travel as files
    assert 'backups/20260101T000000Z/old.db' not in shipper.sources
    status = standby.status()
    assert status['alarms'] == {} and status['stale'] == [] and status['lag_seconds'] < 5
    assert shipper.status()['lag_seconds'] == 0 and shipper.status()['alarms'] == {}
    shipper.close()


def test_the_standby_cannot_fork_the_record(tmp_path):
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path)
    _counter(primary).execute('INSERT INTO c(n) VALUES (1)')
    shipper, standby = Shipper(primary, cp), Standby(standby_home, cs)
    _sync(shipper, standby)
    replica = connect(standby_home, 'counter.db')
    with pytest.raises(sqlite3.OperationalError):
        replica.execute('INSERT INTO c(n) VALUES (2)')
    path = standby_home / '.twin' / 'replica' / 'db' / 'counter.db'
    assert stat.S_IMODE(path.stat().st_mode) == 0o400
    result = subprocess.run([sys.executable, '-m', 'colonymind', '--instance', str(standby_home), 'twin', 'query',
                             'counter.db', 'DELETE FROM c'], capture_output=True, text=True, cwd=Path(__file__).parents[1])
    assert result.returncode != 0 and 'not authorized' in result.stderr
    result = subprocess.run([sys.executable, '-m', 'colonymind', '--instance', str(standby_home), 'twin', 'query',
                             'counter.db', 'SELECT n FROM c'], capture_output=True, text=True, cwd=Path(__file__).parents[1])
    assert result.returncode == 0 and json.loads(result.stdout) == {'n': 1}
    shipper.close()


def test_promote_with_writes_in_flight_yields_a_prefix_and_fences_the_primary(tmp_path):
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path)
    CommsLog(primary / 'colonymind-comms.db')._conn.close()   # the store exists before the first ship
    outbox = TurnOutbox(outbox_path)
    counter = _counter(primary)
    _vectors(primary)
    shipper, standby = Shipper(primary, cp), Standby(standby_home, cs)
    stop, written, errors = threading.Event(), [0], []

    def writer():
        comms = CommsLog(primary / 'colonymind-comms.db')   # its connection belongs to the writer's thread
        i = 0
        while not stop.is_set():
            try:
                counter.execute('INSERT INTO c(n, blob) VALUES (?, ?)', (i, os.urandom(random.randint(1, 3000))))
                comms.log('p-1', channel='system', summary=f'{i:06d}')
                outbox.enqueue(f'turn-{i:06d}', {'turn_id': f'turn-{i:06d}'})
            except Exception as error:   # the primary must never see an error from the twin
                errors.append(error)
            i += 1
            written[0] = i
            time.sleep(0.001)

    def ship():
        while not shipper.fenced and not stop.is_set():
            shipper.tick()
            time.sleep(0.005)

    threads = [threading.Thread(target=writer), threading.Thread(target=ship)]
    for thread in threads:
        thread.start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            standby.cycle()
            time.sleep(0.005)
        started = time.monotonic()
        report = standby.promote()
        promote_seconds = time.monotonic() - started
        deadline = time.monotonic() + 5
        while not shipper.fenced and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        stop.set()
        for thread in threads:
            thread.join()
    assert errors == []
    assert shipper.fenced and 'fenced' in shipper.journal.alarms
    promoted = sqlite3.connect(standby_home / 'counter.db')
    assert promoted.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    ours = [r[0] for r in promoted.execute('SELECT n FROM c ORDER BY n')]
    theirs = [r[0] for r in counter.execute('SELECT n FROM c ORDER BY n')]
    assert ours and ours == theirs[:len(ours)], 'the promoted store must be a prefix of the primary'
    assert list(promoted.execute('SELECT * FROM c ORDER BY n')) == list(counter.execute('SELECT * FROM c ORDER BY n'))[:len(ours)]
    ids = [r[0] for r in sqlite3.connect(standby_home / 'colonymind-comms.db').execute(
        'SELECT summary FROM communications ORDER BY summary')]
    assert ids == [f'{i:06d}' for i in range(len(ids))] and ids
    body = tmp_path / 'standby-body' / 'state' / 'colonymind-turn-outbox.sqlite3'
    pending = TurnOutbox(body).pending_count()
    assert pending > 0, 'captured turns not yet delivered must survive the failover'
    assert (standby_home / 'instance-id').read_text() == 'twin-test-instance\n'
    import lancedb
    assert lancedb.connect(str(standby_home / 'lancedb')).open_table('conversations').count_rows() == 3
    config = twin_config.load(standby_home)
    assert config.role == 'primary' and config.promoted['epoch'] == report['promoted']['epoch']
    from colonymind.twin.transport import TransportError
    with pytest.raises(TransportError):
        shipper.transport.mirror(primary / '.twin' / 'outbox')     # nothing more lands on it
    lost = len(theirs) - len(ours)
    (tmp_path / 'metrics.json').write_text(json.dumps({'promote_seconds': promote_seconds, 'rpo_commits': lost}))
    with pytest.raises(PromoteError):
        Standby(standby_home, twin_config.TwinConfig(role='standby')).promote()
    shipper.close()


def test_spool_cap_reseeds_instead_of_filling_the_disk(tmp_path):
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path, max_spool_bytes=1 << 20)
    counter = _counter(primary)
    shipper = Shipper(primary, cp)
    first_generation = shipper.generation
    peak = 0
    for i in range(400):                       # the standby is away: nothing is acknowledged
        counter.execute('INSERT INTO c(n, blob) VALUES (?, ?)', (i, os.urandom(8000)))
        shipper.tick()
        peak = max(peak, shipper.capped_bytes())
        assert shipper.spool_bytes() - shipper.capped_bytes() <= (primary / 'counter.db').stat().st_size + (2 << 20)
    assert shipper.generation != first_generation
    assert peak <= cp.max_spool_bytes + (64 << 10), 'beyond the base seeds, the spool is capped (plus one tick)'
    events = [json.loads(line) for line in (primary / '.twin' / 'events.jsonl').read_text().splitlines()]
    assert any(e['event'] == 'generation' and e['reason'] == 'spool-cap' for e in events)
    assert any(e['event'] == 'alarm' and e['name'] == 'spool_cap' for e in events)
    standby = Standby(standby_home, cs)         # the standby returns and catches up from the new seed
    _sync(shipper, standby)
    replica = connect(standby_home, 'counter.db')
    assert list(replica.execute('SELECT * FROM c ORDER BY n')) == list(counter.execute('SELECT * FROM c ORDER BY n'))
    assert len(list((standby_home / '.twin' / 'inbox').glob('gen-*'))) == 1
    shipper.close()


def test_reseed_on_request_from_either_side(tmp_path):
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path)
    counter = _counter(primary)
    counter.execute('INSERT INTO c(n) VALUES (1)')
    shipper, standby = Shipper(primary, cp), Standby(standby_home, cs)
    _sync(shipper, standby)
    generation = shipper.generation
    (primary / '.twin' / 'reseed.request').write_text('operator test\n')
    _sync(shipper, standby)
    assert shipper.generation != generation and standby.state['generation'] == shipper.generation
    generation = shipper.generation
    standby.request_reseed('standby test')
    _sync(shipper, standby, rounds=6)
    assert shipper.generation != generation and standby.state['generation'] == shipper.generation
    assert standby.state['reseed'] is None
    assert list(connect(standby_home, 'counter.db').execute('SELECT n FROM c')) == [(1,)]
    shipper.close()


def test_a_tampered_replica_is_detected_and_reseeded(tmp_path):
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path, anti_entropy_interval=3600)
    counter = _counter(primary)
    counter.executemany('INSERT INTO c(n, blob) VALUES (?, ?)', [(i, os.urandom(500)) for i in range(100)])
    shipper, standby = Shipper(primary, cp), Standby(standby_home, cs)
    _sync(shipper, standby)
    replica = standby_home / '.twin' / 'replica' / 'db' / 'counter.db'
    os.chmod(replica, 0o600)
    with open(replica, 'r+b') as handle:        # silent corruption on the standby
        handle.seek(4096 * 2 + 100)
        handle.write(b'\xff' * 16)
    shipper.last_anti_entropy = 0.0            # the next tick runs the anti-entropy pass
    _sync(shipper, standby, rounds=8)
    events = [json.loads(line) for line in (standby_home / '.twin' / 'events.jsonl').read_text().splitlines()]
    assert any(e['event'] == 'alarm' and e['name'] == 'diverged' and e['state'] == 'raised' for e in events)
    assert any(e['event'] == 'reseed-requested' for e in events)
    assert 'diverged' not in standby.journal.alarms
    fixed = connect(standby_home, 'counter.db')
    assert fixed.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    assert list(fixed.execute('SELECT * FROM c ORDER BY n')) == list(counter.execute('SELECT * FROM c ORDER BY n'))
    shipper.close()


def test_an_unreachable_standby_never_blocks_the_primary_and_raises_alarms(tmp_path):
    primary, standby_home, outbox_path = _instance(tmp_path)
    marker = tmp_path / 'alarms.log'
    hook = tmp_path / 'hook.sh'
    hook.write_text(f'#!/bin/sh\necho "$COLONYMIND_TWIN_ALARM $COLONYMIND_TWIN_STATE" >> {marker}\n')
    hook.chmod(0o755)
    cp, _ = _configs(primary, standby_home, outbox_path, tmp_path, max_lag_seconds=0.2, alarm_command=[str(hook)])
    shutil.rmtree(standby_home / '.twin')          # the standby is gone
    counter = _counter(primary)
    shipper = Shipper(primary, cp)
    started = time.monotonic()
    for i in range(30):
        counter.execute('INSERT INTO c(n) VALUES (?)', (i,))
        shipper.tick()
        time.sleep(0.02)
    assert time.monotonic() - started < 5
    assert counter.execute('SELECT count(*) FROM c').fetchone()[0] == 30
    assert {'transport', 'lag'} <= set(shipper.journal.alarms)
    lines = marker.read_text().split('\n')
    assert 'transport raised' in lines and 'lag raised' in lines
    shipper.close()


def test_rsync_transport_through_a_stand_in_ssh(tmp_path):
    if shutil.which('rsync') is None:
        pytest.skip('rsync is not installed')
    fake_ssh = tmp_path / 'fake-ssh'
    # Drops ssh options and the host, then runs the remote command locally: the real rsync
    # protocol over a pipe, as over ssh.
    fake_ssh.write_text('#!/bin/sh\nwhile [ "$1" = "-o" ] || [ "$1" = "-p" ]; do shift 2; done\nshift\nexec sh -c "$*"\n')
    fake_ssh.chmod(0o755)
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path)
    cp.target = f'ssh://twin-host{standby_home}'
    transport = RsyncTransport(cp.target, ssh_command=str(fake_ssh))
    counter = _counter(primary)
    counter.execute('INSERT INTO c(n) VALUES (7)')
    shipper, standby = Shipper(primary, cp, transport=transport), Standby(standby_home, cs)
    _sync(shipper, standby)
    assert list(connect(standby_home, 'counter.db').execute('SELECT n FROM c')) == [(7,)]
    assert shipper.acked >= shipper.seq - 1 > 0
    shipper.close()


def test_vector_snapshots_hold_one_consistent_version_under_concurrent_writes(tmp_path):
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path)
    table, add = _vectors(primary)
    shipper, standby = Shipper(primary, cp), Standby(standby_home, cs)
    stop = threading.Event()

    def writer():
        n = 1000
        while not stop.is_set():
            add(n, 5)
            n += 5

    thread = threading.Thread(target=writer)
    thread.start()
    installed = set()
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            _sync(shipper, standby, rounds=1)
            info = standby.state['trees'].get('vectors')
            if info:
                installed.add(info['seq'])
    finally:
        stop.set()
        thread.join()
    _sync(shipper, standby)
    assert len(installed) >= 2
    assert standby.state['tree_failed'].get('vectors') is None
    root = tree_path(standby_home, 'vectors') / 'lancedb' / 'conversations.lance' / '_versions'
    assert len(list(root.glob('*.manifest'))) == 1
    import lancedb
    assert lancedb.connect(str(tree_path(standby_home, 'vectors') / 'lancedb')).open_table(
        'conversations').count_rows() == table.count_rows()
    shipper.close()


def test_a_corrupt_tree_blob_is_rejected_and_replaced(tmp_path):
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path)
    shipper, standby = Shipper(primary, cp), Standby(standby_home, cs)
    shipper.tick()
    blobs = standby_home / '.twin' / 'inbox' / 'trees' / 'blobs'
    victim = next(p for p in blobs.iterdir() if p.stat().st_size > 1000)
    victim.write_bytes(b'x' * victim.stat().st_size)     # same size: a size-only mirror skips it
    standby.cycle()
    assert standby.state['tree_failed'].get('files')
    _sync(shipper, standby, rounds=6)                    # nobody removes the bad blob by hand
    assert standby.state['trees']['files']['version'] > standby.state['tree_failed']['files']
    installed = tree_path(standby_home, 'files') / 'media' / 'original.bin'
    assert installed.read_bytes() == (primary / 'media' / 'original.bin').read_bytes()
    events = [json.loads(line) for line in (standby_home / '.twin' / 'events.jsonl').read_text().splitlines()]
    assert [e['state'] for e in events if e['event'] == 'alarm' and e['name'] == 'tree'] == ['raised', 'cleared']
    shipper.close()


def test_a_corrupt_segment_in_the_inbox_is_fetched_again(tmp_path):
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path)
    counter = _counter(primary)
    counter.execute('INSERT INTO c(n) VALUES (1)')
    shipper, standby = Shipper(primary, cp), Standby(standby_home, cs)
    shipper.tick()
    victim = next((standby_home / '.twin' / 'inbox').glob('gen-*/000000000001-*.seg'))
    victim.write_bytes(b'x' * victim.stat().st_size)
    _sync(shipper, standby)
    assert list(connect(standby_home, 'counter.db').execute('SELECT n FROM c')) == [(1,)]
    shipper.close()


def test_promotion_merges_an_existing_outbox_keeps_local_config_and_displaces_old_stores(tmp_path):
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path)
    outbox = TurnOutbox(outbox_path)
    for i in range(3):
        outbox.enqueue(f'turn-{i}', {'turn_id': f'turn-{i}'})
    _counter(primary).execute('INSERT INTO c(n) VALUES (1)')
    body = Path(cs.external_dir)
    body.mkdir(parents=True)
    local = TurnOutbox(body / outbox_path.name)          # the standby's own body already has one
    local.enqueue('turn-local', {'turn_id': 'turn-local'})
    local.enqueue('turn-1', {'turn_id': 'turn-1'})
    (standby_home / 'colonymind.yaml').write_text('sidecar: {port: 8888}\n')
    _counter(standby_home).execute('INSERT INTO c(n) VALUES (99)')   # a stale store from before
    shipper, standby = Shipper(primary, cp), Standby(standby_home, cs)
    _sync(shipper, standby)
    report = standby.promote()
    assert sorted(r['turn_id'] for r in TurnOutbox(body / outbox_path.name).rows('pending')) == [
        'turn-0', 'turn-1', 'turn-2', 'turn-local']
    assert (standby_home / 'colonymind.yaml').read_text() == 'sidecar: {port: 8888}\n'
    assert (standby_home / 'colonymind.yaml.from-primary').read_text() == 'sidecar: {port: 7777}\n'
    assert list(sqlite3.connect(standby_home / 'counter.db').execute('SELECT n FROM c')) == [(1,)]
    displaced = Path(report['displaced'])
    assert list(sqlite3.connect(displaced / 'counter.db').execute('SELECT n FROM c')) == [(99,)]
    shipper.close()


def test_anti_entropy_walks_one_store_per_tick(tmp_path):
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, _ = _configs(primary, standby_home, outbox_path, tmp_path, anti_entropy_interval=3600)
    for name in ('a.db', 'b.db', 'c.db'):
        _counter(primary, name).execute('INSERT INTO c(n) VALUES (1)')
    shipper = Shipper(primary, cp)
    shipper.tick()
    shipper.tick()
    shipper.last_anti_entropy = 0.0
    checked = []
    for _ in range(6):
        before = {key: source.stats.anti_entropy_runs for key, source in shipper.sources.items()}
        shipper.tick()
        ran = [key for key, source in shipper.sources.items() if source.stats.anti_entropy_runs > before[key]]
        assert len(ran) <= 1, 'one store per tick, never the whole instance at once'
        checked += ran
    assert checked == ['a.db', 'b.db', 'c.db']
    events = [json.loads(line) for line in (primary / '.twin' / 'events.jsonl').read_text().splitlines()]
    passes = [e for e in events if e['event'] == 'anti-entropy']
    assert len(passes) == 1 and passes[0]['stores'] == 3 and passes[0]['repaired'] == 0
    shipper.close()


def test_a_failed_publication_is_never_acknowledged(tmp_path, monkeypatch):
    from colonymind.twin import standby as standby_module
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path)
    counter = _counter(primary)
    counter.execute('INSERT INTO c(n) VALUES (1)')
    shipper, standby = Shipper(primary, cp), Standby(standby_home, cs)
    shipper.tick()
    real, failed = os.replace, []

    def replace(source, destination, *args, **kwargs):
        if str(destination).endswith('/replica/db/counter.db') and not failed:
            failed.append(destination)
            raise OSError(28, 'No space left on device')
        return real(source, destination, *args, **kwargs)

    monkeypatch.setattr(standby_module.os, 'replace', replace)
    with pytest.raises(OSError):
        standby.cycle()
    assert failed and standby.state['seq'] == 0, 'nothing was published, so nothing is applied'
    assert (twin_config.read_json(standby_home / '.twin' / 'ack.json') or {}).get('seq', 0) == 0
    _sync(shipper, standby)
    assert list(connect(standby_home, 'counter.db').execute('SELECT n FROM c')) == [(1,)]
    assert Standby(standby_home, cs).state['seq'] == standby.state['seq'] > 0
    shipper.close()


def test_a_kept_local_file_from_the_primary_keeps_its_restrictive_mode(tmp_path):
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path)
    (primary / 'api.key').write_text('primary-secret\n')
    os.chmod(primary / 'api.key', 0o600)
    (standby_home / 'api.key').write_text('standby-secret\n')
    os.chmod(standby_home / 'api.key', 0o600)
    cs.keep_local = ['colonymind.yaml', 'api.key']
    _counter(primary).execute('INSERT INTO c(n) VALUES (1)')
    shipper, standby = Shipper(primary, cp), Standby(standby_home, cs)
    _sync(shipper, standby)
    previous = os.umask(0o022)
    try:
        standby.promote()
    finally:
        os.umask(previous)
    copied = standby_home / 'api.key.from-primary'
    assert copied.read_text() == 'primary-secret\n'
    assert stat.S_IMODE(copied.stat().st_mode) & 0o077 == 0, 'a secret must not become readable by others'
    assert (standby_home / 'api.key').read_text() == 'standby-secret\n'
    shipper.close()


def test_a_store_deleted_while_the_shipper_was_stopped_is_not_resurrected(tmp_path):
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path)
    _counter(primary, 'a.db').execute('INSERT INTO c(n) VALUES (1)')
    _counter(primary, 'b.db').execute('INSERT INTO c(n) VALUES (2)')
    shipper, standby = Shipper(primary, cp), Standby(standby_home, cs)
    _sync(shipper, standby)
    assert {'a.db', 'b.db'} <= set(standby.state['stores'])
    shipper.close()                                   # the shipper stops; the owner deletes a.db
    for suffix in ('', '-wal', '-shm'):
        (primary / f'a.db{suffix}').unlink(missing_ok=True)
    shipper = Shipper(primary, cp)                    # and it starts again, knowing nothing of a.db
    _sync(shipper, standby)
    assert 'a.db' not in standby.state['stores']
    with pytest.raises(FileNotFoundError):
        connect(standby_home, 'a.db')
    standby.promote()
    assert not (standby_home / 'a.db').exists(), 'a store the owner deleted must not come back'
    assert list(sqlite3.connect(standby_home / 'b.db').execute('SELECT n FROM c')) == [(2,)]
    shipper.close()


def test_tree_snapshots_keep_flowing_after_the_shipper_restarts(tmp_path):
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path)
    shipper, standby = Shipper(primary, cp), Standby(standby_home, cs)
    for i in range(4):
        (primary / 'identity.yaml').write_text(f'agent: {{name: Test{i}}}\n')
        _sync(shipper, standby, rounds=2)
    assert (tree_path(standby_home, 'files') / 'identity.yaml').read_text() == 'agent: {name: Test3}\n'
    assert standby.state['trees']['files']['seq'] >= 3
    shipper.close()
    shipper = Shipper(primary, cp)                   # its snapshot numbers start again
    (primary / 'identity.yaml').write_text('agent: {name: After}\n')
    _sync(shipper, standby, rounds=6)
    assert (tree_path(standby_home, 'files') / 'identity.yaml').read_text() == 'agent: {name: After}\n'
    assert 'files' not in shipper.tree_pending, 'acknowledged only once the standby installed it'
    shipper.close()


def test_an_interrupted_promotion_resumes(tmp_path, monkeypatch):
    from colonymind.twin import standby as standby_module
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path)
    _counter(primary).execute('INSERT INTO c(n) VALUES (1)')
    _counter(primary, 'other.db').execute('INSERT INTO c(n) VALUES (2)')
    shipper, standby = Shipper(primary, cp), Standby(standby_home, cs)
    _sync(shipper, standby)
    real, calls = standby_module._place, []

    def place(source, destination, *args):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError(16, 'Device or resource busy')
        return real(source, destination, *args)

    monkeypatch.setattr(standby_module, '_place', place)
    with pytest.raises(OSError):
        standby.promote()
    assert twin_config.load(standby_home).role == 'standby'
    monkeypatch.setattr(standby_module, '_place', real)
    report = Standby(standby_home, cs).promote()     # a new process, after the failure
    assert report['resumed']
    assert list(sqlite3.connect(standby_home / 'counter.db').execute('SELECT n FROM c')) == [(1,)]
    assert list(sqlite3.connect(standby_home / 'other.db').execute('SELECT n FROM c')) == [(2,)]
    assert (standby_home / 'instance-id').read_text() == 'twin-test-instance\n'
    config = twin_config.load(standby_home)
    assert config.role == 'primary' and config.promoted['epoch'] == report['promoted']['epoch']
    with pytest.raises(PromoteError, match='already promoted'):
        Standby(standby_home, cs).promote()
    shipper.close()


def test_promotion_requires_the_file_and_vector_snapshots(tmp_path, monkeypatch):
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path)
    _counter(primary).execute('INSERT INTO c(n) VALUES (1)')
    _vectors(primary)
    shipper, standby = Shipper(primary, cp), Standby(standby_home, cs)
    monkeypatch.setattr(shipper, '_maybe_snapshot_tree', lambda kind, now: None)   # trees withheld
    _sync(shipper, standby)
    assert standby.state['stores']['counter.db']['generation'] == standby.state['generation']
    with pytest.raises(PromoteError) as refused:
        standby.promote()
    assert 'files: no snapshot installed' in str(refused.value)
    assert 'vectors: no snapshot installed' in str(refused.value)
    assert not standby.state['promoted'] and twin_config.load(standby_home).role == 'standby'
    monkeypatch.undo()
    shipper.tick()                                     # the trees arrive, one of them damaged
    blobs = standby_home / '.twin' / 'inbox' / 'trees' / 'blobs'
    victim = next(p for p in blobs.iterdir() if p.stat().st_size == 3 << 20)    # media/original.bin
    victim.write_bytes(b'x' * victim.stat().st_size)
    with pytest.raises(PromoteError, match='files: '):
        standby.promote()
    assert standby.state['tree_failed'].get('files')
    shipper.close()


def test_seeding_never_takes_the_disk_below_its_free_space_floor(tmp_path, monkeypatch):
    import collections
    primary, standby_home, outbox_path = _instance(tmp_path)
    floor = 64 << 20
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path, min_free_bytes=floor)
    counter = _counter(primary)
    counter.executemany('INSERT INTO c(n, blob) VALUES (?, ?)', [(i, os.urandom(4000)) for i in range(512)])
    usage = collections.namedtuple('usage', 'total used free')
    free = [floor + (1 << 20)]          # above the floor, but not by the size of the store's snapshot
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: usage(1 << 40, 0, free[0]))
    shipper = Shipper(primary, cp)
    shipper.tick()
    shipper.tick()
    assert not shipper.sources['counter.db'].seeded
    assert not list((primary / '.twin').rglob('seed-*.db')) and not list((primary / '.twin').rglob('.snap-*'))
    assert 'disk' in shipper.journal.alarms and 'counter.db' in shipper.journal.alarms['disk']
    free[0] = 1 << 40                   # space is back: the store is seeded and the alarm clears
    standby = Standby(standby_home, cs)
    _sync(shipper, standby)
    assert shipper.sources['counter.db'].seeded and 'disk' not in shipper.journal.alarms
    assert connect(standby_home, 'counter.db').execute('SELECT count(*) FROM c').fetchone()[0] == 512
    free[0] = floor + (1 << 20)         # short again: an anti-entropy check waits, the fast path does not
    shipper.last_anti_entropy = 0.0
    for i in range(3):
        counter.execute('INSERT INTO c(n) VALUES (?)', (1000 + i,))
        _sync(shipper, standby, rounds=1)
    _sync(shipper, standby, rounds=2)
    assert shipper.sources['counter.db'].pinned and 'counter.db' not in shipper.failing
    assert connect(standby_home, 'counter.db').execute('SELECT count(*) FROM c').fetchone()[0] == 515
    shipper.close()


def test_a_blocked_transfer_never_holds_the_primary_log(tmp_path):
    from colonymind.twin.transport import LocalTransport

    class Blocking(LocalTransport):
        """A partition: every call waits (as a black-holed ssh does until its timeout)."""
        def __init__(self, target):
            super().__init__(target)
            self.block, self.release = threading.Event(), threading.Event()

        def mirror(self, outbox):
            if self.block.is_set():
                self.release.wait(20)
            return super().mirror(outbox)

        def read_ack(self):
            if self.block.is_set():
                self.release.wait(20)
            return super().read_ack()

    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, _ = _configs(primary, standby_home, outbox_path, tmp_path, release_bytes=64 << 10)
    cp.transfer_wait = 0.05
    counter = _counter(primary)
    counter.execute('PRAGMA wal_autocheckpoint=16')
    counter.execute('INSERT INTO c(n) VALUES (-1)')
    transport = Blocking(standby_home)
    shipper = Shipper(primary, cp, transport=transport)
    for _ in range(3):
        shipper.tick()
    assert shipper.sources['counter.db'].pinned
    transport.block.set()
    stop, peak, errors, ticks = threading.Event(), [0], [], [0]

    def ticking():
        while not stop.is_set():
            shipper.tick()
            ticks[0] += 1
            time.sleep(0.01)

    def writing():
        i = 0
        while not stop.is_set():
            try:
                counter.execute('INSERT INTO c(n, blob) VALUES (?, ?)', (i, os.urandom(4000)))
            except sqlite3.Error as error:
                errors.append(error)
            i += 1
            peak[0] = max(peak[0], (primary / 'counter.db-wal').stat().st_size)
            time.sleep(0.001)

    threads = [threading.Thread(target=ticking), threading.Thread(target=writing)]
    for thread in threads:
        thread.start()
    time.sleep(1.5)
    stop.set()
    transport.release.set()
    for thread in threads:
        thread.join()
    assert errors == []
    assert ticks[0] > 10, 'the shipper keeps ticking while a transfer is blocked'
    assert peak[0] < 2 << 20, f'the log grew to {peak[0]} bytes behind a blocked transfer'
    shipper.close()


def test_a_store_the_primary_cannot_read_is_not_hidden_by_heartbeats(tmp_path, monkeypatch):
    from colonymind.twin.source import SourceError
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path, max_lag_seconds=5)
    cs.max_lag_seconds = 5
    clock = [1_000_000.0]
    _counter(primary).execute('INSERT INTO c(n) VALUES (1)')
    _counter(primary, 'commitments.db').execute('INSERT INTO c(n) VALUES (1)')
    shipper = Shipper(primary, cp, clock=lambda: clock[0])
    standby = Standby(standby_home, cs, clock=lambda: clock[0])
    _sync(shipper, standby)
    assert standby.state['stores']['commitments.db']['generation'] == standby.state['generation']

    def unreadable(**_):
        raise SourceError('store is missing or not a SQLite database')

    monkeypatch.setattr(shipper.sources['commitments.db'], 'poll', unreadable)
    for _ in range(10):                               # heartbeats flow and are acknowledged
        clock[0] += 1.0
        shipper.tick()
        standby.cycle()
    assert shipper.acked >= shipper.seq - 1 > 0
    assert 'commitments.db' in shipper.journal.alarms.get('store', ''), shipper.journal.alarms
    assert shipper.lag >= 9 and 'lag' in shipper.journal.alarms
    with pytest.raises(PromoteError, match='commitments.db'):
        standby.promote()
    shipper.close()


def test_a_store_larger_than_the_standby_spool_cap_does_not_loop_reseeds(tmp_path):
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path)
    cs.max_spool_bytes = 1 << 20
    counter = _counter(primary)
    counter.executemany('INSERT INTO c(n, blob) VALUES (?, ?)', [(i, os.urandom(4000)) for i in range(512)])
    shipper, standby = Shipper(primary, cp), Standby(standby_home, cs)
    generation = shipper.generation
    for i in range(6):
        counter.execute('INSERT INTO c(n) VALUES (?)', (1000 + i,))
        _sync(shipper, standby, rounds=2)
    assert standby.state['reseed'] is None and 'spool_cap' not in standby.journal.alarms
    assert shipper.generation == generation == standby.state['generation']
    assert connect(standby_home, 'counter.db').execute('SELECT count(*) FROM c').fetchone()[0] == 518
    shipper.close()


def test_a_failed_segment_write_leaves_no_gap_and_loses_no_change(tmp_path, monkeypatch):
    import contextlib
    from colonymind.twin import segment as segment_module
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path)
    counter = _counter(primary)
    counter.execute('INSERT INTO c(n) VALUES (1)')
    shipper = Shipper(primary, cp)
    real, failed = segment_module.write, []

    def write(*args, **kwargs):
        if not failed:
            failed.append(True)
            raise OSError(5, 'Input/output error')
        return real(*args, **kwargs)

    monkeypatch.setattr(segment_module, 'write', write)
    with contextlib.suppress(OSError):
        shipper.tick()
    assert failed
    counter.execute('INSERT INTO c(n) VALUES (2)')
    shipper.tick()
    assert sorted(shipper.unacked) == [1], 'the next segment takes the number that failed'
    spooled = {p for p in (primary / '.twin' / 'outbox' / shipper.generation).iterdir()}
    assert spooled == set(shipper.unacked[1][2]), 'nothing orphaned by the failed write'
    standby = Standby(standby_home, cs)
    _sync(shipper, standby)
    assert list(connect(standby_home, 'counter.db').execute('SELECT n FROM c ORDER BY n')) == [(1,), (2,)]
    shipper.close()


def _step_clock_back(monkeypatch, seconds):
    real = time.time_ns
    monkeypatch.setattr(time, 'time_ns', lambda: real() - int(seconds * 1e9))


@pytest.mark.parametrize('ids', ['kept', 'lost'])
def test_replication_keeps_flowing_after_the_clock_steps_back_across_a_restart(tmp_path, monkeypatch, ids):
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path)
    counter = _counter(primary)
    counter.execute('INSERT INTO c(n) VALUES (1)')
    shipper, standby = Shipper(primary, cp), Standby(standby_home, cs)
    _sync(shipper, standby, rounds=3)
    assert standby.state['trees']['files']['version']
    shipper.close()
    if ids == 'lost':                     # the primary's record of the names it used is gone too
        (primary / '.twin' / 'ids.json').unlink(missing_ok=True)
    _step_clock_back(monkeypatch, 3600)
    shipper = Shipper(primary, cp)
    (primary / 'identity.yaml').write_text('agent: {name: After}\n')
    counter.execute('INSERT INTO c(n) VALUES (2)')
    _sync(shipper, standby, rounds=8)
    assert (tree_path(standby_home, 'files') / 'identity.yaml').read_text() == 'agent: {name: After}\n'
    assert _rows(connect(standby_home, 'counter.db'), 'SELECT n FROM c ORDER BY n') == [(1,), (2,)]
    assert 'files' not in shipper.tree_pending
    shipper.close()


@pytest.mark.parametrize('fence', ['ack', 'marker'])
def test_a_promotion_interrupted_before_its_fence_publishes_it_on_retry(tmp_path, monkeypatch, fence):
    from colonymind.twin import standby as standby_module
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path)
    _counter(primary).execute('INSERT INTO c(n) VALUES (1)')
    shipper, standby = Shipper(primary, cp), Standby(standby_home, cs)
    _sync(shipper, standby)
    real = standby_module.write_json

    def write_json(path, value, **kwargs):
        if fence == 'ack' and Path(path).name == 'ack.json' and value.get('promoted'):
            raise OSError(28, 'No space left on device')
        if fence == 'marker' and Path(path).name == twin_config.PROMOTED:
            raise OSError(28, 'No space left on device')
        return real(path, value, **kwargs)

    monkeypatch.setattr(standby_module, 'write_json', write_json)
    with pytest.raises(OSError):
        standby.promote()
    monkeypatch.setattr(standby_module, 'write_json', real)
    report = Standby(standby_home, cs).promote()     # a new process, after the failure
    ack = json.loads((standby_home / '.twin' / 'ack.json').read_text())
    assert (ack.get('promoted') or {}).get('epoch') == report['promoted']['epoch']
    assert (standby_home / '.twin' / twin_config.PROMOTED).exists()
    for _ in range(3):
        shipper.tick()
    assert shipper.fenced
    shipper.close()


@pytest.mark.parametrize('crash', ['chmod', 'after-move'])
def test_an_interrupted_promotion_leaves_every_store_writable(tmp_path, monkeypatch, crash):
    from colonymind.twin import standby as standby_module
    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path)
    _counter(primary).execute('INSERT INTO c(n) VALUES (1)')
    shipper, standby = Shipper(primary, cp), Standby(standby_home, cs)
    _sync(shipper, standby)
    real_chmod, real_place, failed = os.chmod, standby_module._place, []

    def chmod(path, mode, *args, **kwargs):
        if mode == 0o600 and not failed:
            failed.append(path)
            raise OSError(5, 'Input/output error')
        return real_chmod(path, mode, *args, **kwargs)

    def place(source, destination, *args):
        real_place(source, destination, *args)
        if destination.name == 'counter.db' and not failed:
            failed.append(destination)
            raise OSError(5, 'Input/output error')        # the process dies right after the move

    if crash == 'chmod':
        monkeypatch.setattr(os, 'chmod', chmod)
    else:
        monkeypatch.setattr(standby_module, '_place', place)
    with pytest.raises(OSError):
        standby.promote()
    monkeypatch.undo()
    Standby(standby_home, cs).promote()
    for name in ('counter.db',):
        assert os.stat(standby_home / name).st_mode & 0o200, f'{name} is not writable after the promotion'
        conn = sqlite3.connect(standby_home / name)
        conn.execute('INSERT INTO c(n) VALUES (2)')
        conn.commit()
        conn.close()
    shipper.close()


@pytest.mark.parametrize('answer', ['promoted', 'reseed'])
def test_a_transfer_finishing_mid_tick_is_handled_before_the_next_starts(tmp_path, monkeypatch, answer):
    from colonymind.twin.source import SqliteSource
    from colonymind.twin.transport import LocalTransport

    class Gated(LocalTransport):
        """Each ack read waits for its gate; the test opens it while the tick polls the stores."""
        def __init__(self, target):
            super().__init__(target)
            self.calls = []

        def read_ack(self):
            gate = threading.Event()
            self.calls.append((gate, threading.current_thread()))
            number = len(self.calls)
            gate.wait(20)
            if answer == 'promoted':
                return {'promoted': {'epoch': f'epoch-{number}'}}
            return {'generation': '', 'seq': 0, 'reseed': {'nonce': f'nonce-{number}', 'reason': 'test'}}

        def mirror(self, outbox):
            return 0

    primary, standby_home, outbox_path = _instance(tmp_path)
    cp, _ = _configs(primary, standby_home, outbox_path, tmp_path)
    cp.transfer_wait = 0.05
    _counter(primary).execute('INSERT INTO c(n) VALUES (1)')
    transport = Gated(standby_home)
    shipper = Shipper(primary, cp, transport=transport)
    real_poll = SqliteSource.poll

    def poll(self, **kwargs):
        for gate, thread in [c for c in transport.calls if not c[0].is_set()]:
            gate.set()                   # the transfer finishes while the stores are polled
            thread.join(5)
        return real_poll(self, **kwargs)

    monkeypatch.setattr(SqliteSource, 'poll', poll)
    try:
        for _ in range(4):
            shipper.tick()
        finished = [i + 1 for i, (gate, _) in enumerate(transport.calls) if gate.is_set()]
        assert len(finished) >= (1 if answer == 'promoted' else 2)
        if answer == 'promoted':
            assert shipper.fenced, 'a promotion ack was dropped'
        else:
            assert {f'nonce-{i}' for i in finished} <= shipper.handled_requests
    finally:
        for gate, _ in transport.calls:
            gate.set()
        shipper.close()


@pytest.mark.parametrize('key_in_backups', [False, True])
def test_the_vault_key_travels_to_the_standby_only_when_backups_may_carry_it(tmp_path, key_in_backups):
    """The standby is a copy like a backup: it holds the sealed store's ciphertext, and its key only when
    ``secrets.vault.key_in_backups`` is on (docs/SECRETS.md)."""
    primary, standby_home, outbox_path = _instance(tmp_path)
    (primary / 'colonymind.yaml').write_text('sidecar: {port: 7777}\n'
                                            f'secrets: {{vault: {{key_in_backups: {str(key_in_backups).lower()}}}}}\n')
    (primary / 'secrets-vault.key').write_bytes(os.urandom(32))
    os.chmod(primary / 'secrets-vault.key', 0o600)
    cp, cs = _configs(primary, standby_home, outbox_path, tmp_path)
    shipper, standby = Shipper(primary, cp), Standby(standby_home, cs)
    _sync(shipper, standby, rounds=4)
    files = tree_path(standby_home, 'files')
    assert (files / 'identity.yaml').read_text() == 'agent: {name: Test}\n', 'the file snapshot arrived'
    assert (files / 'secrets-vault.key').is_file() is key_in_backups
    shipper.close()
