"""Bounded notice delivery and activity outside the watcher write transaction."""
from contextlib import closing
from dataclasses import replace
import json
import os
import sqlite3
import threading

import pytest

from colonymind import health_counters
from colonymind.jobwatch import activity, runner, store
from colonymind.work.hermes_jobs import HermesJob
from test_jobwatch import cfg, invoke, notice_rows, snapshot


def receipt(home, identity, ended, command):
    directory = home / 'logs' / 'process-results'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'proc_{identity}.json'
    path.write_text(json.dumps({'id': f'proc_{identity}', 'command': command, 'started_at': ended - 30,
                               'exit_code': 2, 'output': 'same failure'}))
    os.utime(path, (ended, ended))


def test_unset_command_never_accumulates_a_backlog(tmp_path, cfg, caplog):
    home = tmp_path / 'hermes'
    home.mkdir()
    (home / 'processes.json').write_text('[]')
    cfg['notice_command'] = []
    for day in range(5):
        now = 1000 + day * 2 * 86400
        for index in range(3):
            receipt(home, f'{day}_{index}', now + index, f'job{day} --private-argument')
        findings = invoke(tmp_path, cfg, now + 600)
        assert len(findings) == 1
    invoke(tmp_path, cfg, now + 2 * 86400)
    assert notice_rows(tmp_path) == []
    assert sum('notice command is unset' in record.message for record in caplog.records) == 6
    assert 'delivery failed' not in caplog.text
    notices = []
    cfg['notice_command'] = ['test-sender']
    invoke(tmp_path, cfg, now + 2 * 86400 + 300, notices=notices)
    assert notices == []


@pytest.mark.parametrize('count', [2, 6])
def test_receipt_expiry_groups_closes_and_names_safe_subjects(tmp_path, cfg, count):
    home = tmp_path / 'hermes'
    home.mkdir()
    (home / 'processes.json').write_text('[]')
    for command in range(count):
        for index in range(3):
            receipt(home, f'{command}_{index}', 1000 + index, f'tool{command} --private-argument')
    notices = []
    invoke(tmp_path, cfg, 1600, notices=notices)
    opened = list(notices)
    invoke(tmp_path, cfg, 1000 + 26 * 3600, notices=notices)
    closed = notices[len(opened):]
    assert len(opened) == len(closed) == (count if count <= 3 else 1)
    assert all('no new failures in the last 24 hours.' in message for message in closed)
    assert 'private-argument' not in repr(notices)
    if count > 3:
        assert opened[0].startswith('Job watch opened 6 findings: repeat: 6.')
        assert closed[0].startswith('Job watch closed 6 findings: repeat: 6.')
        assert opened[0].count('(process, started ') == closed[0].count('(process, started ') == 3


def test_repeat_success_closes_with_the_opening_subject(tmp_path, cfg):
    job = HermesJob('first', 'process', 'make', 'finished', 0, ended_at=1, exit_code=1,
                    signature='command', error_hash='same', source='process')
    notices = []
    invoke(tmp_path, cfg, 10, snapshot(*(replace(job, id=str(i), ended_at=i) for i in range(3))), notices=notices)
    invoke(tmp_path, cfg, 20, snapshot(replace(job, id='success', label='other label', started_at=15,
                                             ended_at=19, exit_code=0)), notices=notices)
    assert len(notices) == 2
    assert notices[-1].startswith('make (process, started ')
    assert notices[-1].endswith(': repeated failure has cleared.')


@pytest.mark.parametrize('age,expired', [(86399, False), (86401, True)])
def test_pending_notices_expire_without_reusing_delivery_ids(tmp_path, cfg, age, expired):
    cfg['dependency_failure_passes'] = 1
    key = 'writes_dropped.jobwatch_notice_expired'
    before = health_counters.snapshot().get(key, 0)
    def fail(*args):
        raise OSError
    runner.run(cfg, tmp_path, 'http://sidecar', jobs=snapshot(), now=0,
               request=lambda *a, **k: (503, None), notify=fail)
    first_id = notice_rows(tmp_path)[0]['id']
    notices = []
    invoke(tmp_path, cfg, age, snapshot(), request=lambda *a, **k: (503, None), notices=notices)
    assert len(notices) == int(not expired)
    assert health_counters.snapshot().get(key, 0) == before + int(expired)
    if expired:
        assert notice_rows(tmp_path) == []
    invoke(tmp_path, cfg, age + 1, snapshot(), notices=notices)
    invoke(tmp_path, cfg, age + 2, snapshot(), request=lambda *a, **k: (503, None), notices=notices)
    assert notice_rows(tmp_path)[-1]['id'] > first_id


@pytest.mark.parametrize('legacy', [False, True])
def test_partial_batch_cleanup_keeps_the_retry_identity(tmp_path, legacy):
    with closing(store.connect(tmp_path / 'jobwatch.db')) as conn:
        with conn:
            watch = runner.Pass(conn, 0)
            for index in range(4):
                watch.finding('endpoint', str(index), True, notice_data={'name': f'service{index}'})
        attempts = []
        def fail(message, command, notice_id):
            attempts.append((message, notice_id))
            raise OSError
        runner.deliver_pending(conn, ['test-sender'], fail, 0, 30)
        with conn:
            if legacy:
                conn.execute("DELETE FROM snapshots WHERE id LIKE 'notice_identity:%'")
            runner.Pass(conn, 1).finding('endpoint', '0', False)
        sent = []
        runner.deliver_pending(conn, ['test-sender'], lambda m, c, ident: sent.append((m, ident)), 2, 30)
        assert len(attempts) == len(sent) == 1
        assert sent[0][1] == attempts[0][1]
        assert 'service0' not in sent[0][0]
        assert all(f'service{index}' in sent[0][0] for index in range(1, 4))
        assert conn.execute('SELECT count(*) FROM notices WHERE sent=0').fetchone()[0] == 0


def test_activity_probe_does_not_hold_the_write_transaction(tmp_path, cfg, monkeypatch):
    invoke(tmp_path, cfg, 0, snapshot())
    entered, release = threading.Event(), threading.Event()
    errors = []
    def block(*args):
        entered.set()
        assert release.wait(5)
    monkeypatch.setattr(activity, 'record_activity', block)
    def run():
        try:
            invoke(tmp_path, cfg, 300, snapshot())
        except BaseException as error:
            errors.append(error)
    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(3)
        with closing(sqlite3.connect(tmp_path / 'jobwatch.db', timeout=.1)) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            store.put(conn, 'concurrent-write', 1)
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and not errors
    with closing(store.connect(tmp_path / 'jobwatch.db')) as conn:
        assert store.get(conn, 'concurrent-write') == 1
