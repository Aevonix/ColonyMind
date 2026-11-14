"""Regression cases from the second independent job watch review."""
from contextlib import closing
from copy import deepcopy
import json
import sqlite3
import sys
from types import SimpleNamespace

import pytest
import yaml

from colonymind import config, health_counters
from colonymind.jobwatch import cli, runner, store
from colonymind.memory import producers, turn_budget
from colonymind.turns import keywords
from colonymind.work.hermes_jobs import JobSnapshot, read_jobs


def test_unreadable_checkpoint_preserves_open_finding(tmp_path):
    home = tmp_path / 'hermes'
    home.mkdir()
    checkpoint = home / 'processes.json'
    row = {'session_id': 'proc_existing', 'command': 'make', 'started_at': 0}
    checkpoint.write_text(json.dumps([row]))
    cfg = deepcopy(config.DEFAULTS['watch'])
    cfg.update(hermes_home=str(home), tmux=False)
    jobs = read_jobs(home, 1800)
    jobs[0].cpu, jobs[0].output = 0, 'unchanged-output-digest'
    with closing(store.connect(tmp_path / 'jobwatch.db')) as conn, conn:
        for now in range(0, 1801, 300):
            runner.Pass(conn, now).jobs(jobs, cfg, None)
    assert len(store.read_findings(tmp_path / 'jobwatch.db')) == 1
    row['started_at'] = 'unparseable-time'
    checkpoint.write_text(json.dumps([row, {'session_id': 'proc_other', 'command': 'make', 'started_at': 2000}]))
    partial = read_jobs(home, 2100, processes_only=True)
    assert [job.id for job in partial] == ['proc_other']
    assert 'process' not in partial.available
    notices = []
    findings = runner.run(cfg, tmp_path, 'http://sidecar', now=2100,
        request=lambda *a, **k: (200, None), notify=lambda msg, argv: notices.append(msg))
    assert len(findings) == 1 and findings[0]['check'] == 'stuck'
    assert not any('resumed or ended' in msg for msg in notices)


@pytest.mark.parametrize('watch,key', [
    ({'endpoints': [{'name': 'local-service', 'url': 8080}]}, 'watch.endpoints'),
    ([], 'watch'), ({'endpoints': None}, 'watch.endpoints'),
])
def test_wrong_watch_types_are_isolated_and_cli_refuses(tmp_path, monkeypatch, caplog, capsys, watch, key):
    monkeypatch.setattr(config, '_watch_warnings', set())
    (tmp_path / 'colonymind.yaml').write_text(yaml.safe_dump({'watch': watch}))
    for _ in range(2):
        assert config.load_config(tmp_path, environ={}).data['watch'] == config.DEFAULTS['watch']
    warnings = [record for record in caplog.records if 'Ignoring invalid' in record.message]
    assert len(warnings) == 1 and key in warnings[0].message
    assert '8080' not in caplog.text and 'local-service' not in caplog.text
    assert cli.main(SimpleNamespace(instance=str(tmp_path), list=True, since=None, dry_run=False)) == 1
    assert key in capsys.readouterr().err


def test_unexpected_watch_validation_error_is_isolated(monkeypatch, caplog):
    from colonymind.jobwatch import settings

    def broken(value):
        raise AttributeError('PRIVATE_VALUE')

    monkeypatch.setattr(settings, 'validate_watch', broken)
    monkeypatch.setattr(config, '_watch_warnings', set())
    assert config.validate(deepcopy(config.DEFAULTS))['watch'] == config.DEFAULTS['watch']
    assert 'Ignoring invalid watch (AttributeError)' in caplog.text
    assert 'PRIVATE_VALUE' not in caplog.text
    with pytest.raises(config.ConfigError, match='watch is invalid'):
        config.validate(deepcopy(config.DEFAULTS), strict_watch=True)


@pytest.mark.parametrize('grouped', [False, True])
def test_notice_ids_survive_timeout_and_distinguish_events(tmp_path, grouped):
    cfg = deepcopy(config.DEFAULTS['watch'])
    cfg.update(dependency_failure_passes=1, notice_timeout_s=.2, tmux=False)
    if grouped:
        cfg['endpoints'] = [{'name': str(i), 'url': f'http://service/{i}', 'expect_status': 200} for i in range(4)]
    deliveries = tmp_path / 'notice-ids'
    script = ("import os,sys,time; from pathlib import Path; "
        "p=Path(sys.argv[1]); f=p.open('a'); "
        "f.write(os.environ.get('COLONYMIND_NOTICE_ID', '') + '\\n'); f.flush(); f.close(); "
        "time.sleep(float(sys.argv[2]))")
    cfg['notice_command'] = [sys.executable, '-c', script, str(deliveries), '2']
    for now in (0, 300, 3900):
        runner.run(cfg, tmp_path, 'http://sidecar', now=now, jobs=JobSnapshot(),
                   request=lambda *a, **k: (503, None))
    ids = deliveries.read_text().splitlines()
    assert len(ids) == 2 and ids[0] and ids[0] == ids[1]
    cfg['notice_command'][-1] = '0'
    runner.run(cfg, tmp_path, 'http://sidecar', now=7800, jobs=JobSnapshot(),
               request=lambda *a, **k: (503, None))
    runner.run(cfg, tmp_path, 'http://sidecar', now=7900, jobs=JobSnapshot(),
               request=lambda *a, **k: (200, None))
    ids = deliveries.read_text().splitlines()
    assert ids[2] == ids[0]
    closed = ids[3:]
    assert len(closed) == 1
    assert len(set([ids[0], *closed])) == len(closed) + 1
    runner.run(cfg, tmp_path, 'http://sidecar', now=8100, jobs=JobSnapshot(),
               request=lambda *a, **k: (503, None))
    reopened = deliveries.read_text().splitlines()[-1]
    assert reopened not in ids


@pytest.mark.parametrize('stemmed', [False, True])
async def test_keyword_reader_denies_attach_and_keeps_temp_ranking(tmp_path, monkeypatch, stemmed):
    path = tmp_path / 'terms.db'
    with closing(sqlite3.connect(path)) as conn, conn:
        for table in ('turn_source_search', 'turn_source_stems'):
            conn.execute(f'CREATE VIRTUAL TABLE {table} USING fts5(content)')
            conn.execute(f"INSERT INTO {table} VALUES ('neutral evidence')")
    original = keywords._document_counts
    checked = []

    def counts(conn, terms, **kwargs):
        with pytest.raises(sqlite3.DatabaseError) as caught:
            conn.execute('ATTACH DATABASE ? AS other', (str(tmp_path / 'forbidden.db'),))
        assert caught.value.sqlite_errorname == 'SQLITE_AUTH'
        assert not (tmp_path / 'forbidden.db').exists()
        result = original(conn, terms, **kwargs)
        checked.append(result)
        return result

    monkeypatch.setattr(keywords, '_document_counts', counts)
    pool = producers.ProducerPool(workers=1)
    try:
        with turn_budget.turn(turn_budget.TurnBudget(wait_s=2)) as budget:
            result = await pool.read(lambda: keywords.document_counts(
                producers.connection(path), ['neutral', 'absent'], stemmed=stemmed))
            assert result == {'neutral': 1} and checked == [result]
            assert budget.sync_read_failed == 1  # The denied ATTACH remains visible to the turn.
    finally:
        pool.executor.shutdown(wait=True)
