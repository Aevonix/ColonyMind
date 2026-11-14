"""Independent watch passes over synthetic native records and dependencies."""
from contextlib import closing
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from colonymind.config import DEFAULTS, ConfigError, load_config
from colonymind.jobwatch import activity, cli, runner, store
from colonymind.work.hermes_jobs import HermesJob, JobSnapshot, digest, read_jobs


@pytest.fixture
def cfg(tmp_path):
    result = deepcopy(DEFAULTS['watch'])
    result.update(hermes_home=str(tmp_path / 'hermes'), tmux=False, notice_command=['test-sender'])
    return result


def snapshot(*jobs, available=('process',)):
    result = JobSnapshot()
    result.extend(jobs)
    result.available.update(available)
    return result


def healthy(url, timeout, **kwargs):
    return (200, {'health_counters': {}}) if url.endswith('/status') else (200, None)


def invoke(tmp_path, cfg, now, jobs=None, request=healthy, notices=None, **kwargs):
    return runner.run(cfg, tmp_path, 'http://sidecar', now=now, jobs=jobs, request=request,
                      notify=lambda message, argv: notices.append(message) if notices is not None else None, **kwargs)


def test_stuck_and_busy_compile(tmp_path, cfg):
    notices = []
    idle = HermesJob('idle', 'process', 'build', 'running', 0, cpu=1, output=digest('0'), source='process')
    busy = replace(idle, id='busy')
    for now in range(0, 2101, 300):
        findings = invoke(tmp_path, cfg, now, snapshot(idle, replace(busy, cpu=now + 1)), notices=notices)
    assert len(findings) == 1
    assert findings[0]['subject'] == idle.subject
    assert len(notices) == 1
    invoke(tmp_path, cfg, 2400, snapshot(replace(idle, state='finished'), busy), notices=notices)
    assert len(notices) == 2


def test_unknown_activity_or_missing_source_never_claims_recovery(tmp_path, cfg):
    job = HermesJob('idle', 'process', 'build', 'running', 0, cpu=1, output='hash', source='process')
    notices = []
    for now in range(0, 1801, 300):
        invoke(tmp_path, cfg, now, snapshot(job), notices=notices)
    findings = invoke(tmp_path, cfg, 2100, snapshot(available=()), notices=notices)
    assert len(findings) == 1
    assert len(notices) == 1
    invoke(tmp_path, cfg, 2400, snapshot(replace(job, cpu=None, output=None)), notices=notices)
    assert len(notices) == 1
    invoke(tmp_path, cfg, 4500, snapshot(job), notices=notices)
    assert len(notices) == 1
    invoke(tmp_path, cfg, 4800, snapshot(replace(job, cpu=2)), notices=notices)
    assert len(notices) == 2


def test_three_failures_success_and_different_error_reset(tmp_path, cfg):
    notices = []
    def receipt(i, code=1, error='same'):
        return HermesJob(str(i), 'process', 'build', 'finished', 0, i * 10, code,
                         signature='command-hash', error_hash=error, source='process')
    jobs = snapshot(receipt(1), receipt(2), receipt(3))
    assert len(invoke(tmp_path, cfg, 100, jobs, notices=notices)) == 1
    invoke(tmp_path, cfg, 200, jobs, notices=notices)
    assert len(notices) == 1
    jobs.append(receipt(4, 0))
    assert invoke(tmp_path, cfg, 300, jobs, notices=notices) == []
    jobs.extend([receipt(5), receipt(6)])
    assert invoke(tmp_path, cfg, 400, jobs, notices=notices) == []
    jobs.append(receipt(7, error='other'))
    assert invoke(tmp_path, cfg, 500, jobs, notices=notices) == []
    jobs.extend([receipt(8, error='other'), receipt(9, error='other')])
    assert len(invoke(tmp_path, cfg, 600, jobs, notices=notices)) == 1
    assert len(notices) == 3


def test_over_limit_and_turn_limit(tmp_path, cfg):
    jobs = snapshot(HermesJob('time', 'kanban', 'task', 'running', 0, limits={'wall_seconds': 10}),
                    HermesJob('turn', 'hand', 'task', 'running', 0, limits={'max_turns': 3}, usage={'turns': 4}),
                    HermesJob('none', 'cron', 'task', 'running', 0))
    findings = invoke(tmp_path, cfg, 11, jobs)
    assert len(findings) == 2
    assert {item['check'] for item in findings} == {'limit'}


def test_repeat_expires_once_and_a_new_burst_reopens(tmp_path, cfg):
    notices = []
    cfg['repeat_window_hours'] = 1
    def burst(start):
        return snapshot(*(HermesJob(str(start + i), 'process', 'make', 'finished', start,
            start + i, 1, signature='command-hash', error_hash='same', source='process') for i in range(3)))
    assert len(invoke(tmp_path, cfg, 100, burst(10), notices=notices)) == 1
    assert len(notices) == 1
    assert invoke(tmp_path, cfg, 4000, snapshot(), notices=notices) == []
    assert len(notices) == 2 and 'no new failures in the last 1 hours.' in notices[-1]
    assert invoke(tmp_path, cfg, 4100, snapshot(), notices=notices) == []
    assert len(notices) == 2
    assert len(invoke(tmp_path, cfg, 4200, burst(4150), notices=notices)) == 1
    invoke(tmp_path, cfg, 4300, snapshot(), notices=notices)
    assert len(notices) == 3 and 'failed repeatedly' in notices[-1]


def test_shell_comment_quote_keeps_both_processes(tmp_path):
    command = "/usr/bin/make private-target # don't parse this comment"
    (tmp_path / 'processes.json').write_text(json.dumps([
        {'session_id': 'proc_first', 'command': command},
        {'session_id': 'proc_second', 'command': 'python private-script'}]))
    jobs = read_jobs(tmp_path, 100)
    assert [job.id for job in jobs] == ['proc_first', 'proc_second']
    assert [job.label for job in jobs] == ['make', 'python']
    assert jobs[0].signature == digest(command)
    assert 'process' in jobs.available
    assert 'private-' not in repr(jobs)


def test_malformed_checkpoint_row_does_not_hide_other_rows(tmp_path, caplog):
    from colonymind.health_counters import snapshot as counters
    (tmp_path / 'processes.json').write_text(json.dumps([
        {'session_id': 'bad', 'command': 'make', 'started_at': 'private-invalid-time'},
        {'session_id': 'proc_good', 'command': 'python private-script'}]))
    before = counters().get('context_reads_failed', 0)
    jobs = read_jobs(tmp_path, 100)
    assert [job.id for job in jobs] == ['proc_good']
    assert 'process' not in jobs.available
    assert counters()['context_reads_failed'] == before + 1
    assert 'ValueError' in caplog.text and 'private-invalid-time' not in caplog.text


def test_sidecar_down_once_and_recovery(tmp_path, cfg):
    notices = []
    def down(*args, **kwargs):
        raise TimeoutError('must never be logged')
    for now in (0, 300, 600):
        result = invoke(tmp_path, cfg, now, snapshot(), request=down, notices=notices)
        assert [item['check'] for item in result] == ([] if now == 0 else ['sidecar'])
    assert len(notices) == 1
    assert 'still running independently' in notices[0]
    assert invoke(tmp_path, cfg, 900, snapshot(), notices=notices) == []
    invoke(tmp_path, cfg, 1200, snapshot(), notices=notices)
    assert len(notices) == 2
    assert 'recovered' in notices[-1]


def test_endpoint_break_and_recover(tmp_path, cfg):
    cfg['endpoints'] = [{'name': 'dependency', 'url': 'http://endpoint', 'expect_status': 204}]
    notices = []
    state = [503]
    def request(url, timeout, **kwargs):
        return (state[0], None) if url == 'http://endpoint' else healthy(url, timeout, **kwargs)
    for now in (0, 300):
        invoke(tmp_path, cfg, now, snapshot(), request=request, notices=notices)
    state[0] = 204
    for now in (600, 900):
        invoke(tmp_path, cfg, now, snapshot(), request=request, notices=notices)
    assert len(notices) == 2
    assert 'dependency' in notices[0]


def test_download_stall_and_unreachable_skip(tmp_path, cfg, caplog):
    cfg['downloads']['url'] = 'http://downloads'
    notices = []
    state = [{'hash': 'abc', 'progress': .5, 'num_seeds': 0, 'num_leechs': 0}]
    def request(url, timeout, **kwargs):
        if url.startswith('http://downloads'):
            if state is None:
                raise ConnectionError('private-response')
            return 200, state
        return healthy(url, timeout, **kwargs)
    for now in range(0, 3601, 300):
        result = invoke(tmp_path, cfg, now, snapshot(), request=request, notices=notices)
    assert result[0]['check'] == 'download'
    assert len(notices) == 1
    state = None
    result = invoke(tmp_path, cfg, 3900, snapshot(), request=request, notices=notices)
    assert len(result) == 1
    assert len(notices) == 1
    assert 'private-response' not in caplog.text
    state = [{'hash': 'abc', 'progress': 1}]
    assert invoke(tmp_path, cfg, 4200, snapshot(), request=request, notices=notices) == []
    assert len(notices) == 2


def test_counter_summary_once_per_day_and_reset(tmp_path, cfg):
    notices = []
    counts = {'context_writes_dropped': 4, 'mind_incomplete.test': 3}
    def request(url, timeout, **kwargs):
        return 200, {'health_counters': dict(counts)}
    for now in (0, 300, 600):
        invoke(tmp_path, cfg, now, snapshot(), request=request, notices=notices)
        counts['context_writes_dropped'] += 1
    assert len(notices) == 1
    counts['context_writes_dropped'] = 1
    invoke(tmp_path, cfg, 86401, snapshot(), request=request, notices=notices)
    assert len(notices) == 2


def test_missing_sources_permissions_and_readonly_list(tmp_path, cfg):
    assert invoke(tmp_path, cfg, 1) == []
    path = tmp_path / 'jobwatch.db'
    assert path.stat().st_mode & 0o777 == 0o600
    before = path.read_bytes()
    assert store.read_findings(path) == []
    assert path.read_bytes() == before
    assert store.read_findings(tmp_path / 'absent.db') == []
    assert not (tmp_path / 'absent.db').exists()


def test_dry_run_never_writes_or_sends(tmp_path, cfg):
    cfg['dependency_failure_passes'] = 1
    notices = []
    def down(*args, **kwargs):
        raise TimeoutError
    result = invoke(tmp_path, cfg, 0, snapshot(), request=down, notices=notices, dry_run=True)
    assert len(result) == 1
    assert not (tmp_path / 'jobwatch.db').exists()
    invoke(tmp_path, cfg, 0, snapshot())
    before = (tmp_path / 'jobwatch.db').read_bytes()
    result = invoke(tmp_path, cfg, 300, snapshot(), request=down, notices=notices, dry_run=True)
    assert len(result) == 1
    assert not notices
    assert (tmp_path / 'jobwatch.db').read_bytes() == before


def make_db(path, sql, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(sql)
        table = sql.split()[2]
        conn.execute(f'INSERT INTO {table} VALUES ({",".join("?" for _ in values)})', values)
        conn.commit()


def test_real_native_shapes_and_readonly_sources(tmp_path, monkeypatch):
    home = tmp_path / 'hermes'
    home.mkdir()
    (home / 'processes.json').write_text(json.dumps([{'session_id': 'proc_running', 'command': 'make all',
        'started_at': 1, 'pid': 42, 'pid_scope': 'host', 'host_start_time': 2, 'owner_task_id': 't'}]))
    results = home / 'logs' / 'process-results'
    results.mkdir(parents=True)
    for i in range(3):
        path = results / f'proc_{i}.json'
        path.write_text(json.dumps({'id': f'proc_{i}', 'command': 'make all', 'started_at': 1,
                                   'exit_code': 2, 'completion_reason': 'exited', 'output': 'private\nerror-secret\n\n'}))
        os.utime(path, (10 + i, 10 + i))
    make_db(home / 'cron' / 'executions.db',
            'CREATE TABLE executions (id,job_id,status,claimed_at,started_at,finished_at)',
            ('cron', 'schedule', 'running', '1970-01-01T00:00:01Z', None, None))
    make_db(home / 'kanban.db',
            'CREATE TABLE task_runs (id,task_id,status,worker_pid,started_at,ended_at,max_runtime_seconds)',
            (1, 't', 'running', 42, 1, None, 60))
    make_db(tmp_path / 'hands.db',
            'CREATE TABLE hands (id,task_id,state,started_at,ended_at,budgets,usage)',
            ('hand', 't', 'running', '1970-01-01T00:00:01Z', None, '{"wall_seconds":120}', '{"turns":4}'))
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    connect = sqlite3.connect
    def readonly(path, *args, **kwargs):
        assert 'mode=ro' in str(path)
        return connect(path, *args, **kwargs)
    monkeypatch.setattr(sqlite3, 'connect', readonly)
    jobs = read_jobs(home, 100, data_dir=tmp_path)
    assert len(jobs) == 7
    assert jobs[0].label == 'make'
    assert jobs[1].error_hash == digest('error-secret')
    assert 'private' not in repr(jobs)
    assert 'error-secret' not in repr(jobs)
    assert {p: p.read_bytes() for p in before} == before
    assert len(read_jobs(home, 100, data_dir=tmp_path, since=12)) == 5
    assert next(j for j in jobs if j.kind == 'kanban').limits == {'wall_seconds': 60}
    assert next(j for j in jobs if j.kind == 'hand').usage == {'turns': 4}


def test_malformed_source_visible_without_content(tmp_path, cfg, caplog):
    home = Path(cfg['hermes_home'])
    home.mkdir()
    (home / 'processes.json').write_text('secret-broken-json')
    from colonymind.health_counters import snapshot as counters
    before = counters().get('context_reads_failed', 0)
    assert invoke(tmp_path, cfg, 0) == []
    assert 'JSONDecodeError' in caplog.text
    assert 'secret-broken-json' not in caplog.text
    assert counters()['context_reads_failed'] == before + 1


def test_tmux_process_tree_cpu_and_metadata(monkeypatch):
    monkeypatch.setattr(activity.Path, 'is_file', lambda _: False)
    def command(argv, **kwargs):
        if argv[0] == 'tmux':
            return SimpleNamespace(returncode=0, stdout='$1|%1|10|1|2|10|0|0|0|0|make|1\n')
        return SimpleNamespace(stdout='10 1 10 10 00:00:00 Mon Nov 2 00:00:00 2026\n11 10 11 10 00:04:00 Mon Nov 2 00:00:01 2026\n')
    monkeypatch.setattr(activity.shutil, 'which', lambda _: '/tmux')
    monkeypatch.setattr(activity.subprocess, 'run', command)
    jobs = snapshot()
    activity.probe(jobs)
    assert len(jobs) == 1
    assert jobs[0].cpu == 240
    assert jobs[0].output
    assert 'tmux' in jobs.available


def test_notice_failure_redacts_command_output(tmp_path, caplog, monkeypatch):
    class Sender(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            assert kwargs['stdout'] == subprocess.DEVNULL
            assert kwargs['stderr'] == subprocess.DEVNULL
            super().__init__(*args, **kwargs)

        def communicate(self, input=None, timeout=None):
            assert input == runner.MESSAGES['sidecar'][0] + '\n'
            return super().communicate(input, timeout)
    monkeypatch.setattr(runner.subprocess, 'Popen', Sender)
    cfg = deepcopy(DEFAULTS['watch'])
    cfg['dependency_failure_passes'] = 1
    cfg['notice_command'] = [sys.executable, '-c',
        "import sys; print('secret-output'); print('secret-error', file=sys.stderr); sys.exit(1)"]
    runner.run(cfg, tmp_path, 'http://sidecar', jobs=snapshot(), request=lambda *a, **k: (503, None))
    assert 'CalledProcessError' in caplog.text
    assert 'secret-' not in caplog.text
    assert store.read_status(tmp_path / 'jobwatch.db')['delivery_failures'] == 1


@pytest.mark.parametrize('change', [
    {'repeat_failures': 0}, {'stuck_minutes': 'nan'}, {'notice_command': 'shell command'},
    {'downloads': {'url': 'file:///tmp/source'}}, {'endpoints': [{'name': 'a\nb', 'url': 'http://a'}]},
    {'endpoints': [{'name': 'a', 'url': 'http://a', 'expect_status': True}]},
])
def test_invalid_config(tmp_path, change, capsys):
    import yaml
    (tmp_path / 'colonymind.yaml').write_text(yaml.safe_dump({'watch': change}))
    assert load_config(tmp_path, environ={}).data["watch"]
    with pytest.raises(ConfigError):
        load_config(tmp_path, environ={}, strict_watch=True)
    assert cli.main(SimpleNamespace(instance=str(tmp_path), list=True, since=None, dry_run=False)) == 1
    assert "watch" in capsys.readouterr().err


def test_cli_list_no_network_or_key_read(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, 'read_api_key', lambda _: pytest.fail('list read key'))
    assert cli.main(SimpleNamespace(instance=str(tmp_path), list=True, since=None, dry_run=False)) == 0
    assert json.loads(capsys.readouterr().out) == {'findings': [], 'delivery_failures': 0}
    assert not (tmp_path / 'jobwatch.db').exists()


@pytest.mark.asyncio
async def test_findings_route_off_thread_and_owner_only(tmp_path, monkeypatch):
    import threading
    from fastapi import HTTPException
    from colonymind.api.routers import watch
    monkeypatch.setattr(watch, 'instance_home', lambda: tmp_path)
    monkeypatch.setattr(watch, 'speaks_for_owner', lambda _: True)
    main_thread = threading.get_ident()
    def reader(path):
        assert threading.get_ident() != main_thread
        assert path == tmp_path / 'jobwatch.db'
        return [{'id': 'finding'}]
    monkeypatch.setattr(watch, 'read_findings', reader)
    assert await watch.findings(None) == [{'id': 'finding'}]
    monkeypatch.setattr(watch, 'speaks_for_owner', lambda _: False)
    monkeypatch.setattr(watch, 'request_authority', lambda _: None)
    monkeypatch.setattr(watch, 'owner_not_granted', lambda _: HTTPException(403))
    with pytest.raises(HTTPException) as error:
        await watch.findings(None)
    assert error.value.status_code == 403


@pytest.mark.asyncio
async def test_http_deadline_covers_body(monkeypatch):
    import asyncio
    from colonymind.jobwatch import network
    class Response:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        async def aiter_bytes(self):
            yield b'partial'
            await asyncio.sleep(10)
            yield b'body'
    class Client:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        def stream(self, *args, **kwargs):
            return Response()
    monkeypatch.setattr(network.httpx, 'AsyncClient', Client)
    with pytest.raises(TimeoutError):
        await network._get('http://example', .01, None, False)


def test_unknown_download_gap_does_not_report_recovery(tmp_path, cfg):
    cfg['downloads']['url'] = 'http://downloads'
    notices = []
    def request(url, timeout, **kwargs):
        if 'downloads' in url:
            return 200, [{'hash': 'abc', 'progress': .5}]
        return healthy(url, timeout, **kwargs)
    for now in range(0, 3601, 300):
        invoke(tmp_path, cfg, now, snapshot(), request=request, notices=notices)
    assert len(invoke(tmp_path, cfg, 5000, snapshot(), request=request, notices=notices)) == 1
    assert len(notices) == 1


def test_process_activity_counts_exited_children_and_guards_pid_reuse(monkeypatch):
    monkeypatch.setattr(activity.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        stdout='10 1 10 10 00:00:00 Mon Nov 2 00:00:00 2026\n'))
    monkeypatch.setattr(activity.Path, 'is_file', lambda _: True)
    fields = ['0'] * 20
    fields[13], fields[14], fields[19] = '1000', '1000', '42'
    def read(path):
        return 'wchar: 123\n' if str(path).endswith('/io') else '10 (process) ' + ' '.join(fields)
    monkeypatch.setattr(activity.Path, 'read_text', read)
    job = HermesJob('id', 'process', 'build', 'running', pid=10, host_start_time=42)
    activity.probe(snapshot(job), tmux=False)
    assert job.cpu > 0
    assert job.output
    reused = replace(job, host_start_time=99)
    activity.probe(snapshot(reused), tmux=False)
    assert reused.cpu is None and reused.output is None


def test_cli_wiring(monkeypatch):
    import sys
    from colonymind import cli as root_cli
    monkeypatch.setattr(sys, 'argv', ['colonymind', 'watch', '--once', '--dry-run', '--since', '2026-10-28'])
    def run(args):
        assert args.once and args.dry_run and args.since == '2026-10-28'
        return 0
    monkeypatch.setattr(cli, 'main', run)
    with pytest.raises(SystemExit) as result:
        root_cli.main()
    assert result.value.code == 0


def notice_rows(tmp_path):
    with closing(store.connect(tmp_path / 'jobwatch.db')) as conn:
        return [dict(row) for row in conn.execute('SELECT * FROM notices ORDER BY id')]


def test_failed_delivery_retried_once_and_cli_reports_count(tmp_path, cfg, monkeypatch, capsys, caplog):
    cfg['dependency_failure_passes'] = 1
    delivered = []
    attempts = []
    def notify(message, command):
        attempts.append(message)
        if len(attempts) == 1:
            raise OSError('private delivery detail')
        delivered.append(message)
    for now in (0, 300, 600):
        result = runner.run(cfg, tmp_path, 'http://sidecar', jobs=snapshot(), now=now,
                            request=lambda *a, **k: (503, None), notify=notify)
        assert len(result) == 1
    assert len(attempts) == 2
    assert delivered == [runner.MESSAGES['sidecar'][0]]
    assert notice_rows(tmp_path)[0]['sent'] == 1
    assert 'OSError' in caplog.text
    assert 'private delivery detail' not in caplog.text
    monkeypatch.setattr(cli, 'read_api_key', lambda _: pytest.fail('list read key'))
    before = (tmp_path / 'jobwatch.db').read_bytes()
    assert cli.main(SimpleNamespace(instance=str(tmp_path), list=True)) == 0
    status = json.loads(capsys.readouterr().out)
    assert status['delivery_failures'] == 1
    assert len(status['findings']) == 1
    assert (tmp_path / 'jobwatch.db').read_bytes() == before


def test_repeated_delivery_failure_keeps_one_finding_pending(tmp_path, cfg):
    def notify(*args):
        raise TimeoutError
    for now in (0, 300, 600):
        runner.run(cfg, tmp_path, 'http://sidecar', jobs=snapshot(), now=now,
                   request=lambda *a, **k: (503, None), notify=notify)
    rows = notice_rows(tmp_path)
    assert len(rows) == 1 and rows[0]['sent'] == 0
    with closing(store.connect(tmp_path / 'jobwatch.db')) as conn:
        assert conn.execute('SELECT COUNT(*) FROM findings').fetchone()[0] == 1
    assert store.read_status(tmp_path / 'jobwatch.db')['delivery_failures'] == 1


def test_pending_pairs_are_dropped_across_reopening(tmp_path, cfg):
    cfg['dependency_failure_passes'] = 1
    attempts = []
    def notify(message, command):
        attempts.append(message)
        raise OSError
    for now, status in enumerate((503, 200, 503, 200)):
        runner.run(cfg, tmp_path, 'http://sidecar', jobs=snapshot(), now=now,
                   request=lambda *a, **k: (status, None), notify=notify)
    assert attempts == [runner.MESSAGES['sidecar'][0]] * 2
    assert notice_rows(tmp_path) == []
    delivered = []
    invoke(tmp_path, cfg, 300, snapshot(), notices=delivered)
    assert delivered == []
    assert notice_rows(tmp_path) == []


@pytest.mark.parametrize('after_delivery', [False, True])
def test_crash_leaves_committed_notice_pending(tmp_path, cfg, monkeypatch, after_delivery):
    cfg['dependency_failure_passes'] = 1
    delivered = []
    original = runner.deliver_pending
    def crash(*args):
        raise KeyboardInterrupt
    if not after_delivery:
        monkeypatch.setattr(runner, 'deliver_pending', crash)
    def notify(message, command):
        delivered.append(message)
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        runner.run(cfg, tmp_path, 'http://sidecar', jobs=snapshot(), now=0,
                   request=lambda *a, **k: (503, None), notify=notify)
    assert notice_rows(tmp_path)[0]['sent'] == 0
    monkeypatch.setattr(runner, 'deliver_pending', original)
    invoke(tmp_path, cfg, 300, snapshot(), request=lambda *a, **k: (503, None), notices=delivered)
    assert delivered == [runner.MESSAGES['sidecar'][0]] * (2 if after_delivery else 1)
    assert notice_rows(tmp_path)[0]['sent'] == 1


def test_dry_run_preserves_pending_open_and_close_notices(tmp_path, cfg):
    cfg['dependency_failure_passes'] = 1
    def notify(*args):
        raise OSError
    runner.run(cfg, tmp_path, 'http://sidecar', jobs=snapshot(), now=0,
               request=lambda *a, **k: (503, None), notify=notify)
    with closing(store.connect(tmp_path / 'jobwatch.db')) as conn, conn:
        runner.Pass(conn, 300).finding('sidecar', 'sidecar', False)
    before = (tmp_path / 'jobwatch.db').read_bytes()
    runner.run(cfg, tmp_path, 'http://sidecar', jobs=snapshot(), now=600, request=healthy,
               notify=lambda *a: pytest.fail('dry run delivered'), dry_run=True)
    assert (tmp_path / 'jobwatch.db').read_bytes() == before
    assert [row['sent'] for row in notice_rows(tmp_path)] == [0, 0]


@pytest.mark.parametrize('check', ['stuck', 'repeat', 'limit', 'endpoint', 'download', 'sidecar', 'counters'])
def test_every_check_names_its_subject_on_open_and_close(tmp_path, cfg, check):
    from datetime import datetime
    notices = []
    job = HermesJob('job', 'process', 'make', 'running', 0, cpu=1, output='hash', source='process')
    jobs = snapshot()
    if check == 'stuck':
        jobs = snapshot(job)
    elif check == 'limit':
        jobs = snapshot(replace(job, cpu=None, output=None, limits={'wall_seconds': 1}))
    elif check == 'repeat':
        jobs = snapshot(*(replace(job, id=str(i), state='finished', ended_at=i, exit_code=1,
                                  signature='command', error_hash='error') for i in range(3)))
    cfg['endpoints'] = [{'name': 'artifact service', 'url': 'http://endpoint', 'expect_status': 200}]
    cfg['downloads']['url'] = 'http://downloads'
    recovered = False
    counts = {'loop_stalls': 4, 'context_writes_dropped': 1, 'mind_incomplete.test': 2}
    def request(url, timeout, **kwargs):
        if url == 'http://endpoint':
            return (503 if check == 'endpoint' and not recovered else 200), None
        if url.startswith('http://downloads'):
            return 200, ([{'hash': 'hash', 'name': 'release archive', 'progress': .5}]
                         if check == 'download' and not recovered else [])
        if url.endswith('/health'):
            return (503 if check == 'sidecar' and not recovered else 200), None
        return 200, {'health_counters': dict(counts)}
    for now in range(0, 3601, 300):
        invoke(tmp_path, cfg, now, jobs, request=request, notices=notices)
        if now == 0 and check == 'counters':
            counts = {'loop_stalls': 6, 'context_writes_dropped': 4, 'mind_incomplete.test': 3}
    assert len(notices) == 1
    recovered = True
    jobs = snapshot(replace(job, state='finished', ended_at=3700, exit_code=0, signature='command'))
    invoke(tmp_path, cfg, 3900, jobs, request=request, notices=notices)
    assert len(notices) == (1 if check == 'counters' else 2)
    subjects = {'endpoint': ['artifact service'], 'download': ['release archive'], 'sidecar': ['ColonyMind'],
                'counters': ['ColonyMind', 'context_writes_dropped +3', 'mind_incomplete.test +1']}
    expected = subjects.get(check, ['make', 'process', 'started ' + datetime.fromtimestamp(0).strftime('%H:%M')])
    for message in notices:
        assert all(subject in message for subject in expected)


def test_process_notice_never_contains_command_arguments(tmp_path, cfg):
    home = Path(cfg['hermes_home'])
    home.mkdir()
    (home / 'processes.json').write_text(json.dumps([
        {'session_id': 'job', 'command': '/usr/bin/make private-argument', 'started_at': 0}]))
    jobs = read_jobs(home, 10)
    jobs[0].limits = {'wall_seconds': 1}
    notices = []
    invoke(tmp_path, cfg, 10, jobs, notices=notices)
    assert len(notices) == 1 and 'make (process, started ' in notices[0]
    assert '/usr/bin' not in notices[0] and 'private-argument' not in notices[0]
    data = (tmp_path / 'jobwatch.db').read_bytes()
    assert b'private-argument' not in data and b'/usr/bin' not in data


@pytest.mark.parametrize('check', ['sidecar', 'endpoint'])
def test_dependency_requires_consecutive_failures(tmp_path, cfg, check):
    cfg['endpoints'] = [{'name': 'dependency', 'url': 'http://endpoint', 'expect_status': 200}]
    notices = []
    for now, status in enumerate((503, 200, 503, 503, 503, 200, 200)):
        def request(url, timeout, **kwargs):
            target = url.endswith('/health') if check == 'sidecar' else url == 'http://endpoint'
            return (status, None) if target else healthy(url, timeout, **kwargs)
        result = invoke(tmp_path, cfg, now * 300, snapshot(), request=request, notices=notices)
        assert len(result) == (1 if now in (3, 4) else 0)
        assert len(notices) == (0 if now < 3 else 1 if now < 5 else 2)
    assert 'recovered' in notices[-1]


def test_loop_stalls_do_not_send_notices(tmp_path, cfg):
    notices = []
    counts = {'loop_stalls': 0, 'context_writes_dropped': 0}
    def request(*args, **kwargs):
        return 200, {'health_counters': dict(counts)}
    for now in (0, 300, 86400, 86700):
        counts['loop_stalls'] += 1
        assert invoke(tmp_path, cfg, now, snapshot(), request=request, notices=notices) == []
    assert notices == []
    for now in (87000, 87300):
        counts['context_writes_dropped'] += 1
        invoke(tmp_path, cfg, now, snapshot(), request=request, notices=notices)
    assert len(notices) == 1
    assert 'context_writes_dropped +1' in notices[0]
    assert 'loop_stalls' not in notices[0]


@pytest.mark.parametrize('field,value', [
    ('dependency_failure_passes', 0), ('dependency_failure_passes', 1.5),
    ('dependency_failure_passes', True), ('dependency_failure_passes', 'nan'),
    ('notice_timeout_s', 0), ('notice_timeout_s', -1),
    ('notice_timeout_s', True), ('notice_timeout_s', 'inf'),
])
def test_notice_settings_validation(cfg, field, value):
    from colonymind.jobwatch.settings import validate_watch
    cfg[field] = value
    with pytest.raises(ConfigError, match='watch.' + field):
        validate_watch(cfg)


def test_notice_settings_defaults_and_overrides(tmp_path):
    cfg = load_config(tmp_path, environ={}).get('watch')
    assert cfg['dependency_failure_passes'] == 2
    assert cfg['notice_timeout_s'] == 30
    (tmp_path / 'colonymind.yaml').write_text('watch:\n  dependency_failure_passes: 3\n  notice_timeout_s: .25\n')
    cfg = load_config(tmp_path, environ={}).get('watch')
    assert cfg['dependency_failure_passes'] == 3
    assert cfg['notice_timeout_s'] == .25


def test_unset_notice_command_records_findings_without_notices(tmp_path, cfg, caplog):
    cfg.update(dependency_failure_passes=1, notice_command=[])
    runner.run(cfg, tmp_path, 'http://sidecar', jobs=snapshot(), now=0,
               request=lambda *a, **k: (503, None))
    assert notice_rows(tmp_path) == []
    status = store.read_status(tmp_path / 'jobwatch.db')
    assert len(status['findings']) == 1 and status['delivery_failures'] == 0
    assert len(caplog.records) == 1 and 'notice command is unset' in caplog.text
    assert runner.MESSAGES['sidecar'][0] not in caplog.text


def test_notice_timeout_kills_process_group_and_backs_off(tmp_path, cfg):
    cfg.update(dependency_failure_passes=1, notice_timeout_s=.3)
    marker = tmp_path / 'sender-pids'
    cfg['notice_command'] = [sys.executable, '-c',
        'import os, pathlib, subprocess, sys, time; '
        'child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]); '
        'pathlib.Path(sys.argv[1]).write_text(f"{os.getpid()} {child.pid}"); time.sleep(60)', str(marker)]
    def run(now):
        runner.run(cfg, tmp_path, 'http://sidecar', jobs=snapshot(), now=now,
                   request=lambda *a, **k: (503, None))
    run(0)
    pids = marker.read_text()
    def alive(pid):
        state = subprocess.run(['ps', '-o', 'stat=', '-p', pid], capture_output=True,
                               text=True, check=False).stdout.strip()
        return bool(state) and not state.startswith('Z')
    deadline = time.monotonic() + 2
    while any(alive(pid) for pid in pids.split()) and time.monotonic() < deadline:
        time.sleep(.01)
    assert not any(alive(pid) for pid in pids.split())
    for now in (300, 600, 3599):
        run(now)
    assert marker.read_text() == pids
    assert store.read_status(tmp_path / 'jobwatch.db')['delivery_failures'] == 1
    assert notice_rows(tmp_path)[0]['sent'] == 0
    run(3602)
    assert marker.read_text() != pids
    assert store.read_status(tmp_path / 'jobwatch.db')['delivery_failures'] == 2
    cfg['notice_command'] = [sys.executable, '-c', 'import sys; sys.stdin.read()']
    run(7204)
    assert notice_rows(tmp_path)[0]['sent'] == 1


@pytest.mark.parametrize('pause', ['http', 'notice'])
def test_overlapping_pass_does_not_wait_for_io(tmp_path, cfg, pause):
    cfg['dependency_failure_passes'] = 1
    invoke(tmp_path, cfg, 0, snapshot())
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    errors, notices = [], []
    def request(*args, **kwargs):
        if pause == 'http':
            entered.set()
            assert release.wait(5)
        return 503, None
    def notify(message, command):
        if pause == 'notice':
            entered.set()
            assert release.wait(5)
        notices.append(message)
    def first():
        try:
            runner.run(cfg, tmp_path, 'http://sidecar', jobs=snapshot(), now=300,
                       request=request, notify=notify)
        except BaseException as error:
            errors.append(error)
    def second():
        try:
            invoke(tmp_path, cfg, 301, snapshot(), request=lambda *a, **k: (503, None), notices=notices)
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()
    one, two = threading.Thread(target=first), threading.Thread(target=second)
    one.start()
    try:
        assert entered.wait(3)
        two.start()
        assert finished.wait(2)
        assert not errors
    finally:
        release.set()
        one.join(5)
        if two.ident is not None:
            two.join(5)
    assert not one.is_alive() and not two.is_alive()
    assert not errors
    assert notices == [runner.MESSAGES['sidecar'][0]]
    assert notice_rows(tmp_path)[0]['sent'] == 1


@pytest.mark.parametrize('count', [2, 3, 4, 6, 10])
def test_opening_and_closing_bursts_group_notices(tmp_path, cfg, count):
    jobs = snapshot(*(HermesJob(str(i), 'process', 'make', 'running', 0,
                               limits={'wall_seconds': 1}, source='process') for i in range(count)))
    notices = []
    result = invoke(tmp_path, cfg, 10, jobs, notices=notices)
    assert len(result) == count
    assert len(notices) == (count if count <= 3 else 1)
    if count > 3:
        assert notices[0].startswith(f'Job watch opened {count} findings: limit: {count}.')
        assert notices[0].count('make (process, started ') == 3
    assert all(row['sent'] == 1 for row in notice_rows(tmp_path))
    invoke(tmp_path, cfg, 20, jobs, notices=notices)
    opens = len(notices)
    assert invoke(tmp_path, cfg, 30, snapshot(), notices=notices) == []
    assert len(notices) == opens + (count if count <= 3 else 1)
    assert all('over-limit task has ended' in message for message in notices[opens:])
    if count > 3:
        assert notices[-1].startswith(f'Job watch closed {count} findings: limit: {count}.')
        assert notices[-1].count('make (process, started ') == 3


def test_combined_pending_open_and_close_are_dropped(tmp_path, cfg):
    cfg['dependency_failure_passes'] = 1
    cfg['endpoints'] = [{'name': str(i), 'url': f'http://endpoint{i}', 'expect_status': 200}
                        for i in range(4)]
    attempts = []
    def notify(message, command):
        attempts.append(message)
        raise TimeoutError
    runner.run(cfg, tmp_path, 'http://sidecar', now=0, jobs=snapshot(),
               request=lambda *a, **k: (503, None), notify=notify)
    assert len(attempts) == 1
    assert attempts[0].startswith('Job watch opened 5 findings: endpoint: 4, sidecar: 1.')
    notices = []
    invoke(tmp_path, cfg, 300, snapshot(), notices=notices)
    assert notices == []
    assert notice_rows(tmp_path) == []
    invoke(tmp_path, cfg, 3602, snapshot(), notices=notices)
    assert notices == []
    assert notice_rows(tmp_path) == []
