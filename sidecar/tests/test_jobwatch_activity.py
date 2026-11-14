"""Non-/proc progress records and detached terminal activity."""
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import subprocess
from types import SimpleNamespace

import pytest

from colonymind.config import DEFAULTS
from colonymind.health_counters import snapshot as counters
from colonymind.jobwatch import activity, runner, store
from colonymind.work.hermes_jobs import HermesJob, JobSnapshot


@pytest.fixture
def watch(tmp_path, monkeypatch):
    config = deepcopy(DEFAULTS['watch'])
    home = tmp_path / 'hermes'
    home.mkdir()
    config.update(hermes_home=str(home), tmux=False, notice_command=['test-sender'])
    is_file = Path.is_file
    monkeypatch.setattr(Path, 'is_file', lambda path: False if str(path).startswith('/proc/') else is_file(path))
    state = {'cpu': 1, 'start': 'Mon Nov 2 00:00:00 2026', 'pane': 'make', 'tpgid': 10}
    def command(argv, **kwargs):
        if argv[0] == 'tmux':
            assert '#{pane_current_command}' in argv[-1]
            return SimpleNamespace(returncode=0, stdout=f'$1|%1|10|0|2|10|0|0|0|0|{state["pane"]}|{state.get("default_shell", 1)}\n')
        if state.get('unsupported') and 'tpgid=' in argv[-1]:
            raise subprocess.CalledProcessError(1, argv)
        groups = 'pgid=' in argv[-1]
        assert argv == ['ps', '-axo', 'pid=,ppid=,' + ('pgid=,tpgid=,' if groups else '') + 'time=,lstart=']
        fields = f'{state.get("pgid", 10)} {state["tpgid"]} ' if groups else ''
        child = f'11 10 {fields}0 {state["start"]}\n' if state.get('children') else ''
        return SimpleNamespace(stdout=f'10 1 {fields}{state["cpu"]} {state["start"]}\n' + child)
    monkeypatch.setattr(activity.subprocess, 'run', command)
    monkeypatch.setattr(activity.shutil, 'which', lambda _: '/tmux')
    notices = []
    def run(now):
        return runner.run(config, tmp_path, 'http://sidecar', now=now,
                          request=lambda *a, **k: (200, {'health_counters': {}}),
                          notify=lambda message, argv: notices.append(message))
    return SimpleNamespace(home=home, config=config, state=state, notices=notices, run=run)


def process(home, output='private-output', **extra):
    row = {'session_id': 'proc_job', 'command': 'make private-argument', 'pid': 10, 'started_at': 0, **extra}
    if output is not None:
        row['output'] = output
    (home / 'processes.json').write_text(json.dumps([row]))


def progress_source(watch, kind, progress=1):
    if kind == 'process':
        process(watch.home, output=f'private-output-{progress}')
    elif kind == 'cron':
        path = watch.home / 'cron'
        path.mkdir(exist_ok=True)
        with closing(sqlite3.connect(path / 'executions.db')) as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS executions (id, job_id, status, started_at, pid)')
            conn.execute('DELETE FROM executions')
            conn.execute("INSERT INTO executions VALUES ('run', 'schedule', 'running', 0, 10)")
            conn.commit()
        (path / 'jobs.json').write_text(json.dumps({'jobs': [{'id': 'schedule', 'schedule': {'kind': 'once'},
            'run_claim': {'by': 'private-owner', 'at': str(progress)}}]}))
    else:
        path = watch.home / 'kanban' / 'boards' / 'board'
        path.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path / 'kanban.db')) as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS task_runs '
                         '(id, task_id, status, started_at, worker_pid, last_heartbeat_at)')
            conn.execute('DELETE FROM task_runs')
            conn.execute("INSERT INTO task_runs VALUES (1, 'task', 'running', 0, 10, ?)", (progress,))
            conn.commit()


@pytest.mark.parametrize('kind', ['process', 'cron', 'kanban'])
def test_without_proc_stuck_once_and_private(watch, tmp_path, kind, caplog):
    progress_source(watch, kind)
    before = {path: path.read_bytes() for path in watch.home.rglob('*') if path.is_file()}
    for now in range(0, 2401, 300):
        findings = watch.run(now)
        assert len(findings) == (1 if now >= 1800 else 0)
    assert findings[0]['check'] == 'stuck'
    assert len(watch.notices) == 1
    assert {path: path.read_bytes() for path in before} == before
    assert b'private-' not in (tmp_path / 'jobwatch.db').read_bytes()
    assert 'private-' not in caplog.text + repr(watch.notices)


@pytest.mark.parametrize('kind', ['process', 'cron', 'kanban'])
@pytest.mark.parametrize('signal', ['cpu', 'output', 'pid_reuse'])
def test_without_proc_progress_or_pid_reuse_resets(watch, kind, signal):
    progress_source(watch, kind)
    for now in range(0, 1500, 300):
        assert watch.run(now) == []
    if signal == 'cpu':
        watch.state['cpu'] = 2
    elif signal == 'output':
        progress_source(watch, kind, progress=2)
    else:
        watch.state['start'] = 'Mon Nov 2 00:25:00 2026'
    for now in range(1500, 3300, 300):
        assert watch.run(now) == []
    assert watch.notices == []
    assert len(watch.run(3300)) == 1
    assert len(watch.notices) == 1


def test_without_proc_keeps_cpu_but_requires_output(watch):
    jobs = JobSnapshot()
    jobs.append(HermesJob('proc_job', 'process', 'make', 'running', pid=10))
    activity.probe(jobs, tmux=False)
    assert jobs[0].cpu == 1 and jobs[0].output is None
    process(watch.home, output=None)
    for now in range(0, 2401, 300):
        assert watch.run(now) == []
    assert watch.notices == []


def test_empty_checkpoint_output_is_a_signal(watch):
    process(watch.home, output='')
    for now in range(0, 1801, 300):
        findings = watch.run(now)
    assert len(findings) == 1


@pytest.mark.parametrize('kind', ['cron', 'kanban'])
def test_missing_progress_stays_unknown(watch, kind):
    progress_source(watch, kind, progress=None)
    if kind == 'cron':
        (watch.home / 'cron' / 'jobs.json').write_text(json.dumps({'jobs': [
            {'id': 'schedule', 'schedule': {'kind': 'interval'}}]}))
    for now in range(0, 2401, 300):
        assert watch.run(now) == []
    assert not watch.notices


def test_progress_read_failure_visible_without_content(watch, caplog):
    process(watch.home)
    jobs = JobSnapshot()
    jobs.append(HermesJob('proc_job', 'process', 'make', 'running', pid=10, cpu=1))
    (watch.home / 'processes.json').write_text('private-broken-json')
    before = counters().get('context_reads_failed', 0)
    activity.record_activity(jobs, str(watch.home))
    assert jobs[0].output is None
    assert counters()['context_reads_failed'] == before + 1
    assert 'JSONDecodeError' in caplog.text and 'private-' not in caplog.text


@pytest.mark.parametrize('shell', sorted(activity.SHELLS))
def test_idle_shell_pane_is_not_a_job(watch, shell):
    watch.config['tmux'] = True
    watch.state['pane'] = shell
    for now in range(0, 2401, 300):
        assert watch.run(now) == []
    assert not watch.notices


def test_detached_program_stuck_once_then_shell_closes(watch, tmp_path):
    watch.config['tmux'] = True
    for now in range(0, 2401, 300):
        findings = watch.run(now)
    assert len(findings) == 1 and findings[0]['check'] == 'stuck'
    assert len(watch.notices) == 1
    watch.state['pane'] = 'zsh'
    assert watch.run(2700) == []
    assert len(watch.notices) == 2
    with closing(store.connect(tmp_path / 'jobwatch.db')) as conn:
        assert store.get(conn, 'inventory') == {}


def test_overlapping_cron_attempts_have_no_run_specific_signal(watch):
    progress_source(watch, 'cron')
    with closing(sqlite3.connect(watch.home / 'cron' / 'executions.db')) as conn:
        conn.execute("INSERT INTO executions VALUES ('other', 'schedule', 'running', 0, 10)")
        conn.commit()
    for now in range(0, 2401, 300):
        assert watch.run(now) == []
    assert not watch.notices


def test_missing_cron_declaration_does_not_hide_kanban_progress(watch):
    progress_source(watch, 'kanban')
    path = watch.home / 'cron'
    path.mkdir()
    with closing(sqlite3.connect(path / 'executions.db')) as conn:
        conn.execute('CREATE TABLE executions (id, job_id, status, started_at, pid)')
        conn.execute("INSERT INTO executions VALUES ('run', 'schedule', 'running', 0, 10)")
        conn.commit()
    for now in range(0, 1801, 300):
        findings = watch.run(now)
    assert len(findings) == 1
    assert len(watch.notices) == 1 and '(kanban,' in watch.notices[0]


def test_linux_record_fallback_does_not_read_profile(watch, monkeypatch):
    jobs = JobSnapshot()
    jobs.append(HermesJob('proc_job', 'process', 'make', 'running', pid=10, cpu=1))
    monkeypatch.setattr(Path, 'is_file', lambda _: True)
    monkeypatch.setattr(activity, '_json', lambda _: pytest.fail('Linux read checkpoint activity'))
    activity.record_activity(jobs, str(watch.home))
    assert jobs[0].cpu == 1 and jobs[0].output is None


@pytest.mark.parametrize('shell', ['bash', 'zsh'])
def test_shell_with_running_script_or_child_is_watched(watch, shell):
    watch.config['tmux'] = True
    watch.state['pane'] = shell
    watch.state.update(children=True, tpgid=11)
    jobs = JobSnapshot()
    activity.probe(jobs)
    job, = jobs
    assert job.kind == 'tmux' and job.pid == 10 and job.cpu == 1
    for now in range(0, 2401, 300):
        findings = watch.run(now)
    assert len(findings) == 1 and findings[0]['check'] == 'stuck'
    assert len(watch.notices) == 1
    watch.state.update(children=False, tpgid=10)
    assert watch.run(2700) == []
    assert len(watch.notices) == 2


@pytest.mark.parametrize('shell', ['bash', 'zsh'])
@pytest.mark.parametrize('terminal', [10, 20])
def test_prompt_with_background_child_or_coproc_is_idle(watch, shell, terminal):
    watch.config['tmux'] = True
    watch.state.update(pane=shell, children=True, pgid=terminal, tpgid=terminal)
    for now in range(0, 2401, 300):
        assert watch.run(now) == []
    assert not watch.notices


@pytest.mark.parametrize('children', [False, True])
def test_explicit_shell_script_pane_is_watched(watch, children):
    watch.config['tmux'] = True
    watch.state.update(pane='bash', default_shell=0, children=children)
    for now in range(0, 1801, 300):
        findings = watch.run(now)
    assert len(findings) == 1 and findings[0]['check'] == 'stuck'
    assert len(watch.notices) == 1


@pytest.mark.parametrize('terminal', [-1, '-', '?', 'unsupported'])
@pytest.mark.parametrize('children', [False, True])
def test_missing_foreground_group_falls_back_to_children(watch, terminal, children, caplog):
    watch.config['tmux'] = True
    watch.state.update(pane='bash', tpgid=terminal, children=children,
                       unsupported=terminal == 'unsupported')
    before = counters().get('context_reads_failed', 0)
    jobs = JobSnapshot()
    activity.probe(jobs)
    assert len(jobs) == int(children)
    if terminal == 'unsupported':
        assert counters()['context_reads_failed'] == before + 1
        assert 'CalledProcessError' in caplog.text
