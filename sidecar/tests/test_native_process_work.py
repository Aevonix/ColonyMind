"""Hermes process observations share the owner-only work projection."""
import asyncio
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from onekey import RequestAuthority
from colonymind import config
from colonymind.api.routers import executions, host
from colonymind.api.schemas.host import ContextAssembleRequest
from colonymind.health_counters import snapshot
from colonymind.turns import board_observations, hermes_work, reported_workers
from colonymind.turns.executions import _process_view, has_work_records, request_work_context
from colonymind.work import hermes_jobs


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setattr(hermes_jobs, '_process_read_warnings', set())
    home = tmp_path / 'hermes'
    home.mkdir()
    now = time.time()
    (home / 'processes.json').write_text(json.dumps([
        {'session_id': 'proc_running', 'command': '/usr/bin/python --private-argument',
         'started_at': now - 120, 'output': 'PRIVATE_OUTPUT'}]))
    receipts = home / 'logs' / 'process-results'
    receipts.mkdir(parents=True)
    for name, code, reason, age in (
            ('failed', 2, 'exited', 60), ('success', 0, 'exited', 60),
            ('old', 1, 'exited', 3601)):
        path = receipts / f'proc_{name}.json'
        path.write_text(json.dumps({'id': f'proc_{name}', 'command': '/usr/bin/make private-target',
            'started_at': now - 300, 'exit_code': code, 'completion_reason': reason,
            'output': 'PRIVATE_OUTPUT'}))
        os.utime(path, (now - age, now - age))
    monkeypatch.setattr(config, 'load_config', lambda: SimpleNamespace(
        data={'watch': {'hermes_home': str(home)}}))
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', 'owner')
    for module, name in ((hermes_work, 'cron_view'), (board_observations, 'kanban_view'),
                         (reported_workers, 'reported_worker_view')):
        monkeypatch.setattr(module, name, lambda **_: {'available': True, 'items': []})
    return home


def authority(person):
    return RequestAuthority(principal_id='voice-fixture', credential_id='fixture',
        scopes=frozenset({'context:read'}), viewer_person_id=person,
        person_ids=frozenset({person}), audiences=frozenset({'viewer'}), authenticated=True)


def assert_processes(view):
    rows = view['native_processes']['items']
    assert [row['id'] for row in rows] == ['proc_running', 'proc_failed']
    assert [row['label'] for row in rows] == ['python', 'make']
    assert [row['state'] for row in rows] == ['running', 'finished']
    assert all(set(row) == {'id', 'label', 'state', 'started_at', 'ended_at',
                            'exit_code', 'end_reason'} for row in rows)
    assert view['native_processes']['available']
    assert has_work_records(view)


def assert_text(text):
    rows = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    processes = [row for row in rows if row.get('source') == 'native_processes']
    assert [row['label'] for row in processes] == ['python', 'make']
    assert processes[0]['age'] == 'started 2m ago'
    assert processes[1]['age'] == 'ended 1m ago'
    assert 'proc_success' not in text and 'proc_old' not in text
    assert 'private-' not in text and 'PRIVATE_OUTPUT' not in text
    assert len(text) <= 4000


def test_process_records_and_compact_text(hermes_home):
    before = {path: path.read_bytes() for path in hermes_home.rglob('*.json')}
    view = {'native_processes': _process_view()}
    assert_processes(view)
    result = request_work_context(view)
    assert_text(result['text'])
    assert result['work_sources']['native_processes']['status'] == 'observed'
    assert {path: path.read_bytes() for path in before} == before
    limited = _process_view(limit=1)
    assert len(limited['items']) == 1 and limited['truncated'] and limited['total'] == 2


@pytest.mark.parametrize('reason', ['failed', 'timeout', 'killed', 'error'])
def test_failure_reason_without_nonzero_exit(hermes_home, reason):
    path = hermes_home / 'logs' / 'process-results' / 'proc_success.json'
    row = json.loads(path.read_text())
    row['completion_reason'] = reason
    path.write_text(json.dumps(row))
    result = _process_view()
    assert any(item['id'] == 'proc_success' and item['end_reason'] == reason for item in result['items'])


def test_missing_home_is_unavailable(hermes_home, monkeypatch):
    monkeypatch.setattr(config, 'load_config', lambda: SimpleNamespace(
        data={'watch': {'hermes_home': str(hermes_home / 'missing')}}))
    view = {'native_processes': _process_view()}
    assert view['native_processes']['items'] == []
    assert not has_work_records(view)
    coverage = request_work_context(view)['work_sources']['native_processes']
    assert coverage['status'] == 'unavailable' and coverage['items_returned'] == 0


@pytest.mark.parametrize('code', [0, 1])
@pytest.mark.parametrize('age', [60, 7200])
def test_terminal_receipt_wins_over_running_checkpoint(hermes_home, monkeypatch, code, age):
    now = time.time()
    path = hermes_home / 'logs' / 'process-results' / 'proc_running.json'
    path.write_text(json.dumps({'id': 'proc_running', 'command': 'python private-script',
        'started_at': now - 8000, 'exit_code': code, 'completion_reason': 'exited'}))
    os.utime(path, (now - age, now - age))
    scans = []
    original = Path.glob
    def glob(path, pattern):
        scans.append((path, pattern))
        return original(path, pattern)
    monkeypatch.setattr(Path, 'glob', glob)
    view = _process_view()
    rows = [row for row in view['items'] if row['id'] == 'proc_running']
    assert len(rows) == (1 if code and age < 3600 else 0)
    assert all(row['state'] == 'finished' and row['exit_code'] == 1 for row in rows)
    assert view['available']
    assert scans.count((path.parent, 'proc_*.json')) == 1


def test_default_home_uses_watch_configuration(tmp_path, monkeypatch):
    selected = []
    defaults = config.load_config(tmp_path, environ={})
    monkeypatch.setattr(config, 'load_config', lambda: defaults)
    def empty(home, now, **kwargs):
        selected.append(home)
        return hermes_jobs.JobSnapshot()
    monkeypatch.setattr(hermes_jobs, 'read_jobs', empty)
    assert _process_view()['items'] == []
    assert selected == ['~/.hermes']


def test_unreadable_records_are_unavailable_without_payload(hermes_home, monkeypatch, caplog):
    original = Path.read_text
    def unreadable(path, *args, **kwargs):
        if path.is_relative_to(hermes_home):
            raise PermissionError('PRIVATE_DRIVER_TEXT')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', unreadable)
    before = snapshot().get('context_reads_failed', 0)
    result = _process_view()
    assert result['items'] == [] and result['available'] is False
    assert snapshot()['context_reads_failed'] > before
    assert 'PermissionError' in caplog.text and 'PRIVATE_DRIVER_TEXT' not in caplog.text


@pytest.mark.asyncio
async def test_route_and_turn_start_share_processes_for_owner_only(hermes_home):
    app = FastAPI()
    @app.middleware('http')
    async def identity(request, call_next):
        request.state.colonymind_authority = authority(request.headers.get('fixture-person', 'owner'))
        return await call_next(request)
    app.include_router(executions.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://fixture') as client:
        response = await client.get('/v1/host/executions', params={'contact_id': 'owner'})
        assert response.status_code == 200
        assert_processes(response.json())
        assert 'private-' not in response.text and 'PRIVATE_OUTPUT' not in response.text
        response = await client.get('/v1/host/executions', params={
            'contact_id': 'owner', 'projection': 'request', 'reserve_chars': 500})
        assert response.status_code == 200
        assert_text(response.json()['text'])
        assert_text(response.json()['reserved']['text'])
        response = await client.get('/v1/host/executions', params={'contact_id': 'guest'},
                                    headers={'fixture-person': 'guest'})
        assert response.status_code == 200
        body = response.json()
        assert body['items'] == [] and 'native_processes' not in body and 'native_cron' not in body
        denied = await client.get('/v1/host/executions', params={
            'contact_id': 'guest', 'projection': 'request'}, headers={'fixture-person': 'guest'})
        assert denied.status_code == 403
    for person in ('owner', 'guest'):
        body = ContextAssembleRequest(identity={'host_id': 'voice-fixture'},
            context={'contact_id': person, 'session_id': 'session'},
            incoming_message={'role': 'user', 'content': 'Show current work.'})
        result = await host.context_assemble(body, SimpleNamespace(
            state=SimpleNamespace(colonymind_authority=authority(person))))
        sections = [section for section in result.sections if section.id == 'colonymind-executions']
        if person == 'owner':
            assert len(sections) == 1
            assert_text(sections[0].body)
        else:
            assert sections == []


@pytest.mark.asyncio
async def test_missing_home_route_reports_unavailable(hermes_home, monkeypatch):
    monkeypatch.setattr(config, 'load_config', lambda: SimpleNamespace(
        data={'watch': {'hermes_home': str(hermes_home / 'missing')}}))
    request = SimpleNamespace(state=SimpleNamespace(colonymind_authority=authority('owner')))
    result = await executions.active(request, 'owner', limit=8, reserve_chars=0)
    assert result['native_processes']['items'] == []
    assert result['work_sources']['native_processes']['status'] == 'unavailable'


@pytest.mark.asyncio
async def test_slow_reader_is_bounded_and_independent(hermes_home, monkeypatch, caplog):
    started, release = threading.Event(), threading.Event()
    def slow(*args, **kwargs):
        started.set()
        release.wait(2)
        return hermes_jobs.JobSnapshot()
    monkeypatch.setattr(hermes_jobs, 'read_jobs', slow)
    monkeypatch.setattr(hermes_work, 'cron_view', lambda **_: {
        'available': True, 'items': [{'execution_id': 'independent-cron'}]})
    before = snapshot().get('context_reads_failed', 0)
    begin = time.monotonic()
    task = asyncio.create_task(executions.with_queue_work({}, owner=True))
    try:
        await asyncio.sleep(.03)
        assert started.is_set() and not task.done()
        view = await task
        assert time.monotonic() - begin < .3
        assert view['native_processes']['items'] == []
        assert view['work_sources']['native_processes']['status'] == 'unavailable'
        assert view['native_cron']['items'][0]['execution_id'] == 'independent-cron'
        assert snapshot()['context_reads_failed'] > before
        assert 'TimeoutError' in caplog.text
    finally:
        release.set()


def test_process_reader_avoids_other_sources(hermes_home, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('Process view read a cron or kanban database')
    monkeypatch.setattr(hermes_jobs, 'rows', unexpected)
    (hermes_home / 'cron').mkdir()
    (hermes_home / 'cron' / 'executions.db').touch()
    (hermes_home / 'kanban.db').touch()
    assert_processes({'native_processes': _process_view()})


def test_process_read_warning_once_but_every_failure_counted(hermes_home, caplog):
    (hermes_home / 'processes.json').write_text('{')
    before = snapshot().get('context_reads_failed', 0)
    _process_view()
    _process_view()
    assert snapshot()['context_reads_failed'] == before + 2
    assert sum('JSONDecodeError' in record.message for record in caplog.records) == 1
