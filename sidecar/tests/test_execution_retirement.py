"""Execution registry retirement preserves live readers and historical backups."""
import sqlite3

from fastapi import FastAPI, HTTPException
import pytest
from types import SimpleNamespace

from onekey import RequestAuthority
from colonymind import init
from colonymind.api.routers import executions
from colonymind.api.viewer import authorized_viewer
from colonymind.self_model.task_assessments import VERSION, quotation_metadata
from colonymind.turns.executions import has_work_records, request_work_context
from test_upgrade_retirement import ManagedSidecar, instance, rows  # noqa: F401


def test_upgrade_backs_up_and_retires_both_execution_tables(instance, capsys):
    path = instance.home / 'turn-idempotency.db'
    writer = sqlite3.connect(path)
    writer.execute('PRAGMA journal_mode=WAL')
    writer.execute('PRAGMA wal_autocheckpoint=0')
    tables = ('execution_observations', 'execution_runtime_observations')
    for table in tables:
        writer.execute(f'CREATE TABLE {table} (execution_id TEXT PRIMARY KEY, retained TEXT)')
        writer.execute(f'INSERT INTO {table} VALUES (?, ?)', ('execution-1', 'fixture'))
    writer.execute('CREATE TABLE kept (id TEXT PRIMARY KEY)')
    writer.execute("INSERT INTO kept VALUES ('survives')")
    writer.commit()
    instance.service['current'] = ManagedSidecar(instance.events, writer)

    assert init.run_upgrade(instance.args) == 0
    output = capsys.readouterr().out
    backup, = (instance.home / 'backups').glob('*/turn-idempotency.db')
    for table in tables:
        assert (table,) not in rows(path, 'SELECT name FROM sqlite_master')
        assert rows(backup, f'SELECT * FROM {table}') == [('execution-1', 'fixture')]
        assert f'retired table {table}' in output
    assert rows(path, 'SELECT * FROM kept') == [('survives',)]
    events = instance.events
    assert events.index('stop') < events.index('backup_instance') < events.index('retire_tables') < events.index('start')
    assert init.retired_tables_present(instance.home) == []
    assert init.run_upgrade(instance.args) == 0
    assert 'retired table execution_' not in capsys.readouterr().out
    for table in tables:
        assert rows(backup, f'SELECT * FROM {table}') == [('execution-1', 'fixture')]


def test_only_the_live_execution_read_route_remains():
    app = FastAPI()
    app.include_router(executions.router)
    assert set(app.openapi()['paths']) == {'/v1/host/executions'}
    assert set(app.openapi()['paths']['/v1/host/executions']) == {'get'}


def test_registry_rows_and_forecasts_are_absent_from_work_context():
    view = {'items': [{'execution_id': 'retired'}], 'recent': [{'execution_id': 'retired-recent'}]}
    assert not has_work_records(view)
    view['native_cron'] = {'items': [{'job_id': 'live', 'forecast': {'status': 'retired-forecast'}}]}
    result = request_work_context(view)
    assert has_work_records(view)
    assert 'live' in result['text'] and 'retired' not in result['text']
    assert set(result['work_sources']) == {'native_cron', 'native_kanban', 'reported_worker', 'native_processes'}
    assert 'input_provenance' not in result and 'native_task_ids' not in result


def test_shared_viewer_requires_authentication_and_resolves_owner(monkeypatch):
    monkeypatch.setenv('COLONYMIND_OWNER_PERSON_ID', 'owner')
    authority = RequestAuthority(principal_id='fixture', credential_id='fixture',
        scopes=frozenset({'context:read'}), viewer_person_id='owner',
        person_ids=frozenset({'owner'}), audiences=frozenset({'viewer'}), authenticated=True)
    request = SimpleNamespace(state=SimpleNamespace(colonymind_authority=authority))
    assert authorized_viewer(request, 'owner') == ('owner', True)
    assert authorized_viewer(request, 'other') == ('other', False)
    request.state.colonymind_authority = RequestAuthority(authenticated=False)
    with pytest.raises(HTTPException) as error:
        authorized_viewer(request, 'owner')
    assert error.value.status_code == 403


def test_retained_assessment_quotations_keep_their_attribution():
    metadata = quotation_metadata({'role': 'assistant', '_task_artifact_assessment': VERSION})
    assert metadata['assessment_context']['kind'] == VERSION
    assert metadata['assessment_context']['owner_approval'] == 'unobserved'
    assert quotation_metadata({'role': 'user', '_task_artifact_assessment': VERSION}) == {}
