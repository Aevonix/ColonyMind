"""Retired adapters cannot contribute routes, work or observation views."""
import json
from types import SimpleNamespace

from fastapi import FastAPI
import pytest

from onekey import RequestAuthority
from colonymind.api.routers import executions, hands, host
from colonymind.api.schemas.host import ContextAssembleRequest, MemoryReadRequest
from colonymind.initiatives.store import InitiativeStore
from colonymind.turns import board_observations, hermes_work, reported_workers


def test_retired_routes_and_observation_schema_are_absent():
    app = FastAPI()
    for router in (host.router, host.v2_router, hands.router, executions.router):
        app.include_router(router)
    paths = app.openapi()['paths']
    assert '/v1/hands/triage' not in paths
    assert '/v2/host/turns/source-observation/{turn_id}' not in paths
    assert not any('/local-work' in path or '/local-draft' in path for path in paths)
    assert '/v1/host/executions' in paths
    assert 'observations' not in MemoryReadRequest.model_json_schema()['properties']['source_view']['enum']


@pytest.mark.asyncio
async def test_stale_assigned_initiatives_do_not_render_on_owner_turns(tmp_path, monkeypatch):
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(tmp_path))
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', 'owner')
    store = InitiativeStore(tmp_path)
    try:
        stale = store.create(type='RESEARCH_DEEP_DIVE', description='Historical accepted draft',
            source_type='installed_capabilities', created_by='native_local_work',
            context={'native_job_id': 'retired-job', 'native_execution_id': 'retired-execution'})
        store.update(stale.id, status='assigned', assigned_agent_id='native-cron:retired-execution')
        monkeypatch.setattr(host, '_initiative_store', store)
        for module, name, key in ((hermes_work, 'cron_view', 'job_id'),
                                  (board_observations, 'kanban_view', 'native_task_id'),
                                  (reported_workers, 'reported_worker_view', 'task_id')):
            monkeypatch.setattr(module, name, lambda key=key, **_: {
                'available': True, 'items': [{key: 'live-' + key, 'status': 'running'}], 'recent': []})
        authority = RequestAuthority(principal_id='native', credential_id='fixture',
            scopes=frozenset({'context:read'}), viewer_person_id='owner', person_ids=frozenset({'owner'}),
            audiences=frozenset({'viewer'}), authenticated=True)
        request = SimpleNamespace(state=SimpleNamespace(colonymind_authority=authority))
        full = await executions.active(request, 'owner', limit=8, reserve_chars=0)
        assert set(full['work_sources']) == {'native_cron', 'native_kanban', 'reported_worker', 'native_processes'}
        assert 'local_work' not in full and stale.id not in json.dumps(full)
        projected = await executions.active(request, 'owner', limit=8, projection='request', reserve_chars=0)
        assert stale.id not in projected['text'] and 'retired-job' not in projected['text']
        body = ContextAssembleRequest(identity={'host_id': 'native'},
            context={'contact_id': 'owner', 'session_id': 'later'},
            incoming_message={'role': 'user', 'content': 'What work is active?'})
        context = await host.context_assemble(body, request)
        section, = [section for section in context.sections if section.id == 'colonymind-executions']
        assert stale.id not in section.body and 'retired-job' not in section.body
        assert all(name in section.body for name in ('live-job_id', 'live-native_task_id', 'live-task_id'))
        assert store.get(stale.id).status == 'assigned'
    finally:
        store.close()
