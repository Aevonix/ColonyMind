"""Initial recall uses the same bounded operational view as native requests."""
import copy
import json
from types import SimpleNamespace

import pytest

from onekey import RequestAuthority
from colonymind.api.routers import executions, host
from colonymind.api.schemas.host import ContextAssembleRequest
from colonymind.turns import executions as work
from colonymind.turns import TurnIdempotencyLedger


def shaped_work(view):
    """Generic reader shapes: a completed task and bulky reports."""
    view = copy.deepcopy(view)
    view['native_kanban'] = {
        'available': True, 'items': [], 'boards': [], 'selection': 'explicit',
        'recent': [{'native_board': 'operations', 'native_task_id': 'finished-task',
            'native_run_id': None, 'status': 'done', 'liveness': 'unknown',
            'terminal_result': {'summary': ('已检查所选文件；仍有未验证的配置声明。' * 100)[:1200],
                'summary_chars': 2000, 'truncated': True,
                'run_id': 7, 'authority': 'worker report; external effects not verified',
                'reader': {'tool': 'kanban_show',
                    'arguments': {'task_id': 'finished-task', 'board': 'operations'}}}}]}
    view['reported_worker'] = {'available': True, 'items': [{
        'label': 'Background reader', 'state': 'uncertain', 'liveness': 'unknown',
        'detail_code': 'provider_outcome_uncertain',
        'unrelated_diagnostic': 'Historical diagnostic. ' * 1000}]}
    return view


@pytest.mark.asyncio
async def test_initial_owner_work_is_bounded_and_preserves_result(tmp_path, monkeypatch):
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', 'owner')
    ledger = TurnIdempotencyLedger(tmp_path / 'turns.db')
    retained = []
    async def readers(view, **kwargs):
        result = shaped_work(view)
        retained.append(copy.deepcopy(result))
        return result
    monkeypatch.setattr(executions, 'with_queue_work', readers)
    before = ledger.erasure_watermark('owner')
    for person in ('owner', 'guest'):
        authority = RequestAuthority(principal_id='native-' + person, credential_id='fixture',
            scopes=frozenset({'context:read'}), viewer_person_id=person,
            person_ids=frozenset({person}), audiences=frozenset({'viewer'}), authenticated=True)
        body = ContextAssembleRequest(identity={'host_id': 'native'},
            context={'contact_id': person, 'session_id': 'session'},
            incoming_message={'role': 'user', 'content': 'Inspect the selected source records.'})
        result = await host.context_assemble(body, SimpleNamespace(
            state=SimpleNamespace(colonymind_authority=authority)))
        sections = [s for s in result.sections if s.id == 'colonymind-executions']
        if person == 'guest':
            assert sections == []
            continue
        section, = sections
        assert len(section.body) <= 4000
        rows = [json.loads(line) for line in section.body.splitlines() if line.startswith('{')]
        terminal, = [r for r in rows if r.get('native_task_id') == 'finished-task']
        assert terminal['status'] == 'done' and terminal['liveness'] == 'unknown'
        assert terminal['terminal_result']['run_id'] == 7
        assert terminal['terminal_result']['reader'] == retained[0]['native_kanban']['recent'][0]['terminal_result']['reader']
        assert terminal['terminal_result']['authority'] == 'worker report; external effects not verified'
        original = retained[0]['native_kanban']['recent'][0]['terminal_result']['summary']
        assert terminal['terminal_result']['summary'] != original
        assert original.startswith(terminal['terminal_result']['summary'])
        assert terminal['terminal_result']['truncated'] and terminal['terminal_result']['summary_chars'] == 2000
        assert 'unrelated_diagnostic' not in section.body
        assert 'provider_outcome_uncertain' in section.body
        assert 'external effects remain unverified' in section.body
        assert 'request_input' not in section.body
    assert len(retained) == 1
    assert ledger.erasure_watermark('owner') == before


def test_existing_work_cap_includes_coverage_and_unavailable_footer():
    view = shaped_work({'items': [], 'truncated': False})
    for source in ('native_cron',):
        view[source] = {'available': False, 'items': [], 'reason': 'reader_unavailable'}
    original = copy.deepcopy(view)
    result = work.request_work_context(view, max_chars=4000)
    assert len(result['text']) <= 4000
    assert 'Unavailable sources: native_cron.' in result['text']
    assert not result['complete'] and result['work_sources']['native_cron']['status'] == 'unavailable'
    assert view == original
