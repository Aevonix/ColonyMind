"""Selected work readers share a bounded model-request snapshot."""
import copy
import json

from colonymind.turns.executions import request_work_context


def rows(result):
    return [json.loads(line) for line in result['text'].splitlines()
            if line.startswith('{') and '"source": "native_kanban_coverage"' not in line]


def concurrent_view():
    return {'native_kanban': {'available': True, 'items': [], 'total': 0, 'recent_total': 8,
                'selection': 'configured_boards', 'partial': False,
                'boards': [{'board': board, 'available': True} for board in ('default', 'colonymind-drafts')],
                'recent': [{'native_task_id': 'old-board-task', 'label': 'Previous inventory review',
                    'status': 'done', 'liveness': 'native_terminal_record'}]},
            'reported_worker': {'available': True, 'items': [
                {'label': 'Feed transport', 'state': 'uncertain', 'liveness': 'unverified'},
                {'label': 'Old model download', 'state': 'exited', 'freshness': 'stale',
                 'liveness': 'unverified', 'record_kind': 'terminal_report',
                 'result_refs': [{'kind': 'artifact', 'reference': 'r' * 512,
                                  'verification': 'v' * 160} for _ in range(4)]},
                {'label': 'Delivery', 'state': 'reported_ready', 'liveness': 'unverified'}]},
            'worker_work': {'available': True, 'items': [], 'total': 0},
            'native_cron': {'available': True, 'total': 1, 'items': [
                {'execution_id': 'c' * 32, 'job_id': 'scheduled-review', 'status': 'running',
                 'name': 'Continuity reviewer', 'liveness': 'unknown', 'record_age_seconds': 134}],
                'recent': [{'execution_id': 'd' * 32, 'name': 'Prior scheduled job',
                            'status': 'completed', 'liveness': 'native_terminal_record'}]}}


def test_real_multi_reader_shape_preserves_coverage_and_budget():
    view = concurrent_view()
    original = copy.deepcopy(view)
    result = request_work_context(view, session_id='observer')
    projected = rows(result)
    assert len(result['text']) <= 4000 and len(projected) <= 8
    assert result['truncated'] and not result['complete']
    assert result['work_sources']['native_kanban']['recent_total'] == 8
    assert result['work_sources']['native_cron']['total'] == 1
    assert 'Additional operational records omitted.' in result['text']
    assert view == original


def test_short_work_remains_visible_with_a_tight_budget():
    result = request_work_context({'native_cron': {
        'available': True, 'items': [{'job_id': 'short-worker', 'status': 'running'}]}}, limit=2)
    projected = rows(result)
    assert len(projected) <= 2
    result = request_work_context({'native_cron': {
        'available': True, 'items': [{'job_id': 'short-worker', 'status': 'running'}]}}, max_chars=950)
    assert any(row.get('job_id') == 'short-worker' for row in rows(result))
    assert len(result['text']) <= 950


def test_many_board_details_respect_the_budget():
    view = concurrent_view()
    view['native_kanban']['boards'] = [{'board': 'b' * 80 + str(n), 'available': False,
                                      'reason': 'unavailable'} for n in range(50)]
    result = request_work_context(view, session_id='observer')
    assert result['truncated'] and len(result['text']) <= 4000


def test_active_sources_precede_history():
    view = {'native_kanban': {'items': [{'native_task_id': 'active-board-task', 'status': 'running'}]},
        'native_cron': {'items': [{'job_id': 'active-schedule', 'status': 'running'}]},
        'reported_worker': {'items': [
            {'label': 'active-download', 'state': 'running'},
            {'label': 'old-download', 'state': 'exited', 'record_kind': 'terminal_report'}]}}
    result = request_work_context(view, session_id='current-child')
    projected = rows(result)
    assert {row['source'] for row in projected} == {
        'native_kanban', 'native_cron', 'reported_worker'}
    assert any(row.get('label') == 'active-download' for row in projected)
    assert projected[-1]['label'] == 'old-download'
    assert len(projected) <= 8 and len(result['text']) <= 4000
    assert not result['truncated'] and not result['complete']


def test_terminal_task_outcome_is_visible():
    view = {'native_kanban': {'available': True, 'items': [], 'total': 0,
            'recent': [{'native_task_id': 'finished-repair', 'status': 'done',
                        'liveness': 'native_terminal_record'}]}}
    original = copy.deepcopy(view)
    result = request_work_context(view, limit=2)
    projected = rows(result)
    assert any(row.get('native_task_id') == 'finished-repair' for row in projected)
    assert view == original
