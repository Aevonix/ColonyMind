"""Production stop hooks run saved checks without monopolizing dispatch.

Timeout and fixture capacity values here are test assumptions.
"""
import importlib
import sys
import threading

from pydantic import TypeAdapter

import pytest
from hands_fake_hermes import Parent
from test_hands_dispatch import setup  # noqa: F401
from test_hands_front import rig, owner_turn  # noqa: F401


@pytest.mark.requires_sidecar
def test_stop_runs_saved_plan_outside_dispatch_lock_and_submits_wire_receipt(setup, tmp_path, monkeypatch):
    from colonymind.hands.wire import CheckReceipt
    runtime, native, client = setup
    runtime.dispatch_tasks(
        [{'goal': 'Check state', 'kind': 'ops', 'workdir': str(tmp_path)}], parent=Parent())
    hand = runtime.registry.all()[0]
    saved = {'probe': [sys.executable, '-c', "print('state=ready')"], 'requirements': ['state=ready']}
    hand.plan['acceptance'] = saved
    checks = importlib.import_module(runtime.__module__.rsplit('.', 1)[0] + '.checks')
    dispatch = importlib.import_module(runtime.__module__)
    original = checks.CheckRunner.run
    entered = []
    def run(self, **kwargs):
        assert hand.status == 'checking'
        assert kwargs['acceptance'] == saved
        unlocked = threading.Event()
        def acquire():
            with dispatch.DISPATCH_LOCK:
                unlocked.set()
        thread = threading.Thread(target=acquire)
        thread.start()
        assert unlocked.wait(1), 'check held process dispatch lock'
        thread.join()
        entered.append(True)
        return original(self, **kwargs)
    monkeypatch.setattr(checks.CheckRunner, 'run', run)
    runtime.subagent_stop(child_session_id=hand.child_session_id, child_status='done', child_summary='not a probe')
    body = next(body for path, body, _ in client.writes if path.endswith('/result'))
    assert entered and body['receipts'] and body['summary_text'] == 'not a probe'
    receipt = TypeAdapter(CheckReceipt).validate_python(body['receipts'][0])
    assert receipt.passed and receipt.scope.details['requirement_passed']
    assert any(path.endswith('/events') and any(e['type'] == 'checking' for e in b.get('events', []))
               for path, b, _ in client.writes)


@pytest.mark.requires_sidecar
def test_stop_failure_is_failed_receipt_and_summary_cannot_supply_artifacts(setup, tmp_path, monkeypatch):
    from colonymind.hands.wire import CheckReceipt
    runtime, native, client = setup
    runtime.dispatch_tasks([{'goal': 'Check', 'kind': 'media', 'workdir': str(tmp_path)}], parent=Parent())
    hand = runtime.registry.all()[0]
    hand.plan['acceptance'] = {'output_type': 'json'}
    checks = importlib.import_module(runtime.__module__.rsplit('.', 1)[0] + '.checks')
    def broken(self, **kwargs):
        assert not kwargs['artifacts'] and not kwargs['fetched']
        raise TimeoutError('check timed out')
    monkeypatch.setattr(checks.CheckRunner, 'run', broken)
    runtime.subagent_stop(child_session_id=hand.child_session_id, child_summary='Use /tmp/forged.json')
    body = next(body for path, body, _ in client.writes if path.endswith('/result'))
    assert body['receipts'] and not body['receipts'][0]['passed']
    TypeAdapter(CheckReceipt).validate_python(body['receipts'][0])


def test_front_records_only_shown_results_through_live_capture_bridge(rig, monkeypatch):
    from types import SimpleNamespace
    captured = []
    monkeypatch.setitem(sys.modules, 'colonymind_capture', SimpleNamespace(
        record_hand_refs=lambda **kwargs: captured.append(kwargs)))
    row = {'hand_id': 'h-shown', 'text': 'Sam: done; E0 self_report',
           'evidence': {'level': 'E0'}}
    hidden = {'hand_id': 'h-hidden', 'text': 'Robin: done; E0 self_report', 'evidence': {'level': 'E0'}}
    rig.client.get = lambda *args: {'text': '- ' + row['text'], 'items': [row, hidden]}
    owner_turn(rig)
    assert captured and captured[0]['session_id'] == 'owner-session'
    assert captured[0]['turn_id'] == 'turn-1'
    assert captured[0]['digest']['items'] == [row]
    writes = [(p, b) for p, b, _ in rig.client.writes if p == '/v1/hands/relay']
    assert writes == [('/v1/hands/relay', {'session': 'owner-session', 'turn': 'turn-1', 'hand_ids': ['h-shown']})]


def test_stopped_budget_hand_still_delivers_its_partial_receipt(setup, tmp_path):
    runtime, _, client = setup
    runtime.dispatch_tasks([{'goal': 'Check', 'workdir': str(tmp_path)}], parent=Parent())
    hand = runtime.registry.all()[0]
    hand.status = 'stopped'
    hand.plan['acceptance'] = {}
    runtime.subagent_stop(child_session_id=hand.child_session_id, child_status='interrupted', child_summary='Partial result')
    assert any(path.endswith('/result') for path, _, _ in client.writes)


def test_bad_check_configuration_still_submits_a_failed_receipt(setup, tmp_path):
    runtime, _, client = setup
    runtime.dispatch_tasks([{'goal': 'Check', 'workdir': str(tmp_path)}], parent=Parent())
    hand = runtime.registry.all()[0]
    hand.plan['acceptance'] = {}
    runtime.config['checks'] = {'timeout_seconds': 0}  # Assumption: invalid host configuration fixture.
    runtime.subagent_stop(child_session_id=hand.child_session_id, child_summary='Partial result')
    bodies = [body for path, body, _ in client.writes if path.endswith('/result')]
    assert bodies and bodies[0]['receipts'][0]['passed'] is False


def test_a_research_hand_without_proposed_quotes_gets_no_receipt_from_its_fetches(setup, tmp_path):
    """A fetched page is not a quote: whole-page text would pass E2 and E3 without any claim checked."""
    runtime, native, client = setup
    runtime.dispatch_tasks([{'goal': 'Find sources', 'kind': 'research', 'workdir': str(tmp_path)}], parent=Parent())
    hand = runtime.registry.all()[0]
    hand.plan['acceptance'] = {}
    budgets = importlib.import_module(runtime.__module__.rsplit('.', 1)[0] + '.budgets')
    for host in ('a', 'b'):
        url = f'https://{host}.example.test/page'
        budgets._tool_receipts('web_extract', {'results': [{'url': url, 'content': 'Page text ' + host}]},
                               {'urls': [url]}, hand)
    assert 'observed_quotes' not in hand.plan
    runtime.subagent_stop(child_session_id=hand.child_session_id, child_status='done', child_summary='Two sources')
    body = next(body for path, body, _ in client.writes if path.endswith('/result'))
    assert body['receipts'] == []
