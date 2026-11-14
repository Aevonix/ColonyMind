"""Excessively nested tool results retain the existing malformed-result handling."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from colonymind_hermes import guard, tools


# Deep enough to raise RecursionError in the JSON decoder on Python 3.11, 3.12 and 3.13.
DEEP_RESULT = '[' * 100000 + ']'


@pytest.fixture
def budgets(monkeypatch):
    name = '_tool_result_depth_budgets'
    path = Path(__file__).resolve().parents[2] / 'plugins/colonymind-hands/budgets.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_deep_tool_evidence_is_hashed_as_plain_content(budgets):
    assert budgets._evidence(DEEP_RESULT) == {'content:' + hashlib.sha256(DEEP_RESULT.encode()).hexdigest()}


def test_deep_tool_receipt_keeps_its_digest_without_inventing_sources(budgets):
    hand = budgets.HandState('hand-1')
    assert budgets._tool_receipts('web_extract', DEEP_RESULT, {}, hand) == [
        {'tool': 'web_extract', 'result_hash': hashlib.sha256(json.dumps(DEEP_RESULT).encode()).hexdigest()}]
    assert 'fetched_texts' not in hand.plan


@pytest.mark.parametrize('decode', [1, 2], ids=['creator-result', 'reported-result'])
def test_deep_creation_result_reports_failure_without_publishing_authority(monkeypatch, decode):
    # Exercise each decode independently: the other already took its empty-result fallback.
    calls = []
    def loads(value):
        calls.append(value)
        return json.loads(value) if len(calls) == decode else {}

    popped, completed, reported = [], [], []
    standings = SimpleNamespace(
        pop=lambda *args: popped.append(args) or ('mind', 'worker'),
        complete=lambda *args: completed.append(args) or ('mind', 'worker'))
    gate = guard.Guard.__new__(guard.Guard)
    gate.sessions = SimpleNamespace(standings=standings)
    gate.client = SimpleNamespace(post=lambda path, **kwargs: reported.append((path, kwargs['json'])))
    gate._task_id = lambda session, task: task
    monkeypatch.setattr(guard, 'json', SimpleNamespace(loads=loads, dumps=json.dumps))
    gate._report('cronjob_manage', {'action': 'create'}, DEEP_RESULT, 'session-1', 'task-1', 'call-1')
    assert calls == [DEEP_RESULT, DEEP_RESULT]
    assert len(popped) == 1 and completed == []
    assert reported == [(guard.CRON_ORIGIN_ROUTE,
                         {'job_id': '', 'task_id': 'task-1', 'tool_call_id': 'call-1',
                          'action': 'create', 'ok': False})]


def test_deep_read_result_keeps_literal_text_and_recording_note():
    assert json.loads(tools._noted(DEEP_RESULT)) == {'result': DEEP_RESULT, 'note': tools.RECORDED}
