"""Pinned source checks and explicitly opt-in integration receipts.

Set HERMES_SOURCE_ROOT to either pinned Hermes source for compatibility.
Live cancellation needs the integrator's isolated executor and engine probe.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess

import pytest
from hermes_source import assert_supported_source


@pytest.fixture
def source():
    path = os.environ.get('HERMES_SOURCE_ROOT')
    if not path:
        pytest.skip('integrator: set HERMES_SOURCE_ROOT to the pinned Hermes checkout')
    root = Path(path)
    assert_supported_source(root)
    return root


def test_native_signature_anchor_and_fallbacks(source):
    tree=ast.parse((source/'tools/delegate_tool.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='delegate_task')
    assert {'credentials_cfg','background','parent_agent'} <= {a.arg for a in fn.args.args}
    progress=(source/'tools/delegate_tool_progress.py').read_text()
    assert progress.count('You are a focused subagent working on a specific delegated task.') == 1
    dispatch=(source/'tools/delegate_tool_dispatch.py').read_text()
    assert '_run_sync_with_note(batch, "no_async")' in dispatch
    assert '_run_sync_with_note(batch, "at_capacity")' in dispatch
    assert 'slot_key=slot_key' in dispatch
    delegate=(source/'tools/delegate_tool.py').read_text()
    assert '_MODEL_HIDDEN_TASK_FIELDS = {"acp_command", "acp_args"}' in delegate


def test_only_known_production_callers_import_dispatch_at_call_time(source):
    found=set()
    for directory in ('tools','agent','gateway','hermes_cli','cron'):
        for path in (source/directory).rglob('*.py'):
            if path.name=='async_delegation.py':
                continue
            tree=ast.parse(path.read_text())
            parents={child:parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom) and node.module=='tools.async_delegation':
                    if not any(n.name.startswith('dispatch_async_delegation') for n in node.names):
                        continue
                    found.add(str(path.relative_to(source)))
                    enclosing=parents.get(node)
                    while enclosing is not None and not isinstance(enclosing,(ast.FunctionDef,ast.AsyncFunctionDef)):
                        enclosing=parents.get(enclosing)
                    assert enclosing is not None, 'import-time binding bypasses the shared module wrapper'
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                    assert not node.func.attr.startswith('dispatch_async_delegation'), 'new qualified caller needs review'
    assert found=={'tools/delegate_tool_dispatch.py','tools/cronjob_tools.py'}


def test_stop_hook_omits_agent_and_iteration_budget_is_native(source):
    results=(source/'tools/delegate_tool_results.py').read_text()
    tree=ast.parse(results)
    stop=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and n.args
              and isinstance(n.args[0],ast.Constant) and n.args[0].value=='subagent_stop')
    keys={kw.arg for kw in stop.keywords}
    assert 'child_session_id' in keys
    assert not {'child_subagent_id','agent','parent_agent'} & keys
    delegate=(source/'tools/delegate_tool.py').read_text()
    assert 'max_iterations=default_max_iter' in delegate or 'max_iterations = default_max_iter' in delegate


def test_cancel_releases_live_engine_and_recorded_process_integrator():
    receipt=os.environ.get('HANDS_CANCEL_RECEIPT')
    if not receipt:
        pytest.skip('integrator: run scenario 2 with live Hermes and write HANDS_CANCEL_RECEIPT')
    evidence=json.loads(Path(receipt).read_text())
    assert evidence['hand_started'] is True
    assert evidence['hard_interrupt_requested'] is True
    assert evidence['process_identity_checked'] is True
    assert evidence['recorded_process_exited'] is True
    assert evidence['stream_closed'] is True
    assert evidence['engine_running_after'] < evidence['engine_running_before']
    assert evidence['unrelated_process_survived'] is True


def test_socket_adapter_integrator():
    if os.environ.get('HANDS_RUN_ADAPTER_CHECK') != '1':
        pytest.skip('integrator: run adapter suite where local sockets and installed Hermes are available')
    import sys
    root=Path(__file__).resolve().parents[2]
    commands = [
        [sys.executable,'scripts/check_hermes_record.py'],
        [sys.executable,'-m','pytest','tests/hermes_adapter','-q',
         '--ignore=tests/hermes_adapter/test_hands_compat.py'],
    ]
    for command in commands:
        result=subprocess.run(command,cwd=root,capture_output=True,text=True)
        assert result.returncode==0, result.stdout+result.stderr
