"""Socket-free Hermes seams model delegation and admission ordering.

Construction calls subagent_start before detached admission. A native batch uses
one slot and falls back inline at capacity or without a detached consumer. The
fake deliberately retains those hazards so the plugin must prevent them.
"""
from __future__ import annotations

import importlib.util
import ast
from contextlib import nullcontext
import json
import logging
import os
from pathlib import Path
import sys
import threading
from types import MethodType, SimpleNamespace
from typing import Optional
import uuid


def attach_turn_leases(runner, state):
    """Use the selected native lease methods with an in-memory registry.

    Extracting the unchanged methods avoids importing optional transport SDKs.
    Ordinary unit runs without a selected tree use the same small lease shape.
    """
    class Registry:
        def __init__(self):
            self.released = []
            self.result = True
            self.error = None

        def release(self, token):
            if self.error is not None:
                raise self.error
            if self.result:
                self.released.append(token)
            return self.result

        def rebind(self, token, session_id):
            token.session_id = session_id
            return True

    def held(self, session_key, run_generation):
        native = self._peek_session_state(session_key)
        if native is not None and run_generation in native.turn.lease_tokens:
            return self._turn_leases, native.turn.lease_tokens
        return None

    def release(self, session_key, run_generation):
        found = self._held_turn_lease(session_key, run_generation)
        if found is None:
            return False
        registry, tokens = found
        try:
            return registry.release(tokens.pop(run_generation))
        except Exception:
            return False

    def clear(self, session_key, *, run_generation=None):
        if run_generation is not None and not self._is_session_run_current(session_key, run_generation):
            return False
        state.turn.clear()
        self._persist_active_agents()
        return True

    methods = {'_held_turn_lease': held, '_release_turn_lease': release, '_release_running_agent_state': clear}
    if os.environ.get('HERMES_SOURCE_ROOT'):
        from hermes_source import HERMES_ROOT, assert_supported_source
        assert_supported_source()
        source = HERMES_ROOT / 'gateway/run_agent_cache.py'
        tree = ast.parse(source.read_text())
        names = {'_held_turn_lease', '_release_turn_lease', '_rebind_turn_lease', '_release_running_agent_state'}
        body = [node for cls in tree.body if isinstance(cls, ast.ClassDef)
                for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {'logger': logging.getLogger(__name__), 'nullcontext': nullcontext, 'Optional': Optional}
        exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), namespace)
        methods = {name: namespace[name] for name in names}
    # retired_agents: the agents a release hands on where the source's turn state keeps them; none here.
    state.turn = SimpleNamespace(lease_tokens={}, lease=None, retired_agents=[],
        clear=lambda: getattr(runner, '_running_agents', {}).pop('route', None))
    runner._persist_active_agents = lambda: None
    runner._turn_leases = Registry()
    for name, method in methods.items():
        setattr(runner, name, MethodType(method, runner))
    return runner._turn_leases


def hold_turn_lease(runner, state, session_id='owner-session'):
    generation = state.persistent.run_generation
    return state.turn.lease_tokens.setdefault(generation, SimpleNamespace(session_id=session_id))


def load_hands():
    name = 'hands_test_' + uuid.uuid4().hex
    root = Path(__file__).resolve().parents[2] / 'plugins' / 'colonymind-hands'
    spec = importlib.util.spec_from_file_location(name, root / '__init__.py', submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeContext:
    def __init__(self):
        self.hooks, self.middleware, self.tools, self.commands = {}, {}, {}, {}
        self.unloads = []

    def register_hook(self, name, fn, **kwargs):
        self.hooks.setdefault(name, []).append(fn)

    def register_middleware(self, name, fn, **kwargs):
        self.middleware.setdefault(name, []).append(fn)

    def register_tool(self, name, **kwargs):
        self.tools[name] = kwargs

    def register_command(self, name, handler, **kwargs):
        self.commands[name] = handler

    def on_unload(self, fn):
        self.unloads.append(fn)

    def fire(self, name, **kwargs):
        return [fn(**kwargs) for fn in self.hooks.get(name, [])]


class FakeController:
    def __init__(self):
        self.responses, self.writes, self.down, self.rows = {}, [], False, {}
        self.lock = threading.RLock()

    def post(self, path, body, key):
        with self.lock:
            if self.down:
                raise ConnectionError('controller unavailable')
            if key in self.responses:
                return self.responses[key]
            self.writes.append((path, body, key))
            if path == '/v1/hands/definitions/queue/claim':
                answer = {'tasks': []}
            elif path.endswith('/plan'):
                plans = []
                for task in body['tasks']:
                    suffix = uuid.uuid4().hex[:12]
                    plans.append(dict(task_id='t-' + suffix, hand_id='h-' + suffix,
                        name=task.get('name') or 'research-' + suffix[:4], kind=task.get('kind', 'research'),
                        effort=2, risk='normal', role='research', contract={'provider':'text-route','version':'1'},
                        budgets={'wall_seconds':1200,'tool_calls':40,'web_searches':12,'tokens':400000},
                        brief=task['goal'], admit='start', lease_token='lease-' + suffix))
                answer = {'plans': plans}
                for plan in plans:
                    self.rows[plan['hand_id']] = {'id':plan['hand_id'],'task_id':plan['task_id'],
                        'parent_session':body['parent_session'],'intent_version':1,'name':plan['name'],
                        'state':'reserved'}
            else:
                answer = {'ok': True}
            self.responses[key] = answer
            return answer

    def get(self, path, params=None):
        if self.down:
            raise ConnectionError('controller unavailable')
        if path.rsplit('/',1)[-1] in self.rows:
            return dict(self.rows[path.rsplit('/',1)[-1]])
        return {'hands': list(self.rows.values())}


class Parent:
    def __init__(self, sid='owner-session'):
        self.session_id, self._current_turn_id, self._delegate_depth = sid, 'turn-1', 0


class FakeHermes:
    def __init__(self, capacity=4, platform='sms', consumer=True):
        self.capacity, self.platform, self.consumer = capacity, platform, consumer
        self.calls, self.inline, self.live, self.max_seen = [], [], {}, 0
        self.runtime = None
        self.before_admission = None
        self.steers, self.stops = [], []
        self.async_module = SimpleNamespace(dispatch_async_delegation=self.cron,
                                            dispatch_async_delegation_batch=self.admit)
        self.delegate_module = SimpleNamespace(delegate_task=self.delegate_task)

    def context(self, parent):
        return {'platform': self.platform, 'chat_type': 'dm',
                'origin_ref': {'platform': self.platform, 'chat':'5550100', 'message_id':'message-1'},
                'origin':'owner_turn', 'turn':parent._current_turn_id, 'tool_call_id':'call-1'}

    def push_supported(self, context):
        return self.consumer is True

    def active_count(self):
        return len(self.live)

    def max_children(self):
        return self.capacity

    def native_policy(self):
        return {'provider':'text-route','model':'text-model','fallback_providers':[]}

    def delegate_task(self, goal=None, context=None, tasks=None, max_iterations=None, role=None,
                      background=None, output_schema=None, images=None, action=None,
                      subagent_id=None, message=None, parent_agent=None, credentials_cfg=None):
        if action:
            return json.dumps({'ok':True})
        self.calls.append({'tasks':tasks,'background':background,'parent_agent':parent_agent,'credentials_cfg':credentials_cfg})
        child = 'child-' + uuid.uuid4().hex
        subagent = 'subagent-' + uuid.uuid4().hex
        self.runtime.subagent_start(parent_session_id=parent_agent.session_id, child_session_id=child,
                                    child_subagent_id=subagent, child_goal=tasks[0]['goal'])
        if self.before_admission:
            self.before_admission()
        if not background or not self.consumer:
            self.inline.append(tasks)
            return json.dumps({'results': []})
        result = self.async_module.dispatch_async_delegation_batch(goals=[t['goal'] for t in tasks])
        if result['status'] != 'dispatched':
            self.inline.append(tasks)
            return json.dumps({'results': []})
        return json.dumps({**result,'subagent_ids':[subagent]})

    def admit(self, **kwargs):
        if len(self.live) >= self.capacity:
            return {'status':'error','error':'capacity'}
        ident = 'delegation-' + uuid.uuid4().hex
        self.live[ident] = {'delegation_id':ident,'status':'running'}
        self.max_seen = max(self.max_seen, len(self.live))
        return {'status':'dispatched','delegation_id':ident}

    def cron(self, **kwargs):
        return self.admit(**kwargs)

    def ledger(self, ident):
        return self.live.get(ident)

    def steer(self, subagent_id, text):
        self.steers.append((subagent_id,text))
        return True

    def stop(self, subagent_id):
        self.stops.append(subagent_id)
        return True
