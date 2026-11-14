"""Detached admission, local bindings and the shared cron dispatch lock.

The plugin never imports the sidecar. All writes use its frozen HTTP contracts.
The default limits below are design assumptions, not guarantees about a host.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any
import uuid
import weakref

from . import wrap

logger = logging.getLogger(__name__)

DISPATCH_LOCK = threading.RLock()
_RUNTIME = None
TERMINAL = {'done','below_minimum','failed','stopped','superseded','lost','unknown_effect','refused','expired'}
PLAN_FIELDS = {'goal','context','kind','effort','risk','needs','accept','touches','parts','depends_on',
               'name','priority','continue','reconciled','workdir','hand','queue_claim'}
NATIVE_FIELDS = {'goal','context','role','output_schema','images','group'}
# What the front can do about a controller's refusal of its plan. Fixed text: a refusal never echoes values.
REFUSAL_HINTS = {
    'coding_requires_workdir_or_resources': 'Kind coding needs workdir (the repository or directory it changes) or '
        'touches (what it changes). For shell-only work with no repository, use kind ops.',
    'invalid_fields': 'Correct the named task fields to values the delegate_task schema lists.',
    'invalid_name': 'A name takes 1 to 24 letters, digits or hyphens and no reserved word; omit it for a default.',
    'name_in_use': 'Open work already has that name; choose another or omit it.',
    'parts_require_planning_unit': 'Send each part as its own task.',
    'completion_requires_task': 'A completion turn can only continue an existing hand.',
    'task_active': 'That task already has a running attempt; wait for its result.',
    'task_closed': 'That task is closed; plan new work instead of continuing it.',
    'unknown_hand_definition': 'No Hand has that id or exact name. Use colonymind_hands list_hands.',
    'disabled_hand_definition': 'That Hand is disabled. Choose an enabled Hand.',
    'archived_hand_definition': 'That Hand is archived. Choose an enabled Hand.',
}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


class LaunchParent:
    """Forward live parent state, with the admitted identity fixed for native hooks.

    Hermes may compress the original during child construction. Its constructor,
    registry and lifecycle hooks must all see the identity checked at launch.
    """
    def __init__(self, parent, session_id):
        object.__setattr__(self, '_parent', parent)
        object.__setattr__(self, 'session_id', session_id)

    def __getattr__(self, name):
        return getattr(self._parent, name)

    def __setattr__(self, name, value):
        if name == 'session_id':
            raise AttributeError('launch identity is immutable')
        setattr(self._parent, name, value)


def refusal_detail(exc):
    """A controller refusal's stable code, and for a validation refusal its field names; never values."""
    try:
        detail = exc.response.json().get('detail')
    except Exception:
        return {}
    detail = detail.get('code') if isinstance(detail, dict) else detail
    if isinstance(detail, str) and re.fullmatch(r'[a-z][a-z0-9_]{0,63}', detail):
        return {'detail': detail}
    if isinstance(detail, list):
        fields = set()
        for row in detail:
            loc = list(row.get('loc') or ()) if isinstance(row, dict) else []
            loc = loc[1:] if loc[:1] == ['body'] else loc
            if loc and all(isinstance(part, int) or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,39}', str(part))
                           for part in loc):
                fields.add('.'.join(map(str, loc)))
        return {'detail': 'invalid_fields', 'fields': sorted(fields)[:8]}
    return {}


def refusal_answer(planned):
    answer = {'status': 'refused', 'reason': 'controller_refused'}
    detail = planned.get('detail')
    if detail:
        answer['detail'] = detail
        if planned.get('fields'):
            answer['fields'] = list(planned['fields'])
        if detail in REFUSAL_HINTS:
            answer['hint'] = REFUSAL_HINTS[detail]
    return answer


class ControllerClient:
    """Instance bearer authentication with per-write idempotency headers."""
    def __init__(self, settings):
        self.settings = settings

    def _request(self, method, path, *, body=None, key=None, params=None):
        import httpx
        # Assumption: controller calls may take at most two seconds.
        headers = {'Authorization':'Bearer ' + self.settings.api_key}
        if key:
            headers['Idempotency-Key'] = key
        # Planning builds the brief, with recall, before it answers: up to about 2 s measured on the rc clone, so a
        # 2 s bound refused hands the controller had already reserved. Assumption: 15 s for a plan, 2 s otherwise.
        with httpx.Client(timeout=15 if path == '/v1/hands/plan' else 2, trust_env=False) as client:
            response = client.request(method, self.settings.sidecar_url.rstrip('/') + path,
                                      json=body, params=params, headers=headers)
            response.raise_for_status()
            return response.json()

    def post(self, path, body, key):
        return self._request('POST', path, body=body, key=key)

    def get(self, path, params=None):
        return self._request('GET', path, params=params)


class NativeHermes:
    """The six plugin seams pinned by the source compatibility tests."""
    def __init__(self):
        self.delegate_module = importlib.import_module('tools.delegate_tool')
        self.async_module = importlib.import_module('tools.async_delegation')
        self.policy = importlib.import_module('tools.delegate_tool_config')
        self.registry = importlib.import_module('tools.delegate_tool_registry')
        self.results_module = importlib.import_module('tools.delegate_tool_results')
        child_run = importlib.import_module('tools.delegate_tool_child_run')
        self.inactivity_timeout = callable(getattr(child_run._ChildRun, 'wait_liveness_aware', None))

    def context(self, parent):
        from gateway.session_context import get_session_env
        from tools.delegate_tool_dispatch import _capture_origin
        import sys
        origin = _capture_origin()
        chat_type = get_session_env('HERMES_SESSION_CHAT_TYPE').strip().lower()
        platform = get_session_env('HERMES_SESSION_PLATFORM')
        turn_id = getattr(parent, '_current_turn_id', '') or ''
        bridge = sys.modules.get('colonymind_hands_front')
        if bridge is None:
            origin_ref = turn_id or parent.session_id
        else:
            # A missing ledger row has no owner message identity to invent.
            try:
                origin_ref = bridge.origin_ref(parent.session_id, turn_id)
            except Exception:
                origin_ref = None
        from .definitions import DIRECT_CHATS
        if (chat_type not in DIRECT_CHATS or (isinstance(origin_ref, dict) and (
                origin_ref.get('platform') != platform
                or origin_ref.get('chat') != get_session_env('HERMES_SESSION_CHAT_ID')
                or origin_ref.get('chat_type', chat_type) not in DIRECT_CHATS))):
            origin_ref = None
        return {'platform':platform,
                'chat_type':chat_type,
                'origin':'owner_turn', 'origin_ref':origin_ref,
                'turn':turn_id or parent.session_id,
                'wake_sid':origin[0], 'history_delivery':origin[4]}

    def push_supported(self, context):
        from gateway.session_context import async_delivery_supported
        from tools.delegate_tool_dispatch import _resolve_async_wake_sid
        # Hermes' resolver catches some errors permissively. Explicit capability
        # is required as well as its empty-string push target.
        return bool(async_delivery_supported()) and _resolve_async_wake_sid(
            context.get('wake_sid',''), context.get('history_delivery',False)) == ''

    def capture_context(self):
        import contextvars
        return contextvars.copy_context()

    def active_count(self):
        # Count slots, including finalizing/checking, rather than child agents.
        with self.async_module._records_lock:
            return len({r.get('slot_key') or r['delegation_id']
                        for r in self.async_module._records.values()
                        if r.get('status') in self.async_module._ACTIVE_STATES})

    def max_children(self):
        return self.policy._get_max_async_children()

    def native_policy(self):
        return dict(self.policy._load_config())

    def tool_entry(self, name):
        from tools.registry import registry
        return registry._tools.get(name)

    def ledger(self, ident):
        return self.async_module.get_durable_delegation(ident)

    def steer(self, subagent_id, text):
        return self.registry.steer_subagent(subagent_id, text)

    def stop(self, subagent_id):
        return self.registry.interrupt_subagent(subagent_id)


class Runtime:
    def __init__(self, *, config, client, spool_dir, native=None, owner_check=None, sessions=None, config_loader=None):
        from .budgets import HandRegistry
        from .spool import Spool
        from .effects import EffectJournal
        from .degraded import DegradedMode
        self.config, self.client = config, client
        self.native = native or NativeHermes()
        self.owner_check, self.sessions = owner_check, sessions
        self.config_loader = config_loader
        self.registry = HandRegistry()
        self.spool_dir = Path(spool_dir)
        self.spool = Spool(self.spool_dir)
        try:
            from gateway.status import get_process_start_time
            started = get_process_start_time(os.getpid())
        except (ImportError, AttributeError):
            started = time.time()
        self.epoch = {'owner_pid':os.getpid(), 'owner_started_at':started or time.time()}
        self.effects = EffectJournal(self.spool, self.epoch)
        self.degraded = DegradedMode(settings=lambda: self.config.get('degraded', {}))
        self.parents = weakref.WeakValueDictionary()
        self.contexts, self.call_ids, self.dispatch_answers = {}, {}, {}
        self._parent_seen = {}
        self._pending = threading.local()
        self._wrapped_async = {}
        self._wrapped_results = None
        self._schema_verdicts = {}
        self._load_context = None
        self._closed = threading.Event()
        self._resume_ready = threading.Event()
        self._send_lock = threading.RLock()
        self._reconciled = False
        self._thread = None
        self.gateway = None
        self.original_delegate = self.native.delegate_module.delegate_task
        self._restore()
        from .resumes import ApprovalResumes
        self.approval_resumes = ApprovalResumes(self)

    def _restore(self):
        """Rebuild local control records, without retrying any executor action."""
        from .budgets import HandState
        records = self.spool.records()
        for row in records:
            if row['path'] != '/v1/hands/plan':
                continue
            request = row['body']
            local_id = 'h-' + hashlib.sha256(row['key'].encode()).hexdigest()[:12]
            response = self.spool.response(row['key'])
            plans = (response or {}).get('plans',[])
            local_rows = [r for r in records if r['path'].startswith('/v1/hands/'+local_id+'/')]
            plan = (self._local_plan(request['tasks'][0],row['key']) if local_rows or not plans else dict(plans[0]))
            own = local_rows or [r for r in records if r['path'].startswith('/v1/hands/'+plan['hand_id']+'/')]
            if not own:
                continue  # Unconfirmed reservation only; the controller lease expires.
            plan.update(task=request['tasks'][0],parent_session=request['parent_session'])
            # New spools retain the creating turn's authority. Older spools
            # need a resolvable parent; the registry's existence grants none.
            plan['standing'] = (row.get('metadata') or {}).get('standing', 'unknown')
            if plan['standing'] == 'unknown' and self.sessions is not None:
                try:
                    plan['standing'] = self.sessions.standing(request['parent_session'])[0]
                except Exception:
                    pass
            epoch = next((r['body']['epoch'] for r in own if 'epoch' in r['body']),self.epoch)
            hand = HandState(hand_id=plan['hand_id'],task_id=plan['task_id'],kind=plan['kind'],plan=plan,
                             parent_session_id=request['parent_session'],epoch=epoch)
            hand.status = 'lost'
            self.registry.add(hand)
            for event_row in own:
                body=event_row['body']
                action=event_row['path'].rsplit('/',1)[-1]
                if action=='dispatched': hand.delegation_id=body['delegation_id']
                if action=='bind':
                    self.registry.bind(hand,body['child_session'],subagent_id=body['executor_ref'])
                    hand.plan['subagent_id']=body['executor_ref']
                if action=='hold':
                    hand.activity='awaiting_approval'
                    hand.blocked=True
                if action=='result':
                    hand.plan['_result_submitted'] = True
                    hand.status='done' if body['status'] in {'done','partial','needs_input','blocked'} else body['status']
                    hand.plan['intent_version']=body['intent_version']
                if action=='stop': hand.status='stopped'
                for event in body.get('events',[]):
                    hand.seq=max(hand.seq,event['seq'])
                    if event['type']=='dispatch_intent':
                        hand.plan['intent_version']=event['payload'].get('intent_version',1)
                        schema = (event_row.get('metadata') or {}).get('output_schema')
                        if isinstance(schema, dict):
                            hand.plan['output_schema'] = deepcopy(schema)
                        policy = (event_row.get('metadata') or {}).get('definition_policy')
                        if isinstance(policy, dict):
                            hand.plan['definition_policy'] = deepcopy(policy)
            if self.effects.open_effects(hand.hand_id): hand.status='unknown_effect'
            if local_rows and plans and plans[0].get('admit')=='start':
                self.registry.alias(plans[0]['hand_id'],hand)
            self.dispatch_answers[row['key']]={'hand_id':hand.hand_id,'name':plan['name'],
                'status':hand.status,'delegation_id':hand.delegation_id,'reason':'previous_attempt'}

    def authorized_owner(self, session):
        try:
            return bool(session and not self.registry.for_session(session)
                        and self.owner_check and self.owner_check(session))
        except Exception:
            return False

    def parent_standing(self, session):
        """Preserve the verified internal parent's lane, including mind and system work."""
        resolver = getattr(self.sessions, 'standing', None)
        if callable(resolver):
            try:
                standing = resolver(session)[0]
                return standing if standing in {'owner', 'mind', 'system'} else 'unknown'
            except Exception:
                return 'unknown'
        # An embedded host may supply the explicit owner-check seam without a
        # SessionMap. Its successful check is the standing source in that host.
        return 'owner' if self.authorized_owner(session) else 'unknown'

    def remember_parent(self, parent):
        sid = str(getattr(parent, 'session_id', '') or '')
        if not self.authorized_owner(sid) or getattr(parent, '_delegate_depth', 0):
            raise PermissionError('owner parent required')
        self.parents[sid] = parent
        context = self.native.context(parent)
        if hasattr(self.native, 'capture_context'):
            context['_execution_context'] = self.native.capture_context()
        self.contexts[sid] = context
        self._parent_seen[sid] = time.monotonic()
        self._resume_ready.set()
        return sid

    def emit(self, path, body, key, *, metadata=None):
        try:
            if metadata is None:
                self.spool.append(path, body, key)
            else:
                self.spool.append(path, body, key, metadata=metadata)
        except OSError:
            from .notices import journal_failure
            journal_failure(self)
            raise
        saved = self.spool.response(key)
        if saved is not None:
            return saved
        # All delivery is FIFO, including concurrent effects and reconnect. A
        # recovered plan must bind its server ids before its effects are sent.
        with self._send_lock:
            self.spool.replay(self._send)
            return self.spool.response(key)

    def _send(self, path, body, key):
        try:
            # A disabled faculty keeps a local journal and the launch controls.
            # Its pending plans can be adopted only after the faculty is enabled.
            if path == '/v1/hands/plan' and not self.config.get('faculty',True):
                return None
            from .degraded import remap_request
            path, body, key = remap_request(self.spool, path, body, key)
            result = self.client.post(path, body, key)
            if path == '/v1/hands/plan':
                local_id = 'h-' + hashlib.sha256(key.encode()).hexdigest()[:12]
                local = self.registry.get(local_id)
                plans = result.get('plans',[])
                if local is not None and plans and plans[0].get('admit') == 'start':
                    self.registry.alias(plans[0]['hand_id'],local)
                    local.plan['controller_hand_id'] = plans[0]['hand_id']
                    local.plan['controller_task_id'] = plans[0]['task_id']
            self.degraded.success()
            return result
        except Exception as exc:
            status = getattr(getattr(exc,'response',None),'status_code',None)
            if isinstance(status,int) and (400 <= status < 500 or status == 501):
                # A reachable controller's refusal is authoritative, not an
                # outage license. Retain its receipt and advance the FIFO.
                parts=path.strip('/').split('/')
                hand=self.registry.get(parts[2]) if len(parts)>3 else None
                if hand is not None:
                    hand.blocked=True
                    try:
                        self.native.stop(hand.subagent_id)
                    except Exception:
                        pass
                return {'error':'controller_refused','status_code':status,
                        **(refusal_detail(exc) if path == '/v1/hands/plan' else {})}
            refused = getattr(exc,'hand_id',None)
            if refused:
                hand = self.registry.get(refused)
                if hand is not None and not hand.blocked:
                    hand.blocked = True
                    try:
                        self.native.stop(hand.subagent_id)
                    finally:
                        from .reconcile import stop_recorded_processes
                        stop_recorded_processes(hand)
                        hand.status = 'unknown_effect' if self.effects.open_effects(refused) else 'refused'
            self.degraded.failure()
            return None

    def event(self, hand, event_type, payload=None, *, metadata=None):
        hand.seq += 1
        key = f'ev:{hand.hand_id}:{hand.seq}'
        return self.emit(f'/v1/hands/{hand.hand_id}/events', {'epoch':self.epoch, 'events':[
            {'seq':hand.seq, 'at':utcnow(), 'type':event_type, 'payload':payload or {},
             'source':'plugin', 'key':key}]}, key, **({'metadata':metadata} if metadata is not None else {}))

    def schema_evidence(self, hand, verdict):
        """Send bounded evidence and retain the full verdict in the private spool."""
        payload = deepcopy(verdict)
        errors = payload.get('schema_errors', [])
        encoded = json.dumps(verdict, sort_keys=True, separators=(',', ':'))
        if any(len(str(error)) > 280 for error in errors) or len(encoded.encode()) > 1800:
            payload['schema_error_count'] = len(errors)
            payload['schema_evidence_hash'] = hashlib.sha256(encoded.encode()).hexdigest()
            payload['schema_errors'] = [str(error)[:280] for error in errors[:3]]
            while len(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()) > 1800:
                summaries = payload['schema_errors']
                if len(summaries) > 1:
                    summaries.pop()
                else:
                    summaries[0] = summaries[0][:len(summaries[0]) // 2]
        self.event(hand, 'schema_validation', payload, metadata={'schema_verdict':verdict})

    def install(self):
        global _RUNTIME
        with DISPATCH_LOCK:
            if not self.available():
                raise RuntimeError('hands load abandoned')
            for name in ('dispatch_async_delegation', 'dispatch_async_delegation_batch'):
                original = getattr(self.native.async_module, name)
                if hasattr(original, '_hands_runtime'):
                    if original._hands_runtime is self:
                        continue
                    raise RuntimeError('async dispatcher already wrapped')
                @wraps(original)
                def locked(*args, _original=original, **kwargs):
                    with DISPATCH_LOCK:
                        if not self.available():
                            return {'status':'error','error':'hands runtime unavailable','retry':False}
                        return _original(*args, **kwargs)
                locked._hands_runtime = self
                self._wrapped_async[name] = (original, locked)
                setattr(self.native.async_module, name, locked)
            results = getattr(self.native, 'results_module', None)
            if results is not None and self._wrapped_results is None:
                original = results._fire_subagent_stop_hooks
                if hasattr(original, '_hands_runtime'):
                    raise RuntimeError('stop dispatcher already wrapped')
                @wraps(original)
                def stopped(entries, children, parent):
                    try:
                        self.capture_schema_verdicts(entries, children, parent)
                    except Exception:
                        logger.exception('Hands schema evidence unavailable')
                    return original(entries, children, parent)
                stopped._hands_runtime = self
                self._wrapped_results = (results, original, stopped)
                results._fire_subagent_stop_hooks = stopped
            wrap.install(self)
            _RUNTIME = self

    def available(self):
        return not self._closed.is_set() and not getattr(self._load_context, '_load_abandoned', False)

    def capture_schema_verdicts(self, entries, children, parent):
        """Keep native validation evidence before the public hook drops it."""
        parent_session = getattr(parent, 'session_id', None)
        parent_turn = getattr(parent, '_current_turn_id', '') or ''
        with DISPATCH_LOCK:
            for entry in entries:
                child = children.get(entry.get('task_index', -1))
                child_session = getattr(child, 'session_id', None)
                hand = self.registry.for_session(child_session)
                if (hand is None or (hand.parent_session_id != parent_session
                                    and self.parents.get(hand.parent_session_id) is not parent)
                        or hand.plan.get('_result_submitted')):
                    continue
                key = (child_session, parent_session, parent_turn)
                # A detached child's parent can begin another turn or rotate
                # its session before this callback. The child registry binds
                # the attempt; the current native fields bind this callback.
                verdict = ({
                        name: deepcopy(entry[name]) for name in ('schema_valid','schema_errors','schema_retries')
                        if name in entry} if isinstance(entry.get('schema_valid'), bool) else None)
                # A local approval response ends this attempt without producing
                # its declared output. Native schema failure cannot turn that
                # verified hold into a task failure. Native execution errors do.
                local_hold = (bool(hand.local_requests) and hand.blocked
                    and hand.activity == 'awaiting_approval' and not hand.stop_reason
                    and entry.get('exit_reason') == 'completed'
                    and entry.get('status') in {'completed', 'failed'} and not entry.get('failure_reason'))
                self._schema_verdicts[key] = (hand, verdict, local_hold)

    def _route(self, plan, task):
        route = dict(wrap.resolve(plan.get('role') or task.get('kind') or 'deep', int(plan.get('effort',2))))
        credentials = dict(route.get('credentials_cfg', route))
        contracts = self.config.get('contracts', {})
        needs = set(task.get('needs') or [])
        def valid(candidate):
            provider = candidate.get('provider')
            contract = contracts.get(provider)
            if not isinstance(contract, dict):
                return False
            required = {'text'} | ({'image'} if 'vision' in needs else set()) | ({'audio'} if 'audio' in needs else set())
            if not required <= set(contract.get('modalities', [])):
                return False
            if contract.get('tool_calls') not in (True, 'qualified') and not (
                isinstance(contract.get('tool_calls'), dict) and contract['tool_calls'].get('qualified') is True):
                return False
            # Conservative character bound needs no tokenizer and never understates
            # the byte length of a brief. Growth uses the contracted output cap.
            size = len(str(plan.get('brief','')).encode()) + int(contract.get('max_output_tokens',0))
            return int(contract.get('context_tokens',0)) >= size and bool(contract.get('version'))
        if not valid(credentials):
            raise ValueError('contract')
        credentials['fallback_providers'] = [dict(c) for c in credentials.get('fallback_providers', []) if valid(c)]
        plan['contract'] = {'provider':credentials['provider'], 'version':contracts[credentials['provider']]['version']}
        plan['route_contract'] = deepcopy(contracts[credentials['provider']])
        # Keep only routing identity in memory; credentials never enter the spool.
        plan['routes'] = [{**{k:c[k] for k in ('provider','model','base_url') if k in c},
                           'contract':{'provider':c['provider'],'version':contracts[c['provider']]['version']}}
                          for c in [credentials, *credentials['fallback_providers']]]
        return credentials

    def _local_plan(self, task, key):
        # Stable local ids are necessary when an outage spans retries. The local
        # plan is reconciled explicitly, never confused with a remote reservation.
        digest = hashlib.sha256(key.encode()).hexdigest()[:12]
        kind = task.get('kind') or 'deep'
        from .budgets import default_budget
        return {'task_id':'t-'+digest,'hand_id':'h-'+digest,'name':task.get('name') or kind+'-'+digest[:4],
                'kind':kind,'effort':task.get('effort') or 2,'role':kind, 'risk':task.get('risk') or 'normal',
                'budgets':default_budget(kind, task.get('effort') or 2),
                'brief':'You are doing one piece of your own work. Report receipts and uncertainty. '
                        'Do not send messages or change memory. Ask before floor effects.\n' + task['goal'],
                'admit':'start','lease_token':'local-'+digest,'local':True}

    def dispatch_tasks(self, tasks, *, parent=None, session_id=None, tool_call_id=None, native_options=None,
                       plan_key=None, context_override=None, approved_call=None):
        from .budgets import HandState
        if not self.available():
            return {'hands':[], 'error':'hands runtime unavailable', 'retry':False}
        if self.config_loader:
            self.config = self.config_loader()
        if parent is not None:
            session_id = self.remember_parent(parent)
        else:
            parent = self.parents.get(session_id)
        if parent is None or parent.session_id != session_id or not self.authorized_owner(session_id):
            return {'hands':[], 'error':'owner parent unavailable', 'retry':False}
        parent_standing = self.parent_standing(session_id)
        if parent_standing == 'unknown':
            return {'hands':[], 'error':'parent standing unavailable', 'retry':False}
        context = context_override if context_override is not None else self.contexts[session_id]
        if not isinstance(tasks, list) or not tasks:
            return {'hands':[], 'error':'tasks required'}
        call_id = tool_call_id or self.call_ids.get(session_id) or context.get('tool_call_id') or uuid.uuid4().hex
        answers = []
        for index, raw in enumerate(tasks):
            if not isinstance(raw, dict) or not isinstance(raw.get('goal'),str) or not raw['goal'].strip():
                answers.append({'status':'refused','reason':'goal'})
                continue
            if context.get('platform') not in self.config.get('platforms',[]):
                # W0-1: the wrapper refuses on a platform without a detached completion consumer, before planning;
                # a plan there could queue at the controller and later start for nobody.
                logger.info('hands refused on platform %s', context.get('platform') or 'unknown')
                answers.append({'status':'refused','reason':'platform'})
                continue
            task = {k:v for k,v in raw.items() if k in PLAN_FIELDS and v is not None}
            key = f'{plan_key}:{index}' if plan_key else f"plan:{session_id}:{context['turn']}:{call_id}:{index}"
            with DISPATCH_LOCK:
                previous = self.registry.get(task.get('continue'))
                resumed = self.approval_resumes.existing(previous) if previous and not plan_key else None
                if resumed:
                    answers.append(resumed)
                    continue
                if key in self.dispatch_answers:
                    answers.append(self.dispatch_answers[key])
                    continue
                planned = self.emit('/v1/hands/plan', {'parent_session':session_id,'turn':context['turn'],
                    'origin':context.get('origin','owner_turn'), 'origin_ref':context['origin_ref'], 'tasks':[task]}, key,
                    metadata={'standing': parent_standing})
                if planned is not None and planned.get('error'):
                    # The front corrects its own plan only if it learns why the controller refused it.
                    answers.append(refusal_answer(planned))
                    continue
                if planned is not None and planned.get('local_refusal'):
                    answers.append({'status':'refused','reason':planned['local_refusal']})
                    continue
                if planned is None:
                    if (task.get('hand') or task.get('continue', '').startswith('t-')
                            or (previous and previous.plan.get('definition_policy') is not None)):
                        self._never_ran(key, 'definition_unavailable')
                        answers.append({'status': 'refused', 'reason': 'definition_unavailable'})
                        continue
                    if plan_key:
                        # An automatic resume keeps its durable request pending. A timeout may have
                        # reserved the continuation already; replay must reuse that exact plan key.
                        answers.append({'status': 'queued', 'reason': 'controller_unavailable', 'retry': True})
                        continue
                    if not self.degraded.is_degraded and self.config.get('faculty',True):
                        self._never_ran(key,'controller_unavailable')
                        answers.append({'status':'refused','reason':'controller_unavailable'})
                        continue
                    plan = self._local_plan(task,key)
                    plan['plan_key'] = key
                else:
                    plan = dict(planned['plans'][0])
                    if plan.get('admit') in {'start', 'queued'}:
                        try:
                            row = self.client.get('/v1/hands/'+plan['hand_id'])
                            if row.get('parent_session') != session_id or row.get('task_id') != plan['task_id']:
                                raise ValueError('plan ownership mismatch')
                            version = row.get('intent_version')
                            if not isinstance(version,int) or isinstance(version,bool) or version < 1:
                                raise ValueError('intent version unavailable')
                            plan['intent_version'] = version
                            saved = row.get('decisions', {})
                            if saved.get('definition_snapshot') is not None:
                                from .definitions import compile_policy
                                plan['definition_snapshot'] = deepcopy(saved['definition_snapshot'])
                                plan['definition_policy'] = compile_policy(saved['definition_snapshot'])
                            elif task.get('hand'):
                                raise ValueError('definition policy unavailable')
                            if 'acceptance' in saved:
                                plan['acceptance'] = deepcopy(saved['acceptance'])
                            plan['workdir'] = saved.get('workdir') or task.get('workdir')
                        except Exception:
                            plan['admit'],plan['reason']='refuse','plan_metadata_unavailable'
                hand = self.registry.get(plan['hand_id'])
                if hand is None:
                    plan.update({'task':task,'parent_session':session_id,'origin_context':{k:v for k,v in context.items() if not k.startswith('_')},
                                 'priority':task.get('priority','owner'), 'standing':parent_standing})
                    plan.setdefault('intent_version',1)
                    if context.get('_execution_context') is not None:
                        plan['_execution_context'] = context['_execution_context']
                    if approved_call is not None:
                        # Set before the child exists: its first call may be the approved one.
                        plan['approved_call'] = deepcopy(approved_call)
                    hand = HandState(hand_id=plan['hand_id'],task_id=plan['task_id'],kind=plan['kind'],
                                     plan=plan,parent_session_id=session_id,epoch=self.epoch)
                    hand.status = 'reserved'
                    self.registry.add(hand)
                if plan.get('admit') == 'queued':
                    # Kept for promotion: the controller's sweeper reserves queued work later.
                    hand.plan['native_task'] = {k:v for k,v in raw.items() if k in NATIVE_FIELDS}
                    hand.plan['native_options'] = deepcopy(native_options or {})
                answer = self._admit(hand, plan, task, raw, parent, context, native_options)
                if task.get('kind') and plan.get('kind') and plan['kind'] != task['kind']:
                    answer = {**answer, 'kind': plan['kind'], 'kind_adjusted': {'from': task['kind'], 'to': plan['kind']}}
                self.dispatch_answers[key] = answer
                answers.append(answer)
        return {'hands':answers}

    def _admit(self, hand, plan, task, raw, parent, context, native_options):
        """The executor half of admission (3.5 step 2 on): platform, push delivery,
        slots and route, then one background dispatch or a release. Caller holds DISPATCH_LOCK."""
        reason = None
        if not self.available():
            reason = 'hands runtime unavailable'
        elif plan.get('admit') != 'start':
            reason = plan.get('reason') or plan.get('admit')
        elif context.get('platform') not in self.config.get('platforms',[]):
            reason = 'platform'
        else:
            execution = context.get('_execution_context')
            supported = execution.copy().run(self.native.push_supported,context) if execution else self.native.push_supported(context)
            if not supported:
                reason = 'no_async'
        # Queued hands ('planned') hold no slot; counting them refused every new hand while one waited.
        running = sum(h is not hand and h.status not in TERMINAL and h.status != 'planned'
                      and h.activity not in {'awaiting_input','awaiting_approval'} for h in self.registry.all())
        limit = int(self.config.get('max_running',1))
        if not self.config.get('faculty',True) or self.degraded.is_degraded:
            # Config default (an assumption): one running hand without the controller.
            limit = min(limit, int(self.config.get('degraded',{}).get('max_running',1)))
            if self.degraded.is_degraded and task.get('priority','owner') != 'owner':
                reason = reason or 'degraded_owner_only'
        if running >= limit or self.native.active_count() + 1 >= self.native.max_children():
            reason = reason or 'slots'
        if not self.spool.writable:
            reason = 'cannot record work safely'
        credentials = None
        if not reason:
            try:
                credentials = self._route(plan,task)
            except Exception:
                reason = 'contract'
        if reason:
            hand.status = 'planned' if plan.get('admit') == 'queued' else 'refused'
            if plan.get('local'):
                # A local outage plan that does not start has no controller reservation: nothing to release, and
                # its plan must not be replayed.
                self._never_ran(plan.get('plan_key'), reason)
            elif plan.get('lease_token'):
                self._release(hand, reason)
            answer = {'hand_id':hand.hand_id,'name':plan['name'],
                      'status':'queued' if plan.get('admit') == 'queued' else 'refused','reason':reason}
        else:
            answer = self._spawn(hand,raw,parent,credentials,native_options or {},context)
        return answer

    def promote(self):
        """Start queued hands that the controller's sweeper has reserved (3.5, 3.12 rule 6).

        The sweeper re-runs admission for queued work and holds a lease; nothing else
        dispatches it, so without this a queued hand only ever lapses. The owner's parent
        agent and captured context come from the turn that planned it; without them the
        reservation is released, never run inline or under another session.
        """
        for hand in self.registry.all():
            if hand.status != 'planned' or hand.plan.get('admit') != 'queued' or hand.plan.get('local'):
                continue
            try:
                row = self.client.get('/v1/hands/'+hand.hand_id)
            except Exception:
                continue
            state = row.get('state')
            if state in TERMINAL:
                hand.status = state
                continue
            token = ((row.get('decisions') or {}).get('admission') or {}).get('lease_token')
            if state != 'reserved' or not token or row.get('parent_session') != hand.parent_session_id:
                continue
            with DISPATCH_LOCK:
                if hand.status != 'planned':
                    continue
                if self.native.active_count() + 1 >= self.native.max_children():
                    # Hermes still holds a slot (a finishing hand's stop hook, or cron):
                    # keep the reservation and retry on the next pass, within its lease.
                    continue
                hand.status = 'reserved'
                plan = hand.plan
                plan.update(admit='start', lease_token=token, reason=None)
                version = row.get('intent_version')
                if isinstance(version,int) and not isinstance(version,bool) and version >= 1:
                    plan['intent_version'] = version
                parent = self.parents.get(hand.parent_session_id)
                context = self.contexts.get(hand.parent_session_id)
                if parent is None or context is None or not self.authorized_owner(hand.parent_session_id):
                    hand.status = 'refused'
                    self._release(hand, 'parent_unavailable')
                    continue
                raw = dict(plan.get('native_task') or {'goal':plan['task']['goal']})
                self._admit(hand, plan, plan['task'], raw, parent, context, plan.get('native_options', {}))

    def _never_ran(self, key, reason):
        """Settle a spooled plan for work that never started, so a replay never plans it at the controller (where
        it would take a slot from work that did run during the outage, and block that work's adoption)."""
        if key and self.spool.response(key) is None:
            self.spool.acknowledge(key, {'local_refusal': reason})

    def _release(self, hand, reason):
        return self.emit(f'/v1/hands/{hand.hand_id}/release', {'epoch':self.epoch,
            'delegation_id':hand.delegation_id or 'not-dispatched', 'lease_token':hand.plan['lease_token'],
            'reason':reason}, f'release:{hand.hand_id}:{self.epoch_key}')

    @property
    def epoch_key(self):
        return f"{self.epoch['owner_pid']}:{self.epoch['owner_started_at']}"

    def _spawn(self, hand, raw, parent, credentials, options, context):
        if parent.session_id != hand.parent_session_id:
            hand.status = 'refused'
            self._release(hand, 'waiting_for_main_session')
            return {'hand_id': hand.hand_id, 'status': 'refused', 'reason': 'waiting_for_main_session'}
        token = hand.plan.get('task', {}).get('queue_claim')
        if token:
            try:
                from .definitions import pickup_parent
                if not pickup_parent(self, hand.parent_session_id, context, parent):
                    hand.status = 'refused'
                    self._release(hand, 'waiting_for_main_session')
                    return {'hand_id': hand.hand_id, 'status': 'refused', 'reason': 'waiting_for_main_session'}
                self.client.post('/v1/hands/definitions/queue/bind', {
                    'task_id': hand.task_id, 'hand_id': hand.hand_id, 'claim_token': token},
                    'queue-bind:' + token)
            except Exception as exc:
                code = refusal_detail(exc).get('detail', 'definition_claim_invalid')
                hand.status = 'refused'
                self._release(hand, code)
                return {'hand_id': hand.hand_id, 'status': 'refused', 'reason': code}
        native_task = {k:v for k,v in raw.items() if k in NATIVE_FIELDS}
        native_task['context'] = hand.plan['brief']
        # The controller's task wire excludes native-only fields. Retain the
        # effective schema locally so a missing verdict cannot become success.
        hand.plan['output_schema'] = deepcopy(native_task.get('output_schema', options.get('output_schema')))
        hand.plan['parent_turn_id'] = getattr(parent, '_current_turn_id', '') or ''
        self._pending.hand = hand
        hand.plan['_bind_pending'] = []
        # The dispatch intent is durable before any child exists. It contains no
        # credentials, brief or endpoint, and lets restart identify a lost attempt.
        schema = hand.plan.get('output_schema')
        metadata = {'output_schema': schema} if isinstance(schema, dict) else {}
        if hand.plan.get('definition_policy') is not None:
            metadata['definition_policy'] = deepcopy(hand.plan['definition_policy'])
        self.event(hand,'dispatch_intent',{'task_id':hand.task_id,'intent_version':hand.plan['intent_version']},
                   **({'metadata': metadata} if metadata else {}))
        try:
            launch_parent = LaunchParent(parent, hand.parent_session_id) if token else parent
            refused = object()
            def spawn():
                # Bind and intent persistence can block while authority changes.
                # Nothing may perform I/O between this check and native entry.
                if token and not pickup_parent(self, hand.parent_session_id, context, parent):
                    return refused
                return self.original_delegate(tasks=[native_task], parent_agent=launch_parent, background=True,
                                              credentials_cfg=credentials, **options)
            execution = context.get('_execution_context')
            value = execution.copy().run(spawn) if execution else spawn()
            if value is refused:
                hand.status = 'refused'
                # The consumed queue binding stays consumed; release the attempt
                # visibly for owner controls, never automatically retry it.
                self._release(hand, 'waiting_for_main_session')
                return {'hand_id': hand.hand_id, 'status': 'refused', 'reason': 'waiting_for_main_session'}
            result = json.loads(value) if isinstance(value,str) else value
        except Exception:
            hand.status = 'unknown_effect' if self.effects.open_effects(hand.hand_id) else 'lost'
            self.event(hand,'dispatch_uncertain')
            return {'hand_id':hand.hand_id,'name':hand.plan['name'],'status':hand.status,'reason':'dispatch_uncertain'}
        finally:
            self._pending.hand = None
        if result.get('status') != 'dispatched':
            hand.status = 'failed'
            inline = 'results' in result
            self.event(hand,'ran_inline' if inline else 'dispatch_refused',
                       {'result_hash':hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()})
            self._release(hand,'ran_inline' if inline else 'dispatch_refused')
            return {'hand_id':hand.hand_id,'name':hand.plan['name'],'status':'refused',
                    'reason':'ran_inline' if inline else 'dispatch_refused','result':result}
        hand.delegation_id = result['delegation_id']
        hand.status = 'dispatched'
        self.emit(f'/v1/hands/{hand.hand_id}/dispatched', {'epoch':self.epoch,
            'delegation_id':hand.delegation_id,'lease_token':hand.plan['lease_token']},
            f'disp:{hand.hand_id}:{self.epoch_key}')
        for binding in hand.plan.pop('_bind_pending',[]):
            self._emit_bind(hand,binding)
        return {'hand_id':hand.hand_id,'name':hand.plan['name'],**result}

    def subagent_start(self, *, child_session_id='', child_subagent_id='', parent_session_id='', **kwargs):
        try:
            hand = getattr(self._pending,'hand',None)
            if hand is None or hand.parent_session_id != parent_session_id:
                return
            self.registry.bind(hand,child_session_id,subagent_id=child_subagent_id,task_id=child_subagent_id)
            if self.sessions is not None:
                self.sessions.inherit(child_session_id, parent_session_id,
                                      standing=hand.plan.get('standing', 'unknown'), platform='subagent')
            hand.plan['subagent_id'] = child_subagent_id
            binding = {'executor_ref':child_subagent_id,'child_session':child_session_id,'epoch':self.epoch}
            if hand.status == 'reserved':
                hand.plan['_bind_pending'].append(binding)
            else:
                self._emit_bind(hand,binding)
        except Exception:
            if getattr(self._pending,'hand',None) is not None:
                self._pending.hand.blocked = True

    def _emit_bind(self, hand, binding):
        hand.status = 'running'
        self.emit(f'/v1/hands/{hand.hand_id}/bind',binding,
                  f"bind:{hand.hand_id}:{binding['child_session']}:{self.epoch_key}")

    def _checks(self, hand):
        from .checks import CheckRunner, CHECK_TIMEOUT
        acceptance = hand.plan.get('acceptance')
        workdir = hand.plan.get('workdir') or hand.plan.get('task', {}).get('workdir')
        try:
            runner = CheckRunner(timeout=self.config.get('checks', {}).get('timeout_seconds', CHECK_TIMEOUT))
            if acceptance is None:
                task = self.client.get('/v1/hands/tasks/' + hand.task_id)
                acceptance = task['intents'][str(hand.plan['intent_version'])]['acceptance']
                hand.plan['acceptance'] = acceptance
            receipts = runner.run(kind=hand.kind, acceptance=acceptance, workdir=workdir,
                artifacts=hand.plan.get('observed_artifacts', ()), fetched=hand.plan.get('fetched_texts', {}),
                quotes=hand.plan.get('proposed_quotes', ()))
            return runner.finalize(receipts, workdir)
        except Exception as error:
            return [CheckRunner.failure(hand.kind, acceptance or {}, error)]

    def probe_unknown(self, hand_id):
        """The controller must classify the effect before the executor observes it."""
        from .checks import CheckRunner
        try:
            hand = self.client.get('/v1/hands/' + hand_id)
            if hand.get('state') != 'unknown_effect' or hand.get('decisions', {}).get('probe'):
                return
            task = self.client.get('/v1/hands/tasks/' + hand['task_id'])
            saved = task['intents'][str(hand['intent_version'])]['acceptance']
            local = self.registry.get(hand_id)
            if local:
                local.status = 'unknown_effect'
            receipt = CheckRunner().probe(kind=hand['kind'], acceptance=saved,
                                           workdir=hand['decisions'].get('workdir'))
            self.emit('/v1/hands/' + hand_id + '/probe', {'epoch': hand['epoch'], 'receipt': receipt},
                      'probe:' + hand_id)
        except Exception:
            logger.exception('Hands reconciliation probe unavailable')

    def subagent_stop(self, *, child_session_id='', child_status='', child_summary='',
                      parent_session_id='', parent_turn_id='', **kwargs):
        try:
            with DISPATCH_LOCK:
                hand = self.registry.for_session(child_session_id)
                if hand is None or hand.status == 'checking' or hand.plan.get('_result_submitted') or hand.status in TERMINAL - {'stopped'}:
                    return
                key = (child_session_id, parent_session_id, parent_turn_id)
                captured = self._schema_verdicts.pop(key, None)
                if captured is not None and captured[0] is not hand:
                    return  # The child identity was rebound after capture.
                if captured is None and parent_session_id and parent_session_id != hand.parent_session_id:
                    return
                if captured is None and parent_turn_id and hand.plan.get('parent_turn_id', parent_turn_id) != parent_turn_id:
                    return
                verdict = captured[1] if captured else None
                local_hold = bool(captured and captured[2] and hand.status not in TERMINAL
                                  and hand.blocked and hand.activity == 'awaiting_approval'
                                  and not hand.stop_reason)
                failures = {'failed','error'}
                if getattr(self.native, 'inactivity_timeout', False):
                    failures.update({'timeout','stalled'})
                native_failure = child_status in failures | {'interrupted'}
                if (isinstance(hand.plan.get('output_schema'), dict) and verdict is None
                        and not native_failure and not local_hold):
                    # No contract evidence means unresolved, never permission to
                    # repeat an external effect or declare the attempt done.
                    hand.status = 'unknown_effect' if self.effects.open_effects(hand.hand_id) else 'lost'
                    self.event(hand, 'schema_unresolved', {'child_session':child_session_id})
                    return
                hand.status = 'checking'
            # Hermes retains the slot through the hook. Other dispatch and owner
            # work must still acquire DISPATCH_LOCK while checks execute.
            self.event(hand, 'checking')
            receipts = self._checks(hand)
            if verdict is not None:
                self.schema_evidence(hand, verdict)
            from .budgets import REPORT_STOPS
            status_value = ('blocked' if local_hold else
                'stopped' if child_status == 'interrupted' else
                'failed' if child_status in failures or (verdict or {}).get('schema_valid') is False else
                'needs_input' if hand.activity == 'awaiting_input' else (
                'blocked' if hand.activity == 'awaiting_approval' else
                # A budget stop's report is the attempt's partial result, never a finished task.
                'partial' if hand.stop_reason in REPORT_STOPS else
                # An executor that ended on the local reply after a runtime stop was stopped.
                ('failed' if hand.stop_reason == 'stale' else 'stopped')
                if hand.stop_reason and hand.local_requests else
                'done'))
            self.event(hand,'usage',{'tools':hand.tools,'web_searches':hand.searches,'tokens':hand.tokens,
                                     'raw_tokens':hand.raw_tokens})
            outcome = self.emit(f'/v1/hands/{hand.hand_id}/result', {'epoch':self.epoch,
                'intent_version':hand.plan.get('intent_version',1),'status':status_value,
                'summary_ref':hand.child_session_id or child_session_id,
                'summary_text':str(child_summary or ''),
                'summary_hash':hashlib.sha256(str(child_summary or '').encode()).hexdigest(),
                'tool_digest':hand.plan.get('tool_digest',[]),'receipts':receipts,
                'families_used':sorted(hand.families_used),
                'contracts_used':hand.plan.get('contracts_used',[])},
                f'res:{hand.hand_id}:{self.epoch_key}')
            hand.plan['_result_submitted'] = True
            with DISPATCH_LOCK:
                self._schema_verdicts = {key:value for key,value in self._schema_verdicts.items()
                                         if value[0] is not hand}
            hand.status = (outcome or {}).get('outcome') or (status_value if status_value in TERMINAL else 'done')
            if self.effects.open_effects(hand.hand_id):
                hand.status = 'unknown_effect'
        except Exception:
            logger.exception('Hands stop hook could not record its result')

    def _owned(self, hand, calling_session):
        session = calling_session or hand.parent_session_id
        if not self.authorized_owner(session) or session != hand.parent_session_id:
            raise PermissionError('hand belongs to another owner session')

    def steer(self, hand, text, calling_session=None):
        """A steer a session asks for: only the hand's owner session may give it."""
        self._owned(hand,calling_session)
        return self.native.steer(hand.plan.get('subagent_id',''),text)

    def system_steer(self, hand, text):
        """The runtime's own nudges (wall warning, status check, no new information).

        No session asked for them, so there is no owner session to check. They run on
        delegate and maintenance threads, where the owner's session may not resolve.
        """
        return self.native.steer(hand.plan.get('subagent_id',''),text)

    def stop(self, hand, reason, calling_session=None):
        """A stop a session asks for: only the hand's owner session may make it."""
        self._owned(hand,calling_session)
        return self._stop(hand, reason, calling_session)

    def system_stop(self, hand, reason, *, interrupt=True):
        """A stop the runtime itself decides: budgets, clocks, accounting and journal faults.

        No session asked for it, so there is no owner session to check; the delegate or
        maintenance thread that notices the cause often has no owner session current. Every
        other effect of a stop still applies: the journal entry, process cleanup and status.
        interrupt=False leaves the executor running for its bounded final report
        (budgets.REPORT_STOPS); calling again with the same reason interrupts it, and the
        journal entry repeats with the same key and body.
        """
        return self._stop(hand, reason, None, interrupt=interrupt)

    def _stop(self, hand, reason, calling_session, *, interrupt=True):
        stopped = False
        # Journal before changing the executor; stop still reaches it if storage fails.
        try:
            if calling_session or reason == 'parent stopped':
                self.approval_resumes.cancel(hand)
            self.emit(f'/v1/hands/{hand.hand_id}/stop',{'reason':reason,'epoch':self.epoch},
                      f'stop:{hand.hand_id}:{self.epoch_key}')
        finally:
            try:
                if interrupt:
                    stopped = self.native.stop(hand.plan.get('subagent_id',''))
            finally:
                from .reconcile import stop_recorded_processes
                stop_recorded_processes(hand)
                hand.blocked = True
                hand.status = ('unknown_effect' if self.effects.open_effects(hand.hand_id) else
                               'failed' if reason == 'stale' else 'stopped')
        return stopped

    def continue_hand(self, hand, answer='', reconciled=False):
        self._owned(hand,hand.parent_session_id)
        with DISPATCH_LOCK:
            resumed = self.approval_resumes.existing(hand)
            if resumed:
                return resumed
        if (hand.status == 'unknown_effect' or self.effects.open_effects(hand.hand_id)) and not reconciled:
            return {'error':'reconciliation required'}
        task = dict(hand.plan['task'])
        task.update({'continue':hand.hand_id,'reconciled':reconciled})
        if isinstance(hand.plan.get('output_schema'), dict):
            task['output_schema'] = deepcopy(hand.plan['output_schema'])
        if answer:
            task['context'] = task.get('context','') + '\nAnswer from the main context: ' + answer
        return self.dispatch_tasks([task],session_id=hand.parent_session_id,
                                   tool_call_id='continue-'+uuid.uuid4().hex)

    def resume_approvals(self, session_id, grants, *, approved=()):
        with DISPATCH_LOCK:
            # Polling may already have started and consumed this grant before the answer response.
            # Its durable queue receipt still supplies the answer turn's no-duplicate instruction.
            known = [row['body'] for row in self.approval_resumes.spool.records()
                     if row['path'] == 'approval-resume' and row['body']['ask_id'] in approved
                     and row['body']['parent_session'] == session_id]
            return self.approval_resumes.enqueue(session_id, [*grants, *known])

    def pre_tool_call(self, tool_name='', session_id='', tool_call_id='', **kwargs):
        try:
            if tool_name == 'delegate_task' and self.authorized_owner(session_id):
                with DISPATCH_LOCK:
                    self.call_ids[session_id] = tool_call_id
        except Exception:
            logger.exception('Hands call identity observer unavailable')

    def refresh_parent(self, gateway=None, **kwargs):
        try:
            self.gateway = weakref.ref(gateway) if gateway is not None else None
            for agent in (getattr(gateway,'_running_agents',{}) or {}).values():
                if hasattr(agent,'session_id') and self.authorized_owner(agent.session_id):
                    # This hook precedes route ContextVar binding. Refresh only
                    # object identity here, never another session's origin.
                    self.parents[agent.session_id]=agent
        except Exception:
            pass

    def current_parent(self, session_id='', **kwargs):
        try:
            if not self.authorized_owner(session_id):
                return
            gateway=self.gateway() if self.gateway else None
            parent=self.parents.get(session_id)
            if parent is None:
                parent=next((a for a in getattr(gateway,'_running_agents',{}).values()
                             if getattr(a,'session_id',None)==session_id),None)
            if parent is not None:
                self.remember_parent(parent)
        except Exception:
            pass

    def agent_loop_stopped(self, session_id='', **kwargs):
        for hand in self.registry.all():
            if hand.parent_session_id == session_id and hand.status not in TERMINAL:
                try:
                    self.stop(hand,'parent stopped')
                except Exception:
                    hand.blocked=True

    def maintain(self):
        if not self.available():
            return
        from .definitions import pump
        pump(self)
        try:
            if self.config_loader:
                self.config = self.config_loader()
            if not self.spool.writable:
                from .notices import journal_failure
                journal_failure(self)
            with self._send_lock:
                self.spool.replay(self._send)
                if not self._reconciled and not self.spool.pending():
                    from .reconcile import reconcile
                    reconcile(self, self.native.ledger)
                    self._reconciled = True
            if self.config.get('faculty',True) and not self.degraded.is_degraded:
                self.promote()
            from . import budgets, notices
            for hand in self.registry.all():
                # hands.dispatch_start_seconds (default 120, an assumption from 3.5).
                start_limit = float(self.config.get('dispatch_start_seconds',120))
                if hand.status == 'dispatched' and time.monotonic() - hand.started_at >= start_limit:
                    durable = self.native.ledger(hand.delegation_id)
                    if not durable or durable.get('state',durable.get('status')) in {'unknown','failed','interrupted'}:
                        hand.status = 'unknown_effect' if self.effects.open_effects(hand.hand_id) else 'lost'
                        self.event(hand,'missing_start',{'outcome':hand.status})
            tick=getattr(budgets,'tick',None)
            if callable(tick): tick(self)
            pump=getattr(notices,'pump',None)
            if callable(pump): pump(self)
            self.approval_resumes.pump()
        except Exception:
            pass

    def start(self):
        if self._thread is not None:
            return
        def run():
            self.maintain()
            # hands.spool.replay_seconds (default 10, an assumption from 3.4).
            while not self._closed.is_set():
                self._resume_ready.wait(min(2.0, float(self.config.get('spool',{}).get('replay_seconds',10))))
                self._resume_ready.clear()
                if self._closed.is_set():
                    break
                self.maintain()
        self._thread=threading.Thread(target=run,name='hands-maintenance',daemon=True)
        self._thread.start()

    def close(self):
        global _RUNTIME
        self._closed.set()
        self._resume_ready.set()
        if self._thread:
            self._thread.join(timeout=1)
        with DISPATCH_LOCK:
            if self._wrapped_results is not None:
                results, original, installed = self._wrapped_results
                if results._fire_subagent_stop_hooks is installed:
                    results._fire_subagent_stop_hooks = original
            self._schema_verdicts.clear()
            for name,(original,installed) in self._wrapped_async.items():
                if getattr(self.native.async_module,name) is installed:
                    setattr(self.native.async_module,name,original)
            if getattr(self.native.delegate_module.delegate_task,'_hands_runtime',None) is self:
                self.native.delegate_module.delegate_task=self.original_delegate
            try:
                lookup=getattr(self.native,'tool_entry',None)
                entry=lookup('delegate_task') if callable(lookup) else None
                overrides=getattr(entry,'dynamic_schema_overrides',None)
                if getattr(overrides,'_hands_runtime',None) is self:
                    entry.dynamic_schema_overrides=overrides._hands_native
            except ImportError:
                pass
        self.spool.close()
        if _RUNTIME is self: _RUNTIME=None


def refuse_delegation(owner=None, context=None):
    """Replace native delegation with a refusal: no hands controls, no delegation."""
    try:
        module = owner.native.delegate_module if owner is not None else importlib.import_module('tools.delegate_tool')
        runtime=getattr(module.delegate_task,'_hands_runtime',None)
        if owner is not None and runtime is not None and runtime is not owner:
            return  # A late abandoned loader cannot replace its successor.
        if context is not None and runtime is not None and getattr(runtime, '_load_context', None) is not context:
            return
        if runtime is not None:
            runtime._closed.set()
        current = module.delegate_task
        def refused(*args, **kwargs):
            return json.dumps({'error':'hands runtime unavailable','retry':False})
        refused._hands_native = runtime.original_delegate if runtime else getattr(current, '_hands_native', current)
        module.delegate_task = refused
    except ImportError:
        pass


def get_runtime(ctx=None):
    return getattr(ctx,'hands_runtime',None) or _RUNTIME


def register(ctx: Any) -> None:
    global _RUNTIME
    runtime=get_runtime(ctx)
    if runtime is not None and runtime._load_context is not ctx and not runtime.available():
        runtime.close()
        runtime = None
    if runtime is None:
        try:
            from colonymind_hermes.client import load_settings, read_yaml
            import colonymind_hermes
            settings=load_settings()
            config=read_yaml(settings.home/'colonymind.yaml')
            hands=dict(config.get('hands') or {})
            hands['faculty']=config.get('mind',{}).get('faculties',{}).get('hands',True)
            def config_loader():
                latest=read_yaml(settings.home/'colonymind.yaml')
                result=dict(latest.get('hands') or {})
                result['faculty']=latest.get('mind',{}).get('faculties',{}).get('hands',True)
                return result
            def owner_check(sid):
                body=getattr(colonymind_hermes,'_BODY',None)
                return body is not None and body.sessions.standing(sid)[0] in {'owner', 'mind', 'system'}
            body=getattr(colonymind_hermes,'_BODY',None)
            runtime=Runtime(config=hands,client=ControllerClient(settings),spool_dir=settings.home/'hands-spool',
                            owner_check=owner_check,sessions=getattr(body,'sessions',None),config_loader=config_loader)
        except Exception:
            # Without runtime controls delegation must still be refused. Existing
            # conversation tools are unaffected. A load warning explains the fault.
            refuse_delegation(context=ctx)
            from .notices import journal_failure
            journal_failure()
            return
    ctx.hands_runtime = runtime
    _RUNTIME = runtime
    runtime._load_context = ctx
    if callable(getattr(ctx,'on_unload',None)): ctx.on_unload(runtime.close)
    ctx.register_hook('pre_tool_call',runtime.pre_tool_call)
    ctx.register_hook('pre_gateway_dispatch',runtime.refresh_parent)
    ctx.register_hook('pre_llm_call',runtime.current_parent)
    ctx.register_hook('subagent_start',runtime.subagent_start)
    ctx.register_hook('subagent_stop',runtime.subagent_stop)
    ctx.register_hook('agent_loop_stopped',runtime.agent_loop_stopped)


def activate(ctx):
    """Expose delegation only after all guards and unload callbacks exist."""
    runtime = get_runtime(ctx)
    if runtime is None:
        return
    if getattr(ctx, '_load_abandoned', False):
        runtime.close()
        refuse_delegation(owner=runtime)
        return
    runtime.install()
    runtime.start()
