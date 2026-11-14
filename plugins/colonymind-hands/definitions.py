"""Definition policy compilation at admission; no sidecar or agent runner imports."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime


def compile_policy(snapshot):
    """Resolve known toolsets once. Unknown names remain exact tool names.

    The existing kind policy still runs on every call. A saved definition can
    only remove capabilities; empty lists authorize no tools/connectors.
    """
    tools = set()
    try:
        from toolsets import get_toolset, resolve_toolset
    except ImportError:
        get_toolset = lambda name: None
    for name in snapshot['tools']:
        if get_toolset(name) is not None:
            tools.update(resolve_toolset(name))
        else:
            tools.add(name)
    policy = {'tools': sorted(tools), 'connectors': list(snapshot['connectors'])}
    if 'source_reads' in snapshot:
        policy['source_reads'] = deepcopy(snapshot['source_reads'])
    return policy


def check_source_read(reads, tool, args):
    """Only the exact frozen reads; shell, wrappers and every write fail closed."""
    for read in reads:
        server, operation = read['server'], read['tool']
        names = {f'connectors__{server}__{operation}', f'connector.{server}.{operation}'}
        if tool.startswith('mcp_'):
            # Use Hermes' own naming and registry identity, including server
            # underscores and clamped names. Never guess a server from a prefix.
            try:
                from tools.mcp_tool_schema import mcp_prefixed_tool_name
                if connector_name(tool) == server:
                    names.add(mcp_prefixed_tool_name(server, operation))
            except ImportError:
                pass
        if tool not in names:
            continue
        expected = {'account': read['account']}
        if operation == 'gmail_get':
            expected.update(message_id=read['provider_id'])
            if 'format' in args:
                expected['format'] = 'full'
        elif operation == 'calendar_events':
            expected.update({key: read[key] for key in ('calendar_id', 'time_min', 'time_max')})
            expected.update(single_events=True, page_size=16)
        else:
            continue
        actual = dict(args)
        if operation == 'calendar_events':
            try:
                for key in ('time_min', 'time_max'):
                    actual[key] = datetime.fromisoformat(actual[key].replace('Z', '+00:00'))
                    expected[key] = datetime.fromisoformat(expected[key].replace('Z', '+00:00'))
            except (KeyError, AttributeError, TypeError, ValueError):
                continue
        if actual == expected:
            return None
    return 'tool or arguments are outside the frozen source read scope; external effects require separate owner review'


def connector_name(tool):
    """Prefer registry identity over parsing MCP names (server names contain `_`)."""
    try:
        from tools.registry import registry
        entry = registry.get_entry(tool)
        if entry and entry.toolset.startswith('mcp-'):
            return entry.toolset[4:]
    except ImportError:
        pass
    if tool.startswith('mcp_'):
        # Unresolved MCP identities fail closed; guessing a server from a prefix
        # could authorize a different server with an overlapping name.
        return ''
    if tool.startswith('connectors__'):
        parts = tool.split('__', 2)
        return parts[1] if len(parts) == 3 and parts[1] and parts[2] else ''
    if tool.startswith('connector.'):
        return tool.split('.', 2)[1]
    return None


def check_policy(hand, tool, args):
    policy = hand.plan.get('definition_policy')
    if policy is None:
        return ('definition policy unavailable' if hand.plan.get('definition_snapshot')
                or hand.plan.get('task', {}).get('hand') else None)
    if tool not in policy['tools']:
        return f'{tool} is outside the Hand definition allow-list'
    connector = connector_name(tool)
    if connector is not None and (not connector or connector not in policy['connectors']):
        return 'connector is outside the Hand definition allow-list'
    # Dispatch bridges can execute names hidden inside an allowed wrapper.
    # Refuse rather than let a wrapper widen this task's compiled allowlist.
    if tool in {'tool_call', 'execute_code', 'connectors', 'connector_batch', 'connectors__execute'}:
        return 'dispatch bridges are unavailable for a restricted Hand'
    if 'source_reads' in policy:
        return check_source_read(policy['source_reads'], tool, args)
    return None


def list_hands(runtime):
    result = runtime.client.get('/v1/hands/definitions', params={'enabled_only': True})
    return {'hands': [{key: row[key] for key in ('id', 'name', 'role')}
                      for row in result['definitions'] if row['enabled'] and not row['archived_at']]}


DIRECT_CHATS = {'dm', 'direct', 'private'}


def pickup_parent(runtime, sid, context=None, parent=None):
    """Current owner-only authority and the exact captured private destination.

    General internal authorization also admits mind/system work. Pickup must
    instead check the live sender, without SessionInfo.owner's cached answer.
    """
    parent = parent if parent is not None else runtime.parents.get(sid)
    captured = runtime.contexts.get(sid, {})
    context = captured if context is None else context
    ref = context.get('origin_ref')
    if (parent is None or parent.session_id != sid or getattr(parent, '_delegate_depth', 0)
            or not runtime.authorized_owner(sid) or runtime.parent_standing(sid) != 'owner'
            or captured.get('origin') != 'owner_turn' or not isinstance(ref, dict)
            or not ref.get('chat') or not ref.get('message_id')
            or ref != captured.get('origin_ref') or ref.get('platform') != context.get('platform')
            or context.get('chat_type') not in DIRECT_CHATS
            or ref.get('chat_type', 'dm') not in DIRECT_CHATS):
        return False
    sessions = runtime.sessions
    try:
        if callable(getattr(sessions, 'get', None)):
            info = sessions.get(sid)
            # Resolve live owner evidence before inspecting mutable destination
            # and standing again; contact resolution may itself block.
            owner = info and sessions.sender_is_owner(info.platform, info.sender_id, timeout=2.0)
            chat, chat_type = sessions.chat(sid)
            return bool(owner and info.sender_id and info.platform == ref['platform']
                and info.inherited_standing not in {'mind', 'system', 'guest'}
                and chat == ref['chat'] and chat_type in DIRECT_CHATS
                and parent.session_id == sid)
        # Embedded hosts must capture explicit direct-chat evidence too.
        return parent.session_id == sid
    except Exception:
        return False


def owner_session(runtime):
    # Use captured ingress context and a live weak parent; never reconstruct a
    # parent, a destination or an owner turn from a saved database record.
    for sid, context in sorted(list(runtime.contexts.items()),
                               key=lambda row: runtime._parent_seen.get(row[0], 0), reverse=True):
        parent = runtime.parents.get(sid)
        if parent is not None and parent.session_id != sid:
            # Compression rotates the object in place. Never authorize its old
            # weak-map alias; capture the new identity through normal ingress.
            runtime.parents.pop(sid, None)
            runtime.contexts.pop(sid, None)
            runtime._parent_seen.pop(sid, None)
            try:
                sid = runtime.remember_parent(parent)
            except Exception:
                continue
        if pickup_parent(runtime, sid):
            return sid
    return None


def dispatch_claim(runtime, request, session_id, *, kind=None):
    import uuid
    task = {'goal': request['brief'], 'continue': request['task_id'],
            'queue_claim': request['claim_token']}
    if kind is not None:
        task['kind'] = kind
    context = {**runtime.contexts[session_id], 'origin': 'own'}
    try:
        answer = runtime.dispatch_tasks([task], session_id=session_id,
            plan_key='definition-claim:' + request['claim_token'], context_override=context)
        row = next(iter(answer.get('hands', [])), {})
        if row.get('status') in {'dispatched', 'running', 'lost', 'unknown_effect'}:
            return answer
        code = row.get('detail') or row.get('reason') or 'waiting_for_main_session'
    except Exception:
        # An ambiguous native outcome must never be dispatched again. A consumed
        # binding rejects release; an unbound claim is safe to retry.
        code = 'dispatch_unavailable'
        answer = {'hands': [], 'error': code}
    try:
        runtime.client.post('/v1/hands/definitions/queue/release', {
            'task_id': request['task_id'], 'claim_token': request['claim_token'], 'code': code}, uuid.uuid4().hex)
    except Exception:
        pass  # Lease recovery handles an unavailable sidecar.
    return answer


def pump(runtime):
    import uuid
    try:
        sid = owner_session(runtime)
        for _ in range(max(1, int(runtime.config.get('max_running', 1)))):
            response = runtime.client.post('/v1/hands/definitions/queue/claim',
                                           {'parent_session': sid}, uuid.uuid4().hex)
            if not response['tasks']:
                break
            for request in response['tasks']:
                dispatch_claim(runtime, request, sid)
    except Exception:
        pass  # The existing maintenance loop retries; no degraded definition dispatch.


def dispatch_request(runtime, task_id, session_id, *, kind=None):
    import uuid
    if not pickup_parent(runtime, session_id):
        return {'hands': [], 'error': 'owner parent unavailable', 'retry': False}
    response = runtime.client.post('/v1/hands/definitions/queue/claim',
        {'parent_session': session_id, 'task_id': task_id}, uuid.uuid4().hex)
    if not response['tasks']:
        raise ValueError('Unknown, unavailable or already claimed Hand task request')
    return dispatch_claim(runtime, response['tasks'][0], session_id, kind=kind)
