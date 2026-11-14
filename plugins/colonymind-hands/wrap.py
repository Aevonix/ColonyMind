"""One delegation wrapper and a host-independent role resolver."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from functools import wraps
from inspect import signature
import json
from typing import Any, Protocol

# Her task fields (design 6.2), shown on delegate_task's task items so the front can set them: Hermes' own schema
# advertises only goal, context, output_schema, images and group, so without these every hand was a deep hand
# (read-only tools) with no acceptance check fixed before work. Values follow the controller's wire contract.
TASK_FIELDS = {
    'hand': {'type': 'string', 'description': 'Persistent Hand definition id or exact name. '
             'Use colonymind_hands action list_hands to see enabled Hands.'},
    'kind': {'type': 'string', 'enum': ['coding', 'deep', 'research', 'ops', 'media', 'plan', 'synthesize', 'verify',
                                        'integrate'],
             'description': 'What the hand does; sets its tools and route. coding, ops and integrate have a shell; '
                            'research has web search; deep, plan, synthesize and verify only read. Default deep.'},
    'effort': {'type': 'integer', 'minimum': 0, 'maximum': 3, 'description': 'Requested depth, 0 to 3 (default 2).'},
    'risk': {'type': 'string', 'enum': ['low', 'normal', 'high', 'critical'],
             'description': 'What a mistake would cost (default normal).'},
    'needs': {'type': 'array', 'items': {'type': 'string', 'enum': ['code', 'shell', 'web', 'vision', 'audio',
                                                                   'long_context']},
              'description': 'Capabilities the work requires.'},
    'accept': {'type': 'object', 'description': 'The check fixed before work starts, run on the final result.',
               'properties': {'test_command': {'type': 'string', 'description': 'Test command run in the workdir.'},
                              'probe': {'type': 'string', 'description': 'Command whose output shows the result.'},
                              'requirements': {'type': 'array', 'items': {'type': 'string'},
                                               'description': 'Text the probe output must contain.'},
                              'output_type': {'type': 'string', 'description': 'Expected artifact type.'},
                              'required_sources': {'type': 'integer', 'minimum': 1,
                                                   'description': 'Research: independent sources to fetch.'}}},
    'touches': {'type': 'array', 'items': {'type': 'string'},
                'description': 'Repositories, files or services the work changes (locked one writer at a time).'},
    'workdir': {'type': 'string', 'description': 'Exact working directory.'},
    'name': {'type': 'string', 'pattern': '^[A-Za-z0-9-]{1,24}$', 'description': 'Optional label for the hand.'},
    'priority': {'type': 'string', 'enum': ['owner', 'own', 'auto', 'upkeep'],
                 'description': 'Who is waiting: owner (default in his conversation), own, auto or upkeep.'},
}


class RouteResolver(Protocol):
    def __call__(self, role: str, effort: int) -> Mapping[str, Any]: ...


_route_resolver: RouteResolver | None = None


def register_route_resolver(fn: RouteResolver) -> None:
    if not callable(fn):
        raise TypeError('route resolver must be callable')
    global _route_resolver
    _route_resolver = fn


def get_route_resolver() -> RouteResolver | None:
    return _route_resolver


def resolve(role: str, effort: int) -> Mapping[str, Any]:
    if _route_resolver is not None:
        return _route_resolver(role, effort)
    from tools.delegate_tool_config import _load_config
    return dict(_load_config())


def install(runtime):
    """Wrap only the module attribute, with the pinned call signature preserved."""
    module = runtime.native.delegate_module
    original = getattr(module.delegate_task, '_hands_native', module.delegate_task)
    if getattr(original, '_hands_runtime', None) is runtime:
        return
    if hasattr(original, '_hands_runtime'):
        raise RuntimeError('another hands runtime already owns delegation')
    runtime.original_delegate = original
    call_signature = signature(original)

    @wraps(original)
    def delegate_task(*args, **kwargs):
        try:
            if not runtime.available():
                return json.dumps({'error':'hands runtime unavailable','retry':False})
            selector = kwargs.pop('hand', None)
            bound = call_signature.bind(*args, **kwargs)
            values = bound.arguments
            parent = values.get('parent_agent')
            action = values.get('action')
            if action in {'list', 'stop', 'steer'}:
                # The pinned native control path checks the parent ownership chain.
                if parent is None or not runtime.authorized_owner(parent.session_id):
                    return json.dumps({'error':'owner session required'})
                return original(*args, **kwargs)
            if action not in {None, '', 'spawn'}:
                return json.dumps({'error':'unknown action'})
            tasks = values.get('tasks')
            if tasks is None:
                tasks = [{'goal':values.get('goal'), 'context':values.get('context') or ''}]
            if selector is not None:
                tasks = [{**task, 'hand': task.get('hand', selector)} for task in tasks]
            result = runtime.dispatch_tasks(tasks, parent=parent,
                native_options={k:values[k] for k in ('output_schema','images') if values.get(k) is not None})
            return json.dumps(result)
        except Exception as exc:
            # No exception or uncontrolled original call may escape into Hermes.
            return json.dumps({'error':'hands dispatch refused', 'reason':type(exc).__name__, 'retry':False})

    delegate_task._hands_runtime = runtime
    module.delegate_task = delegate_task
    runtime.wrapper = delegate_task
    extend_schema(runtime)


def extend_schema(runtime):
    """Add TASK_FIELDS to delegate_task's task items through the tool entry's per-call schema overrides."""
    lookup = getattr(runtime.native, 'tool_entry', None)
    entry = lookup('delegate_task') if callable(lookup) else None
    if entry is None:
        return
    current = getattr(entry, 'dynamic_schema_overrides', None)
    native = getattr(current, '_hands_native', current)

    def overrides():
        value = dict(native() or {}) if callable(native) else {}
        parameters = deepcopy(value.get('parameters') or entry.schema.get('parameters') or {})
        items = parameters.get('properties', {}).get('tasks', {}).get('items')
        if isinstance(items, dict):
            items['properties'] = {**items.get('properties', {}), **deepcopy(TASK_FIELDS)}
            value['parameters'] = parameters
        return value

    overrides._hands_runtime = runtime
    overrides._hands_native = native
    entry.dynamic_schema_overrides = overrides


def register(ctx: Any) -> None:
    # dispatch.register owns initialization after this resolver seam is loaded.
    return None
