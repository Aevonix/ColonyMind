"""Bounded work metadata captured without I/O on the tool lifecycle hooks.

Original arguments and results remain in the host's evidence stores. This module
keeps only typed postcondition facts and per-root opaque resource references in
the durable turn envelope. Resource equality is checked in transient memory.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, replace
import json
import math
import os
import posixpath
import re
import shlex
import threading
import time
from typing import Any, Mapping
import uuid

from .client import Settings

MAX_TURNS = 256
MAX_OPERATIONS = 256
MAX_TARGETS = 64
MAX_RESULT_CHARS = 16384
MAX_BINDINGS = 512
CONTEXT_SECONDS = 86400
AUTOMATED = frozenset({"cron", "worker", "system", "internal", "subagent", "kanban", "mind"})
READ_TOOLS = frozenset({"read_file", "search_files", "web_search", "web_extract", "session_search",
                        "skills_list", "skill_view", "kanban_get", "kanban_list", "tool_search", "tool_describe"})
JOB_FIELDS = frozenset({"name", "prompt", "schedule", "repeat", "deliver", "failure_deliver", "script",
                        "skill", "workdir", "no_agent", "paused", "monitor", "attach_to_session"})
FAILURE_KINDS = {"invalid_tool_arguments": "malformed_arguments", "tool_scope_block": "scope_block",
                 "guardrail_block": "guardrail_block", "plugin_block": "plugin_block",
                 "tool_timeout": "timeout", "tool_interrupted": "cancellation",
                 "tool_error": "tool_error", "ContentPolicyBlocked": "refusal"}


def duration(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def safe_identifier(value: Any, limit: int = 256) -> str | None:
    """Withhold detected values before they could enter an outbox digest."""
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        return None
    value = str(value)
    if not value or len(value) > limit:
        return None
    try:
        from .secrets import hide_secrets
        clean = hide_secrets(value)
    except Exception:
        return None
    return value if clean == value else None


def classification(tool: str, args: Mapping[str, Any]) -> tuple[str, str]:
    """Classify the actual action, never a remote tool's suggestive name."""
    action = str(args.get("action") or "")
    action = action if re.fullmatch(r"[a-z_]{0,32}", action) else ""
    if tool == "terminal":
        if args.get("background") is True or action == "background":
            return "background", "background"
        # Only literal probe calls without shell control, expansion or a
        # configurable external preprocessor have a known read-only effect.
        command = str(args.get("command") or "")
        if command and not re.search(r"[;&|`$><()\n]", command):
            try:
                parts = shlex.split(command)
            except ValueError:
                parts = []
            if parts and parts[0] in {"grep", "rg", "test", "["} and not any(
                    part.startswith(("--pre", "--hostname-bin")) for part in parts[1:]):
                return "probe", "read"
        return (action, "read") if action in {"read", "probe"} and not command else ("execute", "execute")
    if tool == "send_message":
        action = (action or "send") if action in {"", "send", "list"} else ""
        return action, {"list": "read", "send": "send"}.get(action, "effect_unknown")
    if tool in {"cronjob", "cronjob_manage"}:
        effects = {"list": "read", "get": "read", "create": "create", "edit": "edit", "update": "edit",
                   "remove": "remove", "pause": "edit", "resume": "edit", "run": "dispatch",
                   "resnap": "edit", "run_now": "dispatch", "trigger": "dispatch"}
        return action if action in effects else "", effects.get(action, "effect_unknown")
    if tool in {"delegate_task", "kanban_create"}:
        return "list" if action == "list" else "dispatch", "read" if action == "list" else "dispatch"
    if tool in {"write_file", "patch"}:
        mode = str(args.get("mode") or "write")
        return mode if mode in {"write", "append", "patch", "replace", "insert", "delete"} else "", "write"
    if tool in READ_TOOLS:
        return "read", "read"
    return "", "effect_unknown"


def result_mapping(result: Any) -> tuple[Mapping[str, Any], bool]:
    if isinstance(result, Mapping):
        return result, False
    if isinstance(result, str):
        if len(result) > MAX_RESULT_CHARS:
            return {}, True
        try:
            value = json.loads(result)
            return (value, False) if isinstance(value, Mapping) else ({}, False)
        except (ValueError, TypeError):
            pass
    return {}, False


@dataclass
class Turn:
    turn_id: str
    session_id: str
    root_turn_id: str
    parent_turn_id: str | None = None
    origin_ref: str | None = None
    info: Any = None
    candidate: bool = False
    operations: list[dict[str, Any]] = field(default_factory=list)
    children: list[dict[str, str]] = field(default_factory=list)
    resources: dict[tuple[str, str, str], str] = field(default_factory=dict)
    jobs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = field(default_factory=dict)
    seen_operations: set[str] = field(default_factory=set)
    seen_calls: set[str] = field(default_factory=set)
    request_models: dict[str, str | None] = field(default_factory=dict)
    sequence: int = 0
    pending_id: str | None = None
    frozen: dict[str, Any] | None = None
    truncated: bool = False
    seen_at: float = field(default_factory=time.monotonic)


class WorkCapture:
    """Turn identity, bounded metadata, and detached-child event envelopes.

    Configuration refresh belongs to registration and the body thread. The
    hooks read only a cached boolean, then update bounded state under one lock.
    """

    def __init__(self, settings: Settings, outbox: Any, *, limit: int = MAX_TURNS):
        self.settings, self.outbox, self.limit = settings, outbox, limit
        self._lock = threading.RLock()
        self._turns: OrderedDict[str, Turn] = OrderedDict()
        self._children: OrderedDict[str, tuple[str, str]] = OrderedDict()
        self._evicted: OrderedDict[str, Turn] = OrderedDict()
        self._pending: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._frozen_events: set[str] = set()
        self.enabled = True
        self.refresh()

    def refresh(self) -> None:
        """Refresh outside lifecycle hooks, including when the mind is off."""
        faculties = self.settings.mind().get("faculties") if self.settings is not None else {}
        enabled = self.settings is not None and not (
            isinstance(faculties, Mapping) and faculties.get("work_records") is False)
        with self._lock:
            self.enabled = enabled
            if not enabled:
                self._turns.clear()
                self._children.clear()
                self._pending.clear()
                self._frozen_events.clear()
                self._evicted.clear()

    def _trim(self) -> None:
        now = time.monotonic()
        while self._turns and (len(self._turns) > self.limit
                              or now - next(iter(self._turns.values())).seen_at > CONTEXT_SECONDS):
            ident, old = self._turns.popitem(last=False)
            if old.frozen is None or old.operations or old.truncated:
                evicted = Turn(old.turn_id, old.session_id, old.root_turn_id,
                    old.parent_turn_id, old.origin_ref,
                    replace(old.info, user_message="", said="") if old.info is not None else None,
                    old.candidate, truncated=True)
                if old.frozen is not None:
                    # A saturated pending queue can leave late facts waiting in
                    # the context. Eviction must still invalidate complete coverage.
                    evicted.frozen = self._envelope(evicted)
                self._evicted[ident] = evicted
                while len(self._evicted) > MAX_BINDINGS:
                    self._evicted.popitem(last=False)
            old.resources.clear()
            old.jobs.clear()

    def start(self, info: Any, *, task_id: str = "", turn_id: str = "", **_: Any) -> None:
        if not self.enabled or info is None or not turn_id or os.environ.get("HERMES_KANBAN_TASK"):
            return
        platform = str(info.platform or "").lower()
        with self._lock:
            if turn_id in self._turns:
                return
            binding = self._children.get(info.session_id)
            if binding is not None:
                root, parent = binding
                candidate = False
            else:
                root, parent = turn_id, None
                from .capture import ingress_verified
                candidate = (platform not in AUTOMATED
                             and not str(task_id).startswith(("cron:", "mind:"))
                             and not info.session_id.startswith(("cron_", "mind:"))
                             and ((platform == "cli" and not info.sender_id)
                                  or (bool(info.sender_id) and ingress_verified(info, info.session_id, turn_id))))
            origin = None
            if candidate:
                heard = info.heard
                if heard is not None and heard.message_id:
                    origin = json.dumps({"platform": heard.platform, "chat": heard.chat_id,
                                         "message_id": heard.message_id}, sort_keys=True, separators=(",", ":"))
                    origin = safe_identifier(origin, 1024)
                origin = origin or turn_id
            state = Turn(turn_id, info.session_id, root, parent, origin, replace(info), candidate)
            state.truncated = turn_id in self._evicted
            self._turns[turn_id] = state
            if parent:
                ancestor = self._turns.get(parent)
                child = {"turn_id": turn_id, "parent_turn_id": parent}
                if ancestor is not None and ancestor.frozen is None:
                    if len(ancestor.children) < MAX_OPERATIONS:
                        ancestor.children.append(child)
                    else:
                        ancestor.truncated = True
            self._trim()

    def subagent_start(self, *, parent_turn_id: str = "", child_session_id: str = "", **_: Any) -> None:
        if not self.enabled or not parent_turn_id or not child_session_id:
            return
        with self._lock:
            parent = self._turns.get(parent_turn_id)
            if parent is None or not (parent.candidate or parent.parent_turn_id):
                return
            self._children[child_session_id] = (parent.root_turn_id, parent_turn_id)
            while len(self._children) > MAX_BINDINGS:
                self._children.popitem(last=False)

    def info(self, turn_id: str, *, unfinished: bool = False) -> Any:
        with self._lock:
            state = self._turns.get(turn_id) or self._evicted.get(turn_id)
            return state.info if self.enabled and state is not None and not (unfinished and (
                state.frozen is not None or not (state.candidate or state.parent_turn_id))) else None

    def _resource(self, state: Turn, family: str, value: Any, *, task_id: str = "") -> str | None:
        if not isinstance(value, str) or not value or len(value) > 4096:
            return None
        if family == "file":
            value = posixpath.normpath(value)
        # Root scope and exact full value, never basename or URL host equality.
        root = self._turns.get(state.root_turn_id) or state
        key = (family, task_id or state.session_id, value)
        if key not in root.resources:
            if len(root.resources) >= MAX_OPERATIONS:
                state.truncated = True
                return None
            root.resources[key] = "resource:" + uuid.uuid4().hex
        return root.resources[key]

    def pre_api_request(self, *, turn_id: str = "", api_request_id: str = "", model: str = "",
                        **kwargs: Any) -> None:
        """Observe submitted recall and cache the exact request producer."""
        from .capture import pre_api_request
        pre_api_request(turn_id=turn_id, **kwargs)
        if not self.enabled:
            return
        with self._lock:
            state = self._turns.get(turn_id)
            if state is None or not (state.candidate or state.parent_turn_id):
                return
            request = safe_identifier(api_request_id)
            if request is None or len(state.request_models) >= MAX_OPERATIONS and request not in state.request_models:
                state.truncated = True
                return
            state.request_models[request] = safe_identifier(model)
            state.truncated = state.truncated or bool(model and state.request_models[request] is None)

    def post_api_request(self, *, turn_id: str = "", api_request_id: str = "", model: str = "",
                         response_model: str = "", finish_reason: str = "", api_duration: Any = None,
                         **kwargs: Any) -> None:
        self.pre_api_request(turn_id=turn_id, api_request_id=api_request_id, model=response_model or model)
        if finish_reason in {"content_filter", "refusal", "guardrail_intervened"}:
            self.api_request_error(turn_id=turn_id, api_request_id=api_request_id,
                                   api_duration=api_duration, error={"type": "ContentPolicyBlocked"}, **kwargs)

    def api_request_error(self, *, turn_id: str = "", api_request_id: str = "", model: str = "",
                          error: Any = None, reason: str = "", api_duration: Any = None, **_: Any) -> None:
        """Request failures have no effect intent or necessarily a named tool.

        model_response is an explicit request-level marker, not an inferred tool.
        Raw provider messages, request bodies and refusal text are never retained.
        """
        error_type = error.get("type") if isinstance(error, Mapping) else None
        kind = ("refusal" if error_type == "ContentPolicyBlocked" or reason == "content_policy_blocked" else
                "parse_failure" if error_type in {"JSONDecodeError", "ResponseParseError", "InvalidAPIResponse"}
                or reason == "invalid_response" else None)
        if not self.enabled or kind is None:
            return
        with self._lock:
            state = self._turns.get(turn_id)
            request = safe_identifier(api_request_id)
            if state is None or not (state.candidate or state.parent_turn_id):
                return
            if request is None:
                state.truncated = True
                return
            if request not in state.request_models:
                self.pre_api_request(turn_id=turn_id, api_request_id=api_request_id, model=model)
            # A refusal can arrive via both response and error observers.
            call = "request-failure:" + uuid.uuid5(uuid.NAMESPACE_URL, f"{request}:{kind}").hex
            if call in state.seen_calls:
                return
            seconds = duration(api_duration)
            self.post_tool_call(turn_id=turn_id, tool_call_id=call, tool_name="model_response",
                                api_request_id=api_request_id, error_type=error_type,
                                failure_kind=kind, duration_ms=seconds * 1000 if seconds is not None else None,
                                status="error")

    def post_tool_call(self, *, session_id: str = "", turn_id: str = "", tool_call_id: str = "",
                       tool_name: str = "", task_id: str = "", args: Any = None, result: Any = None, status: str = "",
                       error_type: str = "", api_request_id: str = "", duration_ms: Any = None,
                       failure_kind: str | None = None,
                       **_: Any) -> None:
        """Metadata only: no contact lookup, filesystem read or delivery."""
        if not self.enabled or not turn_id or not tool_call_id:
            return
        with self._lock:
            state = self._turns.get(turn_id)
            if state is None or not (state.candidate or state.parent_turn_id):
                return
            # A rotated session still belongs to the stable turn identity.
            arguments = args if isinstance(args, Mapping) else {}
            value, oversized = result_mapping(result)
            action, effect = classification(tool_name, arguments)
            kind = failure_kind or FAILURE_KINDS.get(error_type) or {
                "timeout": "timeout", "cancelled": "cancellation", "interrupted": "cancellation",
                "refused": "refusal"}.get(status)
            success = (False if error_type or kind or status in {"error", "failed", "cancelled", "interrupted"}
                       or value.get("error") or value.get("success") is False else
                       True if status in {"ok", "success"} or value.get("success") is True else None)
            state.truncated = state.truncated or oversized
            family, targets = "", [(effect, None)]
            if tool_name in {"write_file", "patch", "read_file"}:
                family = "file"
                targets = [(effect, value.get("resolved_path") or arguments.get("path"))]
                if tool_name == "patch":
                    files = []
                    for key, changed in (("files_modified", "write"), ("files_created", "write"),
                                         ("files_deleted", "remove")):
                        paths = value.get(key)
                        if isinstance(paths, list):
                            room = MAX_TARGETS - len(files)
                            files.extend((changed, path) for path in paths[:room])
                            state.truncated = state.truncated or len(paths) > room
                    if files:
                        targets = files
            elif tool_name == "send_message":
                # A destination identifies a conversation, not the sent object.
                # Only a native receipt identifies a particular message there.
                receipt = next((safe_identifier(value.get(key)) for key in
                                ("provider_message_id", "message_id", "receipt_id")
                                if safe_identifier(value.get(key))), None)
                destination = arguments.get("target")
                exact = (f"{destination}\0{receipt}" if receipt and isinstance(destination, str)
                         and len(destination) <= 1024 else None)
                family, targets = "message", [(effect, exact)]
            elif tool_name in {"cronjob", "cronjob_manage"}:
                family = "job"
                saved_job = value.get("job")
                job_id = (saved_job.get("job_id") or saved_job.get("id")) if isinstance(saved_job, Mapping) else None
                targets = [(effect, value.get("job_id") or job_id or arguments.get("job_id"))]
            safe_turn = safe_identifier(turn_id)
            name, call = safe_identifier(tool_name, 128), safe_identifier(tool_call_id)
            if safe_turn is None or name is None or call is None:
                state.truncated = True
                if state.frozen is not None:
                    self._queue_event(state)
                return
            safe_action = safe_identifier(action, 64) or ""
            request = safe_identifier(api_request_id)
            producer = state.request_models.get(request) if request else None
            native_error = safe_identifier(error_type, 128)
            elapsed = duration(duration_ms)
            call_facts = {"request_id": request, "model": producer, "error_type": native_error,
                          "failure_kind": kind, "duration_ms": elapsed}
            state.truncated = state.truncated or bool(
                api_request_id and (request is None or producer is None)
                or error_type and native_error is None or duration_ms is not None and elapsed is None)
            state.sequence += 1
            state.truncated = state.truncated or call in state.seen_calls
            if len(state.seen_calls) < MAX_OPERATIONS:
                state.seen_calls.add(call)
            captured_call = False
            call_ident = uuid.uuid5(uuid.NAMESPACE_URL, f"{safe_turn}:{state.sequence}:{call}").hex
            for index, (target_effect, target) in enumerate(targets):
                ident = f"operation:{call_ident}:{index}"
                if len(state.seen_operations) >= MAX_OPERATIONS:
                    state.truncated = True
                    break
                state.seen_operations.add(ident)
                ref = self._resource(state, family, target, task_id=task_id) if family else None
                evidence: dict[str, Any] = {"kind": "read" if effect == "read" else "none"}
                if (ref is not None and target_effect == "write" and tool_name in {"write_file", "patch"}
                        and value.get("verified") is True and success is not False):
                    evidence = {"kind": "file_hash", "method": "native_post_write_sha256", "verified": True}
                elif effect == "send" and success is True:
                    for source, dest in (("receipt_id", "receipt_id"), ("provider_message_id", "provider_message_id"),
                                         ("message_id", "provider_message_id")):
                        clean = safe_identifier(value.get(source))
                        if clean:
                            evidence = {"kind": "message_receipt", dest: clean}
                            break
                operation = {"operation_id": ident, "tool_call_id": call, "tool": name,
                             "action": safe_action, "effect": target_effect,
                             "resource_ref": ref, "success": success, "evidence": evidence, "source_refs": []}
                operation.update(call_facts)
                state.operations.append(operation)
                captured_call = True
                if family == "job" and effect in {"create", "edit"} and isinstance(target, str):
                    requested = {key: arguments[key] for key in JOB_FIELDS if key in arguments
                                 and isinstance(arguments[key], (str, int, float, bool, type(None)))
                                 and len(str(arguments[key])) <= 1024}
                    if requested and len(requested) == sum(key in arguments for key in JOB_FIELDS):
                        state.jobs[target] = (operation, requested)
            if captured_call and tool_name in {"cronjob", "cronjob_manage"} and effect == "read" and success is True:
                self._job_reads(state, value, tool_call_id, call_facts)
            if state.frozen is not None and (state.operations or state.truncated):
                self._queue_event(state)

    def _job_reads(self, state: Turn, value: Mapping[str, Any], tool_call_id: str,
                   call_facts: Mapping[str, Any]) -> None:
        jobs = value.get("jobs")
        if not isinstance(jobs, list):
            jobs = [value.get("job")]
        for job in jobs[:MAX_TARGETS]:
            if not isinstance(job, Mapping):
                continue
            pending = state.jobs.get(str(job.get("id") or job.get("job_id") or ""))
            if pending is None:
                continue
            operation, requested = pending
            # Hermes lists formatted jobs: prompt is a preview; repeat and
            # schedule are display text, so equality cannot prove or refute them.
            display_only = {"prompt", "repeat", "schedule"} if "prompt_preview" in job else set()
            if any(key not in job or key in display_only for key in requested):
                continue
            # Opaque field comparison facts preserve equality without retaining
            # prompts, destinations, schedules or arbitrary stored job values.
            evidence = {"kind": "saved_job", "method": "native_saved_job_fields",
                        "requested_fields": {key: True for key in requested},
                        "saved_fields": {key: type(job[key]) is type(val) and job[key] == val
                                         for key, val in requested.items()}}
            if state.frozen is None:
                operation["evidence"] = evidence
            elif len(state.operations) < MAX_OPERATIONS:
                check = {**operation, **call_facts, "operation_id": "check:" + uuid.uuid4().hex,
                         "tool_call_id": tool_call_id, "action": "list", "effect": "read",
                         "success": True, "evidence": {**evidence,
                             "checked_operation_id": operation["operation_id"]}}
                state.operations.append(check)

    def _envelope(self, state: Turn) -> dict[str, Any]:
        return {"version": 1, "root_turn_id": state.root_turn_id, "parent_turn_id": state.parent_turn_id,
                "origin_ref": state.origin_ref, "operations": deepcopy(state.operations),
                "children": deepcopy(state.children), "truncated": state.truncated}

    def freeze(self, turn_id: str) -> dict[str, Any] | None:
        with self._lock:
            state = self._turns.get(turn_id) or self._evicted.get(turn_id)
            if not self.enabled or state is None or not state.candidate:
                return None
            if state.frozen is None:
                state.frozen = self._envelope(state)
                state.operations.clear()
            return deepcopy(state.frozen)

    def _queue_event(self, state: Turn) -> None:
        if not state.operations and not state.children and not state.truncated:
            return
        existing = (self._pending.get(state.pending_id or "")
                    if state.pending_id not in self._frozen_events else None)
        if existing is not None:
            envelope = existing["work"]
            room = MAX_OPERATIONS - len(envelope["operations"])
            envelope["operations"].extend(deepcopy(state.operations[:room]))
            child_room = MAX_OPERATIONS - len(envelope["children"])
            envelope["children"].extend(deepcopy(state.children[:child_room]))
            envelope["truncated"] = (envelope["truncated"] or state.truncated
                                     or len(state.operations) > room or len(state.children) > child_room)
            state.operations.clear()
            state.children.clear()
            state.truncated = False
            return
        if len(self._pending) >= MAX_BINDINGS:
            # Retain the current facts and mark incomplete coverage. A later
            # body flush retries them after draining admitted pending envelopes.
            state.truncated = True
            return
        event_id = "work-event:" + uuid.uuid4().hex
        event = {"event_id": event_id, "root_turn_id": state.root_turn_id,
                 "session_id": state.session_id, "work": self._envelope(state), "source_refs": []}
        self._pending[event_id] = {"kind": "work_event", **event}
        state.pending_id = event_id
        state.operations.clear()
        state.children.clear()
        state.truncated = False

    def finish_child(self, turn_id: str) -> bool:
        with self._lock:
            state = self._turns.get(turn_id) or self._evicted.get(turn_id)
            if not self.enabled or state is None or not state.parent_turn_id:
                return False
            if state.frozen is not None and not state.operations:
                return True
            state.children = [{"turn_id": turn_id, "parent_turn_id": state.parent_turn_id}]
            self._queue_event(state)
            state.frozen = self._envelope(state)
        # Same durable boundary as a root's final capture, never a tool hook.
        self.flush()
        return True

    def flush(self) -> None:
        """Body-thread persistence of late metadata, safe to replay on failure."""
        with self._lock:
            if not self.enabled:
                return
            events = list(self._pending.items())
            # Once persistence starts an ambiguous failure may mean the outbox
            # already committed. Retrying that ID must keep the same envelope.
            self._frozen_events.update(self._pending)
            self._pending.clear()
        for index, (event_id, payload) in enumerate(events):
            try:
                self.outbox.enqueue(event_id, payload)
            except Exception:
                with self._lock:
                    # Preserve exactly the failed envelope identity for retry.
                    self._pending.update(events[index:])
                raise
            else:
                with self._lock:
                    self._frozen_events.discard(event_id)
        with self._lock:
            for state in [*self._turns.values(), *self._evicted.values()]:
                if state.frozen is not None and (state.operations or state.truncated):
                    self._queue_event(state)
