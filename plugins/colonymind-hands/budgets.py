"""Local attempt accounting and clocks, independent of controller availability."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
from pathlib import Path
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)
_DEFAULTS = {
    "coding": (1800, 100, 6, 1000000), "deep": (1200, 20, 6, 300000),
    "research": (1200, 40, 12, 400000), "ops": (900, 60, 4, 300000),
    "media": (600, 20, 4, 200000), "plan": (600, 20, 6, 200000),
    "synthesize": (600, 20, 6, 200000), "verify": (600, 20, 6, 200000),
    "integrate": (600, 20, 6, 200000),
}
TERMINAL = frozenset({"done", "below_minimum", "failed", "stopped", "superseded", "lost", "unknown_effect", "refused", "expired"})
WAITING = frozenset({"awaiting_input", "awaiting_approval"})
# Stops that end with the hand's own report. Tool refusals after them say "write up what you
# have", so the child gets one bounded model call for that report (middleware.llm_request).
# A wall stop interrupts at once: its report prompt is the earlier wrap-up steer.
REPORT_STOPS = frozenset({"token_budget"})
# hands.budgets.final_report_seconds: how long that report may take before the executor is
# interrupted, and hands.budgets.final_report_tokens: its output bound. Both are assumptions.
FINAL_REPORT_SECONDS = 300
FINAL_REPORT_TOKENS = 8192
_STOP_WHY = {
    "token_budget": "its token budget was spent",
    "wall_budget": "its time budget was spent",
    "stale": "it stopped making progress",
    "queue_timeout": "it waited too long for a model",
    "accounting_unavailable": "its token accounting was unavailable",
}


@dataclass
class HandState:
    hand_id: str
    task_id: str = ""
    kind: str = "research"
    plan: dict[str, Any] = field(default_factory=dict)
    child_session_id: str = ""
    parent_session_id: str = ""
    delegation_id: str = ""
    subagent_id: str = ""
    agent: Any = None
    epoch: Any = ""
    status: str = "running"
    activity: str = "queued"
    blocked: bool = False
    started_at: float = field(default_factory=time.monotonic)
    activity_since: float = field(default_factory=time.monotonic)
    expected_seconds: float = 0.0
    tools: int = 0
    searches: int = 0
    # tokens is what the budget counts: uncached prompt plus output. raw_tokens also counts every
    # prompt token a provider served from its prefix cache, as tokens did before; kept as evidence.
    tokens: int = 0
    raw_tokens: int = 0
    seq: int = 0
    no_new_information: int = 0
    session_ids: set[str] = field(default_factory=set)
    families_used: set[str] = field(default_factory=set)
    contract_versions: set[tuple[str, str]] = field(default_factory=set)
    repeats: dict[str, int] = field(default_factory=dict)
    seen_evidence: set[str] = field(default_factory=set)
    counted_calls: set[str] = field(default_factory=set)
    refused_calls: set[str] = field(default_factory=set)
    counted_results: set[str] = field(default_factory=set)
    counted_requests: set[str] = field(default_factory=set)
    local_requests: set[str] = field(default_factory=set)
    wall_warned: bool = False
    stall_steered_at: float | None = None
    stop_reason: str = ""
    anchor_warned: bool = False
    request_family: str = ""
    request_id: str = ""
    request_refused: bool = False
    # The bounded final report after a REPORT_STOPS stop: "" (none), "allowed", "claimed" (one
    # request id holds it), "used" (a response arrived) or "expired" (the executor was interrupted).
    final_report: str = ""
    final_report_request: str = ""
    final_report_since: float = 0.0
    final_report_text: str = ""
    lock: Any = field(default_factory=threading.RLock, repr=False, compare=False)


class HandRegistry:
    """Bind stable child task ids to all session ids, including compression rotations."""
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self._hands: dict[str, HandState] = {}
        self._aliases: dict[str, str] = {}
        self._sessions: dict[str, str] = {}
        self._tasks: dict[str, str] = {}
        self.requests = threading.Condition(self.lock)
        self.inflight: dict[str, str] = {}

    def add(self, hand: HandState) -> HandState:
        with self.lock:
            if hand.hand_id in self._hands and self._hands[hand.hand_id] is not hand:
                raise ValueError("hand already registered")
            self._hands[hand.hand_id] = hand
            self.bind(hand, hand.child_session_id, agent=hand.agent, subagent_id=hand.subagent_id)
            return hand

    def bind(self, hand: HandState | str, session_id: str, agent: Any = None,
             subagent_id: str | None = None, task_id: str | None = None) -> HandState:
        with self.lock:
            state = self.get(hand) if isinstance(hand, str) else hand
            if state is None:
                raise KeyError(hand)
            for key, index in ((session_id, self._sessions), (subagent_id, self._tasks), (task_id, self._tasks)):
                if key and key in index and index[key] != state.hand_id:
                    raise ValueError("child identity already belongs to another hand")
            if session_id:
                self._sessions[session_id] = state.hand_id
                state.session_ids.add(session_id)
                state.child_session_id = session_id
            if subagent_id:
                state.subagent_id = subagent_id
                self._tasks[subagent_id] = state.hand_id
            if task_id:
                self._tasks[task_id] = state.hand_id
            if agent is not None:
                state.agent = agent
            return state

    def get(self, hand_id: str) -> HandState | None:
        with self.lock:
            return self._hands.get(self._aliases.get(hand_id, hand_id))

    def alias(self, hand_id: str, hand: HandState | str) -> HandState:
        with self.lock:
            state = self.get(hand) if isinstance(hand, str) else hand
            if state is None or state.hand_id not in self._hands:
                raise KeyError(hand)
            if self.get(hand_id) not in (None, state):
                raise ValueError("hand alias already belongs to another hand")
            self._aliases[hand_id] = state.hand_id
            return state

    def for_session(self, session_id: str, agent: Any = None,
                    subagent_id: str | None = None, task_id: str | None = None) -> HandState | None:
        with self.lock:
            ids = {value for value in (self._sessions.get(session_id), self._tasks.get(subagent_id),
                                      self._tasks.get(task_id)) if value}
            if agent is not None:
                ids.update(h.hand_id for h in self._hands.values() if h.agent is agent)
            elif session_id:
                ids.update(h.hand_id for h in self._hands.values()
                           if h.agent is not None and getattr(h.agent, "session_id", None) == session_id)
            if len(ids) > 1:
                raise ValueError("conflicting child identity")
            state = self._hands.get(next(iter(ids))) if ids else None
            if state is not None:
                self.bind(state, session_id, agent=agent, subagent_id=subagent_id, task_id=task_id)
            return state

    def all(self) -> list[HandState]:
        with self.lock:
            return list(self._hands.values())

    def remove(self, hand_id: str) -> HandState | None:
        with self.lock:
            hand_id = self._aliases.get(hand_id, hand_id)
            state = self._hands.pop(hand_id, None)
            self._aliases = {k: v for k, v in self._aliases.items() if v != hand_id}
            self._sessions = {k: v for k, v in self._sessions.items() if v != hand_id}
            self._tasks = {k: v for k, v in self._tasks.items() if v != hand_id}
            self.inflight.pop(hand_id, None)
            self.requests.notify_all()
            return state


def settings(runtime: Any) -> dict[str, Any]:
    config = runtime.config
    return config.get("hands", config)


def default_budget(kind: str, effort: int = 1) -> dict[str, int]:
    values = dict(zip(("wall_seconds", "tool_calls", "web_searches", "tokens"), _DEFAULTS[kind]))
    if int(effort) == 3:
        values["tokens"] *= 2
        values["wall_seconds"] *= 2
    return values


def limits(runtime: Any, hand: HandState) -> dict[str, float]:
    budget = settings(runtime).get("budgets", {})
    values = default_budget(hand.kind)
    values.update(budget.get("attempt", {}).get(hand.kind, {}))
    if int(hand.plan.get("effort", 1)) == 3:
        for key in ("tokens", "wall_seconds"):
            values[key] *= budget.get("effort_3_multiplier", 2)
    # The controller already applies effort to its final attempt allocation.
    values.update(hand.plan.get("budgets", {}))
    return values


def resolve_hand(runtime: Any, session_id: str = "", **ids: Any) -> HandState | None:
    return runtime.registry.for_session(session_id, agent=ids.get("agent"),
        subagent_id=ids.get("subagent_id"), task_id=ids.get("task_id"))


def is_child_context(**ids: Any) -> bool:
    if ids.get("platform") == "subagent" or ids.get("subagent_id") or str(ids.get("task_id") or "").startswith("sa-"):
        return True
    try:
        from agent.delegation_context import is_delegated_child_context
        return is_delegated_child_context()
    except ImportError:
        return False


def block(message: str) -> dict[str, str]:
    return {"action": "block", "message": message}


def refuse(hand: HandState | None, message: str, tool_call_id: str = "") -> dict[str, str]:
    # Hermes runs every hook before resolving the veto. Later effect observers
    # need to know this call will not run before writing an effect intent.
    if hand is not None and tool_call_id:
        with hand.lock:
            hand.refused_calls.add(tool_call_id)
    return block(message)


def ended_message(hand: HandState) -> str:
    """What a refused tool call tells a hand that has ended or is waiting."""
    if hand.stop_reason in REPORT_STOPS:
        spent = hand.stop_reason.replace("_", " ")
        return spent + " spent: tools are closed; write the final report now with what you have"
    return "hand has ended or is awaiting input: write the final report"


def stopped_text(hand: HandState) -> str:
    """The local reply for a model call refused after a stop: the report, or why the hand ended."""
    if hand.final_report_text:
        return hand.final_report_text
    why = _STOP_WHY.get(hand.stop_reason, "it was stopped (" + str(hand.stop_reason or "stopped") + ")")
    return f"This hand ended because {why}. No further model calls run for it."


def stop(runtime: Any, hand: HandState, reason: str) -> None:
    """Stop a hand for the runtime's own reasons. Never raises.

    Budget, clock, queue and accounting stops come from the runtime itself, often on a delegate or
    maintenance thread where no owner session is current. They use the runtime's system path, which
    has no owner-session check and keeps every other effect of a stop. A REPORT_STOPS stop journals
    the stop and leaves the executor running for one bounded final report; the clock interrupts it
    if that report overruns. Every other stop interrupts the executor at once.
    """
    if hand.stop_reason:
        return
    hand.stop_reason = reason
    hand.blocked = True
    report = reason in REPORT_STOPS
    if report:
        hand.final_report = "allowed"
        hand.final_report_since = time.monotonic()
    try:
        runtime.system_stop(hand, reason, interrupt=not report)
    except Exception:
        # The stop state above already refuses tools and further model calls.
        logger.exception("hand %s: %s stop incomplete", hand.hand_id, reason)


def _report_clock(runtime: Any, hand: HandState, now: float) -> None:
    """Interrupt a hand whose final report outlasts hands.budgets.final_report_seconds."""
    if hand.plan.get("_result_submitted") or hand.status not in {"stopped", "unknown_effect"}:
        return  # The result is recorded or another path now owns the outcome.
    grace = float(settings(runtime).get("budgets", {}).get("final_report_seconds", FINAL_REPORT_SECONDS))
    if now - hand.final_report_since < grace:
        return
    hand.final_report = "expired"
    try:
        # The stop journal entry repeats with the same key and body; only the interrupt is new.
        runtime.system_stop(hand, hand.stop_reason)
    except Exception:
        logger.exception("hand %s: final report interrupt incomplete", hand.hand_id)


def _clock(runtime: Any, hand: HandState, now: float) -> None:
    if hand.final_report in {"allowed", "claimed"}:
        _report_clock(runtime, hand, now)
        return
    if hand.status in TERMINAL or hand.activity in WAITING or hand.stop_reason:
        return
    cap = limits(runtime, hand)["wall_seconds"]
    elapsed = now - hand.started_at
    if elapsed >= cap:
        stop(runtime, hand, "wall_budget")
        return
    fraction = settings(runtime).get("budgets", {}).get("wall_warning_fraction", 0.8)
    if elapsed >= cap * fraction and not hand.wall_warned:
        hand.wall_warned = True
        runtime.system_steer(hand, "wrap up and report")
    if hand.activity not in {"generating", "executing"} or hand.expected_seconds <= 0:
        return
    stalls = settings(runtime).get("stalls", {})
    grace = (float(stalls.get("generation_multiplier", 2)) * hand.expected_seconds if hand.activity == "generating"
             else hand.expected_seconds + float(stalls.get("tool_grace_seconds", 60)))
    if now - hand.activity_since <= grace:
        return
    if hand.stall_steered_at is None:
        hand.stall_steered_at = now
        runtime.system_steer(hand, "status?")
    elif now - hand.stall_steered_at >= max(1, grace):
        stop(runtime, hand, "stale")


def tick(runtime: Any, now: float | None = None) -> None:
    """Called by the runtime's periodic worker even when a model or tool hangs."""
    for hand in runtime.registry.all():
        try:
            with hand.lock:
                _clock(runtime, hand, time.monotonic() if now is None else now)
        except Exception:
            logger.exception("hand clock failed for %s", hand.hand_id)


# Hermes tools whose schemas declare a timeout, by the names hooks see. Other tools' arguments carry no
# timeout, so a model-invented one is ignored. Defaults are Hermes' own: terminal 180 s (TERMINAL_TIMEOUT) and
# execute_code 300 s (code_execution timeout; it has no timeout parameter).
TOOL_TIMEOUT_KEYS = {"terminal": "timeout", "process_manage": "timeout", "browser_cdp": "timeout",
                     "manage_connections": "timeout_seconds"}
TOOL_DEFAULT_SECONDS = {"terminal": 180.0, "execute_code": 300.0}


def expected_seconds(tool_name: str, arguments: dict[str, Any]) -> float:
    """How long this call may take before stall handling: the tool's own timeout argument when it declares one and
    the value is a usable number, else the tool's default (120 s if Hermes names none). At least 1 s, so stall
    handling stays on.

    Models send null, text or invented timeouts; none of those may fail the budget check, which refuses the call
    and stops the hand."""
    default = TOOL_DEFAULT_SECONDS.get(tool_name, 120.0)
    key = TOOL_TIMEOUT_KEYS.get(tool_name)
    value = arguments.get(key) if key else None
    if value is None or isinstance(value, bool):
        return default
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(1.0, seconds) if math.isfinite(seconds) else default


def pre_tool_call(runtime: Any, tool_name: str, args: dict[str, Any] | None = None,
                  session_id: str = "", tool_call_id: str = "", **ids: Any) -> dict | None:
    hand = None
    try:
        hand = resolve_hand(runtime, session_id, **ids)
        if hand is None:
            return block("unbound hand session: tool refused") if is_child_context(**ids) else None
        with hand.lock:
            _clock(runtime, hand, time.monotonic())
            if hand.blocked or hand.activity in WAITING or hand.status in TERMINAL:
                return refuse(hand, ended_message(hand), tool_call_id)
            cap = limits(runtime, hand)
            budget = settings(runtime).get("budgets", {})
            if hand.tokens >= cap["tokens"]:
                stop(runtime, hand, "token_budget")
                return refuse(hand, "token budget spent: write up what you have", tool_call_id)
            if hand.tools >= cap["tool_calls"]:
                return refuse(hand, "tool budget spent: write up what you have", tool_call_id)
            if tool_name == "web_search" and hand.searches >= cap["web_searches"]:
                return refuse(hand, "search budget spent: write up what you have", tool_call_id)
            if tool_name in {"web_search", "web_extract"} and hand.no_new_information >= budget.get("no_new_information_stop", 5):
                return refuse(hand, "no new information: write up what you have", tool_call_id)
            arguments = args or {}
            seconds = expected_seconds(tool_name, arguments)  # before any counter changes
            # Non-finite or unserializable arguments raise here and the call is refused without a stop: the effect
            # spool could not record them either.
            digest = tool_name + ":" + hashlib.sha256(json.dumps(arguments, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            if tool_call_id and tool_call_id in hand.counted_calls:
                return None
            if hand.repeats.get(digest, 0) + 1 >= budget.get("repeat_tool_calls", 3):
                return refuse(hand, "repeated identical tool call: write up what you have", tool_call_id)
            hand.repeats[digest] = hand.repeats.get(digest, 0) + 1
            if tool_call_id:
                hand.counted_calls.add(tool_call_id)
            hand.tools += 1
            hand.searches += tool_name == "web_search"
            hand.activity = "executing"
            hand.activity_since = time.monotonic()
            hand.stall_steered_at = None
            hand.expected_seconds = seconds
        return None
    except Exception:
        logger.exception("hand budget check failed")
        if hand is None and not is_child_context(**ids):
            return None
        return refuse(hand, "hand budget check unavailable: tool refused", tool_call_id)


def _evidence(value: Any) -> set[str]:
    """Use result structure, excluding query text, timing, ranks and prose metadata."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError, RecursionError):
            return {"content:" + hashlib.sha256(value.encode()).hexdigest()} if value else set()
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"url", "source_url", "link"} and isinstance(item, str):
                found.add("url:" + item)
            elif key in {"content_hash", "sha256"} and isinstance(item, str):
                found.add("content:" + item)
            elif key in {"content", "text", "snippet", "markdown", "raw_content"} and isinstance(item, str):
                found.add("content:" + hashlib.sha256(item.encode()).hexdigest())
            elif isinstance(item, (dict, list)):
                found.update(_evidence(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_evidence(item))
    return found


def _tool_receipts(tool: str, result: Any, args: dict, hand: HandState) -> list[dict]:
    digest = hashlib.sha256(json.dumps(result, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (ValueError, RecursionError):
            result = {}
    base = {"tool": tool, "result_hash": digest}
    if isinstance(result, dict) and isinstance(result.get("exit_code"), int):
        base["exit_code"] = result["exit_code"]
    if tool in {"read_file", "write_file", "edit_file"} and args.get("path"):
        base["path"] = str(args["path"])[:2048]
        # Hash actual local bytes only when the task explicitly binds a directory.
        workdir = (hand.plan.get("workdir") or hand.plan.get("task", {}).get("workdir")
                   or hand.plan.get("cwd") or hand.plan.get("workspace"))
        if workdir and tool in {"write_file", "edit_file"}:
            target = Path(args["path"])
            target = target if target.is_absolute() else Path(workdir) / target
            try:
                if target.is_file() and target.stat().st_size <= 1048576:
                    base["content_hash"] = hashlib.sha256(target.read_bytes()).hexdigest()
                    outputs = hand.plan.setdefault("observed_artifacts", [])
                    if str(target.resolve()) not in outputs:
                        outputs.append(str(target.resolve()))
            except OSError:
                pass
    sources: list[dict] = []
    def visit(value: Any) -> None:
        if len(sources) >= 20:
            return
        if isinstance(value, dict):
            url = value.get("url") or value.get("source_url")
            if isinstance(url, str):
                receipt = {**base, "url": url[:2048]}
                content = next((value[key] for key in ("content", "text", "markdown", "raw_content", "snippet")
                                if isinstance(value.get(key), str)), None)
                if content is not None:
                    receipt["content_hash"] = hashlib.sha256(content.encode()).hexdigest()
                    if tool == "web_extract":
                        # The fetched text is what a proposed quote is checked against, never a quote itself.
                        hand.plan.setdefault("fetched_texts", {})[url] = content
                sources.append(receipt)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    if tool in {"web_search", "web_extract"}:
        visit(result)
    return sources or [base]


def post_tool_call(runtime: Any, tool_name: str, result: Any = None, session_id: str = "",
                   tool_call_id: str = "", **ids: Any) -> None:
    try:
        hand = resolve_hand(runtime, session_id, **ids)
        if hand is None:
            return
        with hand.lock:
            if tool_call_id in hand.refused_calls:
                return
            if tool_call_id and tool_call_id in hand.counted_results:
                return
            if tool_call_id:
                hand.counted_results.add(tool_call_id)
            receipts = hand.plan.setdefault("tool_digest", [])
            receipts.extend(_tool_receipts(tool_name, result, ids.get("args") or {}, hand))
            del receipts[500:]
            if tool_name in {"web_search", "web_extract"}:
                evidence = _evidence(result)
                hand.no_new_information = 0 if evidence - hand.seen_evidence else hand.no_new_information + 1
                hand.seen_evidence.update(evidence)
                threshold = settings(runtime).get("budgets", {}).get("no_new_information_warn", 3)
                if hand.no_new_information == threshold:
                    runtime.system_steer(hand, "No new information in three searches: write up what you have.")
            if hand.activity not in WAITING:
                hand.activity = "queued"
                hand.activity_since = time.monotonic()
                hand.stall_steered_at = None
    except Exception:
        logger.exception("hand tool accounting failed")


def register(ctx: Any) -> None:
    from .dispatch import get_runtime

    def before(**kwargs: Any) -> dict | None:
        try:
            return pre_tool_call(get_runtime(ctx), **kwargs)
        except Exception:
            logger.exception("hand budget runtime unavailable")
            return block("hand budget check unavailable: tool refused") if is_child_context(**kwargs) else None

    def after(**kwargs: Any) -> None:
        try:
            post_tool_call(get_runtime(ctx), **kwargs)
        except Exception:
            logger.exception("hand budget runtime unavailable")

    ctx.register_hook("pre_tool_call", before)
    ctx.register_hook("post_tool_call", after)
