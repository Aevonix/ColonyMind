"""Allowlisted sidecar facts; the Hermes plugin owns final submission and alarms.

No content enters observations. The existing assembly response carries a snapshot;
post-response capture callbacks can only report their outcome separately.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import json
import logging
import time

logger = logging.getLogger(__name__)
ALARM_ENV = "COLONYMIND_RECALL_EXIT_ALARM_TURNS"
ORDER = ("no_turn_id", "not_owner", "runtime_unavailable", "circuit_open", "review",
         "binding_refused", "error", "cancelled", "deadline", "no_candidates", "host_drop", "ok")
EXITS = frozenset(ORDER)
STAGES = frozenset({"embed", "search", "owner_search", "rerank", "assembly_left", "recall"})
_POST = ContextVar("recall_exit_post_response", default=None)
_debug_at = float("-inf")
_debug_pending = False


def safe(call):
    @wraps(call)
    def guarded(*args, **kwargs):
        global _debug_pending
        try:
            return call(*args, **kwargs)
        except BaseException:
            _debug_pending = True
    return guarded


@dataclass
class Observation:
    exit: str = "ok"
    stage: str | None = None
    subcode: str | None = None
    exception: str | None = None
    recall_ms: int = 0
    budget_ms: int = 0
    sections_returned: int = 0
    rerank: bool = False
    capture: str = "none"
    turn_id: str = ""
    started: float | None = None
    emitted: bool = False


def current():
    from . import turn_budget
    budget = turn_budget.current()
    return budget.recall_exit if budget is not None else _POST.get()


@safe
def begin(owner, attempted, turn_id=""):
    from . import turn_budget
    budget = turn_budget.current()
    # A host owner can disagree with sidecar authorization; retain the sidecar's
    # not_owner fact too. The host alone decides which turns enter its streak.
    if budget is not None and budget.recall_exit is None:
        budget.recall_exit = Observation(budget_ms=round(budget.wait_s * 1000),
                                         turn_id=str(turn_id or ""))
    if budget is not None:
        if not turn_id:
            code("no_turn_id")
        if owner is False:
            code("not_owner")
        if not attempted:
            code("no_candidates")


@safe
def code(value, stage=None, subcode=None, exception=None):
    state = current()
    if state is not None and value in EXITS and ORDER.index(value) < ORDER.index(state.exit):
        state.exit = value
        state.stage = stage if stage in STAGES else None
        state.subcode = subcode if subcode == "identity" else None
        state.exception = exception


@safe
def failed(error, stage="search"):
    from . import turn_budget
    if isinstance(error, asyncio.CancelledError):
        budget = turn_budget.current()
        own = budget is not None and budget.assembly_left() <= 0
        code("deadline" if own else "cancelled", "assembly_left" if own else stage)
    elif isinstance(error, TimeoutError):
        code("deadline", stage)
    else:
        # Class only. Never inspect the message or instance attributes.
        code("error", stage, exception=type(error).__name__)


@safe
def producer_started():
    state = current()
    if state is not None:
        state.started = time.monotonic()


@safe
def producer_finished(section, completed=True):
    state = current()
    if state is not None:
        if section is None and completed:
            code("no_candidates")
        if state.started is not None:
            state.recall_ms = max(0, round((time.monotonic() - state.started) * 1000))


@safe
def queued():
    state = current()
    if state is not None:
        state.capture = "pending"


@safe
def captured(value):
    state = current()
    if state is not None and value in {"c", "d", "recorded"}:
        state.capture = value


@safe
def reranked():
    state = current()
    if state is not None:
        state.rerank = True


@safe
def accepted(sections):
    state = current()
    if state is not None:
        state.sections_returned = sum(s.id == "colonymind-memory" and bool(s.body) for s in sections)
        if not state.sections_returned and state.capture == "pending":
            state.capture = "c"


@safe
def snapshot(state):
    if state is None:
        return None
    return {key: getattr(state, key) for key in (
        "exit", "stage", "subcode", "exception", "recall_ms", "budget_ms",
        "sections_returned", "rerank", "capture", "turn_id")}


@safe
def bind_post(state):
    return _POST.set(state)


@safe
def reset_post(token):
    if token is not None:
        _POST.reset(token)


@safe
def emit(state):
    # There is deliberately no sidecar RECALL_EXIT line or sidecar streak.
    if state is not None and not state.emitted:
        state.emitted = True
        if state.capture in {"c", "d", "recorded"} and state.exit != "not_owner":
            logger.info("RECALL_CAPTURE turn_id=%s capture=%s", json.dumps(state.turn_id, ensure_ascii=True), state.capture)


@safe
def finish_post(state, token):
    global _debug_at, _debug_pending
    try:
        emit(state)
    finally:
        reset_post(token)
        if _debug_pending and time.monotonic() - _debug_at >= 300:
            _debug_pending = False
            _debug_at = time.monotonic()
            logger.debug("RECALL_EXIT_COUNTER_FAILED")
