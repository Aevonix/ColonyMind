"""Reconcile old attempts with Hermes' durable ledger without executing work."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable


def _process_dependencies(registry=None, start_time=None):
    if registry is None:
        from tools.process_registry import process_registry
        registry = process_registry
    if start_time is None:
        from gateway.status import get_process_start_time
        start_time = get_process_start_time
    return registry, start_time


def record_process(hand, result, *, task_id: str, registry=None, start_time=None) -> bool:
    """Record only a returned local background handle with verified ownership."""
    try:
        if isinstance(result, str):
            result = json.loads(result)
        if not isinstance(result, dict) or not task_id or not result.get("session_id") or not result.get("pid"):
            return False
        registry, start_time = _process_dependencies(registry, start_time)
        session = registry.get(str(result["session_id"]))
        if (session is None or session.pid_scope != "host" or session.exited or
                session.owner_task_id != task_id or session.pid != result["pid"] or
                session.host_start_time is None or start_time(session.pid) != session.host_start_time):
            return False
        handles = hand.plan.setdefault("process_handles", {})
        handles[session.id] = {"pid": session.pid, "started_at": session.host_start_time,
                               "owner_task_id": task_id}
        return True
    except Exception:
        return False


def stop_recorded_processes(hand, *, registry=None, start_time=None) -> list[str]:
    """Kill recorded handles only after rechecking the kernel process identity."""
    stopped = []
    handles = hand.plan.get("process_handles", {})
    if not handles:
        return stopped
    try:
        registry, start_time = _process_dependencies(registry, start_time)
    except Exception:
        return stopped
    for session_id, expected in list(handles.items()):
        try:
            session = registry.get(session_id)
            if (session is None or session.exited or session.pid_scope != "host" or
                    session.pid != expected["pid"] or session.host_start_time != expected["started_at"] or
                    session.owner_task_id != expected["owner_task_id"] or
                    start_time(session.pid) != expected["started_at"]):
                continue
            result = registry.kill_process(session_id, source="hands.stop", consume_output=False)
            # Only the newer registry checks for survivors after signaling.
            # Preserve the older registry's established non-raising reporting.
            if callable(getattr(registry, '_post_kill_survivors', None)):
                if (not isinstance(result, dict) or result.get('status') not in {'killed','already_exited'}
                        or result.get('process_running') is True or result.get('error')):
                    continue
            stopped.append(session_id)
        except Exception:
            continue
    return stopped


def reconcile(runtime, ledger_lookup: Callable[[str], dict | None] | None = None) -> dict[str, str]:
    """Return local terminal classifications and send the frozen executor view.

    Hermes' durable lookup exposes state but not its owner process identity. The
    epoch therefore comes from our fsynced dispatch, never from the new process.
    No callback here dispatches, continues or retries a tool.
    """
    if ledger_lookup is None:
        from tools.async_delegation import get_durable_delegation
        ledger_lookup = get_durable_delegation
    attempts = {}
    for row in runtime.spool.records():
        parts = row["path"].strip("/").split("/")
        if len(parts) != 4 or parts[:2] != ["v1", "hands"]:
            continue
        hand_id, action = parts[2:]
        body = row["body"]
        if action == "dispatched":
            attempts[hand_id] = {"hand_id": hand_id, "executor_ref": body["delegation_id"],
                                 "delegation_id": body["delegation_id"], "epoch": body["epoch"],
                                 "status": "unknown", "child_session": None}
        elif action == "events" and body.get("epoch") and (body.get("effects") or any(
                event.get("type") == "dispatch_intent" for event in body.get("events", []))):
            # A child may reach its first effect before Hermes returns the
            # delegation id. A crash in that window must still surface it.
            attempts.setdefault(hand_id, {"hand_id": hand_id, "executor_ref": f"unbound:{hand_id}",
                                          "delegation_id": None, "epoch": body["epoch"],
                                          "status": "unknown", "child_session": None})
        elif hand_id in attempts and action == "bind":
            attempts[hand_id].update(executor_ref=body["executor_ref"], child_session=body["child_session"])
        elif hand_id in attempts and action in {"result", "stop"}:
            attempts[hand_id]["terminal"] = True
    outcomes, ledger, live = {}, [], []
    for hand_id, attempt in attempts.items():
        open_effects = runtime.effects.open_effects(hand_id)
        if attempt["epoch"] == runtime.epoch or (attempt.get("terminal") and not open_effects):
            continue
        durable = ledger_lookup(attempt["delegation_id"]) if attempt["delegation_id"] else None
        attempt["status"] = str((durable or {}).get("state") or "unknown")
        ledger.append({key: value for key, value in attempt.items() if key not in {"delegation_id", "terminal"}})
        outcomes[hand_id] = "unknown_effect" if open_effects else "lost"
    for hand in runtime.registry.all():
        # Only an executor Hermes accepted is live; a reserved, refused or queued hand has no executor reference
        # and would make the frozen wire refuse the whole reconciliation.
        if (hand.epoch != runtime.epoch or not (hand.subagent_id or hand.delegation_id)
                or hand.status not in {"dispatched", "running", "checking", "awaiting_input"}):
            continue
        live.append({"hand_id": hand.hand_id, "executor_ref": hand.subagent_id or hand.delegation_id,
                     "epoch": runtime.epoch, "status": hand.status,
                     "child_session": hand.child_session_id or None})
    if ledger or live:
        body = {"epoch": runtime.epoch, "live": live, "ledger": ledger}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:20]
        response = runtime.emit("/v1/hands/reconcile", body, f"reconcile:{digest}")
        for hand in (response or {}).get("hands", []):
            if hand.get("state") == "unknown_effect":
                runtime.probe_unknown(hand["id"])
    return outcomes


def register(ctx: Any) -> None:
    """Runtime calls reconcile once at startup, after ordered spool replay."""
