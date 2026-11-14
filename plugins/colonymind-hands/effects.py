"""Write an effect intent before the tool and its observed outcome afterwards."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import threading
from typing import Any


EFFECT_KINDS = {"coding", "ops", "integrate"}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


class EffectJournal:
    def __init__(self, spool, epoch: dict):
        self.spool, self.epoch = spool, deepcopy(epoch)
        self._lock = threading.RLock()
        self._rejected_duplicates: set[str] = set()

    def _effects(self, hand_id: str) -> dict[str, dict]:
        effects = {}
        for row in self.spool.records():
            if row["path"] == f"/v1/hands/{hand_id}/events":
                for effect in row["body"].get("effects", []):
                    effects[effect["key"]] = deepcopy(effect)
        return effects

    def get(self, hand_id: str, tool_call_id: str) -> dict | None:
        with self._lock:
            return self._effects(hand_id).get(f"eff:{hand_id}:{tool_call_id}")

    def call(self, hand_id: str, tool_call_id: str) -> dict | None:
        key = f"eff:{hand_id}:{tool_call_id}:intent"
        for row in self.spool.records():
            if row["key"] == key:
                return deepcopy(row.get("metadata", {}).get("call"))
        return None

    def open_effects(self, hand_id: str) -> list[dict]:
        with self._lock:
            return [item for item in self._effects(hand_id).values() if not item.get("done_at")]

    def has_result_to_review(self, hand_id: str) -> bool:
        """Keep observed work visible; a plain workdir observation adds no result.

        This only selects notification handling. It grants no tool authority.
        Unknown calls and failed or incomplete observations always need review.
        """
        with self._lock:
            blocked = {row['key'].removesuffix(':outcome') for row in self.spool.records()
                       if row['path'] == f'/v1/hands/{hand_id}/events'
                       and row['key'].endswith(':outcome') and row.get('metadata', {}).get('status') == 'blocked'}
            for effect in self._effects(hand_id).values():
                if effect['key'] in blocked and effect.get('done_at'):
                    continue  # Hermes vetoed the call before executing it.
                prefix = f"eff:{hand_id}:"
                call = self.call(hand_id, effect['key'][len(prefix):])
                if (effect.get('outcome') != 'ok' or not effect.get('done_at')
                        or not isinstance(call, dict) or call.get('tool') != 'terminal'
                        or (call.get('arguments') or {}).get('command', '').strip() != 'pwd'):
                    return True
            return False

    def intent(self, hand_id: str, tool_call_id: str, tool: str, args: dict,
               workdir: str = "", *, reject_duplicate: bool = False) -> dict:
        if not tool_call_id:
            raise ValueError("an effect requires a tool call id")
        with self._lock:
            key = f"eff:{hand_id}:{tool_call_id}"
            digest = _digest({"tool": tool, "arguments": args})
            existing = self.get(hand_id, tool_call_id)
            if existing is not None:
                if reject_duplicate or existing["call_hash"] != digest:
                    self._rejected_duplicates.add(key)
                    raise ValueError("effect already journaled; reconcile before retrying")
                return self._record(hand_id, existing, "intent")
            effect = {"key": key, "tool": tool, "call_hash": digest, "workdir": workdir,
                      "intent_at": _timestamp(), "outcome": "unknown", "exit_code": None,
                      "result_hash": None, "done_at": None}
            return self._record(hand_id, effect, "intent", metadata={
                "call": {"tool": tool, "arguments": args, "workdir": workdir}})

    def _record(self, hand_id: str, effect: dict, phase: str, *, metadata=None) -> dict:
        # The effect entity keeps its stable key; each HTTP mutation needs a
        # distinct key so the outcome does not replay the intent's response.
        return self.spool.append(f"/v1/hands/{hand_id}/events",
                                 {"epoch": self.epoch, "events": [], "effects": [effect]},
                                 f"{effect['key']}:{phase}", metadata=metadata)

    def outcome(self, hand_id: str, tool_call_id: str, *, status: str = "success",
                result: Any = None, exit_code: int | None = None) -> dict | None:
        with self._lock:
            if status == "blocked" and f"eff:{hand_id}:{tool_call_id}" in self._rejected_duplicates:
                return None
            effect = self.get(hand_id, tool_call_id)
            if effect is None:
                return None
            if effect.get("done_at"):
                return None
            if exit_code is None and isinstance(result, dict):
                exit_code = result.get("exit_code")
            uncertain = status in {"cancelled", "timeout", "unknown", "abandoned"}
            effect.update(outcome="unknown" if uncertain else (
                "ok" if status in {"success", "ok", "completed"} and exit_code in {None, 0} else "error"),
                done_at=None if uncertain else _timestamp(), exit_code=exit_code,
                result_hash=_digest(result))
            return self._record(hand_id, effect, "outcome", metadata={"status": status})


def register(ctx: Any) -> None:
    def pre_tool_call(tool_name="", args=None, session_id="", tool_call_id="", **kwargs):
        hand = None
        try:
            from .dispatch import get_runtime
            runtime = get_runtime(ctx)
            hand = runtime.registry.for_session(session_id, task_id=kwargs.get("task_id"))
            if hand is None or hand.kind not in EFFECT_KINDS:
                return None
            if tool_call_id in getattr(hand, "refused_calls", set()):
                return {"action": "block", "message": "hand tool call was refused"}
            if hand.blocked:
                return {"action": "block", "message": "hand is blocked"}
            arguments = args if isinstance(args, dict) else {}
            workdir = str(arguments.get("workdir") or arguments.get("cwd") or hand.plan.get("workdir") or "")
            row = runtime.effects.intent(hand.hand_id, tool_call_id, tool_name, arguments,
                                         workdir, reject_duplicate=True)
            runtime.emit(row["path"], row["body"], row["key"])
        except Exception:
            if hand is not None:
                hand.blocked = True
                try:
                    runtime.system_stop(hand, "cannot record effect safely")
                except Exception:
                    pass
                return {"action": "block", "message": "cannot record effect safely"}
        return None

    def post_tool_call(tool_name="", args=None, result=None, session_id="", tool_call_id="",
                       status="success", **kwargs):
        hand = None
        try:
            from .dispatch import get_runtime
            runtime = get_runtime(ctx)
            hand = runtime.registry.for_session(session_id, task_id=kwargs.get("task_id"))
            if hand is None or hand.kind not in EFFECT_KINDS:
                return
            if tool_name == "terminal":
                from .reconcile import record_process
                record_process(hand, result, task_id=str(kwargs.get("task_id") or ""))
            row = runtime.effects.outcome(hand.hand_id, tool_call_id, status=status or "success", result=result)
            if row:
                runtime.emit(row["path"], row["body"], row["key"])
        except Exception:
            if hand is not None:
                hand.blocked = True
                try:
                    runtime.system_stop(hand, "cannot record effect outcome safely")
                except Exception:
                    pass

    ctx.register_hook("pre_tool_call", pre_tool_call)
    ctx.register_hook("post_tool_call", post_tool_call)
