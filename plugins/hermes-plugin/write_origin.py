"""Host-only call provenance and one-use owner actions for gateway tools."""
from collections import OrderedDict
from contextvars import ContextVar
import hashlib
from inspect import getclosurevars
import json
import logging
import os
import re
import threading
import time

from colonymind_shared.secret_detection import hide_secrets
from colonymind_shared.write_origin import owner_started_turn

_CALL = ContextVar("colonymind_write_call", default=None)
logger = logging.getLogger(__name__)
ADDRESS_CAP = 256
CONTACT_SECONDS = 1.0
_EMAIL = re.compile(r"[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,63}")


def _emails(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return (m.group().lower() for m in _EMAIL.finditer(text))


def _hash(tool, args):
    raw = json.dumps({"tool": tool, "args": args}, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _current():
    frame = _CALL.get()
    return frame if frame and frame["active"] and frame["turn"].get("active") else None


def call_origin() -> dict | None:
    """A copy of the current pre-call stamp; absent outside a running tool."""
    frame = _current()
    return dict(frame["stamp"]) if frame else None


def untrusted_addresses() -> set[str]:
    """This turn's result addresses, less owner words and known contacts."""
    frame = _current()
    return frame["bridge"].addresses(frame["turn"]) if frame else set()


def request_owner_action(tool, args, reason) -> dict:
    """Ask once for this exact current call; consume a verified yes once."""
    frame = _current()
    if not frame or tool != frame["stamp"]["tool"]:
        return {"state": "denied"}
    try:
        if _hash(tool, args) != frame["stamp"]["args_hash"]:
            return {"state": "denied"}
        return frame["bridge"].request(tool, args, frame["stamp"]["args_hash"])
    except Exception:
        return {"state": "denied"}


class WriteOrigin:
    def __init__(self, guard):
        self.guard, self.turns = guard, OrderedDict()
        self.lock = threading.RLock()

    def register(self, tool, args):
        try:
            from tools.registry import registry
            entry = registry.get_entry(tool)
            if entry is None or not entry.toolset.startswith("mcp-"):
                return
            # Hermes handlers close over original names, including sanitized/clamped ones.
            names = getclosurevars(entry.handler).nonlocals
            server, bare = names.get("server_name"), names.get("tool_name")
            if (server not in getattr(self.guard.settings, "tool_origin_servers", ("google",)) or not bare
                    or entry.toolset != "mcp-" + server):
                return
            stamp = call_origin()
            if stamp is None:
                raise ValueError("missing_tool_origin")
            payload = {key: value for key, value in stamp.items() if key != "verified"}
            payload.update(tool=bare, args_hash=_hash(bare, args),
                           untrusted_addresses=sorted(untrusted_addresses()))
            response = self.guard.client.post("/v1/mind/tool-origins", timeout=0.5, json=payload)
            response.raise_for_status()
        except Exception as error:
            logger.warning("tool-origin registration failed (%s)", type(error).__name__)
            self.clear_call()
            return {"action": "block", "message": "tool-origin registration unavailable; retry: false"}

    def begin(self, session_id, turn_id):
        if not session_id or not turn_id:
            return {"active": True, "seen": set(), "owner": set(), "contacts": {}}
        key = (session_id, turn_id)
        with self.lock:
            if key not in self.turns:
                self.turns[key] = {"active": True, "seen": set(), "owner": set(), "contacts": {}}
            self.turns.move_to_end(key)
            while len(self.turns) > 128:
                self.turns.popitem(last=False)[1].clear()
            return self.turns[key]

    def end(self, session_id, turn_id):
        with self.lock:
            state = self.turns.pop((session_id, turn_id), None)
            if state is not None:
                state.clear()
        frame = _CALL.get()
        if frame and frame["stamp"]["session_id"] == session_id and frame["stamp"]["turn_id"] == turn_id:
            self.clear_call()

    @staticmethod
    def clear_call():
        frame = _CALL.get()
        if frame:
            frame["active"] = False
        _CALL.set(None)

    def enter(self, tool, args, session_id, turn_id):
        self.clear_call()
        info = self.guard.sessions.get(session_id)
        turn_id = turn_id or getattr(info, "turn_id", "")
        state = self.begin(session_id, turn_id)
        _CALL.set({"bridge": self, "turn": state, "active": True, "stamp": {
            "origin": "other", "channel": "", "turn_id": turn_id, "session_id": session_id,
            "tool": tool, "args_hash": _hash(tool, args), "verified": False}})

    def resolve(self, run, platform, session_id, task_id, hand):
        frame = _CALL.get()
        if not frame or frame["bridge"] is not self:
            return
        stamp = frame["stamp"]
        info = self.guard.sessions.get(session_id)
        stamp.update(channel=platform, verified=run in {"owner", "guest", "mind", "system"})
        from .capture import session_env, owner_input_text, genuine
        bound_platform, bound_sender, _, _ = session_env()
        own_message = info and (not bound_sender or (bound_sender == info.sender_id and bound_platform == info.platform))
        if run == "owner" and not hand:
            worker = os.environ.get("HERMES_KANBAN_TASK")
            job = self.guard._cron_job(session_id, task_id) if platform == "cron" else ""
            saved = self.guard.sessions.standings.get("kanban" if worker else "cron", worker or job)
            if saved and saved[0] == "owner":
                stamp.update(origin="autonomy", run_id=task_id or session_id)
            elif not worker and own_message and owner_started_turn(session_id, stamp["turn_id"], info, genuine):
                stamp["origin"] = "owner_channel"
        if stamp["origin"] == "owner_channel" and own_message and info.turn_id == stamp["turn_id"]:
            words = owner_input_text(session_id, info.turn_id, info.said or info.user_message)
            frame["turn"]["owner"].update(list(_emails(words or ""))[:ADDRESS_CAP])

    def result(self, result, session_id, turn_id):
        frame = _CALL.get()
        if not turn_id and frame and frame["stamp"]["session_id"] == session_id:
            turn_id = frame["stamp"]["turn_id"]
        with self.lock:
            state = self.turns.get((session_id, turn_id))
            if state is None:
                return
            for address in _emails(result):
                if len(state["seen"]) >= ADDRESS_CAP and address not in state["seen"]:
                    state["overflow"] = True
                    break
                state["seen"].add(address)

    def addresses(self, state):
        deadline = time.monotonic() + CONTACT_SECONDS
        with self.lock:
            if state.get("overflow"):
                return {"*"}
            result = set()
            for address in sorted(state["seen"] - state["owner"]):
                if address not in state["contacts"]:
                    known = False
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        try:
                            known = bool(self.guard.client.resolve_contact("email", address,
                                         timeout=min(0.2, remaining)))
                        except Exception:
                            pass
                    state["contacts"][address] = known
                if not state["contacts"][address]:
                    result.add(address)
            return result

    def request(self, tool, args, digest):
        # Never serialize body, headers, attachments, arbitrary reason or other arguments to the ask service.
        recipients = sorted(set(_emails({key: args[key] for key in
            ("to", "cc", "bcc", "recipients", "attendees") if key in args})))
        subject = hide_secrets(str(args.get("subject", args.get("title", ""))))[:300]
        response = self.guard.client.post("/v1/mind/owner-actions", timeout=2, json={
            "tool": tool, "args_hash": digest, "summary": {"recipients": recipients,
            "subject_or_title": subject}, "reason": "this external write needs the owner's yes"})
        response.raise_for_status()
        return response.json()
