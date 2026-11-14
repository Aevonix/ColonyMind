"""Schema-only external-write asks and single-use settlement receipts."""
from datetime import datetime, timezone
import json
import re
import threading

from colonymind.api.auth import request_authority, speaks_for_owner
from colonymind_shared.secret_detection import hide_secrets


def _terms(body):
    from .tool_origins import key
    key(body)
    if (set(body) != {"tool", "args_hash", "summary", "reason"}
            or not isinstance(body["summary"], dict)
            or set(body["summary"]) != {"recipients", "subject_or_title"}
            or not isinstance(body["reason"], str) or len(body["reason"]) > 300):
        raise ValueError("invalid_owner_action")
    recipients = body["summary"]["recipients"]
    if (not isinstance(recipients, list) or len(recipients) > 256
            or any(not isinstance(address, str) or len(address) > 254 or
           not re.fullmatch(r"[^\s<>@]+@[^\s<>@]+\.[^\s<>@]+", address) or hide_secrets(address) != address
           for address in recipients)):
        raise ValueError("invalid_owner_action")
    subject = body["summary"]["subject_or_title"]
    if not isinstance(subject, str) or len(subject) > 300:
        raise ValueError("invalid_owner_action")
    terms = {"recipients": sorted(set(address.lower() for address in recipients)),
             "subject_or_title": hide_secrets(subject)}
    return terms


def create(mind, body, principal):
    terms = _terms(body)
    call = json.dumps({"tool": body["tool"], "args": terms, "targets": []}, sort_keys=True)
    held = mind._hold(None, {"effect": {"cls": "external_write", "tool": body["tool"], "call": call,
        "digest": body["args_hash"], "text": body["tool"] + ": " + json.dumps(terms, ensure_ascii=False)},
        "reason": "this external write needs the owner's yes"}, task_id="owner-action:" + principal,
        now=mind.clock(), origin="hand")
    row = mind.store.get(held["ask"])
    mind.store.update(row.id, context={**row.context, "owner_action_principal": principal})
    return {"ask_id": row.id}


class OwnerActions:
    """Move the adapter's ask/poll/consume path next to the existing hold ledger."""
    def __init__(self, mind):
        self.mind, self.records, self.lock = mind, {}, threading.RLock()

    def request(self, body, principal):
        from .permit_asks import view
        from .tool_origins import key
        _terms(body)
        binding = (principal, *key(body))
        with self.lock:
            record = self.records.get(binding)
            if record is None:
                if len(self.records) >= 1024:
                    return {"state": "denied", "ask_id": None}
                result = create(self.mind, body, principal)
                record = self.records[binding] = {"ask_id": result["ask_id"], "used": False}
                return {"state": "asked", "ask_id": record["ask_id"]}
            ident, digest = record["ask_id"], body["args_hash"]
            denied = {"state": "denied", "ask_id": ident}
            row = self.mind.store.get(ident)
            if row is None or row.context.get("owner_action_principal") != principal:
                return denied
            value = view(self.mind.store, row, now=self.mind.clock())
            if (value.get("id") != ident or value.get("terms", {}).get("call_digest") != digest
                    or value.get("terms", {}).get("call", {}).get("tool") != body["tool"]):
                return denied
            if value.get("open") is True:
                return {"state": "asked", "ask_id": ident}
            winner = value.get("winner") or {}
            if winner.get("choice") != "yes" or record["used"] or value.get("status") != "approved":
                return denied
            try:
                settled = datetime.fromisoformat(winner["at"].replace("Z", "+00:00"))
                age = (self.mind.clock().astimezone(timezone.utc) - settled).total_seconds()
            except (ValueError, TypeError, KeyError):
                return denied
            if not 0 <= age < 600:
                return denied
            record["used"] = True
            return {"state": "approved", "ask_id": ident, "owner_action": {
                "ask_id": ident, "args_hash": digest, "settled_at": winner["at"]}}


def may_read(request, mind):
    """Only this creator's external-write asks; never lists, captures or answers."""
    from .permit_asks import is_permit_ask
    authority = request_authority(request)
    prefix = "/v1/mind/permit-asks/"
    path = request.url.path
    if (request.method != "GET" or not path.startswith(prefix) or "/" in path[len(prefix):]
            or not speaks_for_owner(request) or "manage" not in authority.access or authority.scopes is not None
            or mind is None):
        return False
    row = mind.store.get(path[len(prefix):])
    return is_permit_ask(row) and row.context.get("owner_action_principal") == authority.principal_id
