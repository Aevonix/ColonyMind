"""Companion device reports and exact-call invocations, using the mind's permits.

No transport or native execution lives here. The authenticated phone relay reports
OS state, claims an authorized invocation once, and returns its native result.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
import threading
from typing import Annotated, Any, Literal

from fastapi import HTTPException
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from colonymind.mind.authority import Effect
from colonymind.mind.outcomes import owner_answer
from colonymind.redact import redact_sensitive_text

TOOL = "colonymind_device_invoke"
DEVICE = "device.registry:"
INVOCATION = "device.invocation:"
LOCK = threading.RLock()
Id = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}$")]
Capability = Literal["health", "alarms", "home"]
Method = Literal["read", "write"]
Mode = Literal["allow", "ask", "deny"]
OSGrant = Literal["unknown", "not_determined", "granted", "limited", "denied", "restricted", "unsupported"]
COMMANDS = {
    "health.read": ("health", "read"), "health.write": ("health", "write"),
    "alarm.list": ("alarms", "read"),
    **{f"alarm.{action}": ("alarms", "write") for action in ("set", "update", "dismiss", "dismiss_all")},
    "home.read": ("home", "read"), "home.execute": ("home", "write"),
}
TERMINAL = {"succeeded", "failed", "timed_out", "denied", "cancelled"}
_PRIVATE_KEY = re.compile(r"secret|token|password|credential|authorization|phone|bearer|api.?key", re.I)


def fail(status, code):
    raise HTTPException(status, detail={"code": code})


def dumps(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def private(value):
    """Reject credential fields and recognized sensitive text before persistence or approval display."""
    if isinstance(value, dict):
        return any(_PRIVATE_KEY.search(key) or private(item) for key, item in value.items())
    if isinstance(value, list):
        return any(private(item) for item in value)
    return isinstance(value, str) and redact_sensitive_text(value) != value


class Wire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def safe_payload(self):
        value = self.model_dump(mode="json")
        if len(dumps(value).encode()) > 65536 or private(value):
            raise ValueError("device_payload_refused")
        return self


class CapabilityReport(Wire):
    capability: Capability
    os_grants: dict[Method, OSGrant]
    commands: list[str] = Field(max_length=16)

    @model_validator(mode="after")
    def commands_match(self):
        if len(set(self.commands)) != len(self.commands) or any(
            COMMANDS.get(command, (None,))[0] != self.capability for command in self.commands
        ):
            raise ValueError("unsupported_device_command")
        return self


class DeviceReport(Wire):
    display_name: str = Field(min_length=1, max_length=80)
    platform: Literal["ios"]
    app_version: str = Field(min_length=1, max_length=32)
    state: Literal["online", "offline", "error", "disconnected"]
    capabilities: list[CapabilityReport] = Field(max_length=3)

    @model_validator(mode="after")
    def unique_capabilities(self):
        if len({item.capability for item in self.capabilities}) != len(self.capabilities):
            raise ValueError("duplicate_device_capability")
        return self


class GrantEdit(Wire):
    expected_revision: int = Field(ge=1)
    method: Method
    mode: Mode


class Invoke(Wire):
    id: Id
    device_id: Id
    capability: Capability
    command: str = Field(max_length=64)
    parameters: dict[str, Any] = Field(default_factory=dict)
    deadline: AwareDatetime

    @model_validator(mode="after")
    def command_matches(self):
        if COMMANDS.get(self.command, (None,))[0] != self.capability:
            raise ValueError("unsupported_device_command")
        return self


class Result(Wire):
    ok: bool
    payload: dict[str, Any] | None = None
    error: Id | None = None
    card: dict[str, Any] | None = None

    @model_validator(mode="after")
    def error_matches(self):
        if self.ok == (self.error is not None):
            raise ValueError("result_requires_error_on_failure_only")
        return self


class Devices:
    def __init__(self, mind):
        self.mind = mind
        self.state = mind.mind_state

    def _get(self, prefix, ident):
        value = self.state.get(prefix + ident)
        if value is None:
            fail(404, "unknown_device" if prefix == DEVICE else "unknown_device_invocation")
        return json.loads(value["text"])

    def _save(self, prefix, value):
        if prefix == INVOCATION and value["state"] in {"cancelled", "denied", "timed_out"} \
                and value["approval"] and not value.get("dispatched_at"):
            ask = self.mind.store.get(value["approval"])
            if ask and ask.status in {"asked", "approved", "dispatched"}:
                self.mind.outcomes.record(ask.id,
                    status="expired" if value["state"] == "timed_out" else "cancelled", by="mind",
                    implicit_verdict=False, summary="the device invocation is no longer executable",
                    details={"device_invocation": value["id"], "state": value["state"]})
        self.state.set(prefix + value["id"], text=dumps(value), now=self.mind.clock())
        return value

    def list_devices(self):
        return [json.loads(item["text"]) for item in self.state.items(DEVICE)]

    def device(self, ident):
        return self._get(DEVICE, ident)

    def report(self, ident, body):
        with LOCK:
            prior = self.state.get(DEVICE + ident)
            old = json.loads(prior["text"]) if prior else {}
            value = {"id": ident, **body.model_dump(mode="json")}
            for item in value["capabilities"]:
                before = next((c for c in old.get("capabilities", []) if c["capability"] == item["capability"]), {})
                item["authorization"] = before.get("authorization", {"read": "ask", "write": "ask"})
            # Repeated reports refresh presence without changing approval bindings.
            changed = any(value[key] != old.get(key) for key in value)
            value.update(revision=old.get("revision", 0) + int(changed), reported_at=self.mind.clock().isoformat())
            return self._save(DEVICE, value)

    def grant(self, ident, capability, body):
        with LOCK:
            value = self.device(ident)
            if value["revision"] != body.expected_revision:
                fail(409, "device_revision_changed")
            item = self._capability(value, capability)
            item["authorization"][body.method] = body.mode
            value["revision"] += 1
            return self._save(DEVICE, value)

    @staticmethod
    def _capability(device, capability):
        item = next((c for c in device["capabilities"] if c["capability"] == capability), None)
        if item is None:
            fail(404, "unknown_device_capability")
        return item

    def _availability(self, row):
        device = self.device(row["device_id"])
        capability = self._capability(device, row["capability"])
        method = COMMANDS[row["command"]][1]
        if device["revision"] != row["device_revision"]:
            return "cancelled"
        if capability["authorization"][method] == "deny":
            return "denied"
        if device["state"] != "online" or row["command"] not in capability["commands"]:
            return "unavailable"
        if capability["os_grants"].get(method) != "granted":
            return "os_permission_required"
        return None

    def _authorize(self, row):
        args = {key: row[key] for key in ("id", "device_id", "capability", "command", "parameters", "deadline", "device_revision")}
        # Use the existing off switch and deny rules before considering the owner's device grant.
        verdict = self.mind.authority.guard(tool=TOOL, args=args, run="mind")
        if verdict.get("action") == "block":
            return verdict
        if row["mode"] == "allow":
            return {"allow": True, "action": "allow", "reason": "the owner's device capability grant"}
        # Approval terms reach the audit and autobiography logs. Bind opaque
        # parameters by digest so those records never receive their values.
        binding = {key: row[key] for key in ("id", "device_id", "capability", "command")}
        binding["parameters_digest"] = hashlib.sha256(dumps(row["parameters"]).encode()).hexdigest()
        # Include every original approval term (also deadline and revision) in
        # the identity without copying any further invocation data into logs.
        call = dumps({"tool": TOOL, "args": binding,
                      "invocation_digest": hashlib.sha256(dumps(args).encode()).hexdigest()})
        effect = Effect("unbounded", TOOL, call=call,
                        problem="this device capability asks for each exact invocation")
        ask = self.mind.store.get(row["approval"]) if row["approval"] else None
        # The same owner-answer evidence as Hands: an operator or a claimed yes grants nothing.
        if ask is not None and not any(h.action == "queued" and owner_answer(
            (h.details or {}).get("by"), h.details) is True and (h.details or {}).get("answer", {}).get("message")
            for h in self.mind.store.get_history(ask.id, limit=200)):
            ask = None
        verdict = self.mind.authority.effect_verdict(effect, task=ask)
        if verdict.get("hold") and not row["approval"]:
            held = self.mind._hold(None, verdict, task_id=row["id"], now=self.mind.clock(), origin="hand")
            row["approval"] = held["ask"]
            self.mind.store.update(held["ask"], expires_at=datetime.fromisoformat(row["deadline"]))
        return verdict

    def request(self, body):
        with LOCK:
            requested = body.model_dump(mode="json")
            requested["deadline"] = body.deadline.astimezone(timezone.utc).isoformat()
            prior = self.state.get(INVOCATION + body.id)
            if prior:
                row = json.loads(prior["text"])
                if any(row[key] != value for key, value in requested.items()):
                    fail(409, "invocation_id_conflict")
                return self.invocation(body.id)
            seconds = (body.deadline - self.mind.clock()).total_seconds()
            if not 0 < seconds <= 3600:
                fail(422, "invocation_deadline_out_of_range")
            device = self.device(body.device_id)
            capability = self._capability(device, body.capability)
            row = {**requested, "device_revision": device["revision"], "created_at": self.mind.clock().isoformat(),
                   "mode": capability["authorization"][COMMANDS[body.command][1]],
                   "approval": None, "result": None, "state": "requested"}
            # Persist before asking. A crash cannot silently create a second command.
            self._save(INVOCATION, row)
            return self.invocation(body.id)

    def invocation(self, ident):
        with LOCK:
            row = self._get(INVOCATION, ident)
            if row["state"] in TERMINAL:
                return row
            if row["state"] == "dispatched":
                if datetime.fromisoformat(row["deadline"]) <= self.mind.clock():
                    row["state"] = "timed_out"
                    return self._save(INVOCATION, row)
                return row
            if datetime.fromisoformat(row["deadline"]) <= self.mind.clock():
                row["state"] = "timed_out"
            else:
                try:
                    blocked = self._availability(row)
                except HTTPException as error:
                    if error.status_code != 404:
                        raise
                    blocked = "cancelled"
                if blocked:
                    row["state"] = blocked
                elif row["state"] != "ready":
                    verdict = self._authorize(row)
                    if verdict.get("allow"):
                        row["state"] = "ready"
                    elif verdict.get("action") == "block":
                        row["state"] = "denied"
                    else:
                        ask = self.mind.store.get(row["approval"]) if row["approval"] else None
                        row["state"] = ("denied" if ask and ask.status in {"denied", "cancelled", "failed"}
                                        else "awaiting_approval")
            return self._save(INVOCATION, row)

    def invocations(self, device_id):
        self.device(device_id)
        return [self.invocation(json.loads(item["text"])["id"]) for item in self.state.items(INVOCATION)
                if json.loads(item["text"])["device_id"] == device_id]

    def claim(self, device_id, ident):
        with LOCK:
            row = self._get(INVOCATION, ident)
            if row["device_id"] != device_id:
                fail(404, "unknown_device_invocation")
            row = self.invocation(ident)
            # Readiness spends no authority. Revalidate the exact, unspent permit
            # (including its configured expiry), off/deny rules and owner undo.
            if row["state"] == "ready":
                verdict = self._authorize(row)
                if not verdict.get("allow"):
                    row["state"] = "denied"
                    self._save(INVOCATION, row)
            if row["state"] != "ready":
                fail(409, "invocation_not_ready")
            # Consume only at the dispatch boundary, under the same claim lock.
            self.mind._allowed(TOOL, {}, verdict, None, [], task_id=ident, now=self.mind.clock())
            row.update(state="dispatched", dispatched_at=self.mind.clock().isoformat())
            return self._save(INVOCATION, row)

    def result(self, device_id, ident, body):
        with LOCK:
            row = self._get(INVOCATION, ident)
            if row["device_id"] != device_id:
                fail(404, "unknown_device_invocation")
            value = body.model_dump(mode="json")
            if row["result"] is not None:
                if row["result"] != value:
                    fail(409, "invocation_result_conflict")
                return row
            if row["state"] != "dispatched" and not (row["state"] == "timed_out" and row.get("dispatched_at")):
                fail(409, "invocation_not_dispatched")
            late = datetime.fromisoformat(row["deadline"]) <= self.mind.clock()
            row.update(result=value, completed_at=self.mind.clock().isoformat(), late=late,
                       state="timed_out" if late else "succeeded" if body.ok else "failed")
            return self._save(INVOCATION, row)
