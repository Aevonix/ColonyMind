"""Bounded, process-local, single-claim attestations from the gateway adapter."""
from collections import OrderedDict
from copy import deepcopy
import re
import threading
import time

TTL_S, CAPACITY = 120, 4096


def key(body):
    if (not isinstance(body, dict) or not isinstance(body.get("tool"), str)
            or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", body["tool"])
            or not isinstance(body.get("args_hash"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", body["args_hash"])):
        raise ValueError("invalid_tool_origin")
    return body["tool"], body["args_hash"]


class ToolOrigins:
    def __init__(self, clock=time.monotonic, capacity=CAPACITY):
        self.clock, self.capacity = clock, capacity
        self.records, self.lock = OrderedDict(), threading.Lock()

    def register(self, body):
        binding = key(body)
        with self.lock:
            now = self.clock()
            self._expire(now)
            previous = self.records.pop(binding, None)
            # Pop before validation: a rejected replacement cannot leave owner authority behind.
            value = self._value(body)
            if previous and (previous[1] != value or previous[2] != body["session_id"]):
                # Stdio claims cannot identify an invocation. Conflicting live calls need a tap.
                value.update(origin="other", untrusted_addresses=["*"])
                value.pop("run_id", None)
            self.records[binding] = (now + TTL_S, deepcopy(value), body["session_id"])
            while len(self.records) > self.capacity:
                self.records.popitem(last=False)

    @staticmethod
    def _value(body):
        if (set(body) - {"tool", "args_hash", "origin", "channel", "turn_id", "session_id", "run_id",
                         "untrusted_addresses"}
                or not {"origin", "channel", "turn_id", "session_id", "untrusted_addresses"} <= set(body)
                or body["origin"] not in ("owner_channel", "autonomy", "other")):
            raise ValueError("invalid_tool_origin")
        for field in ("channel", "turn_id", "session_id", "run_id"):
            if field in body and (not isinstance(body[field], str) or len(body[field]) > 256
                                  or any(ord(c) < 32 for c in body[field])):
                raise ValueError("invalid_tool_origin")
        if (body["origin"] == "autonomy") != bool(body.get("run_id")):
            raise ValueError("invalid_tool_origin")
        addresses = body["untrusted_addresses"]
        if (not isinstance(addresses, list) or len(addresses) > 256
                or any(not isinstance(item, str) or len(item) > 254
                       or (item != "*" and not re.fullmatch(r"[^\s<>@]+@[^\s<>@]+\.[^\s<>@]+", item))
                       for item in addresses)):
            raise ValueError("invalid_tool_origin")
        value = {field: body[field] for field in ("origin", "channel", "turn_id", "run_id") if field in body}
        value.update(verified=True, untrusted_addresses=["*"] if "*" in addresses else
                     sorted(set(item.lower() for item in addresses)))
        return value

    def _expire(self, now):
        while self.records and next(iter(self.records.values()))[0] <= now:
            self.records.popitem(last=False)

    def claim(self, body):
        binding = key(body)
        if set(body) != {"tool", "args_hash"}:
            raise ValueError("invalid_tool_origin")
        with self.lock:
            self._expire(self.clock())
            record = self.records.pop(binding, None)
            return deepcopy(record[1]) if record else None
