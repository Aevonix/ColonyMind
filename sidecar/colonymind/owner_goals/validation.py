"""Strict wire admission for the owner-goals contract."""
from __future__ import annotations

from datetime import datetime
import json
import math
import re
import unicodedata
import uuid

MAX_INTEGER = 9007199254740991
REQUEST_BYTES = 16384
RESPONSE_BYTES = 1048576
TIERS = ("Public", "Everyday", "Personal", "Private", "Secret")
ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z", re.ASCII)
STAMP = re.compile(
    r"([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})"
    r"(?:\.([0-9]+))?(Z|[+-][0-9]{2}:[0-9]{2})\Z", re.ASCII)


class GoalError(Exception):
    def __init__(self, status: int, code: str):
        self.status, self.code = status, code
        super().__init__(code)


def invalid():
    raise GoalError(422, "invalid_request")


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def loads(raw: bytes):
    if len(raw) > REQUEST_BYTES:
        invalid()

    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                invalid()
            result[key] = value
        return result

    try:
        body = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=lambda _: invalid())
        # Validate Unicode scalars even in keys and strings not used by a schema.
        dumps(body).encode("utf-8")
    except (ValueError, UnicodeError, RecursionError):
        invalid()
    if not isinstance(body, dict):
        invalid()
    return body


def fields(body, required, optional=()):
    if not isinstance(body, dict) or set(body) - set(required) - set(optional) or set(required) - set(body):
        invalid()


def text(value, maximum, *, blank=True, multiline=False):
    if not isinstance(value, str) or len(value) > maximum or (not blank and not value.strip()):
        invalid()
    for char in value:
        if (unicodedata.category(char) in {"Cc", "Cs"}
                and not (multiline and char in "\n\t")):
            invalid()
    return value


def identifier(value, *, job=False, goal=False):
    if not isinstance(value, str) or not ID.fullmatch(value) or (job and len(value) > 64):
        invalid()
    if goal and value == "proposals":
        invalid()
    return value


def integer(value):
    if type(value) is not int or not 0 <= value <= MAX_INTEGER:
        invalid()
    return value


def operation(value):
    try:
        if not isinstance(value, str) or str(uuid.UUID(value)) != value:
            invalid()
    except (ValueError, AttributeError):
        invalid()
    return value


def tier(value):
    if value not in TIERS:
        invalid()
    return value


def timestamp(value):
    text(value, 64, blank=False)
    match = STAMP.fullmatch(value)
    if match is None:
        invalid()
    date, hour, minute, second, fraction, offset = match.groups()
    h, m, s = int(hour), int(minute), int(second)
    if h > 23 or m > 59 or s > 59:
        invalid()
    delta = 0
    if offset != "Z":
        oh, om = int(offset[1:3]), int(offset[4:6])
        if oh > 23 or om > 59:
            invalid()
        delta = (oh * 3600 + om * 60) * (1 if offset[0] == "+" else -1)
    try:
        day = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        invalid()
    # A sortable exact UTC instant, including fractions beyond microseconds and year boundaries.
    seconds = day.toordinal() * 86400 + h * 3600 + m * 60 + s - delta
    return f"{seconds:012d}." + (fraction or "").rstrip("0")


def source(body):
    if "source_id" in body:
        text(body["source_id"], 256, blank=False)


def create(body, *, provenance=True):
    fields(body, ("title", "description", "category"),
           ("parent_id", "muted", "sensitivity", *(["source_id"] if provenance else [])))
    result = {"title": text(body["title"], 200, blank=False),
              "description": text(body["description"], 8000, multiline=True),
              "category": text(body["category"], 80, blank=False),
              "parent_id": body.get("parent_id"), "muted": body.get("muted", False)}
    if result["parent_id"] is not None:
        identifier(result["parent_id"], goal=True)
    if type(result["muted"]) is not bool:
        invalid()
    if "sensitivity" in body:
        result["sensitivity"] = tier(body["sensitivity"])
    if provenance:
        source(body)
    return result


def progress(body, *, provenance=True):
    fields(body, ("at", "text"), ("value", "unit", "sensitivity", *(["source_id"] if provenance else [])))
    timestamp(body["at"])
    result = {"at": body["at"], "text": text(body["text"], 8000, multiline=True),
              "value": body.get("value"), "unit": body.get("unit")}
    value = result["value"]
    if value is not None:
        try:
            if type(value) not in (int, float) or not math.isfinite(value):
                invalid()
        except OverflowError:
            invalid()
    if result["unit"] is not None:
        text(result["unit"], 80, blank=False)
    if "sensitivity" in body:
        result["sensitivity"] = tier(body["sensitivity"])
    if provenance:
        source(body)
    return result


def validate(action, body):
    if action == "create":
        return create(body)
    if action == "progress":
        return progress(body)
    if action == "patch":
        edits = {"title", "description", "category", "status", "muted", "sensitivity"}
        fields(body, ("expected_revision",), (*edits, "source_id"))
        integer(body["expected_revision"])
        if not edits.intersection(body):
            invalid()
        for key in edits.intersection(body):
            value = body[key]
            if key in {"title", "description", "category"}:
                text(value, {"title": 200, "description": 8000, "category": 80}[key],
                     blank=key == "description", multiline=key == "description")
            elif key == "status":
                if value not in ("active", "completed"):
                    invalid()
            elif key == "muted":
                if type(value) is not bool:
                    invalid()
            else:
                tier(value)
    elif action in {"delete", "detach", "confirm"}:
        fields(body, ("expected_revision",), ("source_id",))
        integer(body["expected_revision"])
    elif action == "attach":
        fields(body, ("type", "ref", "label"), ("source_id",))
        artifact_ref(body["type"], body["ref"])
        text(body["label"], 200, blank=False)
    elif action == "propose":
        fields(body, ("kind", "target_id", "payload"), ("source_id",))
        if body["kind"] == "goal":
            if body["target_id"] is not None:
                invalid()
            create(body["payload"], provenance=False)
        elif body["kind"] == "progress":
            identifier(body["target_id"], goal=True)
            progress(body["payload"], provenance=False)
        else:
            invalid()
    else:
        invalid()
    source(body)
    return {key: value for key, value in body.items() if key != "source_id"}


def artifact_ref(kind, ref):
    if kind not in ("library_item", "cron_job"):
        invalid()
    identifier(ref, job=kind == "cron_job")


def pagination(pairs):
    result = {"limit": 50, "offset": 0}
    seen = set()
    for key, value in pairs:
        if key not in result or key in seen or not re.fullmatch(r"[0-9]+", value, re.ASCII):
            invalid()
        seen.add(key)
        # Avoid giant integer parsing and preserve leading zero decimal admission.
        significant = value.lstrip("0") or "0"
        if len(significant) > 16:
            invalid()
        result[key] = integer(int(significant))
    if not 1 <= result["limit"] <= 100:
        invalid()
    return result
