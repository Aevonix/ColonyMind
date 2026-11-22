"""One recorder for everything the agent decides and does.

One line per event, card, check and error: which source, which choice, who
chose, what came of it and how long it took. A line holds code labels and
timings only. It never holds message text, names, addresses, handles or any
identifier that comes from outside this instance.

This module is the whole interface. ``begin`` and ``finish`` bracket work
that has a start and an end. ``once`` records something settled in one step.
``mark`` adds a later label to lines that share a ``ref`` (for example that
the owner acted on a card, or corrected a decision). ``recent`` reads lines
back. A line that was begun and never finished is itself a signal.

Calls never raise and never block the caller. Recording is never a reason
for a turn, a task or a check to fail.

Labels
    kind      one of ``KINDS``
    source    short code label: a channel, a job or a module (``whatsapp``,
              ``check.mail``, ``mind.task``). Never a person, chat or address.
    ref       key of a row in this instance's own stores, used to join lines
              (``initiative:42``). Never an outside identifier.
    choice    one of ``CHOICES``
    chooser   one of ``CHOOSERS``
    result    one of ``RESULTS``
    detail    one short code label (an error class, a rule name). Never text.

A value outside its set is recorded as ``other``. Adding a label is one entry
in the tuple below.
"""

from __future__ import annotations

import re
import threading
import time
import uuid

KINDS = ("event", "card", "check", "error")
CHOICES = ("ignore", "remember", "react", "reply", "act", "tell_owner", "ask_owner")
CHOOSERS = ("rule", "instinct", "model", "owner")
RESULTS = ("sent", "reacted", "silent", "failed", "done", "blocked", "pending")

_LABEL = re.compile(r"^[a-z][a-z0-9_.:\-]{0,63}$")
_LOCK = threading.Lock()
_LINES: dict[str, dict] = {}
_KEEP = 2000


def _label(value: str, allowed: tuple[str, ...] | None = None) -> str:
    text = str(value or "")
    if allowed is not None:
        return text if text in allowed else ("" if not text else "other")
    return text if _LABEL.match(text) else ("" if not text else "other")


def begin(kind: str, source: str, ref: str = "") -> str:
    """Open a line and return its id. Never raises."""
    line_id = uuid.uuid4().hex
    try:
        line = {
            "id": line_id,
            "kind": _label(kind, KINDS),
            "source": _label(source),
            "ref": _label(ref),
            "choice": "",
            "chooser": "",
            "result": "",
            "detail": "",
            "marks": [],
            "began_at": time.time(),
            "took_ms": None,
        }
        with _LOCK:
            _LINES[line_id] = line
            while len(_LINES) > _KEEP:
                _LINES.pop(next(iter(_LINES)))
    except Exception:  # noqa: BLE001 - recording never fails the caller
        pass
    return line_id


def finish(line_id: str, *, result: str, choice: str = "", chooser: str = "", detail: str = "") -> None:
    """Close a line opened by ``begin``. Unknown ids are ignored. Never raises."""
    try:
        with _LOCK:
            line = _LINES.get(line_id)
            if line is None:
                return
            line["result"] = _label(result, RESULTS)
            line["choice"] = _label(choice, CHOICES)
            line["chooser"] = _label(chooser, CHOOSERS)
            line["detail"] = _label(detail)
            line["took_ms"] = int((time.time() - line["began_at"]) * 1000)
    except Exception:  # noqa: BLE001
        pass


def once(
    kind: str,
    source: str,
    *,
    result: str,
    ref: str = "",
    choice: str = "",
    chooser: str = "",
    detail: str = "",
    took_ms: int | None = None,
) -> str:
    """Record something settled in one step and return the line id. Never raises."""
    line_id = begin(kind, source, ref)
    finish(line_id, result=result, choice=choice, chooser=chooser, detail=detail)
    if took_ms is not None:
        try:
            with _LOCK:
                if line_id in _LINES:
                    _LINES[line_id]["took_ms"] = int(took_ms)
        except Exception:  # noqa: BLE001
            pass
    return line_id


def mark(ref: str, label: str) -> int:
    """Add one later label to every line with this ``ref``. Returns how many. Never raises."""
    count = 0
    try:
        key, tag = _label(ref), _label(label)
        if not key or not tag:
            return 0
        with _LOCK:
            for line in _LINES.values():
                if line["ref"] == key:
                    line["marks"].append(tag)
                    count += 1
    except Exception:  # noqa: BLE001
        pass
    return count


def recent(limit: int = 50, *, kind: str = "", source: str = "") -> list[dict]:
    """Newest lines first, as plain dicts. Never raises."""
    try:
        with _LOCK:
            rows = [dict(line, marks=list(line["marks"])) for line in _LINES.values()]
        rows.reverse()
        if kind:
            rows = [row for row in rows if row["kind"] == kind]
        if source:
            rows = [row for row in rows if row["source"] == source]
        return rows[: max(0, int(limit))]
    except Exception:  # noqa: BLE001
        return []
