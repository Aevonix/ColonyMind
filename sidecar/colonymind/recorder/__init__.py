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
    marks     later labels added by ``mark``. ``MARKS`` lists the ones in use.

A value outside its set is recorded as ``other``. Adding a label is one entry
in the tuple below. A ``ref`` is the one exception: it has the form
``<store>:<row key>``, and one that does not is left empty, so that lines
never share a ref by accident and ``mark`` reaches only the lines it names.

The lines are one table in the instance's state directory (``store``). A
call that cannot store its line drops it and leaves an ``error`` line for the
next call that can. ``watch`` tells the owner when something stops, adds the
daily line to the digest and prunes old lines. Each alert it sends is itself
a line of kind ``alert``, open while its condition holds.
"""

from __future__ import annotations

import re
import time
import uuid

from colonymind import get_state_dir
from colonymind.recorder.store import INSERT, Store, new_line, store_at

KINDS = ("event", "card", "check", "error", "alert")
CHOICES = ("ignore", "remember", "react", "reply", "act", "tell_owner", "ask_owner")
CHOOSERS = ("rule", "instinct", "model", "owner")
RESULTS = ("sent", "reacted", "silent", "failed", "done", "blocked", "pending")
MARKS = ("owner_acted", "owner_declined", "owner_said_should_have", "owner_said_no_need")

_LABEL = re.compile(r"^[a-z][a-z0-9_.:\-]{0,63}$")
_DETAIL = re.compile(r"^[A-Za-z][A-Za-z0-9_.:\-]{0,63}$")  # an error class is written with capitals
_REF = re.compile(r"^[a-z][a-z0-9_]{0,31}:[A-Za-z0-9][A-Za-z0-9_.:\-]{0,119}$")  # a row key may be a digest or a code


def _label(value: str, allowed: tuple[str, ...] | None = None, pattern: re.Pattern[str] = _LABEL) -> str:
    text = str(value or "")
    if allowed is not None:
        return text if text in allowed else ("" if not text else "other")
    return text if pattern.match(text) else ("" if not text else "other")


def _ref(value: str) -> str:
    text = str(value or "")
    return text if _REF.match(text) else ""


def _store() -> Store:
    return store_at(get_state_dir())


def begin(kind: str, source: str, ref: str = "") -> str:
    """Open a line and return its id. Never raises."""
    line_id = uuid.uuid4().hex
    try:
        _store().write(INSERT, new_line(_label(kind, KINDS), _label(source), ref=_ref(ref), line_id=line_id))
    except Exception:  # noqa: BLE001 - recording never fails the caller
        pass
    return line_id


def finish(line_id: str, *, result: str, choice: str = "", chooser: str = "", detail: str = "") -> None:
    """Close a line opened by ``begin``. Unknown ids are ignored. Never raises."""
    try:
        _store().write(
            "UPDATE lines SET result = ?, choice = ?, chooser = ?, detail = ?, "
            "took_ms = CAST((? - began_at) * 1000 AS INTEGER) WHERE id = ?",
            (_label(result, RESULTS), _label(choice, CHOICES), _label(chooser, CHOOSERS),
             _label(detail, pattern=_DETAIL), time.time(), str(line_id or "")),
        )
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
    line_id = uuid.uuid4().hex
    try:
        try:
            took = 0 if took_ms is None else int(took_ms)
        except (TypeError, ValueError, OverflowError):
            took = 0
        _store().write(INSERT, new_line(
            _label(kind, KINDS), _label(source), ref=_ref(ref), choice=_label(choice, CHOICES),
            chooser=_label(chooser, CHOOSERS), result=_label(result, RESULTS),
            detail=_label(detail, pattern=_DETAIL), took_ms=took, line_id=line_id))
    except Exception:  # noqa: BLE001
        pass
    return line_id


def mark(ref: str, label: str) -> int:
    """Add one later label to every line with this ``ref``. Returns how many. Never raises."""
    try:
        key, tag = _ref(ref), _label(label)
        if not key or not tag:
            return 0
        return _store().write(
            "UPDATE lines SET marks = CASE WHEN marks = '' THEN ? ELSE marks || ' ' || ? END WHERE ref = ?",
            (tag, tag, key))
    except Exception:  # noqa: BLE001
        return 0


def recent(limit: int = 50, *, kind: str = "", source: str = "") -> list[dict]:
    """Newest lines first, as plain dicts. Never raises."""
    try:
        where = " AND ".join(f"{name} = ?" for name, value in (("kind", kind), ("source", source)) if value)
        rows = _store().read(
            f"SELECT * FROM lines {'WHERE ' + where if where else ''} ORDER BY rowid DESC LIMIT ?",
            (*(value for value in (kind, source) if value), max(0, int(limit))))
        return [dict(row, marks=row["marks"].split()) for row in rows]
    except Exception:  # noqa: BLE001
        return []
