"""What the recorder's lines tell the owner.

``upkeep`` runs on the mind's tick: it tells the owner when something stops,
closes the alerts whose condition cleared and drops old lines once a day.
``digest_lines`` is the recorder's line in the daily digest. Both read the
store in the mind's own state directory and do nothing while it has none.

An alert is one function over the store that yields what holds now as
``(subject, title, text)``. A condition that starts to hold becomes one line
of kind ``alert`` (source ``recorder.<function>``, the subject in ``detail``)
and one notice to the owner. The line stays open while the condition holds,
so the owner is told once, and it is closed when the condition clears. A new
alert is one function and one entry in ``ALERTS``.

A notice and the digest line hold source labels, counts and durations only.
"""

from __future__ import annotations

import weakref
from collections import Counter
from collections.abc import Iterator, Mapping
from datetime import datetime
from typing import Any

from colonymind.config import DEFAULTS, load_config
from colonymind.recorder import CHOICES, KINDS, RESULTS
from colonymind.recorder.store import INSERT, Store, new_line, store_at

DAY_S = 86400
WEEK_S = 7 * DAY_S
CLOSE = "UPDATE lines SET result = ?, took_ms = CAST((? - began_at) * 1000 AS INTEGER) WHERE id = ?"

Alert = Iterator[tuple[str, str, str]]


def _positive(value: Any, default: Any) -> Any:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else default


def settings() -> dict[str, Any]:
    """The ``recorder`` settings as they are now. A value that is not a positive number counts as its default; a
    kind whose ``open_minutes`` is not a positive number is not watched."""
    defaults = DEFAULTS["recorder"]
    try:
        found = load_config().get("recorder")
    except Exception:  # noqa: BLE001 - settings that cannot be read leave the defaults
        found = None
    found = found if isinstance(found, Mapping) else defaults
    cfg = {name: _positive(found.get(name), default) for name, default in defaults.items()}
    minutes = found.get("open_minutes")
    minutes = minutes if isinstance(minutes, Mapping) else defaults["open_minutes"]
    cfg["open_minutes"] = {str(kind): value for kind, value in minutes.items() if _positive(value, 0)}
    return cfg


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _span(seconds: float) -> str:
    """A duration in words: whole days or hours from two of them up, else minutes."""
    for size, unit in ((DAY_S, "day"), (3600, "hour")):
        if seconds >= 2 * size:
            return _plural(int(seconds // size), unit)
    return _plural(int(seconds // 60), "minute")


# -- the alerts ---------------------------------------------------------------------------------------------------


def quiet(store: Store, now: float, cfg: Mapping[str, Any]) -> Alert:
    """A source that normally produces lines has gone quiet: it recorded on ``quiet_active_days`` of the 7 days
    before its last line, and that line is ``quiet_hours`` old. It counts for one week from then on."""
    gap = cfg["quiet_hours"] * 3600
    for row in store.rows(
            "SELECT l.source, COUNT(DISTINCT CAST(l.began_at / 86400 AS INTEGER)) AS days, MAX(m.last) AS last "
            "FROM (SELECT source, MAX(began_at) AS last FROM lines WHERE kind NOT IN ('error', 'alert') "
            "AND source != '' AND began_at >= ? GROUP BY source HAVING MAX(began_at) <= ?) AS m "
            "JOIN lines AS l ON l.source = m.source AND l.kind NOT IN ('error', 'alert') "
            "AND l.began_at >= m.last - ? GROUP BY l.source",
            (now - gap - WEEK_S, now - gap, WEEK_S)):
        if row["days"] >= cfg["quiet_active_days"]:
            yield (row["source"], f"Recorder: {row['source']} has gone quiet",
                   f"Nothing has been recorded from {row['source']} for {_span(now - row['last'])}. "
                   f"In the week before, it recorded on {_plural(min(row['days'], 7), 'day')}.")


def failing(store: Store, now: float, cfg: Mapping[str, Any]) -> Alert:
    """A check keeps failing: its last ``failing_runs`` runs of the past 7 days all failed."""
    for row in store.rows(
            "SELECT l.source, COUNT(*) AS runs FROM lines AS l LEFT JOIN ("
            "SELECT source, MAX(rowid) AS last FROM lines WHERE kind = 'check' AND result NOT IN ('', 'failed') "
            "AND began_at >= ? GROUP BY source) AS ok ON ok.source = l.source "
            "WHERE l.kind = 'check' AND l.result = 'failed' AND l.began_at >= ? AND l.rowid > COALESCE(ok.last, 0) "
            "GROUP BY l.source HAVING COUNT(*) >= ?",
            (now - WEEK_S, now - WEEK_S, int(cfg["failing_runs"]))):
        yield (row["source"], f"Recorder: {row['source']} keeps failing",
               f"{row['source']} has failed its last {_plural(row['runs'], 'run')}.")


def unfinished(store: Store, now: float, cfg: Mapping[str, Any]) -> Alert:
    """Lines that were begun and never finished: open for the ``open_minutes`` of their kind. A line counts for one
    day from then on, so a source that loses no more lines clears."""
    for kind, minutes in cfg["open_minutes"].items():
        late = now - minutes * 60
        for row in store.rows(
                "SELECT source, COUNT(*) AS n FROM lines WHERE kind = ? AND result = '' AND began_at < ? "
                "AND began_at >= ? GROUP BY source", (kind, late, late - DAY_S)):
            source = row["source"] or "other"
            yield (f"{kind}:{source}", f"Recorder: {source} has unfinished {kind} lines",
                   f"{_plural(row['n'], kind)} from {source} began more than {_span(minutes * 60)} ago "
                   f"and never got a result.")


ALERTS = (quiet, failing, unfinished)


# -- the tick's call ----------------------------------------------------------------------------------------------

_BY_OUTBOX: "weakref.WeakKeyDictionary[Any, Store]" = weakref.WeakKeyDictionary()  # each mind's outbox and its store


def upkeep(mind: Any, now: datetime) -> None:
    """One run on the mind's tick. Never raises."""
    try:
        store = _BY_OUTBOX[mind.outbox] = store_at(mind.state_dir)
    except Exception:  # noqa: BLE001 - a mind without a state directory has no lines
        return
    try:
        if store.path.exists():
            _run(store, mind.outbox, now.timestamp(), settings())
    except Exception as error:  # noqa: BLE001 - the recorder never fails the tick
        store.owe("recorder.watch", error)


def _run(store: Store, outbox: Any, now: float, cfg: Mapping[str, Any]) -> None:
    holding = {(f"recorder.{alert.__name__}", subject): (title, text)
               for alert in ALERTS for subject, title, text in alert(store, now, cfg)}
    told = {(row["source"], row["detail"]): row["id"] for row in store.rows(
        "SELECT id, source, detail FROM lines WHERE kind = 'alert' AND result = ''")}
    for condition in told.keys() - holding.keys():  # it cleared
        store.write(CLOSE, ("done", now, told[condition]))
    for condition in holding.keys() - told.keys():  # it holds since this run: tell the owner once
        line = new_line("alert", condition[0], detail=condition[1], began_at=now)
        if not store.write(INSERT, line):
            continue  # not stored, so not told: the next run tries again
        key = f"recorder:{line[0]}"
        title, text = holding[condition]
        try:
            outbox.notice(type="health_notice", title=title, text=text, dedup_key=key, extra={"topic": key})
        except Exception:  # noqa: BLE001 - not told: the line closes as failed and the next run tries again
            store.write(CLOSE, ("failed", now, line[0]))
    day = int(now // DAY_S)
    if store.pruned_on != day:
        store.pruned_on = day
        store.prune(now - cfg["keep_days"] * DAY_S)


# -- the digest's line --------------------------------------------------------------------------------------------


def digest_lines(outbox: Any, since: datetime, now: datetime) -> list[str]:
    """The recorder's line for the digest of this outbox that covers ``since`` to ``now``: counts by kind, choice
    and result, the failures and the lines still open. None when nothing was recorded. Never raises."""
    store = None
    try:
        store = _BY_OUTBOX.get(outbox)
        if store is None or not store.path.exists():
            return []
        line = _digest_line(store, since.timestamp(), now.timestamp())
        return [line] if line else []
    except Exception as error:  # noqa: BLE001 - the digest goes out without the line
        if store is not None:
            store.owe("recorder.watch", error)
        return []


def _listed(counts: Mapping[str, int], order: tuple[str, ...]) -> str:
    """Counts in words, the known labels first: ``20 ignore, 12 reply``."""
    names = [name for name in order if name in counts] + sorted(name for name in counts if name not in order)
    return ", ".join(f"{counts[name]} {name.replace('_', ' ')}" for name in names)


def _digest_line(store: Store, since: float, until: float) -> str:
    rows = store.rows(
        "SELECT kind, choice, result, COUNT(*) AS n FROM lines WHERE began_at >= ? AND began_at <= ? "
        "GROUP BY kind, choice, result", (since, until))
    if not rows:
        return ""
    total: Counter = Counter()
    choices: dict[str, Counter] = {}
    results: dict[str, Counter] = {}
    for row in rows:
        kind = row["kind"] or "other"
        total[kind] += row["n"]
        if row["choice"]:
            choices.setdefault(kind, Counter())[row["choice"]] += row["n"]
        results.setdefault(kind, Counter())[row["result"] or "open"] += row["n"]
    parts = []
    for kind in [name for name in KINDS if name in total] + sorted(name for name in total if name not in KINDS):
        labels = (_listed(choices.get(kind, {}), CHOICES), _listed(results[kind], RESULTS))
        parts.append(f"{_plural(total[kind], kind)} ({'; '.join(text for text in labels if text)})")
    failed = store.rows(
        "SELECT source, COUNT(*) AS n FROM lines WHERE result = 'failed' AND began_at >= ? AND began_at <= ? "
        "GROUP BY source ORDER BY n DESC, source", (since, until))
    where = [f"{row['n']} {row['source'] or 'other'}" for row in failed[:4]]
    if failed[4:]:
        where.append(f"{sum(row['n'] for row in failed[4:])} elsewhere")
    count = sum(row["n"] for row in failed)
    still = {row["kind"] or "other": row["n"] for row in store.rows(
        "SELECT kind, COUNT(*) AS n FROM lines WHERE result = '' AND began_at <= ? GROUP BY kind", (until,))}
    kinds = [name for name in KINDS if name in still] + sorted(name for name in still if name not in KINDS)
    return (f"Recorded: {', '.join(parts)}. Failed: {count}{' (' + ', '.join(where) + ')' if where else ''}. "
            f"Still open: {', '.join(_plural(still[name], name) for name in kinds) or '0'}.")
