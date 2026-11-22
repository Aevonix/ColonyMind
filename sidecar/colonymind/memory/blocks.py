"""Resident blocks: a hard limit per block, one global budget, fresh state first, and a fossil guard.

Every text block the Hermes plugin carries on every turn (the sections of ``/v1/host/context/assemble``:
current state, recorded views, lessons, the owner's profile, recall ...) and the mind's recipient packet
pass through :func:`fit`; the self-narrative admits its lines with the same rule. Three guarantees:

1. **No block outgrows its limit, and none is cut silently.** A block over its limit keeps its first whole
   lines (every producer writes most important first) and ends with a marker that says how many lines and
   characters were held back and why. The held-back text is not lost: it stays in the stores it was read
   from, and the report carries its SHA-256.
2. **The whole never outgrows the host's budget, and nothing vanishes without a trace.** Over budget,
   blocks are admitted in :data:`RANK` order (fresh state before recall before the slow profile: current
   state before identity) in two passes: every block first gets a floor (its head), then the rest fills in
   rank order. A block that does not get even its floor is named in one ``Held back this turn`` section.
   A host that replaces an over-cap payload with a preview (Hermes spills memory context over
   ``hooks.output_spill.max_chars``) therefore never receives one: the failure where every block vanishes
   from the wake cannot happen.
3. **A block cannot fossilise unnoticed.** :class:`BlockLedger` keeps, per persisted block, when its content
   last changed, the sources it rests on and when its writer last rebuilt it. Re-writing the same content
   moves nothing (a timestamp bump grows nothing), and a block whose sources moved while it went
   ``fossil_days`` without a rebuild is flagged, logged once and served on ``GET /v1/mind/blocks``.

Configuration (``colonymind.yaml``)::

    context:
      budget_chars: 16000   # hard ceiling; a host's inline_chars can shorten it
      recall_items: 20      # owner-adjustable; matches the independent rerank candidate bound
      fossil_days: 7
      limits: {}            # per-block overrides, {section id: characters}
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from colonymind.util.temporal import now_utc

logger = logging.getLogger(__name__)

#: Admission order under a short budget: fresh state first, then what this turn recalled, then the slow
#: profile and lists (Compass 5.6: current state before identity; identity is already in the session prompt).
RANK: Tuple[str, ...] = (
    "temporal-context",             # the clock
    "colonymind-current-state",      # observed state and stale observations
    "colonymind-owner-continuity",   # the owner's last exchange on another platform
    "colonymind-executions",         # work running now
    "colonymind-mind",               # on my mind, open asks
    "colonymind-status",             # the agent's posture now (colonymind.mind.status)
    "colonymind-commitments",
    "colonymind-waiting",
    "colonymind-open-questions",
    "colonymind-corrections",        # corrections to earlier context (owner-critical)
    "colonymind-standing-lessons",
    "colonymind-page-person",        # the agent's page on the turn's person, its boundaries first (pages.turn)
    "colonymind-memory",             # this turn's recall
    "colonymind-lessons",
    "colonymind-stances",            # recorded views
    "colonymind-page-self",          # the agent's page on itself
    "colonymind-appraisals",
    "colonymind-owner-preferences",  # the owner's profile
    "colonymind-self-perspective",
    "colonymind-person",
    "colonymind-relationship",
    "colonymind-comms-landscape",
    "colonymind-goals",
    "colonymind-initiatives",
)
#: Hard per-block limits (characters of the body). Every producer with its own cap fits inside its limit
#: (tests pin it), so a limit only ever acts on a block that outgrew what its producer promised.
LIMITS: Dict[str, int] = {
    "colonymind-standing-lessons": 900,
    "temporal-context": 2000,
    "colonymind-current-state": 3000,
    "colonymind-owner-continuity": 2200,  # six summaries of 280 characters plus local labels
    "colonymind-executions": 4000,       # request_work_context(max_chars=4000)
    "colonymind-mind": 600,              # MIND_SECTION_CHARS
    "colonymind-commitments": 3000,
    "colonymind-memory": 24000,          # the recall packet's own ceiling (memory/search.py)
    "colonymind-stances": 1500,          # opinions.CONTEXT_CHARS = 1400
    "colonymind-appraisals": 3000,
    "colonymind-person": 600,            # contacts.digest.MAX_CHARS
    "colonymind-relationship": 500,
    "colonymind-page-person": 2000,      # a page past it keeps its first lines: the boundaries, then the oldest
}
DEFAULT_LIMIT = 1500
DEFAULT_BUDGET = 16000
FLOOR = 240                  # the head every admitted block keeps under a short budget
FOSSIL_DAYS = 7
HELD_ID, HELD_TITLE = "colonymind-held-back", "Held back this turn"
#: Sections never collapsed to a line by the provider (its ``_ALWAYS_SENT``).
ALWAYS_WHOLE = frozenset({"temporal-context", "colonymind-standing-lessons", "colonymind-owner-continuity"})
#: What the provider sends instead of a section its session's last turn already carried (provider.py).
UNCHANGED_LINE = "unchanged since your last turn: {title}"
MARK = "[{chars} characters in {lines} line{s} held back: {why}]"


@dataclass(frozen=True)
class Block:
    id: str
    title: str
    body: str
    ref: int = -1          # the caller's own index, carried through ``fit`` untouched


def cost(title: str, body: str) -> int:
    """What a block adds to the rendered context: ``## title\\nbody`` and the blank line after it."""
    return len(title) + len(body) + 6


def fingerprint(title: str, body: str) -> str:
    """The provider's per-section fingerprint (``ColonyMindMemoryProvider._unchanged``)."""
    return hashlib.sha256(f"{title}\n{body}".encode("utf-8", "replace")).hexdigest()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _mark(lines: int, chars: int, why: str) -> str:
    return MARK.format(lines=lines, s="" if lines == 1 else "s", chars=chars, why=why)


def rotate(body: str, limit: int, *, why: str) -> str:
    """``body`` in at most ``limit`` characters: its first whole lines, then a marker naming what was held
    back and why. A first line too long to keep whole is cut at a word and ends with an ellipsis; the marker
    counts it among the held-back lines. ``body`` is returned as it is when it fits."""
    if len(body) <= limit:
        return body
    lines = body.split("\n")
    widest = len(_mark(len(lines), len(body), why))      # no marker this body needs is longer
    kept: List[str] = []
    used = 0
    for line in lines:
        grown = used + len(line) + (1 if kept else 0)
        if grown + 1 + widest > limit:
            break
        kept.append(line)
        used = grown
    if kept:
        text = "\n".join(kept)
        held, shown = len(lines) - len(kept), len(text)
    else:                                                  # one long first line: cut it at a word, and say so
        head = lines[0][:max(0, limit - widest - 2)]
        if " " in head[len(head) // 2:]:
            head = head.rsplit(" ", 1)[0]
        head = head.rstrip()
        text, held, shown = head + "…", len(lines), len(head)
    result = f"{text}\n{_mark(held, len(body) - shown, why)}"
    return result if len(result) <= limit else _mark(len(lines), len(body), why)[:limit]


def rank_of(block_id: str) -> int:
    try:
        return RANK.index(block_id)
    except ValueError:
        return len(RANK)


def fit(blocks: Sequence[Block], *, budget: Optional[int], limits: Optional[Mapping[str, int]] = None,
        shown: Iterable[str] = (), floor: int = FLOOR) -> Tuple[List[Block], Dict[str, Any]]:
    """``blocks`` within their limits and, together, within ``budget`` characters as the provider renders
    them. Display order is kept; only admission follows :data:`RANK`. ``shown`` is the provider's
    fingerprints of what its session's last turn carried: those blocks cost one line (the provider
    collapses them) and are never rotated or held back. Returns the blocks to show (a ``Held back this
    turn`` block last when any were held back) and the report every action is on."""
    limits = {**LIMITS, **dict(limits or {})}
    shown = set(shown or ())
    report: Dict[str, Any] = {"budget": budget, "chars_before": sum(cost(b.title, b.body) for b in blocks),
                              "rotated": [], "held_back": []}
    limited: List[Block] = []
    for block in blocks:
        limit = max(1, int(limits.get(block.id, DEFAULT_LIMIT)))
        body = rotate(block.body, limit, why=f"this block's {limit}-character limit")
        if body != block.body:
            report["rotated"].append({"id": block.id, "reason": "limit", "chars": len(block.body),
                                      "kept": len(body), "sha256": _sha(block.body)})
        limited.append(replace(block, body=body))

    def collapsed(block: Block) -> bool:
        return block.id not in ALWAYS_WHOLE and fingerprint(block.title, block.body) in shown

    def weight(block: Block) -> int:
        return len(UNCHANGED_LINE.format(title=block.title)) + 2 if collapsed(block) else cost(block.title, block.body)

    total = sum(weight(block) for block in limited)
    if budget is None or total <= budget:
        report["chars_after"] = total
        return limited, report
    order = sorted(range(len(limited)), key=lambda index: (rank_of(limited[index].id), index))
    held: set = set()
    while True:
        note = _held_body(budget, [limited[i].title for i in sorted(held)]) if held else ""
        remaining = budget - (cost(HELD_TITLE, note) if note else 0)
        alloc: Dict[int, int] = {}
        for index in order:                                   # pass 1: every block its floor, in rank order
            block = limited[index]
            if index in held:
                continue
            if collapsed(block):
                need, give = weight(block), len(block.body)
            else:
                give = min(len(block.body), floor)
                need = cost(block.title, "") + give
            if need <= remaining:
                alloc[index], remaining = give, remaining - need
        grown = held | {index for index in order if index not in alloc}
        if grown == held:
            break
        held = grown
    for index in order:                                       # pass 2: the rest, in rank order
        if index in alloc and not collapsed(limited[index]):
            extra = min(len(limited[index].body) - alloc[index], max(0, remaining))
            alloc[index] += extra
            remaining -= extra
    result: List[Block] = []
    for index, block in enumerate(limited):
        if index in held:
            report["held_back"].append({"id": block.id, "chars": len(block.body), "sha256": _sha(block.body)})
            continue
        if alloc[index] < len(block.body):
            body = rotate(block.body, alloc[index], why=f"this turn's {budget}-character context budget")
            report["rotated"].append({"id": block.id, "reason": "budget", "chars": len(block.body),
                                      "kept": len(body), "sha256": _sha(block.body)})
            block = replace(block, body=body)
        result.append(block)
    if note:
        result.append(Block(HELD_ID, HELD_TITLE, note))
    report["chars_after"] = sum(weight(block) for block in result)
    return result, report


def _held_body(budget: int, titles: Sequence[str]) -> str:
    """The ``Held back this turn`` body within ``budget``: every held block named; a budget too small for the
    names gets the count alone, and one too small for that gets none (the report still names every one)."""
    full = (f"Not shown, over this turn's {budget}-character context budget: {'; '.join(titles)}. "
            "Their records are intact and a memory search reaches them.")
    if cost(HELD_TITLE, full) <= budget:
        return full
    count = f"{len(titles)} block{'' if len(titles) == 1 else 's'} not shown, over this turn's context budget."
    return count if cost(HELD_TITLE, count) <= budget else ""


# -- configuration ---------------------------------------------------------------------------------------

_CONFIG_TTL_S = 60.0
_configured: Dict[str, Any] = {"read_at": -1e18, "value": None}
_config_lock = threading.Lock()


def settings() -> Dict[str, Any]:
    """``context`` from ``colonymind.yaml`` (read at most once a minute), with mandatory caps defaulted."""
    clock = time.monotonic()
    with _config_lock:
        if clock - _configured["read_at"] < _CONFIG_TTL_S and _configured["value"] is not None:
            return dict(_configured["value"])
    value: Dict[str, Any] = {"budget_chars": DEFAULT_BUDGET, "recall_items": 20,
                             "fossil_days": FOSSIL_DAYS, "limits": {}}
    try:
        from colonymind.config import load_config
        section = load_config().get("context") or {}
        if isinstance(section, Mapping):
            value.update({key: section[key] for key in value if key in section})
    except Exception:
        logger.debug("context settings unavailable; the defaults apply", exc_info=True)
    with _config_lock:
        _configured.update(read_at=clock, value=dict(value))
    return value


def reset_settings() -> None:
    """Forget the cached ``context`` settings (tests, and a config write that must apply at once)."""
    with _config_lock:
        _configured.update(read_at=-1e18, value=None)


# -- the last fit, for the owner's inspection ------------------------------------------------------------

_last: Dict[str, Any] = {}


def record_fit(report: Dict[str, Any], *, where: str) -> None:
    """Keep the latest report and log any action, so what a turn did not show is always on record."""
    _last.clear()
    _last.update({"where": where, "at": now_utc().isoformat(), **report})
    if report.get("rotated") or report.get("held_back"):
        logger.info("context blocks fitted (%s): budget=%s chars %s -> %s, rotated=%s, held back=%s", where,
                    report.get("budget"), report.get("chars_before"), report.get("chars_after"),
                    [item["id"] for item in report.get("rotated") or []],
                    [item["id"] for item in report.get("held_back") or []])


def last_fit() -> Dict[str, Any]:
    return dict(_last)


# -- the fossil guard ------------------------------------------------------------------------------------

class BlockLedger:
    """Per persisted block (``mind_state`` key ``block:<id>``): the content fingerprint and size, when the
    content last changed (``changed_at``), the sources it rests on and when its writer last rebuilt it from
    its sources (``built_at``), and the sources seen now with the time they last moved.

    Re-persisting the same content never moves ``changed_at`` and never changes ``chars``: a writer that
    only bumps a timestamp grows nothing and makes nothing look fresher. A block is *stale* when the sources
    seen now are not the ones it rests on (``stale_since``: the first such observation since it last rested
    on its sources), and *fossilised* when it has been stale with no rebuild for ``fossil_days``: flagged on
    every read, logged once per episode, cleared by the next rebuild. A rebuild
    that keeps the same text (the writer read the new sources and changed nothing) is not a fossil; a text
    that only shrinks (lines expiring) without a rebuild is."""

    PREFIX = "block:"

    def __init__(self, mind_state: Any, *, fossil_days: Optional[float] = None,
                 clock: Optional[Callable[[], datetime]] = None) -> None:
        self.state = mind_state
        self.fossil_days = fossil_days
        self.clock = clock or (lambda: now_utc())

    def _days(self) -> float:
        return float(self.fossil_days if self.fossil_days is not None else settings().get("fossil_days", FOSSIL_DAYS))

    def get(self, block_id: str, now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        record = self._load(block_id)
        return self._view(record, now or self.clock()) if record is not None else None

    def _load(self, block_id: str) -> Optional[Dict[str, Any]]:
        row = self.state.get(self.PREFIX + block_id)
        try:
            record = json.loads((row or {}).get("text") or "")
        except ValueError:
            return None
        return record if isinstance(record, dict) else None

    def observe(self, block_id: str, text: str, *, sources: Iterable[Any], built_from: Optional[Iterable[Any]] = None,
                rebuilt: bool = False, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Record what block ``block_id`` holds now and the sources it answers to now. ``built_from`` names the
        sources the content rests on when the writer knows them (a digest keeps its own); ``rebuilt`` says the
        writer rebuilt it now, from ``built_from`` or else from ``sources``. A block seen for the first time
        rests on ``built_from`` or else on ``sources``."""
        now = now or self.clock()
        stamp = now.isoformat()
        content, current = _sha(text or ""), _sources(sources)
        record = self._load(block_id)
        # A record written before ``stale_since`` existed and already stale carries its episode start only in
        # the read view's fallback; observing it must not restart that clock (it would clear a fossil unbuilt).
        inherited = _stale_start(record) if record is not None else None
        if record is None:
            record = {"id": block_id, "content": content, "chars": len(text or ""), "changed_at": stamp,
                      "built_from": current if built_from is None else _sources(built_from), "built_at": stamp}
        else:
            if record.get("content") != content:
                record.update(content=content, chars=len(text or ""), changed_at=stamp)
            if built_from is not None or rebuilt:
                record["built_from"] = current if built_from is None else _sources(built_from)
            if rebuilt:
                record["built_at"] = stamp
        if record.get("sources") != current:
            record.update(sources=current, sources_changed_at=stamp)
        # The episode starts at the first move its writer has not answered, not at the last rebuild: a block
        # current for weeks whose first new source arrives tonight is stale from tonight.
        if record.get("sources") != record.get("built_from"):
            record.setdefault("stale_since", inherited or stamp)
        else:
            record.pop("stale_since", None)
        record["seen_at"] = stamp
        view = self._view(record, now)
        if view["fossilised"] and record.get("flagged") != record["stale_since"]:
            record["flagged"] = record["stale_since"]
            logger.warning("resident block %s is fossilised: stale since %s with no rebuild (last built %s)",
                           block_id, record["stale_since"], record["built_at"])
        self.state.set(self.PREFIX + block_id, text=json.dumps(record, sort_keys=True), now=now)
        return view

    def _view(self, record: Mapping[str, Any], now: datetime) -> Dict[str, Any]:
        stale = record.get("sources") != record.get("built_from")
        since = _parse(_stale_start(record))
        fossil = bool(stale and since and now - since >= timedelta(days=self._days()))
        return {key: record.get(key) for key in ("id", "chars", "changed_at", "built_at", "sources_changed_at",
                                                 "stale_since", "seen_at")} | {"stale": stale, "fossilised": fossil}

    def records(self, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        now = now or self.clock()
        out = []
        for row in self.state.items(self.PREFIX):
            try:
                record = json.loads(row.get("text") or "")
            except ValueError:
                continue
            if isinstance(record, dict):
                out.append(self._view(record, now))
        return out

    def fossils(self, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        return [record for record in self.records(now) if record["fossilised"]]


def _stale_start(record: Mapping[str, Any]) -> Optional[str]:
    """When ``record``'s current stale episode began, or None when it rests on its sources. A record from
    before ``stale_since`` was kept was stale at least since its sources last moved."""
    if record.get("sources") == record.get("built_from"):
        return None
    return record.get("stale_since") or record.get("sources_changed_at")


def _sources(refs: Iterable[Any]) -> str:
    return _sha("\n".join(sorted({str(ref) for ref in refs or ()})))


def _parse(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


__all__ = ["ALWAYS_WHOLE", "DEFAULT_BUDGET", "DEFAULT_LIMIT", "FLOOR", "FOSSIL_DAYS", "HELD_ID", "HELD_TITLE",
           "LIMITS", "RANK", "UNCHANGED_LINE", "Block", "BlockLedger", "cost", "fingerprint", "fit", "last_fit",
           "rank_of", "record_fit", "reset_settings", "rotate", "settings"]
