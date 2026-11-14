"""Why a recall kept or left out each candidate: the stage-level trace (docs/RECALL-HYBRID.md, Tracing a recall).

A caller asks for one with ``trace`` (memory search; context assembly for a credential that manages the instance).
The recall then fills a :class:`RecallTrace` as it runs: each retrieval lane's ranks where the lanes are fused
(``memory.search.collect_sources``), the fused order offered to the reranker, what the reranker was sent (when the
call starts, whatever comes of it), what it scored and the calibrated cutoff (``memory.selection``), and what the
packer kept or left out (``memory.recall``). ``report()`` then gives one row per candidate: its rank in each lane,
its place in the fused order, whether the reranker was sent it and its score, and one exit reason from a closed set
(``EXITS``), none when it is in the packet. A message a lane found that no candidate carried to ranking gets a row
of its own.

The trace holds ids, ranks, scores and closed-set values: never text, never the question. It is built for one
response and never stored. Reported, every id in it is read back against the ledger in the recall's own scope: a
candidate keeps its ids only while every source it names is readable there and every message it carries is still in
its source (erasing a message also takes its copies out of the checkpoints that keep other messages). Any other
keeps one row, ``erased``, without its ids.

With tracing off the recall is handed ``NULL_TRACE``: falsy, with methods that do nothing, so a recall nobody asked
to explain does no work for it.
"""
from __future__ import annotations

import hashlib
import json
import re
from contextlib import closing
from itertools import islice
from typing import Any, Literal, get_args

#: Why a candidate is not in the packet, in the order of the stages that leave it out:
#: ``no_lane``: no lane carried it to ranking (not found, or found and then left out before ranking: a repeat of
#: words already ranked, outside the question's time, a reply shown only beside its input);
#: ``below_submission_bound``: the reranker was sent at most the submission bound of candidates and this one was
#: below it in the fused order; ``over_session_share``: it was within the bound in the fused order, and the
#: submission, spread across sessions, went to the other sessions' candidates and its own session's stronger ones;
#: ``below_cutoff``: it scored below the calibrated cutoff (only ever while a verified cutoff applied);
#: ``over_packet_limit``: the packet was full; ``over_char_budget``: it did not fit the character budget;
#: ``over_host_allowance``: it fitted recall's budget, not the host's inline allowance less the items kept whole
#: beside the memory (context assembly, when the host names its allowance);
#: ``removed_by_recheck``: its source or its corrections changed while it was ranked;
#: ``erased``: its source was erased (or is no longer readable in the scope) by the time the trace was reported.
Exit = Literal["no_lane", "below_submission_bound", "over_session_share", "below_cutoff", "over_packet_limit",
               "over_char_budget", "over_host_allowance", "removed_by_recheck", "erased"]
EXITS: tuple[str, ...] = get_args(Exit)
#: What form the candidate took in ranking.
Form = Literal["quotation", "claims", "conversation", "correction", "media", "media_name", "belief", "other"]
FORMS: tuple[str, ...] = get_args(Form)
#: What the reranker did: ``off`` (not configured to run), ``not_needed`` (every candidate already fit and no cutoff
#: applied), ``scored``, ``shadow`` (scored and logged; the order and the cutoff were not used) or ``missed``.
RerankOutcome = Literal["off", "not_needed", "scored", "shadow", "missed"]
#: The calibrated cutoff: ``none`` configured, or the configuration check's outcome (the rerank's own terms).
Calibration = Literal["none", "unverified", "mismatch", "configuration_verified",
                      "configuration_verified_weights_unverified"]
#: Context assembly only: whether the recalled section reached the returned context as recalled.
Section = Literal["in_context", "shortened", "held_back"]
#: A candidate's id as the trace shows it: a kind prefix and a digest.
CANDIDATE_ID = r"^[a-z][a-z-]{0,39}:[0-9a-f]{64}$"
#: A lane's name.
LANE_NAME = r"^[a-z][a-z0-9_]{0,31}$"
#: The most rows a trace returns (my choice): recall collects tens of candidates, so only a pathological recall
#: reaches it; the rest are counted in ``rows_omitted``.
MAX_ROWS = 256

_CANDIDATE = re.compile(CANDIDATE_ID)
_LANE = re.compile(LANE_NAME)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
# How far a candidate got, for a message several candidates carry: it got as far as the furthest of them.
_REACH = {"no_lane": 0, "below_submission_bound": 1, "over_session_share": 1, "below_cutoff": 2,
          "over_packet_limit": 3, "over_char_budget": 3, "over_host_allowance": 3, "removed_by_recheck": 4,
          "erased": -1, None: 5}


class NullTrace:
    """Tracing off: every recording method of :class:`RecallTrace`, doing nothing."""

    __slots__ = ()

    def __bool__(self) -> bool:
        return False

    def lane(self, name, hits, *, media=False): pass

    def recall(self, collected, *, limit, max_chars): pass

    def fused(self, candidates): pass

    def calibration(self, status, cutoff): pass

    def submitted(self, rows, *, bound): pass

    def rerank(self, outcome, *, mode, candidates=(), scores=None, cutoff=None, missed=None, late=False): pass

    def packing(self, rows, *, limit, max_chars): pass

    def left(self, rows, reason, *, start=0): pass

    def dropped(self, row, reason): pass

    def shown(self, row, by): pass

    def replaced(self, selected, kept, by): pass

    def packed(self, selected): pass

    def rechecked(self, selected, retained): pass

    def packet(self, selected, content): pass

    def context_section(self, body): pass


NULL_TRACE = NullTrace()


class RecallTrace:
    """One recall's trace, filled by the recall as it runs and read once with :meth:`report`."""

    def __init__(self) -> None:
        # Per lane, the rank of each (source, message hash, image asset) it found: the asset for an image lane only.
        self._lanes: dict[str, dict[tuple[str, str | None, str | None], int]] = {}
        self._media_lanes: set[str] = set()
        self._lane_hits: dict[str, int] = {}
        self._scope: tuple[Any, str, str] | None = None
        self._rows: dict[str, dict] = {}
        self._fused: dict[str, int] = {}
        self._order: list[str] | None = None
        self._pack: tuple[int | None, int] | None = None
        self._outcome: dict[str, Any] = {}
        self._submitted: set[str] = set()
        self._sent: list[str] = []
        self._bound: int | None = None
        self._scores: dict[str, float] = {}
        self._rerank: dict | None = None
        self._calibration: dict = {"status": "none", "cutoff": None}
        self._selected: list[str] | None = None
        self._digest: str | None = None
        self._section: str | None = None
        self._resolved: dict | None = None

    def __bool__(self) -> bool:
        return True

    # --- recording, in the order the recall calls it ------------------------------------------------------------

    def lane(self, name: str, hits, *, media: bool = False) -> None:
        """A retrieval lane's hits, best first (``collect_sources``, where the lanes are fused). A ``media`` lane
        finds images by their descriptions: each rank counts for the image it found (its asset, so two images of
        one message keep their own ranks), never for the message's words."""
        if not _LANE.match(name):
            raise ValueError(f"a lane's name is lower-case letters, digits and underscores: {name!r}")
        if media:
            self._media_lanes.add(name)
        ranks = self._lanes.setdefault(name, {})
        count = 0
        for rank, hit in enumerate(hits, 1):
            count += 1
            turn = hit.get("turn_id") or hit.get("source_turn_id")
            asset = hit.get("asset_id") if media else None
            if turn:
                ranks.setdefault((str(turn), hit.get("source_message_hash") or None, str(asset) if asset else None),
                                 rank)
        self._lane_hits[name] = self._lane_hits.get(name, 0) + count

    def recall(self, collected, *, limit: int | None, max_chars: int) -> None:
        """The recall's scope, read back when the trace is reported (``select_memory``)."""
        self._scope = (collected.ledger, collected.contact_id, collected.session_id)
        self._pack = (limit, max_chars)

    def fused(self, candidates) -> None:
        """The fused candidates, in the order offered to the reranker."""
        for rank, row in enumerate(candidates, 1):
            key = self._see(row)
            if key:
                self._fused.setdefault(key, rank)

    def calibration(self, status: str, cutoff: float | None) -> None:
        """The configured cutoff's check, and the cutoff it applies (None when it applies none)."""
        self._calibration = {"status": status, "cutoff": cutoff}

    def submitted(self, rows, *, bound: int | None) -> None:
        """The reranker was sent ``rows``, at most ``bound`` of the fused order spread across sessions: recorded
        when the call starts, so a call that then times out, fails or answers invalidly still says what it was
        sent."""
        self._sent = [self._see(row) for row in rows]
        self._submitted.update(self._sent)
        self._bound = bound

    def rerank(self, outcome: str, *, mode: str, candidates=(), scores=None, cutoff=None, missed=None,
               late=False) -> None:
        """What came of the rerank of ``candidates``: the ``scores`` of what it was sent (by position) and the
        cutoff it applied, or why it did not score (``missed``)."""
        sent = self._sent
        self._rerank = {"mode": mode if mode in ("shadow", "on") else "off", "outcome": outcome,
                        "missed": missed, "late": bool(late), "submission_bound": self._bound if sent else None,
                        "submitted": len(sent)}
        if outcome not in ("scored", "shadow"):
            return
        for index, key in enumerate(sent):
            if scores is not None and index in scores:
                self._scores[key] = float(scores[index])
        if outcome != "scored":
            return
        read, bound = set(sent), self._bound
        for position, row in enumerate(candidates):
            key = self._see(row)
            if key not in read:
                # Within the bound in the fused order and not sent: the spread across sessions sent others instead.
                within = bound is not None and position < bound
                self._outcome[key] = "over_session_share" if within else "below_submission_bound"
        if cutoff is not None:
            for index, key in enumerate(sent):
                if scores[index] < cutoff:
                    self._outcome[key] = "below_cutoff"

    def packing(self, rows, *, limit: int | None, max_chars: int) -> None:
        """The packer's input, in the order it considers it, and its limit (None: no count limit) and budget (the
        first packing: a recheck packs again)."""
        keys = [self._see(row) for row in rows]
        if self._order is None:
            self._order, self._pack = keys, (limit, max_chars)

    def left(self, rows, reason: str, *, start: int = 0) -> None:
        """``rows[start:]`` left out for ``reason``."""
        for row in islice(rows, start, None):
            self._outcome[self._see(row)] = reason

    def dropped(self, row, reason: str) -> None:
        self._outcome[self._see(row)] = reason

    def shown(self, row, by) -> None:
        """``row``'s words are in the packet through ``by`` (a pair that shows a copy's whole message)."""
        self._outcome[self._see(row)] = ("shown", self._see(by))

    def replaced(self, selected, kept, by) -> None:
        """``by`` took the place of what ``selected`` held and ``kept`` does not."""
        remaining = {id(row) for row in kept}
        for row in selected:
            if id(row) not in remaining:
                self.shown(row, by)

    def packed(self, selected) -> None:
        for row in selected:
            self._outcome[self._see(row)] = "packed"

    def rechecked(self, selected, retained) -> None:
        """What the recheck after ranking kept of ``selected``."""
        kept = {str(row.get("id") or "") for row in retained}
        for row in selected:
            key = self._see(row)
            if key not in kept:
                self._outcome[key] = "removed_by_recheck"

    def packet(self, selected, content: str) -> None:
        """The packet the recall returns (again, when context assembly packs it within the host's allowance)."""
        self._selected = [self._see(row) for row in selected]
        self._digest = hashlib.sha256(content.encode("utf-8")).hexdigest() if content else None

    def context_section(self, body: str | None) -> None:
        """Context assembly: the recalled section's body as returned (None when it was held back)."""
        if self._digest is None:
            self._section = None
        elif body is None:
            self._section = "held_back"
        else:
            same = hashlib.sha256(body.encode("utf-8")).hexdigest() == self._digest
            self._section = "in_context" if same else "shortened"

    # --- reading --------------------------------------------------------------------------------------------------

    def report(self) -> dict:
        """The trace as the API returns it (``RecallTraceReport``)."""
        lanes = dict(self._lane_hits)
        if self._selected is None or self._scope is None:
            # The selection did not complete: there is no packet to explain.
            return {"lanes": lanes, "rerank": self._rerank, "calibration": dict(self._calibration),
                    "packet": None, "selected": [], "rows": [], "rows_omitted": 0, "section": self._section}
        resolved = self._resolve()
        carried = resolved["carried"]
        rows, found, lineage = [], set(), []
        for key in resolved["order"]:
            candidate = self._rows[key]
            exit_, shown_by = resolved["exits"][key]
            turn, digest = _primary(candidate, carried[key])
            image = candidate.get("kind") == "media_description"
            asset = str(candidate.get("asset_id") or "") if image else ""
            ranks = {}
            for lane, hits in self._lanes.items():
                if (lane in self._media_lanes) != image:
                    continue
                matched = [(hit, rank) for hit, rank in hits.items() if _finds(hit, carried[key], asset)]
                found.update((lane, hit) for hit, _ in matched)
                if matched:
                    ranks[lane] = min(rank for _, rank in matched)
            rows.append({"turn_id": turn, "message_hash": digest, "candidate": _shown_id(key), "form": _form(candidate),
                         "lanes": ranks, "fused_rank": self._fused.get(key), "submitted": key in self._submitted,
                         "rerank_score": self._scores.get(key), "exit": exit_,
                         "shown_by": _shown_id(shown_by) if shown_by else None})
            # Its lineage: every message it carries and every source it names.
            lineage.append([*carried[key], *((source, None) for source in _turns(candidate)), (turn, digest)])
        # What a lane found and no candidate carried to ranking.
        uncarried: dict[tuple[str, str | None, str | None], dict[str, int]] = {}
        for lane, hits in self._lanes.items():
            for hit, rank in hits.items():
                if (lane, hit) not in found:
                    uncarried.setdefault(hit, {})[lane] = rank
        for (turn, digest, _), ranks in sorted(uncarried.items(), key=lambda item: (
                min(item[1].values()), item[0][0], item[0][1] or "", item[0][2] or "")):
            rows.append({"turn_id": turn, "message_hash": digest, "candidate": None, "form": None, "lanes": ranks,
                         "fused_rank": None, "submitted": False, "rerank_score": None, "exit": "no_lane",
                         "shown_by": None})
            lineage.append([(turn, digest)])
        # Every id shown is one the recall's own scope can read now: a row whose lineage is not all there any more
        # (a source erased or out of reach, a message erased from a source that stays) is erased and loses its ids.
        ledger, contact_id, session_id = self._scope
        held = _held(ledger, [source for keys in lineage for source, _ in keys], contact_id, session_id)
        gone = set()
        for row, keys in zip(rows, lineage):
            if not all(_holds(held, key) for key in keys):
                if row["candidate"]:
                    gone.add(row["candidate"])
                row.update(turn_id=None, message_hash=None, candidate=None, form=None, lanes={}, fused_rank=None,
                           submitted=False, rerank_score=None, exit="erased", shown_by=None)
            elif row["message_hash"] is not None and not _DIGEST.match(str(row["message_hash"])):
                row["message_hash"] = None
        for row in rows:
            if row["shown_by"] in gone:
                row["shown_by"] = None
        selected = [_shown_id(key) for key in self._selected]
        # ``limit`` None: no count limit (a turn's packet behind a verified cutoff).
        limit, max_chars = self._pack or (0, 0)
        return {"lanes": lanes, "rerank": self._rerank, "calibration": dict(self._calibration),
                "packet": {"limit": limit, "max_chars": max_chars, "count": len(self._selected)},
                "selected": [key for key in selected if key not in gone],
                "rows": rows[:MAX_ROWS], "rows_omitted": max(0, len(rows) - MAX_ROWS), "section": self._section}

    def exit_for(self, turn_id: str, message_hash: str | None = None) -> str | None:
        """Where one message left this recall: None when a candidate in the packet carries it (the same words, from
        the same session and speaker, count wherever they are held), else the exit of the candidate that carried it
        furthest; ``no_lane`` when no candidate carried it to ranking; ``erased`` when its source is no longer
        readable in the recall's scope or no longer holds the message. Without a message hash, any candidate from
        that source counts."""
        if self._selected is None or self._scope is None:
            raise RuntimeError("the recall has not finished: there is no packet to explain")
        resolved = self._resolve()
        wanted = (str(turn_id), message_hash)
        reached = [resolved["exits"][key][0] for key, keys in resolved["carried"].items() if _covers(keys, wanted)]
        found = any(_covers(hits, wanted) for hits in self._lanes.values())
        if not reached and not found:
            return "no_lane"
        ledger, contact_id, session_id = self._scope
        if not _holds(_held(ledger, [wanted[0]], contact_id, session_id), wanted):
            return "erased"
        if not reached:
            return "no_lane"
        return max(reached, key=lambda exit_: _REACH[exit_])

    # --- internals ------------------------------------------------------------------------------------------------

    def _see(self, row) -> str:
        key = str(row.get("id") or "")
        if key and key not in self._rows:
            self._rows[key] = row
        return key

    def _resolve(self) -> dict:
        if self._resolved is not None:
            return self._resolved
        final = set(self._selected or ())
        exits: dict[str, tuple[str | None, str | None]] = {}
        limit = (self._pack or (0, 0))[0]

        def resolve(key, depth=0):
            if key in exits:
                return exits[key]
            outcome = self._outcome.get(key)
            if key in final:
                result = (None, None)
            elif isinstance(outcome, tuple) and depth < 8:
                # Its words are in the packet through the candidate that showed them, if that one is there.
                by = outcome[1]
                exit_, through = resolve(by, depth + 1)
                result = (exit_, (through or by) if exit_ is None else None)
            elif outcome in EXITS:
                result = (outcome, None)
            else:
                # It reached the packer and is not in the packet: the packer leaves a candidate out only when the
                # packet is full or the candidate does not fit.
                full = limit is not None and len(final) >= limit
                result = ("over_packet_limit" if full else "over_char_budget", None)
            exits[key] = result
            return result

        order = list(dict.fromkeys([*(self._order or ()), *sorted(self._fused, key=self._fused.get), *self._rows]))
        for key in order:
            resolve(key)
        self._resolved = {"exits": exits, "order": order,
                          "carried": {key: _carried(row) for key, row in self._rows.items()}}
        return self._resolved


def _held(ledger, turns, contact_id, session_id) -> dict[str, frozenset[str]]:
    """The sources of ``turns`` the scope can read now (not erased, not invalidated, the person's or the session's),
    each with the hashes of the messages it holds now. A source can stay and lose a message: erasing a message also
    takes its copies out of the checkpoints that hold other messages."""
    from colonymind.memory import curated
    turns = list(turns)
    return curated.fenced_read(ledger, lambda: _read_held(ledger, turns, contact_id, session_id),
                              empty={})

def _read_held(ledger, turns, contact_id, session_id) -> dict[str, frozenset[str]]:
    turns = list(dict.fromkeys(str(turn) for turn in turns if turn))
    if ledger is None or not turns:
        return {}
    from colonymind.turns.idempotency import source_message_hash
    from colonymind.memory import curated
    held = {}
    with closing(ledger._connect()) as conn:
        # The scope rule is the ledger's own (``source_references``), read in the same connection as the messages.
        for ref in ledger._source_references(conn, turns, contact_id=contact_id, session_id=session_id):
            row = conn.execute("SELECT session_id, messages_json FROM turn_sources WHERE turn_id=?",
                               (ref["source_id"],)).fetchone()
            if row is not None:
                hashes = frozenset(source_message_hash(row[0], message)
                    for message in json.loads(row[1]) if isinstance(message, dict)
                    and not curated.is_forgotten(ledger, {
                        'source_id': ref['source_id'],
                        'message_hash': source_message_hash(row[0], message)}, conn=conn))
                if hashes:
                    held[ref['source_id']] = hashes
    return held


def _holds(held, key) -> bool:
    """Whether the message ``key`` (a source and a message hash, or a source alone) is readable now."""
    turn, digest = key[0], key[1]
    return bool(turn) and turn in held and (not digest or digest in held[turn])


def _turns(row) -> list[str]:
    turns = [row.get("source_turn_id")]
    for name in ("source_read", "history_anchor"):
        if isinstance(row.get(name), dict):
            turns.append(row[name].get("source_id"))
    turns.extend(row.get("source_turn_ids") or ())
    uri = str(row.get("source_uri") or "")
    if uri.startswith("turn:"):
        turns.append(uri[5:])
    return list(dict.fromkeys(str(turn) for turn in turns if turn))


def _carried(row) -> list[tuple[str, str | None]]:
    """The messages a candidate carries: its exact message memberships, else the sources it names."""
    keys = []
    turn, digest = row.get("source_turn_id"), row.get("source_message_hash")
    if turn and digest:
        keys.append((str(turn), str(digest)))
    for name in ("_source_message_hashes", "_annotation_message_hashes"):
        value = row.get(name)
        if isinstance(value, dict):
            keys.extend((str(source), str(message)) for source, messages in value.items() for message in messages or ())
    if not keys:
        keys = [(source, None) for source in _turns(row)]
    return list(dict.fromkeys(keys))


def _covers(keys, wanted) -> bool:
    """Whether a candidate carrying ``keys`` carries the message ``wanted``: the same message hash (which covers the
    session, the speaker and the words, so a checkpoint's copy is its original's message), or, for a message
    without one, the same source. Keys and ``wanted`` start with a source and a message hash."""
    turn, digest = wanted[0], wanted[1]
    if digest:
        return any(digest == key[1] for key in keys)
    return any(turn == key[0] for key in keys)


def _finds(hit, carried, asset: str) -> bool:
    """Whether a lane's ``hit`` found the candidate: an image lane's hit, the same image (asset); any other, a
    message the candidate carries."""
    if asset and hit[2]:
        return hit[2] == asset
    return _covers(carried, hit)


def _primary(row, keys) -> tuple[str | None, str | None]:
    turns = _turns(row)
    turn = turns[0] if turns else (keys[0][0] if keys else None)
    digest = row.get("source_message_hash") or next(
        (message for source, message in keys if source == turn and message), None)
    return turn, (str(digest) if digest else None)


def _form(row) -> str:
    if row.get("epistemic_state") == "correction_evidence":
        return "correction"
    if row.get("kind") == "conversation_pair":
        return "conversation"
    if row.get("content_format") == "source_assertions_v1":
        return "claims"
    return {"source_quote": "quotation", "media_description": "media", "media_locator": "media_name",
            "belief": "belief"}.get(row.get("kind"), "other")


def _shown_id(key: str) -> str:
    """A candidate id as the trace shows it: as it is when it is a kind and a digest, else a digest of it."""
    return key if _CANDIDATE.match(key) else "candidate:" + hashlib.sha256(key.encode("utf-8")).hexdigest()


__all__ = ["CANDIDATE_ID", "EXITS", "FORMS", "LANE_NAME", "MAX_ROWS", "NULL_TRACE", "Calibration", "Exit", "Form",
           "NullTrace", "RecallTrace", "RerankOutcome", "Section"]
