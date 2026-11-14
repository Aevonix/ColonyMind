"""What memory holds about one person: claim cards, a page at a time, and one claim's own history
(docs/SOURCE-ANNOTATIONS.md, The memory inventory).

A read of existing records, never a new store. A person's source claims are read as recall reads them
(``SourceClaimProjection._rows``: the person's own scope, current attribution, the exact quoted message still
retained, and every claim a claim is built on still readable), less the claims of a source an erasure touched. Each
card carries the claim's state, its quotation and the exact source revision (``source_id``, ``source_version``) the
annotate and forget routes act on.

States follow the claim's dates, as recall reads them: ``retracted`` (a correction or the owner's answer withdrew
it), ``superseded`` (a change replaced it, and the change's date has come), ``expired`` (its stated validity has
ended), ``upcoming`` (its stated validity has not begun: a change dated ahead, until that date), ``in_conflict``
(recall shows it beside a live statement of a different value for the same property, or a change to it waits for the
owner), else ``current``. A claim a change replaces from a later date is live until then, and can be in conflict.
Every claim is formed and reviewed by a model, so it is ``unverified`` unless the owner kept it in an answer
(``owner_confirmed``).

History: the claim's own chain, read from the links the claims already carry (what each claim superseded or
retracted, what superseded or retracted it), with the owner's answers about its claims (the learning log) and the
attributed corrections recorded against the messages that state them. Each step carries the source reference of the
statement that made it. An erased claim or correction is a content-free marker (its id, ``erased``), never its words:
a claim whose row is gone is found through the links other claims keep to it.

Paging: newest first, in the order the claims were formed. A listing covers the claims that existed when its first
page was read. Its ``read_revision`` covers the person's erasures, attribution changes and removed claims, and a next
page asked for after any of them is refused (``InventoryChanged``): a caller holding an earlier page lists again
rather than keep a claim that is gone. Ordinary writes (a new claim, a correction, a change) never move a claim to
another page; each page shows the states as they are when it is read.
"""
from __future__ import annotations

from collections import deque
from contextlib import closing
import hashlib
import json
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from colonymind.util.temporal import now_utc

#: Claims on a page by default, and at most (a builder's choice: a page an operator reads at once; the bound keeps
#: one request's reads small).
PAGE_DEFAULT = 50
PAGE_MAX = 200
#: Claims one history walks (a builder's choice: a chain of changes to one property rarely passes a few; the bound
#: keeps one request's reads small, and ``complete`` says when a history reached it).
HISTORY_LIMIT = 64
#: The ids one query hands to ``_rows``.
_CHUNK = 400
_TOKEN = re.compile(r"^(\d{1,19})\.(\d{1,19})\.([0-9a-f]{64})$")
STATES = ("current", "in_conflict", "upcoming", "expired", "superseded", "retracted")


class InventoryChanged(Exception):
    """A next page asked for after an erasure, an attribution change or a removed claim in the person's memory."""


class InvalidPage(ValueError):
    """A page token this listing did not give."""


def _projection(ledger):
    from colonymind.beliefs.source_projection import SourceClaimProjection
    return SourceClaimProjection(ledger)


def _marks(values) -> str:
    return ",".join("?" for _ in values)


def _chunks(values: Iterable[Any], size: int = _CHUNK):
    values = list(values)
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _revision(conn, person: str, top: int) -> Tuple[str, int]:
    """The listing's revision and the person's erasure watermark. It moves with an erasure, an attribution change
    and any removed claim of the listing's range; a new claim (above ``top``) or a changed state leaves it."""
    watermark = int(conn.execute("SELECT coalesce(max(sequence),0) FROM source_erasures WHERE contact_id=?",
                                 (person,)).fetchone()[0])
    # An attribution change: the replies it invalidated in this person's scope, and every source it moved from or to
    # this person (a claim that was anchored there may stay, moved onto a witness, with the same count).
    moved = int(conn.execute("SELECT count(*) FROM source_attribution_invalidations i JOIN turn_sources s "
                             "ON s.turn_id=i.source_id WHERE s.contact_id=?", (person,)).fetchone()[0])
    moved += int(conn.execute("SELECT count(*) FROM source_attribution_revisions WHERE old_contact_id=? "
                              "OR contact_id=?", (person, person)).fetchone()[0])
    held = int(conn.execute("SELECT count(*) FROM source_claims c JOIN turn_sources s ON s.turn_id=c.turn_id "
                            "WHERE s.contact_id=? AND s.scope='person' AND c.rowid<=?", (person, top)).fetchone()[0])
    curated_head = (int(conn.execute("SELECT coalesce(max(sequence),0) FROM curated_memory_events WHERE operation='forget' AND memory_id LIKE 'sources:%'").fetchone()[0])
                    if conn.execute("SELECT 1 FROM sqlite_master WHERE name='curated_memory_events'").fetchone() else 0)
    digest = hashlib.sha256(json.dumps(["memory-inventory-v1", person, top, watermark, moved, held, curated_head]).encode())
    return digest.hexdigest(), watermark


def _erased_turns(conn, turns: Iterable[str]) -> set:
    erased = set()
    for chunk in _chunks(set(turns)):
        erased.update(row[0] for row in conn.execute(
            f"SELECT DISTINCT turn_id FROM source_projection_erasures WHERE turn_id IN ({_marks(chunk)})", chunk))
    return erased


def visible(conn, person: str, ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """The claims of ``ids`` recall would read in the person's own scope, less those of a source an erasure
    touched, by id."""
    ids = list(dict.fromkeys(ids))
    projection, rows = _projection(None), []
    for chunk in _chunks(ids):
        rows += projection._rows(conn, person, "", ids=chunk, limit=len(chunk) + 1)
    erased = _erased_turns(conn, (row["turn_id"] for row in rows))
    return {row["id"]: row for row in rows if row["turn_id"] not in erased}


def _sources(conn, turns: Iterable[str]) -> Dict[str, Dict[str, str]]:
    from colonymind.turns.idempotency import canonical_turn_digest
    found = {}
    for chunk in _chunks(set(turns)):
        for row in conn.execute("SELECT turn_id, session_id, messages_json FROM turn_sources "
                                f"WHERE turn_id IN ({_marks(chunk)})", chunk):
            found[row["turn_id"]] = {"source_id": row["turn_id"],
                                     "source_version": canonical_turn_digest(json.loads(row["messages_json"])),
                                     "session_id": row["session_id"]}
    return found


def _places(conn, person: str, rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[str]]:
    """Every message that states each claim, by ``(source, message)``, with the claims it states: the claim's
    anchor, and each witness message of a folded claim still retained in the person's own scope (current
    attribution, no erasure touched its source), as recall reads a folded claim through every message that states
    it. A correction recorded against an earlier statement stays the claim's when a repeat moves its anchor."""
    from colonymind.turns.idempotency import source_message_hash
    places: Dict[Tuple[str, str], List[str]] = {}
    for row in rows:
        places.setdefault((row["turn_id"], row["message_hash"]), []).append(row["id"])
    retained: Dict[str, set] = {}
    for chunk in _chunks({row["id"] for row in rows}):
        for edge in conn.execute(
                "SELECT w.claim_id, w.turn_id, w.message_hash, s.session_id, s.messages_json "
                "FROM source_claim_witnesses w JOIN turn_sources s ON s.turn_id=w.turn_id "
                "WHERE s.contact_id=? AND s.scope='person' "
                "AND NOT EXISTS (SELECT 1 FROM source_projection_erasures e WHERE e.turn_id=s.turn_id) "
                "AND NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=s.turn_id) "
                f"AND w.claim_id IN ({_marks(chunk)}) ORDER BY w.recorded_at, w.turn_id", [person, *chunk]):
            if edge["turn_id"] not in retained:
                retained[edge["turn_id"]] = {source_message_hash(edge["session_id"], message)
                                             for message in json.loads(edge["messages_json"])}
            if edge["message_hash"] in retained[edge["turn_id"]]:
                stated = places.setdefault((edge["turn_id"], edge["message_hash"]), [])
                if edge["claim_id"] not in stated:
                    stated.append(edge["claim_id"])
    return places


def owner_answers(conn, ids: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
    """The owner's answers recorded about each claim (the learning log, stage ``owner_answer``), oldest first."""
    found: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in _chunks(set(ids)):
        for row in conn.execute(
                "SELECT item_id, action, at, refs_json FROM learning_log WHERE store='claim' AND stage='owner_answer' "
                f"AND item_id IN ({_marks(chunk)}) ORDER BY id", chunk):
            refs = json.loads(row["refs_json"] or "[]")
            found.setdefault(row["item_id"], []).append(
                {"action": row["action"], "at": _when(row["at"]), "ref": refs[0] if refs else None})
    return found


def _when(epoch: Any) -> Optional[str]:
    from datetime import timezone
    try:
        return datetime.fromtimestamp(float(epoch), timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _ended(row: Dict[str, Any], now: datetime) -> bool:
    from colonymind.beliefs.source_time import utc_timestamp
    end = utc_timestamp(row.get("valid_to"))
    return end is not None and end <= now


def _begun(row: Dict[str, Any], now: datetime) -> bool:
    from colonymind.beliefs.source_time import utc_timestamp
    start = utc_timestamp(row.get("valid_from"))
    return start is None or start <= now


def _live(row: Dict[str, Any], now: datetime) -> bool:
    """Read by recall as current at ``now``: not retracted, and inside its stated validity. A change that
    supersedes it from a later date ends that validity on that date, not before."""
    return not row.get("retracted_by") and _begun(row, now) and not _ended(row, now)


def _live_values(conn, person: str, key: Tuple[str, str], query) -> set:
    """The distinct literal values of one property live at the query's time, as recall reads them (its current-time
    read with each value once, the newest statement of it; quoted preferences never compared), less the claims of a
    source an erasure touched. Every claim of the property is read, a chunk at a time, so one value said many times
    in a store that does not fold restatements never crowds out another."""
    from colonymind.beliefs.source_claims import norm_value
    projection = _projection(None)
    ids = [row[0] for row in conn.execute(
        "SELECT c.id FROM source_claims c JOIN turn_sources s ON s.turn_id=c.turn_id WHERE s.contact_id=? "
        "AND s.scope='person' AND c.subject_key=? AND c.predicate=? AND c.retracted_by IS NULL "
        "AND NOT EXISTS (SELECT 1 FROM source_projection_erasures e WHERE e.turn_id=c.turn_id) ORDER BY c.rowid",
        (person, *key))]
    values = set()
    for chunk in _chunks(ids):
        values.update(norm_value(row["value"]) for row in projection._rows(
            conn, person, "", ids=chunk, key=key, time_query=query, distinct_values=True, limit=len(chunk) + 1)
            if row.get("representation") != "preference")
    return values


def _conflicted(conn, person: str, rows: List[Dict[str, Any]], now: datetime) -> set:
    """The live claims recall shows beside a live statement of a different value for the same property (its
    ``unresolved_conflict`` rule: literal values that differ, with overlapping validity, quoted preferences never
    compared), and both sides of a change to a claim that waits for the owner. Every claim live now overlaps every
    other at now, so a property with two live values puts every live card of either value in conflict."""
    from colonymind.beliefs.source_claims import norm_value
    from colonymind.beliefs.source_time import MemoryTimeQuery
    query = MemoryTimeQuery("current", now.isoformat())
    live = [row for row in rows if _live(row, now) and row.get("representation") != "preference"]
    conflicted = set()
    for key in {(row["subject_key"], row["predicate"]) for row in live}:
        values = _live_values(conn, person, key, query)
        if len(values) > 1:
            conflicted.update(row["id"] for row in live if (row["subject_key"], row["predicate"]) == key
                              and norm_value(row["value"]) in values)
    for row in conn.execute("SELECT target_id, by_id FROM learning_pending WHERE store='claim' AND status='pending' "
                            "AND contact_id=?", (person,)):
        conflicted.update((row[0], row[1]))
    return conflicted


def state_of(row: Dict[str, Any], now: datetime, conflicted: set) -> str:
    if row.get("retracted_by"):
        return "retracted"
    if _ended(row, now):
        return "superseded" if row.get("superseded_by") else "expired"
    if not _begun(row, now):
        return "upcoming"
    return "in_conflict" if row["id"] in conflicted else "current"


def cards(conn, person: str, rows: List[Dict[str, Any]], now: datetime) -> List[Dict[str, Any]]:
    """One card per claim row (``visible``), in the order given."""
    if not rows:
        return []
    sources = _sources(conn, (row["turn_id"] for row in rows))
    notes = _correction_steps(conn, person, rows)
    answers = owner_answers(conn, (row["id"] for row in rows))
    conflicted = _conflicted(conn, person, rows, now)
    made = []
    for row in rows:
        kept = answers.get(row["id"], [])
        card = {
            "claim_id": row["id"], "state": state_of(row, now, conflicted),
            "subject": row.get("subject"), "predicate": row.get("predicate"), "value": row.get("value"),
            "quote": row.get("evidence"),
            "verification": "owner_confirmed" if kept and kept[-1]["action"] == "ratified" else "unverified",
            "source": {**sources[row["turn_id"]], "message_hash": row["message_hash"]},
            "said_at": row.get("observed_at"), "recorded_at": row.get("recorded_at"),
            "valid_from": row.get("valid_from"), "valid_to": row.get("valid_to"),
            "operation": row.get("operation") or "assert",
            "prior_claim_id": row.get("prior_claim_id"),
            "superseded_by": row.get("superseded_by"), "retracted_by": row.get("retracted_by"),
            "corrections": sum(1 for note in notes.get(row["id"], []) if not note["by"].get("erased")),
        }
        for key in ("representation", "epistemic_state", "source_modality", "witnesses"):
            if row.get(key):
                card[key] = row[key]
        made.append(card)
    return made


def _collect(conn, person: str, *, top: int, before: int, limit: int) -> Tuple[List[Dict[str, Any]], int, bool]:
    """Up to ``limit`` readable claims below ``before`` (and at most ``top``), newest first; the position the next
    page starts from, and whether a readable claim remains there."""
    rows: List[Dict[str, Any]] = []
    scan = last = before
    batch_size = min(_CHUNK, max(2 * limit, 64))
    while True:
        batch = conn.execute(
            "SELECT c.rowid, c.id FROM source_claims c JOIN turn_sources s ON s.turn_id=c.turn_id "
            "WHERE s.contact_id=? AND s.scope='person' AND c.rowid<? AND c.rowid<=? ORDER BY c.rowid DESC LIMIT ?",
            (person, scan, top, batch_size)).fetchall()
        if not batch:
            return rows, last, False
        readable = visible(conn, person, (row[1] for row in batch))
        for seq, ident in batch:
            scan = seq
            if ident not in readable:
                continue
            if len(rows) == limit:
                return rows, last, True
            rows.append(readable[ident])
            last = seq


def page(ledger, person: str, *, token: Optional[str]=None, limit: int=PAGE_DEFAULT, now: Optional[datetime]=None) -> Dict[str, Any]:
    """One page of the person's claim cards. ``token``: a ``next_page`` this listing gave. Raises ``InvalidPage``
    for a token it did not give and ``InventoryChanged`` when the listing must start again."""
    from colonymind.memory import curated
    return curated.fenced_read(ledger, lambda: _read_page(ledger, person, token=token, limit=limit, now=now),
                              empty={'person_id': person, 'claims': [], 'read_revision': '', 'erasure_watermark': curated.watermark(ledger, person), 'limit': int(limit), 'next_page': None, 'complete': False})

def _read_page(ledger, person: str, *, token: Optional[str] = None, limit: int = PAGE_DEFAULT,
         now: Optional[datetime] = None) -> Dict[str, Any]:
    if not 1 <= int(limit) <= PAGE_MAX:
        raise InvalidPage("invalid_limit")
    now = now or now_utc()
    with closing(ledger._connect()) as conn:
        conn.execute("BEGIN")            # one snapshot: the revision and the page are read together
        try:
            if token is None:
                top = int(conn.execute("SELECT coalesce(max(rowid),0) FROM source_claims").fetchone()[0])
                before, expected = top + 1, None
            else:
                match = _TOKEN.fullmatch(token) if isinstance(token, str) else None
                if match is None or int(match[2]) > int(match[1]) + 1:
                    raise InvalidPage("invalid_page")
                top, before, expected = int(match[1]), int(match[2]), match[3]
            revision, watermark = _revision(conn, person, top)
            if expected is not None and expected != revision:
                raise InventoryChanged("inventory_changed_restart")
            rows, cursor, more = _collect(conn, person, top=top, before=before, limit=int(limit))
            made = cards(conn, person, rows, now)
        finally:
            if conn.in_transaction:
                conn.execute("COMMIT")
    return {"person_id": person, "claims": made, "read_revision": revision, "erasure_watermark": watermark,
            "limit": int(limit), "next_page": f"{top}.{cursor}.{revision}" if more else None, "complete": not more}


# -- one claim's own history ------------------------------------------------------------------------------------------

def claim_person(ledger, claim_id: str) -> Optional[str]:
    """The person whose own scope holds ``claim_id`` (its anchor source's), or None."""
    with closing(ledger._connect()) as conn:
        row = conn.execute("SELECT s.contact_id FROM source_claims c JOIN turn_sources s ON s.turn_id=c.turn_id "
                           "WHERE c.id=? AND s.scope='person'", (claim_id,)).fetchone()
    return row[0] if row else None


def _ended_ones(conn, person: str, ident: str) -> List[Tuple[str, str]]:
    """The claims of the person's scope that ``ident`` ended, oldest first, each with how."""
    from colonymind.memory import curated
    return [(row[0], "retracted" if row[1] == ident else "superseded") for row in conn.execute(
        "SELECT c.id, c.retracted_by FROM source_claims c JOIN turn_sources s ON s.turn_id=c.turn_id "
        "WHERE s.contact_id=? AND s.scope='person' AND (c.superseded_by=? OR c.retracted_by=?) ORDER BY c.rowid",
        (person, ident, ident)) if not curated.is_forgotten(conn, {'source_claim_id': row[0]})]


def _exists(conn, ident: str) -> bool:
    from colonymind.memory import curated
    if curated.is_forgotten(conn, {'source_claim_id': ident}):
        return False
    return conn.execute("SELECT 1 FROM source_claims WHERE id=?", (ident,)).fetchone() is not None


def _record(conn, ident: str) -> Optional[Dict[str, Any]]:
    """The links of a claim still held but not readable in the person's scope (never its words), or None once its
    row is gone (erased)."""
    from colonymind.memory import curated
    if curated.is_forgotten(conn, {'source_claim_id': ident}):
        return None
    raw = conn.execute("SELECT data_json, valid_from, superseded_by, retracted_by FROM source_claims WHERE id=?",
                       (ident,)).fetchone()
    if raw is None:
        return None
    data = json.loads(raw[0])
    return {"operation": data.get("operation"), "prior_claim_id": data.get("prior_claim_id"),
            "valid_from": raw[1], "superseded_by": raw[2], "retracted_by": raw[3]}


def _applied(conn, prior: str, by: str, operation: Any, valid_from: Any) -> bool:
    """Whether ``by``'s change or correction of ``prior`` took effect, read from what is kept once ``prior``'s row
    (which held the link) is gone: a correction, or a change with a date (commit applies nothing else), that was
    not held for the owner or that the owner ratified. A held change the owner has not ratified retired nothing:
    its pointer only names what it proposed."""
    if not (operation == "correct" or (operation == "change" and valid_from)):
        return False
    held = [row[0] for row in conn.execute(
        "SELECT status FROM learning_pending WHERE store='claim' AND target_id=? AND by_id=?", (prior, by))]
    return not held or "ratified" in held


def _ended_by(conn, person: str, ident: str, row: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """What ended ``ident`` (``row``: its record, None once erased): a claim (``superseded`` or ``retracted``), a
    reference that is not a claim (an owner's answer: ``reference``), or nothing. An erased claim's own link went
    with its row: it is ended by the one claim that names it as the prior it changed or corrected, when that change
    took effect (``_applied``); without one, what ended it is not known (``unrecovered``), which is not the same as
    nothing."""
    if row is not None:
        link = row.get("retracted_by") or row.get("superseded_by")
        if not link:
            return None, None
        if not str(link).startswith("claim:"):
            return str(link), "reference"
        return str(link), "retracted" if row.get("retracted_by") else "superseded"
    from colonymind.memory import curated
    found = [(by, operation) for by, operation, valid_from in conn.execute(
        "SELECT c.id, json_extract(c.data_json,'$.operation'), c.valid_from FROM source_claims c JOIN turn_sources s "
        "ON s.turn_id=c.turn_id WHERE s.contact_id=? AND s.scope='person' "
        "AND json_extract(c.data_json,'$.prior_claim_id')=? "
        "AND json_extract(c.data_json,'$.operation') IN ('change','correct') ORDER BY c.rowid",
        (person, ident)).fetchall() if _applied(conn, ident, by, operation, valid_from)
        and not curated.is_forgotten(conn, {'source_claim_id': by})]
    if len(found) != 1:
        return None, "unrecovered"
    return found[0][0], "retracted" if found[0][1] == "correct" else "superseded"


def _invalidated(conn, ids: Iterable[str]) -> set:
    found = set()
    for chunk in _chunks(set(ids)):
        found.update(row[0] for row in conn.execute(
            f"SELECT source_id FROM source_attribution_invalidations WHERE source_id IN ({_marks(chunk)})", chunk))
    return found


def _versions(conn, turns: Iterable[str]) -> Dict[str, str]:
    """The current revision of each source still held."""
    from colonymind.turns.idempotency import canonical_turn_digest
    found = {}
    for chunk in _chunks(set(turns)):
        for row in conn.execute(f"SELECT turn_id, messages_json FROM turn_sources WHERE turn_id IN ({_marks(chunk)})",
                                chunk):
            found[row[0]] = canonical_turn_digest(json.loads(row[1]))
    return found


def _correction_steps(conn, person: str, rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """The attributed corrections of each claim, once each: those recorded against a message that states it
    (``_places``), and in turn those recorded against such a correction (recall's expansion follows a correction as
    it does any source). Each has its words and source, and the source it corrects (``corrects``), or is a
    content-free marker once the correction is erased, no longer this person's, or was recorded against a revision
    of its target that is gone; the walk never goes past a marker."""
    from colonymind.turns.idempotency import canonical_turn_digest
    from colonymind.memory import curated
    found: Dict[str, List[Dict[str, Any]]] = {}
    seen = set()
    versions: Dict[str, str] = {}
    # What to look for: (the source, the message recorded against, or None for any of its messages) -> the claims.
    targets: Dict[Tuple[str, Optional[str]], List[str]] = dict(_places(conn, person, rows))
    while targets:
        relations = []
        for chunk in _chunks({turn for turn, _ in targets}):
            relations += conn.execute(
                "SELECT a.annotation_source_id, a.target_source_id, a.target_version, h.value, s.messages_json, "
                "s.contact_id FROM source_annotations a JOIN json_each(a.target_message_hashes_json) h "
                "LEFT JOIN turn_sources s ON s.turn_id=a.annotation_source_id "
                f"WHERE a.target_source_id IN ({_marks(chunk)}) ORDER BY a.annotation_source_id", chunk).fetchall()
        moved = _invalidated(conn, (relation[0] for relation in relations))
        versions.update(_versions(conn, {relation[1] for relation in relations} - set(versions)))
        following: Dict[Tuple[str, Optional[str]], List[str]] = {}
        for annotation, target, target_version, message_hash, messages_json, contact in relations:
            for ident in targets.get((target, message_hash), []) + targets.get((target, None), []):
                if (ident, annotation) in seen:
                    continue
                seen.add((ident, annotation))
                step: Dict[str, Any] = {"event": "corrected", "claim_id": ident, "corrects": target,
                                        "by": {"source_id": annotation, "erased": True}}
                if (messages_json and contact == person and annotation not in moved
                        and versions.get(target) == target_version
                        and not curated.is_forgotten(conn, annotation)
                        and not curated.is_forgotten(conn, {'source_id': target, 'message_hash': message_hash})):
                    try:
                        messages = json.loads(messages_json)
                        content = json.loads(messages[0]["content"])
                        step = {"event": "corrected", "claim_id": ident, "corrects": target,
                                "at": content.get("recorded_at"), "by": {
                                    "source": {"source_id": annotation,
                                               "source_version": canonical_turn_digest(messages)},
                                    "excerpt": content.get("excerpt"), "correction": content.get("correction"),
                                    "author": content.get("author_principal"),
                                    "recorded_at": content.get("recorded_at")}}
                        following.setdefault((annotation, None), []).append(ident)
                    except (ValueError, LookupError, TypeError, AttributeError):
                        pass
                found.setdefault(ident, []).append(step)
        targets = following
    return found


def _proposals(conn, person: str, ident: str) -> List[Tuple[str, str]]:
    """The changes to a quoted preference held for the owner (``learning_pending``) that ``ident`` is either end of,
    whatever became of them: (the claim it would retire, the claim that proposes it)."""
    return [(row[0], row[1]) for row in conn.execute(
        "SELECT target_id, by_id FROM learning_pending WHERE store='claim' AND contact_id=? "
        "AND (target_id=? OR by_id=?) ORDER BY created_at", (person, ident, ident))]


def _preference_changes(conn, person: str, ids: Iterable[str]) -> List[Tuple[str, str, str, Optional[str]]]:
    """Changes to a quoted preference held for the owner (``learning_pending``) that a claim of ``ids`` is either
    end of: (the claim it would retire, the claim that proposes it, its status, when it was proposed or decided)."""
    found: Dict[Tuple[str, str], Tuple[str, str, str, Optional[str]]] = {}
    for chunk in _chunks(set(ids)):
        for row in conn.execute(
                "SELECT target_id, by_id, status, created_at, decided_at FROM learning_pending WHERE store='claim' "
                f"AND contact_id=? AND (target_id IN ({_marks(chunk)}) OR by_id IN ({_marks(chunk)})) "
                "ORDER BY created_at", [person, *chunk, *chunk]):
            found[(row[0], row[1])] = (row[0], row[1], row[2], _when(row[4] if row[4] is not None else row[3]))
    return list(found.values())


def _steps(conn, person: str, rows: Dict[str, Optional[Dict[str, Any]]], depth: Dict[str, int],
           ended: List[Tuple[str, str, str]], now: datetime) -> List[Dict[str, Any]]:
    changes = _preference_changes(conn, person, rows)
    proposers = [by for _, by, _, _ in changes if by not in rows]
    shown = {**{ident: row for ident, row in rows.items() if row is not None},
             **visible(conn, person, proposers)}
    made = {card["claim_id"]: card for card in cards(conn, person, list(shown.values()), now)}
    seq = {}
    for chunk in _chunks(rows):
        seq.update((row[0], row[1]) for row in conn.execute(
            f"SELECT id, rowid FROM source_claims WHERE id IN ({_marks(chunk)})", chunk))
    answers = owner_answers(conn, rows)
    notes = _correction_steps(conn, person, [row for row in rows.values() if row is not None])

    def claim(ident: str) -> Dict[str, Any]:
        return made[ident] if ident in made else {"claim_id": ident, "erased": True}

    steps: List[Dict[str, Any]] = []
    for ident in sorted(rows, key=lambda item: (depth[item], seq.get(item, 0), item)):
        into = sorted(((old, kind) for old, new, kind in ended if new == ident and kind != "reference"),
                      key=lambda item: (seq.get(item[0], 0), item[0]))
        if not into:
            steps.append({"event": "stated", "claim_id": ident,
                          **({"claim": made[ident]} if ident in made else {"erased": True})})
        steps += [{"event": kind, "claim_id": old, "by": claim(ident)} for old, kind in into]
        attached = list(notes.get(ident, []))
        attached += [{"event": "kept_by_owner", "claim_id": ident, "at": answer["at"],
                      "by": {"ref": answer["ref"], "owner_answer": True}}
                     for answer in answers.get(ident, []) if answer["action"] == "ratified"]
        # A change is a step of the claim it would retire (of its proposer when the walk's bound left that out).
        attached += [{"event": "preference_change", "claim_id": target, "status": status, "at": at, "by": claim(by)}
                     for target, by, status, at in changes if (target if target in rows else by) == ident]
        attached.sort(key=lambda step: step.get("at") or "")
        steps += attached
        for old, link, kind in ended:
            if old == ident and kind == "reference":
                declined = [answer for answer in answers.get(ident, [])
                            if answer["action"] == "declined" and answer["ref"] == link]
                steps.append({"event": "retracted", "claim_id": ident,
                              **({"at": declined[-1]["at"]} if declined else {}),
                              "by": {"ref": link, "owner_answer": bool(declined)}})
    return steps


def history(ledger, person: str, claim_id: str, *, now: Optional[datetime]=None) -> Optional[Dict[str, Any]]:
    """The claim's own chain, first statement first, or None when the claim is not readable in the person's
    scope. ``complete`` is False when the chain may go on past what is shown: the walk reached ``HISTORY_LIMIT``
    claims (``limit_reached``), or an erased claim's link on one side is gone with it and no retained claim
    establishes it (``gaps``: the claim and the side, ``before`` or ``after``). An erased claim is never taken for
    the chain's first or last statement: only a claim whose record is held can show it is one."""
    from colonymind.memory import curated
    return curated.fenced_read(ledger, lambda: _read_history(ledger, person, claim_id, now=now),
                              empty=None)

def _read_history(ledger, person: str, claim_id: str, *, now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    now = now or now_utc()
    with closing(ledger._connect()) as conn:
        conn.execute("BEGIN")
        try:
            first = visible(conn, person, [claim_id]).get(claim_id)
            if first is None:
                return None
            rows: Dict[str, Optional[Dict[str, Any]]] = {claim_id: first}   # None: erased or no longer readable
            depth = {claim_id: 0}
            ended: List[Tuple[str, str, str]] = []
            gaps: List[Tuple[str, str]] = []
            limit_reached = False

            def admit(ident: str, level: int) -> bool:
                nonlocal limit_reached
                if ident in rows:
                    return True
                if len(rows) >= HISTORY_LIMIT:
                    limit_reached = True
                    return False
                rows[ident], depth[ident] = visible(conn, person, [ident]).get(ident), level
                return True

            links: set = set()

            def link(old: str, new: str, kind: str) -> None:
                if (old, new, kind) not in links:
                    links.add((old, new, kind))
                    ended.append((old, new, kind))

            # One walk over the chain, both ways from every claim it reaches: what each claim ended and what ended
            # it. A claim can end several (a correction retracts every row of a value), so the walk from any claim
            # of the chain reaches the same claims.
            queue = deque([claim_id])
            while queue:
                ident = queue.popleft()
                row = rows[ident] if rows[ident] is not None else _record(conn, ident)
                before = _ended_ones(conn, person, ident)
                proposals = _proposals(conn, person, ident)
                prior = row.get("prior_claim_id") if row and row.get("operation") in ("change", "correct") else None
                if (prior and prior not in {old for old, _ in before} and not _exists(conn, prior)
                        and _applied(conn, prior, ident, row["operation"], row.get("valid_from"))):
                    before.insert(0, (prior, "retracted" if row["operation"] == "correct" else "superseded"))
                if row is None and not before and not any(by == ident for _, by in proposals):
                    # Erased, no retained claim names it as what ended that claim, and it proposed no held change
                    # (which names its own prior): whether it replaced an earlier claim went with its row.
                    gaps.append((ident, "before"))
                for old, kind in before:
                    new = old not in rows
                    if admit(old, depth[ident] - 1):
                        link(old, ident, kind)
                        if new:
                            queue.append(old)
                # A change to a quoted preference held for the owner joins its two statements whatever the
                # owner made of it, so the history is the same from either; it ends one only once applied.
                for target, by in proposals:
                    other, level = (by, depth[ident] + 1) if target == ident else (target, depth[ident] - 1)
                    new = other not in rows
                    if admit(other, level) and new:
                        queue.append(other)
                after, kind = _ended_by(conn, person, ident, row)
                if kind == "unrecovered":
                    gaps.append((ident, "after"))
                if after is None:
                    continue
                if kind == "reference":
                    link(ident, after, kind)
                    continue
                new = after not in rows
                if admit(after, depth[ident] + 1):
                    link(ident, after, kind)
                    if new:
                        queue.append(after)
            steps = _steps(conn, person, rows, depth, ended, now)
        finally:
            if conn.in_transaction:
                conn.execute("COMMIT")
    gaps.sort(key=lambda gap: (depth[gap[0]], gap[0], gap[1] == "after"))
    return {"person_id": person, "claim_id": claim_id, "steps": steps,
            "complete": not limit_reached and not gaps, "limit_reached": limit_reached,
            "gaps": [{"claim_id": ident, "side": side} for ident, side in gaps]}


__all__ = ["HISTORY_LIMIT", "InvalidPage", "InventoryChanged", "PAGE_DEFAULT", "PAGE_MAX", "STATES", "cards",
           "claim_person", "history", "owner_answers", "page", "state_of", "visible"]
