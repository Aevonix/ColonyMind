"""Cross-channel communication ledger + outreach governance.

Gives the agent the WHOLE picture of its relationship traffic with a contact —
every inbound/outbound exchange across every channel (WhatsApp now, email/SMS
later) — and a principled decision on whether / how / when to (re)initiate, so it
never spams and always references the last discussion + open follow-ups before
reaching out. ``evaluate_outreach`` is the social drive's timing policy;
permission is ``contacts.may_contact``, decided by the mind's authority.
"""
from __future__ import annotations

from colonymind.memory.producers import ReadableConnection

import copy
import json
import logging
import os
from pathlib import Path
import sqlite3
import time
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from colonymind.contacts.source_lineage import SourceLinkedStore

logger = logging.getLogger(__name__)


def _now() -> datetime:
    """UTC now from ``time.time``, the one clock contact stamps and the mind share."""
    return datetime.fromtimestamp(time.time(), timezone.utc)


def _parse(ts: Any) -> Optional[datetime]:
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


# ``external_ref`` of a message the mind itself sent: ``mind:<intention type>:<intention id>``.
MIND_REF = "mind:"
# A conversation row: a message in or out, not a delivery receipt, the mind's own record or a system or automated
# row (``history(conversation_only=True)``, the contacts' reply-timing learner).
CONVERSATION = ("direction IN ('in','out') AND coalesce(receipt_ref,'')=''"
                " AND lower(coalesce(external_ref,'')) NOT GLOB 'mind:*'"
                " AND lower(coalesce(external_ref,'')) NOT GLOB 'system:*'"
                " AND lower(coalesce(external_ref,'')) NOT GLOB 'automated:*'")


class CommsLog(SourceLinkedStore):
    """SQLite ledger of communications with each contact, across all channels."""

    _conn = ReadableConnection()

    def __init__(self, db_path: str, *, source_ledger=None) -> None:
        self._db_path = str(db_path)
        self._source_ledger = source_ledger
        # (erasure mark, erased pairs, checked turns): refreshed for new erasures or linked turns.
        self._erased: Optional[tuple] = None
        self._conn = sqlite3.connect(str(db_path), uri=True)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS communications (
                id TEXT PRIMARY KEY,
                contact_id TEXT NOT NULL,
                channel TEXT NOT NULL DEFAULT 'unknown',
                direction TEXT NOT NULL,        -- 'in' (they->us) | 'out' (us->them)
                summary TEXT,
                session_id TEXT,
                ts TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_comms_contact ON communications(contact_id, ts);
            """
        )
        existing = {str(row[1]) for row in self._conn.execute(
            "PRAGMA table_info(communications)").fetchall()}
        for name, sql_type in {
            "external_ref": "TEXT",
            "reply_to_ref": "TEXT",
            "reaction": "TEXT",
            "receipt_ref": "TEXT",
            "outbound_ref": "TEXT",
            "source_lineage_json": "TEXT",
        }.items():
            if name not in existing:
                self._conn.execute(
                    f"ALTER TABLE communications ADD COLUMN {name} {sql_type}")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_comms_reply_ref "
            "ON communications(contact_id,reply_to_ref,ts)")
        self._conn.execute("""CREATE INDEX IF NOT EXISTS idx_comms_source_lineage
            ON communications(contact_id,json_extract(source_lineage_json,'$.turn_id'))
            WHERE source_lineage_json IS NOT NULL""")
        self._conn.commit()
        from .transport_ingress import TransportIngress, ensure_schema
        ensure_schema(self._conn)
        self._transport_ingress = TransportIngress(self._conn)

    def _ledger(self):
        # An offline copy can belong to another profile. Missing local source
        # rows are not evidence that its linked communications were erased.
        if self._source_ledger is None:
            raise ValueError('communications_source_ledger_required')
        return self._source_ledger

    def _erased_pairs(self, turn_ids) -> frozenset:
        """Erased (contact, turn) pairs relevant to this communications ledger.

        Direct erasures are small; projection tombstones can number in the millions.
        Probe their existing turn-id index for exactly the caller's selected turns,
        and require cache coverage for every one of those turns.
        """
        ledger = self._ledger()
        linked_turns = frozenset(turn_ids)
        try:
            stat = os.stat(ledger.db_path)
            place = (id(ledger), str(ledger.db_path), stat.st_dev, stat.st_ino)
        except (AttributeError, OSError):
            place = (id(ledger), None, None, None)
        with closing(ledger._connect()) as conn:
            # The newest tombstones name themselves, so another ledger with the same counts has another mark.
            mark = place + tuple(conn.execute(
                'SELECT (SELECT max(sequence) FROM source_erasures),'
                ' (SELECT turn_id FROM source_erasures ORDER BY sequence DESC LIMIT 1),'
                ' (SELECT count(*) FROM source_erasure_revisions),'
                ' (SELECT max(rowid) FROM source_projection_erasures),'
                ' (SELECT turn_id || char(31) || source_turn_id FROM source_projection_erasures'
                '  ORDER BY rowid DESC LIMIT 1)').fetchone())
            cached = self._erased
            if cached is not None and cached[0] == mark and linked_turns <= cached[2]:
                return cached[1]
            erased = set(tuple(row) for row in conn.execute(
                '''SELECT e.contact_id,coalesce(r.source_turn_id,e.turn_id)
                FROM source_erasures e LEFT JOIN source_erasure_revisions r USING(sequence)
                WHERE r.whole_source IS NOT 1 OR e.turn_id=r.source_turn_id'''))
            contacts_by_source = {}
            for contact, turn in erased:
                contacts_by_source.setdefault(turn, set()).add(contact)
            turns = list(linked_turns)
            for start in range(0, len(turns), 400):
                batch = turns[start:start + 400]
                for turn, source in conn.execute(
                    'SELECT p.turn_id,p.source_turn_id FROM source_projection_erasures p'
                    ' WHERE p.turn_id IN (' + ','.join('?' for _ in batch) + ')', batch):
                    erased.update((contact, turn) for contact in contacts_by_source.get(source, ()))
            erased = frozenset(erased)
        self._erased = (mark, erased, linked_turns)
        return erased

    def _reconcile_erasure_ids(self, contact_id=None) -> None:
        """Remove the linked summaries of erased sources: the linked rows are few, the erasures many."""
        where, params = (' AND contact_id=?', (contact_id,)) if contact_id else ('', ())
        linked = self._conn.execute("SELECT id,contact_id,json_extract(source_lineage_json,'$.turn_id')"
                                    ' FROM communications WHERE source_lineage_json IS NOT NULL' + where,
                                    params).fetchall()
        if not linked:
            return
        erased = self._erased_pairs(row[2] for row in linked)
        doomed = [tuple(row) for row in linked if (row[1], row[2]) in erased]
        if doomed:
            # The erased pair stays in the predicate: a row moved to another contact since the scan is kept.
            self._conn.executemany("DELETE FROM communications WHERE id=? AND contact_id=?"
                                   " AND json_extract(source_lineage_json,'$.turn_id')=?", doomed)

    def _reconciled_rows(self, query, params, *, contact_id=None):
        """Select, clean and read one communications snapshot.

        Reserve the writer before selecting linked rows. Otherwise a cleaner
        and a stale writer can change that selection, or insert unchecked rows
        before the aggregate. SQLite's connection timeout bounds admission.
        """
        from colonymind.memory import producers
        with self._conn:
            if not self._conn.in_transaction:
                self._conn.execute('BEGIN' if producers.active() else 'BEGIN IMMEDIATE')
            self._reconcile_erasure_ids(contact_id)
            return self._conn.execute(query, params).fetchall()

    def _summary_rows(self, query, params, fields, *, contact_id=None):
        from colonymind.memory import curated
        return curated.fenced_read(self._source_ledger,
            lambda: self._current_summary_rows(query, params, fields, contact_id=contact_id), empty=[])

    def _current_summary_rows(self, query, params, fields, *, contact_id=None):
        from colonymind.memory import producers
        from colonymind.memory import curated
        def current(rows):
            linked = [row for row in rows if row['source_lineage_json'] is not None]
            if not linked:
                return rows
            hidden = curated.forgotten_mask(self._ledger(),
                (json.loads(row['source_lineage_json']) for row in linked))
            rejected = {row['id'] for row, withdrawn in zip(linked, hidden) if withdrawn}
            return [row for row in rows if row['id'] not in rejected]
        if producers.active():
            # Read validity without the writer's opportunistic erasure cleanup.
            rows = current(self._conn.execute(query, params).fetchall())
            # Use lineage from these rows, never a second communications scan:
            # another writer may delete/replace their ids after selection.
            lineage = {row['id']: json.loads(row['source_lineage_json']).get('turn_id')
                       for row in rows if row['source_lineage_json'] is not None}
            erased = self._erased_pairs(lineage.values()) if lineage else ()
            rows = [row for row in rows if row['source_lineage_json'] is None or
                    (lineage.get(row['id']) is not None and
                     (row['contact_id'], lineage[row['id']]) not in erased)]
            invalid = self._invalid_sources([row for row in rows if row['source_lineage_json'] is not None])
            rejected = {row['id'] for row in invalid}
            return [{key: row[key] for key in fields} for row in rows if row['id'] not in rejected]
        rows = current(self._reconciled_rows(query, params, contact_id=contact_id))
        invalid = self._invalid_sources([row for row in rows if row['source_lineage_json'] is not None])
        rejected = {row['id'] for row in invalid}
        with self._conn:
            self._conn.executemany('DELETE FROM communications WHERE id=?', [(key,) for key in rejected])
        return [{key: row[key] for key in fields} for row in rows if row['id'] not in rejected]

    def purge_erased_sources(self, turn_ids=None, *, contact_id=None) -> int:
        sql = ('SELECT id,contact_id,source_lineage_json FROM communications '
               'WHERE source_lineage_json IS NOT NULL')
        rows = self._conn.execute(sql + (' AND contact_id=?' if contact_id else ''),
                                  (contact_id,) if contact_id else ()).fetchall()
        invalid = self._invalid_sources(rows, turn_ids)
        with self._conn:
            self._conn.executemany('DELETE FROM communications WHERE id=?', [(row['id'],) for row in invalid])
        return len(invalid)

    def _metadata_rows(self, query, params, fields, *, contact_id=None, connection=None):
        """Aggregate identifiers without rehashing canonical conversation prose."""
        from colonymind.memory import curated
        return curated.fenced_read(self._source_ledger,
            lambda: self._current_metadata_rows(query, params, fields,
                contact_id=contact_id, connection=connection), empty=[])

    def _current_metadata_rows(self, query, params, fields, *, contact_id=None, connection=None):
        from colonymind.memory import curated, producers
        reader = producers.active() or connection is not None
        if reader:
            db = connection if connection is not None else self._conn
            with db:
                if not db.in_transaction:
                    db.execute('BEGIN')
                rows = db.execute(query, params).fetchall()
        else:
            rows = self._reconciled_rows(query, params, contact_id=contact_id)
        encoded = dict.fromkeys(row['source_lineage_json'] for row in rows
                                if row['source_lineage_json'] is not None)
        lineages = {raw: json.loads(raw) for raw in encoded}
        erased = self._erased_pairs(item['turn_id'] for item in lineages.values()) if lineages and reader else ()
        sources = list(dict.fromkeys(item['turn_id'] for item in lineages.values()))
        visible = dict(zip(sources, (not hidden for hidden in
            curated.forgotten_mask(self._ledger(), sources)))) if sources else {}
        selected = []
        for row in rows:
            lineage = lineages.get(row['source_lineage_json'])
            if lineage:
                source = lineage['turn_id']
                if (row['contact_id'], source) in erased:
                    continue
                if not visible[source]:
                    continue
            selected.append({key: row[key] for key in fields})
        return selected

    def unlinked_summary_count(self, contact_id: str) -> int:
        """Historical/unattributed prose cannot be erased by guessing its source."""
        return self._conn.execute('''SELECT count(*) FROM communications WHERE contact_id=?
            AND source_lineage_json IS NULL AND coalesce(summary,'')!=''
            AND coalesce(receipt_ref,'')='' ''', (contact_id,)).fetchone()[0]

    def read_connection(self):
        """An independent read connection for projections running on a worker thread.

        The caller closes it. The transport writer's thread-bound connection
        and transaction ownership remain unchanged.
        """
        db = sqlite3.connect(Path(self._db_path).resolve().as_uri()+'?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        return db

    def log_receipt(self, *, event_id, contact_id, channel, direction, external_ref,
                    receipt_ref, occurred_at, status, reply_to_ref='', outbound_ref=''):
        """Idempotent metadata from a trusted transport adapter, never prose."""
        import hashlib
        stamp = _parse(occurred_at)
        if stamp is None or stamp > _now() or direction not in {'in', 'out'}:
            raise ValueError('invalid_transport_observation')
        identifier = 'transport:' + hashlib.sha256(event_id.encode()).hexdigest()
        values = (identifier, contact_id, channel, direction, 'Transport '+status, '',
                  stamp.isoformat(), external_ref, reply_to_ref or None, None,
                  receipt_ref, outbound_ref or None)
        columns = 'id,contact_id,channel,direction,summary,session_id,ts,external_ref,reply_to_ref,reaction,receipt_ref,outbound_ref'
        with self._conn:
            previous = self._conn.execute('SELECT '+columns+' FROM communications WHERE id=?', (identifier,)).fetchone()
            if previous:
                if tuple(previous) != values:
                    raise ValueError('transport_event_conflict')
                return False
            self._conn.execute('INSERT INTO communications ('+columns+') VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', values)
        return True

    def outbound_receipts(self, *, contact_id, outbound_ref):
        return self._summary_rows('''SELECT * FROM communications WHERE contact_id=?
            AND direction='out' AND (outbound_ref=? OR external_ref=?)
            AND receipt_ref IS NOT NULL AND external_ref IS NOT NULL ORDER BY ts,id LIMIT 100''',
            (contact_id, outbound_ref, outbound_ref),
            ('id', 'contact_id', 'channel', 'direction', 'summary', 'session_id', 'ts',
             'external_ref', 'reply_to_ref', 'reaction', 'receipt_ref', 'outbound_ref',
             'source_lineage_json'), contact_id=contact_id)

    def log(self, contact_id: str, *, channel: str = "unknown", direction: str = "in",
            summary: str = "", session_id: str = "",
            external_ref: str = "", reply_to_ref: str = "",
            reaction: str = "", receipt_ref: str = "",
            ts: Optional[str] = None, source_lineage=None) -> None:
        if not contact_id or direction not in ("in", "out"):
            return
        if source_lineage is not None and not self._source_visible(contact_id, source_lineage):
            from colonymind.turns.idempotency import SourceErased
            raise SourceErased('source_erased')
        normalized_reaction = (reaction or "").strip().lower()
        if normalized_reaction not in {
                "", "accepted", "acknowledged", "actioned", "negative",
                "dismissed", "corrected", "rejected", "neutral"}:
            normalized_reaction = "neutral"
        self._conn.execute(
            "INSERT INTO communications "
            "(id,contact_id,channel,direction,summary,session_id,ts,"
            "external_ref,reply_to_ref,reaction,receipt_ref,source_lineage_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, contact_id, channel or "unknown", direction,
             (summary or "")[:500], session_id or "", ts or _now().isoformat(),
             (external_ref or "")[:512] or None,
             (reply_to_ref or "")[:512] or None,
             normalized_reaction or None,
             (receipt_ref or "")[:512] or None,
             json.dumps(source_lineage) if source_lineage is not None else None),
        )
        self._conn.commit()
        # A source erasure can finish while this separate projection is written.
        if source_lineage is not None and not self._source_visible(contact_id, source_lineage):
            self.purge_erased_sources(contact_id=contact_id)
            from colonymind.turns.idempotency import SourceErased
            raise SourceErased('source_erased')

    def history(self, contact_id: str, limit: Optional[int] = 15, *,
                since: Optional[str] = None, until: Optional[str] = None,
                conversation_only: bool = False) -> List[Dict[str, Any]]:
        where, params = "contact_id=?", [contact_id]
        if since is not None:
            where += " AND ts>=?"
            params.append(since)
        if until is not None:
            where += " AND ts<=?"
            params.append(until)
        if conversation_only:
            where += " AND " + CONVERSATION
        tail = " ORDER BY ts DESC, rowid DESC"
        # A time window does not bound the number of rows. Keep the legacy
        # unwindowed API, but never materialize an unlimited continuity window.
        if since is not None or until is not None:
            limit = min(100, limit) if limit is not None else 100
        if limit is not None:
            tail += " LIMIT ?"
            params.append(limit)
        return self._summary_rows(
            "SELECT id,contact_id,source_lineage_json,channel,direction,summary,ts "
            "FROM communications WHERE " + where + tail,
            params, ('channel', 'direction', 'summary', 'ts'), contact_id=contact_id)

    def elsewhere_history(self, contact_id: str, current_channel: str, *,
                          since: str, until: str) -> List[Dict[str, Any]]:
        """Six live summaries on the newest other platform, read in bounded index pages.

        Choose the platform after source validation: erasing its last message
        must expose the next platform, even after a full page of erased rows.
        """
        from .continuity import _AUTOMATED, platform
        current = platform(current_channel)
        if not current:
            return []
        # Use the renderer's exact normalization, including whitespace that
        # SQLite's default trim does not recognize. The ts index still bounds
        # the scan; this function is only a channel filter within that range.
        self._conn.create_function("comms_platform", 1, platform, deterministic=True)
        channel_platform = "comms_platform(channel)"
        excluded = sorted(_AUTOMATED | {current})
        where = ("contact_id=? AND ts>=? AND ts<=?"
                 " AND direction IN ('in','out') AND coalesce(receipt_ref,'')=''"
                 " AND coalesce(summary,'')!=''"
                 " AND lower(coalesce(external_ref,'')) NOT GLOB 'mind:*'"
                 " AND lower(coalesce(external_ref,'')) NOT GLOB 'system:*'"
                 " AND lower(coalesce(external_ref,'')) NOT GLOB 'automated:*'"
                 f" AND {channel_platform} NOT IN (" + ','.join('?' for _ in excluded) + ")")
        params = [contact_id, since, until, *excluded]
        cursor, chosen, result = None, None, []
        while len(result) < 6:
            page_where, page_params = where, list(params)
            if cursor is not None:
                page_where += " AND (ts,rowid)<(?,?)"
                page_params.extend(cursor)
            if chosen is not None:
                page_where += f" AND {channel_platform}=?"
                page_params.append(chosen)
            # Only ids and the index cursor are materialized before erasure
            # checks. Prose and lineage validation are limited to this page.
            page = self._conn.execute(
                "SELECT id,ts,rowid AS position FROM communications WHERE " + page_where
                + " ORDER BY ts DESC,rowid DESC LIMIT 100", page_params).fetchall()
            if not page:
                break
            cursor = (page[-1]['ts'], page[-1]['position'])
            rows = self._summary_rows(
                "SELECT id,contact_id,source_lineage_json,channel,direction,summary,ts "
                "FROM communications WHERE contact_id=? AND id IN ("
                + ','.join('?' for _ in page) + ") ORDER BY ts DESC,rowid DESC LIMIT 100",
                [contact_id, *(row['id'] for row in page)],
                ('channel', 'direction', 'summary', 'ts'), contact_id=contact_id)
            for row in rows:
                other = platform(row['channel'])
                if chosen is None:
                    chosen = other
                if other == chosen:
                    result.append(row)
                    if len(result) == 6:
                        break
            if len(page) < 100:
                break
        return result

    def last_per_channel(self, contact_id: str) -> Dict[str, Dict[str, Any]]:
        rows = self._summary_rows(
            "SELECT id,contact_id,source_lineage_json,channel,direction,summary,MAX(ts) AS ts "
            "FROM communications WHERE contact_id=? GROUP BY channel", (contact_id,),
            ('channel', 'direction', 'summary', 'ts'), contact_id=contact_id)
        return {r["channel"]: {"direction": r["direction"], "summary": r["summary"], "ts": r["ts"]}
                for r in rows}

    def last_outbound(self, contact_id: str) -> Optional[Dict[str, Any]]:
        rows = self._summary_rows(
            "SELECT id,contact_id,source_lineage_json,channel,summary,ts "
            "FROM communications WHERE contact_id=? AND direction='out' ORDER BY ts DESC LIMIT 1",
            (contact_id,), ('channel', 'summary', 'ts'), contact_id=contact_id)
        return rows[0] if rows else None

    def mind_sends(self, contact_id: str, *, limit: int = 20) -> List[Dict[str, Any]]:
        """The mind's own messages to a contact, newest first: ``{intention_id, type, ts}``. The mind
        logs each one it sent with ``external_ref`` ``mind:<type>:<intention id>`` (``MIND_REF``), so
        the social drive's streak and last send outlive the intention rows retention prunes, and
        follow the person through a merge (architecture 4.7 item 6)."""
        rows = self._metadata_rows(
            "SELECT contact_id,source_lineage_json,external_ref,ts FROM communications WHERE contact_id=? AND direction='out' "
            "AND external_ref LIKE ? ORDER BY ts DESC LIMIT ?",
            (contact_id, MIND_REF + "%", max(1, min(int(limit), 200))),
            ('external_ref', 'ts'), contact_id=contact_id)
        sends = []
        for row in rows:
            kind, _, intention_id = str(row["external_ref"])[len(MIND_REF):].partition(":")
            if kind and intention_id:
                sends.append({"intention_id": intention_id, "type": kind, "ts": row["ts"]})
        return sends

    def inbound_since(self, contact_id: str, since_iso: str) -> List[str]:
        """Timestamps of inbound rows from a contact since an ISO instant
        (selfhood benchmark: did the owner respond after a delivery)."""
        rows = self._metadata_rows(
            "SELECT contact_id,source_lineage_json,ts FROM communications WHERE contact_id=? AND"
            " direction='in' AND ts >= ? ORDER BY ts ASC LIMIT 5000",
            (contact_id, since_iso), ('ts',), contact_id=contact_id)
        return [r["ts"] for r in rows]

    def reactions_for_refs(
        self,
        contact_id: str,
        refs: List[str],
        *,
        since_iso: str,
        until_iso: str,
    ) -> List[Dict[str, Any]]:
        """Explicit inbound reactions tied to exact outbound references."""
        bounded = sorted({str(ref) for ref in refs if str(ref).strip()})[:5000]
        if not bounded:
            return []
        placeholders = ",".join("?" for _ in bounded)
        return self._summary_rows(
            "SELECT id,contact_id,source_lineage_json,channel,summary,ts,external_ref,reply_to_ref,reaction,"
            "receipt_ref FROM communications WHERE contact_id=? "
            "AND direction='in' AND ts>=? AND ts<? AND reply_to_ref IN ("
            f"{placeholders}) AND reaction IS NOT NULL ORDER BY ts",
            [contact_id, since_iso, until_iso, *bounded],
            ('id', 'channel', 'summary', 'ts', 'external_ref', 'reply_to_ref', 'reaction', 'receipt_ref'),
            contact_id=contact_id)

    def match_reply(self, *, contact_id: str, outbound_ref: str, since_iso: str,
                    until_iso: Optional[str] = None, connection=None) -> Dict[str, Any]:
        """Exact receipt-backed reply references for a previously resolved contact.

        A reply link proves which message was addressed, not whether its requested
        answer or artifact was supplied. The waiting-condition owner checks that.
        Neither similar text nor an unrelated inbound message counts as a reply.
        """
        start, end = _parse(since_iso), _parse(until_iso) if until_iso else _now()
        if not contact_id or not outbound_ref or start is None or end is None or end < start:
            raise ValueError('invalid_reply_window')
        rows = self._metadata_rows('''SELECT contact_id,source_lineage_json,channel,ts,external_ref,reply_to_ref,reaction,receipt_ref
            FROM communications WHERE contact_id=? AND direction='in' AND reply_to_ref=?
            AND external_ref IS NOT NULL AND external_ref!=''
            AND receipt_ref IS NOT NULL AND receipt_ref!='' ORDER BY ts LIMIT 500''',
            (contact_id, outbound_ref),
            ('channel', 'ts', 'external_ref', 'reply_to_ref', 'reaction', 'receipt_ref'),
            contact_id=contact_id, connection=connection)
        matches, seen = [], set()
        for row in rows:
            at = _parse(row['ts'])
            key = (row['external_ref'], row['receipt_ref'])
            if at is not None and start <= at <= end and key not in seen:
                matches.append(dict(row))
                seen.add(key)
        return {'status': 'matched' if matches else 'unrelated', 'contact_id': contact_id,
                'outbound_ref': outbound_ref, 'matches': matches,
                'condition_satisfied': False, 'coverage_limited': len(rows) == 500}

    def outbound_between(
        self,
        contact_id: str,
        since_iso: str,
        until_iso: str,
        *,
        require_receipt: bool = False,
    ) -> List[Dict[str, Any]]:
        """One auditable outbound cohort for benchmark denominators."""
        query = (
            "SELECT id,contact_id,source_lineage_json,channel,summary,ts,external_ref,receipt_ref "
            "FROM communications WHERE contact_id=? AND direction='out' "
            "AND ts>=? AND ts<?")
        if require_receipt:
            query += " AND receipt_ref IS NOT NULL AND receipt_ref!=''"
        query += " ORDER BY ts"
        return self._summary_rows(query, (contact_id, since_iso, until_iso),
            ('id', 'channel', 'summary', 'ts', 'external_ref', 'receipt_ref'), contact_id=contact_id)

    def counts(self, contact_id: str) -> Dict[str, Any]:
        rows = self._metadata_rows('SELECT id,contact_id,source_lineage_json,direction,channel '
            'FROM communications WHERE contact_id=?', (contact_id,), ('direction', 'channel'), contact_id=contact_id)
        return {'inbound': sum(row['direction'] == 'in' for row in rows),
                'outbound': sum(row['direction'] == 'out' for row in rows),
                'channels': len({row['channel'] for row in rows})}

    def stats(self, contact_id: str, *, since_days: int = 90) -> Dict[str, Any]:
        """Channel-usage counts and interaction-hour histogram (UTC hours;
        the consumer shifts into the contact's timezone). Feeds the
        relationship profiler's approach guidance (preferred channel,
        best time to reach)."""
        rows = self._metadata_rows(
            "SELECT id,contact_id,source_lineage_json,channel,ts FROM communications"
            " WHERE contact_id=? AND ts >= datetime('now', ?)",
            (contact_id, f"-{int(since_days)} day"), ('channel', 'ts'), contact_id=contact_id)
        channels: Dict[str, int] = {}
        hours = [0] * 24
        for r in rows:
            channels[r["channel"]] = channels.get(r["channel"], 0) + 1
            try:
                hours[int(str(r["ts"])[11:13])] += 1
            except (ValueError, IndexError):
                pass
        return {"total": len(rows), "channels": channels, "hours_utc": hours}

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """The newest exchanges across ALL contacts and channels, most recent
        first. The per-contact reads above answer 'how do I stand with X';
        this answers 'what has been flowing lately' for an ops ledger view."""
        limit = max(1, min(500, int(limit)))
        return self._summary_rows(
            "SELECT id,contact_id,source_lineage_json,channel,direction,summary,ts "
            "FROM communications ORDER BY ts DESC LIMIT ?", (limit,),
            ('contact_id', 'channel', 'direction', 'summary', 'ts'))

    def recent_snapshot(self, *, limit: int = 50, since_days: int = 30) -> dict:
        """Read the ops ledger on a worker-owned connection, including erasure cleanup."""
        from colonymind.memory import curated
        with closing(sqlite3.connect(self._db_path, uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            reader = copy.copy(self)
            reader._conn = conn
            return curated.fenced_read(self._source_ledger,
                lambda: {"entries": reader.recent(limit=limit),
                         "by_channel": reader.rollup(since_days=since_days)},
                empty={"entries": [], "by_channel": {}})

    def rollup(self, *, since_days: int = 30) -> Dict[str, Dict[str, int]]:
        """Inbound/outbound counts per channel over a window: the ledger's
        flow summary. ``{channel: {"in": n, "out": n}}``."""
        from colonymind.memory import curated
        return curated.fenced_read(self._source_ledger,
            lambda: self._current_rollup(since_days=since_days), empty={})

    def _current_rollup(self, *, since_days):
        from colonymind.memory import curated, producers

        def no_withdrawals():
            if self._source_ledger is None:
                return True  # Reconciliation still refuses unbound linked rows.
            try:
                ledger = self._ledger()
                cache = curated._cache_of(ledger)
                return not cache.source_fences and not curated._state(
                    None, ledger, '', source_only=True, cache=cache).forgotten
            except Exception:
                return False  # The metadata fence owns failure handling.

        window = f"-{int(since_days)} day"
        rows = None
        if not producers.active() and no_withdrawals():
            rows = self._reconciled_rows(
                "SELECT channel,direction,COUNT(*) AS n FROM communications"
                " WHERE ts >= datetime('now', ?) GROUP BY channel,direction", (window,))
            # A withdrawal can commit while this separate aggregate runs.
            # Recheck the live monitor before publishing the aggregate.
            if not no_withdrawals():
                rows = None
        if rows is None:
            rows = self._metadata_rows(
                "SELECT NULL AS contact_id,NULL AS source_lineage_json,channel,direction,COUNT(*) AS n"
                " FROM communications WHERE source_lineage_json IS NULL AND ts >= datetime('now', ?)"
                " GROUP BY channel,direction UNION ALL"
                " SELECT contact_id,source_lineage_json,channel,direction,1 AS n"
                " FROM communications WHERE source_lineage_json IS NOT NULL AND ts >= datetime('now', ?)",
                (window,) * 2, ('channel', 'direction', 'n'))
        out: Dict[str, Dict[str, int]] = {}
        for r in rows:
            ch = out.setdefault(r["channel"], {"in": 0, "out": 0})
            if r["direction"] in ("in", "out"):
                ch[r["direction"]] += r['n']
        return out

    def reattribute(self, old_id: str, new_id: str) -> int:
        """Move the exchanges of ``old_id`` to ``new_id`` (a merge); the number of rows moved. A
        sourced row moves once the ledger moved its source (``_movable``): the merge calls this
        before the source move and the reconciliation again after it."""
        if not old_id or not new_id or old_id == new_id:
            return 0
        rows = self._conn.execute("SELECT id, source_lineage_json FROM communications WHERE contact_id=?",
                                  (old_id,)).fetchall()
        movable = self._movable(rows, new_id)
        with self._conn:
            self._conn.executemany("UPDATE communications SET contact_id=? WHERE id=?", [(new_id, key) for key in movable])
        return len(movable)


# ---------------------------------------------------------------------------
# Outreach policy (architecture 4.7 items 5 and 6): the social drive's one rule
# ---------------------------------------------------------------------------
def evaluate_outreach(
    contact: Any,
    *,
    is_owner: bool = False,
    now: Optional[datetime] = None,
    cadence_minutes: Optional[float] = None,
    last_interaction_ts: Any = None,
    first_seen_ts: Any = None,
    last_outbound_ts: Any = None,
    last_attempt_ts: Any = None,
    ignored_streak: int = 0,
    open_followups: Optional[List[str]] = None,
    affect_declining: bool = False,
    suggested_channel: str = "",
) -> Dict[str, Any]:
    """Whether a check-in to ``contact`` is warranted now, and when it next would be.

    The reference is the later of the last conversation (or, before any, when
    the contact was first seen), the last message sent to them and the end of
    the last check-in that never went out (an ask that expired or was refused);
    a check-in is due one cadence after it. Silence moves the next one back: the cooldown
    after a send is ``cadence x 2**ignored_streak``, capped at four cadences,
    measured from the last send; a reply resets the streak (the caller
    computes it from the sent rows and the contact's last interaction).
    Declining contact affect holds outreach altogether. ``may_contact`` is not
    read here: authority decides permission, this decides timing. Returns
    ``should_contact``, ``reason``, ``cooldown_active``, ``next_eligible_at``
    (a datetime, None without a cadence), ``cooldown_hours`` and
    ``talking_points`` (the open follow-ups, at most five).
    """
    now = now or _now()
    followups = [str(item) for item in (open_followups or []) if str(item or "").strip()]
    result: Dict[str, Any] = {"should_contact": False, "reason": "", "cooldown_active": False,
                              "next_eligible_at": None, "cooldown_hours": None,
                              "suggested_channel": suggested_channel or "", "talking_points": followups[:5]}
    try:
        minutes = float(cadence_minutes) if cadence_minutes is not None else 0.0
    except (TypeError, ValueError):
        minutes = 0.0
    if is_owner:
        result["reason"] = "the owner is never a check-in target"
        return result
    if minutes <= 0:
        result["reason"] = "no cadence: neither owner-set nor estimated"
        return result
    cadence = timedelta(minutes=minutes)
    streak = max(0, int(ignored_streak or 0))
    cooldown = min(cadence * (2 ** streak), 4 * cadence)
    result["cooldown_hours"] = cooldown.total_seconds() / 3600.0
    last_in, first_seen, last_out = _parse(last_interaction_ts), _parse(first_seen_ts), _parse(last_outbound_ts)
    anchors = [value for value in (last_in or first_seen, last_out, _parse(last_attempt_ts)) if value is not None]
    if not anchors:
        result["reason"] = "no history to time a check-in from"
        return result
    reference = max(anchors)
    due_at = reference + cadence
    cooldown_end = last_out + cooldown if last_out is not None else None
    next_eligible = max(due_at, cooldown_end) if cooldown_end is not None else due_at
    result["next_eligible_at"] = next_eligible
    if affect_declining:
        result["reason"] = "contact affect declining; holding outreach"
        return result
    if cooldown_end is not None and now < cooldown_end:
        result["cooldown_active"] = True
        result["reason"] = (f"in cooldown until {cooldown_end.isoformat()} after {streak} ignored check-in(s)"
                            if streak else f"reached out at {last_out.isoformat()}; next eligible {cooldown_end.isoformat()}")
        return result
    if now < due_at:
        result["reason"] = f"not due until {due_at.isoformat()} (every {minutes:g} min)"
        return result
    reasons = [f"overdue vs cadence (every {minutes:g} min; last contact {reference.isoformat()})"]
    if followups:
        reasons.append(f"{len(followups)} open follow-up(s)")
    result["should_contact"] = True
    result["reason"] = "; ".join(reasons)
    return result


# The conversation-based cadence of a contact the owner gave no cadence (tier ``regular`` or above):
# the mean gap between conversations, a week before there are two, never under a day.
ESTIMATE_DEFAULT_MINUTES = 7 * 24 * 60
ESTIMATE_FLOOR_MINUTES = 24 * 60
ESTIMATE_CEILING_MINUTES = 90 * 24 * 60


def conversation_cadence_minutes(*, first_seen_ts: Any, last_interaction_ts: Any, conversations: Any) -> float:
    """How often this contact and the agent talk, in minutes, from the conversation count (C3)."""
    first, last = _parse(first_seen_ts), _parse(last_interaction_ts)
    try:
        count = int(conversations or 0)
    except (TypeError, ValueError):
        count = 0
    if count >= 2 and first is not None and last is not None and last > first:
        minutes = (last - first).total_seconds() / 60.0 / (count - 1)
    else:
        minutes = float(ESTIMATE_DEFAULT_MINUTES)
    return max(float(ESTIMATE_FLOOR_MINUTES), min(minutes, float(ESTIMATE_CEILING_MINUTES)))
