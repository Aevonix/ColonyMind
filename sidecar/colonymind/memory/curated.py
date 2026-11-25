"""Append-only curated memory over the existing turn ledger; never erase source history.

Source history stays intact; source forget also erases its derived vector projections.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
from contextvars import ContextVar
from copy import copy
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
from functools import lru_cache
from pathlib import Path

from colonymind.memory import inventory
from colonymind.memory import curated_health
from colonymind.util.temporal import now_utc


logger = logging.getLogger(__name__)


class Refused(ValueError):
    def __init__(self, code, status=409):
        self.code, self.status = code, status
        super().__init__(code)


def initialize(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS curated_memory_events (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id TEXT NOT NULL, memory_id TEXT NOT NULL, revision INTEGER NOT NULL,
        operation TEXT NOT NULL CHECK(operation IN ('remember','correct','forget')),
        request_id TEXT NOT NULL, request_digest TEXT NOT NULL, applied_at TEXT NOT NULL,
        data_json TEXT, source_ids_json TEXT NOT NULL,
        UNIQUE(person_id,request_id), UNIQUE(person_id,memory_id,revision))''')
    conn.execute('''CREATE TABLE IF NOT EXISTS curated_memory_fence_hashes (
        event_sequence INTEGER NOT NULL, person_id TEXT NOT NULL, message_hash TEXT NOT NULL,
        PRIMARY KEY(event_sequence, person_id, message_hash))''')
    conn.execute('''CREATE INDEX IF NOT EXISTS curated_memory_fence_person
        ON curated_memory_fence_hashes(person_id,event_sequence)''')
    for action in ('UPDATE', 'DELETE'):
        conn.execute(f'''CREATE TRIGGER IF NOT EXISTS curated_fence_hashes_no_{action.lower()}
            BEFORE {action} ON curated_memory_fence_hashes BEGIN
            SELECT RAISE(ABORT, 'curated_memory_is_append_only'); END''')
    # Guard the history even against accidental future callers of this table.
    for action in ('UPDATE', 'DELETE'):
        conn.execute(f'''CREATE TRIGGER IF NOT EXISTS curated_memory_no_{action.lower()}
            BEFORE {action} ON curated_memory_events BEGIN
            SELECT RAISE(ABORT, 'curated_memory_is_append_only'); END''')

    # REPLACE deletes conflicts without firing DELETE triggers when recursive_triggers=0.
    conn.execute("""CREATE TRIGGER IF NOT EXISTS curated_memory_no_replace
        BEFORE INSERT ON curated_memory_events WHEN EXISTS (
            SELECT 1 FROM curated_memory_events WHERE sequence=NEW.sequence
            OR (person_id=NEW.person_id AND request_id=NEW.request_id)
            OR (person_id=NEW.person_id AND memory_id=NEW.memory_id AND revision=NEW.revision))
        BEGIN SELECT RAISE(ABORT, 'curated_memory_is_append_only'); END""")


class _Cache:
    def __init__(self, path):
        self.lock = threading.RLock()
        self.path = Path(path)
        self.monitor = None
        self.identity = None
        self.states = {}
        # Source withdrawal is monotonic even while ordinary curated recall is
        # degraded. A successful append publishes this content-free fallback.
        self.source_fences = set()
        self.source_hashes = {}
        self.refresh()

    def refresh(self):
        stat = self.path.stat()
        identity = stat.st_dev, stat.st_ino
        if identity != self.identity:
            monitor = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True, check_same_thread=False)
            monitor.row_factory = sqlite3.Row
            if self.monitor is not None:
                self.monitor.close()
            self.monitor, self.identity = monitor, identity
            self.states.clear()

    def __del__(self):
        monitor = getattr(self, 'monitor', None)
        if monitor is not None:
            monitor.close()


@lru_cache(maxsize=32)
def _cache(path):
    return _Cache(path)


def _ledger_path(ledger):
    """Reuse canonicalization while the configured path still names the same file."""
    path = ledger.db_path.absolute()
    configured = str(path)
    try:
        stat = path.stat()
    except OSError:
        return str(path.resolve())
    identity = (configured, stat.st_dev, stat.st_ino)
    prior = getattr(ledger, '_curated_path', None)
    if prior and prior[0] == identity:
        if prior[1] == configured:
            return prior[1]
        try:
            target = os.stat(prior[1])
            if (target.st_dev, target.st_ino) == identity[1:]:
                return prior[1]
        except OSError:
            pass
    resolved = str(path.resolve())
    ledger._curated_path = (identity, resolved)
    return resolved


def _cache_of(ledger):
    return _cache(_ledger_path(ledger))


_disabled = set()


def enable(ledger):
    ledger._curated_available = True
    _disabled.discard(_ledger_path(ledger))
    curated_health.for_ledger(ledger).recover()


def available(ledger):
    return (ledger is not None and getattr(ledger, '_curated_available', True)
            and _ledger_path(ledger) not in _disabled)


def disable(ledger, error=None):
    ledger._curated_available = False
    _disabled.add(_ledger_path(ledger))
    curated_health.for_ledger(ledger).degrade(error)


def _grams(text):
    # One/two-character searches need their own postings; longer substrings use trigrams.
    return {text[i:i + size] for size in (1, 2, 3) for i in range(len(text) - size + 1)}


class _State:
    def __init__(self, events):
        self.events = events
        self.items = {}
        self.order = {ident: index for index, ident in enumerate(events)}
        self.postings = {}
        for ident, event in events.items():
            if event['operation'] == 'forget':
                continue
            item = json.loads(event['data_json'])
            self.items[ident] = item
            for gram in _grams(item['content'].casefold()):
                self.postings.setdefault(gram, set()).add(ident)

    def matching(self, terms):
        if not terms:
            return self.items.values()
        found = set()
        for term in terms:
            grams = {term[i:i + 3] for i in range(max(1, len(term) - 2))}
            lists = sorted((self.postings.get(gram, set()) for gram in grams), key=len)
            if not lists or not lists[0]:
                continue
            possible = lists[0].intersection(*lists[1:])
            found.update(ident for ident in possible if term in self.items[ident]['content'].casefold())
        # Preserve the current-event query's offer order through indexed matching.
        return (self.items[ident] for ident in sorted(found, key=self.order.__getitem__))


def _state(conn, ledger, person, *, source_only=False, cache=None):
    cache = cache if cache is not None else _cache_of(ledger)
    with cache.lock:
        cache.refresh()
        # Voice can hold an older native snapshot. Read the withdrawal journal
        # through its own read-only monitor so that snapshot cannot revive a
        # tombstone committed by another process. Never borrow its transaction.
        current = cache.monitor
        key = ('sources', person) if source_only else person
        while True:
            version = current.execute('PRAGMA data_version').fetchone()[0]
            prior = cache.states.get(key)
            if prior and prior[1] == version:
                return prior[2]
            sequence = current.execute('SELECT max(sequence) FROM curated_memory_events').fetchone()[0]
            current.execute('BEGIN')
            try:
                rebuild = not prior or prior[0] != sequence
                state = _State.__new__(_State) if rebuild else copy(prior[2])
                if rebuild:
                    state.events = ({row['memory_id']: dict(row) for row in current.execute("""SELECT * FROM
                        curated_memory_events WHERE operation='forget' AND memory_id LIKE 'sources:%'""")}
                        if source_only else _events(current, person)) if sequence else {}
                else:
                    state.events = prior[2].events
                source_events = {key: event for key, event in state.events.items() if key.startswith('sources:')}
                state.forgotten = _fenced_sources(current, source_events)
                state.forgotten.update(cache.source_fences)
                cache.source_fences.update(state.forgotten)
                # Publish the source-only fallback before indexing ordinary
                # curated content. A malformed memory cannot undo a withdrawal.
                indexed = current.execute('''SELECT h.message_hash,e.memory_id FROM curated_memory_fence_hashes h
                    JOIN curated_memory_events e ON e.sequence=h.event_sequence WHERE h.person_id=?''', (person,)).fetchall() if sequence else []
                state.forgotten_hashes = _source_hashes(current, state.forgotten, person=person, events=source_events)
                state.forgotten_hashes.update(row['message_hash'] for row in indexed if row['memory_id'].startswith('sources:'))
                state.forgotten_hashes.update(cache.source_hashes.get(person, set()))
                if person:
                    cache.source_hashes.setdefault(person, set()).update(state.forgotten_hashes)
                if source_only:
                    state.fenced, state.hashes = set(state.forgotten), set(state.forgotten_hashes)
                else:
                    state.fenced = _fenced_sources(current, state.events)
                    state.fenced.update(state.forgotten)
                    state.hashes = _source_hashes(current, state.fenced, person=person, events=state.events)
                    state.hashes.update(row['message_hash'] for row in indexed)
                    state.hashes.update(state.forgotten_hashes)
                if rebuild:
                    _State.__init__(state, state.events)
            finally:
                current.rollback()
            if current.execute('PRAGMA data_version').fetchone()[0] != version:
                continue
            cache.source_fences.update(state.forgotten)
            if person:
                cache.source_hashes.setdefault(person, set()).update(state.forgotten_hashes)
            cache.states[key] = (sequence, version, state)
            return state


class _ReadSnapshot:
    """Open native evidence only when a curated read actually needs it."""
    def __init__(self, ledger):
        self.ledger, self.connection, self.own_transaction = ledger, None, False

    def __getattr__(self, name):
        if self.connection is None:
            connect = getattr(self.ledger, '_read_connect', None) or self.ledger._connect
            self.connection = connect()
            self.own_transaction = not self.connection.in_transaction
            if self.own_transaction:
                self.connection.execute('BEGIN')
        return getattr(self.connection, name)

    def close(self):
        if self.connection is not None:
            try:
                if self.own_transaction:
                    self.connection.rollback()
            finally:
                self.connection.close()
                self.connection = None


@contextmanager
def _snapshot(ledger):
    """Keep the failure boundary even when the live journal needs no native read."""
    with closing(_ReadSnapshot(ledger)) as snapshot:
        yield snapshot


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _events(conn, person):
    return {row['memory_id']: dict(row) for row in conn.execute('''
        SELECT e.* FROM curated_memory_events e WHERE
        (person_id=? AND sequence IN
            (SELECT max(sequence) FROM curated_memory_events WHERE person_id=? GROUP BY memory_id))
        OR (operation='forget' AND memory_id LIKE 'sources:%') ORDER BY sequence''', (person, person))}


def _claim_items(conn, person, events):
    ids = [row[0] for row in conn.execute('''SELECT c.id FROM source_claims c JOIN turn_sources s
        ON s.turn_id=c.turn_id WHERE s.contact_id=? AND s.scope='person' ORDER BY c.id''', (person,))]
    rows = inventory.visible(conn, person, ids)
    items = {}
    rows = {row["id"]: row for row in _filter(conn, person, list(rows.values()), events)}
    for card in inventory.cards(conn, person, list(rows.values()), now_utc()):
        if card['state'] not in ('current', 'in_conflict'):
            continue
        ident = card['claim_id']
        row = rows[ident]
        relationships = [{'kind': 'contact', 'id': person, 'relation': 'participant', 'label': None}]
        if row.get('subject_key'):
            relationships.append({'kind': 'entity', 'id': row['subject_key'], 'relation': 'subject',
                                  'label': row.get('subject')})
        refs = inventory._sources(conn, _claim_sources(conn, ident))
        memory_id = 'claim:' + ident
        items[memory_id] = {
            'id': memory_id, 'person_id': person, 'revision': 1,
            'content': card.get('quote') or card['value'], 'kind': 'claim', 'state': 'active',
            'verification': card['verification'], 'applied_at': None,
            'source_refs': sorted(refs.values(), key=lambda ref: ref['source_id']),
            'relationship_refs': relationships}
    return items


def _dependency_claims(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('subject_basis_claim_id', 'source_claim_id', 'claim_id') and isinstance(item, str):
                yield item
            elif isinstance(item, (dict, list)):
                yield from _dependency_claims(item)
    elif isinstance(value, list):
        for item in value:
            yield from _dependency_claims(item)


def _claim_sources(conn, claim_id):
    sources, visited, pending = set(), set(), [claim_id]
    while pending:
        ident = pending.pop()
        if ident in visited:
            continue
        visited.add(ident)
        rows = conn.execute("""SELECT turn_id,message_hash,data_json FROM source_claims WHERE id=?
                UNION ALL SELECT turn_id,message_hash,data_json FROM source_claim_witnesses WHERE claim_id=?""", (ident, ident)).fetchall()
        # Folded statements are alternative witnesses. A withdrawn witness
        # cannot invalidate an independently retained statement of that claim.
        live = [row for row in rows if not is_forgotten(conn, {
            'source_id': row['turn_id'], 'message_hash': row['message_hash']})]
        for row in (live[:1] or rows[:1]):
            data = json.loads(row['data_json'])
            sources.add(row['turn_id'])
            sources.update(_lineage(data)[0])
            pending.extend(_dependency_claims(data))
    return sources


def _candidate_lineage(conn, row):
    sources, hashes = _lineage(row)
    for ident in _dependency_claims(row):
        sources.update(_claim_sources(conn, ident))
    # The native anchor and all witnesses are part of a projected claim's lineage.
    ident = row.get('id', '')
    if ident and not row.get('_curated_id') and not (row.get('turn_id') or row.get('source_turn_id')):
        sources.update(_claim_sources(conn, ident))
    return sources, hashes


def _items(conn, person):
    events = _events(conn, person)
    items = _claim_items(conn, person, events)
    for ident, event in events.items():
        if event['operation'] == 'forget':
            items.pop(ident, None)
        else:
            items[ident] = json.loads(event['data_json'])
    # Source forgetting also withdraws an existing correction of that source.
    offered = [dict(item, _curated_id=ident, _curated_revision=item['revision'])
               for ident, item in items.items() if ident in events]
    retained = {row['_curated_id'] for row in _filter(conn, person, offered, events)}
    for ident in set(events) - retained:
        items.pop(ident, None)
    return items


def _read(items, memory_id, revision=None):
    item = items.get(memory_id)
    if item is None:
        raise Refused('memory_not_found', 404)
    if revision is not None and revision != item['revision']:
        raise Refused('revision_conflict')
    return item


def read(ledger, person, memory_id, revision=None):
    item = fenced_read(ledger, lambda: _read_item(ledger, person, memory_id, revision), empty=None)
    if item is None:
        raise Refused('curated_changed_restart')
    return item


def _read_item(ledger, person, memory_id, revision=None):
    with _snapshot(ledger) as conn:
        return _read(_items(conn, person), memory_id, revision)


def search(ledger, person, *, query='', cursor=None, limit=20):
    page = fenced_read(ledger, lambda: _search(ledger, person, query=query, cursor=cursor, limit=limit), empty=None)
    if page is None:
        raise Refused('curated_changed_restart')
    return page


def _search(ledger, person, *, query='', cursor=None, limit=20):
    query = query.strip().casefold()
    with _snapshot(ledger) as conn:
        items = _items(conn, person)
        revision = _digest([person, query, limit, sorted(items.values(), key=lambda item: item['id'])])
        offset = 0
        if cursor is not None:
            match = re.fullmatch(r'(0|[1-9][0-9]{0,18})\.([a-f0-9]{64})', cursor)
            if not match:
                raise Refused('invalid_cursor', 422)
            offset = int(match[1])
            if match[2] != revision:
                raise Refused('curated_changed_restart')
        matches = [items[key] for key in sorted(items) if query in items[key]['content'].casefold()]
        if cursor is not None and (offset <= 0 or offset >= len(matches)):
            raise Refused('invalid_cursor', 422)
        page = matches[offset:offset + limit]
        after = offset + len(page)
        return {'items': page, 'next_cursor': f'{after}.{revision}' if after < len(matches) else None,
                'read_revision': revision, 'status': 'ready' if page else 'empty'}


def _receipt(row):
    return {'id': row['memory_id'], 'revision': row['revision'], 'applied_at': row['applied_at'],
            'operation': row['operation'], 'request_id': row['request_id']}


def mutate(ledger, person, memory_id, operation, *, request_id, expected_revision=None, content=None, kind='fact'):
    digest = _digest([person, memory_id, operation, request_id, expected_revision, content, kind])
    if not available(ledger):
        raise sqlite3.OperationalError('curated_backend_unavailable')
    with closing(ledger._connect()) as conn, conn:
        # Reconstruct outside the writer lock. A concurrent native/curated commit
        # changes data_version; retry before comparing and appending.
        while True:
            version = conn.execute('PRAGMA data_version').fetchone()[0]
            conn.execute('BEGIN')
            items = _items(conn, person) if memory_id.startswith('claim:') else {}
            conn.rollback()
            conn.execute('BEGIN IMMEDIATE')
            if conn.execute('PRAGMA data_version').fetchone()[0] == version:
                break
            conn.rollback()
        prior = conn.execute('SELECT * FROM curated_memory_events WHERE person_id=? AND request_id=?',
                             (person, request_id)).fetchone()
        if prior is not None:
            if prior['request_digest'] != digest:
                raise Refused('request_id_conflict')
            return _receipt(prior)
        previous = conn.execute('''SELECT * FROM curated_memory_events WHERE person_id=? AND memory_id=?
            ORDER BY sequence DESC LIMIT 1''', (person, memory_id)).fetchone()
        if previous is not None and not memory_id.startswith('claim:') and previous['data_json']:
            items[memory_id] = json.loads(previous['data_json'])
        sources = set(json.loads(previous['source_ids_json'])) if previous else set()
        if memory_id.startswith('claim:'):
            sources.update(_claim_sources(conn, memory_id[6:]))
        applied_at = now_utc().isoformat()
        if operation == 'remember':
            if memory_id in items or previous is not None:
                raise Refused('memory_id_conflict')
            item = {'id': memory_id, 'person_id': person, 'revision': 1, 'content': content,
                    'kind': kind, 'state': 'active', 'verification': 'owner_asserted',
                    'applied_at': applied_at, 'source_refs': [],
                    'relationship_refs': [{'kind': 'contact', 'id': person, 'relation': 'participant', 'label': None}]}
        else:
            item = dict(_read(items, memory_id, expected_revision))
            item.update(revision=item['revision'] + 1, content=content, applied_at=applied_at,
                        verification='owner_asserted')
        row = {'person_id': person, 'memory_id': memory_id, 'revision': item['revision'],
               'operation': operation, 'request_id': request_id, 'request_digest': digest,
               'applied_at': applied_at, 'data_json': None if operation == 'forget' else _json(item),
               'source_ids_json': _json(sorted(sources))}
        conn.execute('''INSERT INTO curated_memory_events
            (person_id,memory_id,revision,operation,request_id,request_digest,applied_at,data_json,source_ids_json)
            VALUES (:person_id,:memory_id,:revision,:operation,:request_id,:request_digest,:applied_at,:data_json,:source_ids_json)''', row)
        if operation != 'remember':
            # An empty partial rule invalidates prepared packets without erasing
            # source bytes or dependencies. Corrections also withdraw old content.
            event_id = 'curated-item:' + digest
            ledger._append_erasure(conn, turn_id=event_id, contact_id=person,
                session_id='', scope='curated-item', messages=[], removed=[], whole=False,
                event_id=event_id)
        return _receipt(row)


def _fenced_sources(conn, events):
    sources = set()
    for ident, event in events.items():
        sources.update(json.loads(event['source_ids_json']))
        if ident.startswith('claim:'):
            # Folding a later restatement onto a claim must not un-forget it.
            sources.update(_claim_sources(conn, ident[6:]))
        elif ident.startswith('sources:'):
            sources.update(source for source, _ in _speaker_sources(conn, event))
    return sources


def _speaker_sources(conn, event):
    """Older observed-conversation writers also retained per-speaker sources."""
    from colonymind.turns.observed import is_conversation_source, speaker_sources
    return [pair for source in json.loads(event['source_ids_json']) if is_conversation_source(source)
            for pair in speaker_sources(conn, source)]


def _resolve_sources(conn, ledger, person, source_ids, old_text, session_id):
    """The former exact-text selection rules, with only reads and no erasure."""
    if bool(source_ids) == bool(old_text):
        raise Refused('invalid_source_selection', 422)
    rows = conn.execute('SELECT * FROM turn_sources WHERE contact_id=?', (person,)).fetchall()
    by_id = {row['turn_id']: row for row in rows}
    if old_text:
        if not session_id:
            raise Refused('invalid_source_selection', 422)
        from colonymind.secrets.seal import Scope, seal_text
        texts = {old_text, seal_text(old_text, scope=Scope(contact_id=person, session_id=session_id),
                                    vault=ledger.vault, existing_only=True)}
        source_ids = [row['turn_id'] for row in rows if row['session_id'] == session_id
                      and any(isinstance(message.get('content'), str) and message['content'] in texts
                              for message in json.loads(row['messages_json']))]
        if len(source_ids) != 1:
            raise Refused('ambiguous_source' if source_ids else 'source_not_found', 409 if source_ids else 422)
    selected = sorted(set(source_ids))
    if not 1 <= len(selected) <= 100 or any(not item or len(item) > 256 for item in selected):
        raise Refused('invalid_source_selection', 422)
    missing = set(selected) - by_id.keys()
    if missing:
        # A receipt remains retryable after a later attribution correction.
        # Prior source tombstones prove this person's earlier selection; they
        # grant no authority over a previously unknown id.
        known = {source for event in conn.execute('''SELECT source_ids_json FROM curated_memory_events
            WHERE person_id=? AND operation='forget' AND memory_id LIKE 'sources:%' ''', (person,))
                 for source in json.loads(event['source_ids_json'])}
        if missing - known:
            raise Refused('source_not_found', 422)
    return selected


def forget_sources(ledger, person, *, source_ids=None, old_text=None, session_id=None):
    """Append a content-free source fence and its stable, selection-addressed receipt.

    Repeating a selection (including through exact-text resolution) returns the
    original event. Historical source rows remain intact; the host erasure feed
    publishes the withdrawal in the same transaction. Derived vector text is
    scheduled for erasure after that transaction commits; the receipt does not
    wait for the erase or its compaction.
    """
    if not available(ledger):
        raise sqlite3.OperationalError('curated_backend_unavailable')
    with closing(ledger._connect()) as conn, conn:
        while True:
            version = conn.execute('PRAGMA data_version').fetchone()[0]
            conn.execute('BEGIN')
            selected = _resolve_sources(conn, ledger, person, source_ids, old_text, session_id)
            event = {'source_ids_json': _json(selected), 'person_id': person}
            events = {'sources:pending': event}
            prepared_fences = _fenced_sources(conn, events)
            hashes = _source_hashes(conn, prepared_fences, person=person,
                                    events=events, expand=True, ledger=ledger)
            conn.rollback()
            conn.execute('BEGIN IMMEDIATE')
            if conn.execute('PRAGMA data_version').fetchone()[0] == version:
                break
            conn.rollback()
        digest = _digest(['source-forget-v1', person, selected])
        ident, request_id = 'sources:' + digest, 'source-forget:' + digest
        prior = conn.execute('SELECT * FROM curated_memory_events WHERE person_id=? AND request_id=?',
                             (person, request_id)).fetchone()
        if prior is not None and (prior['memory_id'] != ident or prior['request_digest'] != digest):
            raise Refused('request_id_conflict')
        if prior is None:
            prior = {'person_id': person, 'memory_id': ident, 'revision': 1, 'operation': 'forget',
                     'request_id': request_id, 'request_digest': digest, 'applied_at': now_utc().isoformat(),
                     'data_json': None, 'source_ids_json': _json(selected)}
            conn.execute('''INSERT INTO curated_memory_events
                (person_id,memory_id,revision,operation,request_id,request_digest,applied_at,data_json,source_ids_json)
                VALUES (:person_id,:memory_id,:revision,:operation,:request_id,:request_digest,:applied_at,:data_json,:source_ids_json)''', prior)
            sequence = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
            conn.executemany('INSERT INTO curated_memory_fence_hashes VALUES (?,?,?)',
                             ((sequence, person, value) for value in hashes))
            for source in sorted(prepared_fences):
                row = conn.execute('SELECT * FROM turn_sources WHERE turn_id=?', (source,)).fetchone()
                if row is None:
                    continue
                messages = json.loads(row['messages_json'])
                # Publish both stored and preserved input message identities.
                # Keep opaque storage IDs for native historical attribution;
                # erasure_feed exposes the actual turn ID for direct outbox purge.
                ledger._append_erasure(conn, turn_id=source, contact_id=row['contact_id'],
                    session_id=row['session_id'], scope=row['scope'], messages=messages,
                    removed=messages, whole=True,
                    event_id='curated-source:' + _digest([row['contact_id'], source]))
        cache = _cache_of(ledger)
        with cache.lock:
            cache.source_fences.update(prepared_fences)
            cache.source_hashes.setdefault(person, set()).update(hashes)
        # Only fenced turns and turns containing fenced message hashes can
        # hold projections of this withdrawal. Each row is still checked.
        from colonymind.turns.idempotency import source_message_hash
        projection_sources = prepared_fences | {row['turn_id'] for row in conn.execute(
            'SELECT turn_id,session_id,messages_json FROM turn_sources WHERE contact_id=?', (person,))
            if any(source_message_hash(row['session_id'], message) in hashes
                   for message in json.loads(row['messages_json']))}
    from colonymind.vector import get_store
    store = get_store()
    if (store is not None and store.catalog is not None
            and Path(store.catalog.ledger.db_path).resolve() == Path(ledger.db_path).resolve()):
        try:
            store.erase_forgotten_sources(projection_sources)
        except Exception:
            logger.exception("Could not schedule source vector erasure")
    return {'source_erased': True, 'source_ids': selected, 'receipt': _receipt(prior),
            'scope': 'curated_source_tombstone',
            'note': 'source_erased means a curated tombstone was recorded; historical data is retained.'}


def backfill_source_fences(ledger):
    """Upgrade pre-index source withdrawals at startup, never on a recall read."""
    with closing(ledger._connect()) as conn, conn:
        while True:
            version = conn.execute('PRAGMA data_version').fetchone()[0]
            conn.execute('BEGIN')
            pending = conn.execute("""SELECT e.* FROM curated_memory_events e
                WHERE e.operation='forget' AND e.memory_id LIKE 'sources:%'
                AND NOT EXISTS (SELECT 1 FROM curated_memory_fence_hashes h WHERE h.event_sequence=e.sequence)""").fetchall()
            if not pending:
                conn.rollback()
                return
            prepared = []
            for event in pending:
                events = {event['memory_id']: dict(event)}
                hashes = _source_hashes(conn, _fenced_sources(conn, events), person=event['person_id'],
                    events=events, expand=True, ledger=ledger)
                # An empty marker also records a completed migration if an
                # independent historical retention policy removed every root.
                prepared.extend((event['sequence'], event['person_id'], value) for value in hashes or {''})
            conn.rollback()
            conn.execute('BEGIN IMMEDIATE')
            if conn.execute('PRAGMA data_version').fetchone()[0] != version:
                conn.rollback()
                continue
            conn.executemany('INSERT INTO curated_memory_fence_hashes VALUES (?,?,?)', prepared)
            return


def _lineage(value):
    """Read recorded lineage recursively, including conversation/context dependencies."""
    sources, hashes = set(), set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('turn_id', 'source_id', 'source_turn_id') and isinstance(item, str):
                sources.add(item)
            elif key == 'source_turn_ids' and isinstance(item, list):
                sources.update(item)
            elif key in ('source_message_hash', 'message_hash') and isinstance(item, str):
                hashes.add(item)
            elif key in ('_annotation_message_hashes', '_source_message_hashes') and isinstance(item, dict):
                sources.update(item)
                hashes.update(digest for values in item.values() for digest in values)
            elif key == 'source_uri' and isinstance(item, str) and item.startswith('turn:'):
                sources.add(item[5:])
            if isinstance(item, (dict, list)):
                nested_sources, nested_hashes = _lineage(item)
                sources.update(nested_sources)
                hashes.update(nested_hashes)
    elif isinstance(value, list):
        for item in value:
            nested_sources, nested_hashes = _lineage(item)
            sources.update(nested_sources)
            hashes.update(nested_hashes)
    return sources, hashes


def _message_dependencies(conn, message):
    sources, hashes = _lineage(message)
    opinion = (message.get('metadata') or {}).get('opinion_id')
    if opinion is not None:
        dependency = conn.execute('SELECT dependency_json FROM self_judgment_revisions WHERE id=?', (opinion,)).fetchone()
        if dependency:
            more_sources, more_hashes = _lineage(json.loads(dependency[0]))
            sources.update(more_sources)
            hashes.update(more_hashes)
    return sources, hashes


def _message_membership(message):
    membership = {source: set(hashes) for source, hashes in (message.get('_source_message_hashes') or {}).items()}
    for ref in message.get('_supplied_inputs') or []:
        if isinstance(ref, dict) and ref.get('source_id') and ref.get('input_message_hash'):
            membership.setdefault(ref['source_id'], set()).add(ref['input_message_hash'])
    return membership


def index_source_fences(ledger, conn, source_id, person, session, messages, *, prior=None):
    """Index newly retained dependent copies in the owned append-only fence.

    The source and these hashes commit together. This lets a later plain copy
    inherit a withdrawal without a history scan or any native row mutation.
    """
    try:
        # This runs inside the source writer transaction, so no peer can append
        # a withdrawal before the source commit. An unavailable curated backend
        # must preserve ordinary native capture's existing failure contract.
        if not conn.execute("""SELECT 1 FROM curated_memory_events WHERE operation='forget'
                AND memory_id LIKE 'sources:%' LIMIT 1""").fetchone():
            return
        cache = _cache_of(ledger)
        _state(conn, ledger, '', source_only=True)
        if not cache.source_fences:
            return
        from colonymind.turns.idempotency import source_message_hash
        state = _state(conn, ledger, person, source_only=True)
        hashes = {source_message_hash(session, message) for message in messages
                  if is_forgotten(ledger, {'source_id': source_id,
                      'message_hash': source_message_hash(session, message)}, conn=conn)
                  or (prior or {}).get(source_message_hash(session, message), set()) & state.forgotten_hashes}
        if not hashes:
            return
        sequence = conn.execute("""SELECT max(sequence) FROM curated_memory_events
            WHERE operation='forget' AND memory_id LIKE 'sources:%'""").fetchone()[0]
        if sequence is not None:
            conn.executemany('INSERT OR IGNORE INTO curated_memory_fence_hashes VALUES (?,?,?)',
                             ((sequence, person, value) for value in hashes))
    except (sqlite3.Error, OSError) as error:
        disable(ledger, error)
        curated_health.failed_open(ledger)


def _source_hashes(conn, fenced, *, person=None, events=None, expand=False, ledger=None):
    from colonymind.turns.idempotency import source_message_hash
    # Evaluate the former copy/dependency closure in memory. In particular a
    # quotation hit need not carry the assistant's supplied-source metadata.
    if not fenced:
        return set()
    roots = [dict(row) for source in fenced for row in conn.execute(
        'SELECT turn_id,contact_id,session_id,messages_json FROM turn_sources WHERE turn_id=?', (source,))]
    contacts = {person} if person is not None else {row['contact_id'] for row in roots}
    exact_sources = _fenced_sources(conn, {ident: event for ident, event in events.items()
        if not ident.startswith('sources:') or event['person_id'] == person}) if events and person else fenced
    if events and person:
        exact_sources.difference_update(source for ident, event in events.items() if ident.startswith('sources:')
                                        for source, contact in _speaker_sources(conn, event) if contact != person)
        exact_sources.update(source for ident, event in events.items() if ident.startswith('sources:')
                             for source, contact in _speaker_sources(conn, event) if contact == person)
    rows = [dict(row) for person in contacts for row in conn.execute(
        'SELECT turn_id,session_id,messages_json FROM turn_sources WHERE contact_id=?', (person,))] if expand else []
    for row in [*roots, *rows]:
        row['messages'] = json.loads(row['messages_json'])
    hashes, removed = set(), {}
    for row in roots:
        digests = {source_message_hash(row['session_id'], message) for message in row['messages']}
        if ledger is not None:
            from colonymind.secrets.seal import erasure_forms
            for message in row['messages']:
                digests.update(source_message_hash(row['session_id'], {**message, 'content': form})
                    for form in erasure_forms(message.get('content'), contact_id=row['contact_id'], vault=ledger.vault))
        # Exact copies belong to the person who forgot them. Source ids and
        # recorded dependencies stay fenced even after attribution changes.
        exact = digests if row['turn_id'] in exact_sources else set()
        hashes.update(exact)
        removed[row['turn_id']] = digests
    while True:
        changed = False
        for row in rows:
            digests = set()
            for message in row['messages']:
                digest = source_message_hash(row['session_id'], message)
                sources, _ = _message_dependencies(conn, message)
                membership = _message_membership(message)
                if digest in hashes or any(source in removed and (source in fenced
                        or source not in membership or membership[source] & removed[source]) for source in sources):
                    digests.add(digest)
            previous = removed.get(row['turn_id'], set())
            if digests - previous:
                removed[row['turn_id']] = previous | digests
                hashes.update(digests)
                changed = True
        if not changed:
            return hashes


def _filter(conn, person, candidates, events, *, state=None):
    """One conservative lineage fence for curated reads, search and every recall pool."""
    if not candidates or not events and not (state and state.forgotten):
        return list(candidates)
    fenced = state.fenced if state else _fenced_sources(conn, events)
    fenced_hashes = state.hashes if state else _source_hashes(conn, fenced, person=person, events=events)
    source_events = {ident: event for ident, event in events.items() if ident.startswith('sources:')}
    forgotten = state.forgotten if state else _fenced_sources(conn, source_events)
    forgotten_hashes = state.forgotten_hashes if state else _source_hashes(conn, forgotten, person=person, events=source_events)
    retained = []
    for row in candidates:
        curated_id = row.get('_curated_id')
        if curated_id:
            current = events.get(curated_id)
            if current and current['operation'] != 'forget' and current['revision'] == row['_curated_revision']:
                sources, hashes = _lineage(json.loads(current['data_json']))
                if curated_id.startswith('claim:'):
                    sources.update(_claim_sources(conn, curated_id[6:]))
                if not sources.intersection(forgotten) and not hashes.intersection(forgotten_hashes):
                    retained.append(row)
        else:
            sources, hashes = _candidate_lineage(conn, row) if fenced or fenced_hashes else (set(), set())
            if ('claim:' + str(row.get('id', '')) not in events
                    and not sources.intersection(fenced) and not hashes.intersection(fenced_hashes)
                    and (not forgotten or not is_forgotten(conn, row))):
                retained.append(row)
    return retained


def _reader_ledger(ledger):
    """Resolve the existing connection form for both predicates and read tokens."""
    if hasattr(ledger, 'execute'):
        from types import SimpleNamespace
        databases = list(ledger.execute('PRAGMA database_list'))
        path = next((row[2] for row in databases if ledger.execute(
            f"SELECT 1 FROM {row[1]}.sqlite_master WHERE name='curated_memory_events'").fetchone()),
            next(row[2] for row in databases if row[1] == 'main'))
        return SimpleNamespace(db_path=Path(path))
    return ledger


def reader_of(conn):
    """The ledger a connection's reads resolve to, resolved once for a caller that checks many refs through it:
    ``is_forgotten(reader_of(conn), ref, conn=conn)`` reads as ``is_forgotten(conn, ref)`` and
    ``fenced_read(reader_of(conn), ...)`` as ``fenced_read(conn, ...)``. When the probe fails it is the connection
    itself, so each read probes, and fails, as before."""
    try:
        return _reader_ledger(conn)
    except Exception:
        return conn


def is_forgotten(ledger, source_ref, *, conn=None):
    """Shared live read fence, preserving the existing loud fail-open contract."""
    if not source_ref:
        return False
    try:
        if hasattr(ledger, 'execute'):
            conn = ledger
            ledger = _reader_ledger(ledger)
        return _is_forgotten(ledger, source_ref, conn=conn)
    except Exception as error:
        if ledger is not None and hasattr(ledger, 'db_path'):
            disable(ledger, error)
            curated_health.failed_open(ledger)
            try:
                cache = _cache_of(ledger)
            except (OSError, sqlite3.Error):
                return False
            if cache.source_fences:
                try:
                    return _is_forgotten(ledger, source_ref, conn=conn)
                except Exception:
                    pass  # unreadable source evidence cannot certify a fresh use
            sources, hashes = _lineage(source_ref)
            if cache.source_fences and (isinstance(source_ref, str) or sources or hashes
                                        or isinstance(source_ref, dict) and source_ref.get('_curated_id')):
                return True  # known withdrawal + unreadable evidence cannot certify a fresh use
        return False


def withdrawals_known(ledger, *, conn=None):
    """False only when ``is_forgotten`` would read every ref as not forgotten now (its empty-journal check: no
    withdrawal known); True otherwise, a failure included, so each ref is checked as before."""
    try:
        reader = _reader_ledger(ledger) if hasattr(ledger, 'execute') else ledger
        cache = _cache_of(reader)
        return bool(cache.source_fences or _state(conn, reader, '', source_only=True, cache=cache).forgotten)
    except Exception:
        return True


def forgotten_among(ledger, source_refs, *, conn=None):
    """The refs ``is_forgotten`` reads as forgotten, with its empty-journal check made once for them all."""
    refs = [ref for ref in source_refs if ref]
    if not refs or not withdrawals_known(ledger, conn=conn):
        return set()
    return {ref for ref in refs if is_forgotten(ledger, ref, conn=conn)}


def _journal_version(ledger):
    """Read the live journal independently of any caller's native snapshot.

    Inside a hot computation's withdrawal envelope (``withdrawal_fenced_read``, ``context_fenced_read``) on this
    ledger, the envelope's cursor fences withdrawals and the version is the curated journal's alone: any other
    commit to the file (a turn's activity, a capture) would otherwise make every retry of a nested read look
    torn and withhold it whole."""
    try:
        ledger = _reader_ledger(ledger)
        if ledger is None or not hasattr(ledger, 'db_path'):
            return None
        cache = _cache_of(ledger)
        with cache.lock:
            cache.refresh()
            version = cache.monitor.execute('PRAGMA data_version').fetchone()[0]
            sequence = cache.monitor.execute('SELECT max(sequence) FROM curated_memory_events').fetchone()[0]
            if _ledger_path(ledger) in _withdrawal_reads.get():
                version = None
            return cache.identity, version, sequence
    except Exception as error:
        if ledger is not None and hasattr(ledger, 'db_path'):
            disable(ledger, error)
        return None


def recall_protection(ledger):
    """Certify the whole read, including failures followed by handle recovery."""
    try:
        reader = _reader_ledger(ledger)
        if reader is None or not hasattr(reader, 'db_path'):
            return lambda: True
        health = curated_health.for_ledger(reader)
        version = health.failure_version
        was_available = available(reader)
    except Exception as error:
        logger.warning('curated_recall_withheld (%s)', type(error).__name__)
        return lambda: False

    def healthy():
        if not was_available or not available(reader) or health.failure_version != version:
            logger.warning('curated_recall_withheld')
            return False
        return True
    return healthy


def fenced_read(ledger, read, *, empty):
    """Retry the complete preparation if its live journal changed; bound churn.

    An already unknown journal retains the existing loud fail-open boundary.
    Losing a known version or continued concurrent commits cannot certify use.
    Async read callables keep the same fence across their awaits.
    """
    from inspect import iscoroutinefunction
    if iscoroutinefunction(read):
        async def complete():
            version = _journal_version(ledger)
            for _ in range(3):
                result = await read()
                current = _journal_version(ledger)
                if current == version:
                    return result
                if current is None:
                    break
                version = current
            return empty
        return complete()
    version = _journal_version(ledger)
    for _ in range(3):
        result = read()
        current = _journal_version(ledger)
        if current == version:
            return result
        if current is None:
            break
        version = current
    return empty


def _strict_fenced_read(ledger, read, *, empty):
    """Retry the complete preparation if its live journal changed; bound churn.

    An unknown journal or any protection failure withholds the result.
    Losing a known version or continued concurrent commits cannot certify use.
    Async read callables keep the same fence across their awaits.
    """
    if ledger is None:
        return read()  # Standalone stores have no curated-derived evidence.
    from inspect import iscoroutinefunction
    if iscoroutinefunction(read):
        async def complete():
            healthy = recall_protection(ledger)
            if not healthy():
                return empty
            version = _journal_version(ledger)
            if version is None:
                logger.warning('curated_recall_withheld')
                return empty
            for _ in range(3):
                result = await read()
                current = _journal_version(ledger)
                if not healthy():
                    return empty
                if current == version:
                    return result
                if current is None:
                    break
                version = current
            return empty
        return complete()
    healthy = recall_protection(ledger)
    if not healthy():
        return empty
    version = _journal_version(ledger)
    if version is None:
        logger.warning('curated_recall_withheld')
        return empty
    for _ in range(3):
        result = read()
        current = _journal_version(ledger)
        if not healthy():
            return empty
        if current == version:
            return result
        if current is None:
            break
        version = current
    return empty


_withdrawal_reads = ContextVar('curated_withdrawal_reads', default=frozenset())


def _withdrawal_version(ledger):
    """A live withdrawal cursor; ordinary turn/job/annotation writes do not move it."""
    try:
        cache = _cache_of(ledger)
        with cache.lock:
            cache.refresh()
            cursor = cache.monitor.execute("""SELECT
                (SELECT max(sequence) FROM curated_memory_events WHERE operation='forget'),
                (SELECT max(sequence) FROM source_erasures)""").fetchone()
            return cache.identity, tuple(cursor)
    except Exception as error:
        disable(ledger, error)
        return None


def withdrawal_fenced_read(ledger, read, *, empty):
    """Fence one complete hot computation, sharing its envelope with nested readers.

    Use global withdrawal cursors because an assembly can combine multiple
    contacts' source-backed lanes. Leaf reads retain their own membership and
    revision checks. Sync workers and async callers share the same scope,
    including worker-created event loops. No monitor lock is held while running the computation.
    """
    from inspect import iscoroutinefunction

    if iscoroutinefunction(read):
        async def complete():
            if ledger is None or not hasattr(ledger, 'db_path'):
                return await read()
            path = _ledger_path(ledger)
            active = _withdrawal_reads.get()
            if path in active:
                return await read()
            token = _withdrawal_reads.set(active | {path})
            try:
                version = _withdrawal_version(ledger)
                for _ in range(3):
                    result = await read()
                    current = _withdrawal_version(ledger)
                    if current == version:
                        return result
                    if current is None:
                        break
                    version = current
                return empty
            finally:
                _withdrawal_reads.reset(token)

        return complete()
    if ledger is None or not hasattr(ledger, 'db_path'):
        return read()
    path = _ledger_path(ledger)
    active = _withdrawal_reads.get()
    if path in active:
        return read()
    token = _withdrawal_reads.set(active | {path})
    try:
        version = _withdrawal_version(ledger)
        for _ in range(3):
            result = read()
            current = _withdrawal_version(ledger)
            if current == version:
                return result
            if current is None:
                break
            version = current
        return empty
    finally:
        _withdrawal_reads.reset(token)


def _strict_withdrawal_fenced_read(ledger, read, *, empty):
    """Fence one complete hot computation, sharing its envelope with nested readers.

    Use global withdrawal cursors because an assembly can combine multiple
    contacts' source-backed lanes. Leaf reads retain their own membership and
    revision checks. Sync workers and async callers share the same scope,
    including worker-created event loops. No monitor lock is held while running the computation.
    """
    from inspect import iscoroutinefunction

    if iscoroutinefunction(read):
        async def complete():
            if ledger is None or not hasattr(ledger, 'db_path'):
                return await read()
            healthy = recall_protection(ledger)
            if not healthy():
                return empty
            path = _ledger_path(ledger)
            active = _withdrawal_reads.get()
            if path in active:
                result = await read()
                return result if healthy() else empty
            token = _withdrawal_reads.set(active | {path})
            try:
                version = _withdrawal_version(ledger)
                if version is None:
                    logger.warning('curated_recall_withheld')
                    return empty
                for _ in range(3):
                    result = await read()
                    current = _withdrawal_version(ledger)
                    if not healthy():
                        return empty
                    if current == version:
                        return result
                    if current is None:
                        break
                    version = current
                return empty
            finally:
                _withdrawal_reads.reset(token)

        return complete()
    if ledger is None or not hasattr(ledger, 'db_path'):
        return read()
    healthy = recall_protection(ledger)
    if not healthy():
        return empty
    path = _ledger_path(ledger)
    active = _withdrawal_reads.get()
    if path in active:
        result = read()
        return result if healthy() else empty
    token = _withdrawal_reads.set(active | {path})
    try:
        version = _withdrawal_version(ledger)
        if version is None:
            logger.warning('curated_recall_withheld')
            return empty
        for _ in range(3):
            result = read()
            current = _withdrawal_version(ledger)
            if not healthy():
                return empty
            if current == version:
                return result
            if current is None:
                break
            version = current
        return empty
    finally:
        _withdrawal_reads.reset(token)


def forgotten_mask(ledger, source_refs):
    """Fence a materialized batch without repeating empty-journal lookups.

    The monitor is checked on every batch, including with a caller's older
    source snapshot. Withdrawals still use the shared dependency/hash fence.
    """
    refs = list(source_refs)
    if not refs or ledger is None:
        return [False] * len(refs)
    return fenced_read(ledger, lambda: _forgotten_mask(ledger, refs), empty=[True] * len(refs))


def _forgotten_mask(ledger, refs):
    try:
        cache = _cache_of(ledger)
        if not cache.source_fences and not _state(
                None, ledger, '', source_only=True, cache=cache).forgotten:
            return [False] * len(refs)
        with closing(ledger._connect()) as conn:
            return [is_forgotten(ledger, ref, conn=conn) for ref in refs]
    except Exception as error:
        disable(ledger, error)
        return [is_forgotten(ledger, ref) for ref in refs]


def _is_forgotten(ledger, source_ref, *, conn=None):
    """Read fence for active consumers; historical rows and bytes remain intact.

    Source identity remains withdrawn across attribution changes. Walk only the
    requested evidence's recorded dependencies, never the person's history.
    """
    if ledger is None:
        return False
    cache = _cache_of(ledger)
    # Check the current monitor before opening a native body connection.
    if not cache.source_fences:
        if not _state(conn, ledger, '', source_only=True, cache=cache).forgotten:
            return False
    if not available(ledger):
        if cache.source_fences and isinstance(source_ref, dict) and source_ref.get('_curated_id'):
            return True
    if conn is None:
        with closing(ledger._connect()) as snapshot:
            return is_forgotten(ledger, source_ref, conn=snapshot)
    from colonymind.turns.idempotency import source_message_hash
    if not cache.source_fences and not conn.execute("SELECT 1 FROM sqlite_master WHERE name='curated_memory_events' UNION ALL SELECT 1 FROM sqlite_temp_master WHERE name='curated_memory_events'").fetchone():
        # Attached ledgers are handled by the resolved path above. A failed
        # curated initialization keeps the established loud fail-open contract.
        if not any(conn.execute(f"SELECT 1 FROM {row[1]}.sqlite_master WHERE name='curated_memory_events'").fetchone()
                   for row in conn.execute('PRAGMA database_list')):
            return False
    lineage = _candidate_lineage(conn, source_ref)[0] if isinstance(source_ref, dict) else _lineage(source_ref)[0]
    if isinstance(source_ref, dict) and source_ref.get('contact_id'):
        hashes = _lineage(source_ref)[1]
        forgotten_hashes = _state(conn, ledger, source_ref['contact_id'], source_only=True).forgotten_hashes
        if hashes and hashes.intersection(forgotten_hashes):
            return True
    if lineage.intersection(cache.source_fences):
        return True
    pending = [(source_ref, None)] if isinstance(source_ref, str) else [(ident, None) for ident in lineage]
    if isinstance(source_ref, dict):
        for key in ('_source_message_hashes', '_annotation_message_hashes'):
            membership = source_ref.get(key) or {}
            pending = [pair for pair in pending if pair[0] not in membership]
            pending.extend((ident, digest) for ident, digests in membership.items() for digest in digests)
        ident = source_ref.get('source_id') or source_ref.get('turn_id') or source_ref.get('source_turn_id')
        digest = source_ref.get('source_message_hash') or source_ref.get('message_hash')
        if ident and digest:
            pending = [(ident, digest)] + [pair for pair in pending if pair[0] != ident]
    seen = set()
    while pending:
        ident, selected_hash = pending.pop()
        if ident in cache.source_fences:
            return True
        row = conn.execute('SELECT contact_id,session_id,messages_json FROM turn_sources WHERE turn_id=?', (ident,)).fetchone()
        if row is None and ident.startswith('turn:'):
            ident = ident.removeprefix('turn:')
            if ident in cache.source_fences:
                return True
            row = conn.execute('SELECT contact_id,session_id,messages_json FROM turn_sources WHERE turn_id=?', (ident,)).fetchone()
        if not ident or (ident, selected_hash) in seen:
            continue
        seen.add((ident, selected_hash))
        person = row['contact_id'] if row else ''
        state = _state(conn, ledger, person, source_only=True)
        fenced, forgotten_hashes = state.forgotten, state.forgotten_hashes
        if ident in fenced:
            return True
        if row:
            messages = [message for message in json.loads(row['messages_json'])
                        if selected_hash is None or source_message_hash(row['session_id'], message) == selected_hash]
            hashes = set()
            for message in messages:
                more_sources, more_hashes = _message_dependencies(conn, message)
                hashes.update(more_hashes)
                membership = _message_membership(message)
                pending.extend((source, digest) for source in more_sources
                    for digest in (membership[source] if source in membership else [None])
                    if (source, digest) not in seen)
            hashes.update(source_message_hash(row['session_id'], message) for message in messages)
            if hashes & forgotten_hashes:
                return True
    return False


def watermark(ledger, person, *, conn=None):
    """The shared persisted host erasure cursor, including curated withdrawals."""
    if conn is None:
        if not hasattr(ledger, '_connect'):
            return ledger.erasure_watermark(person)
        with closing(ledger._connect()) as source:
            return watermark(ledger, person, conn=source)
    return conn.execute('SELECT coalesce(max(sequence),0) FROM source_erasures WHERE contact_id=?',
                        (person,)).fetchone()[0]


def freshness_feed(ledger, person, after=0, limit=250):
    """Serve the same durable rules and cursor as the GET erasure feed."""
    return ledger.erasure_feed(person, after, limit)


def candidates(ledger, person, query, beliefs, quotations):
    """Offer indexed current content; isolate any curated failure from ordinary recall."""
    from .recall import lexical_terms
    try:
        if not available(ledger):
            curated_health.failed_open(ledger)
            return (_source_current(ledger, beliefs), _source_current(ledger, quotations))
        with _snapshot(ledger) as conn:
            state = _state(conn, ledger, person)
            if not state.events and not state.forgotten:
                return beliefs, quotations
            kept_beliefs = _filter(conn, person, beliefs, state.events, state=state)
            kept_quotations = _filter(conn, person, quotations, state.events, state=state)
            query = query.strip().casefold()
            terms = {query, *lexical_terms(query)} - {''} if query else set()
            overrides = []
            for item in state.matching(terms):
                ident = item['id']
                overrides.append({'id': ident, 'kind': 'curated_memory', 'content': item['content'],
                                'source_uri': 'curated:' + ident, 'epistemic_state': 'owner_asserted',
                                'effective_confidence': 1.0, 'created_at': item['applied_at'],
                                '_curated_id': ident, '_curated_revision': item['revision']})
            if overrides:
                kept_beliefs.extend(_filter(conn, person, overrides, state.events, state=state))
            return kept_beliefs, kept_quotations
    except Exception as exc:
        disable(ledger, exc)
        curated_health.failed_open(ledger)
        return _source_current(ledger, beliefs), _source_current(ledger, quotations)


def strict_candidates(ledger, person, query, beliefs, quotations):
    """Offer indexed current content only with available curated protection."""
    from .recall import lexical_terms
    try:
        if not available(ledger):
            curated_health.failed_open(ledger)
            logger.warning('curated_recall_withheld')
            return [], []
        healthy = recall_protection(ledger)
        with _snapshot(ledger) as conn:
            state = _state(conn, ledger, person)
            if not healthy():
                return [], []
            if not state.events and not state.forgotten:
                return beliefs, quotations
            kept_beliefs = _filter(conn, person, beliefs, state.events, state=state)
            kept_quotations = _filter(conn, person, quotations, state.events, state=state)
            query = query.strip().casefold()
            terms = {query, *lexical_terms(query)} - {''} if query else set()
            overrides = []
            for item in state.matching(terms):
                ident = item['id']
                overrides.append({'id': ident, 'kind': 'curated_memory', 'content': item['content'],
                                'source_uri': 'curated:' + ident, 'epistemic_state': 'owner_asserted',
                                'effective_confidence': 1.0, 'created_at': item['applied_at'],
                                '_curated_id': ident, '_curated_revision': item['revision']})
            if overrides:
                kept_beliefs.extend(_filter(conn, person, overrides, state.events, state=state))
            return (kept_beliefs, kept_quotations) if healthy() else ([], [])
    except Exception as exc:
        disable(ledger, exc)
        curated_health.failed_open(ledger)
        logger.warning('curated_recall_withheld (%s)', type(exc).__name__)
        return [], []


def current_candidates(ledger, candidates, *, contact_id):
    """Recheck the version; reuse the parsed/indexed state until a commit changes it."""
    if not candidates:
        return candidates
    return fenced_read(ledger, lambda: _current_candidates(ledger, candidates, contact_id=contact_id), empty=[])


def strict_current_candidates(ledger, candidates, *, contact_id):
    """Recheck the version; reuse the parsed/indexed state until a commit changes it."""
    if not candidates:
        return candidates
    return strict_fenced_read(ledger, lambda: _strict_current_candidates(ledger, candidates, contact_id=contact_id), empty=[])


def _current_candidates(ledger, candidates, *, contact_id):
    try:
        if not available(ledger):
            curated_health.failed_open(ledger)
            return _source_current(ledger, candidates)
        with _snapshot(ledger) as conn:
            state = _state(conn, ledger, contact_id)
            return _filter(conn, contact_id, candidates, state.events, state=state)
    except Exception as exc:
        disable(ledger, exc)
        curated_health.failed_open(ledger)
        return _source_current(ledger, candidates)


def _strict_current_candidates(ledger, candidates, *, contact_id):
    try:
        if not available(ledger):
            curated_health.failed_open(ledger)
            logger.warning('curated_recall_withheld')
            return []
        with _snapshot(ledger) as conn:
            state = _state(conn, ledger, contact_id)
            return _filter(conn, contact_id, candidates, state.events, state=state)
    except Exception as exc:
        disable(ledger, exc)
        curated_health.failed_open(ledger)
        logger.warning('curated_recall_withheld (%s)', type(exc).__name__)
        return []



def _source_current(ledger, candidates):
    return [row for row in candidates if not is_forgotten(ledger, row)]


_strict_reads = ContextVar('curated_strict_reads', default=None)


def _reprobe(ledger):
    """One read-only, bounded journal probe at a new strict read's start."""
    health = curated_health.for_ledger(ledger)
    version = health.failure_version
    try:
        from types import SimpleNamespace
        reader = SimpleNamespace(_read_connect=lambda: sqlite3.connect(
            Path(ledger.db_path).resolve().as_uri() + '?mode=ro', uri=True, timeout=0.05))
        with _snapshot(reader) as conn:
            conn.execute('SELECT max(sequence) FROM curated_memory_events').fetchone()
        with health.lock:
            if health.failure_version == version:
                ledger._curated_available = True
                _disabled.discard(_ledger_path(ledger))
                # The incident stays open until enable(): a failure this probe cannot
                # see would otherwise open a new incident and notice every read.
    except Exception as error:
        disable(ledger, error)
        logger.warning('curated_recall_withheld (%s)', type(error).__name__)


@contextmanager
def strict_read(ledger):
    """Share one recovery attempt and one failure epoch across a strict read."""
    reader = _reader_ledger(ledger)
    if reader is None or not hasattr(reader, 'db_path'):
        yield lambda: True
        return
    path = _ledger_path(reader)
    active = _strict_reads.get() or {}
    if path in active:
        # A recovered facade may precede a native handle with a stale local flag.
        # Only this strict scope can synchronize it, and only in the same epoch.
        if active[path]():
            reader._curated_available = True
        yield active[path]
        return
    if not available(reader):
        _reprobe(reader)
    healthy = recall_protection(reader)
    token = _strict_reads.set({**active, path: healthy})
    try:
        yield healthy
    finally:
        _strict_reads.reset(token)


def _strict_read_call(fence, ledger, read, *, empty):
    from inspect import iscoroutinefunction
    if iscoroutinefunction(read):
        async def complete():
            with strict_read(ledger) as healthy:
                if not healthy():
                    return empty
                result = await fence(ledger, read, empty=empty)
                return result if healthy() else empty
        return complete()
    with strict_read(ledger) as healthy:
        if not healthy():
            return empty
        result = fence(ledger, read, empty=empty)
        return result if healthy() else empty


def strict_fenced_read(ledger, read, *, empty):
    """Opt in to withholding recall on any unavailable protection or cursor."""
    return _strict_read_call(_strict_fenced_read, ledger, read, empty=empty)


def strict_withdrawal_fenced_read(ledger, read, *, empty):
    """Opt in to strict protection across a complete recall computation."""
    return _strict_read_call(_strict_withdrawal_fenced_read, ledger, read, empty=empty)


async def context_fenced_read(ledger, read, *, empty, withhold):
    """Fence hot-context recall while retaining its completed nonmemory sections."""
    with strict_read(ledger) as healthy:
        if ledger is None or not hasattr(ledger, 'db_path'):
            return await read()
        path = _ledger_path(ledger)
        active = _withdrawal_reads.get()
        if path in active:
            result = await read()
            return result if healthy() else withhold(result)
        token = _withdrawal_reads.set(active | {path})
        try:
            version = _withdrawal_version(ledger)
            for _ in range(3):
                result = await read()
                current = _withdrawal_version(ledger)
                if not healthy() or version is None or current is None:
                    logger.warning('curated_recall_withheld')
                    return withhold(result)
                if current == version:
                    return result
                version = current
            return empty
        finally:
            _withdrawal_reads.reset(token)
