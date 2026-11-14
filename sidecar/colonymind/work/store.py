"""Root-scoped work projection with durable, asynchronous repair."""
from __future__ import annotations

from collections import Counter
from contextlib import closing, contextmanager
import json
import os
from pathlib import Path
import sqlite3
import logging
import threading

from colonymind.turns.idempotency import canonical_turn_digest
from . import ledger as facts
from .models import WorkEvent
from .verification import effect_kind, root_status, verify

logger = logging.getLogger(__name__)

# Reads fence an old projection against the root's own durable generation.
CURRENT = '''generation=(SELECT generation FROM ledger.work_dirty d WHERE d.root_turn_id=work_roots.id)
    AND NOT EXISTS (SELECT 1 FROM ledger.source_attribution_invalidations i WHERE i.source_id=work_roots.source_id)'''

STATUSES = ('proven', 'partial', 'unproven', 'failed')


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def _origin(value):
    try:
        value = json.loads(value)
    except (ValueError, TypeError):
        pass
    return _json(value) if isinstance(value, dict) else value


def _watermark(conn, contact):
    return conn.execute('SELECT coalesce(max(sequence),0) FROM source_erasures WHERE contact_id=?', (contact,)).fetchone()[0]


def _frozen_revision(source, root, envelopes):
    return canonical_turn_digest([{
        'event': row['event_id'], 'envelope': row['envelope_json'],
        'invalidated': row['invalidated'],
        'refs_current': facts.current_refs(source, json.loads(row['source_refs_json']), root['contact_id'])
    } for row in envelopes])


def _effect_id(root, kind, operation):
    return canonical_turn_digest({'root': root, 'kind': kind, 'operation': operation})


def _summary(rows, effects):
    counts = {status: 0 for status in STATUSES}
    overlays = {'confirmed': 0, 'corrected': 0, 'none': 0}
    titles = {'unproven': [], 'failed': []}
    for row in rows:
        counts[row['status']] += 1
        overlays[row.get('overlay') or 'none'] += 1
        if row['status'] in titles and row.get('title') and len(titles[row['status']]) < 5:
            titles[row['status']].append(row['title'])
    ids = {row['id'] for row in rows}
    unknown = Counter((row['tool'], row['action']) for row in effects
                      if row['root_id'] in ids and row['kind'] == 'effect_unknown')
    return {'total': len(rows), 'counts': counts, 'owner_corrected': overlays['corrected'],
            'overlays': overlays, 'titles': titles, 'untitled': sum(row.get('title') is None for row in rows),
            'unknown_effects': [{'tool': tool, 'action': action, 'count': count}
                               for (tool, action), count in sorted(unknown.items())]}


class WorkStore:
    def __init__(self, ledger, path=None, *, read_only=False):
        self.ledger = ledger
        self.path = Path(path or ledger.db_path.parent / 'colonymind-work.db')
        self._read_only = read_only
        if read_only:
            return
        self._lock = threading.Lock()
        self._worker = None
        self._wake = False
        with closing(ledger._connect()) as source, source:
            facts.initialize(source)
        self._initialize_projection()

    def _initialize_projection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute('PRAGMA journal_mode=WAL')
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS work_roots (
                    id TEXT PRIMARY KEY, contact_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
                    title TEXT, revision TEXT NOT NULL, source_id TEXT NOT NULL,
                    erasure_watermark INTEGER NOT NULL, origin_ref TEXT);
                CREATE TABLE IF NOT EXISTS work_effects (
                    id TEXT PRIMARY KEY, root_id TEXT NOT NULL, operation_id TEXT NOT NULL,
                    tool TEXT NOT NULL, action TEXT NOT NULL, resource_ref TEXT,
                    status TEXT NOT NULL, kind TEXT NOT NULL, verification TEXT NOT NULL,
                    source_refs_json TEXT NOT NULL, evidence_json TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS work_effect_root ON work_effects(root_id);
                CREATE TABLE IF NOT EXISTS work_children (
                    root_id TEXT NOT NULL, turn_id TEXT NOT NULL, parent_turn_id TEXT NOT NULL,
                    PRIMARY KEY(root_id,turn_id));
                CREATE TABLE IF NOT EXISTS work_overlays (
                    root_id TEXT PRIMARY KEY, verdict TEXT, source_refs_json TEXT NOT NULL DEFAULT '[]');
                CREATE TABLE IF NOT EXISTS work_titles (
                    root_id TEXT PRIMARY KEY, title TEXT NOT NULL, source_refs_json TEXT NOT NULL,
                    revision TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS work_title_attempts (
                    root_id TEXT PRIMARY KEY, revision TEXT NOT NULL,
                    attempts INTEGER NOT NULL, next_attempt REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS work_untitled ON work_roots(contact_id,created_at,id) WHERE title IS NULL;
                CREATE TABLE IF NOT EXISTS work_verdict_coverage (
                    contact_id TEXT PRIMARY KEY, reviewed_at TEXT NOT NULL, coverage_json TEXT NOT NULL);
            ''')
            columns = {row[1] for row in conn.execute('PRAGMA table_info(work_roots)')}
            if 'generation' not in columns:
                conn.execute('ALTER TABLE work_roots ADD COLUMN generation INTEGER NOT NULL DEFAULT 0')
            if 'frozen_revision' not in columns:
                conn.execute("ALTER TABLE work_roots ADD COLUMN frozen_revision TEXT NOT NULL DEFAULT ''")
            overlay_columns = {row[1] for row in conn.execute('PRAGMA table_info(work_overlays)')}
            if 'verdict_at' not in overlay_columns:
                conn.execute("ALTER TABLE work_overlays ADD COLUMN verdict_at TEXT NOT NULL DEFAULT ''")
            if 'verdict_id' not in overlay_columns:
                conn.execute("ALTER TABLE work_overlays ADD COLUMN verdict_id TEXT NOT NULL DEFAULT ''")
            if 'identity_json' not in overlay_columns:
                conn.execute("ALTER TABLE work_overlays ADD COLUMN identity_json TEXT NOT NULL DEFAULT '{}'")
            title_columns = {row[1] for row in conn.execute('PRAGMA table_info(work_titles)')}
            if 'provenance_version' not in title_columns:
                # Old batches did not retain all exposed inputs. Clear their
                # titles and retry markers atomically before marking the schema.
                conn.execute('BEGIN IMMEDIATE')
                if any(row[1] == 'provenance_version' for row in conn.execute('PRAGMA table_info(work_titles)')):
                    return  # Another initializer completed the migration first.
                conn.execute('UPDATE work_roots SET title=NULL WHERE title IS NOT NULL')
                conn.execute('DELETE FROM work_titles')
                conn.execute('DELETE FROM work_title_attempts')
                conn.execute('ALTER TABLE work_titles ADD COLUMN provenance_version INTEGER NOT NULL DEFAULT 0')

    def _connect(self):
        if self._read_only:
            conn = sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True, timeout=10)
            conn.row_factory = sqlite3.Row
            return conn
        conn = sqlite3.connect(self.path, timeout=10)
        os.chmod(self.path, 0o600)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA busy_timeout=10000')
        conn.execute('PRAGMA secure_delete=ON')
        return conn

    @contextmanager
    def _reading(self):
        # Read snapshots never reserve the canonical ledger's writer lock.
        with closing(self._connect()) as conn:
            source_path = (self.ledger.db_path.resolve().as_uri() + '?mode=ro'
                           if self._read_only else str(self.ledger.db_path))
            conn.execute('ATTACH DATABASE ? AS ledger', (source_path,))
            conn.execute('PRAGMA query_only=ON')
            conn.execute('BEGIN')
            yield conn, conn

    def _projection_missing(self):
        return self._read_only and not (self.path.is_file() and self.ledger.db_path.is_file())

    def schedule(self, root_id=None):
        """Wake one worker; admission already persisted the exact dirty root."""
        with self._lock:
            self._wake = True
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._drain, name='work-projection', daemon=True)
                self._worker.start()

    def _pending_roots(self, source, limit=None):
        from colonymind.memory import curated
        rows = source.execute('''SELECT d.root_turn_id,e.source_id,e.owner_hash
            FROM work_dirty d LEFT JOIN work_envelopes e ON e.root_turn_id=d.root_turn_id AND e.kind='root'
            WHERE d.pending=1''')
        result = []
        for row in rows:
            ref = dict(source_id=row['source_id'] or row['root_turn_id'], message_hash=row['owner_hash'])
            if not curated.is_forgotten(self.ledger, ref, conn=source):
                result.append(row)
                if limit and len(result) >= limit:
                    break
        return result

    def _drain(self):
        while True:
            with self._lock:
                self._wake = False
            try:
                with closing(self.ledger._connect()) as source:
                    pending = self._pending_roots(source, limit=64)
                for row in pending:
                    self.rebuild_root(row[0])
            except Exception:
                logger.exception('work projection remains pending')
                with self._lock:
                    self._worker = None
                return
            with self._lock:
                if not pending and not self._wake:
                    self._worker = None
                    return

    def wait_pending(self, timeout=2):
        """Maintenance/test barrier; never called by an HTTP handler."""
        with self._lock:
            worker = self._worker
        if worker is not None:
            worker.join(timeout)
            if worker.is_alive():
                return False
        with closing(self.ledger._connect()) as source:
            return not self._pending_roots(source, limit=1)

    def admit_event(self, event, *, contact_id, principal_id):
        from colonymind.secrets.seal import Scope, Staged, map_leaves, seal_text
        from colonymind.util.temporal import ledger_stamp
        model = event if isinstance(event, WorkEvent) else WorkEvent.model_validate(event)
        stage = Staged()
        scope = Scope(contact_id=contact_id, session_id=model.session_id, source_turn=model.root_turn_id)
        value = map_leaves(model.model_dump(exclude_none=True),
            lambda text: seal_text(text, scope=scope, vault=self.ledger.vault, stage=stage),
            skip=('source_refs', 'intended_sha256', 'actual_sha256'))
        value = WorkEvent.model_validate(value).model_dump(exclude_none=True)
        pending = None
        with closing(self.ledger._connect()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            now = ledger_stamp()
            try:
                created = facts.admit_event(conn, value, contact_id=contact_id,
                                           principal_id=principal_id, occurred_at=now)
            except ValueError as error:
                if str(error) != 'work_parent_missing':
                    raise
                conn.execute('INSERT OR IGNORE INTO work_event_wait VALUES (?,?)', (model.event_id, now))
                pending = error
            else:
                conn.execute('DELETE FROM work_event_wait WHERE event_id=?', (model.event_id,))
                if created:
                    stage.commit()
        if pending is not None:
            raise pending
        self.schedule(model.root_turn_id)
        return {'accepted': True, 'duplicate': not created}

    def _hand_effects(self, root, envelope):
        path = self.ledger.db_path.parent / 'hands.db'
        if not path.is_file():
            return []
        result = []
        try:
            with closing(sqlite3.connect(f'file:{path}?mode=ro', uri=True)) as conn:
                conn.row_factory = sqlite3.Row
                columns = {row[1] for row in conn.execute('PRAGMA table_info(hands)')}
                decisions = 'h.decisions' if 'decisions' in columns else "'{}'"
                rows = conn.execute(f'''SELECT h.id,h.task_id,h.state,h.evidence,h.intent_version,
                    {decisions} AS decisions,
                    t.intent_version AS current_version,t.origin_ref FROM hands h JOIN tasks t ON t.id=h.task_id
                    WHERE t.origin='owner_turn' ''').fetchall()
                for row in rows:
                    if _origin(row['origin_ref']) != _origin(envelope.get('origin_ref') or root['root_turn_id']):
                        continue
                    if json.loads(row['decisions'] or '{}').get('erased'):
                        result.append(self._unknown(root['root_turn_id'], 'hand_erased:' + row['id']))
                        continue
                    evidence = json.loads(row['evidence'] or '{}')
                    level = evidence.get('level')
                    status = 'dispatched'
                    if row['state'] in {'failed', 'below_minimum', 'lost', 'unknown_effect'}:
                        status = 'failed' if row['state'] in {'failed', 'below_minimum'} else 'unknown'
                    elif row['state'] == 'done':
                        # E5 is an owner verdict, reserved for the separate overlay.
                        checked = level in {'E3', 'E4'} and bool(evidence.get('scope'))
                        current = row['intent_version'] == row['current_version']
                        status = 'verified' if checked and current else 'ran'
                    result.append({'id': _effect_id(root['root_turn_id'], 'hand', row['id']),
                        'root_id': root['root_turn_id'], 'operation_id': 'hand:' + row['id'],
                        'tool': 'hands', 'action': 'outcome', 'resource_ref': 'task:' + row['task_id'],
                        'status': status, 'kind': 'dispatch', 'verification': 'hands_' + str(level or 'pending'),
                        'source_refs': [], 'evidence': {'kind': 'hand_outcome', 'task_id': row['task_id']}})
        except (sqlite3.DatabaseError, ValueError):
            # An unreadable existing controller never becomes proof of success.
            result.append(self._unknown(root['root_turn_id'], 'hands_unavailable'))
        return result

    @staticmethod
    def _unknown(root_id, suffix):
        return {'id': _effect_id(root_id, 'coverage', suffix), 'root_id': root_id,
                'operation_id': 'coverage:' + suffix, 'tool': 'capture', 'action': 'incomplete',
                'resource_ref': None, 'status': 'unknown', 'kind': 'effect_unknown',
                'verification': 'incomplete_capture', 'source_refs': [], 'evidence': {}}

    @staticmethod
    def _delete_root(conn, ident):
        for table, key in (('work_roots', 'id'), ('work_effects', 'root_id'),
                           ('work_children', 'root_id'), ('work_titles', 'root_id'), ('work_overlays', 'root_id')):
            conn.execute(f'DELETE FROM {table} WHERE {key}=?', (ident,))

    def reconcile(self):
        """Explicit maintenance only: repair every root, never from a request."""
        self._initialize_projection()
        with closing(self.ledger._connect()) as source:
            roots = [row[0] for row in source.execute('SELECT root_turn_id FROM work_dirty')]
        for ident in roots:
            self.rebuild_root(ident)
        with closing(self._connect()) as conn:
            return conn.execute('SELECT count(*) FROM work_roots').fetchone()[0]

    def rebuild_root(self, ident):
        """Read one source snapshot, write its projection, then acknowledge briefly."""
        with closing(self.ledger._connect()) as source, source:
            source.execute('BEGIN')
            row = source.execute('SELECT generation FROM work_dirty WHERE root_turn_id=?', (ident,)).fetchone()
            if row is None:
                return
            generation = row[0]
            envelopes = source.execute('SELECT * FROM work_envelopes WHERE root_turn_id=? ORDER BY created_at,event_id', (ident,)).fetchall()
            root = next((row for row in envelopes if row['kind'] == 'root'), None)
            from colonymind.memory import curated
            if root is not None and curated.is_forgotten(self.ledger, dict(source_id=root['source_id'],
                    message_hash=root['owner_hash']), conn=source):
                return  # retained projection; live readers withdraw it without cleanup
            with closing(self._connect()) as conn, conn:
                conn.execute('BEGIN IMMEDIATE')
                with closing(self.ledger._connect()) as fence:
                    current = fence.execute('SELECT generation FROM work_dirty WHERE root_turn_id=?', (ident,)).fetchone()
                    if current is None or current[0] != generation:
                        return
                newer = conn.execute('SELECT generation FROM work_roots WHERE id=?', (ident,)).fetchone()
                if newer and newer[0] > generation:
                    return
                if root is None or not facts.root_live(source, root):
                    self._delete_root(conn, ident)
                else:
                    self._project_root(source, conn, root, envelopes, generation)
        # No projection computation runs while the source write lock is held.
        with closing(self.ledger._connect()) as source, source:
            source.execute('UPDATE work_dirty SET pending=0 WHERE root_turn_id=? AND generation=?', (ident, generation))

    def _project_root(self, source, conn, root, envelopes, generation):
        ident = root['root_turn_id']
        root_work = json.loads(root['envelope_json'])
        effects, children, seen = [], [], {}
        for row in envelopes:
            if (row['invalidated'] or not facts.current_refs(source,
                    json.loads(row['source_refs_json']), root['contact_id'])):
                effects.append(self._unknown(ident, 'erased:' + row['event_id']))
                continue
            work = json.loads(row['envelope_json'])
            children.extend(work['children'])
            for raw in work['operations']:
                if not facts.current_refs(source, raw.get('source_refs', []), root['contact_id']):
                    effects.append(self._unknown(ident, raw['operation_id']))
                    continue
                operation = raw
                if operation['operation_id'] in seen:
                    continue
                seen[operation['operation_id']] = operation
                status, reason = verify(operation)
                effects.append({'id': _effect_id(ident, 'operation', operation['operation_id']), 'root_id': ident,
                    'operation_id': operation['operation_id'], 'tool': operation['tool'],
                    'action': operation['action'], 'resource_ref': operation.get('resource_ref'),
                    'status': status, 'kind': effect_kind(operation), 'verification': reason,
                    'source_refs': operation.get('source_refs', []), 'evidence': operation.get('evidence', {})})
            if work['truncated']:
                effects.append(self._unknown(ident, row['event_id']))
        # A later successful read observes only the same exact resource.
        for index, read in enumerate(effects):
            if read['kind'] != 'read' or read['status'] != 'ran' or not read['resource_ref']:
                continue
            for earlier in effects[:index]:
                check = read['evidence']
                if (check.get('checked_operation_id') == earlier['operation_id']
                        and check.get('kind') == 'saved_job' and read['tool'] == 'cronjob'
                        and earlier['tool'] == 'cronjob' and earlier['kind'] in {'create', 'edit'}
                        and earlier['resource_ref'] == read['resource_ref']
                        and earlier['status'] != 'failed'):
                    checked = {**seen[earlier['operation_id']], 'evidence': check}
                    earlier['status'], earlier['verification'] = verify(checked)
                if (earlier['status'] == 'ran' and earlier['kind'] != 'read'
                        and earlier['resource_ref'] == read['resource_ref']):
                    earlier['status'], earlier['verification'] = 'observed', 'later_exact_resource_read'
        effects.extend(self._hand_effects(root, root_work))
        unique_children = {child['turn_id']: child for child in children}
        watermark = _watermark(source, root['contact_id'])
        # Only frozen root evidence changes the revision. Runtime hands state
        # and another root's erasure cannot erase titles.
        frozen_revision = _frozen_revision(source, root, envelopes)
        revision = canonical_turn_digest({'operations': seen, 'envelopes': frozen_revision})
        if not any(item['kind'] != 'read' for item in effects):
            self._delete_root(conn, ident)
            return
        old = conn.execute('SELECT revision,title FROM work_roots WHERE id=?', (ident,)).fetchone()
        title = old['title'] if old and old['revision'] == revision else None
        if old and old['revision'] != revision:
            conn.execute('DELETE FROM work_overlays WHERE root_id=?', (ident,))
            conn.execute('DELETE FROM work_titles WHERE root_id=?', (ident,))
        for table in ('work_titles', 'work_overlays'):
            derived = conn.execute(f'SELECT source_refs_json FROM {table} WHERE root_id=?', (ident,)).fetchone()
            if derived and not facts.current_refs(source, json.loads(derived[0]), root['contact_id']):
                conn.execute(f'DELETE FROM {table} WHERE root_id=?', (ident,))
                if table == 'work_titles':
                    title = None
        # Vetted decisions are canonical facts. Rehydrate a lost projection only
        # from the latest decision already applied to this semantic revision.
        if not conn.execute('SELECT 1 FROM work_overlays WHERE root_id=?', (ident,)).fetchone():
            verdict = source.execute('''SELECT * FROM work_verdict_pending WHERE root_id=? AND applied_revision=?
                AND state IN ('accepted','linked') ORDER BY verdict_at DESC,id DESC LIMIT 1''', (ident, revision)).fetchone()
            if verdict and facts.current_refs(source, json.loads(verdict['source_refs_json']), root['contact_id']):
                from .verdicts import adjacent
                if adjacent(source, json.loads(verdict['identity_json']), root['contact_id']):
                    conn.execute('INSERT INTO work_overlays VALUES (?,?,?,?,?,?)',
                        (ident, verdict['verdict'], verdict['source_refs_json'], verdict['verdict_at'],
                         verdict['id'], verdict['identity_json']))
        conn.execute('INSERT OR REPLACE INTO work_roots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (ident, root['contact_id'], root['session_id'], ident, root_status(effects), root['created_at'],
             title, revision, root['source_id'], watermark, root_work.get('origin_ref'), generation, frozen_revision))
        conn.execute('DELETE FROM work_effects WHERE root_id=?', (ident,))
        for effect in effects:
            conn.execute('INSERT OR REPLACE INTO work_effects VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                (effect['id'], ident, effect['operation_id'], effect['tool'], effect['action'],
                 effect['resource_ref'], effect['status'], effect['kind'], effect['verification'],
                 _json(effect['source_refs']), _json(effect['evidence'])))
        conn.execute('DELETE FROM work_children WHERE root_id=?', (ident,))
        conn.executemany('INSERT INTO work_children VALUES (?,?,?)',
            [(ident, item['turn_id'], item['parent_turn_id']) for item in unique_children.values()])

    @staticmethod
    def _bounds(contact_id, since, until, status=None):
        where, args = ['contact_id=?'], [contact_id]
        for value, op in ((since, '>='), (until, '<')):
            if value is not None:
                where.append('julianday(created_at)' + op + 'julianday(?)')
                args.append(value.isoformat() if hasattr(value, 'isoformat') else value)
        if status:
            where.append('status=?')
            args.append(status)
        return ' AND '.join(where), args

    @staticmethod
    def _public_root(row, source=None, projection=None):
        row = dict(row)
        projection = projection or source
        if row['title'] is not None and source is not None:
            title = projection.execute('SELECT * FROM work_titles WHERE root_id=?', (row['id'],)).fetchone()
            if (not title or 'provenance_version' not in title.keys() or title['provenance_version'] != 1
                    or title['revision'] != row['revision']
                    or not facts.current_refs(source, json.loads(title['source_refs_json']), row['contact_id'])):
                row['title'] = None
        if source is not None:
            from colonymind.memory import curated
            effects = [WorkStore._unknown(row['id'], 'withdrawn:' + effect['id'])
                       if curated.is_forgotten(source, json.loads(effect['source_refs_json'])) else dict(effect)
                       for effect in projection.execute(
                           'SELECT * FROM work_effects WHERE root_id=? ORDER BY id', (row['id'],))]
            row['status'] = root_status(effects)
        overlay = None
        if source is not None:
            derived = projection.execute('SELECT * FROM work_overlays WHERE root_id=?', (row['id'],)).fetchone()
            if derived and facts.current_refs(source, json.loads(derived['source_refs_json']), row['contact_id']):
                from .verdicts import adjacent
                identity = json.loads(derived['identity_json']) if 'identity_json' in derived.keys() else {}
                if not identity or adjacent(source, identity, row['contact_id']):
                    overlay = derived['verdict']
        return {**{key: row[key] for key in ('id', 'contact_id', 'session_id', 'turn_id', 'status', 'created_at', 'title', 'revision')},
                'overlay': overlay}

    @staticmethod
    def _root_ref(row, source):
        root = source.execute("SELECT owner_hash FROM work_envelopes WHERE root_turn_id=? AND kind='root'",
                              (row['id'],)).fetchone()
        return dict(source_id=row['source_id'], message_hash=root[0]) if root else None

    def _visible(self, row, source):
        from colonymind.memory import curated
        ref = self._root_ref(row, source)
        return ref is not None and not curated.is_forgotten(self.ledger, ref, conn=source)

    def list_roots(self, *, contact_id, since=None, until=None, status=None, limit=50, offset=0):
        from colonymind.memory import curated
        return curated.fenced_read(self.ledger, lambda: self._read_list_roots(contact_id=contact_id, since=since, until=until, status=status, limit=limit, offset=offset),
                                  empty=[])

    def _read_list_roots(self, *, contact_id, since=None, until=None, status=None, limit=50, offset=0):
        if self._projection_missing():
            return []
        where, args = self._bounds(contact_id, since, until, status)
        with self._reading() as (conn, source):
            return [self._public_root(row, source) for row in conn.execute('SELECT * FROM work_roots WHERE ' + where
                + ' AND ' + CURRENT + ' ORDER BY created_at DESC,id LIMIT ? OFFSET ?',
                [*args, min(max(limit, 1), 101), min(max(offset, 0), 10000)]) if self._visible(row, source)]

    def get_root(self, ident, *, contact_id):
        from colonymind.memory import curated
        return curated.fenced_read(self.ledger, lambda: self._read_get_root(ident, contact_id=contact_id),
                                  empty=None)

    def _read_get_root(self, ident, *, contact_id):
        if self._projection_missing():
            return None
        with self._reading() as (conn, source):
            row = conn.execute('SELECT * FROM work_roots WHERE id=? AND contact_id=? AND ' + CURRENT, (ident, contact_id)).fetchone()
            if row is None or not self._visible(row, source):
                return None
            result = self._public_root(row, source)
            effects = [dict(item) for item in conn.execute('SELECT * FROM work_effects WHERE root_id=? ORDER BY id LIMIT 257', (ident,))]
            result['effects_truncated'] = len(effects) > 256
            from colonymind.memory import curated
            effects = [effect for effect in effects if not curated.is_forgotten(self.ledger, json.loads(effect['source_refs_json']), conn=source)]
            for effect in effects:
                effect['source_refs'] = json.loads(effect.pop('source_refs_json'))
                effect.pop('evidence_json')
            result['effects'] = effects[:256]
            children = [dict(item) for item in conn.execute('SELECT turn_id,parent_turn_id FROM work_children WHERE root_id=? LIMIT 257', (ident,))]
            result['children'], result['children_truncated'] = children[:256], len(children) > 256
            return result

    def failures(self, *, contact_id, since, until):
        """Each current root created in ``[since, until)``, oldest first, for the mind's wish stage: its status (as
        ``list_roots`` computes it), the owner's overlay, the tools of its non-read effects in call order, and each failed
        non-read effect's tool with the host's failure kind (``failed`` when the call reported none), leaving out a
        failure a later verified call to the same resource made good (as ``root_status`` does). Codes only: never a
        title, an argument or a result."""
        from colonymind.memory import curated
        return curated.fenced_read(self.ledger, lambda: self._read_failures(contact_id=contact_id, since=since,
                                                                              until=until), empty=[])

    def _read_failures(self, *, contact_id, since, until):
        if self._projection_missing():
            return []
        from colonymind.memory import curated
        where, args = self._bounds(contact_id, since, until)
        result = []
        with self._reading() as (conn, source):
            for row in conn.execute('SELECT * FROM work_roots WHERE ' + where + ' AND ' + CURRENT
                                    + ' ORDER BY created_at,id', args).fetchall():
                if not self._visible(row, source):
                    continue
                public = self._public_root(row, source)
                effects = [effect for effect in conn.execute(
                    'SELECT operation_id,tool,resource_ref,kind,status,source_refs_json FROM work_effects '
                    'WHERE root_id=? ORDER BY id', (row['id'],))
                    if not curated.is_forgotten(self.ledger, json.loads(effect['source_refs_json']), conn=source)]
                kinds, order = {}, {}
                for (envelope,) in source.execute('''SELECT envelope_json FROM ledger.work_envelopes
                        WHERE root_turn_id=? AND contact_id=? AND invalidated=0 ORDER BY created_at,event_id''',
                                                  (row['id'], contact_id)):
                    for operation in json.loads(envelope).get('operations', []):
                        if not isinstance(operation, dict):
                            continue
                        order.setdefault(operation.get('operation_id'), len(order))
                        if operation.get('failure_kind'):
                            kinds[operation.get('operation_id')] = str(operation['failure_kind'])
                # The calls in the order they were made (a hand's effects, in no envelope, last).
                acting = sorted((effect for effect in effects if effect['kind'] != 'read'),
                                key=lambda effect: order.get(effect['operation_id'], len(order)))
                failed, verified = [], set()
                for effect in reversed(acting):
                    target = (effect['tool'], effect['resource_ref'])
                    if effect['status'] == 'verified' and all(target):
                        verified.add(target)
                    elif effect['status'] == 'failed' and not (all(target) and target in verified):
                        failed.append((effect['tool'], kinds.get(effect['operation_id'], 'failed')))
                result.append({'id': row['id'], 'created_at': row['created_at'], 'status': public['status'],
                               'overlay': public['overlay'], 'tools': [effect['tool'] for effect in acting],
                               'failed': failed[::-1]})
        return result

    def summary(self, *, contact_id, since=None, until=None):
        from colonymind.memory import curated
        return curated.fenced_read(self.ledger, lambda: self._read_summary(contact_id=contact_id, since=since, until=until),
                                  empty=_summary([], []))

    def _read_summary(self, *, contact_id, since=None, until=None):
        if self._projection_missing():
            return _summary([], [])
        where, args = self._bounds(contact_id, since, until)
        with self._reading() as (conn, source):
            rows = [self._public_root(row, source) for row in conn.execute('SELECT * FROM work_roots WHERE '
                + where + ' AND ' + CURRENT + ' ORDER BY created_at DESC,id', args) if self._visible(row, source)]
            effects = conn.execute('''SELECT e.root_id,e.tool,e.action,e.kind FROM work_effects e
                JOIN work_roots r ON r.id=e.root_id WHERE r.contact_id=?''', (contact_id,)).fetchall()
            result = _summary(rows, effects)
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='work_verdict_coverage'").fetchone():
                coverage = conn.execute('SELECT * FROM work_verdict_coverage WHERE contact_id=?', (contact_id,)).fetchone()
                if coverage:
                    result['verdict_coverage'] = {'reviewed_at': coverage['reviewed_at'],
                                                 **json.loads(coverage['coverage_json'])}
            return result

    def write_derived(self, ident, *, contact_id, expected_revision, title=None, overlay=None, source_refs=(),
                      expected_generation=None, verdict_order=None, verdict_identity=None):
        """Derived writer with a source-generation compare-and-set fence."""
        with closing(self.ledger._connect()) as source, source:
            source.execute('BEGIN IMMEDIATE')
            if not facts.current_refs(source, source_refs, contact_id):
                return False
            if verdict_identity:
                from .verdicts import adjacent
                if not adjacent(source, verdict_identity, contact_id):
                    return False
            with closing(self._connect()) as conn, conn:
                conn.execute('BEGIN IMMEDIATE')
                row = conn.execute('SELECT * FROM work_roots WHERE id=? AND contact_id=?', (ident, contact_id)).fetchone()
                if (not row or row['revision'] != expected_revision
                        or (expected_generation is not None and row['generation'] != expected_generation)
                        or row['generation'] != source.execute('SELECT generation FROM work_dirty WHERE root_turn_id=?', (ident,)).fetchone()[0]):
                    return False
                if title is not None:
                    from pydantic import TypeAdapter
                    from .models import Title
                    from colonymind.secrets.seal import Scope, Staged, seal_text
                    validator = TypeAdapter(Title)
                    validator.validate_python(title)
                    staged = Staged()
                    title = seal_text(title, scope=Scope(contact_id=contact_id, session_id=row['session_id'],
                        source_turn=row['source_id']), vault=self.ledger.vault, stage=staged)
                    validator.validate_python(title)
                    conn.execute('''INSERT OR REPLACE INTO work_titles
                        (root_id,title,source_refs_json,revision,provenance_version) VALUES (?,?,?,?,1)''',
                                 (ident, title, _json(list(source_refs)), expected_revision))
                    conn.execute('UPDATE work_roots SET title=? WHERE id=?', (title, ident))
                    conn.execute('DELETE FROM work_title_attempts WHERE root_id=?', (ident,))
                    staged.commit()
                if overlay is not None:
                    if overlay not in {'confirmed', 'corrected'}:
                        raise ValueError('invalid work overlay')
                    old = conn.execute('SELECT verdict_at,verdict_id FROM work_overlays WHERE root_id=?', (ident,)).fetchone()
                    if verdict_order and old and tuple(old) >= tuple(verdict_order):
                        return True
                    conn.execute('INSERT OR REPLACE INTO work_overlays VALUES (?,?,?,?,?,?)',
                                 (ident, overlay, _json(list(source_refs)), *(verdict_order or ('', '')),
                                  _json(verdict_identity or {})))
                return True


from colonymind.reports.instance import source_snapshot as _report_source


def read_summary(path, *, contact_id, since=None, until=None):
    """Read-only reporting skips roots behind their own evidence generation."""
    path = Path(path)
    if not path.is_file():
        return _summary([], [])
    ledger_path = path.parent / 'turn-idempotency.db'
    if not ledger_path.is_file():
        return _summary([], [])
    try:
        from colonymind.reports.instance import open_readonly
        from colonymind.memory import curated
        conn = open_readonly(path)
        if conn is None:
            return _summary([], [])
        with _report_source(ledger_path) as source, closing(conn):
            source.execute('BEGIN')
            conn.execute('BEGIN')
            where, args = WorkStore._bounds(contact_id, since, until)
            generations = dict(source.execute('SELECT root_turn_id,generation FROM work_dirty'))
            invalid = {row[0] for row in source.execute('SELECT source_id FROM source_attribution_invalidations')}
            rows = [WorkStore._public_root(row, source, conn) for row in conn.execute(
                'SELECT * FROM work_roots WHERE ' + where + ' ORDER BY created_at DESC,id', args)
                if row['generation'] == generations.get(row['id']) and row['source_id'] not in invalid
                and (ref := WorkStore._root_ref(row, source)) is not None
                and not curated.is_forgotten(source, ref)]
            effects = conn.execute('SELECT root_id,tool,action,kind FROM work_effects').fetchall()
            return _summary(rows, effects)
    except (sqlite3.DatabaseError, OSError):
        return _summary([], [])


def invalidate_projection(ledger):
    """Scrub stale copies immediately; repair only dirty roots on the worker."""
    path = ledger.db_path.parent / 'colonymind-work.db'
    if not path.exists():
        return
    store = WorkStore(ledger, path)
    with closing(store._connect()) as conn, conn:
        # Inspect current facts after acquiring the projection lock. A worker may
        # have completed between erasure and this cleanup; never delete that copy
        # merely because the root was once pending.
        conn.execute('BEGIN IMMEDIATE')
        with closing(ledger._connect()) as source:
            source.execute('BEGIN')
            for row in conn.execute('SELECT id,contact_id,generation,frozen_revision FROM work_roots').fetchall():
                root = source.execute("SELECT * FROM work_envelopes WHERE root_turn_id=? AND kind='root'",
                                      (row['id'],)).fetchone()
                if root is None or not facts.root_live(source, root):
                    store._delete_root(conn, row['id'])
                    continue
                envelopes = source.execute('SELECT * FROM work_envelopes WHERE root_turn_id=? ORDER BY created_at,event_id', (row['id'],)).fetchall()
                erased = any(item['invalidated'] for item in envelopes) and (
                    row['frozen_revision'] != _frozen_revision(source, root, envelopes))
                stale_effects = [effect['id'] for effect in conn.execute(
                    'SELECT id,source_refs_json FROM work_effects WHERE root_id=?', (row['id'],))
                    if not facts.current_refs(source, json.loads(effect['source_refs_json']), row['contact_id'])]
                if erased:
                    conn.execute('DELETE FROM work_effects WHERE root_id=?', (row['id'],))
                elif stale_effects:
                    conn.executemany('DELETE FROM work_effects WHERE id=?', [(ident,) for ident in stale_effects])
                for table in ('work_titles', 'work_overlays'):
                    derived = conn.execute(f'SELECT source_refs_json FROM {table} WHERE root_id=?', (row['id'],)).fetchone()
                    if erased or stale_effects or (derived and not facts.current_refs(
                            source, json.loads(derived[0]), row['contact_id'])):
                        conn.execute(f'DELETE FROM {table} WHERE root_id=?', (row['id'],))
                        if table == 'work_titles':
                            conn.execute('UPDATE work_roots SET title=NULL WHERE id=?', (row['id'],))
    from . import work_records_enabled
    if work_records_enabled():
        store.schedule()
