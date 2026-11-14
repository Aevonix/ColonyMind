"""Durable archive overlay; no session history or reparenting implementation lives here.

Integration seam: ``colonymind.session_archive.reparent.archive_reparent`` is a
synchronous callable with keyword-only arguments ``phase, session_id, request,
intent, connection``. ``request`` is the validated request body, empty on a
receipt read. ``connection`` is None except in commit.

* validate: refuse a request this engine cannot act on, raising ArchiveError with
  contract vocabulary. Return value is unused. Used on POST and receipt GET, so
  it must tolerate an empty request. Session existence is NOT ours to assert: the
  gateway owns the session inventory and refuses an unknown session there.
* plan: return a JSON object with ``jobs: [{id, original_origin,
  intended_origin}]`` containing every in-scope cron job. No mutations.
* apply: idempotently retarget and verify cron jobs; return ``reparented`` (every
  count in COUNTS) and ``tasks_reparented``. Honor reparent_to_main=False,
  reporting true only if there is no work to move.
* commit: reserved for work that must land in the same transaction as the overlay
  receipt. Reparenting moves recurring DELIVERY only, so nothing local moves and
  the engine stages nothing here; a connection with any database attached to it is
  refused, because this store's crash atomicity is a single-store guarantee. Never
  commit that connection or call cron here. Return is unused.
* rollback: restore and verify the original cron origin of every job THIS
  operation moved (``moved``, or ``attempted`` for the window where an update may
  have persisted before a raise); return ``restored: {cron_jobs: int},
  half_applied: bool, jobs_left_moved: [str]`` and, when non-empty,
  ``jobs_moved_elsewhere: [str]`` for jobs found somewhere we did not put them,
  which are left untouched rather than set back over a third party's change.

All phases receive the original request and intent (None in validate/plan).
An exception during apply/commit causes rollback. Interrupted operations always
resume rollback before a new attempt for that session; completed receipts replay
without invoking the engine. The engine may raise ArchiveError with contract
vocabulary; arbitrary exceptions are kept private by the HTTP boundary.
"""
from __future__ import annotations

import base64
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid


STATUSES = {'invalid_request': 400, 'unauthorized': 403, 'not_found': 404,
            'unsupported_capability': 409, 'idempotency_conflict': 409,
            'stale_revision': 409, 'busy': 409, 'uncertain': 409,
            'source_unavailable': 503}
COUNTS = ('cron_jobs',)


class ArchiveError(Exception):
    def __init__(self, code, **details):
        if code not in STATUSES:
            raise ValueError('unknown archive refusal')
        self.code, self.details = code, details
        super().__init__(code)

    @property
    def body(self):
        return {'error': self.code, **self.details}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _parents(path):
    missing = []
    while not path.exists():
        missing.append(path)
        path = path.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700, exist_ok=True)


class ArchiveStore:
    """Append-retaining state and receipts in one private SQLite file.

    An unarchive sets archived=0: inventory removes the row logically, while its
    last receipt and revision remain available. No rows or files are deleted.
    """
    def __init__(self, path, *, clock=None, stamp=None):
        self.path = Path(path)
        # Hermes receives the sidecar's trusted plan timestamp. Its journal can
        # use this store without importing the sidecar or another wall clock.
        self._clock = clock or time.time
        self._stamp = stamp or (lambda: datetime.fromtimestamp(self._clock(), timezone.utc).isoformat())

    @contextmanager
    def _connect(self):
        _parents(self.path.parent)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        with closing(sqlite3.connect(self.path, timeout=0)) as db:
            db.row_factory = sqlite3.Row
            # Rollback journals allow attached local databases to commit together.
            db.execute('PRAGMA journal_mode=DELETE')
            db.execute('PRAGMA synchronous=FULL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS archive_sessions (
                    destination_id TEXT PRIMARY KEY, archived INTEGER NOT NULL,
                    revision TEXT NOT NULL, archived_at TEXT, receipt_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS archive_operations (
                    operation_id TEXT PRIMARY KEY, destination_id TEXT NOT NULL,
                    request_json TEXT NOT NULL, intent_json TEXT NOT NULL,
                    state TEXT NOT NULL, refusal_json TEXT, receipt_json TEXT);
                CREATE INDEX IF NOT EXISTS archive_pending
                    ON archive_operations(destination_id,state);
                CREATE TABLE IF NOT EXISTS archive_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS archive_plans (
                    operation_id TEXT PRIMARY KEY, destination_id TEXT NOT NULL,
                    request_json TEXT NOT NULL, authority_json TEXT NOT NULL,
                    created_at TEXT NOT NULL, worker_id TEXT, claim TEXT,
                    lease_until REAL, result_json TEXT);
            ''')
            db.execute('INSERT OR IGNORE INTO archive_meta VALUES (?,?)',
                       ('cursor_key', uuid.uuid4().hex + uuid.uuid4().hex))
            db.commit()
            yield db

    @contextmanager
    def _writer(self):
        # The intent must commit before cron is called. A separate advisory lock
        # keeps that gap exclusive across processes and threads, without holding
        # a SQLite transaction open around a public core API call.
        _parents(self.path.parent)
        fd = os.open(str(self.path) + '.lock', os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.fchmod(fd, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ArchiveError('busy') from None
            with self._connect() as db:
                yield db
        finally:
            os.close(fd)

    @staticmethod
    def _call(engine, phase, session_id, body, intent=None, connection=None, record=None):
        return engine(phase=phase, session_id=session_id, request=body, intent=intent,
                      connection=connection, record=record)

    def _rollback(self, db, engine, row, diagnosis=None):
        body, intent = json.loads(row['request_json']), json.loads(row['intent_json'])
        # The phase marker is bookkeeping, and bookkeeping must never be the reason
        # a job stays moved: if this write fails, compensation still runs and the
        # answer still reports what could not be put back.
        marked = True
        try:
            db.execute("UPDATE archive_operations SET state='rolling_back' WHERE operation_id=?",
                       (row['operation_id'],))
            db.commit()
        except Exception:
            marked = False
            try:
                db.rollback()
            except Exception:
                pass
        attempted = [str(job['id']) for job in intent.get('jobs', [])
                     if job.get('moved') or job.get('attempted')]
        elsewhere = []
        try:
            result = self._call(engine, 'rollback', row['destination_id'], body, intent)
            restored = result['restored']['cron_jobs']
            left = result['jobs_left_moved']
            half = result['half_applied']
            elsewhere = result.get('jobs_moved_elsewhere', [])
            if (type(restored) is not int or restored < 0 or type(half) is not bool
                    or not isinstance(left, list) or any(not isinstance(job, str) for job in left)
                    or not isinstance(elsewhere, list)
                    or any(not isinstance(job, str) for job in elsewhere)
                    or half != bool(left)):
                raise ValueError('invalid rollback verification')
        except Exception:
            # Unknown is reported as unresolved, never as a clean refusal: only the
            # jobs this operation actually touched can be left moved by it.
            restored, left, half = 0, attempted, bool(attempted)
        archived = self._archived_state(db, row['destination_id'])
        refusal = {'destination_id': row['destination_id'], 'archived': archived,
                   'tasks_reparented': False, 'restored': {'cron_jobs': restored},
                   'half_applied': half, 'jobs_left_moved': left}
        if half:
            refusal['receipt_incomplete'] = True
        if elsewhere:
            refusal['jobs_moved_elsewhere'] = elsewhere
        # The engine's own diagnosis - which job core refused, which one moved
        # under us, and why - is merged BEFORE the row is written, not after it is
        # returned. Persisting the refusal without it meant a receipt read of an
        # interrupted operation replayed a refusal stripped of the only part that
        # tells the owner what to do. The compensation result wins any clash,
        # because it is what actually happened to the jobs.
        for key, value in (diagnosis or {}).items():
            refusal.setdefault(key, value)
        try:
            db.execute('UPDATE archive_operations SET state=?, refusal_json=? WHERE operation_id=?',
                       ('rolling_back' if half else 'refused', _json(refusal), row['operation_id']))
            db.commit()
        except Exception:
            # Bookkeeping must never discard a VERIFIED compensation result, and
            # must never be the reason the owner is told nothing. Compensation has
            # already run and its outcome is in ``refusal``; all that failed is
            # writing it down. The operation row still says 'applying' or
            # 'rolling_back', which the receipt read reports as unresolved and the
            # next attempt recovers, so the refusal says the record is incomplete
            # rather than claiming a clean refusal.
            refusal['receipt_incomplete'] = True
        return refusal

    @staticmethod
    def _operation_row(db, operation_id, fallback):
        """This operation's durable row, or ``fallback`` when the overlay cannot say.

        ``record`` keeps the in-memory intent current before every core mutation,
        so the fallback names the same jobs the row would. A failed read of our own
        bookkeeping must never be the reason compensation does not run.
        """
        try:
            row = db.execute('SELECT * FROM archive_operations WHERE operation_id=?',
                             (operation_id,)).fetchone()
        except Exception:
            row = None
        return fallback if row is None else row

    @staticmethod
    def _archived_state(db, destination_id):
        """The archived flag the overlay still holds; a refusal changed nothing."""
        try:
            row = db.execute('SELECT archived FROM archive_sessions WHERE destination_id=?',
                             (destination_id,)).fetchone()
        except Exception:
            return False
        return bool(row['archived']) if row else False

    def archive(self, session_id, body, engine, *, authorize=None):
        with self._writer() as db:
            request_json = _json(body)
            old = db.execute('SELECT * FROM archive_operations WHERE operation_id=?',
                             (body['operation_id'],)).fetchone()
            if old and (old['destination_id'] != session_id or old['request_json'] != request_json):
                raise ArchiveError('idempotency_conflict')
            if old and old['receipt_json']:
                return json.loads(old['receipt_json'])
            # Recovery never stacks another intent on unresolved cron work.
            pending = db.execute("SELECT * FROM archive_operations WHERE destination_id=? "
                                 "AND state IN ('applying','rolling_back')", (session_id,)).fetchall()
            for row in pending:
                refusal = self._rollback(db, engine, row)
                if refusal['half_applied']:
                    raise ArchiveError('uncertain', **refusal)
            if old:
                row = db.execute('SELECT * FROM archive_operations WHERE operation_id=?',
                                 (body['operation_id'],)).fetchone()
                # A refusal whose record could not be written leaves refusal_json
                # NULL. Report the unresolved state rather than failing on it.
                stored = row['refusal_json'] if row else None
                if not stored:
                    raise ArchiveError('uncertain', destination_id=session_id,
                                       archived=self._archived_state(db, session_id),
                                       tasks_reparented=False, receipt_incomplete=True)
                raise ArchiveError('uncertain', **json.loads(stored))
            if authorize is not None:
                authorize(db)
            self._call(engine, 'validate', session_id, body)
            intent = self._call(engine, 'plan', session_id, body)
            # The intent envelope is deliberately small; engine-specific local
            # plans may be additional JSON fields. Every cron origin is durable.
            if (not isinstance(intent, dict) or not isinstance(intent.get('jobs'), list)
                    or any(not isinstance(job, dict) or not isinstance(job.get('id'), str)
                           or not job['id'] or 'original_origin' not in job or 'intended_origin' not in job
                           for job in intent['jobs'])):
                raise ArchiveError('source_unavailable')
            db.execute('INSERT INTO archive_operations VALUES (?,?,?,?,?,?,?)',
                       (body['operation_id'], session_id, request_json, _json(intent), 'applying', None, None))
            db.commit()
            def record(current):
                # Durable before each core mutation: rollback must know which jobs
                # this operation touched even if the process dies mid-update.
                db.execute('UPDATE archive_operations SET intent_json=? WHERE operation_id=?',
                           (_json(current), body['operation_id']))
                db.commit()

            try:
                moved = self._call(engine, 'apply', session_id, body, intent, record=record)
                counts, verified = moved['reparented'], moved['tasks_reparented']
                if (set(counts) != set(COUNTS) or any(type(counts[k]) is not int or counts[k] < 0 for k in COUNTS)
                        or type(verified) is not bool or (body['reparent_to_main'] and not verified)):
                    raise ValueError('invalid reparent verification')
                db.execute('BEGIN IMMEDIATE')
                self._call(engine, 'commit', session_id, body, intent, db)
                # SQLite cannot guarantee a multi-database crash-atomic commit with
                # an attached WAL database, so this receipt only ever claims what one
                # store can promise. An engine that attached anything is refused
                # rather than issued a success receipt for that weaker transaction.
                if any(row['name'] not in {'main', 'temp'}
                       for row in db.execute('PRAGMA database_list').fetchall()):
                    raise ValueError('archive transaction must stay in one store')
                stamp = self._stamp()
                receipt_id = str(uuid.uuid4())
                receipt = {'destination_id': session_id, 'archived': body['archived'],
                           'tasks_reparented': verified, 'receipt_id': receipt_id,
                           'revision': _digest([session_id, receipt_id]),
                           'archived_at': stamp if body['archived'] else None,
                           'reparented': counts, 'operation_id': body['operation_id']}
                db.execute('INSERT INTO archive_sessions VALUES (?,?,?,?,?) ON CONFLICT(destination_id) '
                           'DO UPDATE SET archived=excluded.archived, revision=excluded.revision, '
                           'archived_at=excluded.archived_at, receipt_json=excluded.receipt_json',
                           (session_id, body['archived'], receipt['revision'], receipt['archived_at'], _json(receipt)))
                db.execute("UPDATE archive_operations SET state='completed', receipt_json=? WHERE operation_id=?",
                           (_json(receipt), body['operation_id']))
                db.commit()
                return receipt
            except Exception as failure:
                try:
                    db.rollback()
                except Exception:
                    pass
                # Re-read: apply records each attempt durably, so this row now says
                # which jobs were touched. The planned intent would not. But if the
                # overlay cannot answer that SELECT, compensation still runs from
                # the intent held in memory, which ``record`` kept current: a failed
                # read of our own bookkeeping must not leave a job moved.
                current = self._operation_row(
                    db, body['operation_id'],
                    {'operation_id': body['operation_id'], 'destination_id': session_id,
                     'request_json': request_json, 'intent_json': _json(intent)})
                diagnosis = {}
                if isinstance(failure, ArchiveError):
                    diagnosis = {key: failure.details[key]
                                 for key in ('refused', 'reason', 'conflicted', 'unverified')
                                 if key in failure.details}
                refusal = self._rollback(db, engine, current, diagnosis)
                raise ArchiveError('uncertain', **refusal) from None

    def receipt(self, session_id, engine):
        self._call(engine, 'validate', session_id, {})
        with self._connect() as db:
            db.execute('BEGIN')
            row = db.execute('SELECT * FROM archive_sessions WHERE destination_id=?', (session_id,)).fetchone()
            stuck = db.execute("SELECT state, refusal_json, intent_json FROM archive_operations "
                               "WHERE destination_id=? AND state IN ('applying','rolling_back')",
                               (session_id,)).fetchall()
            refusal, unresolved = {}, []
            for operation in stuck:
                if operation['refusal_json']:
                    refusal = json.loads(operation['refusal_json'])
                else:
                    # Interrupted with no refusal written: the jobs it recorded as
                    # attempted may still sit at Main. Answering clean here told the
                    # owner nothing was pending while a job was still moved.
                    unresolved += [str(job['id'])
                                   for job in json.loads(operation['intent_json']).get('jobs', [])
                                   if job.get('attempted')]
            already = refusal.get('jobs_left_moved', [])
            pending = already + [job for job in unresolved if job not in already]
            queued = db.execute('SELECT operation_id,result_json FROM archive_plans WHERE destination_id=? '
                                'ORDER BY rowid DESC LIMIT 1', (session_id,)).fetchone()
            finished = db.execute('SELECT result_json FROM archive_plans WHERE destination_id=? '
                                  'AND result_json IS NOT NULL ORDER BY rowid DESC LIMIT 1', (session_id,)).fetchone()
            remote = json.loads(finished[0]) if finished else {}
            pending += [job for job in remote.get('jobs_left_moved', []) if job not in pending]
            result = {'destination_id': session_id, 'archived': bool(row['archived']) if row else False,
                    'revision': row['revision'] if row else _digest([session_id, False]),
                    'archived_at': row['archived_at'] if row else None,
                    'receipt': json.loads(row['receipt_json']) if row else None,
                    'half_applied': bool(pending),
                    'jobs_left_moved': pending}
            if queued and queued['result_json'] is None:
                result.update(pending=True, operation_id=queued['operation_id'])
            return result

    def enqueue(self, session_id, body, authority):
        """Persist a whole request before handing it to the Hermes worker."""
        with self._writer() as db:
            encoded = _json(body)
            row = db.execute('SELECT * FROM archive_plans WHERE operation_id=?',
                             (body['operation_id'],)).fetchone()
            if row:
                if row['destination_id'] != session_id or row['request_json'] != encoded:
                    raise ArchiveError('idempotency_conflict')
                if json.loads(row['authority_json'])['owner_id'] != authority['owner_id']:
                    raise ArchiveError('unauthorized')
                return
            previous = db.execute('SELECT * FROM archive_operations WHERE operation_id=?',
                                  (body['operation_id'],)).fetchone()
            if previous and (previous['destination_id'] != session_id or previous['request_json'] != encoded):
                raise ArchiveError('idempotency_conflict')
            replay = previous['receipt_json'] if previous else None
            if previous and previous['state'] == 'refused' and previous['refusal_json']:
                replay = _json({'error': 'uncertain', **json.loads(previous['refusal_json'])})
            pending = db.execute('SELECT 1 FROM archive_plans WHERE destination_id=? '
                                 'AND result_json IS NULL', (session_id,)).fetchone()
            if pending and not replay:
                raise ArchiveError('busy')
            # Never stack a queued operation on an unresolved older local intent.
            if not replay and db.execute("SELECT 1 FROM archive_operations WHERE destination_id=? "
                          "AND state IN ('applying','rolling_back')", (session_id,)).fetchone():
                raise ArchiveError('uncertain', destination_id=session_id, receipt_incomplete=True)
            db.execute('INSERT INTO archive_plans '
                       '(operation_id,destination_id,request_json,authority_json,created_at,result_json) VALUES (?,?,?,?,?,?)',
                       (body['operation_id'], session_id, encoded, _json(authority), self._stamp(), replay))
            db.commit()

    def plan_result(self, operation_id):
        with closing(sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
            row = db.execute('SELECT result_json FROM archive_plans WHERE operation_id=?',
                             (operation_id,)).fetchone()
            return json.loads(row[0]) if row and row[0] else None

    def worker_available(self):
        """Read-only capability probe: never create a store or infer core support."""
        if not self.path.is_file():
            return False
        try:
            with closing(sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True, timeout=0)) as db:
                row = db.execute("SELECT value FROM archive_meta WHERE key='worker_heartbeat'").fetchone()
                return bool(row and 0 <= self._clock() - float(row[0]) < 60)
        except (sqlite3.Error, OSError, ValueError):
            return False

    def heartbeat(self):
        """One authenticated incoming poll, independent of queue inspection."""
        with self._writer() as db:
            db.execute('INSERT INTO archive_meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                       ('worker_heartbeat', str(self._clock())))
            db.commit()

    @staticmethod
    def _claimable(db, worker_id, owner_id, now):
        rows = db.execute('SELECT * FROM archive_plans WHERE result_json IS NULL '
                          'AND (lease_until IS NULL OR lease_until<=?) '
                          'AND (worker_id IS NULL OR worker_id=?) ORDER BY rowid', (now, worker_id)).fetchall()
        return next((r for r in rows if json.loads(r['authority_json'])['owner_id'] == owner_id), None)

    def claim_plan(self, worker_id, owner_id):
        now = self._clock()
        # Empty held polls are read-only. Recheck under the writer lock before
        # leasing a candidate; another poll may have claimed it in the meantime.
        with closing(sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True, timeout=0)) as db:
            db.row_factory = sqlite3.Row
            if self._claimable(db, worker_id, owner_id, now) is None:
                return None
        with self._writer() as db:
            row = self._claimable(db, worker_id, owner_id, now)
            plan = None
            if row:
                claim = uuid.uuid4().hex
                db.execute('UPDATE archive_plans SET worker_id=?,claim=?,lease_until=? WHERE operation_id=?',
                           (worker_id, claim, now + 60, row['operation_id']))
                plan = {'operation_id': row['operation_id'], 'destination_id': row['destination_id'],
                        'request': json.loads(row['request_json']), 'authority': json.loads(row['authority_json']),
                        'created_at': row['created_at'], 'claim': claim}
                previous = db.execute('SELECT receipt_json FROM archive_sessions WHERE destination_id=?',
                                      (row['destination_id'],)).fetchone()
                plan['previous_receipt'] = json.loads(previous[0]) if previous else None
            db.commit()
            return plan

    def accept_plan_result(self, operation_id, worker_id, claim, result):
        """Exactly one worker result and overlay commit, in the same transaction."""
        with self._writer() as db:
            row = db.execute('SELECT * FROM archive_plans WHERE operation_id=?', (operation_id,)).fetchone()
            if row is None or row['worker_id'] != worker_id:
                raise ArchiveError('unauthorized')
            encoded = _json(result)
            if row['result_json']:
                if row['result_json'] != encoded:
                    raise ArchiveError('idempotency_conflict')
                return
            if row['claim'] != claim:
                raise ArchiveError('unauthorized')
            body = json.loads(row['request_json'])
            if not isinstance(result, dict):
                raise ArchiveError('invalid_request')
            if 'error' in result:
                if result['error'] not in STATUSES:
                    raise ArchiveError('invalid_request')
            else:
                keys = {'destination_id', 'archived', 'tasks_reparented', 'receipt_id', 'revision',
                        'archived_at', 'reparented', 'operation_id'}
                counts = result.get('reparented')
                if (set(result) != keys or result['destination_id'] != row['destination_id']
                        or result['operation_id'] != operation_id or result['archived'] is not body['archived']
                        or type(result['tasks_reparented']) is not bool
                        or (body['reparent_to_main'] and not result['tasks_reparented'])
                        or not isinstance(counts, dict) or set(counts) != {'cron_jobs'}
                        or type(counts['cron_jobs']) is not int or counts['cron_jobs'] < 0
                        or not isinstance(result['receipt_id'], str) or not result['receipt_id']
                        or result['revision'] != _digest([row['destination_id'], result['receipt_id']])
                        or (body['archived'] and not isinstance(result['archived_at'], str))
                        or (not body['archived'] and result['archived_at'] is not None)):
                    raise ArchiveError('invalid_request')
                db.execute('INSERT INTO archive_sessions VALUES (?,?,?,?,?) ON CONFLICT(destination_id) '
                           'DO UPDATE SET archived=excluded.archived,revision=excluded.revision,'
                           'archived_at=excluded.archived_at,receipt_json=excluded.receipt_json',
                           (row['destination_id'], result['archived'], result['revision'], result['archived_at'], encoded))
            db.execute('UPDATE archive_plans SET result_json=? WHERE operation_id=?', (encoded, operation_id))
            db.commit()

    def inventory(self, limit=50, cursor=None):
        with self._connect() as db:
            db.execute('BEGIN')
            rows = db.execute('SELECT * FROM archive_sessions ORDER BY archived_at DESC,destination_id ASC').fetchall()
            revision = _digest([[row['destination_id'], row['revision']] for row in rows])
            secret = db.execute("SELECT value FROM archive_meta WHERE key='cursor_key'").fetchone()[0].encode()
            offset, stamp = 0, self._stamp()
            if cursor is not None:
                try:
                    raw, signature = cursor.split('.')
                    if not hmac.compare_digest(signature, hmac.new(secret, raw.encode(), hashlib.sha256).hexdigest()):
                        raise ValueError('cursor signature')
                    token = json.loads(base64.urlsafe_b64decode(raw + '=' * (-len(raw) % 4)))
                    if (token['limit'] != limit or type(token['offset']) is not int or token['offset'] < 0):
                        raise ValueError('cursor bounds')
                    offset, stamp = token['offset'], token['snapshot_at']
                except (ValueError, KeyError, TypeError, UnicodeError):
                    raise ArchiveError('invalid_request') from None
                if token['revision'] != revision:
                    raise ArchiveError('stale_revision')
            archived = [row for row in rows if row['archived']]
            items = []
            for row in archived[offset:offset + limit]:
                receipt = json.loads(row['receipt_json'])
                items.append({key: receipt[key] for key in ('destination_id', 'revision', 'archived_at',
                                                          'receipt_id', 'tasks_reparented')})
            more = offset + limit < len(archived)
            next_cursor = None
            if more:
                token = {'revision': revision, 'snapshot_at': stamp, 'offset': offset + limit, 'limit': limit}
                raw = base64.urlsafe_b64encode(_json(token).encode()).decode().rstrip('=')
                next_cursor = raw + '.' + hmac.new(secret, raw.encode(), hashlib.sha256).hexdigest()
            return {'revision': revision, 'snapshot_at': stamp, 'items': items,
                    'counts': {'archived': len(archived)}, 'has_more': more, 'next_cursor': next_cursor}
