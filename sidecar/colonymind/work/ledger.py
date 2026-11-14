"""Safe immutable work facts admitted in the canonical source transaction."""
from __future__ import annotations

import json
from datetime import datetime

PARENT_WAIT_SECONDS = 300

from colonymind.turns.idempotency import canonical_turn_digest, source_message_hash


def initialize(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS work_envelopes (
        event_id TEXT PRIMARY KEY, root_turn_id TEXT NOT NULL, source_id TEXT NOT NULL,
        contact_id TEXT NOT NULL, session_id TEXT NOT NULL, principal_id TEXT NOT NULL,
        kind TEXT NOT NULL, content_sha256 TEXT NOT NULL, envelope_json TEXT NOT NULL,
        source_refs_json TEXT NOT NULL, owner_hash TEXT NOT NULL,
        invalidated INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL)''')
    conn.execute('CREATE INDEX IF NOT EXISTS work_envelope_root ON work_envelopes(root_turn_id,contact_id)')
    conn.execute('''CREATE INDEX IF NOT EXISTS work_envelope_origin
        ON work_envelopes(json_extract(envelope_json,'$.origin_ref')) WHERE kind='root' AND invalidated=0''')
    dirty_exists = conn.execute("SELECT 1 FROM sqlite_master WHERE name='work_dirty'").fetchone()
    conn.execute('''CREATE TABLE IF NOT EXISTS work_dirty (
        root_turn_id TEXT PRIMARY KEY, generation INTEGER NOT NULL, pending INTEGER NOT NULL)''')
    conn.execute('CREATE INDEX IF NOT EXISTS work_dirty_pending ON work_dirty(pending)')
    conn.execute('''CREATE TABLE IF NOT EXISTS work_event_wait (
        event_id TEXT PRIMARY KEY, first_seen_at TEXT NOT NULL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS work_verdict_pending (
        id TEXT PRIMARY KEY, root_id TEXT NOT NULL, contact_id TEXT NOT NULL,
        generation INTEGER NOT NULL, verdict TEXT NOT NULL, verdict_at TEXT NOT NULL,
        source_refs_json TEXT NOT NULL, identity_json TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
        next_attempt REAL NOT NULL DEFAULT 0, state TEXT NOT NULL DEFAULT 'pending',
        applied_revision TEXT NOT NULL DEFAULT '', captured_revision TEXT NOT NULL DEFAULT '')''')
    verdict_columns = {row[1] for row in conn.execute('PRAGMA table_info(work_verdict_pending)')}
    if 'applied_revision' not in verdict_columns:
        conn.execute("ALTER TABLE work_verdict_pending ADD COLUMN applied_revision TEXT NOT NULL DEFAULT ''")
    if 'captured_revision' not in verdict_columns:
        conn.execute("ALTER TABLE work_verdict_pending ADD COLUMN captured_revision TEXT NOT NULL DEFAULT ''")
    conn.execute('CREATE INDEX IF NOT EXISTS work_verdict_due ON work_verdict_pending(contact_id,state,next_attempt)')
    conn.execute('CREATE INDEX IF NOT EXISTS work_verdict_root ON work_verdict_pending(root_id,applied_revision,verdict_at)')
    conn.execute('''CREATE INDEX IF NOT EXISTS work_session_chronology
        ON turn_sources(contact_id,session_id,julianday(coalesce(occurred_at,ingested_at))) WHERE scope='person' ''')
    if not dirty_exists:
        conn.execute('''INSERT OR IGNORE INTO work_dirty
            SELECT root_turn_id,1,1 FROM work_envelopes WHERE kind='root' ''')


def mark_dirty(conn, root_turn_id):
    conn.execute('''INSERT INTO work_dirty VALUES (?,1,1) ON CONFLICT(root_turn_id)
        DO UPDATE SET generation=generation+1,pending=1''', (root_turn_id,))


def attribution_changed(conn, source_ids):
    """Fence affected roots inside the source correction transaction."""
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='work_envelopes'").fetchone():
        return False
    selected, roots = set(source_ids), set()
    for row in conn.execute('SELECT * FROM work_envelopes WHERE invalidated=0'):
        work = json.loads(row['envelope_json'])
        refs = json.loads(row['source_refs_json'])
        refs += [ref for op in work.get('operations', []) for ref in op.get('source_refs', [])]
        if row['source_id'] in selected or any(ref['source_id'] in selected for ref in refs):
            roots.add(row['root_turn_id'])
    # Native evidence may have been joined by the worker after admission.
    for sid in selected:
        row = conn.execute('SELECT messages_json FROM turn_sources WHERE turn_id=?', (sid,)).fetchone()
        for message in json.loads(row[0]) if row else []:
            if message.get('_native_tool_observation') == 'native-tool-observation-v1':
                ident = message.get('provenance', {}).get('native', {}).get('turn_id')
                if conn.execute('SELECT 1 FROM work_dirty WHERE root_turn_id=?', (ident,)).fetchone():
                    roots.add(ident)
    for ident in roots:
        mark_dirty(conn, ident)
    return bool(roots)


def current_refs(conn, refs, contact_id):
    from colonymind.memory import curated
    if curated.is_forgotten(conn, refs):
        return False
    for ref in refs:
        row = conn.execute('''SELECT messages_json FROM turn_sources WHERE turn_id=? AND contact_id=?
            AND NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=turn_sources.turn_id)''',
            (ref['source_id'], contact_id)).fetchone()
        if row is None or canonical_turn_digest(json.loads(row[0])) != ref['source_version']:
            return False
    return True


def root_live(conn, row):
    from colonymind.memory import curated
    if row['invalidated'] or curated.is_forgotten(conn, dict(source_id=row['source_id'], message_hash=row['owner_hash'])):
        return False
    source = conn.execute('''SELECT session_id,messages_json FROM turn_sources WHERE turn_id=? AND contact_id=?
        AND NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=turn_sources.turn_id)''',
        (row['source_id'], row['contact_id'])).fetchone()
    return bool(source and any(message.get('role') == 'user'
        and source_message_hash(source['session_id'], message) == row['owner_hash']
        for message in json.loads(source['messages_json'])))


def _check_operations(conn, work, contact_id):
    for operation in work['operations']:
        if not current_refs(conn, operation.get('source_refs', []), contact_id):
            raise ValueError('work_source_revision_missing')
    # An operation identity cannot be silently replaced by a later observation.
    for row in conn.execute('SELECT envelope_json FROM work_envelopes WHERE root_turn_id=? AND invalidated=0',
                            (work['root_turn_id'],)):
        previous = {op['operation_id']: op for op in json.loads(row[0]).get('operations', [])}
        for operation in work['operations']:
            if operation['operation_id'] in previous and previous[operation['operation_id']] != operation:
                raise ValueError('work_operation_conflict')


def admit_turn(conn, *, work, source_id, contact_id, session_id, messages, principal_id, occurred_at):
    initialize(conn)
    if work['root_turn_id'] != source_id or work.get('parent_turn_id') is not None:
        raise ValueError('work_owner_root_mismatch')
    owners = [message for message in messages if message.get('role') == 'user']
    if len(owners) != 1:
        raise ValueError('work_owner_source_required')
    from colonymind.memory import curated
    if curated.is_forgotten(conn, {'source_id': source_id,
            'message_hash': source_message_hash(session_id, owners[0])}):
        raise ValueError('work_source_erased')
    _check_operations(conn, work, contact_id)
    for child in work['children']:
        if child['parent_turn_id'] != source_id or child['turn_id'] == source_id:
            raise ValueError('work_child_parent_mismatch')
    return _insert(conn, event_id='turn:' + source_id, work=work, source_id=source_id,
        contact_id=contact_id, session_id=session_id, principal_id=principal_id, kind='root', refs=[],
        owner_hash=source_message_hash(session_id, owners[0]), occurred_at=occurred_at)


def try_admit_turn(conn, *, result=None, **kwargs):
    """Work is optional: refusing it must never roll back ordinary source admission."""
    try:
        admit_turn(conn, **kwargs)
    except ValueError as error:
        if result is not None:
            result.update(recorded=False, reason=str(error))
        return False
    if result is not None:
        result.update(recorded=True, reason=None)
    return True


def _insert(conn, *, event_id, work, source_id, contact_id, session_id, principal_id, kind, refs, owner_hash, occurred_at):
    digest = canonical_turn_digest({'work': work, 'source_id': source_id, 'contact_id': contact_id,
        'session_id': session_id, 'principal_id': principal_id, 'source_refs': refs})
    previous = conn.execute('SELECT content_sha256,invalidated FROM work_envelopes WHERE event_id=?', (event_id,)).fetchone()
    if previous:
        if previous['content_sha256'] != digest:
            raise ValueError('work_event_conflict')
        if previous['invalidated']:
            raise ValueError('work_source_erased')
        return False
    conn.execute('INSERT INTO work_envelopes VALUES (?,?,?,?,?,?,?,?,?,?,?,0,?)',
        (event_id, work['root_turn_id'], source_id, contact_id, session_id, principal_id, kind, digest,
         json.dumps(work, sort_keys=True), json.dumps(refs, sort_keys=True), owner_hash, occurred_at))
    mark_dirty(conn, work['root_turn_id'])
    return True


def admit_event(conn, event, *, contact_id, principal_id, occurred_at):
    initialize(conn)
    work = event['work']
    root = conn.execute('SELECT * FROM work_envelopes WHERE root_turn_id=? AND kind=? AND contact_id=?',
                        (event['root_turn_id'], 'root', contact_id)).fetchone()
    if not root:
        if conn.execute('SELECT 1 FROM source_erasures WHERE turn_id=?', (event['root_turn_id'],)).fetchone():
            raise ValueError('work_root_erased')
        if conn.execute('SELECT 1 FROM turn_sources WHERE turn_id=?', (event['root_turn_id'],)).fetchone():
            raise ValueError('work_root_ineligible')
        raise ValueError('work_root_missing')
    if not root_live(conn, root):
        raise ValueError('work_root_erased')
    if root['principal_id'] != principal_id:
        raise ValueError('work_principal_mismatch')
    refs = event.get('source_refs', [])
    if not current_refs(conn, refs, contact_id):
        raise ValueError('work_source_revision_missing')
    known = {root['root_turn_id']}
    for row in conn.execute('SELECT envelope_json FROM work_envelopes WHERE root_turn_id=? AND invalidated=0',
                            (root['root_turn_id'],)):
        known.update(child['turn_id'] for child in json.loads(row[0]).get('children', []))
    if work.get('parent_turn_id') is not None and work['parent_turn_id'] not in known:
        wait = conn.execute('SELECT first_seen_at FROM work_event_wait WHERE event_id=?', (event['event_id'],)).fetchone()
        if wait and (datetime.fromisoformat(occurred_at) - datetime.fromisoformat(wait[0])).total_seconds() >= PARENT_WAIT_SECONDS:
            raise ValueError('work_parent_expired')
        raise ValueError('work_parent_missing')
    for child in work['children']:
        if child['parent_turn_id'] not in known or child['turn_id'] == child['parent_turn_id']:
            raise ValueError('work_child_parent_mismatch')
    _check_operations(conn, work, contact_id)
    return _insert(conn, event_id='event:' + event['event_id'], work=work, source_id=root['source_id'],
        contact_id=contact_id, session_id=event['session_id'], principal_id=principal_id, kind='event', refs=refs,
        owner_hash=root['owner_hash'], occurred_at=occurred_at)


def root_candidate(body, *, request, contact_id, resolved_sender):
    """Called after sender resolution and credential checks, never from hooks."""
    from colonymind.api.auth import owner_person_id, owner_platform_granted, request_authority
    from colonymind.api.credentials import platform_of
    from . import work_records_enabled
    if body.work is None or not work_records_enabled() or contact_id != owner_person_id():
        return None
    message = body.user_message
    if message is None or message.role != 'user' or not (
            message.content.strip() if isinstance(message.content, str) else message.content):
        return None
    platform = platform_of(body.sender.platform if body.sender else None, body.context.channel_id)
    lane = (body.context.channel_id or '').lower()
    session = body.context.session_id.lower()
    excluded = {'cron', 'background_review', 'kanban', 'worker', 'mind', 'subagent', 'system', 'internal'}
    if platform in excluded or any(lane.startswith(prefix + ':') for prefix in excluded) or any(
            session.startswith(prefix) for prefix in ('cron_', 'cron:', 'kanban:', 'kanban_', 'mind:', 'worker:', 'subagent:')):
        return None
    authority = request_authority(request)
    if not owner_platform_granted(authority, platform):
        return None
    if body.sender is None:
        if platform != 'cli':
            return None
    elif not resolved_sender:
        return None
    if body.work.parent_turn_id is not None:
        return None
    return {'envelope': body.work.model_dump(exclude_none=True), 'principal_id': authority.principal_id}
