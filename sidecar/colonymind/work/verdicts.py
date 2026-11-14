"""Exact adjacent reply identities and a durable handoff from vetted lessons."""
from contextlib import closing
from datetime import datetime, timezone
import json

from colonymind.turns.idempotency import canonical_turn_digest, source_message_hash
from . import ledger as facts

VERDICTS_PER_NIGHT = 48


def _moment(value):
    try:
        result = datetime.fromisoformat(value)
        return result.replace(tzinfo=result.tzinfo or timezone.utc).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def reply_identities(sessions):
    """Keep full-source revisions and message hashes, never match reply prose."""
    result = {}
    for session in sessions:
        previous, assistants = None, 0
        for turn in session['turns']:
            for index, message in enumerate(turn.get('raw_messages', [])):
                role = message.get('role')
                identity = {'source_id': turn['turn_id'], 'source_version': turn['source_version'],
                            'message_hash': source_message_hash(session['session_id'], message)}
                if role == 'user':
                    if previous and previous['role'] == 'assistant' and assistants == 1:
                        same_source = previous['identity']['source_id'] == turn['turn_id']
                        before, after = _moment(previous['at']), _moment(turn['at'])
                        if before and after and (same_source or before < after):
                            result[(turn['turn_id'], index)] = {
                                'reply': previous['identity'], 'verdict': identity,
                                'session_id': session['session_id'], 'at': after.isoformat(timespec='microseconds'),
                                'position': index}
                    assistants = 0
                elif role == 'assistant':
                    assistants += 1
                previous = {'role': role, 'identity': identity, 'at': turn['at']}
    return result


def _resolve(worker, source, identity):
    """Forward-check the exact reply's single direct input against an owner root."""
    from colonymind.beliefs.source_projection import SourceClaimProjection
    reply, verdict = identity['reply'], identity['verdict']
    refs = [{key: item[key] for key in ('source_id', 'source_version')} for item in (reply, verdict)]
    if not facts.current_refs(source, refs, worker.owner_id) or not adjacent(source, identity, worker.owner_id):
        return None
    row = source.execute('SELECT * FROM turn_sources WHERE turn_id=?', (reply['source_id'],)).fetchone()
    if row['session_id'] != identity['session_id']:
        return None
    messages = json.loads(row['messages_json'])
    matches = [m for m in messages if m.get('role') == 'assistant'
               and source_message_hash(row['session_id'], m) == reply['message_hash']]
    if len(matches) != 1:
        return None
    direct = SourceClaimProjection(worker.ledger)._direct_input(
        source, row, matches[0], contact_id=worker.owner_id, session_id=identity['session_id'])
    if not direct:
        return None
    parent, request = direct
    if parent['session_id'] != identity['session_id']:
        return None
    owner_hash = source_message_hash(parent['session_id'], request)
    root = source.execute('''SELECT e.*,d.generation,d.pending AS projection_pending FROM work_envelopes e
        JOIN work_dirty d USING(root_turn_id) WHERE e.source_id=? AND e.owner_hash=?
        AND e.contact_id=? AND e.kind='root' ''', (parent['turn_id'], owner_hash, worker.owner_id)).fetchone()
    if root is None or not facts.root_live(source, root):
        return None
    refs.append({'source_id': parent['turn_id'],
                 'source_version': canonical_turn_digest(json.loads(parent['messages_json']))})
    refs = list({ref['source_id']: ref for ref in refs}.values())
    return root, refs


def adjacent(conn, identity, contact_id):
    """Recheck chronology after model calls, pending retries and delayed admission."""
    pair = []
    for key, role in (('reply', 'assistant'), ('verdict', 'user')):
        ref = identity[key]
        row = conn.execute('''SELECT *,coalesce(occurred_at,ingested_at) AS at FROM turn_sources
            WHERE turn_id=? AND contact_id=? AND session_id=? AND scope='person' ''',
            (ref['source_id'], contact_id, identity['session_id'])).fetchone()
        if row is None:
            return False
        messages = json.loads(row['messages_json'])
        indices = [i for i, m in enumerate(messages) if m.get('role') == role
                   and source_message_hash(row['session_id'], m) == ref['message_hash']]
        if len(indices) != 1:
            return False
        pair.append((row, indices[0], len(messages)))
    (reply, ri, size), (verdict, vi, _) = pair
    if reply['turn_id'] == verdict['turn_id']:
        return vi == ri + 1
    before, after = _moment(reply['at']), _moment(verdict['at'])
    if ri != size - 1 or vi != 0 or not before or not after or before >= after:
        return False
    # Include ties at both ends: rowid cannot resolve simultaneous source order.
    return conn.execute('''SELECT 1 FROM turn_sources WHERE contact_id=? AND session_id=? AND scope='person'
        AND turn_id NOT IN (?,?) AND julianday(coalesce(occurred_at,ingested_at))>=julianday(?)
        AND julianday(coalesce(occurred_at,ingested_at))<=julianday(?) LIMIT 1''',
        (contact_id, identity['session_id'], reply['turn_id'], verdict['turn_id'], reply['at'], verdict['at'])).fetchone() is None


def capture(worker, packet, judged, night, now):
    """Commit bounded vetted verdicts before Lessons advances its scan cursor."""
    selected = list(judged.items())[:VERDICTS_PER_NIGHT]
    repairs = set()
    with closing(worker.ledger._connect()) as source, closing(worker.store._connect()) as conn:
        for label, _ in selected:
            identity = packet['owner'][label].get('work_identity')
            resolved = (_resolve(worker, source, identity) if identity
                        and label not in packet.get('ambiguous_verdicts', []) else None)
            if resolved:
                root, _ = resolved
                current = conn.execute('SELECT generation FROM work_roots WHERE id=? AND contact_id=?',
                                       (root['root_turn_id'], worker.owner_id)).fetchone()
                if current is None or current['generation'] != root['generation']:
                    repairs.add(root['root_turn_id'])
    # Only bounded candidate roots need repair, outside the canonical write lock.
    for ident in repairs:
        worker.store.rebuild_root(ident)
    accepted, unlinked = 0, 0
    with closing(worker.ledger._connect()) as source, source:
        source.execute('BEGIN IMMEDIATE')
        with closing(worker.store._connect()) as conn, conn:
            for label, value in selected:
                owner = packet['owner'][label]
                identity = owner.get('work_identity')
                resolved = (_resolve(worker, source, identity) if identity
                            and label not in packet.get('ambiguous_verdicts', []) else None)
                if not resolved:
                    unlinked += 1
                    continue
                root, refs = resolved
                current = conn.execute('SELECT generation,revision FROM work_roots WHERE id=? AND contact_id=?',
                                       (root['root_turn_id'], worker.owner_id)).fetchone()
                if current is None and not root['projection_pending']:
                    # A repaired root without write effects has no work projection.
                    unlinked += 1
                    continue
                if current is None or current['generation'] != root['generation']:
                    # Roll back the handoff so Lessons retains its cursor for retry.
                    raise RuntimeError('work projection changed during verdict capture')
                ident = canonical_turn_digest(identity)
                source.execute('''INSERT OR IGNORE INTO work_verdict_pending
                    (id,root_id,contact_id,generation,verdict,verdict_at,source_refs_json,identity_json,captured_revision)
                    VALUES (?,?,?,?,?,?,?,?,?)''', (ident, root['root_turn_id'], worker.owner_id,
                        root['generation'], 'confirmed' if value == 'right' else 'corrected',
                        identity['at'] + f":{identity['position']:08d}", json.dumps(refs), json.dumps(identity),
                        current['revision']))
                accepted += 1
            coverage = {**packet['work_coverage'], 'vetted': len(judged), 'queued': accepted,
                        'unlinked': unlinked, 'verdicts_omitted': max(0, len(judged) - VERDICTS_PER_NIGHT)}
            conn.execute('INSERT OR REPLACE INTO work_verdict_coverage VALUES (?,?,?)',
                         (worker.owner_id, now.isoformat(), json.dumps(coverage)))
    night.count('work_verdicts_queued', accepted)
    night.count('work_verdicts_unlinked', unlinked)
    for key, value in coverage.items():
        night.count('work_verdict_' + key, value)


def apply_pending(worker, night, now):
    """Accept canonical decisions before writing their repairable projection."""
    from .night import retry_at
    with closing(worker.ledger._connect()) as conn, conn:
        rows = conn.execute('''SELECT * FROM work_verdict_pending
            WHERE contact_id=? AND state IN ('pending','accepted') AND next_attempt<=?
            ORDER BY verdict_at,id LIMIT ?''', (worker.owner_id, now.timestamp(), VERDICTS_PER_NIGHT)).fetchall()
        for row in rows:
            conn.execute('UPDATE work_verdict_pending SET attempts=attempts+1,next_attempt=? WHERE id=?',
                         (retry_at(now, row['attempts'] + 1), row['id']))
    for row in rows:
        refs = json.loads(row['source_refs_json'])
        identity = json.loads(row['identity_json'])
        state, root = row['state'], None
        with closing(worker.ledger._connect()) as source, source:
            source.execute('BEGIN IMMEDIATE')
            generation = source.execute('SELECT generation FROM work_dirty WHERE root_turn_id=?',
                                        (row['root_id'],)).fetchone()
            expected_revision = row['applied_revision'] if state == 'accepted' else row['captured_revision']
            stale = (not generation or (not expected_revision and generation[0] != row['generation'])
                     or not facts.current_refs(source, refs, worker.owner_id)
                     or not adjacent(source, identity, worker.owner_id))
            if not stale:
                with closing(worker.store._connect()) as projection:
                    current = projection.execute('SELECT * FROM work_roots WHERE id=? AND contact_id=?',
                                                 (row['root_id'], worker.owner_id)).fetchone()
                if current and current['generation'] == generation[0]:
                    stale = bool(expected_revision and current['revision'] != expected_revision)
                    if not stale:
                        root = dict(current)
                        source.execute('''UPDATE work_verdict_pending SET state='accepted',applied_revision=?
                            WHERE id=?''', (root['revision'], row['id']))
            if stale:
                source.execute("UPDATE work_verdict_pending SET state='stale' WHERE id=?", (row['id'],))
        # The decision and revision are now canonical. A crash or projection error
        # here leaves accepted evidence that retries and root repair both replay.
        if root and worker.store.write_derived(row['root_id'], contact_id=worker.owner_id,
                expected_revision=root['revision'], expected_generation=root['generation'],
                overlay=row['verdict'], source_refs=refs, verdict_order=(row['verdict_at'], row['id']),
                verdict_identity=identity):
            with closing(worker.ledger._connect()) as conn, conn:
                conn.execute("UPDATE work_verdict_pending SET state='linked' WHERE id=?", (row['id'],))
            night.count('work_verdicts_linked')
