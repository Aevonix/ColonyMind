"""Seeded conversation history, imported without model calls before an episode's first turn.

A generated scenario may carry ``history``: earlier owner sessions the agent is
supposed to remember. Inside the container the worker writes them into Hermes'
own ``state.db`` through the stock session import in every arm, so the base
arm's ``session_search`` and memory see them, and in plugin arms it additionally
retains them in the ColonyMind ledger through the reviewed history importer
(``colonymind.turns.hermes_history``), bound to the fixture owner: recorded
messages, with no claim and no model call. Nothing is fetched; the histories are
fixture bytes.

A plan may declare the replay mode instead (``paired plan --history-mode replay``):
a plugin arm then sends every (user, assistant) pair of every session through the
served host's own turn route (``replay_ledger``), on the history's own dates, so
claims, capture, appraisals, opinions and vectors form as they do from live turns.
The clock moves back once, to the first session, before the arm's state exists
(``rewind``); before each session the background work drains, the clock moves to
the session's end, and each pair is dated at its session's start plus its first
message's position, so no replayed time is ever later than the clock; after the
last session the worker lets the clock run again from the episode's pinned start.
"""
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import time

PROTOCOL = 'paired-history-1'
# The platform imported sessions are recorded on: a direct message on the capture
# platform, which the history importer accepts (it refuses automation sources).
SOURCE = 'capture'
CHAT_TYPE = 'dm'
ROLES = ('user', 'assistant')
MAX_SESSIONS = 128
MAX_MESSAGES = 1024
MAX_BYTES = 8 * 1024 * 1024
NAMESPACE = 'paired-history'
# How long an arm with an embedding endpoint waits for the source vector worker to embed the history.
VECTOR_DRAIN_S = 300
# The replay mode: the worker capability a plan with it requires, the channel the pairs arrive on (the owner's own:
# the benchmark credential grants the owner on the cli platform), and the deadline each replayed owner turn adds to
# an episode (my choice: an extraction, its review and the other background calls of one turn, with room).
REPLAY_PROTOCOL = 'paired-history-replay-1'
REPLAY_CHANNEL = 'cli:paired-history'
REPLAY_SECONDS_PER_TURN = 20
HISTORY_MODES = ('import', 'replay')
TURN_ROUTE = '/v1/host/turns/sync'
_LEAF = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}')


def validate_history(value):
    """``[{id, at, messages: [{role, content}]}]``; ids are leaf names, ``at`` is epoch seconds."""
    if not isinstance(value, list) or not value or len(value) > MAX_SESSIONS:
        raise ValueError(f'History is 1..{MAX_SESSIONS} sessions')
    if len(json.dumps(value, ensure_ascii=False).encode()) > MAX_BYTES:
        raise ValueError('History exceeds the size bound')
    identities = set()
    for session in value:
        if (not isinstance(session, dict) or set(session) != {'id', 'at', 'messages'}
                or not isinstance(session['id'], str) or not _LEAF.fullmatch(session['id'])
                or session['id'] in identities or type(session['at']) is not int or session['at'] < 0):
            raise ValueError('Invalid history session')
        identities.add(session['id'])
        messages = session['messages']
        if (not isinstance(messages, list) or not messages or len(messages) > MAX_MESSAGES
                or any(not isinstance(m, dict) or set(m) != {'role', 'content'} or m['role'] not in ROLES
                       or not isinstance(m['content'], str) or not m['content'].strip() for m in messages)):
            raise ValueError('Invalid history messages')
    return value


def hermes_sessions(sessions, *, user_id, chat_id):
    """The stock import payload: closed sessions whose messages keep their stated order and times."""
    payload = []
    for session in sessions:
        at = session['at']
        payload.append({'id': session['id'], 'source': SOURCE, 'user_id': user_id, 'title': session['id'],
                        'started_at': at, 'ended_at': at + len(session['messages']),
                        'end_reason': 'imported',
                        'messages': [{'role': m['role'], 'content': m['content'], 'timestamp': at + index}
                                     for index, m in enumerate(session['messages'])]})
    return payload


def mapping(sessions, *, contact_id, user_id, chat_id):
    """The reviewed binding the ledger importer requires, one per seeded session."""
    return {'version': 1, 'namespace': NAMESPACE, 'reviewed': True, 'bindings': [
        {'session_id': session['id'], 'platform': SOURCE, 'actor_id': user_id, 'chat_id': chat_id,
         'chat_type': CHAT_TYPE, 'contact_id': contact_id,
         'review_evidence': {'kind': 'fixture', 'reference': PROTOCOL}} for session in sessions]}


def seed_hermes(db_path, sessions, *, session_db, user_id, chat_id):
    """Import the sessions into ``state.db`` and stamp the transport columns the importer binds on."""
    db = session_db(Path(db_path))
    try:
        outcome = db.import_sessions(hermes_sessions(sessions, user_id=user_id, chat_id=chat_id))
    finally:
        db.close()
    if not outcome.get('ok') or outcome.get('imported') != len(sessions):
        raise RuntimeError('History import into state.db was incomplete: ' + json.dumps(outcome)[:512])
    with sqlite3.connect(db_path) as conn:
        for session in sessions:
            conn.execute('UPDATE sessions SET chat_id=?, chat_type=? WHERE id=?',
                         (chat_id, CHAT_TYPE, session['id']))
    return {'imported': outcome['imported']}


def snapshot(db_path, target):
    """A consistent copy with no WAL, the only input the ledger importer accepts."""
    with sqlite3.connect(db_path) as source, sqlite3.connect(target) as copy:
        source.backup(copy)
    return Path(target)


def seed_ledger(db_path, sessions, *, state_dir, contact_id, user_id, chat_id):
    """Retain the seeded sessions in the ColonyMind ledger as owner-scoped sources."""
    from colonymind.turns.hermes_history import import_history
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    copy = snapshot(db_path, Path(state_dir) / 'history-snapshot.db')
    document = mapping(sessions, contact_id=contact_id, user_id=user_id, chat_id=chat_id)
    counts, rounds = {}, 0
    while True:
        outcome = import_history(copy, document, state_dir=state_dir, apply=True, limit=10000)
        counts, rounds = outcome['counts'], rounds + 1
        if outcome['complete'] or rounds > 64:
            break
    if not outcome['complete']:
        raise RuntimeError('History import into the ledger did not complete')
    return {'counts': counts, 'models_called': outcome['models_called']}


def drain_vectors(ledger_path, *, timeout=VECTOR_DRAIN_S, poll=0.25):
    """Wait, at most ``timeout`` seconds, until the source vector worker has embedded every queued turn;
    returns what was queued, what was embedded and what was left."""
    def waiting():
        try:
            with closing(sqlite3.connect(Path(ledger_path).resolve().as_uri() + '?mode=ro', uri=True)) as conn:
                return int(conn.execute("SELECT count(*) FROM source_vector_jobs "
                                        "WHERE status IN ('pending', 'running')").fetchone()[0])
        except sqlite3.Error:
            return 0
    started = time.monotonic()
    jobs = left = waiting()
    while left and time.monotonic() - started < timeout:
        time.sleep(poll)
        left = waiting()
    return {'jobs': jobs, 'drained': jobs - left, 'left': left, 'seconds': round(time.monotonic() - started, 1)}


def user_turns(sessions):
    """How many owner turns a history holds: what a replay sends, and the deadline it adds."""
    return sum(message['role'] == 'user' for session in sessions or [] for message in session['messages'])


def schedule(sessions):
    """``[(session, start, end)]`` in the order the sessions were held: a session starts at its own time, or where
    the one before it ended when that is later (two sessions stamped the same second), and ends one second per
    message later, as the stock import stamps its messages."""
    rows, end = [], None
    for session in sorted(sessions, key=lambda item: item['at']):
        start = session['at'] if end is None else max(session['at'], end)
        end = start + len(session['messages'])
        rows.append((session, start, end))
    return rows


def pairs(messages):
    """``[(position, user, reply)]``: each owner message with the assistant's reply after it, if any; an assistant
    message with no owner message before it stands alone."""
    result, index = [], 0
    while index < len(messages):
        message = messages[index]
        if message['role'] == 'user':
            reply = messages[index + 1] if index + 1 < len(messages) and messages[index + 1]['role'] == 'assistant' \
                else None
            result.append((index, message, reply))
            index += 2 if reply is not None else 1
        else:
            result.append((index, None, message))
            index += 1
    return result


def _iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def rewind(sessions, state):
    """Move the clock back to the history's first moment, before the arm's state (``state``) exists; returns it."""
    from . import paired_body
    return paired_body.set_clock(schedule(sessions)[0][1], state=state)


@dataclass(frozen=True)
class Replay:
    """How a plugin arm replays its history: ``post(path, body)`` to its own sidecar with its one key, and
    ``drain()``, the bounded wait for its background work."""
    post: object
    drain: object


def replay_ledger(post, sessions, *, contact_id, state, drain):
    """Send every pair of every session through the served host's turn route on the history's own dates (the
    module docstring), as the owner. A turn the host refuses is recorded with its status, and the replay goes on."""
    from . import paired_body
    record = {'protocol': REPLAY_PROTOCOL, 'mode': 'replay', 'sessions': [], 'turns': 0, 'refused': [],
              'drains': []}
    for session, start, end in schedule(sessions):
        record['drains'].append(drain())
        paired_body.set_clock(end, state=state)
        sent = 0
        for position, (index, user, reply) in enumerate(pairs(session['messages'])):
            turn_id = f"history:{session['id']}:{position}"
            body = {'identity': {'host_id': 'paired-history'}, 'context': {
                'session_id': session['id'], 'contact_id': contact_id, 'turn_id': turn_id,
                'channel_id': REPLAY_CHANNEL, 'timezone': 'UTC', 'metadata': {'occurred_at': _iso(start + index)}}}
            if user is not None:
                body['user_message'] = {'role': 'user', 'content': user['content']}
            if reply is not None:
                body['assistant_message'] = {'role': 'assistant', 'content': reply['content']}
            try:
                result = post(TURN_ROUTE, body)
            except Exception as error:
                status = getattr(getattr(error, 'response', None), 'status_code', None)
                record['refused'].append({'turn_id': turn_id, 'status': status})
                continue
            if not isinstance(result, dict) or not result.get('accepted'):
                record['refused'].append({'turn_id': turn_id, 'status': 200, 'skipped_reason':
                                          result.get('skipped_reason') if isinstance(result, dict) else None})
                continue
            sent += 1
        record['sessions'].append({'session': session['id'], 'start': _iso(start), 'end': _iso(end), 'turns': sent})
        record['turns'] += sent
    record['drains'].append(drain())
    return record


def seed(home, sessions, *, session_db, contact_id, ledger, vectors=False, replay=None):
    """Seed every arm's ``state.db``; plugin arms (``ledger``) also retain the sources in the ledger, and
    with an embedding endpoint (``vectors``) wait, bounded, until the history is embedded. With ``replay`` (a
    ``Replay``: the replay mode) a plugin arm replays the history through its own sidecar instead of importing it;
    the replay's drains wait for the embedding too."""
    sessions = validate_history(sessions)
    home = Path(home)
    user_id, chat_id = contact_id, contact_id
    record = {'protocol': PROTOCOL, 'sessions': len(sessions),
              'messages': sum(len(session['messages']) for session in sessions),
              'hermes': seed_hermes(home / 'state.db', sessions, session_db=session_db,
                                    user_id=user_id, chat_id=chat_id)}
    if ledger and replay is not None:
        record['ledger'] = replay_ledger(replay.post, sessions, contact_id=contact_id, state=home / 'memory-state',
                                         drain=replay.drain)
    elif ledger:
        record['ledger'] = seed_ledger(home / 'state.db', sessions, state_dir=home / 'memory-state',
                                       contact_id=contact_id, user_id=user_id, chat_id=chat_id)
        if vectors:
            record['vector_drain'] = drain_vectors(home / 'memory-state' / 'turn-idempotency.db')
    return record
