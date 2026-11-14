"""Scoped assertion projections owned by the canonical source transaction."""
from __future__ import annotations

import asyncio
from contextlib import closing
import hashlib
import json
import logging
import re
import sqlite3
import time
import uuid

from colonymind.usage import function_usage

from .source_claims import (EXTRACTION_VERSION, SourceClaimOutputError, admission_metadata, durable_message_basis,
                            extract_claims, extraction_diagnostics, extraction_request, literal_subject,
                            projection_timeout_seconds, norm_value, quoted_applicability, words_at)
from .source_time import MemoryTimeQuery, event_interval, filter_unstructured, utc_timestamp

logger = logging.getLogger(__name__)

# Includes older procedures whose exact quotation predates the explicit
# representation field. These records do not certify conditional applicability.
_QUOTED_SQL = """(json_extract(c.data_json,'$.representation') IN ('preference','procedure')
    OR json_extract(c.data_json,'$.memory_quality.memory_kind')='procedure')"""
# How many of a person's recent distinct claim keys the extraction prompt lists (a builder's choice, not a
# measured bound): enough for one person's active properties, few enough to keep the prompt short.
KNOWN_KEYS = 20
# How many turns the vector lane of the prior claims reads: as many as the keyword lane's search returns.
SIMILAR_TURNS = 10
# The conversation's previous user messages the extraction call is offered, where a subject its quotation names only
# by a pronoun may have been named (``SourceClaimProjection.earlier_messages``). All three are a builder's choices,
# not measured bounds: the last few messages a reference reaches back over, said within the last few hours of the
# same conversation (a long-lived session is not one conversation), each offered whole up to a thousand characters
# and a longer one by its end, the part nearest what follows.
EARLIER_MESSAGES = 4
EARLIER_WINDOW_SECONDS = 6 * 3600
EARLIER_CHARS = 1000
# The claims whose subject an earlier message named, by that message's source (erasure reads them per source).
_BASIS_TURN_SQL = "json_extract(data_json,'$.subject_basis_message.turn_id')"
# A claim whose event time is a resolved interval, whatever its precision (never NULL).
_EVENT_INTERVAL_SQL = """(coalesce(json_extract(c.data_json,'$.event_time.status'),'')='resolved'
    AND json_extract(c.data_json,'$.event_time.start') IS NOT NULL
    AND json_extract(c.data_json,'$.event_time.end_exclusive') IS NOT NULL)"""
# When a source was said: its occurrence time, else when it was recorded, as a Julian day (SQLite reads the zone
# offset of each, so two sources compare as instants). Re-extraction (``beliefs.reextract``) orders by it, then by
# ``ingested_at`` and ``turn_id``. ``{s}`` is the alias of ``turn_sources``.
OCCURRENCE_SQL = "coalesce(julianday({s}.occurred_at),julianday({s}.ingested_at))"


def owner_signal_writer(mind_provider):
    """The appraisal's owner net for the running mind (owner outreach, architecture 4.10): the owner's
    opt-out it saw pauses outreach when the phrase match missed it. No mind, or one without the hook: nothing."""
    def write(signal):
        mind = mind_provider()
        hook = getattr(mind, "owner_signal", None)
        return hook(signal) if callable(hook) else None
    return write


def initialize(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS source_claim_jobs (
        turn_id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'pending',
        timezone TEXT NOT NULL DEFAULT 'UTC', attempts INTEGER NOT NULL DEFAULT 0,
        next_attempt REAL NOT NULL DEFAULT 0, lease_until REAL NOT NULL DEFAULT 0,
        error TEXT, model TEXT, extraction_version TEXT, lease_token TEXT NOT NULL DEFAULT '',
        diagnostics_json TEXT, enqueued_at REAL NOT NULL DEFAULT 0, next_message INTEGER NOT NULL DEFAULT 0)''')
    columns = {row[1] for row in conn.execute('PRAGMA table_info(source_claim_jobs)')}
    if 'diagnostics_json' not in columns:
        conn.execute('ALTER TABLE source_claim_jobs ADD COLUMN diagnostics_json TEXT')
    if 'next_message' not in columns:
        conn.execute('ALTER TABLE source_claim_jobs ADD COLUMN next_message INTEGER NOT NULL DEFAULT 0')
    if 'enqueued_at' not in columns:
        # A queue from before the column: a job was enqueued when its source was recorded.
        from colonymind.turns.projection_backlog import EPOCH_SQL
        conn.execute('ALTER TABLE source_claim_jobs ADD COLUMN enqueued_at REAL NOT NULL DEFAULT 0')
        conn.execute('UPDATE source_claim_jobs SET enqueued_at=coalesce((SELECT ' + EPOCH_SQL.format('s.ingested_at')
                     + ' FROM turn_sources s WHERE s.turn_id=source_claim_jobs.turn_id),0)')
    conn.execute('''CREATE TABLE IF NOT EXISTS source_claims (
        id TEXT PRIMARY KEY, turn_id TEXT NOT NULL, message_hash TEXT NOT NULL,
        subject_key TEXT NOT NULL, predicate TEXT NOT NULL, value_key TEXT NOT NULL, data_json TEXT NOT NULL,
        valid_from TEXT, valid_to TEXT, superseded_by TEXT, retracted_by TEXT)''')
    conn.execute('CREATE INDEX IF NOT EXISTS source_claim_turn ON source_claims(turn_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS source_claim_key ON source_claims(subject_key,predicate)')
    from colonymind.contacts import refresh
    refresh.initialize(conn)
    conn.execute('CREATE INDEX IF NOT EXISTS source_claim_message ON source_claims(message_hash)')
    conn.execute('CREATE INDEX IF NOT EXISTS source_claim_subject_basis_message ON source_claims('
                 + _BASIS_TURN_SQL + ')')
    conn.execute('CREATE INDEX IF NOT EXISTS source_contact_scope ON turn_sources(contact_id,scope,session_id)')
    # The learning layer (memory/learning.py): a restated claim is a witness edge on the live claim, not a
    # new row. The claim row is anchored at its newest witness; ``witnesses`` counts them all.
    if 'witnesses' not in {row[1] for row in conn.execute('PRAGMA table_info(source_claims)')}:
        conn.execute('ALTER TABLE source_claims ADD COLUMN witnesses INTEGER NOT NULL DEFAULT 1')
    conn.execute('''CREATE TABLE IF NOT EXISTS source_claim_witnesses (
        claim_id TEXT NOT NULL, turn_id TEXT NOT NULL, message_hash TEXT NOT NULL, data_json TEXT NOT NULL,
        recorded_at TEXT, PRIMARY KEY(claim_id,turn_id,message_hash))''')
    conn.execute('CREATE INDEX IF NOT EXISTS source_claim_witness_turn ON source_claim_witnesses(turn_id)')
    from colonymind.memory import learning
    learning.initialize(conn)
    from . import reextract
    reextract.initialize(conn)


def enqueue(conn, turn_id, messages, *, scope, timezone_name=None):
    # Checkpoint history has no per-message verified speaker attribution/time.
    if scope == "person" and any(m.get("role") == "user" for m in messages):
        from colonymind.turns.observed import is_observed_source
        from colonymind.turns import projection_backlog
        enqueued = projection_backlog.watermark(conn) - 1 if is_observed_source(turn_id) else time.time()
        conn.execute('INSERT OR IGNORE INTO source_claim_jobs(turn_id,timezone,enqueued_at) VALUES (?,?,?)',
                     (turn_id, timezone_name or "UTC", enqueued))


def erase_removed(conn, turn_id, session_id, retained):
    from colonymind.turns.idempotency import source_message_hash
    from .value_revision import erase_dependents
    hashes = {source_message_hash(session_id, message) for message in retained}
    # A witness edge goes with its message; the claim keeps its other witnesses.
    for row in conn.execute('SELECT claim_id,message_hash FROM source_claim_witnesses WHERE turn_id=?',
                            (turn_id,)).fetchall():
        if row['message_hash'] not in hashes:
            drop_witness(conn, row['claim_id'], turn_id, row['message_hash'])
    rows = conn.execute('SELECT id,message_hash FROM source_claims WHERE turn_id=?', (turn_id,)).fetchall()
    # A claim whose subject an erased message named goes with that message, wherever the claim was said.
    named = [row[0] for row in conn.execute(
        'SELECT id,json_extract(data_json,\'$.subject_basis_message.message_hash\') FROM source_claims WHERE '
        + _BASIS_TURN_SQL + '=?', (turn_id,)).fetchall() if row[1] not in hashes]
    # The claims re-extraction retired from the erased messages go with them, with every retired or live claim that
    # carries their subject or words (read before the live claims below leave), and so does its queued job.
    from .reextract import erase_removed as erase_retired
    erase_retired(conn, turn_id, hashes, whole=not retained,
                  roots=[row['id'] for row in rows if row['message_hash'] not in hashes] + named)
    for identifier in named:
        erase_dependents(conn, identifier)
    for row in rows:
        if row["message_hash"] not in hashes:
            # Remove projections depending on the erased quote, including
            # carried values. The owner's other raw corrections survive.
            if promote_witness(conn, row['id']):
                # Another message of the same person still states this value: the claim stays, anchored
                # there, and nothing quoting the erased message survives (its dependents carried it).
                erase_dependents(conn, row['id'], keep_root=True)
                continue
            erase_dependents(conn, row['id'])
    if not retained:
        conn.execute('DELETE FROM source_claim_jobs WHERE turn_id=?', (turn_id,))
    else:
        # Exact-message erasure shifts window positions. Restart unfinished work and revoke its old lease.
        conn.execute("UPDATE source_claim_jobs SET next_message=0,status='pending',lease_until=0,lease_token='' "
                     "WHERE turn_id=? AND status IN ('pending','running') "
                     "AND turn_id LIKE 'observed-conversation:%'", (turn_id,))
    # Supersession/retraction links on surviving claims retain IDs, not erased
    # values. Deleting a correction must never silently revive its old value.


def has_witnesses(conn):
    return conn.execute('SELECT 1 FROM source_claim_witnesses LIMIT 1').fetchone() is not None


def drop_witness(conn, claim_id, turn_id, message_hash):
    if conn.execute('DELETE FROM source_claim_witnesses WHERE claim_id=? AND turn_id=? AND message_hash=?',
                    (claim_id, turn_id, message_hash)).rowcount:
        conn.execute('UPDATE source_claims SET witnesses=max(1,witnesses-1) WHERE id=?', (claim_id,))


def promote_witness(conn, claim_id):
    """The anchor's message is erased or no longer attributable: the newest surviving witness still in the
    same person's scope becomes the anchor, and the old anchor is dropped. False when none is left."""
    from colonymind.turns.idempotency import source_message_hash
    anchor = conn.execute('SELECT c.turn_id,s.contact_id FROM source_claims c LEFT JOIN turn_sources s '
                          'ON s.turn_id=c.turn_id WHERE c.id=?', (claim_id,)).fetchone()
    if anchor is None:
        return False
    for witness in conn.execute("""SELECT w.*,s.session_id,s.messages_json,s.contact_id FROM source_claim_witnesses w
            JOIN turn_sources s ON s.turn_id=w.turn_id WHERE w.claim_id=? AND s.scope='person'
            AND NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=s.turn_id)
            ORDER BY w.recorded_at DESC, w.turn_id DESC""", (claim_id,)).fetchall():
        if anchor['contact_id'] is not None and witness['contact_id'] != anchor['contact_id']:
            continue
        if witness['message_hash'] not in {source_message_hash(witness['session_id'], message)
                                           for message in json.loads(witness['messages_json'])}:
            continue
        conn.execute('UPDATE source_claims SET turn_id=?,message_hash=?,data_json=?,witnesses=max(1,witnesses-1) '
                     'WHERE id=?', (witness['turn_id'], witness['message_hash'], witness['data_json'], claim_id))
        conn.execute('DELETE FROM source_claim_witnesses WHERE claim_id=? AND turn_id=? AND message_hash=?',
                     (claim_id, witness['turn_id'], witness['message_hash']))
        return True
    return False


def reanchor_invalidated(conn, turn_id):
    """A source whose attribution was invalidated hides its claims; one that another witness still states is
    re-anchored there, as the separate rows of an unfolded store would still show it."""
    for row in conn.execute('SELECT id FROM source_claims WHERE turn_id=?', (turn_id,)).fetchall():
        promote_witness(conn, row['id'])
    from colonymind.turns import claim_keys
    claim_keys.index_pending(conn)


def _representation(data):
    if data.get('representation'):
        return data['representation']
    return 'procedure' if (data.get('memory_quality') or {}).get('memory_kind') == 'procedure' else 'assertion'


# Records another faculty reads per message (the opinion pass's premises: judgments._admitted) keep one row
# per message; it deduplicates their content itself.
PREMISE_KINDS = frozenset({'decision', 'procedure', 'substantive_event'})


def fold_key(contact_id, subject_key, predicate, data, valid_from, valid_to, *, whole, premises=False):
    """A claim's identity for the witness fold, or None when it never folds (episodes, partial revisions,
    audio-derived claims, premise records, claims whose subject an earlier message named). A literal value compares
    as the read-time dedupe does (``norm_value``, the same period and event time). A quoted preference folds only on
    a verbatim repeat of the whole statement: the same characters, the entire message. ``premises``: also the
    premise records and the claims an earlier message named, for retiring a corrected value's duplicate rows (they
    never fold)."""
    from colonymind.memory.learning import digest, verbatim
    representation = _representation(data)
    kind = (data.get('memory_quality') or {}).get('memory_kind')
    if (representation == 'episode' or str(subject_key).startswith('episode:') or data.get('subject_basis_claim_id')
            or data.get('value_parts') or data.get('evidence_basis')
            or ((kind in PREMISE_KINDS or data.get('subject_basis_message')) and not premises)):
        return None
    quoted = representation in {'preference', 'procedure'}
    if quoted and not whole:
        return None
    value = verbatim(data.get('value')) if quoted else norm_value(data.get('value'))
    # Time as stated, not only as resolved: "9:30am Monday" and "9:30am Tuesday" left unresolved are two claims.
    time_basis = json.dumps([data.get('event_at'), data.get('event_time'), data.get('validity_basis'),
                             *(data.get(key) for key in ('valid_from_text', 'valid_to_text', 'event_at_text'))],
                            sort_keys=True, default=str)
    return digest(contact_id, subject_key, predicate, representation, kind, value, valid_from, valid_to, time_basis,
                  json.dumps(quoted_applicability(data), sort_keys=True))


def _subject_basis_source_sql(identifier_sql, contact_sql):
    """Shared root lifecycle eligibility; full grounding stays in subject_basis.

    Both arguments are internal SQL fragments, never user-supplied values.
    """
    return f'''FROM source_claims b
        JOIN turn_sources bs ON bs.turn_id=b.turn_id JOIN source_claim_jobs bj ON bj.turn_id=b.turn_id
        WHERE b.id={identifier_sql} AND bs.contact_id={contact_sql} AND bs.scope='person' AND bj.status='complete'
        AND NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=bs.turn_id)
        AND NOT EXISTS (SELECT 1 FROM source_projection_erasures e WHERE e.turn_id=bs.turn_id)
        AND NOT EXISTS (SELECT 1 FROM source_annotations a,json_each(a.target_message_hashes_json) h
                        WHERE a.target_source_id=bs.turn_id AND h.value=b.message_hash)'''


def _message_basis_source_sql(turn_sql, contact_sql, hash_sql):
    """Whether an earlier message still names a subject: its source is the same person's, in person scope, its
    attribution and projections stand, and the message is not annotated. All arguments are internal SQL."""
    return f'''FROM turn_sources bs WHERE bs.turn_id={turn_sql} AND bs.contact_id={contact_sql} AND bs.scope='person'
        AND NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=bs.turn_id)
        AND NOT EXISTS (SELECT 1 FROM source_projection_erasures e WHERE e.turn_id=bs.turn_id)
        AND NOT EXISTS (SELECT 1 FROM source_annotations a,json_each(a.target_message_hashes_json) h
                        WHERE a.target_source_id=bs.turn_id AND h.value={hash_sql})'''


def _contribution_basis_sql(data_sql, contact_sql):
    """Eligibility of a claim carrying words of claims whose subject was named elsewhere (``value_parts``): the
    earlier message that named it still names it (``_message_basis_source_sql``), and the prior claim it came from
    still stands (``_subject_basis_source_sql``). Both arguments are internal SQL."""
    named = "json_extract(pc.data_json,'$.subject_basis_message.{}')".format
    inherited = "json_extract(pc.data_json,'$.subject_basis_claim_id')"
    return (f"NOT EXISTS (SELECT 1 FROM json_each({data_sql},'$.value_parts') part JOIN source_claims pc "
            "ON pc.id=json_extract(part.value,'$.source_claim_id') WHERE (" + named('turn_id') + " IS NOT NULL "
            "AND NOT EXISTS (SELECT 1 " + _message_basis_source_sql(named('turn_id'), contact_sql,
                                                                   named('message_hash')) + ")) OR ("
            + inherited + " IS NOT NULL AND NOT EXISTS (SELECT 1 " + _subject_basis_source_sql(inherited, contact_sql)
            + ')))')


def contribution_subject_bases(conn, claim, *, contact_id):
    """Where the subject of the claims whose words ``claim`` carries (``value_parts``) was named, by contributing
    claim: ``{claim_id: basis}``, the earlier message (``message_subject_basis``) or the prior claim
    (``subject_basis``) a contributing claim took its subject from, or None when one no longer grounds it.

    A partial correction that names its subject itself ("Correction: My sister bought a blue one.") still carries
    the corrected claim's words ("Ford"), and those words are about the subject that claim's earlier message (or
    prior claim) named: without it they are no one's, so the correction goes where the corrected claim goes, and
    cites that source beside the words it carries (the contribution's ``subject_basis``)."""
    found, seen = {}, set()
    for part in claim.get('value_parts') or ():
        identifier = part.get('source_claim_id') if isinstance(part, dict) else None
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)
        row = conn.execute('SELECT data_json FROM source_claims WHERE id=?', (identifier,)).fetchone()
        data = json.loads(row[0]) if row is not None else {}
        for field, reader in (('subject_basis_message', message_subject_basis), ('subject_basis_claim_id',
                                                                                  subject_basis)):
            if data.get(field):
                basis = reader(conn, data, contact_id=contact_id)
                if basis is None:
                    return None
                found[identifier] = basis
    return found


def _value_sources(claim):
    """The sources a claim's carried words cite: each contribution's quotation (``value_basis``) and the message that
    named the contribution's subject, when another did (an earlier message, or a prior claim's)."""
    parts = claim.get('value_basis', [])
    return [*parts, *(part['subject_basis'] for part in parts if part.get('subject_basis'))]


def message_subject_basis(conn, claim, *, contact_id):
    """The earlier message that named a claim's subject (``subject_basis_message``), as recall and the other readers
    show it, or None when it no longer can: the message was erased, reattributed or annotated, its words are gone,
    or they do not name the claim's subject. Like a prior claim's basis (``subject_basis``) it grounds who or what
    the subject is, never the value.

    The span is where the words are in the message as it is now. A message's words change under its identity only
    when a secret in it is sealed or released (``colonymind.secrets.migrate``: the message is rewritten in place and
    its hash re-pointed everywhere, this claim's too): the value and its reference differ in length, so words after
    it move. They are found again as the same occurrence of the same words (``words_at``), never as other words,
    for a live claim, a retired one and one re-extraction set aside alike."""
    basis = durable_message_basis(claim.get('subject_basis_message'))
    if basis is None:
        return None
    start, end, evidence, occurrence = (basis['span_start'], basis['span_end'], basis['evidence'],
                                        basis['occurrence'])
    if (type(start) is not int or type(end) is not int or not 0 <= start < end or not isinstance(evidence, str)
            or end - start != len(evidence) or type(occurrence) is not int or occurrence < 0
            or not isinstance(claim.get('subject'), str) or not literal_subject(claim['subject'], evidence)):
        return None
    from colonymind.turns.idempotency import source_message_hash, canonical_turn_digest
    row = conn.execute('SELECT bs.* ' + _message_basis_source_sql('?', '?', '?'),
                       (basis['turn_id'], contact_id, basis['message_hash'])).fetchone()
    if row is None:
        return None
    messages = json.loads(row['messages_json'])
    message = next((m for m in messages if source_message_hash(row['session_id'], m) == basis['message_hash']), None)
    if message is None or message.get('role') != 'user' or not isinstance(message.get('content'), str):
        return None
    start = words_at(message['content'], evidence, start, occurrence)
    if start is None:
        return None
    end = start + len(evidence)
    return {'turn_id': row['turn_id'], 'message_hash': basis['message_hash'],
            'source_version': canonical_turn_digest(messages), 'basis': 'earlier_message',
            'disposition': 'subject_identity_only', 'value_use': 'not_evidence_for_current_value',
            'role': 'user', 'reported_at': row['occurred_at'], 'subject': claim['subject'],
            'span_start': start, 'span_end': end, 'evidence': evidence}


def subject_basis(conn, claim, *, contact_id):
    """Return grounded identity and explicitly qualified earlier episode context.

    A root's value may be retracted or superseded while its literal subject
    remains grounded. Annotation, erasure or changed attribution revokes it.
    Only one fully grounded ancestor is allowed, not a recursive claim chain.
    An immediate episode correction can qualify the earlier report without
    withdrawing every unchanged detail. Further corrections require the existing
    history reader: the root alone does not represent intervening revisions.
    """
    identifier = claim.get('subject_basis_claim_id')
    if not identifier:
        return None
    from colonymind.turns.idempotency import source_message_hash, canonical_turn_digest
    from colonymind.turns.audio import claim_message
    row = conn.execute('SELECT b.*,bs.session_id,bs.messages_json,bs.occurred_at,bs.ingested_at '
        + _subject_basis_source_sql('?', '?'),
        (identifier, contact_id)).fetchone()
    if row is None:
        return None
    data = json.loads(row['data_json'])
    messages = json.loads(row['messages_json'])
    message = next((claim_message(m) for m in messages
        if source_message_hash(row['session_id'], m) == row['message_hash']), None)
    if (data.get('subject_basis_claim_id') or message is None or message.get('role') != 'user'
            or data.get('subject') != claim.get('subject') or row['subject_key'] != claim['subject_key']
            or row['predicate'] != claim['predicate'] or data['evidence'] not in message.get('content', '')
            or admission_metadata(data) is None
            or ('source_admission' in data and ('_audio_segments' in message
                or data['evidence'] != message.get('content')))):
        return None
    # Reuse the original literal-grounding rule, without inferring an alias.
    from .source_claims import literal_subject
    episode = data.get('representation') == claim.get('representation') == 'episode'
    if episode:
        if row['subject_key'] != 'episode:' + hashlib.sha256(data['evidence'].encode()).hexdigest():
            return None
    elif not literal_subject(data['subject'], data['evidence']):
        return None
    result = {'claim_id': row['id'], 'turn_id': row['turn_id'], 'message_hash': row['message_hash'],
            'source_version': canonical_turn_digest(messages),
            'disposition': 'subject_identity_only',
            'value_use': 'not_evidence_for_current_value',
            **{k: data[k] for k in ('evidence_basis', 'epistemic_state', 'source_modality') if k in data},
            'subject': data['subject'], 'predicate': row['predicate'], 'evidence': data['evidence']}
    if episode:
        immediate = claim.get('prior_claim_id') == identifier
        result.update(
            disposition='prior_episode_report' if immediate else 'episode_history_incomplete',
            value_use=(
                'Read with the current correction. Corrected or withdrawn details are not current. '
                'Unchanged details remain attributed earlier context, not newly verified facts. '
                'A withdrawal of the whole report withdraws all its details.' if immediate else
                'Intermediate corrections are absent here. Do not infer current details from this '
                'original report. Open assertion history; missing or withdrawn revisions cannot '
                'be reconstructed from the original.'),
            role='user', reported_at=row['occurred_at'], recorded_at=row['ingested_at'],
            event_at=data.get('event_at'), event_time=data.get('event_time', {
                'status': 'legacy_precision_unknown' if data.get('event_at') else 'unknown'}))
    return result


def _derived_modality(claims):
    """Where the derived claims among ``claims`` came from: a heard conversation or an audio clip's transcript."""
    return next((c.get('source_modality') for c in claims if c.get('evidence_basis') and c.get('source_modality')),
                'audio_transcript')


def witness_key(row):
    """What the read-time dedupe counts as one value: ``(subject_key, predicate, value)`` over one period (the
    same validity and event time). None for a quoted preference, a source statement that never folds.

    A resolved event time is its precision and both bounds, never its start alone: a day, a month and a year
    that begin at the same instant are three times, and each claim stays a record a correction can name. The
    same time stated in other words ("in 2015", "during 2015") is one."""
    if row.get('representation') == 'preference':
        return None
    time = row.get('event_time') if isinstance(row.get('event_time'), dict) else {}
    if time.get('status') == 'resolved' and time.get('precision'):
        event = (time['precision'], time.get('at') or time.get('start'), time.get('end_exclusive'))
    else:
        event = row.get('event_at') or time.get('start')
    return (row['subject_key'], row['predicate'], norm_value(row.get('value')), row.get('valid_from'),
            row.get('valid_to'), event)


def one_witness_per_value(rows):
    """Claim dedupe at read time: of the scalar claims repeating one value (``witness_key``), only the newest
    witness, the rule ``_rows(..., distinct_values=True)`` applies per key to the claims valid at one time.
    The same value over two periods is two claims (a correction may name either). A quoted preference is a
    source statement, never folded. The stored claims are never rewritten."""
    kept, seen = [], set()
    # A folded claim can arrive once per witness message it was found through: keep its newest view.
    rows = {row['id']: row for row in sorted(rows, key=lambda row: (row.get('recorded_at') or '', row['id']))}.values()
    for row in sorted(rows, key=lambda row: (row.get('recorded_at') or '', row['id']), reverse=True):
        value = witness_key(row)
        if value is not None:
            if value in seen:
                continue
            seen.add(value)
        kept.append(row)
    return kept


# What ``extract_claims`` returns in place of a model id when it made no extraction call.
_NOT_EXTRACTED = frozenset({"unsupported_message", "oversize_message", "local_extraction_role_unavailable"})


class _Metered:
    """One dry run's router: every call goes to ``router``, and the tokens each response reports are added up."""

    def __init__(self, router):
        self._router, self.calls, self.tokens = router, 0, {}

    def __getattr__(self, name):
        return getattr(self._router, name)

    async def complete(self, *args, **kwargs):
        try:
            response = await self._router.complete(*args, **kwargs)
        except Exception as error:
            # A call no attempt completed still used what its endpoints reported (the router's error carries it).
            self._add(function_usage(error))
            raise
        self.calls += 1
        self._add(function_usage(response))
        return response

    def _add(self, used):
        for key, count in used.items():
            self.tokens[key] = self.tokens.get(key, 0) + count

    def usage(self):
        return {"calls": self.calls, **self.tokens}


class _PlanOnly:
    """The vault as a dry run uses it: a detected value takes the reference ingestion would give it, planned
    under the existing key and store, and nothing is stored or created (no first key, no store). A value with
    nothing to plan under is withheld, never shown to a model."""

    def __init__(self, vault):
        self._vault = vault

    def __getattr__(self, name):
        return getattr(self._vault, name)

    def plan(self, value, *, contact_id):
        from colonymind.secrets.vault import VaultUnavailable
        if not self._vault.db_path.exists():
            raise VaultUnavailable("nothing is sealed yet, and a dry run makes no store")
        self._vault._load_key(create=False)
        return self._vault.plan(value, contact_id=contact_id)

    @property
    def available(self):
        from colonymind.secrets.vault import VaultUnavailable
        try:
            self._vault._load_key(create=False)
        except VaultUnavailable:
            return False
        return True

    def resolve(self, ref):
        from colonymind.secrets.vault import UnknownReference
        if not self._vault.db_path.exists():
            raise UnknownReference(str(ref))
        return self._vault.resolve(ref)

    def seal(self, *args, **kwargs):
        from colonymind.secrets.vault import VaultUnavailable
        raise VaultUnavailable("a dry run stores nothing")


def lead_pairs(rows, pairs):
    """Each reply that ``pairs`` links to its input, moved to where the better ranked of the two stands and given
    that one's rank, so the conversation pair made from them stands where the first of its messages was found.

    ``memory.recall.pair_conversation_candidates`` puts a pair where its reply stands and drops its input. A reply
    a question found alone brings (``_direct_reply``) comes after every hit, and a lane can find a reply far below
    its question: the pair would carry the question down with it, and past the reranker's bound
    (``memory.selection.spread_across_sessions``) it would never be read. Only the reply moves, into the input's
    place, and the input goes with the pairing, so every other candidate keeps its place. A reply that is not
    paired is not shown alone, wherever it stands."""
    def identity(row):
        return row.get('source_turn_id'), row.get('source_message_hash')
    first = {}
    for index, row in enumerate(rows):
        first.setdefault(identity(row), index)
    leading = {}
    for response_key, input_key in pairs.items():
        reply, request = first.get(response_key), first.get(input_key)
        if reply is not None and request is not None and request < reply:
            leading.setdefault(request, []).append(reply)
    if not leading:
        return rows
    moved = {reply for replies in leading.values() for reply in replies}
    placed = []
    for index, row in enumerate(rows):
        for reply in sorted(leading.get(index, ())):
            placed.append(dict(rows[reply], **{name: row[name] for name in ('relevance', 'retrieval_method')
                                                if name in row}))
        if index not in moved:
            placed.append(row)
    return placed


def collapse_repeated_pairs(rows, *, corrections=lambda row: frozenset()):
    """Keep the first of conversation pairs that repeat the same words by the same speakers (the same person's same
    question with the same answer, in two conversations), and a later one only when it brings a correction the ones
    kept before it do not. Selection collapses a person's repeated plain quotations the same way
    (``memory.selection.RecallSelector.select_context``); a question found alone now brings its answer, so the
    repeat arrives as a pair. A pair whose passages carry a validity label is kept: a time-qualified question may
    need both occurrences.

    Corrections are attached after this (``turns.source_annotations.expand``), and each pair brings those of its
    own messages and of what they depend on, so the same words can carry different ones. ``corrections(row)`` is
    what expansion attaches to a pair (the ids of its corrections), or None when expansion leaves the pair out:
    a pair whose corrections the kept one covers goes, and a pair left out covers nothing. It is asked only of
    pairs whose words repeat."""
    kept, seen, attached = [], {}, {}

    def brought(row):
        if id(row) not in attached:
            attached[id(row)] = corrections(row)
        return attached[id(row)]

    for row in rows:
        if row.get('kind') == 'conversation_pair' and row.get('contact_id'):
            try:
                pair = json.loads(row['content'])
                passages = [pair['input'], pair['response']]
            except (KeyError, TypeError, ValueError):
                passages = None
            if passages and not any('validity_status' in passage for passage in passages):
                key = (row['contact_id'], *((passage['role'], passage['quote']) for passage in passages))
                earlier = seen.setdefault(key, [])
                own = brought(row) if earlier else None
                if own is not None and any(brought(other) is not None and own <= brought(other) for other in earlier):
                    continue
                earlier.append(row)
        kept.append(row)
    return kept


def interleave_lanes(first, second, *, limit):
    """Two ranked lanes of prior claims taken in turn, one record each, ``first`` first, until ``limit``.

    Each turn a lane gives its next record not yet offered; a claim, or another row with the same value
    (``witness_key``), is offered once. A lane that runs out leaves the rest to the other. Rank order is all a
    lane contributes: no score fusion, so neither lane's scale can crowd out the other.
    """
    lanes, merged, ids, values = [iter(first), iter(second)], [], set(), set()
    while lanes and len(merged) < limit:
        for lane in list(lanes):
            if len(merged) >= limit:
                break
            for row in lane:
                value = witness_key(row)
                if row['id'] in ids or (value is not None and value in values):
                    continue
                ids.add(row['id'])
                if value is not None:
                    values.add(value)
                merged.append(row)
                break
            else:
                lanes.remove(lane)
    return merged


# The switch of the prior records' vector lane (``mind.faculties.priors_by_meaning``, on unless the file sets it to
# false). The claim projection runs outside the mind, so it reaches it as an environment flag, as the learning rules
# do (``config.apply_environment``).
PRIORS_BY_MEANING_FLAG = "COLONYMIND_PRIORS_BY_MEANING"


def priors_by_meaning(environ=None) -> bool:
    """Whether the vector lane of the prior records is on: unless the flag says off."""
    import os
    raw = (os.environ if environ is None else environ).get(PRIORS_BY_MEANING_FLAG, "")
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


class SourceClaimProjection:
    def __init__(self, ledger, vectors=None):
        """``vectors``: the source vector search (``turns.source_vectors.SourceVectors``) that gives ``prior``
        its vector lane, or None for the keyword lane alone."""
        self.ledger = ledger
        self.vectors = vectors

    def deadline(self, *, contact_id, session_id, source_id, source_version, claim_id, timezone_name='UTC', timezone_basis='caller_override'):
        """Resolve an explicitly selected claim's value, without creating work.

        The original source stays pinned even after a correction. A changed
        source, missing correction or annotation must not revive an old date.
        All eligibility, successor and value reads share one SQLite snapshot.
        """
        from colonymind.memory import curated
        return curated.fenced_read(self.ledger, lambda: self._read_deadline(contact_id=contact_id, session_id=session_id, source_id=source_id, source_version=source_version, claim_id=claim_id, timezone_name=timezone_name, timezone_basis=timezone_basis),
                                  empty={'original_source_ref': {'source_id': source_id, 'source_version': source_version}, 'original_claim_id': claim_id, 'timezone_name': timezone_name, 'timezone_basis': timezone_basis, 'status': 'unavailable', 'reason': 'source_changed'})

    def _read_deadline(self, *, contact_id, session_id, source_id, source_version, claim_id,
                 timezone_name="UTC", timezone_basis="caller_override"):
        from .source_time import source_deadline_time
        from colonymind.turns.idempotency import canonical_turn_digest

        original = {"original_source_ref": {"source_id": source_id, "source_version": source_version},
                    "original_claim_id": claim_id, "timezone_name": timezone_name,
                    "timezone_basis": timezone_basis}

        def unavailable(reason):
            return {**original, "status": "unavailable", "reason": reason}

        def expression(claim):
            # Extraction may retain the clock as value and its complete date
            # in event_time. Use that exact quoted operand only when it also
            # contains this claim's value; another event's date is not a join.
            value = claim['value']
            event = claim.get('event_time', {}).get('expression')
            if (isinstance(event, str) and event in claim['evidence']
                    and re.search(r'(?<!\w)' + re.escape(value) + r'(?!\w)', event)):
                return event
            return value

        with closing(self.ledger._connect()) as conn:
            conn.execute("BEGIN")

            def source_ref(identifier):
                row = conn.execute('''SELECT s.messages_json FROM turn_sources s
                    WHERE s.turn_id=? AND s.contact_id=? AND (s.scope='person' OR s.session_id=?)
                    AND NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=s.turn_id)
                    AND NOT EXISTS (SELECT 1 FROM source_projection_erasures e WHERE e.turn_id=s.turn_id)''',
                    (identifier, contact_id, session_id)).fetchone()
                return ({"source_id": identifier, "source_version": canonical_turn_digest(json.loads(row[0]))}
                        if row else None)

            def eligible(row):
                return (source_ref(row['turn_id']) is not None and not conn.execute('''
                    SELECT 1 FROM source_annotations a,json_each(a.target_message_hashes_json) h
                    WHERE a.target_source_id=? AND h.value=?''',
                    (row['turn_id'], row['message_hash'])).fetchone())

            if source_ref(source_id) != original['original_source_ref']:
                return unavailable("original_source_unavailable")
            selected = self._rows(conn, contact_id, session_id, ids=[claim_id])
            if len(selected) != 1 or not eligible(selected[0]):
                return unavailable("claim_unavailable")
            if selected[0]['turn_id'] != source_id:
                return unavailable("original_claim_mismatch")
            chain_sources = {source_id: original['original_source_ref']}
            # The same corrected undertaking can be selected in another
            # conversation. Its stable identity is the exact qualified root,
            # not whichever revision happened to be recalled this time.
            root, ancestors = selected[0], set()
            for _ in range(32):
                if root['id'] in ancestors:
                    return unavailable("invalid_successor_chain")
                ancestors.add(root['id'])
                if root.get('operation') not in {'correct', 'change'}:
                    break
                prior = self._rows(conn, contact_id, session_id, ids=[root.get('prior_claim_id')])
                if len(prior) != 1 or not eligible(prior[0]):
                    return unavailable("ancestor_unavailable")
                previous = prior[0]
                links = {previous[key] for key in ('retracted_by', 'superseded_by') if previous[key]}
                if (links != {root['id']} or (root['subject_key'], root['predicate']) !=
                        (previous['subject_key'], previous['predicate'])):
                    return unavailable("invalid_successor_chain")
                chain_sources[previous['turn_id']] = source_ref(previous['turn_id'])
                root = previous
            else:
                return unavailable("successor_limit")
            original['root_claim_id'] = root['id']
            seen, previous = ancestors - {claim_id}, None
            for _ in range(32):
                if claim_id in seen:
                    return unavailable("invalid_successor_chain")
                if len(seen) >= 32:
                    return unavailable("successor_limit")
                seen.add(claim_id)
                rows = self._rows(conn, contact_id, session_id, ids=[claim_id])
                if len(rows) != 1 or not eligible(rows[0]):
                    return unavailable("claim_unavailable")
                current = rows[0]
                chain_sources[current['turn_id']] = source_ref(current['turn_id'])
                if previous is None:
                    if current['turn_id'] != source_id:
                        return unavailable("original_claim_mismatch")
                elif (current.get('operation') not in {'correct', 'change'}
                      or current.get('prior_claim_id') != previous['id']
                      or (current['subject_key'], current['predicate']) !=
                         (previous['subject_key'], previous['predicate'])):
                    return unavailable("invalid_successor_chain")
                successors = {current[key] for key in ('retracted_by', 'superseded_by') if current[key]}
                if not successors:
                    break
                if len(successors) != 1:
                    return unavailable("ambiguous_successor_chain")
                previous, claim_id = current, successors.pop()
            else:
                return unavailable("successor_limit")

            result = {**original, "source_ref": source_ref(current['turn_id']), "claim_id": current['id'],
                      "source_refs": list(chain_sources.values()),
                      **{key: current[key] for key in ('subject', 'predicate', 'value', 'evidence')}}
            # An independently asserted competing value is not an authorized
            # correction. Never choose a deadline merely because it is newer.
            # Bound raw candidates before _rows filters message membership;
            # filtered early rows cannot hide a later conflicting claim.
            count = conn.execute('''SELECT count(*) FROM (SELECT c.id FROM source_claims c
                JOIN turn_sources s ON s.turn_id=c.turn_id WHERE s.contact_id=?
                AND (s.scope='person' OR s.session_id=?) AND c.subject_key=? AND c.predicate=? LIMIT 257)''',
                (contact_id, session_id, current['subject_key'], current['predicate'])).fetchone()[0]
            if count > 256:
                return {**result, "status": "unresolved", "reason": "claim_limit"}
            peers = self._rows(conn, contact_id, session_id,
                               key=(current['subject_key'], current['predicate']), limit=257)
            if any(not row['superseded_by'] and not row['retracted_by'] and eligible(row)
                   and norm_value(expression(row)) != norm_value(expression(current)) for row in peers):
                return {**result, "status": "unresolved", "reason": "conflicting_claims"}
            deadline = source_deadline_time(expression(current), observed_at=current['observed_at'],
                                            timezone_name=timezone_name, evidence=current['evidence'])
            result['deadline_time'] = deadline
            if deadline.get('status') != 'resolved' or deadline.get('precision') != 'instant':
                return {**result, "status": "unresolved", "reason": "deadline_not_instant"}
            return {**result, "status": "current", "deadline_at": deadline['at']}

    def _rows(self, conn, contact_id, session_id, *, turn_ids=None, message_hashes=None, ids=None, key=None, time_query=None, distinct_values=False, limit=256, before=None):
        """``before``: only the claims of sources said before this ``(occurrence, ingested_at, turn_id)``
        (``OCCURRENCE_SQL``), and of the messages of that source said before the message (``message_hashes``, the
        optional fourth member), for a message re-extracted in its place in history (``beliefs.reextract``)."""
        from colonymind.memory import curated
        return curated.fenced_read(conn, lambda: self._read_rows(conn, contact_id, session_id, turn_ids=turn_ids, message_hashes=message_hashes, ids=ids, key=key, time_query=time_query, distinct_values=distinct_values, limit=limit, before=before),
                                  empty=[])

    def _read_rows(self, conn, contact_id, session_id, *, turn_ids=None, message_hashes=None, ids=None,
              key=None, time_query=None, distinct_values=False, limit=256, before=None):
        from colonymind.turns.idempotency import source_message_hash
        where = ["s.contact_id=?", "(s.scope='person' OR s.session_id=?)",
                 "NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=s.turn_id)"]
        args = [contact_id, session_id]
        if ids is not None:
            if not ids:
                return []
            where.append("c.id IN (" + ",".join("?" for _ in ids) + ")")
            args.extend(ids)
        if turn_ids is not None or message_hashes is not None:
            alternatives = []
            for column, values in (("c.turn_id", turn_ids), ("c.message_hash", message_hashes)):
                if values:
                    alternatives.append(column + " IN (" + ",".join("?" for _ in values) + ")")
                    args.extend(values)
            if not alternatives:
                return []
            where.append("(" + " OR ".join(alternatives) + ")")
            # Looked up by source: the unary plus keeps the planner on the turn and message indexes rather than
            # walking every source of the person (``memory.time_lane`` reads by id the same way). Same rows.
            where[where.index("s.contact_id=?")] = "+s.contact_id=?"
        if key:
            where += ["c.subject_key=?", "c.predicate=?"]
            args.extend(key)
        if before is not None:
            clause = "(" + OCCURRENCE_SQL.format(s="s") + ",s.ingested_at,s.turn_id)<(?,?,?)"
            args.extend(before[:3])
            earlier = list(before[3]) if len(before) > 3 else []
            if earlier:
                clause = (f"({clause} OR (s.turn_id=? AND c.message_hash IN ("
                          + ",".join("?" for _ in earlier) + ")))")
                args.extend((before[2], *earlier))
            where.append(clause)
        if time_query:
            where.append("c.retracted_by IS NULL")
            if time_query.mode == "unresolved_time":
                pass  # Return labelled evidence, never certify a requested time.
            elif time_query.mode == "observed_range":
                # A source calendar day or a vague period ("in 2015") is an
                # interval, not an event at its first midnight. Source and query
                # timezones can give it different UTC boundaries. Retain overlap
                # without claiming an exact instant (``source_time.event_interval``).
                where.append(f"""(({_EVENT_INTERVAL_SQL}
                    AND json_extract(c.data_json,'$.event_time.start')<?
                    AND json_extract(c.data_json,'$.event_time.end_exclusive')>?)
                    OR (NOT {_EVENT_INTERVAL_SQL}
                    AND json_extract(c.data_json,'$.event_at')>=?
                    AND json_extract(c.data_json,'$.event_at')<?))""")
                args.extend((time_query.end, time_query.start, time_query.start, time_query.end))
            elif time_query.mode == "valid_range":
                where += [f"(c.valid_from IS NOT NULL OR {_QUOTED_SQL})",
                          "(c.valid_from IS NULL OR c.valid_from<?)", "(c.valid_to IS NULL OR c.valid_to>?)"]
                args.extend((time_query.end, time_query.start))
            else:
                where += ["(c.valid_from IS NULL OR c.valid_from<=?)", "(c.valid_to IS NULL OR c.valid_to>?)"]
                args.extend((time_query.start, time_query.start))
        from colonymind.memory import curated
        withdrawn = bool(conn.execute("SELECT 1 FROM curated_memory_events WHERE operation='forget' "
            "AND memory_id LIKE 'sources:%' LIMIT 1").fetchone()) if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='curated_memory_events'").fetchone() else False
        deduplicate = distinct_values and withdrawn
        if withdrawn:
            distinct_values = False  # choose a value after withdrawing its unavailable witnesses
        columns = "c.*,s.contact_id,s.session_id,s.scope,s.messages_json,s.occurred_at,s.ingested_at"
        if distinct_values:
            # A revoked inherited claim cannot win value deduplication over
            # an independent witness. Filter lifecycle eligibility inside the
            # scoped query, preserving its value limit and the grounding
            # checks on returned rows below.
            where.append("(json_extract(c.data_json,'$.subject_basis_claim_id') IS NULL OR EXISTS (SELECT 1 "
                + _subject_basis_source_sql("json_extract(c.data_json,'$.subject_basis_claim_id')", 's.contact_id') + '))')
            named = "json_extract(c.data_json,'$.subject_basis_message.{}')".format
            where.append(f"({named('turn_id')} IS NULL OR EXISTS (SELECT 1 "
                         + _message_basis_source_sql(named('turn_id'), 's.contact_id', named('message_hash')) + '))')
            from .value_revision import dependency_sql
            where.append(dependency_sql('c.data_json', 's.contact_id'))
            where.append(_contribution_basis_sql('c.data_json', 's.contact_id'))
            # Quoted preferences are source statements, not comparable scalar
            # values. Even repeated words can have distinct source scope or
            # conditions; retain their witnesses and explicit lifecycle.
            columns += ",row_number() OVER (PARTITION BY " \
                "coalesce(json_extract(c.data_json,'$.representation'),'')='preference'," \
                "CASE WHEN json_extract(c.data_json,'$.representation')='preference' " \
                "THEN c.id ELSE c.value_key END ORDER BY s.ingested_at DESC,c.id) AS value_rank"
        table = "source_claims"
        if (turn_ids is not None or message_hashes is not None or withdrawn) and has_witnesses(conn):
            # Read by source: a folded claim is found through every message that states it, as the
            # separate rows of an unfolded store were (its lifecycle is the claim's own).
            table = ("(SELECT id,turn_id,message_hash,subject_key,predicate,value_key,data_json,valid_from,valid_to,"
                     "superseded_by,retracted_by,witnesses FROM source_claims UNION ALL SELECT w.claim_id,w.turn_id,"
                     "w.message_hash,a.subject_key,a.predicate,a.value_key,w.data_json,a.valid_from,a.valid_to,"
                     "a.superseded_by,a.retracted_by,a.witnesses FROM source_claim_witnesses w "
                     "JOIN source_claims a ON a.id=w.claim_id)")
        query = ("SELECT " + columns + " FROM " + table + " c JOIN turn_sources s ON s.turn_id=c.turn_id WHERE "
                 + " AND ".join(where))
        if distinct_values:
            query = "SELECT * FROM (" + query + ") WHERE value_rank=1"
        query += " ORDER BY ingested_at DESC LIMIT ?"
        rows = conn.execute(query, [*args, limit]).fetchall()
        result, membership = [], {}
        for row in rows:
            turn = row["turn_id"]
            if turn not in membership:
                membership[turn] = {source_message_hash(row["session_id"], message)
                                    for message in json.loads(row["messages_json"])}
            if row["message_hash"] not in membership[turn]:
                continue
            data = json.loads(row["data_json"])
            applicability = quoted_applicability(data)
            if applicability is not None:
                data['applicability'] = applicability
            if data.get('subject_basis_claim_id'):
                basis = subject_basis(conn, data, contact_id=contact_id)
                if basis is None:
                    continue
                data['subject_basis'] = basis
            if data.get('subject_basis_message'):
                basis = message_subject_basis(conn, data, contact_id=contact_id)
                if basis is None:
                    continue
                data['subject_basis'] = basis
            if data.get('value_parts'):
                from .value_revision import value_basis
                basis = value_basis(conn, data, contact_id=contact_id)
                named = contribution_subject_bases(conn, data, contact_id=contact_id) if basis is not None else None
                if basis is None or named is None:
                    continue
                for part in basis:
                    if part['claim_id'] in named:
                        part['subject_basis'] = named[part['claim_id']]
                data['value_basis'] = basis
            data.update({key: row[key] for key in ("id", "turn_id", "message_hash", "subject_key", "predicate",
                        "valid_from", "valid_to", "superseded_by", "retracted_by")})
            if "witnesses" in row.keys() and (row["witnesses"] or 1) > 1:
                data["witnesses"] = row["witnesses"]
            data.update(observed_at=row["occurred_at"], recorded_at=row["ingested_at"])
            result.append(data)
        try:
            result = [row for row in result if not curated.is_forgotten(conn, row)]
            if withdrawn:
                witnesses = {}
                for row in result:
                    witnesses.setdefault(row['id'], []).append(row)
                if turn_ids is None and message_hashes is None:
                    result = [rows[0] for rows in witnesses.values()]
                    for row in result:
                        row['witnesses'] = len(witnesses[row['id']])
            return one_witness_per_value(result) if deduplicate else result
        except sqlite3.Error:
            # Final recall owns the established loud fail-open health contract.
            return result

    def known_keys(self, source, limit=None, *, before=None):
        """The person's recent distinct claim keys (``{subject, predicate}``), newest first, at most
        ``KNOWN_KEYS``: shown to the extractor so a new assertion reuses a key instead of naming the property
        again. Read through the scoped row reader, so never another person's, an erased or reattributed
        message's, or a retracted or superseded claim's; an episode's content-derived identity is no key.
        ``before``: as ``_rows``."""
        from colonymind.memory import curated
        return curated.fenced_read(self.ledger, lambda: self._read_known_keys(source, limit, before=before),
                                  empty=[])

    def _read_known_keys(self, source, limit=None, *, before=None):
        limit = KNOWN_KEYS if limit is None else limit
        keys = {}
        with closing(self.ledger._connect()) as conn:
            rows = self._rows(conn, source["contact_id"], source["session_id"], before=before)
        for row in rows:
            if (row['superseded_by'] or row['retracted_by'] or row.get('representation') == 'episode'
                    or row['turn_id'] == source.get('turn_id')):
                continue
            keys.setdefault((row['subject_key'], row['predicate']),
                            {'subject': row['subject'], 'predicate': row['predicate']})
            if len(keys) == limit:
                break
        return list(keys.values())

    async def similar_turns(self, source, message, limit=SIMILAR_TURNS):
        """The vector lane of ``prior``: the person-scope turns the existing source vector search finds for
        the new message, best first. It makes no vector and changes no index, so nothing new needs erasing.
        Empty without vectors, or with the lane switched off (``priors_by_meaning``); a search that fails leaves the
        keyword lane alone.

        Only the person's own turns carry claims, so the search reads the person scope alone: it is given a
        session no source belongs to, whose session scope is empty, so the scope filter drops every checkpoint
        chunk before the nearest neighbours are ranked and cut to ``limit``. Searching the message's own session
        too would let its checkpoint chunks, which hold no claims, fill every place."""
        text = message.get("content")
        if self.vectors is None or not isinstance(text, str) or not text.strip() or not priors_by_meaning():
            return []
        try:
            hits, _ = await self.vectors.search(text, contact_id=source["contact_id"],
                                                session_id="prior-claims:" + uuid.uuid4().hex, limit=limit)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("prior claims by meaning unavailable (%s); the keyword lane stands", type(exc).__name__)
            return []
        return list(dict.fromkeys(hit["turn_id"] for hit in hits
                                  if hit.get("scope") == "person" and hit["turn_id"] != source.get("turn_id")))

    def earlier_messages(self, source, message, *, limit=None):
        """The conversation's previous user messages, offered to the extraction of ``message`` as where a subject
        its quotation names only by a pronoun may have been named ("My sister moved." before "She lives in Denver
        now.").

        They are the typed user messages of the same person, conversation (session) and person scope said before
        ``message``: those before it in its own source, then those of the conversation's earlier sources (by when
        each was said, then recorded), newest first, said within ``EARLIER_WINDOW_SECONDS`` of it, at most
        ``EARLIER_MESSAGES`` (``limit``), and returned oldest first with the ids the call names them by. Never
        another person's or conversation's message, an erased, reattributed or annotated one, an assistant's or a
        transcript's. A message longer than ``EARLIER_CHARS`` is offered by its end. Each row: ``id``, ``turn_id``,
        ``message_hash``, ``reported_at``, ``text`` (the words offered), ``start`` (where they begin in the message)
        and ``content`` (the whole message, which no call is shown). The dry run and re-extraction offer the same.

        A row that the assistant answered also carries ``assistant_after``: what the assistant said after it and
        before the next user message (the last ``EARLIER_MESSAGES`` of its messages with words, each by its end,
        oldest first), so the calls can tell whom the current message answers ("Where does your mother live?"
        between the two above makes "She" the mother). They are context only: no id names them, so no subject is
        ever taken from them. After a user message that is not offered (annotated, a transcript) the assistant's
        words are not shown either: they answer what is hidden."""
        from colonymind.turns.audio import claim_message
        from colonymind.turns.idempotency import source_message_hash
        limit = EARLIER_MESSAGES if limit is None else limit
        said = source.get('occurred_at') or source.get('ingested_at')
        if source.get('scope', 'person') != 'person' or not said or limit <= 0:
            return []
        session, turn = source['session_id'], source.get('turn_id') or ''
        current = source_message_hash(session, message)
        found, replies = [], []     # replies: the assistant's words since the last user message walked, newest first

        def take(row, raw, annotated):
            content = raw.get('content')
            if raw.get('role') == 'assistant':
                if isinstance(content, str) and content.strip():
                    replies.append(content[-EARLIER_CHARS:])
                return
            if raw.get('role') != 'user':
                return
            identity = source_message_hash(row['session_id'], raw)
            if isinstance(content, str) and content.strip() and identity not in annotated and len(found) < limit:
                start = max(0, len(content) - EARLIER_CHARS)
                found.append({'turn_id': row['turn_id'], 'message_hash': identity, 'reported_at': row['occurred_at'],
                              'text': content[start:], 'start': start, 'content': content,
                              **({'assistant_after': replies[:EARLIER_MESSAGES][::-1]} if replies else {})})
            replies.clear()

        def annotations(conn, turn_id):
            return {value for (raw,) in conn.execute(
                'SELECT target_message_hashes_json FROM source_annotations WHERE target_source_id=?', (turn_id,))
                for value in json.loads(raw)}

        with closing(self.ledger._connect()) as conn:
            own = json.loads(source['messages_json']) if source.get('messages_json') else []
            position = next((index for index, raw in enumerate(own) if current in {
                source_message_hash(session, raw), source_message_hash(session, claim_message(raw) or raw)}), None)
            if position:
                annotated = annotations(conn, turn)
                for raw in reversed(own[:position]):
                    take(source, raw, annotated)
            occurrence = OCCURRENCE_SQL.format(s='s')
            rows = conn.execute(f'''SELECT s.turn_id,s.session_id,s.messages_json,s.occurred_at FROM turn_sources s
                WHERE s.contact_id=? AND s.session_id=? AND s.scope='person' AND s.turn_id<>?
                AND NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=s.turn_id)
                AND NOT EXISTS (SELECT 1 FROM source_projection_erasures e WHERE e.turn_id=s.turn_id)
                AND ({occurrence},s.ingested_at,s.turn_id)<(julianday(?),?,?) AND {occurrence}>=julianday(?)-?
                ORDER BY {occurrence} DESC,s.ingested_at DESC,s.turn_id DESC LIMIT ?''',
                (source['contact_id'], session, turn, said, source.get('ingested_at') or '\uffff', turn, said,
                 EARLIER_WINDOW_SECONDS / 86400, 4 * limit)).fetchall() if len(found) < limit else []
            for row in rows:
                if len(found) >= limit:
                    break
                annotated = annotations(conn, row['turn_id'])
                for raw in reversed(json.loads(row['messages_json'])):
                    take(row, raw, annotated)
        found.reverse()
        return [{'id': f'earlier_{index}', **row} for index, row in enumerate(found, 1)]

    async def prior_claims(self, source, message, limit=16, *, before=None, reviving=()):
        """``prior`` with its vector lane (``similar_turns``) when this projection has vectors."""
        return self.prior(source, message, limit, similar=await self.similar_turns(source, message), before=before,
                          reviving=reviving)

    @staticmethod
    def _as_offered(row, reviving):
        """A record one of ``reviving`` retired is offered as it stood before: live. ``reviving`` holds a
        re-extracted message's own earlier claims, so the message is offered what it was offered the first time
        (``beliefs.reextract``)."""
        if reviving and (row['superseded_by'] in reviving or row['retracted_by'] in reviving):
            return dict(row, superseded_by=None, retracted_by=None)
        return row

    def _offered(self, conn, source, rows, relevance, limit, *, before=None, reviving=frozenset()):
        """A lane's rows as priors: only live claims, one witness per value, and for each relevant episode its
        current revision. Relevance can find a retracted report without finding its latest count-only
        correction: offer the existing current episode, never the stale handle. The normal row reader rechecks
        source scope, attribution and retained basis; no erased history is reconstructed. ``before`` and
        ``reviving``: as ``prior``."""
        rows = [self._as_offered(row, reviving) for row in rows]
        keys = list(dict.fromkeys((row['subject_key'], row['predicate'])
            for row in sorted(rows, key=relevance, reverse=True)
            if row.get('representation') == 'episode'))[:limit]
        for key in keys:
            rows.extend(self._as_offered(row, reviving) for row in self._rows(
                conn, source['contact_id'], source['session_id'], key=key, limit=limit, before=before))
        return one_witness_per_value(row for row in rows if not row['superseded_by'] and not row['retracted_by'])

    def prior(self, source, message, limit=16, *, similar=None, before=None, reviving=()):
        """The prior records offered to the extractor for ``message``, at most ``limit``.

        The keyword lane: the live claims of the turns a keyword search of the message finds (with the claims whose
        subject a message of those turns named), or, when it finds none, the person's most relevant recent claims;
        ranked by shared words, then recency. ``similar`` is the
        vector lane (``similar_turns``): the live claims of those turns, in their search order. The lanes are
        interleaved one for one, keyword first (``interleave_lanes``), not fused by score. Without a vector
        lane the keyword lane is returned as it always was.

        Re-extraction (``beliefs.reextract``) offers a message of history what it was offered the first time:
        ``before`` keeps the claims of sources said before it (``_rows``), and a claim one of ``reviving`` (the
        message's own earlier claims) retired is offered live, as it stood before the message.
        """
        from colonymind.memory import curated
        return curated.fenced_read(self.ledger, lambda: self._read_prior(source, message, limit, similar=similar, before=before, reviving=reviving),
                                  empty=[])

    def _read_prior(self, source, message, limit=16, *, similar=None, before=None, reviving=()):
        reviving = frozenset(reviving)
        words = set(norm_value(message.get("content", "")).split())

        def relevance(row):
            # A partial correction can omit every topic word in the original
            # report. Its retained, scoped basis still identifies that episode,
            # and an earlier message that named a claim's subject holds its words.
            basis = (row.get('subject_basis', {}) if row.get('representation') == 'episode'
                     or row.get('subject_basis_message') else {})
            evidence = row['evidence'] + ' ' + basis.get('evidence', '')
            return len(words & set(norm_value(evidence).split())), row['recorded_at']

        def of_turns(conn, turn_ids):
            """The claims of these turns, and the claims whose subject a message of one of them named."""
            rows = self._rows(conn, source["contact_id"], source["session_id"], turn_ids=turn_ids, before=before)
            if turn_ids:
                named = [row[0] for row in conn.execute(
                    'SELECT id FROM source_claims WHERE ' + _BASIS_TURN_SQL + ' IN ('
                    + ','.join('?' for _ in turn_ids) + ') LIMIT 256', turn_ids)]
                if named:
                    rows += self._rows(conn, source["contact_id"], source["session_id"], ids=named, before=before)
            return rows

        hits = self.ledger.search_sources(message.get("content", ""), contact_id=source["contact_id"],
                                          session_id=source["session_id"], limit=10)
        turn_ids = [hit["turn_id"] for hit in hits if hit["turn_id"] != source["turn_id"]]
        rank = {turn: position for position, turn in enumerate(dict.fromkeys(similar or ()))
                if turn != source["turn_id"]}
        meaning = []
        with closing(self.ledger._connect()) as conn:
            rows = self._offered(conn, source, of_turns(conn, turn_ids or None), relevance, limit,
                                 before=before, reviving=reviving)
            if rank:
                found = of_turns(conn, list(rank))
                # A folded claim is found through each of its witness turns: it takes its best turn's rank,
                # and an episode's current revision takes the rank of the report that led to it. A claim whose
                # subject an earlier message named takes that message's rank when only it was found.
                claim_rank, key_rank = {}, {}
                for row in found:
                    position = rank.get(row['turn_id'], rank.get(
                        (row.get('subject_basis_message') or {}).get('turn_id'), len(rank)))
                    claim_rank[row['id']] = min(claim_rank.get(row['id'], position), position)
                    key = (row['subject_key'], row['predicate'])
                    key_rank[key] = min(key_rank.get(key, position), position)
                meaning = self._offered(conn, source, list(found), relevance, limit, before=before,
                                        reviving=reviving)
                meaning.sort(key=lambda row: claim_rank.get(
                    row['id'], key_rank.get((row['subject_key'], row['predicate']), len(rank))))
        rows.sort(key=relevance, reverse=True)
        if not meaning:
            return rows[:limit]
        return interleave_lanes(rows, meaning, limit=limit)

    async def rehearse(self, router, *, contact_id, session_id, text, occurred_at, timezone_name="UTC",
                       prompt_only=False):
        """A dry run of one message in a person's scope: what extraction would make of ``text`` and what the
        commit would do with it, with nothing written.

        It offers the person's priors (both lanes) and key list, runs extraction, every validation step and
        the review through ``router``, then the commit's own checks and writes for a transient source in a
        transaction it rolls back, so each verdict is the one a real commit of the same input would reach.
        No job, vector, learning-log row, claim or sealed value survives it. ``prompt_only``: the prompts
        extraction would send, with no model call. The caller owns authority: this reads the named person's
        records.

        ``text`` is first admitted as ingestion admits a message (``_admit``): sealed, so the vector lane's query
        and every prompt carry a detected secret's reference, never its value; and an exact copy of an erased
        message is refused (``status: source_erased``) before any model call, as ingestion refuses it.
        """
        source = {"turn_id": "dry-run:" + uuid.uuid4().hex, "contact_id": contact_id, "session_id": session_id,
                  "scope": "person", "occurred_at": occurred_at}
        message = self._admit(source, {"role": "user", "content": text})
        if message is None:
            return {"extraction_version": EXTRACTION_VERSION, "writes": "none", "status": "source_erased"}
        prior = await self.prior_claims(source, message)
        keys = self.known_keys(source)
        earlier = self.earlier_messages(source, message)
        result = {"extraction_version": EXTRACTION_VERSION, "writes": "none", "known_keys": keys,
                  "offered_priors": [{k: row[k] for k in ("id", "representation", "subject", "predicate", "value",
                                                          "evidence", "recorded_at") if k in row} for row in prior],
                  "offered_earlier_messages": [{k: row[k] for k in ("id", "turn_id", "reported_at", "text",
                                                                    "assistant_after") if k in row}
                                               for row in earlier]}
        if prompt_only:
            request = extraction_request(source, message, prior, timezone_name=timezone_name, known_keys=keys,
                                         earlier=earlier)
            return {**result, "status": "prompt_only", "prompts": request["messages"],
                    "response_schema": request["response_schema"]}
        metered = _Metered(router)
        diagnostics, rejected = extraction_diagnostics(), []
        try:
            claims, model = await extract_claims(metered, source, message, prior, timezone_name=timezone_name,
                                                 diagnostics=diagnostics, known_keys=keys, rejections=rejected,
                                                 earlier=earlier)
        except SourceClaimOutputError as exc:
            return {**result, "status": "invalid_output", "reason": str(exc), "rejected": rejected,
                    "usage": metered.usage(), "diagnostics": diagnostics}
        if model in _NOT_EXTRACTED:
            return {**result, "status": model, "usage": metered.usage()}
        outcomes = self._rehearse_commit(source, message, claims, model=model)
        accepted = []
        for index, claim in enumerate(claims):
            outcome = outcomes[index] if index < len(outcomes) else {
                "verdict": "rejected", "stage": "commit", "reason": "source no longer available"}
            if outcome["verdict"] == "rejected":
                rejected.append({"stage": "commit", "reason": outcome["reason"], "proposal": claim})
            else:
                accepted.append({"claim": claim, **outcome})
        return {**result, "status": "complete", "model": model, "accepted": accepted, "rejected": rejected,
                "usage": metered.usage(), "diagnostics": diagnostics}

    def _admit(self, source, arrived):
        """``arrived`` as ``record_source`` would admit it, with nothing written: the message with every detected
        secret replaced by its reference, before the lexical search, the vector query and any model call see
        it; or None when ingestion would refuse it as an exact copy of an erased message of its session, in any
        form the message has had (as it came, sealed, or under a reference its value had before). The
        references are only planned (``secrets.seal.Staged``, never committed, through ``_PlanOnly``), so the
        vault stays exactly as it was. Raises ``VaultUnavailable`` when an erasure in the session cannot be
        checked because the vault key cannot be read, as ingestion does. The message is a plain user message: it
        names no source it depends on."""
        from colonymind.secrets.seal import Scope, Staged, erasure_forms, redact_sensitive_messages
        from colonymind.secrets.vault import VaultUnavailable
        from colonymind.turns.idempotency import _ErasureRules, source_message_hash
        contact_id, session_id = source["contact_id"], source["session_id"]
        vault, stage = _PlanOnly(self.ledger.vault), Staged()
        message = redact_sensitive_messages(
            [arrived], scope=Scope(contact_id=contact_id, session_id=session_id, source_turn=source["turn_id"]),
            vault=vault, stage=stage)[0]
        identity, forms = source_message_hash(session_id, message), set()
        if message.get("content") != arrived.get("content"):
            forms.add(source_message_hash(session_id, arrived))
        with closing(self.ledger._connect()) as conn:
            rules = _ErasureRules(self.ledger._erasure_rules(conn, contact_id))
            from colonymind.memory import curated
            withdrawal = curated._state(conn, self.ledger, contact_id, source_only=True).forgotten
        if rules.exact.get(session_id) or withdrawal:
            # With no vault at all no stored message ever carried a reference: the form it came in is all.
            if stage.withheld and self.ledger.vault.db_path.exists():
                raise VaultUnavailable("a value of this message was withheld, so its erasure cannot be checked now")
            forms.update(source_message_hash(session_id, {"role": message.get("role"), "content": form})
                         for form in erasure_forms(message["content"], contact_id=contact_id, vault=vault,
                                                   stage=stage))
        retained = self.ledger._retained_messages([message], session_id, rules, {identity: forms} if forms else None)
        if any(curated.is_forgotten(self.ledger, {'contact_id': contact_id, 'message_hash': digest})
               for digest in {identity, *forms}):
            return None
        return message if retained else None

    def _rehearse_commit(self, source, message, claims, *, model):
        """The commit's verdicts for ``claims`` of a message not in the ledger: its source row, the commit and
        what the commit changed are read in one transaction that is always rolled back. A new claim's own id
        exists only inside it and is not reported; what it would retract, supersede or hold for the owner is."""
        from colonymind.util.temporal import now_utc
        outcomes = []
        messages = json.dumps([message], ensure_ascii=False)
        with closing(self.ledger._connect()) as conn:
            conn.execute('BEGIN IMMEDIATE')
            try:
                conn.execute('''INSERT INTO turn_sources(turn_id,content_sha256,contact_id,session_id,scope,
                    messages_json,occurred_at,ingested_at) VALUES (?,?,?,?,?,?,?,?)''',
                    (source["turn_id"], hashlib.sha256(messages.encode()).hexdigest(), source["contact_id"],
                     source["session_id"], "person", messages, source["occurred_at"],
                     now_utc().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'))
                self._commit(conn, source, message, claims, model=model, outcomes=outcomes)
                for outcome in outcomes:
                    if outcome.get("effect") != "new":
                        continue
                    cid = outcome.pop("claim_id")
                    outcome.update(
                        retracts=[r[0] for r in conn.execute(
                            'SELECT id FROM source_claims WHERE retracted_by=? ORDER BY id', (cid,))],
                        supersedes=[r[0] for r in conn.execute(
                            'SELECT id FROM source_claims WHERE superseded_by=? ORDER BY id', (cid,))],
                        held_for_owner=[r[0] for r in conn.execute(
                            "SELECT target_id FROM learning_pending WHERE by_id=? AND status='pending' "
                            "ORDER BY target_id", (cid,))])
            finally:
                conn.rollback()
        return outcomes

    def commit(self, source, message, claims, *, model, lease_token=None):
        with closing(self.ledger._connect()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            return self._commit(conn, source, message, claims, model=model, lease_token=lease_token)

    def _commit(self, conn, source, message, claims, *, model, lease_token=None, outcomes=None):
        """The commit's checks and writes on ``conn``, inside its open transaction. ``outcomes`` receives one
        verdict per supplied claim, in order: ``accepted`` (``new``, or ``restates`` a live claim), ``unchanged``
        (this message already recorded it) or ``rejected`` with the commit's reason."""
        from colonymind.turns.idempotency import source_message_hash, canonical_turn_digest
        from colonymind.turns.audio import claim_message, claim_basis
        from colonymind.memory import learning
        message_hash = source_message_hash(source["session_id"], message)
        from colonymind.memory import curated
        if curated.is_forgotten(conn, {'turn_id': source['turn_id'], 'message_hash': message_hash}):
            return 0
        rules = learning.enabled()
        if lease_token is not None and not conn.execute(
            "SELECT 1 FROM source_claim_jobs WHERE turn_id=? AND status='running' AND lease_token=?",
            (source["turn_id"], lease_token)).fetchone():
            return 0
        current = conn.execute('''SELECT * FROM turn_sources s WHERE turn_id=? AND NOT EXISTS
            (SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=s.turn_id)''',
            (source["turn_id"],)).fetchone()
        if current is None:
            return 0
        current_messages = json.loads(current['messages_json'])
        original = next((m for m in current_messages
            if source_message_hash(current['session_id'], m) == message_hash), None)
        current_view = claim_message(original) if original is not None else None
        if (current_view is None or current_view.get('content') != message.get('content')
                or current_view.get('_audio_segments') != message.get('_audio_segments')):
            return 0
        ids = list({claim["prior_claim_id"] for claim in claims if claim.get("prior_claim_id")})
        prior = {row["id"]: row for row in self._rows(conn, current["contact_id"], current["session_id"], ids=ids)}
        written = 0

        def reject(stage, reason):
            # The mandatory rejection log: ids and the reason, never the words.
            learning.log(conn, store="claim", action="rejected", stage=stage, reason=reason,
                         contact_id=current["contact_id"], refs=[source["turn_id"], message_hash])
            verdict(verdict='rejected', stage=stage, reason=reason)

        def verdict(**outcome):
            if outcomes is not None:
                outcomes.append(outcome)
        for raw in claims:
            claim = dict(raw)
            # Validation occurs against exact current message bytes again,
            # after model execution and after any concurrent source erase.
            if current_view["content"][claim["span_start"]:claim["span_end"]] != claim["evidence"]:
                reject("commit", "evidence no longer matches the source message")
                continue
            if 'source_admission' in claim and (admission_metadata(claim) is None
                    or '_audio_segments' in current_view or claim['evidence'] != current_view['content']):
                reject("commit", "whole-source admission no longer valid")
                continue
            if '_audio_segments' in current_view:
                basis = claim_basis(current_view, claim['span_start'], claim['span_end'])
                review = claim.get('admission_review', {})
                if (basis is None or basis != claim.get('evidence_basis')
                        or review.get('version') != 'source-claim-review-v1'
                        or review.get('basis') != 'model_judgment_unverified'):
                    reject("commit", "audio evidence basis changed")
                    continue
                # An observed conversation (``turns.observed``) has no clip: its words are the ledger's own.
                if basis['source_modality'] != 'observed_conversation' and not conn.execute(
                    '''SELECT 1 FROM source_media_links l JOIN source_media m USING(asset_hash)
                    WHERE l.turn_id=? AND l.message_hash=? AND l.asset_hash=?
                    AND m.mime_type='audio/wav' AND m.status='complete' ''',
                    (source['turn_id'], message_hash, basis['segment']['asset_id'][7:])).fetchone():
                    reject("commit", "audio segment media unavailable")
                    continue
                claim.update(epistemic_state='derived_unverified', source_modality=basis['source_modality'],
                    evidence_basis={**basis, 'source_message_hash': message_hash,
                        'source_version_at_formation': canonical_turn_digest(current_messages)})
            basis = [source["turn_id"], message_hash, claim["subject_key"], claim["predicate"], claim["value"], claim["evidence"]]
            if claim.get('representation') == 'preference':
                basis.append('preference')
            if claim.get('subject_basis_claim_id'):
                basis.append(claim['subject_basis_claim_id'])
            if claim.get('subject_basis_message'):
                # The span, not the request's own name for the message, is part of the claim.
                claim['subject_basis_message'] = durable_message_basis(claim['subject_basis_message'])
                if claim['subject_basis_message'] is None or not self._names_subject(conn, current, message_hash,
                                                                                     claim):
                    reject("commit", "subject basis no longer current")
                    continue
                basis.append(claim['subject_basis_message'])
            if claim.get('value_parts'):
                basis.append(claim['value_parts'])
            cid = "claim:" + hashlib.sha256(json.dumps(basis, ensure_ascii=False).encode()).hexdigest()
            if conn.execute('SELECT 1 FROM source_claims WHERE id=?', (cid,)).fetchone():
                verdict(verdict='unchanged', claim_id=cid)
                continue
            old = prior.get(claim.get("prior_claim_id"))
            if claim.get('value_parts'):
                from .value_revision import revision_parts, value_basis
                live = self._rows(conn, current['contact_id'], current['session_id'],
                                  ids=[claim.get('prior_claim_id')])
                old = live[0] if live else None
                if (old is None or old['subject_key'] != claim['subject_key']
                        or old['predicate'] != claim['predicate'] or old['subject'] != claim['subject']
                        or old['superseded_by'] or old['retracted_by']
                        or claim['operation'] not in {'correct', 'change'}
                        or admission_metadata(claim) is None
                        or revision_parts(claim['value'], claim['evidence'], old) != claim['value_parts']
                        or value_basis(conn, claim, contact_id=current['contact_id']) is None
                        or contribution_subject_bases(conn, claim, contact_id=current['contact_id']) is None):
                    reject("commit", "partial revision has no current basis")
                    continue
            if claim.get('subject_basis_claim_id'):
                # Re-read current status even when another candidate in
                # this same batch already corrected the supplied prior.
                live = self._rows(conn, current['contact_id'], current['session_id'],
                                  ids=[claim.get('prior_claim_id')])
                old = live[0] if live else None
                if (old is None or old['subject'] != claim['subject']
                        or old['subject_key'] != claim['subject_key'] or old['predicate'] != claim['predicate']
                        or old['superseded_by'] or old['retracted_by']
                        or claim['operation'] not in {'correct', 'change'}
                        or admission_metadata(claim) is None
                        or claim['subject_basis_claim_id'] != (old.get('subject_basis_claim_id') or old['id'])
                        or subject_basis(conn, claim, contact_id=current['contact_id']) is None):
                    reject("commit", "subject basis no longer current")
                    continue  # A missing dependency cannot become assert.
            if old and (old["subject_key"] != claim["subject_key"] or old["predicate"] != claim["predicate"]
                        or old["superseded_by"] or old["retracted_by"]):
                old = None
            if not old:
                claim["operation"] = "assert"
            claim.update(model=model, extraction_version=EXTRACTION_VERSION, role="user")
            # A reviewed operation selects the quoted statement to retire;
            # normalized wording is not a test of preference equivalence.
            lifecycle = bool(old and (old.get('representation') == 'preference'
                                      or claim.get('representation') == 'preference'
                                      or norm_value(old["value"]) != norm_value(claim["value"])))
            restated = rules and not lifecycle and self._fold(conn, current, current_view, message_hash, claim, cid)
            if restated:
                verdict(verdict='accepted', effect='restates', claim_id=restated)
                written += 1
                continue
            conn.execute('''INSERT INTO source_claims
                (id,turn_id,message_hash,subject_key,predicate,value_key,data_json,valid_from,valid_to)
                VALUES (?,?,?,?,?,?,?,?,?)''', (cid, source["turn_id"], message_hash,
                claim["subject_key"], claim["predicate"], norm_value(claim["value"]), json.dumps(claim, ensure_ascii=False),
                claim["valid_from"], claim["valid_to"]))
            if lifecycle and (claim["operation"] == "correct" or (
                    claim["operation"] == "change" and claim["valid_from"])):
                if rules and self._preference(old, claim) and not learning.owner_links(
                        old.get("evidence"), claim["evidence"], old.get("predicate")):
                    # A preference retires only by its owner: the owner's words carry the cue, but not
                    # what the model says they retire. Both statements stay until the owner ratifies
                    # (``decide``) or declines.
                    learning.propose(conn, store="claim", kind="preference_change",
                                     contact_id=current["contact_id"], target_id=old["id"], by_id=cid,
                                     operation=claim["operation"])
                else:
                    if rules and self._preference(old, claim):
                        learning.log(conn, store="claim", action="ratified", stage="preference_change",
                                     reason="the owner's own words name it", item_id=old["id"],
                                     contact_id=current["contact_id"], refs=[cid])
                    # With the learning rules a correction retracts the value, not only the witness the
                    # model was shown: an earlier duplicate row must not resurrect it.
                    targets = [old["id"], *(self._same_value(conn, current, old) if rules else [])]
                    self._retire(conn, targets, claim, cid)
            verdict(verdict='accepted', effect='new', claim_id=cid)
            written += 1
        # The claim key index (``turns.claim_keys``) is written with the claims it locates, in this transaction.
        from colonymind.turns import claim_keys
        claim_keys.index_pending(conn)
        return written

    def _names_subject(self, conn, current, message_hash, claim):
        """The commit's check of a subject named in an earlier message (``subject_basis_message``), against the
        store as it is now: the message still names the subject (``message_subject_basis``), the claim was reviewed,
        and either the message is one this message's extraction could have been offered (the same person and
        conversation, said before it and within ``EARLIER_WINDOW_SECONDS``), or the claim corrects or changes a live
        prior claim of the same subject and property that the same message named."""
        from colonymind.turns.idempotency import source_message_hash
        basis = claim['subject_basis_message']
        if admission_metadata(claim) is None or message_subject_basis(
                conn, claim, contact_id=current['contact_id']) is None:
            return False
        named = conn.execute('SELECT * FROM turn_sources WHERE turn_id=?', (basis['turn_id'],)).fetchone()
        if named is not None and named['session_id'] == current['session_id']:
            if named['turn_id'] == current['turn_id']:
                order = [source_message_hash(current['session_id'], m) for m in json.loads(current['messages_json'])]
                if (basis['message_hash'] in order and message_hash in order
                        and order.index(basis['message_hash']) < order.index(message_hash)):
                    return True
            else:
                occurrence = OCCURRENCE_SQL.format(s='b'), OCCURRENCE_SQL.format(s='c')
                if conn.execute(f'''SELECT 1 FROM turn_sources b, turn_sources c WHERE b.turn_id=? AND c.turn_id=?
                        AND ({occurrence[0]},b.ingested_at,b.turn_id)<({occurrence[1]},c.ingested_at,c.turn_id)
                        AND {occurrence[0]}>={occurrence[1]}-?''',
                        (named['turn_id'], current['turn_id'], EARLIER_WINDOW_SECONDS / 86400)).fetchone():
                    return True
        if claim.get('operation') not in {'correct', 'change'} or not claim.get('prior_claim_id'):
            return False
        live = self._rows(conn, current['contact_id'], current['session_id'], ids=[claim['prior_claim_id']])
        old = live[0] if live else None
        return bool(old is not None and not old['superseded_by'] and not old['retracted_by']
                    and (old['subject'], old['subject_key'], old['predicate']) == (
                        claim['subject'], claim['subject_key'], claim['predicate'])
                    and durable_message_basis(old.get('subject_basis_message')) == basis)

    # -- the learning layer (memory/learning.py) ------------------------------------------------------

    @staticmethod
    def _preference(old, claim):
        def preference(row):
            return (row.get('representation') == 'preference'
                    or (row.get('memory_quality') or {}).get('memory_kind') == 'preference')
        return preference(old) or preference(claim)

    @staticmethod
    def _retire(conn, targets, claim, cid):
        for target in dict.fromkeys(targets):
            row = conn.execute('SELECT valid_from,valid_to,superseded_by,retracted_by FROM source_claims WHERE id=?',
                               (target,)).fetchone()
            if row is None or row['retracted_by'] or row['superseded_by']:
                continue
            if claim["operation"] == "correct":
                conn.execute('UPDATE source_claims SET retracted_by=? WHERE id=?', (cid, target))
            elif claim["operation"] == "change" and claim["valid_from"]:
                if not row["valid_from"] or row["valid_from"] < claim["valid_from"]:
                    conn.execute('UPDATE source_claims SET valid_to=?,superseded_by=? WHERE id=?',
                                 (min(row['valid_to'], claim['valid_from']) if row['valid_to']
                                  else claim['valid_from'], cid, target))

    def _index(self, conn, current, *, subject_key, predicate, value_key, premises=False):
        """The gate's exact index for one value: every claim of this person with the same key and literal
        value, each with its fold key and whether it is live (current and still eligible to read)."""
        from colonymind.memory.learning import Item, verbatim
        from colonymind.turns.idempotency import source_message_hash
        from colonymind.turns.audio import claim_message
        index = []
        for row in conn.execute('''SELECT c.*,s.session_id,s.messages_json FROM source_claims c
                JOIN turn_sources s ON s.turn_id=c.turn_id WHERE s.contact_id=? AND s.scope='person'
                AND c.subject_key=? AND c.predicate=? AND c.value_key=?''',
                (current['contact_id'], subject_key, predicate, value_key)).fetchall():
            data = json.loads(row['data_json'])
            message = next((claim_message(m) for m in json.loads(row['messages_json'])
                            if source_message_hash(row['session_id'], m) == row['message_hash']), None)
            whole = bool(message) and verbatim(data.get('evidence')) == verbatim(message.get('content'))
            key = fold_key(current['contact_id'], row['subject_key'], row['predicate'], data,
                           row['valid_from'], row['valid_to'], whole=whole, premises=premises)
            if key is None:
                continue
            live = (not row['superseded_by'] and not row['retracted_by'] and bool(self._rows(
                conn, current['contact_id'], current['session_id'], ids=[row['id']])))
            index.append(Item(row['id'], '', key, live=live))
        return index

    def _same_value(self, conn, current, old):
        """Live rows restating ``old``'s value (an unfolded store's duplicates): retired with it."""
        raw = conn.execute('SELECT subject_key,predicate,value_key FROM source_claims WHERE id=?', (old['id'],)).fetchone()
        if raw is None:
            return []
        index = self._index(conn, current, subject_key=raw['subject_key'], predicate=raw['predicate'],
                            value_key=raw['value_key'], premises=True)
        key = next((item.key for item in index if item.id == old['id']), None)
        return [item.id for item in index if key and item.key == key and item.id != old['id'] and item.live]

    def _fold(self, conn, current, current_view, message_hash, claim, cid):
        """The gate's hash and staleness steps for one claim: a verbatim or normalised restatement of a live
        claim becomes a witness edge on it (the row is anchored at the newest witness); a match that was
        corrected, superseded or is no longer eligible is stale, so the claim is new. The live claim's id when
        folded, else False."""
        from colonymind.memory import learning
        whole = learning.verbatim(claim['evidence']) == learning.verbatim(current_view['content'])
        key = fold_key(current['contact_id'], claim['subject_key'], claim['predicate'], claim,
                       claim['valid_from'], claim['valid_to'], whole=whole)
        if key is None:
            return False
        index = self._index(conn, current, subject_key=claim['subject_key'], predicate=claim['predicate'],
                            value_key=norm_value(claim['value']))
        decision = learning.NoveltyGate(use_similarity=False).classify_exact(learning.Item(cid, '', key), index)
        if decision.verdict != 'repeat':
            if decision.step == 'stale':
                learning.log(conn, store="claim", action="labelled", stage="novelty",
                             reason="restates a corrected or superseded value", item_id=decision.match,
                             contact_id=current['contact_id'], refs=[current['turn_id']], decision=decision)
            return False
        match = decision.match
        anchor = conn.execute('SELECT c.*,s.ingested_at FROM source_claims c JOIN turn_sources s '
                              'ON s.turn_id=c.turn_id WHERE c.id=?', (match,)).fetchone()
        here = (current['turn_id'], message_hash)
        if (anchor['turn_id'], anchor['message_hash']) == here or conn.execute(
                'SELECT 1 FROM source_claim_witnesses WHERE claim_id=? AND turn_id=? AND message_hash=?',
                (match, *here)).fetchone():
            return match    # This message already witnesses it (a retried job, a doubled proposal).
        data = dict(claim)
        if data.get('prior_claim_id') == match:
            data.update(operation='assert', prior_claim_id=None)
        encoded = json.dumps(data, ensure_ascii=False)
        if (current['ingested_at'] or '') >= (anchor['ingested_at'] or ''):
            conn.execute('INSERT OR IGNORE INTO source_claim_witnesses(claim_id,turn_id,message_hash,data_json,recorded_at) '
                         'VALUES (?,?,?,?,?)', (match, anchor['turn_id'], anchor['message_hash'], anchor['data_json'],
                                                anchor['ingested_at']))
            conn.execute('UPDATE source_claims SET turn_id=?,message_hash=?,data_json=?,witnesses=witnesses+1 WHERE id=?',
                         (current['turn_id'], message_hash, encoded, match))
        else:
            conn.execute('INSERT INTO source_claim_witnesses(claim_id,turn_id,message_hash,data_json,recorded_at) '
                         'VALUES (?,?,?,?,?)', (match, current['turn_id'], message_hash, encoded, current['ingested_at']))
            conn.execute('UPDATE source_claims SET witnesses=witnesses+1 WHERE id=?', (match,))
        learning.log(conn, store="claim", action="witnessed", stage="novelty", reason="restated",
                     item_id=match, contact_id=current['contact_id'], refs=[current['turn_id']], decision=decision)
        return match

    def decide(self, pending_id, *, ratify, by="owner", reason=""):
        """The owner's verdict on a held preference change: ``ratify`` applies it (the old statement is
        retracted or superseded by the new one), otherwise both statements stay. Returns the pending row with
        ``changed`` (False when it was already decided), or None for an unknown id."""
        from colonymind.memory import learning
        with closing(self.ledger._connect()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute("SELECT * FROM learning_pending WHERE id=?", (pending_id,)).fetchone()
            if row is None or row['status'] != 'pending':
                return dict(row, changed=False) if row else None
            by_row = conn.execute('SELECT data_json,valid_from FROM source_claims WHERE id=?', (row['by_id'],)).fetchone()
            status = 'ratified' if ratify and by_row is not None else 'declined' if not ratify else 'void'
            if status == 'ratified':
                claim = json.loads(by_row['data_json'])
                claim.update(operation=row['operation'], valid_from=by_row['valid_from'])
                # The preference retires, not only the row the extractor named: every live row restating it
                # (written before the rules) goes with it, as on the automatic path.
                source = conn.execute('SELECT s.contact_id,s.session_id FROM source_claims c JOIN turn_sources s '
                                      'ON s.turn_id=c.turn_id WHERE c.id=?', (row['by_id'],)).fetchone()
                duplicates = self._same_value(conn, dict(source), {'id': row['target_id']}) if source else []
                self._retire(conn, [row['target_id'], *(ident for ident in duplicates if ident != row['by_id'])],
                             claim, row['by_id'])
            conn.execute("UPDATE learning_pending SET status=?,decided_at=?,decided_by=?,reason=? WHERE id=?",
                         (status, time.time(), str(by)[:40], str(reason)[:200], pending_id))
            learning.log(conn, store="claim", action="ratified" if status == 'ratified' else "declined",
                         stage="preference_change", reason=reason or status, item_id=row['target_id'],
                         contact_id=row['contact_id'], refs=[row['by_id'], by])
            return dict(conn.execute("SELECT * FROM learning_pending WHERE id=?", (pending_id,)).fetchone(), changed=True)

    def claim_job(self, *, backlog=True):
        """Lease the next due job: a new turn first; backlog and observed conversations only
        when none is due, ``backlog`` is set and the hourly budget has room."""
        from colonymind.turns import projection_backlog
        now = time.time()
        with closing(self.ledger._connect()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            before = projection_backlog.watermark(conn)
            query = '''SELECT j.*,s.contact_id,s.session_id,s.scope,s.messages_json,s.occurred_at,s.ingested_at
                FROM source_claim_jobs j JOIN turn_sources s ON s.turn_id=j.turn_id
                WHERE ((j.status='pending' AND j.next_attempt<=?) OR (j.status='running' AND j.lease_until<=?))
                AND ({}) ORDER BY s.ingested_at'''
            job = next((candidate for candidate in conn.execute(query.format("j.enqueued_at>=? AND j.turn_id NOT LIKE 'observed-conversation:%'"),
                               (now, now, before)) if self._job_current(candidate, conn=conn)), None)
            if job is None and backlog:
                job = next((candidate for candidate in conn.execute(query.format("j.enqueued_at<? OR j.turn_id LIKE 'observed-conversation:%'"),
                                   (now, now, before)) if self._job_current(candidate, conn=conn)), None)
                if job is not None and not projection_backlog.admit(conn, now=now):
                    job = None
            if job is None:
                return None
            token = uuid.uuid4().hex
            conn.execute("UPDATE source_claim_jobs SET status='running',attempts=attempts+1,lease_until=?,lease_token=? WHERE turn_id=?",
                         (now + 60, token, job["turn_id"]))
            return dict(job, lease_token=token)

    def finish_job(self, job, *, model=None, error=None, diagnostics=None):
        # claim_job returns the pre-increment count. Older workers can leave
        # this additive column untouched; never attribute their later attempt
        # to measurements produced by this worker.
        encoded = json.dumps(dict(diagnostics, attempt=job['attempts'] + 1), sort_keys=True) if diagnostics is not None else None
        with closing(self.ledger._connect()) as conn, conn:
            if error:
                delay = min(900, 15 * 2 ** min(job["attempts"], 6))
                conn.execute("UPDATE source_claim_jobs SET status='pending',error=?,next_attempt=?,lease_until=0,diagnostics_json=? WHERE turn_id=? AND lease_token=?",
                             (error, time.time() + delay, encoded, job["turn_id"], job["lease_token"]))
            else:
                done = conn.execute("UPDATE source_claim_jobs SET status='complete',error=NULL,model=?,extraction_version=?,lease_until=0,diagnostics_json=?,next_message=0 WHERE turn_id=? AND lease_token=?",
                                    (model, EXTRACTION_VERSION, encoded, job["turn_id"], job["lease_token"])).rowcount
                if done and diagnostics:
                    # The mandatory rejection log also holds what extraction and review refused.
                    from colonymind.memory import learning
                    for reason, count in sorted((diagnostics.get('rejection_counts') or {}).items()):
                        learning.log(conn, store="claim", action="rejected", stage="extraction",
                                     reason=f"{reason} ({count})", contact_id=job.get('contact_id'), refs=[job["turn_id"]])
                    if diagnostics.get('review_rejected_count'):
                        learning.log(conn, store="claim", action="rejected", stage="review",
                                     reason=f"review refused ({diagnostics['review_rejected_count']})",
                                     contact_id=job.get('contact_id'), refs=[job["turn_id"]])

    def _job_current(self, job, *, conn):
        from colonymind.memory import curated
        from colonymind.turns.idempotency import source_message_hash
        return any(message.get('role') == 'user' and not curated.is_forgotten(self.ledger,
            {'turn_id': job['turn_id'], 'message_hash': source_message_hash(job['session_id'], message)}, conn=conn)
            for message in json.loads(job['messages_json'])[job['next_message']:])

    def renew_job(self, job, request_timeout, *, message_hash=None):
        """Extend this lease for bounded extraction, review and their commit."""
        from colonymind.memory import curated
        ref = {'turn_id': job['turn_id'], 'message_hash': message_hash} if message_hash else job['turn_id']
        if curated.is_forgotten(self.ledger, ref):
            return False
        with closing(self.ledger._connect()) as conn, conn:
            updated = conn.execute('''UPDATE source_claim_jobs SET lease_until=?
                WHERE turn_id=? AND status='running' AND lease_token=?''',
                (time.time() + request_timeout + 30, job['turn_id'], job['lease_token']))
            return updated.rowcount == 1

    async def process_one(self, router, *, backlog=True):
        from colonymind.turns.idempotency import source_message_hash
        job = self.claim_job(backlog=backlog)
        if job is None:
            return False
        model = None
        diagnostics = extraction_diagnostics()
        try:
            from colonymind.turns.observed import is_observed_source
            observed = is_observed_source(job['turn_id'])
            messages = json.loads(job["messages_json"])
            for index, message in enumerate(messages):
                if observed and index < job['next_message']:
                    continue
                from colonymind.memory import curated
                message_hash = source_message_hash(job['session_id'], message)
                if curated.is_forgotten(self.ledger, {'turn_id': job['turn_id'], 'message_hash': message_hash}):
                    continue
                # Retained ASR has a deterministic derived view. Other media
                # remains in its existing caption/source recall path.
                from colonymind.turns.audio import claim_message
                message = claim_message(message)
                if message is None:
                    continue
                content = message.get('content')
                if message.get("role") != "user" or not isinstance(content, str) or not content.strip():
                    continue
                request_timeout = projection_timeout_seconds(router)
                if not self.renew_job(job, request_timeout, message_hash=message_hash):
                    return True  # Forgotten or reclaimed; do not start a stale model call.
                prior = await self.prior_claims(job, message)
                claims, model = await extract_claims(router, job, message, prior,
                                                    timezone_name=job["timezone"], request_timeout=request_timeout,
                                                    diagnostics=diagnostics, known_keys=self.known_keys(job),
                                                    earlier=self.earlier_messages(job, message))
                if model == "local_extraction_role_unavailable":
                    self.finish_job(job, error=model, diagnostics=diagnostics)
                    return True
                self.commit(job, message, claims, model=model, lease_token=job["lease_token"])
                if observed and index + 1 < len(messages):
                    # One window per backlog admission. A failed next window resumes here, even when this
                    # window formed no claims. The lease token keeps an erased or reclaimed job untouched.
                    with closing(self.ledger._connect()) as conn, conn:
                        conn.execute("UPDATE source_claim_jobs SET status='pending',next_message=?,lease_until=0,"
                                     "error=NULL WHERE turn_id=? AND status='running' AND lease_token=?",
                                     (index + 1, job['turn_id'], job['lease_token']))
                    return True
            self.finish_job(job, model=model, diagnostics=diagnostics)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.finish_job(job, error=type(exc).__name__, diagnostics=diagnostics)
            logger.warning("source claim projection deferred (%s)", type(exc).__name__)
        return True

    def status(self, contact_id):
        from colonymind.memory import curated
        return curated.fenced_read(self.ledger, lambda: self._read_status(contact_id),
                                  empty=[])

    def _read_status(self, contact_id):
        with closing(self.ledger._connect()) as conn:
            rows = [dict(row) for row in conn.execute('''SELECT j.turn_id,j.status,j.attempts,j.error,j.model,j.extraction_version,j.diagnostics_json,
                (SELECT count(*) FROM source_claims c WHERE c.turn_id=j.turn_id)
                + (SELECT count(*) FROM source_claim_witnesses w WHERE w.turn_id=j.turn_id) AS claim_count
                FROM source_claim_jobs j JOIN turn_sources s ON s.turn_id=j.turn_id WHERE s.contact_id=?
                ORDER BY s.ingested_at DESC LIMIT 20''', (contact_id,))]
        from colonymind.memory import curated
        rows = [row for row in rows if not curated.is_forgotten(self.ledger, row['turn_id'])]
        for row in rows:
            encoded = row.pop('diagnostics_json')
            data = json.loads(encoded) if encoded is not None else None
            row['diagnostics'] = data if data is not None and data['attempt'] == row['attempts'] else None
        return rows

    def preferences(self, contact_id, session_id='', *, now=None, limit=20):
        """Read admitted speaker preferences without authoring another profile.

        Keep the full quotation, including a recurring activity or condition.
        Explicit date bounds restrict eligibility; quoted applicability remains
        unresolved so consumers can use the statement without assuming it applies.
        A correction makes an interpretation unsuitable as automatic guidance;
        ordinary source recall still presents the original and its annotation.
        """
        from colonymind.memory import curated
        return curated.fenced_read(self.ledger, lambda: self._read_preferences(contact_id, session_id, now=now, limit=limit),
                                  empty=[])

    def _read_preferences(self, contact_id, session_id='', *, now=None, limit=20):
        from datetime import datetime, timezone
        from colonymind.turns.idempotency import canonical_turn_digest
        stamp = datetime.fromtimestamp(time.time() if now is None else now, timezone.utc).isoformat()
        with closing(self.ledger._connect()) as conn:
            ids = [r[0] for r in conn.execute('''SELECT c.id FROM source_claims c
                JOIN turn_sources s ON s.turn_id=c.turn_id
                WHERE s.contact_id=? AND (s.scope='person' OR s.session_id=?)
                AND c.subject_key='speaker' AND (c.superseded_by IS NULL OR c.valid_to>?) AND c.retracted_by IS NULL
                AND json_extract(c.data_json,'$.memory_quality.memory_kind')='preference'
                AND json_extract(c.data_json,'$.admission_review.version')='source-claim-review-v1'
                AND json_extract(c.data_json,'$.admission_review.basis')='model_judgment_unverified'
                AND (c.valid_from IS NULL OR c.valid_from<=?) AND (c.valid_to IS NULL OR c.valid_to>?)
                AND NOT EXISTS (SELECT 1 FROM source_projection_erasures e WHERE e.turn_id=s.turn_id)
                AND NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=s.turn_id)
                ORDER BY s.ingested_at DESC,c.id LIMIT ?''',
                (contact_id, session_id, stamp, stamp, stamp, max(1, min(limit, 100))))]
            witnessed = has_witnesses(conn)
            if witnessed:
                # A folded preference whose newest statement was annotated or lost its projections still
                # stands on an earlier verbatim statement, as the separate rows of an unfolded store did.
                ids += [r[0] for r in conn.execute('''SELECT c.id FROM source_claims c
                    JOIN turn_sources s ON s.turn_id=c.turn_id
                    WHERE s.contact_id=? AND (s.scope='person' OR s.session_id=?) AND c.witnesses>1
                    AND c.subject_key='speaker' AND (c.superseded_by IS NULL OR c.valid_to>?) AND c.retracted_by IS NULL
                    AND json_extract(c.data_json,'$.memory_quality.memory_kind')='preference'
                    AND json_extract(c.data_json,'$.admission_review.version')='source-claim-review-v1'
                    AND json_extract(c.data_json,'$.admission_review.basis')='model_judgment_unverified'
                    AND (c.valid_from IS NULL OR c.valid_from<=?) AND (c.valid_to IS NULL OR c.valid_to>?)
                    AND EXISTS (SELECT 1 FROM source_projection_erasures e WHERE e.turn_id=s.turn_id)
                    ORDER BY s.ingested_at DESC,c.id LIMIT ?''',
                    (contact_id, session_id, stamp, stamp, stamp, max(1, min(limit, 100))))]

            def annotated(turn_id, message_hash):
                return any(message_hash in json.loads(r[0]) for r in conn.execute(
                    'SELECT target_message_hashes_json FROM source_annotations WHERE target_source_id=?', (turn_id,)))

            def eligible_view(claim):
                if not annotated(claim['turn_id'], claim['message_hash']) and not conn.execute(
                        'SELECT 1 FROM source_projection_erasures WHERE turn_id=?', (claim['turn_id'],)).fetchone():
                    return claim
                if not witnessed:
                    return None
                from colonymind.turns.idempotency import source_message_hash
                for edge in conn.execute('''SELECT w.*,s.session_id,s.messages_json FROM source_claim_witnesses w
                        JOIN turn_sources s ON s.turn_id=w.turn_id WHERE w.claim_id=? AND s.contact_id=?
                        AND (s.scope='person' OR s.session_id=?)
                        AND NOT EXISTS (SELECT 1 FROM source_projection_erasures e WHERE e.turn_id=s.turn_id)
                        AND NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=s.turn_id)
                        ORDER BY w.recorded_at DESC''', (claim['id'], contact_id, session_id)).fetchall():
                    if (edge['message_hash'] in {source_message_hash(edge['session_id'], m)
                                                 for m in json.loads(edge['messages_json'])}
                            and not annotated(edge['turn_id'], edge['message_hash'])):
                        return {**claim, **json.loads(edge['data_json']), 'id': claim['id'],
                                'turn_id': edge['turn_id'], 'message_hash': edge['message_hash']}
                return None
            result = []
            for claim in self._rows(conn, contact_id, session_id, ids=list(dict.fromkeys(ids)), limit=len(ids) or 1):
                claim = eligible_view(claim)
                if claim is None:
                    continue
                source = conn.execute('SELECT session_id,messages_json FROM turn_sources WHERE turn_id=?', (claim['turn_id'],)).fetchone()
                refs = [{
                    'source_id': claim['turn_id'], 'source_contact_id': contact_id,
                    'source_version': canonical_turn_digest(json.loads(source['messages_json'])),
                    'message_hash': claim['message_hash']}]
                if claim.get('subject_basis'):
                    basis = claim['subject_basis']
                    refs.append({'source_id': basis['turn_id'], 'source_contact_id': contact_id,
                                 'source_version': basis['source_version'], 'message_hash': basis['message_hash']})
                refs.extend({'source_id': b['turn_id'], 'source_contact_id': contact_id,
                             'source_version': b['source_version'], 'message_hash': b['message_hash']}
                            for b in _value_sources(claim))
                from colonymind.memory import curated
                if curated.is_forgotten(self.ledger, refs, conn=conn):
                    continue
                result.append({**claim, 'session_id': source['session_id'], 'sources': refs})
        return result

    def _direct_input(self, conn, source, message, *, contact_id, session_id):
        """Resolve one exact conversational input, not a factual support claim.

        A native answer is often stored separately from its supplied user input.
        Unknown, corrected, changed, multiple-input and cross-scope provenance
        remains recallable. This transient hint never changes stored history.
        """
        from colonymind.turns.idempotency import canonical_turn_digest, source_message_hash
        from colonymind.self_model.task_assessments import quotation_metadata
        from colonymind.memory import curated
        if (message.get('role') != 'assistant' or source['scope'] != 'person'
                or quotation_metadata(message) or curated.is_forgotten(conn, {
                    'source_id': source['turn_id'], 'message_hash': source_message_hash(source['session_id'], message)})):
            return False
        refs = message.get('_supplied_inputs')
        if refs is not None:
            if not isinstance(refs, list) or len(refs) != 1 or not isinstance(refs[0], dict):
                return False
            ref = refs[0]
            if set(ref) != {'source_id', 'input_message_hash'}:
                return False
            parent = conn.execute('''SELECT * FROM turn_sources WHERE turn_id=? AND contact_id=?
                AND (scope='person' OR session_id=?) AND NOT EXISTS (
                    SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=turn_sources.turn_id)
                ''', (ref['source_id'], contact_id, session_id)).fetchone()
            if parent is None or parent['scope'] != 'person':
                return False
            messages = json.loads(parent['messages_json'])
            if {'source_id': parent['turn_id'], 'source_version': canonical_turn_digest(messages)}\
                    not in message.get('_supplied_sources', []):
                return False
            matches = [m for m in messages if m.get('role') == 'user'
                       and source_message_hash(parent['session_id'], m) == ref['input_message_hash']]
            if len(matches) != 1:
                return False
            request = matches[0]
        else:
            # Ordinary paired turns can carry the request in the same envelope.
            messages = json.loads(source['messages_json'])
            indices = [i for i, m in enumerate(messages) if m == message]
            if len(indices) != 1 or indices[0] == 0:
                return False
            parent, request = source, messages[indices[0] - 1]
            if request.get('role') != 'user':
                return False
        return parent, request

    def _direct_reply(self, conn, source, message, *, contact_id, session_id):
        """The one reply whose direct input is the user ``message`` (``_direct_input`` read the other way), with the
        reply's source; False when there is none, or more than one.

        The reply map (``turns.source_replies``) proposes the replies that name the message. Each is read in the
        viewer's scope with its source's attribution intact, and kept only when ``_direct_input`` of it gives back
        this very message, so the forward reading's checks all hold. Who answered what is never guessed: no reply,
        or several, is no pair. An annotated reply is not attached: a corrected answer is shown when it is found
        itself, with its correction, not brought beside a question as its answer.
        """
        from colonymind.turns import source_replies
        from colonymind.turns.idempotency import source_message_hash
        if message.get('role') != 'user' or source['scope'] != 'person':
            return False
        identity = source['turn_id'], source_message_hash(source['session_id'], message)
        found = {}
        for reply_turn, reply_hash in source_replies.replies(conn, *identity):
            reply_source = conn.execute('''SELECT * FROM turn_sources WHERE turn_id=? AND contact_id=?
                AND (scope='person' OR session_id=?) AND NOT EXISTS (
                    SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=turn_sources.turn_id)''',
                (reply_turn, contact_id, session_id)).fetchone()
            if reply_source is None:
                continue
            reply_source = dict(reply_source)
            matches = [m for m in json.loads(reply_source['messages_json'])
                       if source_message_hash(reply_source['session_id'], m) == reply_hash]
            if len(matches) != 1:
                continue
            pair = self._direct_input(conn, reply_source, matches[0], contact_id=contact_id, session_id=session_id)
            if not pair or (pair[0]['turn_id'], source_message_hash(pair[0]['session_id'], pair[1])) != identity:
                continue
            found[(reply_turn, reply_hash)] = reply_source, matches[0]
        if len(found) != 1:
            return False
        (reply_turn, reply_hash), reply = next(iter(found.items()))
        if any(reply_hash in json.loads(row[0]) for row in conn.execute(
                'SELECT target_message_hashes_json FROM source_annotations WHERE target_source_id=?', (reply_turn,))):
            return False
        return reply

    def _current_work_reply(self, conn, source, message, *, contact_id, session_id):
        from colonymind.memory.selection import current_work_query
        from colonymind.turns.idempotency import source_message_hash
        pair = self._direct_input(conn, source, message, contact_id=contact_id, session_id=session_id)
        if not pair:
            return False
        parent, request = pair
        request_hash = source_message_hash(parent['session_id'], request)
        annotations = conn.execute('SELECT target_message_hashes_json FROM source_annotations WHERE target_source_id=?',
                                   (parent['turn_id'],)).fetchall()
        if any(request_hash in json.loads(row[0]) for row in annotations):
            return False
        return current_work_query(request.get('content'))

    @staticmethod
    def _at_message(conn, rows, identity):
        """Each claim as the message ``identity`` states it: a folded claim anchored elsewhere reads its
        witness edge here (that message's quotation and span) with the claim's id and lifecycle."""
        result = []
        for row in rows:
            if (row['turn_id'], row['message_hash']) != tuple(identity):
                edge = conn.execute('SELECT data_json FROM source_claim_witnesses WHERE claim_id=? AND turn_id=? '
                                    'AND message_hash=?', (row['id'], *identity)).fetchone()
                if edge is not None:
                    row = {**row, **json.loads(edge['data_json']), **{key: row[key] for key in (
                        'id', 'subject_key', 'predicate', 'valid_from', 'valid_to', 'superseded_by', 'retracted_by')},
                        'turn_id': identity[0], 'message_hash': identity[1]}
            result.append(row)
        return result

    def prepare_context(self, beliefs, source_hits, *, contact_id, session_id, time_query: MemoryTimeQuery, classify_work_replies=False, include_conversation_inputs=False):
        """Expand retrieved keys into complete scoped assertion bundles.

        Ranking chooses relevant keys. It cannot choose a winner within an
        unresolved conflict or reintroduce text from a corrected assertion.
        """
        from colonymind.memory import curated
        return curated.fenced_read(self.ledger, lambda: self._read_prepare_context(beliefs, source_hits, contact_id=contact_id, session_id=session_id, time_query=time_query, classify_work_replies=classify_work_replies, include_conversation_inputs=include_conversation_inputs),
                                  empty=([], []))

    def _read_prepare_context(self, beliefs, source_hits, *, contact_id, session_id, time_query: MemoryTimeQuery,
                        classify_work_replies=False, include_conversation_inputs=False):
        from colonymind.turns.idempotency import source_message_hash
        from colonymind.turns.audio import source_text, evidence_metadata
        from colonymind.self_model.task_assessments import quotation_metadata
        from colonymind.memory.recall import (
            collapse_checkpoint_copies, drop_checkpoint_copies, pair_conversation_candidates, source_candidates)
        from colonymind.turns.source_annotations import correction_membership, expand as expand_annotations
        source_hits = list(source_hits)
        input_pairs = {}
        turn_ids = list(dict.fromkeys(
            [row["turn_id"] for row in source_hits] + [str(row["source_uri"])[5:] for row in beliefs
             if str(row.get("source_uri") or "").startswith("turn:")]))
        if not turn_ids:
            return filter_unstructured(beliefs, time_query), []
        with closing(self.ledger._connect()) as conn:
            sources = {row["turn_id"]: dict(row) for row in conn.execute(
                # ``+contact_id``: read by turn id, not by walking every source of the person. Same rows.
                "SELECT * FROM turn_sources WHERE +contact_id=? AND (scope='person' OR session_id=?) AND turn_id IN ("
                + ",".join("?" for _ in turn_ids) + ")", (contact_id, session_id, *turn_ids))}
            if include_conversation_inputs:
                # Expand one exact user input only. Supplied memories and tool
                # observations are dependencies, not conversational parents.
                # A user message found alone brings its single direct reply the
                # same way (``_direct_reply``), unless a lane found the reply too.
                seen = {(hit['turn_id'], hit.get('source_message_hash')) for hit in source_hits}
                for hit in tuple(source_hits):
                    source = sources.get(hit['turn_id'])
                    if not source or hit.get('role') not in ('assistant', 'user') or hit.get('excerpt_truncated'):
                        continue
                    matches = [m for m in json.loads(source['messages_json'])
                        if source_message_hash(source['session_id'], m) == hit.get('source_message_hash')]
                    if len(matches) != 1 or matches[0].get('content') != hit.get('content'):
                        continue
                    if hit['role'] == 'user':
                        reply = self._direct_reply(conn, source, matches[0], contact_id=contact_id,
                                                   session_id=session_id)
                        if not reply:
                            continue
                        reply_source, response = reply
                        reply_key = reply_source['turn_id'], source_message_hash(reply_source['session_id'], response)
                        # A reply a lane found is paired from its own hit, or left as found.
                        if (reply_key in seen or not isinstance(response.get('content'), str)
                                or not response['content'].strip() or evidence_metadata(response)):
                            continue
                        seen.add(reply_key)
                        input_pairs[reply_key] = hit['turn_id'], hit['source_message_hash']
                        sources[reply_source['turn_id']] = reply_source
                        source_hits.append({**{k: reply_source[k] for k in (
                            'turn_id', 'contact_id', 'session_id', 'scope', 'occurred_at', 'ingested_at')},
                            'role': 'assistant', 'content': response['content'], 'source_message_hash': reply_key[1],
                            'retrieval_method': 'conversation_reply'})
                        continue
                    pair = self._direct_input(conn, source, matches[0], contact_id=contact_id, session_id=session_id)
                    if not pair:
                        continue
                    parent, request = pair
                    # Media/transcript evidence keeps its existing representation.
                    if not isinstance(request.get('content'), str) or not request['content'].strip() or evidence_metadata(request):
                        continue
                    parent_key = parent['turn_id'], source_message_hash(parent['session_id'], request)
                    input_pairs[(hit['turn_id'], hit['source_message_hash'])] = parent_key
                    sources[parent['turn_id']] = dict(parent)
                    if parent_key not in seen:
                        seen.add(parent_key)
                        source_hits.append({**{k: parent[k] for k in (
                            'turn_id', 'contact_id', 'session_id', 'scope', 'occurred_at', 'ingested_at')},
                            'role': 'user', 'content': request['content'], 'source_message_hash': parent_key[1],
                            'retrieval_method': 'conversation_input'})
                turn_ids = list(sources)
            hashes = {source_message_hash(source["session_id"], message)
                      for source in sources.values() for message in json.loads(source["messages_json"])
                      if any(
                          hit["turn_id"] == source["turn_id"] and hit["role"] == message.get("role")
                          and hit["content"] in source_text(message.get('content')) for hit in source_hits)}
            claims = self._rows(conn, contact_id, session_id, turn_ids=turn_ids, message_hashes=hashes, limit=512)
            status_replies = set()
            if classify_work_replies:
                # Resolve only exact retrieved messages, never every assistant
                # message in an otherwise long canonical checkpoint.
                pending = {(hit['turn_id'], hit.get('source_message_hash')) for hit in source_hits}
                for source in sources.values():
                    for message in json.loads(source['messages_json']):
                        identity = (source['turn_id'], source_message_hash(source['session_id'], message))
                        if identity not in pending:
                            continue
                        pending.discard(identity)
                        if self._current_work_reply(conn, source, message,
                                                    contact_id=contact_id, session_id=session_id):
                            status_replies.add(identity)
        by_turn, by_hash = {}, {}
        for claim in claims:
            by_turn.setdefault(claim["turn_id"], []).append(claim)
            by_hash.setdefault(claim["message_hash"], []).append(claim)
        keys, retained_beliefs = [], []
        for original in beliefs:
            row = dict(original)
            uri = str(row.get("source_uri") or "")
            turn = uri[5:] if uri.startswith("turn:") else None
            known = by_turn.get(turn, [])
            if known:
                keys.extend((c["subject_key"], c["predicate"]) for c in known)
                continue
            if turn in sources:
                row["occurred_at"] = sources[turn]["occurred_at"]
            retained_beliefs.append(row)
        retained_hits = []
        for original in source_hits:
            hit = dict(original)
            hit.pop('_current_work_status_reply', None)
            hit.pop('assessment_context', None)
            source = sources.get(hit["turn_id"])
            removed = []
            if source:
                # Carry canonical attribution through selection. Checkpoint
                # scope cannot attest the speaker of every historical message.
                hit.update({name: source[name] for name in ("contact_id", "session_id", "scope")})
                for message in json.loads(source["messages_json"]):
                    if hit.get('source_message_hash') and hit['source_message_hash'] != source_message_hash(source['session_id'], message):
                        continue
                    text = message.get("content")
                    if isinstance(text, list):
                        text = source_text(text)
                    if message.get("role") != hit["role"] or not isinstance(text, str):
                        continue
                    if ((source['turn_id'], source_message_hash(source['session_id'], message)) in status_replies
                            and hit.get('source_message_hash') == source_message_hash(source['session_id'], message)):
                        hit['_current_work_status_reply'] = True
                    message_claims = by_hash.get(source_message_hash(source["session_id"], message), [])
                    # Source FTS chunks overlap. Clip exact message spans into
                    # every matching chunk occurrence; never depend on an entire
                    # corrected quotation fitting inside one retrieved chunk.
                    offset = text.find(hit["content"])
                    if offset >= 0:
                        # The canonical admission marker owns this relation;
                        # index hints and words inside a quote cannot mint it.
                        hit.update(quotation_metadata(message))
                    if offset >= 0 and hit["content"] != text:
                        hit["excerpt_truncated"] = True
                    while offset >= 0:
                        procedures = [c for c in message_claims
                            if c.get('memory_quality', {}).get('memory_kind') == 'procedure']
                        if procedures:
                            # Its unclaimed remainder may also contain a
                            # required condition. Select the owning message,
                            # never an independently ranked leftover fragment.
                            keys.extend((c['subject_key'], c['predicate']) for c in procedures)
                            removed.append((0, len(hit['content'])))
                        for claim in message_claims:
                            start = max(0, claim["span_start"] - offset)
                            end = min(len(hit["content"]), claim["span_end"] - offset)
                            if start < end:
                                keys.append((claim["subject_key"], claim["predicate"]))
                                removed.append((start, end))
                        offset = text.find(hit["content"], offset + 1)
            cursor = 0
            # Each remainder is still one exact contiguous source quotation.
            # Concatenating separated fragments would fabricate a quotation.
            for start, end in sorted(removed) + [(len(hit["content"]), len(hit["content"]))]:
                if start > cursor:
                    fragment = hit["content"][cursor:start]
                    if fragment.strip(" .,:;\n\t"):
                        retained_hits.append(dict(hit, content=fragment,
                            excerpt_truncated=bool(removed) or bool(hit.get("excerpt_truncated"))))
                cursor = max(cursor, end)
        bundles, message_contexts, emitted_messages, groups = [], {}, set(), {}
        unresolved_time_keys = set()
        anchor_sessions = {turn: source['session_id'] for turn, source in sources.items()}

        def anchor_session(turn_id):
            """The session of the source a bundle is anchored in. A bundle is grouped by its key, not by a
            conversation; the rerank spread (``memory.selection.spread_across_sessions``) reads this session, so
            the many properties one conversation states count as that conversation."""
            if turn_id not in anchor_sessions:
                with closing(self.ledger._connect()) as conn:
                    row = conn.execute('SELECT session_id FROM turn_sources WHERE turn_id=?', (turn_id,)).fetchone()
                anchor_sessions[turn_id] = row[0] if row else None
            return anchor_sessions[turn_id]

        def current_group(key):
            if key not in groups:
                with closing(self.ledger._connect()) as conn:
                    groups[key] = self._rows(conn, contact_id, session_id, key=key,
                        time_query=time_query, distinct_values=True, limit=9)
                    window = utc_timestamp(time_query.start), utc_timestamp(time_query.end)

                    def wider(claim):
                        interval = event_interval(claim.get('event_time'))
                        return interval is not None and (
                            (window[0] is not None and interval[0] < window[0])
                            or (window[1] is not None and interval[1] > window[1]))
                    if time_query.mode == 'observed_range' and any(wider(c) for c in groups[key]):
                        # The event could fall in the overlapping part or the
                        # remainder of its source day or period. Relevance is not
                        # proof that it happened within the query's narrower window.
                        unresolved_time_keys.add(key)
                    if not groups[key] and time_query.mode == 'observed_range':
                        # Relevance already selected this source. An unknown
                        # episode time may help answer a date question, but it
                        # cannot certify a matching date. Never admit a known
                        # event outside the requested interval through here,
                        # including one known only to a period.
                        groups[key] = [c for c in self._rows(conn, contact_id, session_id, key=key,
                            time_query=MemoryTimeQuery(), distinct_values=True, limit=9)
                            if (c.get('representation') == 'episode' or quoted_applicability(c) is not None)
                            and not c.get('event_at') and event_interval(c.get('event_time')) is None]
                        if groups[key]:
                            unresolved_time_keys.add(key)
            return groups[key]

        def complete_message_context(claim):
            """Keep a current short message or a procedure's conditions together.

            Keep the owning message together when it is still current. A
            changed message needs the existing source/history readers instead
            of reviving obsolete instructions from its original quotation.
            """
            procedure = claim.get('memory_quality', {}).get('memory_kind') == 'procedure'
            identity = (claim['turn_id'], claim['message_hash'])
            if identity not in message_contexts:
                with closing(self.ledger._connect()) as conn:
                    source = conn.execute('''SELECT * FROM turn_sources WHERE turn_id=? AND contact_id=?
                        AND (scope='person' OR session_id=?)''',
                        (claim['turn_id'], contact_id, session_id)).fetchone()
                    message = next((m for m in json.loads(source['messages_json'])
                        if source_message_hash(source['session_id'], m) == claim['message_hash']), None) if source else None
                    ids = [row['id'] for row in conn.execute(
                        'SELECT id FROM source_claims WHERE turn_id=? AND message_hash=? UNION SELECT claim_id '
                        'FROM source_claim_witnesses WHERE turn_id=? AND message_hash=? LIMIT 513', identity * 2)]
                    all_claims = self._at_message(conn, self._rows(conn, contact_id, session_id, ids=ids, limit=513),
                                                  identity)
                    current = self._rows(conn, contact_id, session_id, ids=ids,
                        time_query=time_query, limit=513)
                text = source_text(message.get('content')) if message else ''
                complete = bool(text and all_claims and len(ids) < 513 and len(all_claims) == len(ids)
                    and {c['id'] for c in all_claims} == {c['id'] for c in current}
                    and not any(c['superseded_by'] or c['retracted_by'] for c in all_claims)
                    and all(len(current_group((c['subject_key'], c['predicate']))) == 1 for c in all_claims))
                message_contexts[identity] = {'source': dict(source) if source else None,
                    'message': message, 'text': text, 'complete': complete, 'claims': all_claims}
            value = message_contexts[identity]
            if not procedure:
                # An unresolved time expression cannot certify a dated fact,
                # but must not strip conditions from its attributed source.
                # The bundle retains query_time_unresolved in that case.
                # Resolved historical windows still use qualified assertions.
                if (not value['complete'] or len(value['text']) > 2000
                        or time_query.mode not in {'current', 'unresolved_time'}
                        # Derived media needs its exact segment/recognizer basis;
                        # a transcript's display prefix is not missing context.
                        or any(c.get('evidence_basis') for c in value['claims'])
                        or all(c['evidence'].strip() == value['text'].strip() for c in value['claims'])):
                    return None
                return value
            # A complete one-message assertion keeps its existing property
            # semantics. This boundary concerns partial-message procedures.
            if value['complete'] and claim['evidence'].strip() == value['text'].strip():
                return None
            return value

        for key in dict.fromkeys(keys):
            group = current_group(key)
            if not group:
                continue
            overflow = len(group) > 8
            group.sort(key=lambda c: (c["valid_from"] or "", c["recorded_at"], c["id"]))
            # Literal scalar values retain their existing comparison. Quoted
            # preferences cannot establish agreement or contradiction through
            # wording equality/difference, including beside compact values.
            scalars = [c for c in group if c.get('representation') != 'preference']
            quoted_preferences = len(scalars) != len(group)
            quoted_statements = any(quoted_applicability(c) is not None for c in group)
            values = {norm_value(c["value"]) for c in scalars}
            def overlaps(a, b):
                return (not a["valid_to"] or not b["valid_from"] or b["valid_from"] < a["valid_to"]) and (
                    not b["valid_to"] or not a["valid_from"] or a["valid_from"] < b["valid_to"])
            conflict = any(norm_value(a["value"]) != norm_value(b["value"]) and overlaps(a, b)
                           for i, a in enumerate(scalars) for b in scalars[i + 1:])
            members = [{"claim_id": c["id"], "source": "turn:" + c["turn_id"],
                        "source_message_hash": c["message_hash"], "role": c["role"],
                        "value": c["value"], "quote": c["evidence"], "observed_at": c["observed_at"],
                        "recorded_at": c["recorded_at"], "valid_from": c["valid_from"], "valid_to": c["valid_to"],
                        "event_at": c.get("event_at"), "event_time": c.get("event_time", {
                            "status": "legacy_precision_unknown" if c.get("event_at") else "unknown"}),
                        "reported_at": c["observed_at"], "validity_basis": c["validity_basis"],
                        "operation": c["operation"], "prior_claim_id": c.get("prior_claim_id"),
                        **{k: c[k] for k in ('representation', 'applicability', 'evidence_basis', 'epistemic_state', 'source_modality', 'subject_basis', 'value_basis') if k in c}}
                       for c in group]
            status = ("unresolved_conflict" if conflict else "quoted_preference_statements" if quoted_preferences
                      else "temporal_history" if len(values) > 1 else "source_assertion")
            source_refs = [(c['turn_id'], c['message_hash']) for c in group]
            source_refs.extend((c['subject_basis']['turn_id'], c['subject_basis']['message_hash'])
                               for c in group if c.get('subject_basis'))
            source_refs.extend((b['turn_id'], b['message_hash']) for c in group for b in _value_sources(c))
            identifier = hashlib.sha256(json.dumps([contact_id, key, [c["id"] for c in group]]).encode()).hexdigest()
            bundle = {"id": "assertions:" + identifier, "kind": "source_quote",
                            "history_anchor": {"source_id": group[0]['turn_id'], "claim_id": group[0]['id']},
                            "source_turn_ids": list(dict.fromkeys(turn for turn, _ in source_refs)),
                            "_source_message_hashes": {turn: list(dict.fromkeys(
                                message for source, message in source_refs if source == turn))
                                for turn, _ in source_refs},
                            "source_uri": "turn:" + group[0]["turn_id"], "claim_status": status,
                            **({"session_id": anchor_session(group[0]["turn_id"])}
                               if anchor_session(group[0]["turn_id"]) is not None else {}),
                            "epistemic_state": ('derived_unverified' if any(c.get('evidence_basis') for c in group) else status),
                            **({'source_modality': _derived_modality(group)}
                               if any(c.get('evidence_basis') for c in group) else {}),
                            "atomic_evidence": True,
                            "content_format": "source_assertions_v1",
                            # Rank the grounded language the user supplied.
                            # Administrative IDs/timestamps in the output JSON
                            # are provenance, not the passage's semantic topic.
                            # Conflicting peers remain one indivisible candidate.
                            "ranking_text": "\n".join(dict.fromkeys(c["evidence"] for c in group)),
                            **({"validity_status": "unknown"} if quoted_statements else
                               {"validity_status": "query_time_unresolved"}
                               if time_query.mode == "unresolved_time" or key in unresolved_time_keys else {}),
                            "contradiction_count": len(values) - 1 if conflict else None if quoted_preferences else 0,
                            "relevance": 1 / (61 + len(bundles)),
                            "content": json.dumps({"subject": group[0]["subject"], "predicate": key[1],
                                                   "status": status, "assertions": members,
                                                   **({'comparison_basis': 'quoted_preferences_not_compared'}
                                                      if quoted_preferences else {})}, ensure_ascii=False),
                            **({"excerpt_truncated": True, "content": json.dumps({
                                "subject": group[0]['subject'], "predicate": key[1],
                                "status": "incomplete_assertion_history",
                                **({'statements_at_least': len(group)} if quoted_preferences else
                                   {'distinct_values_at_least': len(values)}),
                                "instruction": "Open assertion history before resolving this property; no value selected."},
                                ensure_ascii=False)} if overflow else {})}
            contexts = [(claim, complete_message_context(claim)) for claim in group]
            contexts = [(claim, context) for claim, context in contexts if context is not None]
            if contexts:
                bundle['source_anchors'] = [{'source_id': turn} for turn in
                    dict.fromkeys(claim['turn_id'] for claim in group)]
                if len(group) == 1 and contexts[0][1]['complete']:
                    claim, context = contexts[0]
                    identity = (claim['turn_id'], claim['message_hash'])
                    if identity in emitted_messages:
                        continue
                    emitted_messages.add(identity)
                    source, message = context['source'], context['message']
                    procedure = claim.get('memory_quality', {}).get('memory_kind') == 'procedure'
                    bundle.update(id=('procedure-source:' if procedure else 'message-source:') + hashlib.sha256(json.dumps(identity).encode()).hexdigest(),
                        content=context['text'], ranking_text=context['text'],
                        **({'procedure_context': 'complete_source_message_text'} if procedure else {
                            'source_context': 'complete_source_message_text',
                            'source_history_anchors': [{'source_id': c['turn_id'], 'claim_id': c['id']}
                                for c in context['claims']]}),
                        claim_status='source_procedure_context' if procedure else 'source_message_context',
                        epistemic_state='derived_unverified' if claim.get('evidence_basis') else 'quotation',
                        source_turn_id=claim['turn_id'], source_message_hash=claim['message_hash'],
                        role=message['role'], contact_id=source['contact_id'], session_id=source['session_id'],
                        scope=source['scope'], occurred_at=source['occurred_at'], ingested_at=source['ingested_at'])
                    # Content now holds raw source text, which can itself look
                    # like JSON. It is no longer an internal assertion card.
                    bundle.pop('content_format', None)
                    derived = [{'claim_id': c['id'], 'evidence_basis': c['evidence_basis']}
                               for c in context['claims'] if c.get('evidence_basis')]
                    if derived:
                        bundle.update(source_evidence_bases=derived, epistemic_state='derived_unverified',
                                      source_modality=_derived_modality(context['claims']))
                    # Other represented properties can inherit an exact subject
                    # from another source. Retain every opening dependency.
                    for represented in context['claims']:
                        bases = ([represented['subject_basis']] if represented.get('subject_basis') else [])
                        for basis in [*bases, *_value_sources(represented)]:
                            if basis['turn_id'] not in bundle['source_turn_ids']:
                                bundle['source_turn_ids'].append(basis['turn_id'])
                            hashes = bundle['_source_message_hashes'].setdefault(basis['turn_id'], [])
                            if basis['message_hash'] not in hashes:
                                hashes.append(basis['message_hash'])
                    bundle['source_anchors'] = [{'source_id': turn} for turn in bundle['source_turn_ids']]
                else:
                    # A conflicting or changed procedure remains discoverable,
                    # but its partial assertion is not sufficient instructions.
                    # Opening only the selected property's history could miss
                    # the sibling whose correction made this message unsafe.
                    anchors = {(c['turn_id'], c['subject_key'], c['predicate']): {
                        'source_id': c['turn_id'], 'claim_id': c['id']}
                        for _, context in contexts for c in context['claims']}
                    bundle.update(procedure_context='full_source_required', excerpt_truncated=True,
                        procedure_history_anchors=list(anchors.values()),
                        content='Incomplete procedure context. Open the full sources in source_anchors '
                            'using their recalled source versions and inspect each procedure_history_anchors entry before '
                            'following the procedure. Property history alone may omit its conditions.')
            bundles.append(bundle)
        # A checkpoint's copy of a message goes with the person-scope candidate that has the same words, wherever
        # that goes, so it is judged before this filter too.
        originals = [*bundles, *source_candidates([hit for hit in retained_hits if hit.get('scope') == 'person'])]
        # A complete selected message already includes its unclaimed remainder.
        # Do not rank those fragments independently against their own context.
        retained_hits = [hit for hit in retained_hits
            if (hit['turn_id'], hit.get('source_message_hash')) not in emitted_messages]
        # A checkpoint's copy of words a person-scope candidate has goes with that candidate, including one the time
        # filter or the reply rule leaves out: the words were said when the original was, and the rule acts on the
        # original. What can bring a correction to either decides whether the copy adds anything
        # (``correction_membership``), and the pair the original joins must show the copy's ranking window, so the
        # pairs are made first (``drop_checkpoint_copies``). Pairing, the time filter and the dedupe act on different
        # rows, or on each row alone, so the order changes nothing else. Identical copies from several checkpoints
        # collapse after the time filter, which judges each by its own date. A pair stands where the better ranked of
        # its two messages was found (``lead_pairs``).
        quotations = source_candidates(retained_hits)
        copied = {row['source_message_hash'] for row in quotations
                  if row.get('scope') == 'session' and row.get('source_message_hash')}
        membership = {}
        if copied:
            with closing(self.ledger._connect()) as conn:
                for row in [*originals, *quotations]:
                    identity = row.get('source_turn_id'), row.get('source_message_hash')
                    if identity[1] in copied and identity not in membership:
                        membership[identity] = correction_membership(conn, *identity)
        paired = pair_conversation_candidates(
            lead_pairs(filter_unstructured(quotations, time_query), input_pairs), input_pairs, sources)
        def pair_corrections(row):
            # What annotation expansion attaches to this pair, read as expansion reads it.
            expanded = expand_annotations(self.ledger, [row], contact_id=contact_id, session_id=session_id)
            return frozenset(expanded[0].get('_annotation_ids') or ()) if expanded else None
        quotations = collapse_repeated_pairs(collapse_checkpoint_copies(
            drop_checkpoint_copies(paired, originals=originals, membership=membership), membership=membership),
            corrections=pair_corrections)
        return filter_unstructured(retained_beliefs, time_query), bundles + quotations


IDENTITY_SECONDS = 30.0     # pending identity reconciliation, at most this often
BUSY_PAUSE, IDLE_PAUSE = .05, 2.0


async def _lane(step, deferred, *, idle=IDLE_PAUSE):
    """Run ``step`` for as long as the worker runs: again at once after it did work, after ``idle``
    seconds when it found none or failed. A failure is logged as ``deferred`` and never ends the lane."""
    while True:
        try:
            worked = await step()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(deferred, type(exc).__name__)
            worked = False
        await asyncio.sleep(BUSY_PAUSE if worked else idle)


async def run_source_claim_worker(ledger, router_provider, *, claims_enabled=True, commitments_provider=None,
                                  vectors=True):
    """One consumer per job family, durable jobs and leases; process loss resumes from SQLite.

    Each family runs on its own lane (``_lane``): identity reconciliation, source-vector indexing,
    media descriptions, the judgment, appraisal, claim and commitment (capture) projections, and the
    owner-started re-extraction of history (``beliefs.reextract``).
    No lane waits for another. A source-vector job commits to a table that can take minutes per
    commit before its first compaction, with tens of thousands of sources queued after an upgrade;
    capture and the other model projections go on meanwhile, and a model call that hangs holds no
    indexing back. Each lane leases its own jobs, so running them side by side changes no job's
    outcome, only when it lands.

    ``vectors=False`` leaves the source-vector jobs to a task on the loop that owns the vector store (the
    paired harness runs this worker on its own thread).

    The model-backed lanes (media, appraisal, claim, commitment) take new jobs first; a job queued
    before this release started is backlog and starts only in a slot of its hourly budget
    (``turns.projection_backlog``), so an upgraded store's history never runs through the model at
    the lanes' full rate."""
    projection = SourceClaimProjection(ledger)
    from colonymind.identity import get_owner_contact_id
    from colonymind.self_model.judgments import SelfJudgments
    judgments = SelfJudgments(ledger, owner_id=get_owner_contact_id())
    from colonymind.self_model.appraisals import AppraisalStore
    from colonymind.contacts.affect_writer import contact_signal_writer
    from colonymind.api.routers import host as _host
    appraisals = AppraisalStore(ledger, owner_id=get_owner_contact_id(), on_contact=contact_signal_writer(
        lambda: _host._affect_store, lambda: _host._contacts_store, owner_id_provider=get_owner_contact_id),
        on_owner=owner_signal_writer(_host._mind))
    from colonymind.commitments.extract import CommitmentExtractor, contact_aliases
    if commitments_provider is None:
        commitments_provider = lambda: _host._commitment_store  # noqa: E731
    # The running mind keeps the owner's interests: this pass lists them and settles one the owner says is answered.
    commitment_extractor = CommitmentExtractor(ledger, commitments_provider,
                                               aliases=contact_aliases(lambda: _host._contacts_store),
                                               interests=_host._mind)
    from colonymind.turns.media import SourceMedia
    media = SourceMedia(ledger)
    from colonymind.turns.source_vectors import SourceVectors
    from colonymind.vector import get_store, get_pipeline
    source_vectors = SourceVectors(ledger, get_store(), get_pipeline()) if vectors else None
    if source_vectors is not None:
        source_vectors.backfill()
    # The claim priors' vector lane reads the same index, from this loop only (the store's own loop).
    projection.vectors = source_vectors
    try:
        media.recover_unowned_files()
    except OSError:
        logger.warning("source media orphan recovery deferred")
    try:
        from colonymind.turns import projection_backlog
        with closing(ledger._connect()) as conn:
            backlog = projection_backlog.status(conn)
        waiting = sum((backlog.get('waiting') or {}).values())
        if waiting:
            logger.info("%d model projection jobs queued before %s are backlog: new turns go first, and at most "
                        "%g backlog jobs start an hour (projections.backlog_per_hour)", waiting,
                        time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(backlog['backlog_before'])),
                        projection_backlog.configured_per_hour())
    except Exception as exc:
        logger.debug("projection backlog not read (%s)", type(exc).__name__)

    async def reconcile_identities():
        from colonymind.api.routers.social_state import reconcile_pending_identities
        await reconcile_pending_identities(ledger)
        return False

    lanes = {'identity': _lane(reconcile_identities, 'identity source reconciliation deferred (%s)',
                               idle=IDENTITY_SECONDS)}
    async def refresh_contact_digests():
        mind = _host._mind()
        if mind is not None:
            async with mind._lock:
                await mind._digest_upkeep(mind.clock())
        return False

    lanes['contact-digests'] = _lane(refresh_contact_digests, 'contact digest refresh deferred (%s)', idle=30)
    if source_vectors is not None:
        lanes['semantic'] = _lane(source_vectors.process_one, 'source semantic projection deferred (%s)')
    if claims_enabled:
        lanes['media'] = _lane(lambda: media.process_one(router_provider()), 'source claim worker deferred (%s)')
        # The owner-started re-extraction of history (``beliefs.reextract``): its own lane, which yields to live
        # recall and to a new turn's extraction. The running mind's relations follow a claim it re-extracts.
        from .reextract import Reextraction
        reextraction = Reextraction(projection, relations=lambda: getattr(_host._mind(), 'mind_state', None))
        for name, reflection in (('judgment', judgments), ('appraisal', appraisals), ('claim', projection),
                                 ('commitment', commitment_extractor), ('reextract', reextraction)):
            lanes[name] = _lane(lambda reflection=reflection: reflection.process_one(router_provider()),
                                'source ' + name + ' deferred (%s)')
    tasks = [asyncio.create_task(lane, name='colonymind-source-' + name) for name, lane in lanes.items()]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
