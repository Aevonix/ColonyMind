"""Conversations a wearable or room microphone heard around the owner, kept in the canonical source ledger.

A capturing device (a hub that transcribes a pendant, a room microphone) sends one complete conversation: its
speaker-labelled segments and what the device derived from them. It is not a turn. Nobody said it to the agent,
and only the words of the speaker labelled ``owner`` (the device verified the owner's voice) are the owner's.

The conversation is one source of the owner's, ``observed-conversation:<source>:<id>``: its own session, person
scope, no expiry. Recall, the source reader, backup, forgetting and orphan cleanup apply as to any source; nothing
here is a parallel store. Every speaker's words stay owner-side only; no contact sources or claims are formed
from them. Forgetting also sweeps any speaker sources an older writer created
(``curated.forget_sources``).

Each message carries one ``observed_conversation`` block: a window of consecutive segment pieces small enough to
be one lexical chunk, so a recalled excerpt keeps its header and every line its speaker label. Only this writer
sets the message's ``_observed_conversation`` marker (no public message schema carries it); a block without it is
kept unretained (``turns.media``), never rendered as a conversation. The claim view of a window adds a few
preceding pieces as context and marks the attributed speaker's pieces as the only quotable spans
(``turns.audio.claim_message``), so another speaker's first-person words never become the owner's claim.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
import re
from typing import Annotated, Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field

PREFIX = 'observed-conversation:'
VERSION = 'observed-conversation-v1'
BLOCK = 'observed_conversation'
MODALITY = 'observed_conversation'
OWNER = 'owner'
PROVENANCE = 'reported_in_observed_conversation'

CONVERSATION_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.:-]*$')
MAX_CONVERSATION_ID = 128
# Bounds (builder's choices, within the turn envelope's 8 MiB source and the 10 MiB request limit): about eleven
# hours of continuous speech, stored with its labels and window metadata well under the source limit.
MAX_SEGMENTS = 10_000
MAX_TOTAL_CHARS = 600_000
MAX_SEGMENT_CHARS = 20_000
MAX_SPEAKERS = 32
MAX_SUMMARY_CHARS = 4_000
MAX_ACTION_ITEMS = 50
MAX_ACTION_ITEM_CHARS = 500
MAX_DURATION_MS = 24 * 3600 * 1000
# A device clock a little ahead of the sidecar's, and segment offsets rounded past the conversation's end.
CLOCK_SKEW_SECONDS = 120
SEGMENT_SLACK_MS = 5_000
# A window renders within one lexical chunk (``idempotency._lexical_chunks``: 2000 characters) and one semantic
# chunk (``source_vectors.CHUNK``), so no recalled excerpt loses its header or a line its label.
WINDOW_CHARS = 1_800
PIECE_CHARS = 1_200
# The claim view of a window opens with the end of what was said just before it: context, never quotable.
CONTEXT_PIECES = 2
CONTEXT_CHARS = 400


class ObservedRefusal(ValueError):
    """A conversation the route refuses: its HTTP status and code."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


class ObservedRecognizer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str = Field(min_length=1, max_length=256)
    revision: Optional[str] = Field(default=None, min_length=1, max_length=256)


class ObservedSpeaker(BaseModel):
    model_config = ConfigDict(extra='forbid')
    label: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')
    contact_id: Optional[str] = Field(default=None, min_length=1, max_length=256)


class ObservedSegment(BaseModel):
    model_config = ConfigDict(extra='forbid')
    speaker: str = Field(min_length=1, max_length=64)
    start_ms: int = Field(ge=0, le=MAX_DURATION_MS, strict=True)
    end_ms: int = Field(ge=0, le=MAX_DURATION_MS, strict=True)
    text: str = Field(min_length=1, max_length=MAX_SEGMENT_CHARS)


class ObservedConversationRequest(BaseModel):
    """One complete conversation a device heard. ``summary`` and ``action_items`` are the device's own derivations,
    kept labelled as derived: never transcript words, never commitments."""
    model_config = ConfigDict(extra='forbid')
    source: str = Field(pattern=r'^[a-z0-9][a-z0-9_.-]{0,31}$')
    captured_at: str = Field(min_length=1, max_length=64)
    ended_at: str = Field(min_length=1, max_length=64)
    recognizer: ObservedRecognizer
    speakers: List[ObservedSpeaker] = Field(min_length=1, max_length=MAX_SPEAKERS)
    segments: List[ObservedSegment] = Field(min_length=1)
    summary: Optional[str] = Field(default=None, min_length=1, max_length=MAX_SUMMARY_CHARS)
    action_items: List[Annotated[str, Field(min_length=1, max_length=MAX_ACTION_ITEM_CHARS)]] = Field(
        default_factory=list, max_length=MAX_ACTION_ITEMS)


class ObservedConversationResponse(BaseModel):
    accepted: bool
    source_recorded: bool
    conversation_id: str
    source_id: str
    speaker_source_ids: List[str] = Field(default_factory=list)
    skipped_reason: Optional[str] = None


def source_id(source: str, conversation_id: str) -> str:
    return f'{PREFIX}{source}:{conversation_id}'


def is_conversation_source(turn_id: str) -> bool:
    """The conversation's own source (not a speaker's)."""
    return turn_id.startswith(PREFIX) and '#' not in turn_id


def is_observed_source(turn_id: str) -> bool:
    return turn_id.startswith(PREFIX)


def check_conversation_id(conversation_id: str) -> str:
    value = conversation_id or ''
    if len(value) > MAX_CONVERSATION_ID or not CONVERSATION_ID.fullmatch(value):
        raise ObservedRefusal(422, 'invalid_conversation_id',
                              'conversation_id must match ^[A-Za-z0-9][A-Za-z0-9_.:-]*$, at most 128 characters')
    return value


def _instant(value: str, name: str) -> datetime:
    try:
        stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        stamp = None
    if stamp is None or stamp.tzinfo is None:
        raise ObservedRefusal(422, 'invalid_timestamp', f'{name} must be an ISO 8601 time with a timezone')
    return stamp.astimezone(timezone.utc)


def validate(body: ObservedConversationRequest, *, now: datetime, reserved_contacts=frozenset()) -> dict:
    """The checks a schema cannot state, in order; the normalized clocks. Raises ``ObservedRefusal``: 413 for a
    conversation past the size bounds, 422 for anything else."""
    if len(body.segments) > MAX_SEGMENTS or sum(len(s.text) for s in body.segments) > MAX_TOTAL_CHARS:
        raise ObservedRefusal(413, 'observed_conversation_too_large',
                              f'a conversation holds at most {MAX_SEGMENTS} segments and {MAX_TOTAL_CHARS} '
                              'characters of segment text')
    captured, ended = _instant(body.captured_at, 'captured_at'), _instant(body.ended_at, 'ended_at')
    if captured > ended:
        raise ObservedRefusal(422, 'conversation_clock_order', 'captured_at must not be after ended_at')
    if (ended - now).total_seconds() > CLOCK_SKEW_SECONDS:
        raise ObservedRefusal(422, 'conversation_in_future', 'ended_at must not be in the future')
    duration_ms = (ended - captured).total_seconds() * 1000
    if duration_ms > MAX_DURATION_MS:
        raise ObservedRefusal(422, 'conversation_too_long', 'a conversation spans at most 24 hours')
    labels = [speaker.label for speaker in body.speakers]
    if len(set(labels)) != len(labels):
        raise ObservedRefusal(422, 'duplicate_speaker', 'each speaker label appears once')
    for speaker in body.speakers:
        if speaker.label == OWNER and speaker.contact_id is not None:
            raise ObservedRefusal(422, 'owner_speaker_contact',
                                  'the owner speaker is the owner; it names no contact')
        if speaker.contact_id is not None and (speaker.contact_id in reserved_contacts
                                               or speaker.contact_id.startswith('credential:')):
            raise ObservedRefusal(422, 'invalid_speaker_contact',
                                  'a speaker contact must be a person other than the owner, shared or global person')
    previous = 0
    for segment in body.segments:
        if segment.speaker not in labels:
            raise ObservedRefusal(422, 'unknown_speaker', 'every segment names a listed speaker')
        if segment.end_ms < segment.start_ms:
            raise ObservedRefusal(422, 'invalid_segment_time', 'a segment ends at or after its start')
        if segment.start_ms < previous:
            raise ObservedRefusal(422, 'segments_out_of_order', 'segments are ordered by start_ms')
        if segment.end_ms > duration_ms + SEGMENT_SLACK_MS:
            raise ObservedRefusal(422, 'segment_outside_conversation',
                                  'segment offsets are from captured_at and end by ended_at')
        if not segment.text.strip():
            raise ObservedRefusal(422, 'empty_segment', 'a segment carries words')
        previous = segment.start_ms
    if body.summary is not None and not body.summary.strip():
        raise ObservedRefusal(422, 'empty_summary', 'a summary carries words, or is null')
    if any(not item.strip() for item in body.action_items):
        raise ObservedRefusal(422, 'empty_action_item', 'an action item carries words')
    return {'captured_at': captured.isoformat(), 'ended_at': ended.isoformat()}


def _split(text: str, limit: int = PIECE_CHARS) -> list[str]:
    """``text`` in consecutive parts of at most ``limit`` characters, cut after a space where one is near: joined
    they are exactly ``text``."""
    parts = []
    while len(text) > limit:
        cut = text.rfind(' ', limit // 2, limit)
        cut = limit if cut < 0 else cut + 1
        parts.append(text[:cut])
        text = text[cut:]
    parts.append(text)
    return parts


def _label(block: dict, label: str) -> str:
    contact = next((s.get('contact_id') for s in block.get('speakers', ()) if s.get('label') == label), None)
    return f'[{label}, contact {contact}]' if contact else f'[{label}]'


def _header(block: dict) -> str:
    if 'derived' in block:
        kind = 'summary and action items derived by the capturing device, unverified'
    else:
        whose = ('every speaker' if block.get('attributed_speaker') == OWNER
                 else f"only the words of {block.get('attributed_speaker')}")
        kind = f"unverified machine transcript of {whose}, part {block.get('part')} of {block.get('parts')}"
    return (f"Observed conversation {source_id(block.get('source'), block.get('conversation_id'))}, "
            f"heard by the owner's {block.get('source')} "
            f"device and not said to the agent; captured {block.get('captured_at')} to {block.get('ended_at')}; "
            f"{kind}.")


def render(block: dict, *, claim_view: bool = False) -> tuple[str, list[dict]]:
    """One block's text, and in the claim view the attributed speaker's pieces as quotable spans (offsets in that
    text). Recall, indexing and the source reader use the plain view; only extraction sees the context lines."""
    lines = [_header(block)]
    length = len(lines[0])
    spans: list[dict] = []

    def add(line: str) -> int:
        nonlocal length
        start = length + 1
        lines.append(line)
        length = start + len(line)
        return start

    derived = block.get('derived') if isinstance(block.get('derived'), dict) else None
    if derived is not None:
        if isinstance(derived.get('summary'), str):
            add('Summary: ' + derived['summary'])
        items = [item for item in derived.get('action_items') or () if isinstance(item, str)]
        if items:
            add('Action items (not commitments):')
            for item in items:
                add('- ' + item)
    if claim_view:
        for piece in block.get('context_before') or ():
            if isinstance(piece, dict) and isinstance(piece.get('text'), str):
                add(f"(said before this part, context only) {_label(block, piece.get('speaker'))} {piece['text']}")
    for piece in block.get('pieces') or ():
        if not isinstance(piece, dict) or not isinstance(piece.get('text'), str):
            continue
        prefix = _label(block, piece.get('speaker')) + ' '
        start = add(prefix + piece['text']) + len(prefix)
        if claim_view and piece.get('speaker') == block.get('attributed_speaker'):
            spans.append({
                'segment_index': piece.get('segment'), 'piece_offset': piece.get('offset'),
                'speaker': piece.get('speaker'),
                'contact_id': next((s.get('contact_id') for s in block.get('speakers', ())
                                    if s.get('label') == piece.get('speaker')), None),
                'conversation_id': block.get('conversation_id'),
                'start_ms': piece.get('start_ms'), 'end_ms': piece.get('end_ms'),
                'source_start': start, 'source_end': start + len(piece['text']),
                'recognizer': block.get('recognizer'), 'received_at': None,
                # The conversation's own capture time, common to its segments: the anchor of relative dates.
                'captured_at': block.get('captured_at'), 'timing_basis': 'supplied_conversation_offsets',
                'confidence': None, 'source_modality': MODALITY, 'provenance': PROVENANCE})
    return '\n'.join(lines), spans


def block_of(message: Any) -> dict | None:
    """The message's observed conversation block, when the ledger wrote it as one."""
    if not isinstance(message, dict) or message.get('_observed_conversation') != VERSION:
        return None
    content = message.get('content')
    if not isinstance(content, list):
        return None
    return next((block for block in content if isinstance(block, dict) and block.get('type') == BLOCK), None)


def handle(block: dict) -> dict:
    """What a recalled excerpt says of where it came from: the conversation, when, and its speaker labels."""
    return {'conversation_id': block.get('conversation_id'), 'device_source': block.get('source'),
            'source_id': source_id(block.get('source'), block.get('conversation_id')),
            'captured_at': block.get('captured_at'), 'ended_at': block.get('ended_at'),
            'speakers': [s.get('label') for s in block.get('speakers', ()) if isinstance(s, dict)],
            'heard_by': 'owner_device', 'said_to_agent': False}


def owner_text(message: Any) -> str | None:
    """The owner's own words in an observed conversation message, for owner-word readers; '' when the message is
    the conversation's but holds none (another speaker's source, the derived summary); None when it is not an
    observed conversation message. The device's action items never count: they are its derivation."""
    block = block_of(message)
    if block is None:
        return None
    if block.get('attributed_speaker') != OWNER:
        return ''
    words = [piece['text'] for piece in block.get('pieces') or ()
             if isinstance(piece, dict) and piece.get('speaker') == OWNER and isinstance(piece.get('text'), str)]
    if not words:
        return ''
    return ("[The owner's own words in a conversation with other people that the owner's device heard; "
            "not said to the assistant.]\n" + '\n'.join(words))


def _common(data: dict, conversation_id: str, clocks: dict, attributed: str) -> dict:
    return {'type': BLOCK, 'conversation_id': conversation_id, 'source': data['source'],
            'captured_at': clocks['captured_at'], 'ended_at': clocks['ended_at'],
            'recognizer': dict(data['recognizer']),
            'speakers': [dict(speaker) for speaker in data['speakers']],
            'attributed_speaker': attributed, 'epistemic_state': 'derived_unverified'}


def _message(block: dict) -> dict:
    return {'role': 'user', 'content': [block], '_observed_conversation': VERSION}


def messages(data: dict, conversation_id: str, clocks: dict, *, attributed: str = OWNER) -> list[dict]:
    """The source messages of one conversation: for the owner every segment (and the derived summary first), for
    another speaker that speaker's segments only."""
    common = _common(data, conversation_id, clocks, attributed)
    result = []
    if attributed == OWNER and (data.get('summary') is not None or data.get('action_items')):
        result.append(_message({**common, 'derived': {'summary': data.get('summary'),
                                                      'action_items': list(data.get('action_items') or ())}}))
    pieces = []
    for index, segment in enumerate(data['segments']):
        if attributed != OWNER and segment['speaker'] != attributed:
            continue
        offset = 0
        for part in _split(segment['text']):
            pieces.append({'segment': index, 'speaker': segment['speaker'], 'start_ms': segment['start_ms'],
                           'end_ms': segment['end_ms'], 'offset': offset, 'text': part})
            offset += len(part)
    budget = WINDOW_CHARS - len(_header({**common, 'part': 10 ** 6, 'parts': 10 ** 6}))
    windows, current, used = [], [], 0
    for piece in pieces:
        cost = len(_label(common, piece['speaker'])) + 2 + len(piece['text'])
        if current and used + cost > budget:
            windows.append(current)
            current, used = [], 0
        current.append(piece)
        used += cost
    if current:
        windows.append(current)
    before: list[dict] = []
    for part, window in enumerate(windows, 1):
        context, room = [], CONTEXT_CHARS
        for piece in reversed(before[-CONTEXT_PIECES:]):
            if room <= 0:
                break
            text = piece['text'][-room:]
            context.insert(0, {'speaker': piece['speaker'], 'text': text})
            room -= len(text)
        result.append(_message({**common, 'part': part, 'parts': len(windows), 'context_before': context,
                                'pieces': window}))
        before = window
    return result


def record(ledger, data: dict, conversation_id: str, clocks: dict, *, owner_id: str) -> list[str]:
    """Write only the owner's source; return an empty list of speaker source ids for contract stability.
    The ledger's idempotent ``record_source`` repeats nothing on retry. Raises its
    ``SourceErased`` and ``ValueError`` as it does."""
    primary = source_id(data['source'], conversation_id)
    ledger.record_source(primary, contact_id=owner_id, session_id=primary, scope='person',
                         messages=messages(data, conversation_id, clocks), occurred_at=clocks['captured_at'])
    return []


def speaker_sources(conn, conversation_source: str) -> list[tuple[str, str]]:
    """``(source id, contact id)`` of each speaker source of a conversation still in the ledger."""
    prefix = conversation_source + '#'
    return [(row[0], row[1]) for row in conn.execute(
        'SELECT turn_id,contact_id FROM turn_sources WHERE substr(turn_id,1,?)=? ORDER BY turn_id',
        (len(prefix), prefix))]


def stored(ledger, source: str, conversation_id: str) -> dict | None:
    """The conversation as the ledger holds it now: what was sent, rebuilt from its windows (a value sealed on
    capture stays a reference), or None when it is not held. ``segments_complete`` is false when an exact-message
    erasure removed part of it."""
    primary = source_id(source, conversation_id)
    with closing(ledger._connect()) as conn:
        row = conn.execute('SELECT * FROM turn_sources WHERE turn_id=?', (primary,)).fetchone()
        if row is None:
            return None
    blocks = [block for block in map(block_of, json.loads(row['messages_json'])) if block is not None]
    if not blocks:
        return None
    first, segments, derived = blocks[0], {}, None
    parts, total = set(), None
    for block in blocks:
        if 'derived' in block:
            derived = block['derived']
            continue
        parts.add(block.get('part'))
        total = block.get('parts')
        for piece in block.get('pieces') or ():
            entry = segments.setdefault(piece['segment'], {'speaker': piece['speaker'], 'start_ms': piece['start_ms'],
                                                           'end_ms': piece['end_ms'], 'pieces': []})
            entry['pieces'].append((piece['offset'], piece['text']))
    complete = isinstance(total, int) and parts == set(range(1, total + 1))
    return {
        'conversation_id': conversation_id, 'source_id': primary, 'speaker_source_ids': [],
        'source': first.get('source'), 'captured_at': first.get('captured_at'), 'ended_at': first.get('ended_at'),
        'recognizer': first.get('recognizer'), 'speakers': first.get('speakers'),
        'segments': [{'speaker': entry['speaker'], 'start_ms': entry['start_ms'], 'end_ms': entry['end_ms'],
                      'text': ''.join(text for _, text in sorted(entry['pieces']))}
                     for _, entry in sorted(segments.items())],
        'summary': derived.get('summary') if derived else None,
        'action_items': list(derived.get('action_items') or ()) if derived else [],
        'derived_by': 'capturing_device', 'epistemic_state': 'derived_unverified',
        'segments_complete': complete, 'ingested_at': row['ingested_at'],
    }
