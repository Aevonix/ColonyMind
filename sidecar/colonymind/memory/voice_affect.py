"""Small deterministic projection of the existing affect and appraisal stores."""
from __future__ import annotations

from datetime import datetime, timezone
import math

from colonymind.memory.recall import estimate_voice_tokens

STALE_SECONDS = 30 * 60
MAX_TOKENS = 25
MAX_CHARS = 120
DEADLINE_SECONDS = 0.020
APPRAISAL_WORDS = {
    'frustration': 'frustrated',
    'annoyance': 'annoyed',
    'interest': 'interested',
    'satisfaction': 'pleased',
}


def fresh_timestamp(value, now):
    """Missing, malformed, naive, future or 30-minute-old clocks say nothing."""
    try:
        stamp = datetime.fromisoformat(value)
        return stamp.tzinfo is not None and 0 <= now - stamp.timestamp() < STALE_SECONDS
    except (TypeError, ValueError, OverflowError):
        return False


def speaker_words(store, person, *, now):
    from colonymind.memory import curated
    ledger = store._ledger() if hasattr(store, '_ledger') else None
    return curated.fenced_read(ledger, lambda: _speaker_words(store, person, now=now), empty='')


def _speaker_words(store, person, *, now):
    project = getattr(store, 'project_state', None)
    if project is None:
        return ''
    # Check canonical erasure and project the stored aggregate at the route's
    # clock. This read never purges events or persists decay.
    state = project(person, now=datetime.fromtimestamp(now, timezone.utc))
    if not state or not state.get('last_event_id') or not state.get('event_count'):
        return ''
    if not fresh_timestamp(store.last_observed_at(person), now):
        return ''
    valence, arousal = float(state['current_valence']), float(state['current_arousal'])
    if not (math.isfinite(valence) and math.isfinite(arousal)
            and -1 <= valence <= 1 and 0 <= arousal <= 1):
        return ''
    mood = 'frustrated' if valence <= -0.25 else 'pleased' if valence >= 0.25 else 'neutral'
    energy = 'low energy' if arousal < 0.35 else 'high energy' if arousal > 0.65 else 'steady energy'
    return f'{mood}, {energy}'


def safe_name(value, fallback):
    # Stored/configured labels are data too; do not put TTS tags, controls,
    # prompt prose, scores or arbitrarily long identifiers into the line.
    if (not isinstance(value, str) or not 1 <= len(value) <= 20
            or not value.strip() or not any(char.isalpha() for char in value)
            or any(not (char.isalpha() or char in " '") for char in value)):
        return fallback
    return ' '.join(value.split())


def render_line(speaker, appraisal, *, speaker_name, persona_name):
    speaker_name = safe_name(speaker_name, 'Speaker')
    persona_name = safe_name(persona_name, 'Assistant')

    def combine(human, agent):
        return ' · '.join(part for part in (
            f'{human}: {speaker}' if speaker else '',
            f'{agent}: {appraisal}' if appraisal else '',
        ) if part)

    candidates = [combine(speaker_name, persona_name), combine('Speaker', 'Assistant')]
    # Keep whole descriptors if unusually costly names or a cautious operator
    # override prevents both parts fitting. Never slice a word or fabricate it.
    if speaker:
        candidates.append(f'Speaker: {speaker}')
    if appraisal:
        candidates.append(f'Assistant: {appraisal}')
    for line in candidates:
        if len(line) <= MAX_CHARS and estimate_voice_tokens(line) <= MAX_TOKENS:
            return line
    return ''
