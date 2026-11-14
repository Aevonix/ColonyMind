"""Visible results captured by running these inputs against 484869214's eraser."""
import json
from pathlib import Path
from datetime import datetime

import pytest

from colonymind.memory import curated
from colonymind.turns.idempotency import canonical_turn_digest
from test_curated_memory import state, client, say, ledger_of, PERSON, recall, historical_bytes
from test_curated_memory_review import packet

AT = '2026-03-01T09:00:00+00:00'
GOLDEN = Path(__file__).parent / 'fixtures' / 'source-forget-legacy-recall.json'


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    from colonymind.util import temporal
    monkeypatch.setattr(temporal, 'now_utc', lambda: datetime.fromisoformat(AT))


def forget(ledger, **selection):
    return curated.forget_sources(ledger, PERSON, **selection)


async def visible(client, ledger):
    result = {}
    for query in ('office', 'bicycle'):
        recalled = await recall(client, query)
        result[query] = {key: recalled[key] for key in ('content', 'count')}
    for path in ('text', 'voice'):
        result[path + '_office'] = (await packet(ledger, path)).content
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize('selection', ['ids', 'old_text'])
async def test_source_forget_recall_parity(state, client, selection):
    await say(state, 'private-office', 'My office is in River.')
    ledger = ledger_of(state)
    message = {'role': 'user', 'content': 'My office is in River.'}
    ledger.record_source('checkpoint', contact_id=PERSON, session_id='s1', occurred_at=AT,
                         messages=[message, {'role': 'user', 'content': 'My bicycle is blue.'}])
    ledger.record_source('answer', contact_id=PERSON, session_id='s1', occurred_at=AT, messages=[
        {'role': 'assistant', 'content': 'Your office is in River.', '_supplied_sources': [
            {'source_id': 'private-office', 'source_version': canonical_turn_digest([message])}]}])
    assert 'River' in (await recall(client, 'office'))['content']
    if selection == 'old_text':
        with pytest.raises(curated.Refused, match='ambiguous_source'):
            forget(ledger, old_text=message['content'], session_id='s1')
    before = historical_bytes(ledger)
    first = forget(ledger, source_ids=['private-office'])
    assert first == forget(ledger, source_ids=['private-office'])
    assert historical_bytes(ledger) == before
    assert await visible(client, ledger) == json.loads(GOLDEN.read_text())


@pytest.mark.asyncio
async def test_exact_text_source_forget_recall_parity(state, client):
    await say(state, 'private-office', 'My office is in River.')
    ledger = ledger_of(state)
    before = historical_bytes(ledger)
    assert 'River' in (await recall(client, 'office'))['content']
    first = forget(ledger, old_text='My office is in River.', session_id='s1')
    assert first == forget(ledger, old_text='My office is in River.', session_id='s1')
    assert historical_bytes(ledger) == before
    assert (await recall(client, 'office'))['content'] == ''
    assert (await packet(ledger, 'text')).content == ''
    assert (await packet(ledger, 'voice')).content == ''
