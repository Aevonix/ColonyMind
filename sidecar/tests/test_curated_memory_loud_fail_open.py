"""Failed curated reads withhold recall; one owner notice reports each incident."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import copy
import json
import sqlite3
from types import SimpleNamespace

import pytest

from colonymind.api.routers import host
from colonymind.memory import curated, curated_health
from colonymind.util import temporal
from test_curated_memory import PERSON, ROOT, client, ledger_of, remember, say, state  # noqa: F401
from test_curated_memory_review import packet, withheld_packet
from test_health_problems import quiet_host  # noqa: F401
from test_mind_loop import fx, pinned_clock  # noqa: F401


PRIVATE = 'synthetic private query and memory text'


@contextmanager
def unavailable(ledger):
    raise sqlite3.OperationalError(PRIVATE)
    yield


@pytest.mark.parametrize('path', ['text', 'voice'])
async def test_1000_failed_recalls_one_alert_recovery_then_second_incident(fx, monkeypatch, path):
    ledger = fx.ledger
    ledger.record_source('native', contact_id=PERSON, session_id='s',
                         messages=[{'role': 'user', 'content': 'My office is in River.'}])
    healthy = await packet(ledger, path)
    health = curated_health.for_ledger(ledger)
    assert health.snapshot() == {'fail_open_recalls': 0, 'degraded': False,
                                 'first_seen': None, 'error_class': None}
    notices = []
    real_notice = fx.mind.outbox.notice

    def notice(**kwargs):
        notices.append(kwargs)
        return real_notice(**kwargs)

    monkeypatch.setattr(fx.mind.outbox, 'notice', notice)
    with monkeypatch.context() as broken:
        broken.setattr(curated, '_snapshot', unavailable)
        for i in range(1000):
            before = len(notices)
            await withheld_packet(ledger, path)
            assert len(notices) == before
            if i == 0:
                first_seen = health.snapshot()['first_seen']
            # The legacy counter counts attempted fallbacks. Later requests
            # are withheld before attempting another curated read or fallback.
            assert health.snapshot() == {'fail_open_recalls': 1, 'degraded': True,
                                         'first_seen': first_seen, 'error_class': 'OperationalError'}
            # Poll several times during the incident. Recall itself never calls notice.
            if i in (0, 499, 999):
                ready = await fx.mind.outbox_ready()
                assert len(notices) == 1
                assert len(ready) == 1
        assert len(notices) == 1

    curated.enable(ledger)  # The existing successful initialization/re-enable recovery boundary.
    assert await packet(ledger, path) == healthy
    assert health.snapshot() == {'fail_open_recalls': 1, 'degraded': False,
                                 'first_seen': None, 'error_class': None}
    # An outstanding notice survives recovery; it is never queued twice.
    assert len(await fx.mind.outbox_ready()) == 1 and len(notices) == 1
    fx.shift(seconds=1)
    with monkeypatch.context() as broken:
        broken.setattr(curated, '_snapshot', unavailable)
        await withheld_packet(ledger, path)
    ready = await fx.mind.outbox_ready()
    assert len(ready) == len(notices) == 2
    assert health.snapshot()['fail_open_recalls'] == 2
    assert health.snapshot()['first_seen'] != first_seen
    assert notices[0]['dedup_key'] != notices[1]['dedup_key']
    for message in ready:
        assert message['type'] == 'health_notice' and message['recipient_is_owner']
        assert message['recipient'] == fx.mind.owner_id
        assert message['recipient_handles']
        assert 'Recall is withheld' in message['text']
        assert all(word not in message['text'] for word in (PRIVATE, 'River', 'MEDIA', '\u2013', '\u2014'))


@pytest.mark.parametrize('path,fail_at', [('text', 2), ('voice', 2), ('voice', 3)])
async def test_late_read_failure_counts_the_plain_rebuild_once(state, monkeypatch, path, fail_at):
    ledger = ledger_of(state)
    ledger.record_source('native', contact_id=PERSON, session_id='s',
                         messages=[{'role': 'user', 'content': 'My office is in River.'}])
    real = curated._snapshot
    calls = 0

    @contextmanager
    def failing(ledger):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise RuntimeError(PRIVATE)
        with real(ledger) as conn:
            yield conn

    monkeypatch.setattr(curated, '_snapshot', failing)
    await withheld_packet(ledger, path)
    assert calls >= fail_at
    assert curated_health.for_ledger(ledger).snapshot()['fail_open_recalls'] == 1


async def test_health_exposes_counter_first_seen_and_only_error_class(state, quiet_host, monkeypatch):
    ledger = ledger_of(state)
    stamp = temporal.parse_utc_text('2026-10-18T19:30:00+00:00')
    monkeypatch.setattr(curated_health, 'now_utc', lambda: stamp)
    monkeypatch.setattr(curated, '_snapshot', unavailable)
    await packet(ledger)
    # A later, different error neither changes the incident stamp nor exposes its message.
    curated.disable(ledger, ValueError(PRIVATE))
    result = await host.health()
    assert result.status == 'degraded'
    assert result.recall['curated'] == {'fail_open_recalls': 1, 'degraded': True,
                                       'first_seen': stamp.isoformat(), 'error_class': 'OperationalError'}
    assert any('forgotten items can resurface' in problem for problem in result.problems)
    assert PRIVATE not in result.model_dump_json()
    curated.enable(ledger)
    result = await host.health()
    assert result.status == 'ok'
    assert result.recall['curated'] == {'fail_open_recalls': 1, 'degraded': False,
                                       'first_seen': None, 'error_class': None}


async def test_api_failure_opens_incident_without_counting_a_recall(state, client, monkeypatch):
    ledger = ledger_of(state)
    monkeypatch.setattr(curated, '_snapshot', unavailable)
    response = await client.get(ROOT + '/search', params={'person_id': PERSON, 'query': PRIVATE})
    assert response.status_code == 503
    health = curated_health.for_ledger(ledger)
    assert health.snapshot()['degraded'] and health.snapshot()['fail_open_recalls'] == 0
    await packet(ledger)
    assert health.snapshot()['fail_open_recalls'] == 0  # Already disabled: no fallback attempted.


async def test_allowance_failure_and_reselection_count_one_recall(state, client, monkeypatch):
    from colonymind.api.schemas.host import ContextSection
    from colonymind.memory import compass
    from colonymind.memory.search import collect_sources, select_memory
    from colonymind.memory.selection import RecallSelector, keep_offered
    from colonymind.memory.trace import NULL_TRACE
    from test_memory_inventory import id_of

    await say(state, 'native', 'My office is in River.')
    ledger = ledger_of(state)
    curated.mutate(ledger, PERSON, 'claim:' + id_of(state, 'River'), 'forget',
                   request_id='forget-native', expected_revision=1)
    await remember(client, 'mem_long', 'My office notes: ' + 'Curated independent detail. ' * 90)
    collected = await collect_sources(ledger, query='office', contact_id=PERSON, session_id='later')

    async def select():
        with keep_offered() as offered:
            result = await select_memory(collected, query='office', selector=RecallSelector(),
                                         timezone_name='UTC', limit=None)
        return result, offered[-1]

    result, offered = await select()
    assert 'River' not in result.content and len(result.content) > 1400
    recalled = SimpleNamespace(packet=result, offered=offered, ledger=ledger, contact_id=PERSON,
                               session_id='later', plain_recall=select, curated_recall=collected.curated_recall)
    monkeypatch.setattr(curated, '_snapshot', unavailable)
    sections, _ = await host._memory_within_allowance(
        [ContextSection(id=compass.RECALLED, title='Relevant Memories', body=result.content, priority=90)],
        recalled, inline_chars=1400, trace=NULL_TRACE)
    assert sections == []
    assert curated_health.for_ledger(ledger).snapshot()['fail_open_recalls'] == 1


async def test_notice_queue_retries_and_keeps_incidents_that_recover_before_poll(fx, monkeypatch):
    ledger, outbox = fx.ledger, fx.mind.outbox
    curated.disable(ledger, RuntimeError(PRIVATE))
    curated.enable(ledger)
    curated.disable(ledger, ValueError(PRIVATE))
    real = outbox.notice
    calls = 0

    def committed_then_failed(**kwargs):
        nonlocal calls
        calls += 1
        result = real(**kwargs)
        if calls == 1:
            raise OSError(PRIVATE)
        return result

    monkeypatch.setattr(outbox, 'notice', committed_then_failed)
    assert len(await fx.mind.outbox_ready()) == 1
    assert len(await fx.mind.outbox_ready()) == 2
    assert len(await fx.mind.outbox_ready()) == 2
    assert calls == 3  # One ambiguous write retried with the same durable key.
    assert len(outbox.store.intentions(types=['health_notice'], kind=['message'])) == 2


def test_concurrent_facades_share_one_incident_and_count_each_recall(state):
    ledger = ledger_of(state)

    def recall(_):
        facade = copy(ledger)
        curated.disable(facade, RuntimeError(PRIVATE))
        with curated_health.recalling():
            curated.candidates(facade, PERSON, PRIVATE, [], [])
            curated.current_candidates(facade, [], contact_id=PERSON)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(recall, range(1000)))
    health = curated_health.for_ledger(ledger)
    assert health.snapshot()['fail_open_recalls'] == 1000
    assert len(health.pending) == 1


def test_startup_failure_records_class_without_counting_recall(tmp_path, monkeypatch):
    from colonymind.turns import TurnIdempotencyLedger

    def failing(conn):
        raise RuntimeError(PRIVATE)

    monkeypatch.setattr(curated, 'initialize', failing)
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    status = curated_health.for_ledger(ledger).snapshot()
    assert status['degraded'] and status['error_class'] == 'RuntimeError'
    assert status['fail_open_recalls'] == 0
    assert PRIVATE not in json.dumps(status)


def test_disabled_old_handle_reopens_incident_after_another_handle_recovers(state):
    ledger = ledger_of(state)
    old = copy(ledger)
    curated.disable(old, ValueError(PRIVATE))
    curated.enable(ledger)
    health = curated_health.for_ledger(ledger)
    assert not health.snapshot()['degraded']
    with curated_health.recalling():
        curated.candidates(old, PERSON, PRIVATE, [], [])
        curated.current_candidates(old, [], contact_id=PERSON)
    status = health.snapshot()
    assert status['degraded'] and status['error_class'] == 'ValueError'
    assert status['fail_open_recalls'] == 1 and len(health.pending) == 2
