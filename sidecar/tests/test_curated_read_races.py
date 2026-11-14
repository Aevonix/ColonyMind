"""A peer withdrawal during a fenced read must precede publication."""
import asyncio
import inspect
import subprocess
import sys
import threading

import pytest

from colonymind.contacts.affect import AffectStore
from colonymind.contacts.comms import CommsLog
from colonymind.memory import curated, producers, turn_budget
from colonymind.memory.turn_budget import TurnBudget
from colonymind.memory.voice_recall import read_voice_packet
from colonymind.turns import TurnIdempotencyLedger


def withdraw(ledger, source):
    subprocess.run([sys.executable, '-c', '''from pathlib import Path
import sys
from colonymind.turns import TurnIdempotencyLedger
from colonymind.memory import curated
curated.forget_sources(TurnIdempotencyLedger(Path(sys.argv[1])),
    'fixture-owner', source_ids=[sys.argv[2]])
''', str(ledger.db_path), source], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@pytest.fixture
def traffic(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    log = CommsLog(str(tmp_path / 'comms.db'), source_ledger=ledger)
    for index, source in enumerate(('first', 'second', 'seed')):
        ledger.record_source(source, contact_id='fixture-owner', session_id=source,
            messages=[{'role': 'user', 'content': 'Fixture reference ' + source}], derive_claims=False)
        if source != 'seed':
            lineage, _ = log.source_input(source, 'fixture-owner')
            log.log('fixture-owner', source_lineage=lineage, summary='Fixture',
                    ts='2099-01-0' + str(index + 1))
    return ledger, log


def second_check(monkeypatch, ledger, target):
    original = curated.is_forgotten
    checks = depth = 0
    fired = []

    def check(reader, ref, **kwargs):
        nonlocal checks, depth
        if depth == 0:
            checks += 1
            if checks == 2:
                withdraw(ledger, target())
                fired.append(True)
        depth += 1
        try:
            return original(reader, ref, **kwargs)
        finally:
            depth -= 1

    monkeypatch.setattr(curated, 'is_forgotten', check)
    return fired


def test_summary_withdrawal_during_native_validation(traffic, monkeypatch):
    ledger, log = traffic
    original = log._invalid_sources
    fired = []

    def invalid(rows, *args, **kwargs):
        if not fired:
            withdraw(ledger, 'first')
            fired.append(True)
        return original(rows, *args, **kwargs)

    monkeypatch.setattr(log, '_invalid_sources', invalid)
    offered = log.history('fixture-owner')
    assert fired
    assert len(log.history('fixture-owner')) == 1
    assert len(offered) == 1


@pytest.mark.parametrize('method', ['counts', 'rollup', 'inbound', 'producer', 'affect', 'curated'])
def test_withdrawal_between_batch_checks(traffic, tmp_path, monkeypatch, method):
    ledger, log = traffic
    affect = AffectStore(str(tmp_path / 'affect.db'), source_ledger=ledger)
    if method == 'affect':
        for index, source in enumerate(('first', 'second')):
            lineage, _ = log.source_input(source, 'fixture-owner')
            affect.create_event(contact_id='fixture-owner', valence=0.2,
                source_lineage=lineage, timestamp='2099-01-0' + str(2 - index))
    curated.forget_sources(ledger, 'fixture-owner', source_ids=['seed'])
    fired = second_check(monkeypatch, ledger, lambda: 'first')

    def read():
        if method in ('counts', 'producer'):
            return log.counts('fixture-owner')['inbound']
        if method == 'rollup':
            return log.rollup().get('unknown', {}).get('in', 0)
        if method == 'inbound':
            return len(log.inbound_since('fixture-owner', '2000'))
        if method == 'affect':
            return len(affect.list_events(contact_id='fixture-owner'))
        return len(curated.current_candidates(ledger,
            [dict(turn_id=source, content='Fixture') for source in ('first', 'second')],
            contact_id='fixture-owner'))

    async def producer_read():
        with turn_budget.turn(TurnBudget(wait_s=7.5)):
            return await producers.read(read)

    answer = asyncio.run(producer_read()) if method == 'producer' else read()
    assert fired
    assert read() == 1
    assert answer == 1


def test_voice_withdrawal_during_final_fence(traffic, monkeypatch):
    ledger, _ = traffic
    curated.forget_sources(ledger, 'fixture-owner', source_ids=['seed'])
    original = curated.strict_current_candidates
    fired = []

    def final_fence(reader, candidates, **kwargs):
        if inspect.currentframe().f_back.f_code.co_name != '_read_voice_packet' or fired:
            return original(reader, candidates, **kwargs)
        assert len(candidates) >= 2
        row = candidates[0]
        target = row.get('turn_id') or row.get('source_turn_id') or row['source_uri'][5:]
        with monkeypatch.context() as patch:
            scheduled = second_check(patch, ledger, lambda: target)
            result = original(reader, candidates, **kwargs)
            fired.extend(scheduled)
            return result

    monkeypatch.setattr(curated, 'strict_current_candidates', final_fence)
    packet = read_voice_packet(ledger, query='Fixture reference', contact_id='fixture-owner',
        session_id='fixture-session', k=5, max_tokens=600, timezone_name='UTC',
        cancelled=threading.Event())
    assert fired
    assert len(original(ledger, packet.selected, contact_id='fixture-owner')) == len(packet.selected)
    assert len(packet.selected) == 1


def test_empty_journal_mask_rechecks_before_return(traffic, monkeypatch):
    ledger, _ = traffic
    original = curated._state
    fired = []

    def state(*args, **kwargs):
        result = original(*args, **kwargs)
        if not fired:
            withdraw(ledger, 'first')
            fired.append(True)
        return result

    monkeypatch.setattr(curated, '_state', state)
    assert curated.forgotten_mask(ledger, ['first', 'second']) == [True, False]
    assert fired


def test_summary_continued_commits_fail_closed(traffic, monkeypatch):
    ledger, log = traffic
    original = log._invalid_sources
    committed = []

    def invalid(rows, *args, **kwargs):
        source = ('seed', 'first', 'second')[len(committed)]
        withdraw(ledger, source)
        committed.append(source)
        return original(rows, *args, **kwargs)

    monkeypatch.setattr(log, '_invalid_sources', invalid)
    assert log.history('fixture-owner') == []
    assert len(committed) == 3


def test_voice_withdrawal_during_packet_preparation(traffic, monkeypatch):
    from colonymind.memory import voice_recall
    ledger, _ = traffic
    original = voice_recall.packet_refs
    fired = []

    def refs(reader, selected, **kwargs):
        result = original(reader, selected, **kwargs)
        if not fired:
            row = selected[0]
            target = row.get('turn_id') or row.get('source_turn_id') or row['source_uri'][5:]
            withdraw(ledger, target)
            fired.append(True)
        return result

    monkeypatch.setattr(voice_recall, 'packet_refs', refs)
    packet = read_voice_packet(ledger, query='Fixture reference', contact_id='fixture-owner',
        session_id='fixture-session', k=5, max_tokens=600, timezone_name='UTC',
        cancelled=threading.Event())
    assert fired
    assert len(curated.current_candidates(ledger, packet.selected,
        contact_id='fixture-owner')) == len(packet.selected)
