"""Synchronous canonical callbacks, without servers or cross-thread event loops.

A canonical reply forecast (``reply_forecasts``) still asks the learner for its probability before it is issued
(``probability``), and the learner labels what it served from the comms ledger within that forecast's own
horizon, as it labels every send. Dates, identifiers and the one-hour horizon are fixture assumptions.
"""
import json
from types import SimpleNamespace

import pytest

from colonymind.api.routers import host
from colonymind.commitments.store import CommitmentStore
from colonymind.contacts.comms import CommsLog
from colonymind.contacts.transport_ingress import TransportIngress
from colonymind.initiatives.temporal_followup import TemporalFollowups
from colonymind.plasticity.learners import contact_reply_timing as timing
from colonymind.self_model import reply_forecasts as forecasts
from colonymind.self_model.expectations import ExpectationEngine, ExpectationStore
from colonymind.turns import TurnIdempotencyLedger, canonical_turn_digest
from test_plasticity_contact_reply_timing import NOW, HOUR, PRIVATE, contacts, service, talk


@pytest.fixture
def callback(tmp_path, monkeypatch):
    now = [NOW]
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(tmp_path))
    monkeypatch.setenv('COLONYMIND_OWNER_PERSON_ID', 'owner')
    monkeypatch.setenv('COLONYMIND_EXPECTATIONS', 'on')
    monkeypatch.setattr(forecasts.time, 'time', lambda: now[0].timestamp())
    store = ExpectationStore(str(tmp_path / 'colonymind-expectations.db'))
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    commitments = CommitmentStore(tmp_path / 'colonymind-commitments.db')
    contacts(tmp_path, live=('Sam', 'owner'))
    comms = CommsLog(str(tmp_path / 'colonymind-comms.db'), source_ledger=ledger)
    ingress = TransportIngress(comms._conn)
    monkeypatch.setattr(host, '_expectations', ExpectationEngine(store))
    monkeypatch.setattr(host, '_commitment_store', commitments)
    monkeypatch.setattr(host, '_comms_log', comms)
    running = service(tmp_path, clock=lambda: now[0])
    monkeypatch.setattr(host, '_mind', lambda: SimpleNamespace(plasticity=running))
    messages = [{'role': 'user', 'content': PRIVATE[-1]}]
    ledger.record_source('owner-task', contact_id='owner', session_id='owner-session',
        messages=messages, occurred_at=(NOW - HOUR).isoformat(), derive_claims=False)
    parent = commitments.create('owner', 'Obtain the agreed report')
    with commitments._connect() as db, db:
        db.execute('UPDATE commitments SET made_at=? WHERE id=?',
                   ((NOW - HOUR).isoformat(), parent['id']))
    waits = TemporalFollowups(commitments, clock=lambda: now[0].timestamp())
    waits.expect_reply(wait_id='wait', commitment_id=parent['id'], work_id='work', contact_id='Sam',
        outbound_ref='logical-out', source_refs=['owner-task'],
        source_versions={'owner-task': canonical_turn_digest(messages)}, source_session_id='owner-session',
        expected_after_seconds=3600, expires_at=(NOW + 2 * HOUR).timestamp())
    row = waits.acknowledge_dispatch('wait', receipt_ref='accepted', occurred_at=NOW.timestamp())
    receipt = SimpleNamespace(direction='out', channel='whatsapp', receipt_ref='accepted',
        outbound_ref='logical-out', external_ref='provider-out', status='accepted')

    def coverage():
        ingress.observe_coverage(producer='provider', account_id='account', epoch='epoch',
            connected_since=(NOW - HOUR).timestamp(), observed_at=now[0].timestamp(),
            watermark=0, connected=True, unavailable=0, now=now[0].timestamp())

    def issue():
        forecasts.observe_dispatch(waits, row, producer='provider', receipt=receipt, created=True)

    coverage()
    yield SimpleNamespace(now=now, store=store, ledger=ledger, comms=comms, waits=waits, ingress=ingress,
        running=running, issue=issue, coverage=coverage, fid=forecasts._fid('wait'))
    running.close()
    store._conn.close()
    comms._conn.close()


def test_canonical_forecast_is_served_once_and_a_comms_reply_labels_it(callback):
    from colonymind.reports.plasticity import collect, learner_line
    r = callback
    r.issue()
    r.issue()
    binding = r.running.loaded['contacts']
    record, = binding.comparator.records()
    assert record['output'] == record['reference_output'] == .5 and record['outcome'] is None
    r.now[0] += HOUR / 2
    talk(r.comms, 'Sam', 'in', r.now[0])
    binding.driver.poll(r.running)
    assert binding.comparator.records()[0]['outcome']['observed'] == 1
    result, = collect(r.running.config, runtime=r.running)['learners']
    assert result['detail']['forecasts'] == result['detail']['outcomes'] == 1
    assert "Contacts' reply timing: 1 forecasts, 1 outcomes." in learner_line(result)
    assert 'constant 0.5: Brier 0.2500' in learner_line(result)


@pytest.mark.parametrize('late', [False, True])
def test_no_reply_inside_the_forecast_horizon_labels_0(callback, late):
    r = callback
    r.issue()
    binding = r.running.loaded['contacts']
    if late:
        talk(r.comms, 'Sam', 'in', NOW + HOUR * 1.5)                    # after the forecast's horizon
    r.now[0] += HOUR / 2
    binding.driver.poll(r.running)
    assert binding.comparator.records()[0]['outcome'] is None           # neither a reply nor the horizon yet
    r.now[0] += HOUR * 1.5
    binding.driver.poll(r.running)
    assert binding.comparator.records()[0]['outcome']['observed'] == 0


def test_off_forecast_matches_anchor_bytes(callback, monkeypatch):
    r = callback
    original = r.store.issue_forecast
    calls = []
    monkeypatch.setattr(r.store, 'issue_forecast', lambda **kwargs: calls.append(kwargs))
    with monkeypatch.context() as patch:
        patch.setattr(host, '_mind', lambda: None)
        r.issue()
    before = json.dumps(calls.pop(), sort_keys=True)
    # Retained canonical facts are exactly the original fixed forecast's facts.
    assert 'probability' not in forecasts._metadata(r.ledger, r.fid + ':dispatch')
    r.issue()
    assert json.dumps(calls.pop(), sort_keys=True) == before
    monkeypatch.setattr(r.store, 'issue_forecast', original)


def test_learned_issue_retry_keeps_probability_and_closure(callback, monkeypatch):
    from colonymind.plasticity import _io
    from colonymind.plasticity.gates import Check, GateReport
    r = callback
    binding = r.running.loaded['contacts']
    state = binding.driver.learner.base()
    state['W'][0, -1] = .75  # Fixture assumption: a finite positive shift.
    closure = binding.store.revisions.compose(state=_io.state_digest(state), objective=timing.OBJECTIVE.identity,
        calibration=('fixture',), dependents={}, parent=binding.store.revisions.serving().id,
        manifests=(), periods=(), audience='owner', mode='rebuild')
    binding.controller.propose(closure, state)
    binding.controller.consider(closure.id, GateReport(closure.id, 'fixture', (Check('budget', True),)))
    with monkeypatch.context() as patch:
        patch.setattr(r.store, 'issue_forecast', lambda **kwargs: (_ for _ in ()).throw(OSError('interrupted')))
        with pytest.raises(OSError):
            r.issue()
    r.running.rollback('contacts')
    r.now[0] += HOUR / 2
    r.issue()
    first, = r.store.forecast_history(r.fid)['forecasts']
    record, = binding.comparator.records()
    assert first['confidence'] > .5
    assert record['output'] == first['confidence'] and record['reference_output'] == .5
    assert closure.id in first['detail']['conditions']['probability_basis']
    assert first['created_at'] == NOW.timestamp()


from colonymind.memory import curated
def test_inbound_feature_provenance_erases_served_forecast(callback, tmp_path):
    r = callback
    r.ledger.record_source('contact-inbound', contact_id='Sam', session_id='contact-session', messages=[{'role': 'user', 'content': PRIVATE[-1]}], occurred_at=(NOW - HOUR).isoformat(), derive_claims=False)
    lineage, _ = r.comms.source_input('contact-inbound', 'Sam')
    talk(r.comms, 'Sam', 'in', NOW - HOUR, source_lineage=lineage)    # the last inbound the feature reads
    r.issue()
    binding = r.running.loaded['contacts']
    assert 'contact-inbound' in binding.driver.rows.source_ids(binding.driver.rows.ref(r.fid))
    for path in (tmp_path / 'plasticity').rglob('*'):
        if path.is_file():
            assert all((value.encode() not in path.read_bytes() for value in PRIVATE)), path
    r.running.close()
    r.running = service(tmp_path, clock=lambda: r.now[0])
    try:
        from test_curated_memory import historical_bytes
        _native_before_withdrawal = historical_bytes(r.ledger)
        curated.forget_sources(r.ledger, 'Sam', source_ids=['contact-inbound'])
        assert historical_bytes(r.ledger) == _native_before_withdrawal
        assert not r.running.loaded['contacts'].comparator.records()
    finally:
        r.running.close()
