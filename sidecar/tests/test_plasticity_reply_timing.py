"""Reply timing's live learner, references, numeric gates and erasure."""
import copy
from dataclasses import replace
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from test_plasticity_wiring_runtime import learner_config as load_config
from colonymind.initiatives.store import InitiativeStore
from colonymind.instinct.outcomes import Sources
from colonymind.plasticity import _io
from colonymind.plasticity.learners.reply_timing import DECLARATION, OBJECTIVE, ReplyLearner, gate_week
from colonymind.mind import outreach
from colonymind.plasticity.nightly import Runtime
from colonymind.plasticity.numeric import Revision
from colonymind.turns.idempotency import TurnIdempotencyLedger

NOW = datetime(2026, 4, 6, 12, tzinfo=timezone.utc)
H = timedelta(hours=1)
WORDS = 'Owner asked Sam about the garden.'


def runtime(path, now, **changes):
    path.mkdir(exist_ok=True)
    spec = copy.deepcopy(DECLARATION)
    spec.update(changes)
    (path / 'colonymind.yaml').write_text(yaml.safe_dump({'plasticity': {'learners': {'timing': spec}}}))
    sources = Sources(path, owner_id='Owner')
    return Runtime(load_config(path, environ={}), sources={'timing': SimpleNamespace(sources=sources)},
                   clock=lambda: now[0])


def turn(ledger, ref, at, channel='a', owner='Owner'):
    ledger.record_source(ref, contact_id=owner, session_id='session-' + ref,
                         messages=[{'role': 'user', 'content': WORDS}], occurred_at=at.isoformat(),
                         channel_id=channel + ':5550100', derive_claims=False)


def send(store, at, *, status='sent', kind='outreach_finding'):
    row, _ = store.create_intention(kind='message', type=kind, title='A garden update', drive='social',
        cls='owner', decision='act', decision_reason='fixture', status=status, dedup_key=None,
        recipient='Owner', hermes_kind='message', created_at=at)
    return store.update(row.id, completed_at=at) if status == 'sent' else row


def test_registered_driver_and_logistic_golden(tmp_path):
    service = runtime(tmp_path, [NOW])
    try:
        assert service.find('reply_timing') == 'timing', service.rejected
        assert service.loaded['timing'].comparator.variant == 'logistic'
        assert service.loaded['timing'].store.mode == 'rebuild'
    finally:
        service.close()
    metadata = dict(schema='numeric-revision-1', learner='timing', variant='logistic', tz='UTC',
        features=[{'time': 'how_sin'}, {'time': 'how_cos'}, {'input': 'hours_since_owner_turn'},
                  {'input': 'channel=a'}, {'const': 1}], outputs=[{'event': 'reply_60m'}],
        standardisation={'hours_since_owner_turn': {'mean': 4., 'scale': 4.},
                         'channel=a': {'mean': 0., 'scale': 1.}}, anchor={'type': 'constant', 'p': .5})
    revision = Revision(metadata, np.array([[.2, -.1, -.3, .6, 0.]], dtype=np.float32))
    result = revision.forecast(at='2026-01-05T00:00:00Z', inputs={'hours_since_owner_turn': 8., 'channel=a': 1.})
    assert result['reply_60m'] == pytest.approx(.5498340, rel=1e-6)


def test_rows_wait_for_window_withheld_and_dependencies(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    store = InitiativeStore(state_dir=tmp_path)
    turn(ledger, 'last-turn', NOW - 3 * H)
    turn(ledger, 'label-turn', NOW - H / 2)
    turn(ledger, 'contact-turn', NOW - H / 4, owner='Sam')
    old = send(store, NOW - H)
    young = send(store, NOW - H / 2)
    send(store, NOW - 2 * H, status='approved')
    service = runtime(tmp_path, [NOW])
    try:
        driver = service.loaded['timing'].driver
        rows = list(driver.source.rows(driver.objective))
        row = next(r for r in rows if old.id in driver.rows.source_ids(r['ref']))
        assert row['label'] == 1.
        assert set(driver.rows.source_ids(row['ref'])) == {old.id, 'last-turn', 'label-turn'}
        assert row['ref'] in driver.rows.refs_resting_on(['last-turn'])
        assert not any(young.id in driver.rows.source_ids(r['ref']) for r in rows)
        assert all(WORDS not in r['state'] for r in rows)
        # The same suppression contract as outcomes.sound.
        driver.rows.sources.withheld = lambda ids: {'label-turn'} & set(ids)
        hidden = list(driver.source.rows(driver.objective))
        assert not any(old.id in driver.rows.source_ids(r['ref']) for r in hidden)
    finally:
        service.close()
        store.close()
    assert all(WORDS.encode() not in p.read_bytes() for p in (tmp_path / 'plasticity').rglob('*') if p.is_file())


def weekly(ledger, *, weeks=12):
    start = NOW - timedelta(weeks=weeks)
    # A balanced weekly cycle; times and channel are the only input signal.
    for i in range(weeks * 168):
        at = start + i * H
        if np.sin(2 * np.pi * (at.weekday() * 24 + at.hour) / 168) > 0:
            turn(ledger, f'turn-{i}', at + H / 2)


def test_nightly_weekly_pattern_and_shuffled_control(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    weekly(ledger)
    InitiativeStore(state_dir=tmp_path).close()
    service = runtime(tmp_path, [NOW])
    try:
        result, = service.run()
        assert result['result'] == 'promoted', (result, service.loaded['timing'].driver.metrics)
        driver = service.loaded['timing'].driver
        assert driver.last_report.passed
        assert driver.metrics['brier_gain_lower'] > 0
        assert driver.metrics['shuffled_gain_lower'] <= 0
        assert driver.learner.budget_used(service.pin('timing', 'owner').state) <= 2.
    finally:
        service.close()


def test_served_once_closed_window_outcome_and_report(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    store = InitiativeStore(state_dir=tmp_path)
    turn(ledger, 'last', NOW - H)
    row = send(store, NOW)
    now = [NOW]
    service = runtime(tmp_path, now)
    try:
        driver = service.loaded['timing'].driver
        driver.poll(service)
        driver.poll(service)
        records = service.loaded['timing'].comparator.records()
        assert len(records) == 1 and records[0]['reference_output'] == .5
        assert records[0]['outcome'] is None
        turn(ledger, 'reply', NOW + H / 2)
        now[0] += H
        driver.poll(service)
        record, = service.loaded['timing'].comparator.records()
        assert record['outcome']['observed'] == 1.
        assert {'last', 'reply', row.id} <= set(driver.rows.source_ids(row.id))
        from colonymind.reports.plasticity import collect, lines
        text = ' '.join(lines(collect(service.config, runtime=service)))
        assert 'Reply timing' in text and 'Brier' in text and 'log loss' in text
        assert '1 served' in text and '1 outcomes' in text
    finally:
        service.close()
        store.close()


def test_forced_share_records_probability_and_weights():
    from colonymind.plasticity.forced import ForcedRuns
    choices = [ForcedRuns(b'fixture key', learner='timing').choose(f'item-{i}', eligible=True)
               for i in range(10000)]
    # Normal binomial 99% interval, fixed fixture size, not a tuning criterion.
    assert abs(sum(item.forced for item in choices) - 500) <= 2.576 * np.sqrt(10000 * .05 * .95)
    assert all(item.manifest_row()['probability'] == .05 for item in choices)
    learner = ReplyLearner('timing')
    from colonymind.plasticity.builds import TrainingRow
    from colonymind.plasticity.learners.reply_timing import OBJECTIVE
    state = json.dumps({'at': NOW.isoformat(), 'channel': 'a', 'hours': 4., 'outreach': 1., 'probability': .05})
    row = TrainingRow('forced', 'reply_60m', state, 1., 'realized_value', False, 'logistic', 1., 'owner')
    forced = learner.fit(learner.base(), [row], objective=OBJECTIVE, seed=0)
    normal = learner.fit(learner.base(), [replace(row, state=state.replace('0.05', '1.0'))], objective=OBJECTIVE, seed=0)
    assert np.linalg.norm(forced['W']) > np.linalg.norm(normal['W'])


def test_logistic_comparator_rolls_back_real_closure(tmp_path):
    from colonymind.plasticity.gates import Check, GateReport
    service = runtime(tmp_path, [NOW])
    try:
        binding = service.loaded['timing']
        state = binding.driver.learner.base()
        state['W'][0, -1] = 1.
        closure = binding.store.revisions.compose(state=_io.state_digest(state), objective=OBJECTIVE.identity,
            calibration=('fixture',), dependents={}, parent=binding.store.revisions.serving().id,
            manifests=(), periods=(), audience='owner', mode='rebuild')
        binding.controller.propose(closure, state)
        binding.controller.consider(closure.id, GateReport(closure.id, 'fixture', (Check('budget', True),)))
        for i in range(30):
            frame = {'at': NOW.isoformat(), 'hours': 4., 'channel': 'a', 'outreach': 1.}
            service.serve('timing', f'notice-{i}', frame,
                predict=lambda pin, frame: float(binding.driver.learner.predict(pin.state, [frame])[0, 0]))
            result = service.outcome('timing', f'notice-{i}', observed=0., teacher='realized_value')
        assert result['rolled_back']
        assert binding.store.revisions.serving().mode == 'base'
        from colonymind.reports.plasticity import collect, lines
        assert 'Rolled back' in ' '.join(lines(collect(service.config, runtime=service)))
    finally:
        service.close()


def test_prepare_persists_first_hold_forces_once_and_releases(tmp_path):
    from colonymind.mind.concerns import Concerns
    from colonymind.plasticity.learners.reply_timing import HOLD_PREFIX, SAMPLE_PREFIX, prepare
    from test_mind_outreach import declared, finding, inputs
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    turn(ledger, 'last', NOW - H)
    InitiativeStore(state_dir=tmp_path).close()
    now = [NOW]
    service = runtime(tmp_path, now, forced_run_share=1.)
    concerns = Concerns(tmp_path / 'mind.db')
    mind = SimpleNamespace(plasticity=service, faculties={'plasticity': True}, owner_id='Owner',
                           mind_state=concerns.state)
    try:
        binding = service.loaded['timing']
        # Use the real prepare path and a fixture pin to isolate the hold from nightly qualification.
        service.pin = lambda *a: SimpleNamespace(closure=SimpleNamespace(mode='rebuild'), state={})
        binding.driver.learner.predict = lambda state, frames: np.array([[.2 if
            datetime.fromisoformat(frames[0]['at']).hour == NOW.hour else .8]])
        state = inputs(now=NOW, interests=[declared()], findings=[finding()])
        prepare(mind, state)
        assert len(outreach.candidates(state)[1]) == 1
        assert len(concerns.state.items(HOLD_PREFIX)) == 1
        sample, = concerns.state.items(SAMPLE_PREFIX)
        assert json.loads(sample['text'])['probability'] == 1.
        assert json.loads(sample['text'])['forced']
        first = concerns.state.items(HOLD_PREFIX)[0]['text']
        prepare(mind, inputs(now=NOW, interests=[declared()], findings=[finding()]))
        assert len(concerns.state.items(SAMPLE_PREFIX)) == 1
        assert concerns.state.items(HOLD_PREFIX)[0]['text'] == first
        now[0] += 2 * H
        later = inputs(now=now[0], interests=[declared()], findings=[finding()])
        prepare(mind, later)
        assert outreach.candidates(later)[1]
        now[0] += timedelta(days=1)
        prepare(mind, inputs(now=now[0]))
        assert not concerns.state.items(HOLD_PREFIX)
    finally:
        service.close()
        concerns.close()


def test_channels_cap_and_failed_or_requested_delivery_are_not_labels(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    store = InitiativeStore(state_dir=tmp_path)
    turn(ledger, 'old', NOW - 100 * H)
    send(store, NOW - 2 * H, status='failed')
    send(store, NOW - 2 * H, kind='outreach_answer')
    now = [NOW]
    service = runtime(tmp_path, now)
    try:
        driver = service.loaded['timing'].driver
        rows = list(driver.source.rows(OBJECTIVE))
        assert rows and all(json.loads(r['state'])['outreach'] == 0. for r in rows)
        assert max(json.loads(r['state'])['hours'] for r in rows) == 72.
        from colonymind.plasticity.builds import TrainingRow
        examples = []
        for index in range(8):
            frame = {'at': (NOW - index * H).isoformat(), 'channel': chr(97 + index),
                     'hours': 4., 'outreach': 0.}
            examples.append(TrainingRow(str(index), 'reply_60m', json.dumps(frame), 1,
                                        'realized_value', False, 'logistic', 1., 'owner'))
        learned = driver.learner.fit(driver.learner.base(), examples, objective=OBJECTIVE, seed=0)
        assert driver.learner.channels(learned) == list('abcdef')
        frame['channel'] = 'h'
        features = driver.learner.frame(frame, driver.learner.channels(learned))['inputs']
        assert features['channel_6'] == 1. and sum(features[f'channel_{i}'] for i in range(7)) == 1.
    finally:
        service.close()
        store.close()


def test_hold_metadata_uses_current_hour_mark_and_report_reads_without_runtime(tmp_path):
    from colonymind.reports.plasticity import collect, lines
    now = [NOW]
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    turn(ledger, 'last', NOW - H)
    store = InitiativeStore(state_dir=tmp_path)
    service = runtime(tmp_path, now)
    row = send(store, NOW)
    store.update(row.id, context={'reply_timing': {'probability': 1., 'hour_mark': 1.}})
    try:
        service.loaded['timing'].driver.poll(service)
        now[0] += H
        service.loaded['timing'].driver.poll(service)
        data = collect(service.config)
        metrics = data['learners'][0]['reply_timing']
        assert metrics['hour_mark_brier'] == pytest.approx(0., abs=1e-10)
        assert metrics['constant_brier'] == .25
        assert 'Reply timing' in ' '.join(lines(data))
    finally:
        service.close()
        store.close()


def test_unreadable_timing_data_leaves_outreach_off_arm(tmp_path, monkeypatch):
    from colonymind.mind.concerns import Concerns
    from colonymind.plasticity.learners.reply_timing import prepare
    from test_mind_outreach import declared, finding, inputs
    import sqlite3
    service = runtime(tmp_path, [NOW])
    concerns = Concerns(tmp_path / 'mind.db')
    mind = SimpleNamespace(plasticity=service, faculties={'plasticity': True}, owner_id='Owner',
                           mind_state=concerns.state)
    state = inputs(interests=[declared()], findings=[finding()])
    baseline = outreach.candidates(state)
    def unavailable():
        raise sqlite3.OperationalError('fixture unavailable ledger')
    monkeypatch.setattr(service.loaded['timing'].driver.rows, 'snapshot', unavailable)
    try:
        prepare(mind, state)
        assert state.reply_p == {} and outreach.candidates(state) == baseline
    finally:
        service.close()
        concerns.close()


def test_outreach_preparation_reads_the_sources_once(tmp_path, monkeypatch):
    from colonymind.mind.concerns import Concerns
    from colonymind.plasticity.learners.reply_timing import prepare
    from test_mind_outreach import declared, finding, inputs
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    turn(ledger, 'last-turn', NOW - 3 * H)
    service = runtime(tmp_path, [NOW])
    concerns = Concerns(tmp_path / 'mind.db')
    mind = SimpleNamespace(plasticity=service, faculties={'plasticity': True}, owner_id='Owner',
                           mind_state=concerns.state)
    state = inputs(interests=[declared()], findings=[finding()])
    rows = service.loaded['timing'].driver.rows
    snapshot, reads = rows.snapshot, []
    monkeypatch.setattr(rows, 'snapshot', lambda: reads.append(snapshot()) or reads[-1])
    try:
        prepare(mind, state)
        assert len(reads) == 1 and [turn['id'] for turn in reads[0][0]] == ['last-turn']
        assert set(state.timing_holds.decisions) == {c.dedup_key for c in outreach.candidates(state)[1]}
    finally:
        service.close()
        concerns.close()


def test_report_hold_counts_expire_even_without_another_outreach_tick(tmp_path):
    from colonymind.mind.concerns import Concerns
    from colonymind.plasticity.learners.reply_timing import SAMPLE_PREFIX, report
    service = runtime(tmp_path, [NOW])
    concerns = Concerns(tmp_path / 'mind.db')
    try:
        concerns.state.set(SAMPLE_PREFIX + 'old', text=json.dumps({'forced': False}), now=NOW - 25 * H)
        metrics = report(service.config, 'timing', runtime=service)
        assert metrics['holds'] == 0
    finally:
        concerns.close()
        service.close()
