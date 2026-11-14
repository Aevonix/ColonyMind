"""Pooled contact forecasts from the comms ledger: numeric evidence, qualification and clean erasure.

Fixture sizes, dates, channels and planted cycles are test assumptions.
"""
import copy
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from test_plasticity_wiring_runtime import learner_config as load_config
from colonymind.contacts.comms import CommsLog
from colonymind.instinct.outcomes import Sources
from colonymind.plasticity import _io
from colonymind.plasticity.builds import TrainingRow
from colonymind.plasticity.gates import Check, GateReport
from colonymind.plasticity.learners import contact_reply_timing as timing
from colonymind.plasticity.nightly import Runtime
from colonymind.turns import TurnIdempotencyLedger

NOW = datetime(2026, 4, 6, 12, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)
PRIVATE = ('Sam', 'Robin', '+15550000011', 'Please send the garden report.')


def service(path, clock=None, **changes):
    path.mkdir(exist_ok=True)
    declaration = copy.deepcopy(timing.DECLARATION)
    declaration.update(changes)
    (path / 'colonymind.yaml').write_text(yaml.safe_dump({'plasticity': {'learners': {'contacts': declaration}}}))
    return Runtime(load_config(path, environ={}), clock=clock or (lambda: NOW),
                   sources={'contacts': SimpleNamespace(sources=Sources(path, owner_id='owner'))})


def contacts(path, live=('cid-sam', 'cid-robin', 'owner'), deleted=()):
    """A contacts store with these live and soft-deleted contacts (the columns the learner reads)."""
    with closing(sqlite3.connect(path / 'contacts.db')) as db, db:
        db.execute('CREATE TABLE IF NOT EXISTS contacts (contact_id TEXT PRIMARY KEY, deleted_at TEXT)')
        db.executemany('INSERT OR REPLACE INTO contacts VALUES (?,?)',
                       [(ident, None) for ident in live] + [(ident, NOW.isoformat()) for ident in deleted])


def comms(path):
    return CommsLog(str(path / 'colonymind-comms.db'))


def talk(log, contact, direction, at, *, channel='whatsapp:+15550000011', **fields):
    """One conversation row whose summary and channel handle are private words the learner never reads."""
    log.log(contact, channel=channel, direction=direction, summary=PRIVATE[-1], ts=at.isoformat(), **fields)


def dispatch(index, at, contact='Sam', label=None):
    """A send as the snapshot gives it; ``label`` plants a reply inside the horizon (1) or none (0)."""
    return {'id': f'dispatch-{index}', 'contact': contact, 'at': at, 'horizon': at + HOUR, 'channel': 'chat',
            'deps': (f'dispatch-source-{index}',), 'reply_at': at + HOUR / 2 if label else None,
            'reply_deps': (f'inbound-{index}',) if label else ()}


def planted():
    start = NOW - timedelta(weeks=12)
    return [dispatch(i, at, contact=PRIVATE[i % 2], label=int(np.sin(
        2 * np.pi * (at.weekday() * 24 + at.hour) / 168) > 0))
        for i in range(12 * 28) if (at := start + 6 * i * HOUR) < NOW - HOUR]


def frame():
    return {'at': NOW.isoformat(), 'contact': 'keyed-contact', 'reply_share': .5,
            'hours': 168., 'sends': 10, 'channel': 'chat', 'horizon_hours': 1.}


def test_registered_predict_only_anchor_and_pooled_descriptors(tmp_path):
    runtime = service(tmp_path)
    try:
        assert runtime.find('contact_reply_timing') == 'contacts', runtime.rejected
        binding = runtime.loaded['contacts']
        assert binding.declaration['effect'] == 'predict'
        assert binding.store.mode == 'rebuild' and binding.comparator.variant == 'logistic'
        learner = binding.driver.learner
        assert learner.predict(learner.base(), [frame()])[0, 0] == .5
        features = learner.metadata['features']
        assert features == ([{'time': 'how_sin'}, {'time': 'how_cos'}] +
            [{'input': name} for name in ('reply_share', 'hours_since_inbound', 'sends_7d',
                'log_horizon_hours', *(f'channel_{i}' for i in range(7)))] + [{'const': 1}])
        assert all(value not in json.dumps(learner.metadata) for value in PRIVATE)
    finally:
        runtime.close()


def test_features_as_of_the_send_counts_caps_and_numeric_only(tmp_path, monkeypatch):
    runtime = service(tmp_path)
    try:
        rows = runtime.loaded['contacts'].driver.rows
        sends = [dispatch(i, NOW - (i + 2) * HOUR, label=i % 2) for i in range(12)]
        sends += [dispatch('old', NOW - timedelta(days=91), label=True),
                  dispatch('other', NOW - 2 * HOUR, contact='Robin', label=True),
                  dispatch('future', NOW + HOUR, label=True)]
        # A reply recorded after the send must not enter its feature.
        sends[1]['reply_at'] = NOW + HOUR
        inbound = [{'id': 'old-inbound', 'contact': 'Sam', 'at': NOW - 200 * HOUR, 'deps': ('old-inbound',)},
                   {'id': 'future-inbound', 'contact': 'Sam', 'at': NOW + HOUR, 'deps': ('future-inbound',)}]
        current = dispatch('current', NOW)
        values, deps = rows.frame(current, sends, inbound)
        assert values['reply_share'] == 6 / 14
        assert values['hours'] == 168 and values['sends'] == 10
        assert values['horizon_hours'] == 1
        assert 'old-inbound' in deps and 'future-inbound' not in deps
        assert 'inbound-1' not in deps and 'inbound-3' in deps and 'dispatch-source-old' not in deps
        assert all(value not in json.dumps(values) for value in PRIVATE)
        learner = runtime.loaded['contacts'].driver.learner
        examples = [TrainingRow(str(i), 'reply_by_horizon', json.dumps({**values, 'channel': chr(97 + i)}),
                    i % 2, 'realized_value', False, 'logistic', 1., 'owner') for i in range(8)]
        state = learner.fit(learner.base(), examples, objective=timing.OBJECTIVE, seed=0)
        assert learner.channels(state) == list('abcdef')
        inputs = learner.frame({**values, 'channel': 'h'}, learner.channels(state))['inputs']
        assert inputs['channel_6'] == 1 and sum(inputs[f'channel_{i}'] for i in range(7)) == 1
    finally:
        runtime.close()


def test_nightly_pattern_beats_anchor_and_shuffled_control_without_personal_bytes(tmp_path, monkeypatch):
    runtime = service(tmp_path)
    try:
        driver = runtime.loaded['contacts'].driver
        sends = planted()
        monkeypatch.setattr(driver.rows, 'snapshot', lambda: (sends, []))
        result, = runtime.run()
        assert result['result'] == 'promoted', (result, driver.metrics)
        assert driver.metrics['brier_gain_lower'] > 0
        assert driver.metrics['shuffled_gain_lower'] <= 0
        assert driver.last_report.passed
        assert driver.learner.budget_used(runtime.pin('contacts', 'owner').state) <= 2
        for path in (tmp_path / 'plasticity').rglob('*'):
            if path.is_file():
                assert all(value.encode() not in path.read_bytes() for value in PRIVATE), path
    finally:
        runtime.close()


@pytest.mark.parametrize('dependency', ['dispatch-source-8', 'inbound-8'])
@pytest.mark.parametrize('loaded', [True, False])
def test_erasure_unserves_and_rebuilds_equal_clean(tmp_path, monkeypatch, dependency, loaded):
    sends = planted()
    runtime = service(tmp_path)
    try:
        binding = runtime.loaded['contacts']
        monkeypatch.setattr(timing.Rows, 'snapshot', lambda self: (sends, []))
        assert runtime.run()[0]['result'] == 'promoted'
        old = binding.store.revisions.serving()
        if not loaded:
            config = runtime.config
            runtime.close()
            config.data['plasticity']['learners'] = {}
            runtime = Runtime(config, clock=lambda: NOW)
        # The source transaction has committed before the shared erasure hook.
        sends[:] = [row for row in sends if dependency not in (*row['deps'], *row['reply_deps'])]
        runtime.erase_sources([dependency])
        assert not binding.store.revisions.exists(old.id)
        if not loaded:
            assert binding.store.revisions.serving().mode == 'base'
            runtime.close()
            runtime = service(tmp_path)
        actual = runtime.loaded['contacts'].driver.build().state
        clean_path = tmp_path / 'clean'
        clean_path.mkdir()
        (clean_path / 'plasticity').mkdir()
        for name in ('comparator.key', 'erased-refs.json', 'erased-sources.json'):
            path = tmp_path / 'plasticity' / name
            if path.exists():
                (clean_path / 'plasticity' / name).write_bytes(path.read_bytes())
        clean = service(clean_path)
        try:
            clean.run()
            expected = clean.loaded['contacts'].driver.build().state
            assert actual.keys() == expected.keys()
            for key in actual:
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-6, atol=1e-6)
        finally:
            clean.close()
    finally:
        runtime.close()


def test_live_comparator_rolls_back_planted_regression(tmp_path, monkeypatch):
    TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    runtime = service(tmp_path)
    try:
        binding = runtime.loaded['contacts']
        state = binding.driver.learner.base()
        state['W'][0, -1] = 1.
        closure = binding.store.revisions.compose(state=_io.state_digest(state), objective=timing.OBJECTIVE.identity,
            calibration=('fixture',), dependents={}, parent=binding.store.revisions.serving().id,
            manifests=(), periods=(), audience='owner', mode='rebuild')
        binding.controller.propose(closure, state)
        binding.controller.consider(closure.id, GateReport(closure.id, 'fixture', (Check('budget', True),)))
        for index in range(30):
            ref = f'forecast-{index}'
            runtime.serve('contacts', ref, frame(), predict=lambda pin, value:
                float(binding.driver.learner.predict(pin.state, [value])[0, 0]))
            result = runtime.outcome('contacts', ref, observed=0., teacher='realized_value')
        assert result['rolled_back'] and binding.store.revisions.serving().mode == 'base'
        records = binding.comparator.records()
        assert len(records) == 30 and all(row['reference_output'] == .5 for row in records)
        detail = timing.report(runtime.config, 'contacts', runtime=runtime)
        assert detail['forecasts'] == detail['outcomes'] == 30
        assert detail['constant_brier'] == .25
        assert detail['constant_log_loss'] == pytest.approx(np.log(2))
        assert 'Brier' in ' '.join(timing.report_lines(detail))
        monkeypatch.setattr('colonymind.util.temporal.now_utc', lambda: NOW)
        assert timing.report(runtime.config, 'contacts') == detail
    finally:
        runtime.close()


@pytest.mark.parametrize('field', ['reply_share', 'hours', 'sends', 'horizon_hours'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_features_are_refused(field, value):
    learner = timing.ContactReplyLearner('contacts')
    with pytest.raises(ValueError):
        learner.predict(learner.base(), [{**frame(), field: value}])


def test_offline_report_honors_retention_and_erasure(tmp_path, monkeypatch):
    TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    runtime = service(tmp_path)
    try:
        binding = runtime.loaded['contacts']
        for ref in ('old', 'erased', 'current'):
            runtime.serve('contacts', ref, frame(), predict=lambda pin, value: .5)
            runtime.outcome('contacts', ref, observed=1., teacher='realized_value')
        records = binding.comparator.records()
        records[0]['at'] = (NOW - timedelta(days=15)).isoformat()
        _io.write_json(runtime.root / 'erased-refs.json', [records[1]['erasure_hash']])
        path, = binding.comparator.directory.glob('numeric-*.jsonl')
        path.write_bytes(b''.join(_io.canonical(row) + b'\n' for row in records) + b'{"partial":')
        monkeypatch.setattr('colonymind.util.temporal.now_utc', lambda: NOW)
        before = path.read_bytes()
        detail = timing.report(runtime.config, 'contacts')
        assert detail['forecasts'] == detail['outcomes'] == 1
        assert path.read_bytes() == before
        assert detail == timing.report(runtime.config, 'contacts', runtime=runtime)
    finally:
        runtime.close()


# ---------------------------------------------------------------------------
# The comms ledger as the source
# ---------------------------------------------------------------------------

def test_the_snapshot_reads_conversation_rows_of_live_contacts_only(tmp_path):
    contacts(tmp_path, deleted=('cid-gone',))
    log = comms(tmp_path)
    for contact in ('cid-sam', 'cid-gone', 'owner', 'default', 'system'):
        talk(log, contact, 'out', NOW - 30 * HOUR)
        talk(log, contact, 'in', NOW - 29 * HOUR)
    talk(log, 'cid-sam', 'in', NOW - 20 * HOUR, receipt_ref='receipt-1')       # a delivery receipt
    for ref in ('system:notice', 'automated:digest', 'mind:check_in:i-1', 'MIND:x'):
        talk(log, 'cid-sam', 'out', NOW - 10 * HOUR, external_ref=ref)
    runtime = service(tmp_path)
    try:
        sends, inbound = runtime.loaded['contacts'].driver.rows.snapshot()
        assert [(send['contact'], send['at'], send['reply_at'], send['channel']) for send in sends] == [
            ('cid-sam', NOW - 30 * HOUR, NOW - 29 * HOUR, 'whatsapp')]
        assert [(row['contact'], row['at']) for row in inbound] == [('cid-sam', NOW - 29 * HOUR)]
        assert all(value not in repr((sends, inbound)) for value in PRIVATE)   # no summary, no handle
    finally:
        runtime.close()
        log._conn.close()


def test_a_burst_is_one_send_labelled_by_a_reply_within_the_declared_horizon(tmp_path):
    contacts(tmp_path)
    log = comms(tmp_path)
    start = NOW - timedelta(days=6)
    for minutes in (0, 2, 4):                                    # one burst: its last row is the send
        talk(log, 'cid-sam', 'out', start + timedelta(minutes=minutes))
    talk(log, 'cid-sam', 'in', start + 10 * HOUR)                # replied within 24 h
    later = start + timedelta(days=2)
    talk(log, 'cid-sam', 'out', later)
    talk(log, 'cid-sam', 'in', later + 25 * HOUR)                # too late: no reply by the horizon
    split = start + timedelta(days=4)
    talk(log, 'cid-sam', 'out', split)
    talk(log, 'cid-sam', 'out', split + timedelta(minutes=6))    # more than five minutes apart: two sends
    talk(log, 'cid-sam', 'out', NOW - 2 * HOUR)                  # its horizon is still ahead: no label yet
    runtime = service(tmp_path)
    try:
        rows = runtime.loaded['contacts'].driver.rows
        sends, _ = rows.snapshot()
        assert [send['at'] for send in sends] == [start + timedelta(minutes=4), later, split,
                                                  split + timedelta(minutes=6), NOW - 2 * HOUR]
        assert [rows.label(send, NOW) for send in sends] == [1, 0, 0, 0, None]
        examples = list(rows.examples())
        assert [row['label'] for row in examples] == [1, 0, 0, 0]
        assert {json.loads(row['state'])['horizon_hours'] for row in examples} == {24.}
        assert json.loads(examples[1]['state'])['reply_share'] == 2 / 3      # the first send's reply came before
    finally:
        runtime.close()
        log._conn.close()


def test_an_inbound_row_ends_a_burst(tmp_path):
    contacts(tmp_path)
    log = comms(tmp_path)
    at = NOW - timedelta(days=2)
    talk(log, 'cid-sam', 'out', at)
    talk(log, 'cid-sam', 'in', at + timedelta(minutes=1))
    talk(log, 'cid-sam', 'out', at + timedelta(minutes=2))
    runtime = service(tmp_path)
    try:
        rows = runtime.loaded['contacts'].driver.rows
        sends, _ = rows.snapshot()
        assert [(send['at'], rows.label(send, NOW)) for send in sends] == [(at, 1), (at + timedelta(minutes=2), 0)]
    finally:
        runtime.close()
        log._conn.close()


def test_one_contact_across_gateways_and_only_known_platforms_in_the_vocabulary(tmp_path):
    contacts(tmp_path)
    log = comms(tmp_path)
    at = NOW - timedelta(days=3)
    talk(log, 'cid-sam', 'out', at, channel='whatsapp:+15550000011')
    talk(log, 'cid-sam', 'in', at + HOUR, channel='sms:+15550000011')      # one contact on every gateway
    talk(log, 'cid-robin', 'out', at - HOUR, channel='+15550000011')        # a bare handle is no platform
    talk(log, 'cid-robin', 'out', at + 2 * HOUR, channel='carrier-pigeon:loft')
    runtime = service(tmp_path)
    try:
        rows = runtime.loaded['contacts'].driver.rows
        sends, _ = rows.snapshot()
        assert [(send['contact'], send['channel'], rows.label(send, NOW)) for send in sends] == [
            ('cid-robin', 'other', 0), ('cid-sam', 'whatsapp', 1), ('cid-robin', 'other', 0)]
    finally:
        runtime.close()
        log._conn.close()


def test_a_merge_moves_the_rows_to_the_kept_contact(tmp_path):
    contacts(tmp_path, deleted=('cid-old',))
    log = comms(tmp_path)
    at = NOW - timedelta(days=3)
    talk(log, 'cid-old', 'out', at)
    talk(log, 'cid-old', 'in', at + HOUR)
    runtime = service(tmp_path)
    try:
        rows = runtime.loaded['contacts'].driver.rows
        assert rows.snapshot() == ([], [])                       # a merged-away contact is deleted
        assert log.reattribute('cid-old', 'cid-sam') == 2
        sends, inbound = rows.snapshot()
        assert [(send['contact'], rows.label(send, NOW)) for send in sends] == [('cid-sam', 1)]
    finally:
        runtime.close()
        log._conn.close()


def test_below_30_training_and_30_gate_labels_nothing_is_promoted(tmp_path):
    contacts(tmp_path)
    log = comms(tmp_path)
    for day in range(3, 9):
        talk(log, 'cid-sam', 'out', NOW - timedelta(days=day))
    runtime = service(tmp_path)
    try:
        result, = runtime.run()
        assert result['result'] != 'promoted'
        assert runtime.loaded['contacts'].store.revisions.serving().mode == 'base'
        assert runtime.loaded['contacts'].driver.label_counts(runtime.loaded['contacts'].last_build)['training'] == 6
    finally:
        runtime.close()
        log._conn.close()
