"""Owner reply timing on kind D, with a bounded discretionary hold.

Plan assumptions: .5 anchor, 60-minute labels, six channels plus other, 72-hour
last-turn cap, two-hour hold, .1 gain and five percent forced runs. Builder
assumptions: 90-day replay, every fourth UTC week reserved for gates, 30 gate
items, rate .001, coordinate clip .5, two-log-odds lifetime fence, eight seeds starting at 7 for the
shuffled control. These weights never change authority or an outreach budget.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import math
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import replace
from datetime import timedelta

import numpy as np

from ...instinct.outcomes import INITIATIVES_DB, TURN_DB, Sources
from ...mind import outreach
from ...mind.rank import OUTREACH_TYPES
from .. import _io, builds
from ..admission import Admission
from ..budgets import Budget, BudgetCheck
from ..certificates import GateRecord
from ..forced import ForcedRuns
from ..gates import Check, GateReport, Item, Z95
from ..locking import StateLockBusy
from ..nightly import register_factory
from ..numeric import NumericLearner, moment
from ..numeric_comparator import configure
from ..objectives import Objective
from ..synthetic import Cycle

logger = logging.getLogger(__name__)

HOUR = timedelta(hours=1)
GATE_MIN_ROWS = 30
TRAINING_MIN_ROWS = 1  # Existing calibration requires build.rows > 0.
SHUFFLE_SEEDS = range(7, 15)
PARITY_RTOL = 1e-6
PARITY_ATOL = 1e-8
HOLD_PREFIX = 'outreach.timing_hold:'
SAMPLE_PREFIX = 'outreach.timing_sample:'
OBJECTIVE = Objective(name='reply_timing', model='numeric_predictor', kind='numeric-online',
    target_type='binary', losses={'realized_value': 'logistic'}, blend={'realized_value': 1.},
    points=('reply_60m',), text=False)
DECLARATION = {
    'kind': 'numeric-online', 'implementation': 'reply_timing', 'base': 'reply-constant@1',
    'teachers': {'realized_value': 'binary'}, 'model': {'roles': ['numeric_predictor'], 'change': 'weights'},
    'budget': {'type': 'anchored_output', 'bound': 2.}, 'effect': 'hold', 'audience': 'owner',
    'personal_data': True, 'erasure_path': 'source erasure and clean rebuild', 'dependents': [],
    'mode': 'rebuild', 'forced_run_share': .05, 'learning_rate': .001, 'clip': .5,
    'retention_margin': 1e-12,
    'gates': ['retention', 'specificity', 'canaries', 'calibration', 'budget', 'finite', 'served_offline'],
    'arms': ['constant', 'hour_mark', 'time_shuffled'],
    'comparator': {'window_items': 200, 'window_hours': 168, 'bar': .02},
}


def keyed(key, value):
    return hmac.new(key, _io.canonical(value), hashlib.sha256).hexdigest()


def gate_week(at):
    """A fixed split cannot move a surviving row into training after erasure."""
    return int(moment(at).timestamp() // (7 * 86400)) % 4 == 0


class ReplyLearner(NumericLearner):
    """Kind D with a revision-owned channel vocabulary and sampling weights.

    Channels are packed as UTF-8 bytes in the closure's numeric state so the
    vocabulary rolls back and erases with W. Slots are filled from the six most
    frequent distinct last turns in training only, never from gate labels.
    The numeric step, fence, loss and comparison are the shared implementation.
    """
    def __init__(self, name, *, rate=.001, clip=.5, bound=2.):
        names = ['hours_since_owner_turn', *[f'channel_{i}' for i in range(7)], 'outreach']
        standard = {name: {'mean': 0., 'scale': 1.} for name in names}
        standard['hours_since_owner_turn'] = {'mean': 36., 'scale': 12.}
        super().__init__(dict(schema='numeric-revision-1', learner=name, variant='logistic', tz='UTC',
            features=[{'time': t} for t in ('how_sin', 'how_cos', 'hod_sin', 'hod_cos')]
                     + [{'input': name} for name in names] + [{'const': 1}],
            outputs=[{'event': 'reply_60m'}], standardisation=standard, anchor={'type': 'constant', 'p': .5}),
            rate=rate, clip=clip, bound=bound)

    def base(self):
        return {**super().base(), 'channels': np.frombuffer(b'[]', dtype=np.uint8).copy()}

    def channels(self, state):
        return json.loads(np.asarray(state.get('channels', self.base()['channels']), dtype=np.uint8).tobytes())

    def frame(self, frame, channels):
        frame = json.loads(frame) if isinstance(frame, str) else frame
        channel = frame['channel']
        slot = channels.index(channel) if channel in channels else 6
        return {'at': frame['at'], 'inputs': {'hours_since_owner_turn': min(72., max(0., frame['hours'])),
                'outreach': float(frame['outreach']), **{f'channel_{i}': float(i == slot) for i in range(7)}}}

    def fit(self, start, rows, *, objective, seed):
        distinct = {(f.get('last_at', f['at']), f['channel']) for r in rows if (f := json.loads(r.state))}
        counts = Counter(channel for _, channel in distinct if channel != 'other')
        channels = sorted(counts, key=lambda name: (-counts[name], name))[:6]
        transformed = []
        for row in rows:
            frame = json.loads(row.state)
            probability = frame.get('probability', 1.)
            if not 0 < probability <= 1:
                raise ValueError('reply timing sampling probability must be in (0, 1]')
            transformed.append(replace(row, state=json.dumps(self.frame(frame, channels)),
                                       weight=row.weight / probability))
        state = super().fit(start, transformed, objective=objective, seed=seed)
        state['channels'] = np.frombuffer(_io.canonical(channels), dtype=np.uint8).copy()
        return state

    def predict(self, state, states):
        state = state or self.base()
        return super().predict(state, [self.frame(frame, self.channels(state)) for frame in states])


class Rows:
    """Re-read times and channel names, never message contents or cached labels."""
    def __init__(self, store, sources, clock):
        self.store, self.sources, self.clock = store, sources, clock
        self.dependencies = {}

    @property
    def key(self):
        return (self.store.root / 'comparator.key').read_bytes()

    def query(self, filename, sql, args=()):
        path = (self.sources.state_dir / filename).resolve()
        if not path.is_relative_to(self.sources.state_dir.resolve()):
            raise ValueError('reply timing sources must stay in the state directory')
        if not path.exists():
            return []
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(sql, args)]

    def snapshot(self):
        if not self.sources.owner_id:
            return [], []
        # One row for all the turns: fetched row by row, each row hands the interpreter lock over and back.
        found = self.query(TURN_DB, "SELECT json_group_array(json_object('id', id, 'at', at, 'channel', channel)) "
            "AS turns FROM (SELECT s.turn_id AS id, coalesce(s.occurred_at,s.ingested_at) AS at, "
            "coalesce(c.platform,'other') AS channel FROM turn_sources s LEFT JOIN source_channels c "
            "ON c.turn_id=s.turn_id WHERE s.contact_id=? AND s.scope='person' AND s.session_id<>'mind' "
            "AND s.turn_id NOT LIKE 'mind:%' AND EXISTS (SELECT 1 FROM json_each(s.messages_json) m "
            "WHERE json_extract(m.value,'$.role')='user'))", (self.sources.owner_id,))
        turns = json.loads(found[0]['turns']) if found else []
        hidden = self.sources.withheld(t['id'] for t in turns) | self.sources.erased(t['id'] for t in turns)
        for turn in turns:
            turn['at'], turn['hidden'] = moment(turn['at']), turn['id'] in hidden
        turns = sorted((t for t in turns if t['at'] <= self.clock()), key=lambda t: (t['at'], t['id']))
        sends = self.query(INITIATIVES_DB, "SELECT id,type,status,created_at,completed_at, "
            "json_extract(context,'$.reply_timing.probability') AS probability, "
            "json_extract(context,'$.reply_timing.hour_mark') AS hour_mark, verdict "
            "FROM initiatives WHERE kind='message' AND entity_id=? AND "
            "type IN ('outreach_finding','outreach_loop','outreach_care') "
            "AND coalesce(json_extract(context,'$.requested'),0)=0", (self.sources.owner_id,))
        return turns, sends

    def frame(self, at, turns, *, outreach_flag, probability=1.):
        previous = next((turn for turn in reversed(turns) if turn['at'] <= at), None)
        if previous is None or previous['hidden']:
            return None, ()
        return {'at': at.isoformat(), 'hours': min(72., (at - previous['at']).total_seconds() / 3600),
                'last_at': previous['at'].isoformat(), 'channel': previous['channel'],
                'outreach': float(outreach_flag), 'probability': probability}, (previous['id'],)

    def labelled(self, at, turns, *, outreach_flag, origin, probability=1.):
        frame, deps = self.frame(at, turns, outreach_flag=outreach_flag, probability=probability)
        replies = [t for t in turns if at < t['at'] <= at + HOUR]
        dependencies = tuple(dict.fromkeys((*origin, *deps, *(t['id'] for t in replies))))
        if frame is None or at + HOUR > self.clock() or any(t['hidden'] for t in replies):
            return None, dependencies
        ref = keyed(self.key, ['reply-row', at.isoformat(), *origin])
        self.dependencies[ref] = dependencies
        state = _io.canonical(frame).decode()
        return {'ref': ref, 'point': 'reply_60m', 'version': 'reply-timing-1', 'state': state,
                'state_hash': keyed(self.key, frame), 'label': int(bool(replies)), 'teacher': 'realized_value',
                'push_away': False, 'source_kind': 'resolver', 'audience': 'owner', 'at': at.isoformat()}, dependencies

    def examples(self):
        turns, sends = self.snapshot()
        cutoff = self.clock() - timedelta(days=90)
        for send in sends:
            if send['status'] != 'sent' or not send['completed_at']:
                continue
            at = moment(send['completed_at'])
            row, deps = self.labelled(at, turns, outreach_flag=True, origin=(send['id'],),
                                      probability=send['probability'] or 1.)
            self.dependencies[send['id']] = deps
            if row is not None and at >= cutoff:
                yield row
        if not turns:
            return
        at = max(cutoff, turns[0]['at']).replace(minute=0, second=0, microsecond=0)
        while at + HOUR <= self.clock():
            # An availability row is not a second copy of a pending outreach window.
            pending = any(moment(s['created_at']) <= at and
                (s['status'] in {'approved', 'sending', 'uncertain'} or
                 (s['status'] == 'sent' and s['completed_at'] and at < moment(s['completed_at']) + HOUR))
                for s in sends)
            if not pending:
                row, _ = self.labelled(at, turns, outreach_flag=False, origin=())
                if row is not None:
                    yield row
            at += HOUR

    def rows(self, objective, *, since=None):
        yield from (r for r in self.examples() if (not since or moment(r['at']) >= moment(since))
                    and (not objective.points or r['point'] in objective.points))

    def source_of(self, row):
        return None

    def source_ids(self, ref):
        return self.dependencies.get(ref)

    def refs_resting_on(self, ids):
        # Saved forward provenance in LearnerStore covers rows whose turn has
        # already been deleted when the shared callback runs.
        wanted = set(ids)
        return [ref for ref, deps in self.dependencies.items() if wanted.intersection(deps)]

    def erase(self, refs):
        for ref in refs:
            self.dependencies.pop(ref, None)


class Driver(Cycle):
    def __init__(self, *, store, controller, source, declaration, clock):
        margin = declaration.get('retention_margin', 1e-12)
        try:
            valid_margin = (not isinstance(margin, bool) and isinstance(margin, (int, float))
                            and math.isfinite(margin) and margin >= 0)
        except OverflowError:
            valid_margin = False
        if not valid_margin:
            raise ValueError('reply_timing: retention_margin must be non-negative and finite; '
                             f'received {margin!r} (type {type(margin).__name__})')
        self.retention_margin = float(margin)
        for field in ('mode', 'personal_data', 'erasure_path', 'effect', 'audience', 'dependents', 'teachers'):
            if declaration.get(field) != DECLARATION[field]:
                raise ValueError('reply_timing: invalid ' + field)
        if declaration['budget']['type'] != 'anchored_output' or not 0 < declaration['budget']['bound'] <= 2.:
            raise ValueError('reply_timing: anchored_output bound must be at most 2 log-odds')
        self.store, self.controller, self.source, self.clock = store, controller, source, clock
        sources = getattr(source.source, 'sources', None)
        if sources is None:
            raise ValueError('reply_timing requires the instance source ledger')
        self.rows = Rows(store, sources, clock)
        source.source = self.rows
        self.objective = OBJECTIVE
        self.learner = ReplyLearner(store.name, rate=declaration.get('learning_rate', .001),
                                   clip=declaration.get('clip', .5), bound=declaration['budget']['bound'])
        identity_path = store.directory / 'reply-layout.json'
        identity = {'identity': self.learner.identity}
        if identity_path.exists() and _io.read_json(identity_path) != identity:
            raise ValueError('reply_timing: settings changed; declare a new learner identity')
        _io.write_json(identity_path, identity)
        self.forced = ForcedRuns(self.rows.key, learner=store.name, share=declaration['forced_run_share'])
        configure(controller, declaration, variant='logistic')
        self.admission = Admission(self.objective, learner=store.name, source_of=source.source_of,
                                   protected_refs=self.protected, clock=clock)
        if store.budgets.declared(store.name) is None:
            store.budgets.declare(Budget(store.name, 'anchored_output', self.learner.bound))
        self.last_period = self.last_report = None
        self.metrics = {}
        start_path = store.directory / 'reply-start.json'
        if not start_path.exists():
            _io.write_json(start_path, {'at': clock().isoformat()})
        self.started_at = moment(_io.read_json(start_path)['at'])
        self.canaries = (Item('missing-input', '{"at":"2026-01-01T00:00:00Z"}', 0.),)

    def protected(self):
        return frozenset(r['ref'] for r in self.rows.examples() if gate_week(r['at']))

    def propose(self, build):
        closure = self.store.revisions.compose(state=build.digest, objective=self.objective.identity,
            calibration=('reply-' + build.digest,), dependents={}, parent=self.store.revisions.serving().id,
            manifests=build.manifests, periods=tuple(p.id for p in self.store.periods.retained()),
            audience=build.audience, mode='rebuild')
        self.controller.propose(closure, build.state)
        self.store.budgets.record(BudgetCheck(self.store.name, build.digest, 'anchored_output',
                                             self.learner.bound, self.learner.budget_used(build.state)))
        return closure

    def gate(self, closure, build):
        record = GateRecord(learner=self.store.name, revision=closure.id, state=build.digest,
            serving_revision=self.store.revisions.serving().id, at=_io.stamp(self.clock()), loss='brier',
            arms=dict.fromkeys(('candidate', 'serving', 'anchor', 'shuffled')),
            comparisons=dict.fromkeys(('anchor_candidate', 'serving_candidate', 'shuffled_candidate',
                                       'anchor_shuffled')),
            counts={'gate': None, 'training': build.rows, 'control_training': None,
                    'shuffles': len(SHUFFLE_SEEDS)},
            thresholds={'gate_min_rows': GATE_MIN_ROWS, 'training_min_rows': TRAINING_MIN_ROWS, 'z': Z95, 'sd_ddof': 1,
                'retention_margin': self.retention_margin, 'anchor_gain_lower': {'gt': 0.},
                'serving_gain_lower': {'ge': -self.retention_margin}, 'specificity_gain_lower': {'gt': 0.},
                'canary_exposure': 'equal_to_base', 'budget_type': 'anchored_output',
                'budget_bound': self.learner.bound, 'finite': 'all_state_values',
                'parity_rtol': PARITY_RTOL, 'parity_atol': PARITY_ATOL, 'parity_equal_nan': False,
                'shuffle_seeds': list(SHUFFLE_SEEDS)})
        try:
            return self._gate(closure, build, record)
        except Exception as error:
            record.error = type(error).__name__
            record.reason = 'gate evaluation error'
            raise
        finally:
            # Persist before the controller can act on the returned report.
            self.store.certificates.write(record)

    def _gate(self, closure, build, record):
        rows = [r for r in self.source.rows(self.objective) if gate_week(r['at'])
                and not json.loads(r['state'])['outreach']]
        record.counts['gate'] = len(rows)
        record.gate_set = 'reply-' + _io.digest([r['ref'] for r in rows])
        enough = len(rows) >= GATE_MIN_ROWS
        checks = []
        agree = False
        if enough:
            states, truth = [r['state'] for r in rows], np.array([r['label'] for r in rows])
            predicted = self.learner.predict(build.state, states)[:, 0]
            previous = self.learner.predict(self._start()[1], states)[:, 0]
            training = builds.training_rows(self.objective, self.store.current_manifests(), self.source)
            record.counts['control_training'] = len(training)
            training.sort(key=lambda row: (moment(json.loads(row.state)['at']), row.point))
            control_losses = []
            for seed in SHUFFLE_SEEDS:
                order = np.random.default_rng(seed).permutation(len(training))
                shuffled = [replace(row, label=training[int(i)].label) for row, i in zip(training, order)]
                control = self.learner.fit(self.learner.base(), shuffled, objective=self.objective, seed=seed)
                control_losses.append((self.learner.predict(control, states)[:, 0] - truth) ** 2)
            control_losses = np.array(control_losses)
            control_loss = control_losses.mean(axis=0)
            loss = (predicted - truth) ** 2

            def lower(delta):
                return float(delta.mean() - Z95 * delta.std(ddof=1) / math.sqrt(len(delta)))

            def measured(value):
                return float(value) if math.isfinite(value) else None

            def comparison(delta, bound):
                return {'n': len(delta), 'mean': measured(delta.mean()), 'sd': measured(delta.std(ddof=1)),
                        'se': measured(delta.std(ddof=1) / math.sqrt(len(delta))),
                        'lower': measured(bound)}

            previous_loss = (previous - truth) ** 2
            record.arms = {name: measured(values.mean()) for name, values in
                           (('candidate', loss), ('serving', previous_loss),
                            ('anchor', np.full(len(rows), .25)), ('shuffled', control_loss))}
            deltas = {'anchor_candidate': .25 - loss, 'serving_candidate': previous_loss - loss,
                      'shuffled_candidate': control_loss - loss, 'anchor_shuffled': .25 - control_loss}
            bounds = {name: lower(delta) for name, delta in deltas.items()}
            record.comparisons = {name: comparison(delta, bounds[name]) for name, delta in deltas.items()}
            seed_means = control_losses.mean(axis=1)
            record.comparisons['anchor_shuffled']['shuffle_sd'] = measured(seed_means.std(ddof=1))
            record.comparisons['anchor_shuffled']['shuffle_se'] = measured(
                seed_means.std(ddof=1) / math.sqrt(len(control_losses)))
            self.metrics = {'brier': float(loss.mean()), 'brier_gain_lower': bounds['anchor_candidate'],
                'shuffled_gain_lower': bounds['anchor_shuffled'] - Z95 *
                    float(control_losses.mean(axis=1).std(ddof=1)) / math.sqrt(len(control_losses))}
            record.comparisons['anchor_shuffled']['adjusted_lower'] = measured(self.metrics['shuffled_gain_lower'])
            checks.extend([Check('retention', self.metrics['brier_gain_lower'] > 0
                                  and bounds['serving_candidate'] >= -self.retention_margin,
                                  'paired Brier gain against .5 and previous closure'),
                           Check('specificity', bounds['shuffled_candidate'] > 0,
                                 'paired Brier gain against time-shuffled targets')])
            record.parity = {'status': 'error', 'passed': None}
            pinned = self.store.revisions.state(closure)
            agree = np.allclose(self.learner.predict(pinned, states)[:, 0], predicted,
                                rtol=PARITY_RTOL, atol=PARITY_ATOL)
            record.parity = {'status': 'measured', 'passed': bool(agree)}
        else:
            checks.extend([Check('retention', False, f'awaiting {GATE_MIN_ROWS} reserved-week items'),
                           Check('specificity', False, f'awaiting {GATE_MIN_ROWS} reserved-week items')])
        budget = self.store.budgets.result(self.store.name, build.digest)
        if budget is not None:
            record.thresholds['budget_bound'] = budget.limit
        checks.extend([Check('canaries', self.learner.exposure(build.state, self.canaries)
                             == self.learner.exposure(self.learner.base(), self.canaries)),
                       Check('calibration', build.rows > 0), Check('budget', budget is not None and budget.ok),
                       Check('finite', all(np.isfinite(v).all() for v in build.state.values())),
                       Check('served_offline', bool(agree))])
        self.last_report = GateReport(closure.id, record.gate_set, tuple(checks))
        record.checks = [{'name': check.name, 'passed': bool(check.passed), 'detail': check.detail}
                         for check in checks]
        record.outcome = 'passed' if self.last_report.passed else 'failed'
        record.reason = ', '.join(check.name for check in checks if not check.passed)
        return self.last_report

    def poll(self, runtime):
        """Confirm delivery, serve once, and label only after the complete hour. Returns the snapshot it read
        (``Rows.snapshot``), None when it read none."""
        if not runtime.enabled or runtime.find('reply_timing') != self.store.name:
            return None
        with runtime.lock:
            turns, sends = snapshot = self.rows.snapshot()
            comparator = runtime.loaded[self.store.name].comparator
            records = {r['ref_hash']: r for r in comparator.records()}
            for send in sends:
                if send['status'] != 'sent' or not send['completed_at']:
                    continue
                at = moment(send['completed_at'])
                if at > self.clock() or at < max(self.started_at, self.clock() - timedelta(days=14)):
                    continue
                ref = send['id']
                frame, deps = self.rows.frame(at, turns, outreach_flag=True, probability=send['probability'] or 1.)
                row, all_deps = self.rows.labelled(at, turns, outreach_flag=True, origin=(ref,),
                                                  probability=send['probability'] or 1.)
                self.rows.dependencies[ref] = all_deps or (ref, *deps)
                if frame is None or ref not in self.source.record_refs((ref,)):
                    continue
                saved = records.get(comparator._path(ref).stem)
                if saved is None:
                    runtime.serve(self.store.name, ref, frame, predict=lambda pin, value:
                        float(self.learner.predict(pin.state if pin else {}, [value])[0, 0]))
                if row is not None and (saved is None or saved['outcome'] is None):
                    runtime.outcome(self.store.name, ref, observed=row['label'], teacher='realized_value')
        return snapshot

    async def live(self, runtime):
        while True:
            await asyncio.to_thread(self.poll, runtime)
            await asyncio.sleep(60)  # Builder assumption: follow delivery within one minute.


class TimingHolds(dict):
    """Snapshot data: first-held times plus the keyed exploration decisions."""
    def __init__(self):
        super().__init__()
        self.forced, self.decisions = set(), {}


def prepare(mind, state):
    """Missing numeric evidence leaves the existing outreach rules in charge."""
    runtime = getattr(mind, 'plasticity', None)
    if runtime is None or not runtime.enabled:
        return
    try:
        with runtime.lock:
            _prepare(mind, state)
    except StateLockBusy:
        state.reply_p.clear()
        state.timing_holds = TimingHolds()
    except Exception as error:  # a learner fault never stops the tick: today's outreach rules decide
        state.reply_p.clear()
        logger.warning('reply timing unavailable (%s)', type(error).__name__)


def _prepare(mind, state):
    runtime = getattr(mind, 'plasticity', None)
    name = runtime.find('reply_timing') if runtime is not None and runtime.enabled else None
    if name is None or not mind.faculties.get('plasticity', True):
        return
    driver = runtime.loaded[name].driver
    driver.rows.sources.owner_id = mind.owner_id
    polled = driver.poll(runtime)
    holds = TimingHolds()
    for item in mind.mind_state.items(HOLD_PREFIX):
        first = moment(item['text'])
        if state.now - first >= timedelta(days=1):
            mind.mind_state.delete(item['key'])
        else:
            holds[item['key'][len(HOLD_PREFIX):]] = first
    for item in mind.mind_state.items(SAMPLE_PREFIX):
        if state.now - moment(item['updated_at']) >= timedelta(days=1):
            mind.mind_state.delete(item['key'])
    state.timing_holds = holds
    # Existing rules form the eligible candidates before the learned hold.
    eligible = outreach.candidates(state)[1]
    # The poll's snapshot, read under the same lock hold.
    turns, _ = polled if polled is not None else driver.rows.snapshot()
    for candidate in eligible:
        holds.decisions[candidate.dedup_key] = {
            **driver.forced.choose(keyed(driver.rows.key, ['candidate', candidate.dedup_key]),
                                  eligible=False).manifest_row(),
            'hour_mark': float(state.timing.get(state.local_hour, 0.))}
    pin = runtime.pin(name, 'owner')
    if pin is None or pin.closure.mode == 'base':
        return
    grid = state.now.replace(minute=0, second=0, microsecond=0)
    for offset in (0, 1, 2):
        frame, _ = driver.rows.frame(grid + offset * HOUR, turns, outreach_flag=True)
        if frame is None:
            state.reply_p.clear()
            return
        state.reply_p[offset] = float(driver.learner.predict(pin.state, [frame])[0, 0])
    for candidate in eligible:
        if any(f.bound_commitment for f in state.findings if candidate.source_id == 'intention:' + f.id):
            continue
        due = next((c.due_at for c in state.cares if candidate.source_id == 'care:' + c.slug), None)
        due = next((c.due_at for c in state.loops if candidate.source_id == 'commitment:' + c.id), due)
        if outreach.holds(state, candidate=candidate, due_at=due) is None:
            continue
        key = candidate.dedup_key
        if key not in holds:
            holds[key] = state.now
            mind.mind_state.set(HOLD_PREFIX + key, text=state.now.isoformat(), now=state.now)
        ref = keyed(driver.rows.key, ['hold', key, holds[key].isoformat()])
        choice = driver.forced.choose(ref, eligible=True)
        sample = {**choice.manifest_row(), 'hour_mark': float(state.timing.get(state.local_hour, 0.))}
        sample_key = SAMPLE_PREFIX + ref
        if mind.mind_state.get(sample_key) is None:
            mind.mind_state.set(sample_key, text=json.dumps(sample), now=state.now)
        holds.decisions[key] = sample
        if choice.forced:
            holds.forced.add(key)


def sent(mind, row):
    """The allowed act seam; confirmation can also arrive after this tick."""
    runtime = getattr(mind, 'plasticity', None)
    name = runtime.find('reply_timing') if runtime is not None and runtime.enabled else None
    if name is not None and row.type in OUTREACH_TYPES and mind.faculties.get('plasticity', True):
        try:
            runtime.loaded[name].driver.poll(runtime)
        except StateLockBusy:
            pass  # The live worker will reconcile delivery on its next poll.
        except Exception as error:  # the outreach already went out; a learner fault never undoes the act
            logger.warning('reply timing delivery check unavailable (%s)', type(error).__name__)


register_factory('reply_timing', Driver)


def report_lines(detail) -> list[str]:
    """Render reply-timing evidence through the shared learner report hook."""
    if not detail:
        return []
    parts = [f"Reply timing: {detail['holds']} holds, {detail['forced']} forced runs (last 24 h); "
             f"{detail['served']} served, {detail['outcomes']} outcomes."]
    for arm, label in (("served", "served"), ("constant", "constant 0.5"), ("hour_mark", "hour mark")):
        if detail.get(arm + "_brier") is not None:
            parts.append(f"{label}: Brier {detail[arm + '_brier']:.4f}, "
                         f"log loss {detail[arm + '_log_loss']:.4f}.")
    useful = detail.get("interruptions_per_useful")
    parts.append(f"Interruptions per useful notice: {useful:g}." if useful is not None
                 else "Interruptions per useful notice: awaiting usefulness verdicts.")
    return parts


def report(config, name, *, runtime=None):
    """Read numeric evidence without loading a learner or keeping another label log.

    The hour-mark arm's probability is .5 * (1 - mark), a builder assumption.
    Useful notices use the outreach verdict, never reply speed as a substitute.
    """
    from ...util.temporal import now_utc
    now = runtime._clock() if runtime is not None else now_utc()
    directory = runtime.data_dir if runtime is not None else config.home
    sources = Sources(directory)
    root = runtime.root if runtime is not None else directory / config.get('plasticity.directory', 'plasticity')
    comparator_dir = root / 'learners' / name / 'comparator'
    if runtime is not None and name in runtime.loaded:
        driver = runtime.loaded[name].driver
        sources = driver.rows.sources
        records = runtime.loaded[name].comparator.records()
        key = driver.rows.key
        comparator_dir = runtime.loaded[name].comparator.directory
    else:
        try:
            key = (root / 'comparator.key').read_bytes()
        except FileNotFoundError:
            return {}
        records = [json.loads(line) for path in comparator_dir.glob('numeric-*.jsonl')
                   for line in path.read_text().splitlines() if line.strip()]
    # Comparator identity uses the shared keyed ref encoding.
    def reference_hash(ref):
        return keyed(key, ['reference', ref])
    sends = sources._query(sources.state_dir / INITIATIVES_DB,
        "SELECT id,verdict,json_extract(context,'$.reply_timing.hour_mark') AS mark "
        "FROM initiatives WHERE kind='message' AND status='sent' AND "
        "type IN ('outreach_finding','outreach_loop','outreach_care')")
    served_refs = {row['ref_hash'] for row in records}
    sends = [row for row in sends if reference_hash(row['id']) in served_refs]
    marks = {reference_hash(row['id']): float(row['mark'] or 0.) for row in sends}
    samples = sources._query(sources.state_dir / 'mind.db',
                            "SELECT text FROM mind_state WHERE key LIKE ? AND updated_at >= ?",
                            (SAMPLE_PREFIX + '%', (now - timedelta(days=1)).timestamp()))
    samples = [json.loads(row['text']) for row in samples]
    labelled = [r for r in records if r['outcome'] is not None]
    result = {'holds': sum(not s['forced'] for s in samples), 'forced': sum(s['forced'] for s in samples),
              'served': len(records), 'outcomes': len(labelled)}
    useful = sum(row['verdict'] == 'useful' for row in sends)
    result['interruptions_per_useful'] = len(sends) / useful if useful else None
    for arm in ('served', 'constant', 'hour_mark'):
        predictions = [r['output'] if arm == 'served' else .5 if arm == 'constant'
                       else .5 * (1 - min(1., max(0., marks.get(r['ref_hash'], 0.)))) for r in labelled]
        truth = np.array([r['outcome']['observed'] for r in labelled])
        p = np.clip(predictions, 1e-12, 1 - 1e-12)  # Builder assumption: logarithm floor.
        result[arm + '_brier'] = float(np.mean((p - truth) ** 2)) if labelled else None
        result[arm + '_log_loss'] = float(np.mean(-truth * np.log(p) - (1 - truth) * np.log1p(-p))) if labelled else None
    return result
