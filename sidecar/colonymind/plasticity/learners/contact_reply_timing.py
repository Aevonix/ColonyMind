"""One pooled contact reply forecast, prediction only.

Its evidence is the comms ledger (``colonymind-comms.db`` ``communications``): the agent's sends to contacts
and their replies, one contact per phone on every gateway (``contacts.store._match_canonical`` resolves a
sender when its row is written; a merge moves the rows). A send is the last outbound row of a five-minute
burst; its label is whether the contact writes back within the declared 24-hour horizon.

Plan assumptions: constant .5 anchor, 90 days of sends, 168-hour last-inbound cap,
ten sends per week, six channels plus other, and a log-hours horizon. Builder
assumptions: UTC weekly split (every fourth week), 30 training and 30 gate labels,
rate .001, coordinate clip .5, two-log-odds fence, eight shuffled seeds starting
at 7, and a 60-second pass that serves new sends and labels served ones.
No parameter belongs to a contact. No forecast schedules, holds or sends.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
from collections import Counter
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import numpy as np

from ...contacts.comms import CONVERSATION
from ...contacts.continuity import PLATFORMS, platform
from ...instinct.outcomes import TURN_DB
from ...util.temporal import parse_utc
from .. import _io, builds
from ..admission import Admission
from ..budgets import Budget, BudgetCheck
from ..gates import Check, GateReport, Item, Z95
from ..locking import on_busy
from ..nightly import register_factory
from ..numeric import NumericLearner, moment
from ..numeric_comparator import configure
from ..objectives import Objective
from ..synthetic import Cycle
from .reply_timing import gate_week, keyed

logger = logging.getLogger(__name__)

COMMS_DB = 'colonymind-comms.db'
TRAINING_MIN_ROWS = GATE_MIN_ROWS = 30
BURST = timedelta(minutes=5)           # outbound rows to one contact this close together are one send: its last row
HORIZON = timedelta(hours=24)          # the declared horizon: a reply is an inbound row within 24 h of the send
# The channel's platform, cut in SQL so no handle after it is read (``continuity.platform`` normalizes it again).
CHANNEL = "CASE WHEN instr(channel,':')>0 THEN substr(channel,1,instr(channel,':')-1) ELSE channel END"
BASE_BASIS = 'fixed uncalibrated 0.5 baseline'
OBJECTIVE = Objective(name='contact_reply_timing', model='numeric_predictor', kind='numeric-online',
    target_type='binary', losses={'realized_value': 'logistic'}, blend={'realized_value': 1.},
    points=('reply_by_horizon',), text=False)
DECLARATION = {
    'kind': 'numeric-online', 'implementation': 'contact_reply_timing', 'base': 'contact-reply-constant@1',
    'teachers': {'realized_value': 'binary'}, 'model': {'roles': ['numeric_predictor'], 'change': 'weights'},
    'budget': {'type': 'anchored_output', 'bound': 2.}, 'effect': 'predict', 'audience': 'owner',
    'personal_data': True, 'erasure_path': 'source erasure and clean rebuild', 'dependents': [],
    'mode': 'rebuild', 'learning_rate': .001, 'clip': .5,
    'gates': ['retention', 'specificity', 'canaries', 'calibration', 'budget', 'finite', 'served_offline'],
    'arms': ['constant', 'time_shuffled'],
    'comparator': {'window_items': 200, 'window_hours': 168, 'bar': .02},
}


class ContactReplyLearner(NumericLearner):
    """Fixed descriptors and one revision-owned channel vocabulary for all contacts."""

    def __init__(self, name, *, rate=.001, clip=.5, bound=2.):
        names = ['reply_share', 'hours_since_inbound', 'sends_7d', 'log_horizon_hours',
                 *[f'channel_{i}' for i in range(7)]]
        # Builder assumptions: fixed scaling, with the shared input box [-3, 3].
        standard = {name: {'mean': 0., 'scale': 1.} for name in names}
        standard.update(reply_share={'mean': .5, 'scale': .5},
                        hours_since_inbound={'mean': 84., 'scale': 84.},
                        sends_7d={'mean': 5., 'scale': 5.},
                        log_horizon_hours={'mean': 0., 'scale': 4.})
        super().__init__(dict(schema='numeric-revision-1', learner=name, variant='logistic', tz='UTC',
            features=[{'time': t} for t in ('how_sin', 'how_cos')]
                     + [{'input': name} for name in names] + [{'const': 1}],
            outputs=[{'event': 'reply_by_horizon'}], standardisation=standard,
            anchor={'type': 'constant', 'p': .5}), rate=rate, clip=clip, bound=bound)

    def base(self):
        return {**super().base(), 'channels': np.frombuffer(b'[]', dtype=np.uint8).copy()}

    def channels(self, state):
        return json.loads(np.asarray(state.get('channels', self.base()['channels']), dtype=np.uint8).tobytes())

    def frame(self, frame, channels):
        frame = json.loads(frame) if isinstance(frame, str) else frame
        if any(type(frame[name]) not in (int, float) or not math.isfinite(frame[name])
               for name in ('reply_share', 'hours', 'sends', 'horizon_hours')):
            raise ValueError('contact reply features must be finite numbers')
        channel = frame['channel']
        slot = channels.index(channel) if channel in channels else 6
        return {'at': frame['at'], 'inputs': {
            'reply_share': min(1., max(0., frame['reply_share'])),
            'hours_since_inbound': min(168., max(0., frame['hours'])),
            'sends_7d': min(10., max(0., frame['sends'])),
            'log_horizon_hours': min(12., max(-12., math.log(frame['horizon_hours']))),
            **{f'channel_{i}': float(i == slot) for i in range(7)}}}

    def fit(self, start, rows, *, objective, seed):
        counts = Counter(json.loads(row.state)['channel'] for row in rows)
        counts.pop('other', None)
        channels = sorted(counts, key=lambda channel: (-counts[channel], channel))[:6]
        transformed = [replace(row, state=json.dumps(self.frame(row.state, channels))) for row in rows]
        state = super().fit(start, transformed, objective=objective, seed=seed)
        state['channels'] = np.frombuffer(_io.canonical(channels), dtype=np.uint8).copy()
        return state

    def predict(self, state, states):
        state = state or self.base()
        return super().predict(state, [self.frame(frame, self.channels(state)) for frame in states])


class Rows:
    """Re-render the comms ledger's conversation rows by reference. Private identifiers stay in memory.

    Each row is read for its contact, direction, time, channel platform and source turn only, never its
    summary, and only a conversation row counts (``comms.CONVERSATION``: no receipt, no mind, system or
    automated row). The owner, placeholder ids (``default``, ``system``) and deleted contacts are no
    contacts here. A send is the last outbound row of a burst (outbound rows to one contact at most
    ``BURST`` apart, with no inbound between); its label is whether an inbound row from that contact comes
    within the declared ``HORIZON`` after it, known once the inbound is recorded or the horizon has passed.
    A row whose source turn was erased or withheld is gone, and so is everything it fed.
    """

    def __init__(self, store, sources, clock):
        self.store, self.sources, self.clock = store, sources, clock
        self.dependencies = {}
        self.pending_frames = {}
        self.pending_sends = {}

    @property
    def key(self):
        return (self.store.root / 'comparator.key').read_bytes()

    def ref(self, fid):
        return keyed(self.key, ['contact-reply', fid])

    def query(self, filename, sql, args=()):
        return [dict(row) for row in self.sources._query(self.sources.state_dir / filename, sql, args)]

    def contacts(self):
        """The live contacts other than the owner."""
        rows = self.sources._query(self.sources.contacts_db,
                                   'SELECT contact_id FROM contacts WHERE deleted_at IS NULL')
        return sorted({str(row['contact_id']) for row in rows} - {self.sources.owner_id, None})

    def snapshot(self):
        from colonymind.memory import curated
        return curated.fenced_read(SimpleNamespace(db_path=self.sources.state_dir / TURN_DB),
                                   self._read_snapshot, empty=([], []))

    def _read_snapshot(self):
        contacts, raw = self.contacts(), []
        for start in range(0, len(contacts), 500):
            chunk = contacts[start:start + 500]
            raw += self.query(COMMS_DB,
                "SELECT id,contact_id AS contact,direction,ts AS at," + CHANNEL + " AS channel,"
                "json_extract(source_lineage_json,'$.turn_id') AS turn FROM communications "
                "WHERE contact_id IN (" + ','.join('?' * len(chunk)) + ") AND " + CONVERSATION, chunk)
        turns = [row['turn'] for row in raw if row['turn']]
        hidden = self.sources.erased(turns) | self.sources.withheld(turns)
        rows = []
        for row in raw:
            at = parse_utc(row['at'])
            if at is None or row['turn'] in hidden:
                continue
            rows.append({'id': str(row['id']), 'contact': row['contact'], 'direction': row['direction'], 'at': at,
                         'channel': channel_of(row['channel']), 'deps': (row['turn'],) if row['turn'] else ()})
        rows.sort(key=lambda row: (row['contact'], row['at'], row['id']))
        inbound = [row for row in rows if row['direction'] == 'in']
        sends, burst = [], []

        def close():
            if burst:
                last = burst[-1]
                send = {'id': last['id'], 'contact': last['contact'], 'at': last['at'],
                        'horizon': last['at'] + HORIZON, 'channel': last['channel'],
                        'deps': tuple(dep for row in burst for dep in row['deps'])}
                sends.append({**send, **reply_of(send, inbound)})
                burst.clear()
        for row in rows:
            if burst and (row['contact'] != burst[-1]['contact'] or row['direction'] == 'in'
                          or row['at'] - burst[-1]['at'] > BURST):
                close()
            if row['direction'] == 'out':
                burst.append(row)
        close()
        sends.sort(key=lambda send: (send['at'], send['id']))
        return sends, inbound

    @staticmethod
    def label(send, now):
        """1 once its reply is recorded, 0 once its horizon passed without one, None while neither."""
        if send.get('reply_at') is not None and send['reply_at'] <= now:
            return 1
        return 0 if now >= send['horizon'] else None

    def frame(self, send, sends, inbound):
        """The send's features from what was known when it went: earlier sends to the contact (90 days), the
        replies recorded before it, and the contact's last inbound row at or before it."""
        at = send['at']
        prior = [row for row in sends if row['contact'] == send['contact']
                 and at - timedelta(days=90) <= row['at'] < at and row['id'] != send['id']]
        deps = list(send['deps'])
        replies = 0
        for row in prior:
            deps.extend(row['deps'])
            if row['reply_at'] is not None and row['reply_at'] < at:
                deps.extend(row['reply_deps'])
                replies += 1
        previous = max((row for row in inbound if row['contact'] == send['contact'] and row['at'] <= at),
                       key=lambda row: (row['at'], row['id']), default=None)
        if previous:
            deps.extend(previous['deps'])
        return {'at': at.isoformat(), 'contact': keyed(self.key, ['contact', send['contact']]),
            'reply_share': (replies + 1) / (len(prior) + 2),
            'hours': min(168., (at - previous['at']).total_seconds() / 3600) if previous else 168.,
            'sends': min(10, sum(row['at'] >= at - timedelta(days=7) for row in prior)),
            'channel': send['channel'], 'horizon_hours': (send['horizon'] - at).total_seconds() / 3600}, tuple(sorted(set(deps)))

    def serve(self, runtime, name, ref, send, sends, inbound, learner):
        """Freeze one served/reference pair for a send, its dependencies recorded first.

        Only a canonical forecast's send (``pending_sends``: no comms row holds it) keeps its frozen frame as
        current. A send from the comms ledger is current while it is still its burst's last row: a later row in the
        same burst supersedes its forecast (``current_records`` no longer finds it), so a burst served while it
        grows keeps one forecast, for its last row."""
        values, deps = self.frame(send, sends, inbound)
        self.dependencies[ref] = deps
        if ref in self.pending_sends:
            self.pending_frames[ref] = values
        return runtime.serve(name, ref, values, predict=lambda pin, value:
            float(learner.predict(pin.state if pin else {}, [value])[0, 0]))

    def examples(self):
        sends, inbound = self.snapshot()
        now = self.clock()
        cutoff = now - timedelta(days=90)
        for send in sends:
            label = self.label(send, now)
            if label is None or not cutoff <= send['at'] <= now:
                continue
            frame, deps = self.frame(send, sends, inbound)
            ref = self.ref(send['id'])
            self.dependencies[ref] = tuple(sorted(set((*deps, *send['reply_deps']))))
            yield {'ref': ref, 'point': 'reply_by_horizon', 'version': 'contact-reply-2',
                'state': _io.canonical(frame).decode(), 'state_hash': keyed(self.key, frame),
                'label': label, 'teacher': 'realized_value', 'push_away': False,
                'source_kind': 'resolver', 'audience': 'owner', 'at': send['at'].isoformat()}

    def rows(self, objective, *, since=None):
        yield from (row for row in self.examples() if (not since or moment(row['at']) >= moment(since))
                    and (not objective.points or row['point'] in objective.points))

    def source_of(self, row):
        return None

    def source_ids(self, ref):
        return self.dependencies.get(ref)

    def refs_resting_on(self, ids):
        wanted = set(ids)
        return [ref for ref, deps in self.dependencies.items() if wanted.intersection(deps)]

    def erase(self, refs):
        for ref in refs:
            self.dependencies.pop(ref, None)
            self.pending_frames.pop(ref, None)
            self.pending_sends.pop(ref, None)


def channel_of(value):
    """The channel's platform when it is a known one, else ``other``: a stored vocabulary never holds a handle."""
    name = platform(value)
    return name if name in PLATFORMS else 'other'


def reply_of(send, inbound):
    """The first inbound row from the send's contact after it and within its horizon, as time and sources."""
    reply = min((row for row in inbound if row['contact'] == send['contact']
                 and send['at'] < row['at'] <= send['horizon']), key=lambda row: (row['at'], row['id']), default=None)
    return {'reply_at': reply['at'] if reply else None, 'reply_deps': reply['deps'] if reply else ()}


class Driver(Cycle):
    minimum_labels = {'training': TRAINING_MIN_ROWS, 'gate': GATE_MIN_ROWS}

    def current_records(self, records, comparator):
        """Serve only pairs whose original evidence and frozen inputs remain current."""
        from colonymind.memory import curated
        return curated.fenced_read(SimpleNamespace(db_path=self.rows.sources.state_dir / TURN_DB),
            lambda: self._read_current_records(records, comparator), empty=[])

    def _read_current_records(self, records, comparator):
        sends, inbound = self.rows.snapshot()
        current = {}
        for send in sends:
            frame, deps = self.rows.frame(send, sends, inbound)
            ref = self.rows.ref(send['id'])
            original = self.rows.dependencies.get(ref, deps)
            # The recorded dependencies include a positive label's reply: withholding or erasing it retracts the label.
            if self.rows.sources.erased(original) or self.rows.sources.withheld(original):
                continue
            current[comparator._path(ref).stem] = comparator._hash(['input', frame])
        for ref, frame in self.rows.pending_frames.items():
            deps = self.rows.dependencies.get(ref, ())
            if not self.rows.sources.erased(deps) and not self.rows.sources.withheld(deps):
                current.setdefault(comparator._path(ref).stem, comparator._hash(['input', frame]))
        known = {comparator._path(ref).stem for ref in self.rows.dependencies}
        known.update(comparator._path(ref).stem for ref, deps in self.store.source_refs().items() if deps)
        return [row for row in records if row['ref_hash'] not in known
                or current.get(row['ref_hash']) == row['input_hash']]

    def label_counts(self, build):
        return {'training': build.rows,
                'gate': sum(1 for row in self.source.rows(self.objective) if gate_week(row['at']))}

    def __init__(self, *, store, controller, source, declaration, clock):
        for field in ('mode', 'personal_data', 'erasure_path', 'effect', 'audience', 'dependents', 'teachers'):
            if declaration.get(field) != DECLARATION[field]:
                raise ValueError('contact_reply_timing: invalid ' + field)
        if declaration['budget']['type'] != 'anchored_output' or not 0 < declaration['budget']['bound'] <= 2.:
            raise ValueError('contact_reply_timing: bound must be at most 2 log-odds')
        self.store, self.controller, self.source, self.clock = store, controller, source, clock
        sources = getattr(source.source, 'sources', None)
        if sources is None:
            raise ValueError('contact_reply_timing requires the instance source ledger')
        self.rows = Rows(store, sources, clock)
        source.source = self.rows
        self.objective = OBJECTIVE
        self.learner = ContactReplyLearner(store.name, rate=declaration.get('learning_rate', .001),
            clip=declaration.get('clip', .5), bound=declaration['budget']['bound'])
        path = store.directory / 'contact-reply-layout.json'
        identity = {'identity': self.learner.identity}
        if path.exists() and _io.read_json(path) != identity:
            raise ValueError('contact_reply_timing: settings changed; declare a new learner identity')
        _io.write_json(path, identity)
        configure(controller, declaration, variant='logistic')
        self.admission = Admission(OBJECTIVE, learner=store.name, source_of=source.source_of,
                                   protected_refs=self.protected, clock=clock)
        if store.budgets.declared(store.name) is None:
            store.budgets.declare(Budget(store.name, 'anchored_output', self.learner.bound))
        self.last_period = self.last_report = None
        self.started = clock()                 # a send before this is history: never served after the fact
        self.metrics = {}
        self.canaries = (Item('missing-input', '{"at":"2026-01-01T00:00:00Z"}', 0.),)

    def protected(self):
        return frozenset(row['ref'] for row in self.rows.examples() if gate_week(row['at']))

    def propose(self, build):
        closure = self.store.revisions.compose(state=build.digest, objective=OBJECTIVE.identity,
            calibration=('contact-reply-' + build.digest,), dependents={},
            parent=self.store.revisions.serving().id, manifests=build.manifests,
            periods=tuple(p.id for p in self.store.periods.retained()), audience=build.audience, mode='rebuild')
        self.controller.propose(closure, build.state)
        self.store.budgets.record(BudgetCheck(self.store.name, build.digest, 'anchored_output',
                                             self.learner.bound, self.learner.budget_used(build.state)))
        return closure

    def gate(self, closure, build):
        rows = [r for r in self.source.rows(OBJECTIVE) if gate_week(r['at'])]
        enough = len(rows) >= GATE_MIN_ROWS and build.rows >= TRAINING_MIN_ROWS
        checks, agree = [], False
        self.metrics = {}
        if enough:
            states, truth = [r['state'] for r in rows], np.array([r['label'] for r in rows])
            predicted = self.learner.predict(build.state, states)[:, 0]
            previous = self.learner.predict(self._start()[1], states)[:, 0]
            training = builds.training_rows(OBJECTIVE, self.store.current_manifests(), self.source)
            training.sort(key=lambda row: (moment(json.loads(row.state)['at']), row.ref))
            controls = []
            for seed in range(7, 15):
                order = np.random.default_rng(seed).permutation(len(training))
                shuffled = [replace(row, label=training[int(i)].label) for row, i in zip(training, order)]
                control = self.learner.fit(self.learner.base(), shuffled, objective=OBJECTIVE, seed=seed)
                controls.append((self.learner.predict(control, states)[:, 0] - truth) ** 2)
            controls = np.array(controls)
            control_loss, loss = controls.mean(axis=0), (predicted - truth) ** 2

            def lower(delta):
                return float(delta.mean() - Z95 * delta.std(ddof=1) / math.sqrt(len(delta)))

            self.metrics = {'brier': float(loss.mean()), 'brier_gain_lower': lower(.25 - loss),
                'shuffled_gain_lower': lower(.25 - control_loss) - Z95 *
                    float(controls.mean(axis=1).std(ddof=1)) / math.sqrt(len(controls))}
            checks.extend([Check('retention', self.metrics['brier_gain_lower'] > 0
                and lower((previous - truth) ** 2 - loss) >= -1e-12, 'paired Brier gain against .5 and previous'),
                Check('specificity', lower(control_loss - loss) > 0, 'paired gain against time-shuffled labels')])
            agree = np.allclose(self.learner.predict(self.store.revisions.state(closure), states)[:, 0],
                                predicted, rtol=1e-6)
        else:
            checks.extend(Check(name, False, 'awaiting 30 training and 30 reserved-week labels')
                          for name in ('retention', 'specificity'))
        budget = self.store.budgets.result(self.store.name, build.digest)
        checks.extend([Check('canaries', self.learner.exposure(build.state, self.canaries)
                             == self.learner.exposure(self.learner.base(), self.canaries)),
                       Check('calibration', enough), Check('budget', budget is not None and budget.ok),
                       Check('finite', all(np.isfinite(value).all() for value in build.state.values())),
                       Check('served_offline', bool(agree))])
        self.last_report = GateReport(closure.id, 'contact-reply-' + _io.digest([r['ref'] for r in rows]), tuple(checks))
        return self.last_report

    def poll(self, runtime):
        """Serve each new send once, as it happens, and label every served send once its label is known.

        A send is served only when it went after this driver started and before its label was known: a
        historical serve is never invented. A canonical forecast served through ``probability`` is labelled
        the same way, from the comms ledger, within its own horizon."""
        if not runtime.enabled or runtime.find('contact_reply_timing') != self.store.name:
            return
        with runtime.lock:
            comparator = runtime.loaded[self.store.name].comparator
            sends, inbound = self.rows.snapshot()
            now = self.clock()
            served = {row['ref_hash'] for row in comparator.records()}
            for send in sends:
                ref = self.rows.ref(send['id'])
                if (comparator._path(ref).stem in served or send['at'] < self.started or send['at'] > now
                        or self.rows.label(send, now) is not None):
                    continue
                try:
                    self.rows.serve(runtime, self.store.name, ref, send, sends, inbound, self.learner)
                except ValueError as error:          # an erased reference cannot serve again
                    logger.info('contact reply forecast not served (%s)', error)
            pending = {row['ref_hash'] for row in comparator.records() if row['outcome'] is None}
            if not pending:
                return
            labelled = [(self.rows.ref(send['id']), send) for send in sends]
            labelled += [(ref, {**send, **reply_of(send, inbound)}) for ref, send in self.rows.pending_sends.items()]
            binding = runtime.loaded[self.store.name]
            for ref, send in labelled:
                label = self.rows.label(send, now)
                if label is None or comparator._path(ref).stem not in pending:
                    continue
                if label and send['reply_deps']:
                    # A positive rests on its reply as well as on the send: record the reply's sources (in memory and
                    # in the persisted index) first, so withholding or erasing the reply takes the positive back.
                    served = self.rows.dependencies.get(ref) or self.rows.frame(send, sends, inbound)[1]
                    self.rows.dependencies[ref] = tuple(sorted({*served, *send['reply_deps']}))
                    if ref not in binding.source.record_refs((ref,)):
                        continue
                runtime.outcome(self.store.name, ref, observed=label, teacher='realized_value')
                self.rows.pending_sends.pop(ref, None)

    async def live(self, runtime):
        while True:
            await asyncio.to_thread(self.poll, runtime)
            await asyncio.sleep(60)


def _runtime():
    from ...api.routers import host
    runtime = getattr(host._mind(), 'plasticity', None)
    if runtime is None or not runtime.enabled:
        return None, None
    return runtime, runtime.find('contact_reply_timing')


@on_busy((.5, BASE_BASIS))
def probability(fid, retained):
    """Freeze one served/reference pair before the canonical forecast is issued."""
    runtime, name = _runtime()
    if name is None:
        return .5, BASE_BASIS
    with runtime.lock:
        binding = runtime.loaded[name]
        driver = binding.driver
        ref = driver.rows.ref(fid)
        comparator = binding.comparator
        saved = next((r for r in comparator.records() if r['ref_hash'] == comparator._path(ref).stem), None)
        if saved is None:
            c = retained['conditions']
            send = {'id': fid, 'contact': c['recipient_id'], 'at': moment(retained['origin_at']),
                'horizon': moment(retained['horizon']), 'channel': channel_of(c['channel']),
                'deps': (fid + ':dispatch', *c['parent_sources'])}
            driver.rows.pending_sends[ref] = send
            driver.rows.serve(runtime, name, ref, send, *driver.rows.snapshot(), driver.learner)
            saved = next((r for r in comparator.records() if r['ref_hash'] == comparator._path(ref).stem), None)
        if saved is None:
            return .5, BASE_BASIS
        basis = BASE_BASIS if saved['closure'] == binding.store.revisions.base.id else (
            'pooled contact reply timing closure ' + saved['closure'])
        return saved['output'], basis


@on_busy()
def outcome(fid):
    """Only an existing reconciled proof observation can label a served forecast."""
    runtime, name = _runtime()
    if name is not None:
        runtime.loaded[name].driver.poll(runtime)


register_factory('contact_reply_timing', Driver)


def report(config, name, *, runtime=None):
    """Counts and paired scores contain no contact identities or message contents."""
    if runtime is not None and name in runtime.loaded:
        records = runtime.loaded[name].comparator.records()
    else:
        from ...util.temporal import now_utc
        root = runtime.root if runtime is not None else config.home / config.get('plasticity.directory', 'plasticity')
        directory = root / 'learners' / name / 'comparator'
        # Match the comparator's 14-day retention without mutating report inputs.
        cutoff = (runtime._clock() if runtime is not None else now_utc()) - timedelta(days=14)
        try:
            erased = set(_io.read_json(root / 'erased-refs.json'))
        except FileNotFoundError:
            erased = set()
        records = [json.loads(line) for path in directory.glob('numeric-*.jsonl')
                   for line in path.read_text().splitlines(keepends=True) if line.endswith('\n')]
        records = [row for row in records if moment(row['at']) >= cutoff
                   and row.get('erasure_hash') not in erased]
    labelled = [row for row in records if row['outcome'] is not None]
    result = {'forecasts': len(records), 'outcomes': len(labelled)}
    truth = np.array([row['outcome']['observed'] for row in labelled])
    for arm in ('served', 'constant'):
        predictions = np.array([row['output'] if arm == 'served' else .5 for row in labelled])
        p = np.clip(predictions, 1e-12, 1 - 1e-12)  # Builder assumption: logarithm floor.
        result[arm + '_brier'] = float(np.mean((predictions - truth) ** 2)) if labelled else None
        result[arm + '_log_loss'] = float(np.mean(-truth * np.log(p) - (1 - truth) * np.log1p(-p))) if labelled else None
    return result


def report_lines(detail):
    lines = [f"Contacts' reply timing: {detail['forecasts']} forecasts, {detail['outcomes']} outcomes."]
    for arm, label in (('served', 'served'), ('constant', 'constant 0.5')):
        if detail[arm + '_brier'] is not None:
            lines.append(f"{label}: Brier {detail[arm + '_brier']:.4f}, log loss {detail[arm + '_log_loss']:.4f}.")
    return lines
