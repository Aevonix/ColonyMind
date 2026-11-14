"""Verified task outcomes forecast success without changing authority.

Plan assumptions: .7 anchor, twelve task types plus other, thirty days of prior
failures capped at five, three lessons, factor [.5, 1], note below .4 rounded to
ten percent, five-minute settlement polling, thirty labelled gate rows. Builder
assumptions: every fourth UTC week reserved, rate .01, coordinate clip .5,
two-log-odds lifetime fence, eight shuffle seeds starting at seven, 1e-6 scale
floor. Features and labels are re-read from primary references for every build.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import numpy as np

from ...instinct.outcomes import INITIATIVES_DB, Sources
from ...mind.authority import Budgets
from ...mind.learned import FactorSpec, register
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

TRAINING_MIN_ROWS = GATE_MIN_ROWS = 30
ANCHOR = .7
DRIVES = ('duty', 'social', 'curiosity', 'mastery', 'upkeep', 'other')
FAILURES = ('failed', 'blocked', 'expired', 'none')
OBJECTIVE = Objective(name='action_success', model='numeric_predictor', kind='numeric-online',
    target_type='binary', losses={'verified_outcome': 'logistic'}, blend={'verified_outcome': 1.},
    points=('action_success',), text=False)
DECLARATION = {
    'kind': 'numeric-online', 'implementation': 'action_success', 'base': 'action-constant@1',
    'teachers': {'verified_outcome': 'binary'}, 'model': {'roles': ['numeric_predictor'], 'change': 'weights'},
    'budget': {'type': 'anchored_output', 'bound': 2.}, 'effect': 'predict', 'audience': 'owner',
    'personal_data': True, 'erasure_path': 'source erasure and clean rebuild', 'dependents': [],
    'mode': 'rebuild', 'learning_rate': .01, 'clip': .5,
    'gates': ['retention', 'specificity', 'canaries', 'calibration', 'budget', 'finite', 'served_offline'],
    'arms': ['constant', 'task_type_prior', 'time_shuffled'],
    'comparator': {'window_items': 200, 'window_hours': 168, 'bar': .02},
}


def keyed(key, value):
    return hmac.new(key, _io.canonical(value), hashlib.sha256).hexdigest()


def gate_week(at):
    """Fixed membership survives erasure and never moves a gate row into training."""
    return int(moment(at).timestamp() // (7 * 86400)) % 4 == 0


def label(row):
    if row.get('outcome') == 'done' and row.get('verified') in {'owner', 'check'}:
        return 1
    if row.get('outcome') in FAILURES[:3] and row.get('verified') in {'owner', 'check', 'hermes_failure'}:
        return 0
    return None


class ActionLearner(NumericLearner):
    """Revision-owned hashed vocabulary and runtime scaling, shared bounded steps."""
    def __init__(self, name, *, rate=.01, clip=.5, bound=2.):
        names = [*[f'type_{i}' for i in range(13)], *[f'drive_{d}' for d in DRIVES],
                 'open_ended', 'goal_mode', 'success_check', 'runtime', 'retries', 'prior_failures',
                 *[f'failure_{f}' for f in FAILURES], 'lessons']
        super().__init__(dict(schema='numeric-revision-1', learner=name, variant='logistic', tz='UTC',
            features=[{'input': name} for name in names]
                     + [{'time': t} for t in ('how_sin', 'how_cos')] + [{'const': 1}],
            outputs=[{'event': 'action_success'}],
            standardisation={name: {'mean': 0., 'scale': 1.} for name in names},
            anchor={'type': 'constant', 'p': ANCHOR}), rate=rate, clip=clip, bound=bound)

    def base(self):
        # The untrained runtime reference is ten minutes, a builder assumption.
        return {**super().base(), 'types': np.frombuffer(b'[]', dtype=np.uint8).copy(),
                'runtime_scale': np.array([math.log(10.), 1.], dtype=np.float64)}

    @staticmethod
    def log_runtime(frame):
        seconds = float(frame['runtime_seconds'])
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('action_success requires a positive runtime budget')
        return math.log(seconds / 60.)

    def frame(self, frame, state):
        frame = json.loads(frame) if isinstance(frame, str) else frame
        types = json.loads(np.asarray(state['types'], dtype=np.uint8).tobytes())
        slot = types.index(frame['task_type']) if frame['task_type'] in types else 12
        drive = frame['drive'] if frame['drive'] in DRIVES else 'other'
        mean, scale = state['runtime_scale']
        inputs = {**{f'type_{i}': float(i == slot) for i in range(13)},
                  **{f'drive_{d}': float(d == drive) for d in DRIVES},
                  **{f'failure_{f}': float(f == frame['last_failure']) for f in FAILURES},
                  **{key: float(bool(frame[key])) for key in ('open_ended', 'goal_mode', 'success_check')},
                  'runtime': float(np.clip((self.log_runtime(frame) - mean) / scale, -3., 3.)),
                  # Retries clipped at three is a builder assumption; it affects only this feature.
                  'retries': min(3., max(0., float(frame['retries']))),
                  'prior_failures': min(5., max(0., float(frame['prior_failures']))) / 5.,
                  'lessons': min(3., max(0., float(frame['lessons']))) / 3.}
        return {'at': frame['at'], 'inputs': inputs}

    def fit(self, start, rows, *, objective, seed):
        frames = [json.loads(row.state) for row in rows]
        counts = Counter(frame['task_type'] for frame in frames)
        types = sorted(counts, key=lambda kind: (-counts[kind], kind))[:12]
        times = [self.log_runtime(frame) for frame in frames]
        layout = self.base()
        layout['types'] = np.frombuffer(_io.canonical(types), dtype=np.uint8).copy()
        if times:
            layout['runtime_scale'] = np.array([np.mean(times), max(float(np.std(times)), 1e-6)])
        transformed = [replace(row, state=json.dumps(self.frame(frame, layout)))
                       for row, frame in zip(rows, frames)]
        state = super().fit(start, transformed, objective=objective, seed=seed)
        return {**layout, **state}

    def predict(self, state, states):
        state = state or self.base()
        return super().predict(state, [self.frame(frame, state) for frame in states])


def turn_refs(context):
    """Read structural source identifiers; never look for identifiers in task prose."""
    found = set()

    def walk(value, field=''):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, key)
        elif isinstance(value, list):
            for item in value:
                walk(item, field)
        elif isinstance(value, str):
            if field in {'turn', 'turn_id', 'turn_ids', 'source_turn', 'source_turn_id', 'source_turn_ids'}:
                found.add(value.removeprefix('turn:'))
            elif field == 'evidence' and value.startswith('turn:'):
                found.add(value[len('turn:'):])

    walk(context)
    return tuple(sorted(ref for ref in found if ref))


class Rows:
    """Primary intention metadata, counts and keyed categories, with full provenance."""
    def __init__(self, store, sources, clock):
        self.store, self.sources, self.clock = store, sources, clock
        self.dependencies = {}
        self.budgets = Budgets()

    @property
    def key(self):
        return (self.store.root / 'comparator.key').read_bytes()

    def query(self, sql, args=(), *, filename=INITIATIVES_DB):
        return [dict(row) for row in self.sources._query(self.sources.state_dir / filename, sql, args)]

    def hidden(self, refs):
        refs = tuple(refs)
        if self.sources.erased(refs) or self.sources.withheld(refs):
            return True
        try:
            erased = set(_io.read_json(self.store.root / 'erased-sources.json'))
        except FileNotFoundError:
            erased = set()
        try:
            erased_refs = set(_io.read_json(self.store.root / 'erased-refs.json'))
        except FileNotFoundError:
            erased_refs = set()
        return any(self.store.source_hash(ref) in erased or keyed(self.key, ['erased', ref]) in erased_refs
                   for ref in refs)

    def snapshot(self):
        rows = self.query('SELECT id,type,drive,kind,context,success_check,lesson_ids,parent_goal_id, '
            'created_at,assigned_at,completed_at,failed_at,cancelled_at,outcome,verified '
            "FROM initiatives WHERE kind='task' ORDER BY created_at,id")
        ratings = {}
        for event in self.query("SELECT initiative_id,timestamp FROM assignment_history WHERE action='rated' "
                                "AND json_extract(details,'$.verdict') IN ('useful','not_useful','wrong')"):
            at = moment(event['timestamp'])
            ref = event['initiative_id']
            ratings.setdefault(ref, []).append(at)
        kept = []
        for row in rows:
            row['context'] = json.loads(row['context'] or '{}')
            row['deps'] = (row['id'], *turn_refs(row['context']))
            # An owner rating can verify a previously unverified terminal result.
            # Its completion time alone must not make that proof available early.
            ended = self.settled_at(row)
            verification = [at for at in ratings.get(row['id'], ()) if ended is not None and at >= ended]
            row['verified_at'] = min(verification, default=None) if row['verified'] == 'owner' else None
            if moment(row['created_at']) <= self.clock() and not self.hidden(row['deps']):
                kept.append(row)
        return kept

    @staticmethod
    def at(row):
        recorded = row['context'].get('action_success', {}).get('at')
        return moment(recorded or row['assigned_at'] or row['created_at'])

    @staticmethod
    def settled_at(row):
        value = row.get('completed_at') or row.get('failed_at') or row.get('cancelled_at')
        observed = row.get('context', {}).get('action_success_observed', {})
        if (not value and observed.get('outcome') == row.get('outcome')
                and observed.get('verified') == row.get('verified')):
            value = observed.get('at')
        if not value:
            return None
        return max(moment(value), row.get('verified_at') or moment(value))

    def available_label(self, row):
        ended = self.settled_at(row)
        return label(row) if ended is not None and ended <= self.clock() else None

    def observe(self):
        """Date an untimed verified block when first seen, never retrospectively.

        The mind's blocked outcome has no terminal timestamp. This numeric source
        metadata records observation time with the primary task, preserving the
        existing outcome and verifier as the authority for the label.
        """
        for row in self.snapshot():
            if label(row) is not None and self.settled_at(row) is None:
                self.remember(row['id'], 'action_success_observed',
                              {'at': self.clock().isoformat(), 'outcome': row['outcome'],
                               'verified': row['verified']})

    def open_ended(self, row):
        context = row['context']
        if 'open_ended' in context:
            return bool(context['open_ended'])
        saved = context.get('action_success', {})
        if 'open_ended' in saved:
            return bool(saved['open_ended'])
        found = self.query('SELECT json_extract(detail_json,\'$.open_ended\') AS value '
                          'FROM concerns WHERE intention_id=?', (row['id'],), filename='mind.db')
        # Legacy intentions lacking the flag use false, a builder assumption.
        return bool(found and found[0]['value'])

    def frame(self, row, at, *, snapshot=None):
        context = row['context']
        deps = (row.get('id'), *turn_refs(context)) if row.get('id') else turn_refs(context)
        deps = tuple(ref for ref in deps if ref)
        if self.hidden(deps):
            return None, deps
        topic = context.get('topic') or ''
        topic_hash = keyed(self.key, ['topic', topic]) if topic else ''
        failures = []
        if topic:
            for previous in self.snapshot() if snapshot is None else snapshot:
                ended = self.settled_at(previous)
                if (previous['id'] != row.get('id') and label(previous) == 0 and ended is not None
                        and at - timedelta(days=30) <= ended < at
                        and previous['context'].get('topic') == topic):
                    failures.append((ended, previous))
        failures.sort(key=lambda item: (item[0], item[1]['id']))
        deps = tuple(dict.fromkeys((*deps, *(ref for _, prior in failures for ref in prior['deps']))))
        runtime, retries = self.budgets.for_task(row['type'])
        lessons = context.get('lesson_ids') or row.get('lesson_ids') or []
        lessons = json.loads(lessons) if isinstance(lessons, str) else lessons
        frame = {'at': at.isoformat(), 'task_type': keyed(self.key, ['type', row['type']]),
                 'drive': row['drive'] if row['drive'] in DRIVES else 'other',
                 'open_ended': self.open_ended(row), 'goal_mode': bool(context.get('goal_mode')),
                 'success_check': bool(row['success_check']),
                 'runtime_seconds': context.get('max_runtime_seconds') or runtime,
                 'retries': retries if context.get('max_retries') is None else context['max_retries'],
                 'prior_failures': min(5, len(failures)),
                 'last_failure': failures[-1][1]['outcome'] if failures else 'none',
                 'lessons': min(3, len(lessons)), 'topic': topic_hash}
        if row.get('id'):
            self.dependencies[row['id']] = deps
        return frame, deps

    def examples(self):
        rows = self.snapshot()
        for row in rows:
            target = self.available_label(row)
            if target is None:
                continue
            at = self.at(row)
            frame, _ = self.frame(row, at, snapshot=rows)
            if frame is not None and at <= self.clock():
                yield {'ref': row['id'], 'point': 'action_success', 'version': 'action-success-1',
                       'state': _io.canonical(frame).decode(), 'state_hash': keyed(self.key, frame),
                       'label': target, 'teacher': 'verified_outcome', 'push_away': False,
                       'source_kind': 'resolver', 'audience': 'owner', 'at': at.isoformat()}

    def rows(self, objective, *, since=None):
        yield from (row for row in self.examples() if (not since or moment(row['at']) >= moment(since))
                    and (not objective.points or row['point'] in objective.points))

    def source_of(self, row):
        return None

    def source_ids(self, ref):
        return self.dependencies.get(ref)

    def refs_resting_on(self, ids):
        return [ref for ref, deps in self.dependencies.items() if set(ids).intersection(deps)]

    def erase(self, refs):
        for ref in refs:
            self.dependencies.pop(ref, None)

    def remember(self, ref, member, detail):
        """Keep numeric dispatch metadata with its primary intention, not in weights.

        The timestamp fixes retrospective features to the first dispatch; the
        marker prevents another comparator item after its fourteen-day retention.
        Only this JSON member changes, preserving concurrent context updates.
        """
        with closing(sqlite3.connect(self.sources.state_dir / INITIATIVES_DB)) as conn:
            conn.execute("UPDATE initiatives SET context=json_set(coalesce(context,'{}'), ?,json(?)) WHERE id=?",
                         ('$.' + member, json.dumps(detail), ref))
            conn.commit()


class Driver(Cycle):
    minimum_labels = {'training': TRAINING_MIN_ROWS, 'gate': GATE_MIN_ROWS}

    def label_counts(self, build):
        return {'training': build.rows,
                'gate': sum(1 for row in self.source.rows(self.objective) if gate_week(row['at']))}

    def __init__(self, *, store, controller, source, declaration, clock):
        for field in ('mode', 'personal_data', 'erasure_path', 'effect', 'audience', 'dependents', 'teachers'):
            if declaration.get(field) != DECLARATION[field]:
                raise ValueError('action_success: invalid ' + field)
        if declaration['budget']['type'] != 'anchored_output' or not 0 < declaration['budget']['bound'] <= 2.:
            raise ValueError('action_success: anchored_output bound must be at most 2 log-odds')
        self.store, self.controller, self.source, self.clock = store, controller, source, clock
        sources = getattr(source.source, 'sources', None)
        if sources is None:
            raise ValueError('action_success requires the instance source ledger')
        self.rows = Rows(store, sources, clock)
        source.source = self.rows
        self.objective = OBJECTIVE
        self.learner = ActionLearner(store.name, rate=declaration.get('learning_rate', .01),
                                    clip=declaration.get('clip', .5), bound=declaration['budget']['bound'])
        path = store.directory / 'action-layout.json'
        identity = {'identity': self.learner.identity}
        if path.exists() and _io.read_json(path) != identity:
            raise ValueError('action_success: settings changed; declare a new learner identity')
        _io.write_json(path, identity)
        configure(controller, declaration, variant='logistic')
        self.admission = Admission(self.objective, learner=store.name, source_of=source.source_of,
                                   protected_refs=self.protected, clock=clock)
        if store.budgets.declared(store.name) is None:
            store.budgets.declare(Budget(store.name, 'anchored_output', self.learner.bound))
        self.last_period = self.last_report = None
        self.metrics = {}
        self.canaries = (Item('missing-input', '{"at":"2026-01-01T00:00:00Z"}', 0.),)

    def protected(self):
        return frozenset(row['ref'] for row in self.rows.examples() if gate_week(row['at']))

    def period(self):
        self.rows.observe()
        return super().period()

    def propose(self, build):
        closure = self.store.revisions.compose(state=build.digest, objective=self.objective.identity,
            calibration=('action-' + build.digest,), dependents={}, parent=self.store.revisions.serving().id,
            manifests=build.manifests, periods=tuple(p.id for p in self.store.periods.retained()),
            audience=build.audience, mode='rebuild')
        self.controller.propose(closure, build.state)
        self.store.budgets.record(BudgetCheck(self.store.name, build.digest, 'anchored_output',
                                             self.learner.bound, self.learner.budget_used(build.state)))
        return closure

    def gate(self, closure, build):
        rows = [row for row in self.source.rows(self.objective) if gate_week(row['at'])]
        checks, agree = [], False
        self.metrics = {}
        if len(rows) >= GATE_MIN_ROWS and build.rows >= TRAINING_MIN_ROWS:
            states, truth = [row['state'] for row in rows], np.array([row['label'] for row in rows])
            predicted = self.learner.predict(build.state, states)[:, 0]
            previous = self.learner.predict(self._start()[1], states)[:, 0]
            training = builds.training_rows(self.objective, self.store.current_manifests(), self.source)
            controls = []
            for seed in range(7, 15):
                order = np.random.default_rng(seed).permutation(len(training))
                shuffled = [replace(row, label=training[int(i)].label) for row, i in zip(training, order)]
                control = self.learner.fit(self.learner.base(), shuffled, objective=self.objective, seed=seed)
                controls.append((self.learner.predict(control, states)[:, 0] - truth) ** 2)
            controls = np.array(controls)
            loss, baseline = (predicted - truth) ** 2, (ANCHOR - truth) ** 2

            def lower(delta):
                return float(delta.mean() - Z95 * delta.std(ddof=1) / math.sqrt(len(delta)))

            self.metrics = {'brier': float(loss.mean()), 'brier_gain_lower': lower(baseline - loss),
                'shuffled_gain_lower': lower(baseline - controls.mean(axis=0)) - Z95 *
                    float(controls.mean(axis=1).std(ddof=1)) / math.sqrt(len(controls))}
            checks.extend([Check('retention', self.metrics['brier_gain_lower'] > 0
                                  and lower((previous - truth) ** 2 - loss) >= -1e-12,
                                  'paired Brier gain against .7 and previous closure'),
                           Check('specificity', lower(controls.mean(axis=0) - loss) > 0,
                                 'paired Brier gain against time-shuffled targets')])
            agree = np.allclose(self.learner.predict(self.store.revisions.state(closure), states)[:, 0],
                                predicted, rtol=1e-6)
        else:
            checks.extend([Check('retention', False, 'awaiting 30 training and 30 reserved-week outcomes'),
                           Check('specificity', False, 'awaiting 30 training and 30 reserved-week outcomes')])
        budget = self.store.budgets.result(self.store.name, build.digest)
        checks.extend([Check('canaries', self.learner.exposure(build.state, self.canaries)
                             == self.learner.exposure(self.learner.base(), self.canaries)),
                       Check('calibration', build.rows >= TRAINING_MIN_ROWS), Check('budget', budget is not None and budget.ok),
                       Check('finite', all(np.isfinite(value).all() for value in build.state.values())),
                       Check('served_offline', bool(agree))])
        self.last_report = GateReport(closure.id, 'action-' + _io.digest([row['ref'] for row in rows]), tuple(checks))
        return self.last_report

    def poll(self, runtime):
        """Only previously served tasks get live outcomes; never replay a dispatch."""
        if not runtime.enabled or runtime.find('action_success') != self.store.name:
            return
        with runtime.lock:
            if self.store.erasure_pending():
                return
            self.rows.observe()
            comparator = runtime.loaded[self.store.name].comparator
            records = {row['ref_hash']: row for row in comparator.records() if row['outcome'] is None}
            for row in self.rows.snapshot():
                saved = records.get(comparator._path(row['id']).stem)
                target = self.rows.available_label(row)
                if saved is not None and target is not None:
                    # Keep the class and verifier that supplied this immutable
                    # comparator label, even if a blocked task later succeeds.
                    self.rows.remember(row['id'], 'action_success_outcome',
                                       {'outcome': row['outcome'], 'verified': row['verified']})
                    runtime.outcome(self.store.name, row['id'], observed=target, teacher='realized_value')

    async def live(self, runtime):
        while True:
            await asyncio.to_thread(self.poll, runtime)
            await asyncio.sleep(300)


@on_busy(1.)
def factor(runtime, name, candidate, now):
    if candidate.kind != 'task' or not candidate.multiplier_keys() or not runtime.enabled:
        return 1.
    with runtime.lock:
        driver = runtime.loaded[name].driver
        driver.rows.budgets = Budgets.from_config(runtime.config.get('mind.budgets', {}))
        # This source flag otherwise vanishes when a Candidate becomes an intention.
        candidate.extra = {**(candidate.extra or {}), 'open_ended': bool(candidate.open_ended)}
        context = {**candidate.extra, 'topic': candidate.topic, 'evidence': candidate.evidence,
                   'goal_mode': bool(candidate.parent_goal_id), 'lesson_ids': candidate.lesson_ids}
        row = {'type': candidate.type, 'drive': candidate.drive, 'context': context,
               'success_check': candidate.success_check}
        frame, _ = driver.rows.frame(row, now)
        pin = runtime.pin(name, 'owner')
        if frame is None or pin is None:
            return 1.
        p = float(driver.learner.predict(pin.state, [frame])[0, 0])
        return min(1., max(.5, p / ANCHOR))


@on_busy()
def note(runtime, name, row, now):
    if row.kind != 'task' or not runtime.enabled:
        return None
    with runtime.lock:
        driver = runtime.loaded[name].driver
        if driver.store.erasure_pending() or row.id in driver.source.blocked:
            return None
        driver.rows.budgets = Budgets.from_config(runtime.config.get('mind.budgets', {}))
        primary = next((item for item in driver.rows.snapshot() if item['id'] == row.id), None)
        if primary is None:
            return None
        detail = primary['context'].get('action_success', {})
        frame, _ = driver.rows.frame(primary, moment(detail['at']) if detail else now)
        if frame is None or row.id not in driver.source.record_refs((row.id,)):
            return None
        comparator = runtime.loaded[name].comparator
        saved = next((item for item in comparator.records() if item['ref_hash'] == comparator._path(row.id).stem), None)
        if saved is not None:
            p = saved['output']
        elif detail:
            # The comparator has expired; the primary dispatch marker still
            # prevents another served item or a forecast from a newer closure.
            p = detail['p']
        else:
            p = runtime.serve(name, row.id, frame, predict=lambda pin, value:
                float(driver.learner.predict(pin.state if pin else {}, [value])[0, 0]))
        if not detail:
            driver.rows.remember(row.id, 'action_success', {'at': now.isoformat(), 'p': p,
                                                          'open_ended': frame['open_ended']})
        if p < .4:
            return (f'A learned forecast puts tasks like this one at about {round(p * 10) * 10}% success; '
                    'check the result against the task before reporting it done.')
        return None


register_factory('action_success', Driver)
register(FactorSpec('action_success', .5, 1., factor, note))


def report(config, name, *, runtime=None):
    """Read paired evidence and the mind's existing weekly settlement counters."""
    from ...util.temporal import now_utc
    now = runtime._clock() if runtime is not None else now_utc()
    directory = runtime.data_dir if runtime is not None else config.home
    root = runtime.root if runtime is not None else directory / config.get('plasticity.directory', 'plasticity')
    sources = Sources(directory)
    comparator_dir = root / 'learners' / name / 'comparator'
    if runtime is not None and name in runtime.loaded:
        driver = runtime.loaded[name].driver
        sources, key = driver.rows.sources, driver.rows.key
        source = driver.rows
        records = runtime.loaded[name].comparator.records()
    else:
        try:
            key = (root / 'comparator.key').read_bytes()
        except FileNotFoundError:
            return {}
        records = [json.loads(line) for path in comparator_dir.glob('numeric-*.jsonl')
                   for line in path.read_bytes().splitlines(keepends=True) if line.endswith(b'\n')]
        records = [row for row in records if now - timedelta(days=14) <= moment(row['at']) <= now]
        store = SimpleNamespace(root=root, source_hash=lambda ref: keyed(key, ['source', ref]))
        source = Rows(store, sources, lambda: now)
    tasks = source.snapshot()
    by_ref = {keyed(key, ['reference', row['id']]): row for row in tasks}
    labelled = [row for row in records if row['outcome'] is not None]
    classifications = [by_ref[r['ref_hash']]['context'].get('action_success_outcome', by_ref[r['ref_hash']])
                       for r in labelled if r['ref_hash'] in by_ref]
    result = {'served': len(records), 'notes': sum(row['output'] < .4 for row in records),
              'factors_below_one': sum(row['context'].get('learned', {}).get('action_success', 1.) < 1
                                       for row in tasks),
              'outcomes': len(labelled),
              'outcomes_by_verifier': dict(Counter(row['verified'] for row in classifications)),
              'outcomes_by_class': dict(Counter(row['outcome'] for row in classifications))}
    priors = []
    for record in labelled:
        row = by_ref.get(record['ref_hash'])
        past = [item for item in tasks if row is not None and item['type'] == row['type']
                and label(item) is not None and (ended := Rows.settled_at(item)) is not None
                and ended < moment(record['at'])]
        priors.append((4 * ANCHOR + sum(label(item) for item in past)) / (4 + len(past)))
    truth = np.array([row['outcome']['observed'] for row in labelled])
    for arm, values in (('served', [row['output'] for row in labelled]),
                        ('constant', [ANCHOR] * len(labelled)), ('task_type_prior', priors)):
        p = np.clip(values, 1e-12, 1 - 1e-12)  # Builder assumption: numerical logarithm floor.
        result[arm + '_brier'] = float(np.mean((p - truth) ** 2)) if labelled else None
        result[arm + '_log_loss'] = float(np.mean(-truth * np.log(p) - (1 - truth) * np.log1p(-p))) if labelled else None
    counters = sources._query(sources.state_dir / INITIATIVES_DB,
        'SELECT h.action,count(*) AS n FROM assignment_history h JOIN initiatives i ON i.id=h.initiative_id '
        "WHERE h.timestamp>=? AND h.timestamp<=? AND ((h.action='outcome_failed' AND i.kind='task') "
        "OR h.action='breaker_trip') GROUP BY h.action", ((now - timedelta(days=7)).isoformat(), now.isoformat()))
    counts = {row['action']: row['n'] for row in counters}
    result.update(failed_dispatches_per_week=counts.get('outcome_failed', 0),
                  breaker_trips_per_week=counts.get('breaker_trip', 0))
    return result


def report_lines(detail):
    if not detail:
        return []
    lines = [f"Action success: {detail['served']} forecasts served, {detail['notes']} notes, "
             f"{detail['factors_below_one']} factors below 1; {detail['outcomes']} outcomes."]
    for arm, name in (('served', 'served'), ('constant', 'constant 0.7'), ('task_type_prior', 'task-type prior')):
        if detail.get(arm + '_brier') is not None:
            lines.append(f"{name}: Brier {detail[arm + '_brier']:.4f}, log loss {detail[arm + '_log_loss']:.4f}.")
    for field, name in (('outcomes_by_verifier', 'verifier'), ('outcomes_by_class', 'class')):
        lines.append(f"Outcomes by {name}: " + (', '.join(f'{key} {value}' for key, value in
                     sorted(detail[field].items())) or 'none') + '.')
    lines.append(f"Last week: {detail['failed_dispatches_per_week']} failed dispatches, "
                 f"{detail['breaker_trips_per_week']} breaker trips.")
    return lines
