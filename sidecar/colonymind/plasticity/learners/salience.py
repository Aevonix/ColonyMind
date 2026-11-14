"""Private numeric salience features and a bounded discretionary ranking factor.

Builder assumptions: twelve type slots plus other, fixed category vocabularies,
72-hour owner-turn cap, rate .001, coordinate clip .5, two-log-odds fence, and
a 1e-6 probability floor for degenerate labels. The revision's anchor is the
useful share of observed owner labels, including dismissals. Proxy targets name
the disfavoured label and are not usefulness observations. The type vocabulary
uses all admitted active rows. A changed vocabulary preserves or lowers each
type's coefficient; if that remap exceeds the fence, a constant residual at the
lower fence replaces the weights (a conservative builder assumption).
Dismissals rebase the anchor conservatively before a
separate nonpositive constant channel lowers predictions: changing the anchor's
denominator must never raise another candidate's factor.
"""
from __future__ import annotations

import asyncio
import copy
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
from types import SimpleNamespace

import numpy as np

from ...initiatives.store import InitiativeStore
from ...instinct.outcomes import INITIATIVES_DB, TURN_DB
from ...mind.learned import FactorSpec, FactorValue, register
from ...mind.rank import Candidate
from .. import _io, builds
from ..admission import Admission
from ..budgets import Budget, BudgetCheck
from ..forced import ForcedRun, ForcedRuns
from ..gates import Check, GateReport, Item, Z95
from ..locking import StateLockBusy, on_busy
from ..nightly import register_factory
from ..numeric import NumericLearner, Revision, bounded_step, moment
from ..numeric_comparator import NumericComparator, configure
from ..objectives import Objective
from ..revisions import Pinned
from ..synthetic import Cycle

GATE_MIN_ROWS = 30
TRAINING_MIN_ROWS = 1  # The existing calibration gate requires build.rows > 0.
DRIVES = ('duty', 'social', 'curiosity', 'mastery', 'upkeep', 'other')
KINDS = ('task', 'message', 'note', 'other')
CONCERNS = ('obligation', 'interest', 'social', 'upkeep', 'question', 'goal', 'goal_step',
            'anomaly', 'failure', 'other')
SOURCES = ('commitment', 'reply_wait', 'observation', 'interest', 'link_candidate',
           'contact', 'goal', 'task', 'concern', 'none', 'other')
TYPE_SLOTS = 12
PROBABILITY_FLOOR = 1e-6

OBJECTIVE = Objective(name='salience', model='numeric_predictor', kind='numeric-online',
    target_type='binary', losses={'owner_label': 'logistic', 'behavioural_proxy': 'margin'},
    blend={'owner_label': 1., 'behavioural_proxy': .1}, points=('useful',), text=False)
DECLARATION = {
    'kind': 'numeric-online', 'implementation': 'salience', 'base': 'salience-constant@1',
    'teachers': {'owner_label': 'binary', 'behavioural_proxy': 'binary'},
    'model': {'roles': ['numeric_predictor'], 'change': 'weights'},
    'budget': {'type': 'anchored_output', 'bound': 2.}, 'effect': 'rank', 'audience': 'owner',
    'personal_data': True, 'erasure_path': 'source erasure and clean rebuild', 'dependents': [],
    'mode': 'rebuild', 'exploration_share': 0., 'learning_rate': .001, 'clip': .5,
    'gates': ['retention', 'specificity', 'canaries', 'calibration', 'budget', 'finite', 'served_offline'],
    'arms': ['constant', 'time_shuffled'],
    'comparator': {'window_items': 200, 'window_hours': 168, 'bar': .02},
}


def keyed(key, value):
    return hmac.new(key, _io.canonical(value), hashlib.sha256).hexdigest()


def _number(value, default=0., low=0., high=1.):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return default
    return max(low, min(high, float(value)))


def _category(value, vocabulary):
    return value if isinstance(value, str) and value in vocabulary else 'other'


def frame(candidate, now, context, key):
    """Extract categories, numbers and keyed ids without copying candidate words."""
    context = context or {}
    at = moment(now)
    drive = getattr(candidate, 'drive', None)
    effective = (context.get('drives') or {}).get(drive, 1.)
    configured = (context.get('base') or context.get('drives') or {}).get(drive, 1.)
    configured = _number(configured, default=1., high=1e6)
    satiation = 1. - _number(effective, default=1., high=1e6) / configured if configured > 0 else 0.
    affect = context.get('affect')
    previous = context.get('last_owner_turn')
    if previous is None:
        hours = _number(context.get('hours_since_owner_turn'), default=72., high=72.)
    else:
        hours = max(0., min(72., (at - moment(previous)).total_seconds() / 3600))
    return {
        'at': at.isoformat(), 'type': keyed(key, ['salience-type', getattr(candidate, 'type', '')]),
        'drive': _category(drive, DRIVES), 'kind': _category(getattr(candidate, 'kind', None), KINDS),
        'concern_kind': _category(getattr(candidate, 'concern_kind', None), CONCERNS),
        'source_type': _category(getattr(candidate, 'source_type', None) or 'none', SOURCES),
        'open_ended': float(bool(getattr(candidate, 'open_ended', False))),
        'salience': _number(getattr(candidate, 'salience', None)),
        'cost': _number(getattr(candidate, 'cost', None)), 'satiation': _number(satiation),
        'worry': _number(getattr(affect, 'worry', None)),
        'curiosity': _number(getattr(affect, 'curiosity', None)), 'hours': hours,
    }


class Exploration:
    """G1 keyed sampling with a disabled default and an explicit .2 ceiling."""
    def __init__(self, key, *, learner, share=0.):
        if (isinstance(share, bool) or not isinstance(share, (int, float))
                or not math.isfinite(share) or not 0 <= share <= .2):
            raise ValueError('salience exploration_share must be between zero and .2')
        # Validate the key and learner through the shared implementation at zero too.
        self._forced = ForcedRuns(key, learner=learner, share=share or 1.)
        self.share, self.learner = float(share), learner

    def choose(self, ref, *, eligible=True):
        selected = self._forced.choose(ref, eligible=eligible)
        if self.share == 0.:
            return ForcedRun(ref, self.learner, eligible, False, 0. if eligible else 1.)
        return selected


class SalienceLearner(NumericLearner):
    """Logistic kind D with revision-owned hashes, anchor and push-away bias."""
    def __init__(self, name, *, rate=.001, clip=.5, bound=2.):
        categories = [('drive', DRIVES), ('kind', KINDS), ('concern_kind', CONCERNS), ('source_type', SOURCES)]
        names = [*[f'type_{i}' for i in range(TYPE_SLOTS + 1)],
                 *[f'{name}_{category}' for name, values in categories for category in values],
                 'open_ended', 'salience', 'cost', 'satiation', 'worry', 'curiosity', 'hours_since_owner_turn']
        standard = {name: {'mean': 0., 'scale': 1.} for name in names}
        standard['hours_since_owner_turn'] = {'mean': 36., 'scale': 12.}
        super().__init__(dict(schema='numeric-revision-1', learner=name, variant='logistic', tz='UTC',
            features=[{'time': name} for name in ('how_sin', 'how_cos')]
                     + [{'input': name} for name in names] + [{'const': 1}],
            outputs=[{'event': 'useful'}], standardisation=standard, anchor={'type': 'constant', 'p': .5}),
            rate=rate, clip=clip, bound=bound)

    def base(self):
        return {**super().base(), 'types': np.frombuffer(b'[]', dtype=np.uint8).copy(),
                'p0': np.array([.5], dtype=np.float64), 'push': np.zeros(1, dtype=np.float32)}

    def anchor(self, state):
        value = float(np.asarray((state or {}).get('p0', [.5])).reshape(-1)[0])
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError('salience revision anchor must be a finite probability')
        return value

    def types(self, state):
        values = json.loads(np.asarray((state or {}).get('types', self.base()['types']), dtype=np.uint8).tobytes())
        if (not isinstance(values, list) or len(values) > TYPE_SLOTS or len(set(values)) != len(values)
                or any(not isinstance(value, str) or len(value) != 64
                       or any(c not in '0123456789abcdef' for c in value) for value in values)):
            raise ValueError('salience types must be keyed hashes')
        return values

    def revision(self, state):
        state = state or self.base()
        metadata = copy.deepcopy(self.metadata)
        metadata['anchor']['p'] = self.anchor(state)
        weights = np.asarray(state['W'], dtype=np.float32).copy()
        push = float(np.asarray(state.get('push', [0.])).reshape(-1)[0])
        if not math.isfinite(push) or push > 0:
            raise ValueError('salience push-away bias must be nonpositive')
        weights[0, -1] += push
        return Revision(metadata, weights)

    def frame(self, value, types):
        value = json.loads(value) if isinstance(value, str) else value
        kind = value['type']
        if (not isinstance(kind, str) or len(kind) != 64
                or any(c not in '0123456789abcdef' for c in kind)):
            raise ValueError('salience type feature must be a keyed hash')
        slot = types.index(kind) if kind in types else TYPE_SLOTS
        inputs = {f'type_{i}': float(i == slot) for i in range(TYPE_SLOTS + 1)}
        for name, vocabulary in (('drive', DRIVES), ('kind', KINDS),
                                  ('concern_kind', CONCERNS), ('source_type', SOURCES)):
            category = _category(value.get(name), vocabulary)
            inputs.update({f'{name}_{entry}': float(entry == category) for entry in vocabulary})
        inputs.update({name: _number(value.get(name)) for name in
                       ('open_ended', 'salience', 'cost', 'satiation', 'worry', 'curiosity')})
        inputs['hours_since_owner_turn'] = _number(value.get('hours'), default=72., high=72.)
        return {'at': value['at'], 'inputs': inputs}

    def budget_used(self, state):
        return super().budget_used(state) + abs(float(np.asarray(state.get('push', [0.])).reshape(-1)[0]))

    def relayout(self, state, types):
        """Install all-row type frequencies without raising any one-hot contribution."""
        result = {name: np.array(value, copy=True) for name, value in state.items()}
        result['types'] = np.frombuffer(_io.canonical(types), dtype=np.uint8).copy()
        self.types(result)
        columns = [self.metadata['features'].index({'input': f'type_{i}'}) for i in range(TYPE_SLOTS + 1)]
        old = dict(zip(self.types(state), state['W'][0, columns]))
        other = state['W'][0, columns[-1]]
        result['W'][0, columns] = 0.
        for i, value in enumerate(types):
            result['W'][0, columns[i]] = old.get(value, other)
        result['W'][0, columns[-1]] = min([other, *(weight for value, weight in old.items() if value not in types)])
        if self.budget_used(result) > self.bound:
            # Copying an old 'other' into several slots can consume more of the
            # shared coordinate fence. Its global minimum is safe for every type.
            result['W'][:] = 0.
            result['push'][0] = -self.bound
            if self.budget_used(result) > self.bound:
                result['push'][0] = np.nextafter(result['push'][0], np.float32(0.))
        return result

    def rebase(self, state, probability):
        """Include dismissal zeros in p0 without increasing any factor.

        Write the old residual as d in [-B, B], its anchor as q, and the new
        anchor as p <= q. For scaled weights sW and bias b, factor nonincrease
        requires b <= (1-s)d + log((1-p)/((1-q)+(q-p)exp(d))). This is concave
        in d, so its two endpoints suffice. Choose the largest s allowed by
        sB+abs(b) <= the lifetime fence; account for float32 rounding inward.
        """
        result = {name: np.array(value, copy=True) for name, value in state.items()}
        previous = self.anchor(state)
        result['p0'][0] = probability
        probability = self.anchor(result)
        if probability > previous or float(state['push'][0]) != 0.:
            raise ValueError('salience anchor rebasing needs a lower anchor and an unpushed state')
        size = super().budget_used(state)
        if probability == previous or size == 0.:
            return result
        if size > self.bound:
            raise ValueError('salience starting state exceeds its lifetime fence')
        # log1p avoids cancellation when one dismissal barely changes p0.
        offset = lambda d: -math.log1p((previous - probability) / (1. - probability) * math.expm1(d))
        low, high = offset(-size), offset(size)
        scale = min(1., max(0., (self.bound + size + high) / (2 * size)))
        radii = np.array([1. if 'time' in f or 'const' in f else 3. for f in self.metadata['features']])
        original = state['W'].astype(np.float64)
        epsilon = np.finfo(np.float32).eps
        for _ in range(32):
            # Leave room for rounding at the analytic fence intersection.
            scale *= 1. - 8 * epsilon
            result['W'] = (original * scale).astype(np.float32)
            error = float(np.sum(np.abs(result['W'].astype(np.float64) - original * scale) * radii))
            bias = min(0., -(1. - scale) * size + low, (1. - scale) * size + high) - error
            # revision() combines the two constant coefficients as float32 too.
            bias -= 4 * epsilon * (abs(float(result['W'][0, -1])) + abs(bias))
            result['push'][0] = np.nextafter(np.float32(bias), np.float32(-math.inf))
            if self.budget_used(result) <= self.bound:
                return result
            scale *= .5
        raise ValueError('salience anchor cannot be rebased inside its numeric fence')

    def push_step(self, state, anchor, *, weight=1.):
        """A margin update moves only a globally nonpositive constant channel."""
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('salience training weight must be nonnegative and finite')
        result = {name: np.array(value, copy=True) for name, value in state.items()}
        old = float(state['push'][0])
        reserve = max(0., self.bound - super().budget_used(state))
        floor = max(-reserve, float(anchor['push'][0]) - self.clip)
        result['push'][0] = min(old, max(floor, old - self.rate * weight))
        # Float32 may round outward from the exact budget boundary.
        if self.budget_used(result) > self.bound:
            result['push'][0] = np.nextafter(result['push'][0], np.float32(0.))
        return result

    def fit(self, start, rows, *, objective, seed):
        del start, seed  # Rebuild revisions train from the origin, with their own layout.
        ordered = sorted(((json.loads(row.state), row) for row in rows),
                         key=lambda pair: (moment(pair[0]['at']), pair[1].ref, pair[1].teacher))
        pulling, pushing = [], []
        for value, row in ordered:
            self.target(row.label)
            if objective.blend[row.teacher] <= 0:
                continue
            (pushing if row.push_away or row.teacher == 'behavioural_proxy'
             or value.get('verdict') == 'dismissed' else pulling).append((value, row))
        state = self.base()
        if pulling:
            state['p0'][0] = min(1. - PROBABILITY_FLOOR, max(PROBABILITY_FLOOR,
                sum(float(row.label) for _, row in pulling) / len(pulling)))
        counts = Counter(value['type'] for value, _ in pulling)
        types = sorted(counts, key=lambda name: (-counts[name], name))[:TYPE_SLOTS]
        state['types'] = np.frombuffer(_io.canonical(types), dtype=np.uint8).copy()
        self.types(state)
        anchor = {name: np.array(value, copy=True) for name, value in state.items()}
        revision = self.revision(state)
        for value, row in pulling:
            probability = value.get('probability', 1.)
            if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                    or not math.isfinite(probability) or not 0 < probability <= 1):
                raise ValueError('salience sampling probability must be in (0, 1]')
            x = revision.features(**self.frame(value, types))
            weights = bounded_step(state['W'], anchor['W'], x, self.target(row.label),
                rate=self.rate * row.weight / probability, clip=self.clip, probability=self.anchor(state))
            state['W'] = self._project(weights, {'W': state['W']})['W']
        observed = [row for _, row in pulling + pushing
                    if row.teacher == 'owner_label' and not row.push_away]
        if observed:
            probability = min(1. - PROBABILITY_FLOOR, max(PROBABILITY_FLOOR,
                sum(float(row.label) for row in observed) / len(observed)))
            state = self.rebase(state, probability)
        # Apply independent lowering only after pulling training, so no later gradient
        # can compensate a proxy's decrease elsewhere in the feature space.
        for _, row in pushing:
            # TrainingRow.weight already includes its objective blend.
            state = self.push_step(state, anchor, weight=row.weight)
        counts = Counter(value['type'] for value, _ in pulling + pushing)
        return self.relayout(state, sorted(counts, key=lambda name: (-counts[name], name))[:TYPE_SLOTS])

    def predict(self, state, states):
        state = state or self.base()
        return super().predict(state, [self.frame(value, self.types(state)) for value in states])

    def factor(self, state, value):
        return max(.5, min(1.5, float(self.predict(state, [value])[0, 0]) / self.anchor(state)))

    def losses(self, state, rows, *, objective):
        result = super().losses(state, rows, objective=objective)
        proxies = [row for row in rows if row.teacher == 'behavioural_proxy']
        if proxies:
            result['behavioural_proxy'] = float(self.predict(state, [row.state for row in proxies]).mean())
        return result


logger = logging.getLogger(__name__)
MAX_CANDIDATES = 20  # Plan assumption: eligible discretionary items per pass.
MAX_PASSES = 200  # Builder assumption: bounded recent propensity history.
RETENTION_DAYS = 14  # Builder assumption: matches the numeric comparator.
REPLAY_DAYS = 90  # Builder assumption: retained by-reference replay horizon.
IGNORE_HOURS = 48  # Plan assumption: no owner reaction in two days is push-only.
BIAS_NOTICE = 'selection bias: outcomes exist only for acted items'


def gate_week(at):
    """Builder assumption: every fourth UTC week is reserved and never trained."""
    return int(moment(at).timestamp() // (7 * 86400)) % 4 == 0


def _dependencies(candidate):
    ids = [candidate.source_id] if candidate.source_id else []
    for value in candidate.evidence:
        if isinstance(value, str) and value:
            ids.append(value)
            prefix, sep, ident = value.partition(':')
            if sep and prefix in {'turn', 'source', 'claim', 'appraisal', 'intention'}:
                ids.append(ident)
    return tuple(dict.fromkeys(ids))


class Rows:
    """Re-render numeric snapshots in the source ledger, never cache row text."""

    def __init__(self, store, sources, clock):
        self.store, self.sources, self.clock = store, sources, clock
        self.dependencies = {}
        self.erasure_epoch = 0
        self.propensity_path = store.directory / 'salience-propensities.json'

    @property
    def key(self):
        return (self.store.root / 'comparator.key').read_bytes()

    def query(self, filename, sql, args=()):
        path = (self.sources.state_dir / filename).resolve()
        if not path.is_relative_to(self.sources.state_dir.resolve()):
            raise ValueError('salience sources must stay in the state directory')
        if not path.exists():
            return []
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(sql, args)]

    def reference(self, intention, verdict=None):
        # A corrected verdict gets a new admission identity. Only the current
        # verdict re-renders, so older proof/proxy rows cease to train.
        return keyed(self.key, ['salience-row', intention, verdict])

    def owner_turn(self, at):
        if not self.sources.owner_id:
            return None
        rows = self.query(TURN_DB, "SELECT s.turn_id AS id, coalesce(s.occurred_at,s.ingested_at) AS at "
            "FROM turn_sources s WHERE s.contact_id=? AND s.scope='person' AND s.session_id<>'mind' "
            "AND s.turn_id NOT LIKE 'mind:%' AND EXISTS (SELECT 1 FROM json_each(s.messages_json) m "
            "WHERE json_extract(m.value,'$.role')='user')", (self.sources.owner_id,))
        return max((r for r in rows if moment(r['at']) <= at), key=lambda r: moment(r['at']), default=None)

    def snapshot(self):
        return self.query(INITIATIVES_DB,
            "SELECT id,type,drive,kind,decision,status,created_at,completed_at,source_type,source_id,"
            "context,verdict FROM initiatives WHERE kind IN ('task','message','note') AND decision='act'")

    def sample(self, row):
        context = json.loads(row['context'] or '{}')
        saved = context.get('salience')
        at = moment(row['created_at'])
        deps = [row['id']]
        if row.get('source_id'):
            deps.append(row['source_id'])
        if isinstance(saved, dict) and isinstance(saved.get('frame'), dict):
            numeric = dict(saved['frame'])
            deps.extend(saved.get('source_ids', ()))
        else:
            # Historical intentions may still have their original concern template.
            concerns = self.query('mind.db',
                'SELECT detail_json FROM concerns WHERE intention_id=? ORDER BY last_touched DESC LIMIT 1',
                (row['id'],))
            detail = json.loads(concerns[0]['detail_json']) if concerns else {}
            if detail:
                candidate = Candidate.from_detail(detail)
            else:
                candidate = Candidate(type=row['type'], drive=row['drive'], kind=row['kind'], title='',
                    dedup_key=row['id'], source_type=row['source_type'], source_id=row['source_id'],
                    salience=context.get('candidate_salience', context.get('salience', .8))
                    if not isinstance(context.get('salience'), dict) else .8,
                    cost=context.get('cost', .1), concern_kind=context.get('concern_kind', 'obligation'),
                    open_ended=bool(context.get('open_ended', False)))
            deps.extend(_dependencies(candidate))
            turn = self.owner_turn(at)
            if turn:
                deps.append(turn['id'])
            numeric = frame(candidate, at, {'last_owner_turn': moment(turn['at']) if turn else None}, self.key)
        numeric['at'] = at.isoformat()
        return numeric, tuple(dict.fromkeys(deps))

    def label(self, row):
        history = self.query(INITIATIVES_DB,
            "SELECT action,details,timestamp FROM assignment_history WHERE initiative_id=? "
            "ORDER BY timestamp DESC,id DESC", (row['id'],))
        reaction = False
        for event in history:
            if moment(event['timestamp']) > self.clock():
                continue
            detail = json.loads(event['details'] or '{}')
            if event['action'] == 'rated':
                # The latest rating must still be the row's current verdict.
                if detail.get('by') != 'owner' or detail.get('verdict') != row['verdict']:
                    return None
                reaction = True
                verdict = detail.get('verdict')
                if verdict in {'useful', 'actioned', 'not_useful', 'wrong', 'dismissed'}:
                    refs = tuple(detail[k] for k in ('turn_id', 'source_id') if detail.get(k))
                    return (int(verdict in {'useful', 'actioned'}), 'owner_label', False, verdict, refs)
                break
            if detail.get('by') == 'owner' or detail.get('path') in {'reaction', 'quoted', 'next', 'list'}:
                reaction = True
        if ((not reaction or row['verdict'] == 'ignored') and row['verdict'] in {None, '', 'ignored'}
                and moment(row['created_at']) + timedelta(hours=IGNORE_HOURS) <= self.clock()):
            # The proxy pushes away from useful; it never asserts a negative proof label.
            return 1, 'behavioural_proxy', True, 'ignored', ()
        return None

    def examples(self):
        cutoff = self.clock() - timedelta(days=REPLAY_DAYS)
        for row in self.snapshot():
            if row['source_type'] == 'commitment' or row['type'] in {'outreach_answer', 'outreach_followup'}:
                continue
            at = moment(row['created_at'])
            if not cutoff <= at <= self.clock():
                continue
            label = self.label(row)
            if label is None:
                continue
            numeric, deps = self.sample(row)
            value, teacher, pushing, verdict, label_refs = label
            deps = tuple(dict.fromkeys((*deps, *label_refs)))
            ref = self.reference(row['id'], verdict)
            self.dependencies[ref] = self.dependencies[row['id']] = deps
            if self.sources.erased(deps) or self.sources.withheld(deps):
                continue
            numeric['verdict'] = verdict
            yield {'ref': ref, 'point': 'useful', 'version': 'salience-1',
                'state': _io.canonical(numeric).decode(), 'state_hash': keyed(self.key, numeric),
                'label': value, 'teacher': teacher, 'push_away': pushing,
                'source_kind': 'behavioural' if pushing else 'explicit', 'audience': 'owner',
                'at': at.isoformat()}

    def rows(self, objective, *, since=None):
        yield from (r for r in self.examples() if (not since or moment(r['at']) >= moment(since))
                    and (not objective.points or r['point'] in objective.points))

    def source_of(self, row):
        return None

    def source_ids(self, ref):
        return self.dependencies.get(ref)

    def refs_resting_on(self, ids):
        wanted = set(ids)
        refs = [ref for ref, deps in self.dependencies.items() if wanted.intersection(deps)]
        # Every rank depends on the whole eligible set, including unacted items
        # with incomplete source provenance. A source erasure drops this window.
        if wanted:
            self.erasure_epoch += 1
            _io.remove(self.propensity_path)
        return refs

    def propensities(self):
        try:
            records = _io.read_json(self.propensity_path)
        except FileNotFoundError:
            return []
        cutoff = self.clock() - timedelta(days=RETENTION_DAYS)
        return [r for r in records if moment(r['at']) >= cutoff][-MAX_PASSES:]

    def _prune_propensities(self, refs=(), *, sources=()):
        if not self.propensity_path.exists():
            return
        hashed = {keyed(self.key, ['intention', ref]) for ref in refs}
        sources = {self.store.source_hash(ref) for ref in sources}
        records = []
        for record in self.propensities():
            candidates = [c for c in record['candidates'] if c.get('intention') not in hashed
                          and not sources.intersection(c.get('sources', ()))]
            if candidates:
                records.append({**record, 'candidates': candidates})
        _io.write_json(self.propensity_path, records)

    def erase(self, refs):
        self.erasure_epoch += 1
        self._prune_propensities(refs)
        for ref in refs:
            self.dependencies.pop(ref, None)


class Driver(Cycle):
    minimum_labels = {'training': TRAINING_MIN_ROWS, 'gate': GATE_MIN_ROWS}

    def label_counts(self, build):
        return {'training': build.rows,
                'gate': sum(1 for row in self.source.rows(self.objective)
                            if gate_week(row['at']) and row['teacher'] == 'owner_label')}

    def __init__(self, *, store, controller, source, declaration, clock):
        for name in ('mode', 'personal_data', 'erasure_path', 'effect', 'audience', 'dependents', 'teachers'):
            if declaration.get(name) != DECLARATION[name]:
                raise ValueError('salience: invalid ' + name)
        if declaration['budget']['type'] != 'anchored_output' or not 0 < declaration['budget']['bound'] <= 2.:
            raise ValueError('salience: anchored_output bound must be at most 2 log-odds')
        self.store, self.controller, self.source, self.clock = store, controller, source, clock
        sources = getattr(source.source, 'sources', None)
        if sources is None:
            raise ValueError('salience requires the instance source ledger')
        self.rows = Rows(store, sources, clock)
        source.source = self.rows
        self.objective = OBJECTIVE
        self.learner = SalienceLearner(store.name, rate=declaration.get('learning_rate', .01),
            clip=declaration.get('clip', .5), bound=declaration['budget']['bound'])
        self.exploration_share = declaration.get('exploration_share', 0.)
        self.exploration = Exploration(self.rows.key, learner=store.name, share=self.exploration_share)
        identity = {'identity': self.learner.identity}
        path = store.directory / 'salience-layout.json'
        if path.exists() and _io.read_json(path) != identity:
            raise ValueError('salience: settings changed; declare a new learner identity')
        _io.write_json(path, identity)
        configure(controller, declaration, variant='logistic')
        self.admission = Admission(self.objective, learner=store.name, source_of=source.source_of,
                                   protected_refs=self.protected, clock=clock)
        if store.budgets.declared(store.name) is None:
            store.budgets.declare(Budget(store.name, 'anchored_output', self.learner.bound))
        self.last_period = self.last_report = None
        self.metrics = {}
        self._pending = None
        self._pass_sequence = max((r.get('sequence', 0) for r in self.rows.propensities()), default=0)
        self.canaries = (Item('missing-input', '{"at":"2026-01-01T00:00:00Z"}', 0.),)

    def protected(self):
        return frozenset(r['ref'] for r in self.rows.examples() if gate_week(r['at']))

    def propose(self, build):
        closure = self.store.revisions.compose(state=build.digest, objective=self.objective.identity,
            calibration=('salience-' + build.digest,), dependents={}, parent=self.store.revisions.serving().id,
            manifests=build.manifests, periods=tuple(p.id for p in self.store.periods.retained()),
            audience=build.audience, mode='rebuild')
        self.controller.propose(closure, build.state)
        self.store.budgets.record(BudgetCheck(self.store.name, build.digest, 'anchored_output',
                                             self.learner.bound, self.learner.budget_used(build.state)))
        return closure

    def gate(self, closure, build):
        self.metrics = {}
        rows = [r for r in self.source.rows(self.objective) if gate_week(r['at']) and r['teacher'] == 'owner_label']
        checks, agree = [], False
        if len(rows) >= GATE_MIN_ROWS:
            states, truth = [r['state'] for r in rows], np.array([r['label'] for r in rows])
            predicted = self.learner.predict(build.state, states)[:, 0]
            p0 = self.learner.anchor(build.state)
            reference_loss = (p0 - truth) ** 2
            previous_state = self._start()[1]
            previous = self.learner.predict(previous_state, states)[:, 0]
            training = builds.training_rows(self.objective, self.store.current_manifests(), self.source)
            control_losses = []
            for seed in range(7, 15):  # Builder assumption: eight shuffled fits.
                pulling = [r for r in training if not r.push_away]
                order = np.random.default_rng(seed).permutation(len(pulling))
                shuffled = [replace(r, label=pulling[int(i)].label) for r, i in zip(pulling, order)]
                control = self.learner.fit(self.learner.base(), shuffled, objective=self.objective, seed=seed)
                control_losses.append((self.learner.predict(control, states)[:, 0] - truth) ** 2)
            control_losses = np.asarray(control_losses)
            control_loss, loss = control_losses.mean(axis=0), (predicted - truth) ** 2

            def lower(delta):
                return float(delta.mean() - Z95 * delta.std(ddof=1) / math.sqrt(len(delta)))

            self.metrics = {'brier': float(loss.mean()), 'p0': p0,
                'brier_gain_lower': lower(reference_loss - loss),
                'shuffled_gain_lower': lower(reference_loss - control_loss) - Z95 *
                    float(control_losses.mean(axis=1).std(ddof=1)) / math.sqrt(len(control_losses))}
            checks.extend([Check('retention', self.metrics['brier_gain_lower'] > 0
                and lower((previous - truth) ** 2 - loss) >= -1e-12, 'paired Brier gain against p0 and previous'),
                Check('specificity', lower(control_loss - loss) > 0, 'paired gain against time-shuffled labels')])
            agree = np.allclose(self.learner.predict(self.store.revisions.state(closure), states)[:, 0],
                                predicted, rtol=1e-6)
        else:
            checks.extend([Check('retention', False, 'awaiting 30 reserved-week owner labels'),
                           Check('specificity', False, 'awaiting 30 reserved-week owner labels')])
        budget = self.store.budgets.result(self.store.name, build.digest)
        checks.extend([Check('canaries', self.learner.exposure(build.state, self.canaries)
                             == self.learner.exposure(self.learner.base(), self.canaries)),
            Check('calibration', build.rows > 0), Check('budget', budget is not None and budget.ok),
            Check('finite', all(np.isfinite(v).all() for v in build.state.values())),
            Check('served_offline', bool(agree))])
        self.last_report = GateReport(closure.id, 'salience-' + _io.digest([r['ref'] for r in rows]), tuple(checks))
        return self.last_report

    def unavailable_pass(self, context, now):
        skipped = getattr(self, '_unavailable_pass', None)
        return skipped is not None and skipped[0] is context and skipped[1] == now

    def pass_view(self, runtime, now, context):
        if self._pending is not None and self._pending['context'] is context and self._pending['now'] == now:
            return self._pending
        binding = runtime.loaded[self.store.name]
        binding.comparator.summary()
        serving = runtime.pin(self.store.name, 'owner')
        previous = self.controller.previous(serving.closure.id)
        reference = Pinned(previous, self.store.revisions.state(previous))
        self._pass_sequence += 1
        ref = keyed(self.rows.key, ['pass', now.isoformat(), self._pass_sequence])
        choice = self.exploration.choose(ref)
        self._pending = {'context': context, 'now': now, 'serving': serving, 'reference': reference,
                         'choice': choice, 'items': {}, 'erasure_epoch': self.rows.erasure_epoch,
                         'ref': ref, 'sequence': self._pass_sequence}
        return self._pending

    def prediction(self, runtime, candidate, now, context):
        pending = self.pass_view(runtime, now, context)
        items = pending['items']
        if id(candidate) not in items:
            value = frame(candidate, now, context or {}, self.rows.key)
            live, reference = pending['serving'], pending['reference']
            p0 = self.learner.anchor(live.state)
            reference_p0 = p0 if reference.closure.mode == 'base' else self.learner.anchor(reference.state)
            p = float(self.learner.predict(live.state, [value])[0, 0])
            reference_p = (reference_p0 if reference.closure.mode == 'base' else
                           float(self.learner.predict(reference.state, [value])[0, 0]))
            if pending['choice'].forced:
                p, reference_p = p0, reference_p0
            items[id(candidate)] = {'candidate': candidate, 'frame': value, 'p': p, 'reference_p': reference_p,
                'p0': p0, 'reference_p0': reference_p0, 'factor': max(.5, min(1.5, p / p0)),
                'reference_factor': max(.5, min(1.5, reference_p / reference_p0))}
        return items[id(candidate)]

    def capture(self, store, ref, prediction, context):
        row = store.get(ref)
        if row is None or row.decision != 'act' or row.kind not in {'task', 'message', 'note'}:
            return False
        saved = (row.context or {}).get('salience', {})
        if 'p0' in saved:
            self.rows.dependencies[ref] = self.rows.dependencies[self.rows.reference(ref)] = tuple(
                dict.fromkeys((ref, *saved.get('source_ids', ()))))
            return True
        candidate = prediction['candidate']
        deps = [ref, *_dependencies(candidate)]
        last = (context or {}).get('last_owner_turn')
        if last is not None:
            turn = self.rows.owner_turn(moment(last))
            if turn and moment(turn['at']) == moment(last):
                deps.append(turn['id'])
        self.rows.dependencies[ref] = self.rows.dependencies[self.rows.reference(ref)] = tuple(dict.fromkeys(deps))
        if self.rows.sources.withheld(deps) or self.rows.sources.erased(deps):
            return False
        store.update(ref, context={**(row.context or {}), 'salience': {'frame': prediction['frame'],
            'source_ids': list(dict.fromkeys(deps)), 'p0': prediction['p0'],
            'concern': keyed(self.rows.key, ['concern', candidate.dedup_key])}})
        return True

    @on_busy()
    def observe(self, runtime, ranking, now):
        if self.unavailable_pass(ranking.context, now):
            return  # Do not record a learned prediction for a neutral fallback.
        with runtime.lock:
            pending = self.pass_view(runtime, now, ranking.context)
            sources = {ref for candidate, _ in ranking.eligible for ref in _dependencies(candidate)}
            last = (ranking.context or {}).get('last_owner_turn')
            if last is not None:
                turn = self.rows.owner_turn(moment(last))
                if turn and moment(turn['at']) == moment(last):
                    sources.add(turn['id'])
            try:
                erased = set(_io.read_json(self.store.root / 'erased-sources.json'))
            except FileNotFoundError:
                erased = set()
            if (pending['erasure_epoch'] != self.rows.erasure_epoch or self.store.erasure_pending()
                    or erased.intersection(self.store.source_hash(ref) for ref in sources)
                    or self.rows.sources.erased(sources) or self.rows.sources.withheld(sources)):
                self._pending = None
                return
            eligible = [(c, score) for c, score in ranking.eligible if c.multiplier_keys()]
            items = {id(c): self.prediction(runtime, c, now, ranking.context) for c, _ in eligible}
            served_ranks = {id(c): i + 1 for i, (c, _) in enumerate(ranking.eligible)}
            reference_order = sorted(ranking.eligible, key=lambda pair: (
                -ranking.rescore(pair[0], {'salience': items[id(pair[0])]['reference_factor']}
                                 if id(pair[0]) in items else {}), pair[0].dedup_key))
            reference_ranks = {id(c): i + 1 for i, (c, _) in enumerate(reference_order)}
            acted = {}
            binding = runtime.loaded[self.store.name]
            # This is the shared comparator with its supported pin callback bound
            # to the pair that ranked, even if formation awaited a later promotion.
            paired_controller = SimpleNamespace(store=self.store, recover_rollback=self.controller.recover_rollback,
                previous=lambda closure: pending['reference'].closure
                if closure == pending['serving'].closure.id else self.controller.previous(closure))
            paired = NumericComparator(paired_controller, key=self.rows.key, clock=lambda: now, lock=runtime.lock,
                pin=lambda requester: pending['serving'], rollback=binding.comparator._rollback)
            existing = {r['ref_hash'] for r in binding.comparator.records()}
            with closing(InitiativeStore(state_dir=self.rows.sources.state_dir)) as intentions:
                for candidate, ref in ranking.acted:
                    item = items.get(id(candidate))
                    if item is None or not self.capture(intentions, ref, item, ranking.context):
                        continue
                    if ref not in self.source.record_refs((ref,)) or self.store.erasure_pending():
                        continue
                    acted[id(candidate)] = ref
                    if paired._path(ref).stem not in existing:
                        def predict(pin, inputs, item=item):
                            return item['p'] if pin.closure.id == pending['serving'].closure.id else item['reference_p']
                        paired.serve(ref, item['frame'], predict)
            rows = []
            for candidate, score in eligible[:MAX_CANDIDATES]:
                item = items[id(candidate)]
                ref = acted.get(id(candidate))
                rows.append({'ref': keyed(self.rows.key, ['candidate', candidate.dedup_key]),
                    'served_factor': item['factor'], 'reference_factor': item['reference_factor'],
                    'served_rank': served_ranks[id(candidate)], 'reference_rank': reference_ranks[id(candidate)],
                    'score': float(score), 'acted': ref is not None,
                    'intention': keyed(self.rows.key, ['intention', ref]) if ref else None,
                    'concern': keyed(self.rows.key, ['concern', candidate.dedup_key]),
                    'p': item['p'], 'reference_p': item['reference_p'], 'p0': item['p0'],
                    'sources': [self.store.source_hash(s) for s in _dependencies(candidate)]})
            choice = pending['choice']
            record = {'at': now.isoformat(), 'ref': pending['ref'], 'sequence': pending['sequence'],
                'closure': pending['serving'].closure.id, 'reference': pending['reference'].closure.id,
                'exploration': choice.forced, 'probability': choice.probability,
                'arm_probability': self.exploration_share if choice.forced else 1. - self.exploration_share,
                'exploration_share': self.exploration_share, 'candidates': rows}
            history = [r for r in self.rows.propensities() if r['ref'] != record['ref']]
            # One constant witness marks unknown provenance of unacted items.
            # If this driver is unavailable at source erasure, the runtime's
            # existing conservative reset clears the whole derived store.
            self.store.record_source_refs({keyed(self.rows.key, ['salience-propensity-provenance']): None})
            _io.write_json(self.rows.propensity_path, (history + [record])[-MAX_PASSES:])
            self._pending = None

    def poll(self, runtime):
        if not runtime.enabled or runtime.find('salience') != self.store.name:
            return
        with runtime.lock:
            records = {r['ref_hash']: r for r in runtime.loaded[self.store.name].comparator.records()}
            for row in self.rows.snapshot():
                ref = row['id']
                record = records.get(runtime.loaded[self.store.name].comparator._path(ref).stem)
                if record is None or record['outcome'] is not None:
                    continue
                label = self.rows.label(row)
                if label is None or label[1] != 'owner_label':
                    continue
                _, deps = self.rows.sample(row)
                deps = tuple(dict.fromkeys((*deps, *label[4])))
                self.rows.dependencies[ref] = deps
                if (self.rows.sources.erased(deps) or self.rows.sources.withheld(deps)
                        or ref not in self.source.record_refs((ref,))):
                    continue
                runtime.outcome(self.store.name, ref, observed=float(label[0]), teacher=label[1])

    async def live(self, runtime):
        while True:
            try:
                await asyncio.to_thread(self.poll, runtime)
            except Exception as error:
                logger.warning('salience live loop unavailable (%s)', type(error).__name__)
            await asyncio.sleep(60)  # Builder assumption: one outcome poll a minute.


def factor(runtime, learner, candidate, now, *, context=None):
    if not runtime.enabled or learner not in runtime.loaded or not candidate.multiplier_keys():
        return 1.
    driver = runtime.loaded[learner].driver
    if driver.unavailable_pass(context, now):
        return 1.
    try:
        with runtime.lock:
            value = driver.prediction(runtime, candidate, now, context)
            if driver._pending['choice'].forced:
                return FactorValue(1., overrides={name: 1. for name in (context or {}).get('implementations', ())})
            return value['factor']
    except StateLockBusy:
        # Pass bookkeeping belongs to this synchronous ranking caller, not the
        # nightly's model state. Keep this pass neutral even if the lock frees
        # between candidates, and don't fabricate learned observation evidence.
        driver._unavailable_pass = (context, now)
        return 1.


def observe(runtime, learner, ranking, now):
    if runtime.enabled and learner in runtime.loaded:
        runtime.loaded[learner].driver.observe(runtime, ranking, now)


def report(config, name, *, runtime=None):
    from ..nightly import Runtime
    own = runtime is None
    runtime = runtime or Runtime.from_config(config)
    try:
        binding = runtime.loaded.get(name)
        if binding is None:
            return {}
        driver = binding.driver
        records = binding.comparator.records()
        labelled = [r for r in records if r['outcome'] is not None]
        propensities = driver.rows.propensities()
        candidates = [c for r in propensities for c in r['candidates']]
        factors = [c['served_factor'] for c in candidates]
        result = {'factors': len(factors), 'factor_mean': float(np.mean(factors)) if factors else 1.,
            'factor_below': sum(v < 1 for v in factors) / len(factors) if factors else 0.,
            'factor_above': sum(v > 1 for v in factors) / len(factors) if factors else 0.,
            'served': len(records), 'outcomes': len(labelled), 'propensity_passes': len(propensities),
            'exploration_share': driver.exploration_share, 'selection_bias': BIAS_NOTICE,
            'teachers': dict(Counter(r['teacher'] for r in driver.rows.examples()))}
        # Resolve each frozen anchor through its source intention, independently
        # of the shorter bounded propensity history.
        intentions = driver.rows.snapshot()
        snapshots = {binding.comparator._path(r['id']).stem: json.loads(r['context'] or '{}').get('salience', {})
                     for r in intentions}
        p0 = {ref: value.get('p0') for ref, value in snapshots.items()}
        paired = [r for r in labelled if p0.get(r['ref_hash']) is not None]
        result['p0_outcomes'] = len(paired)
        for arm in ('served', 'p0'):
            selected = paired
            predicted = np.asarray([r['output'] if arm == 'served' else p0[r['ref_hash']] for r in selected])
            truth = np.asarray([r['outcome']['observed'] for r in selected])
            bounded = np.clip(predicted, 1e-12, 1 - 1e-12)  # Builder assumption: numerical log floor.
            result[arm + '_brier'] = float(np.mean((predicted - truth) ** 2)) if selected else None
            result[arm + '_log_loss'] = float(np.mean(-truth * np.log(bounded) -
                                                     (1 - truth) * np.log1p(-bounded))) if selected else None
        concerns = {}
        for row in labelled:
            concern = snapshots.get(row['ref_hash'], {}).get('concern', row['ref_hash'])
            concerns.setdefault(concern, []).append(row['outcome']['observed'])
        result['usefulness_per_acted_concern'] = float(np.mean([np.mean(v) for v in concerns.values()])) \
            if concerns else None
        dismissed = driver.rows.query(INITIATIVES_DB,
            "SELECT DISTINCT h.initiative_id FROM assignment_history h JOIN initiatives i ON i.id=h.initiative_id "
            "WHERE h.action='rated' AND json_extract(h.details,'$.by')='owner' "
            "AND json_extract(h.details,'$.verdict')='dismissed' AND i.verdict='dismissed' "
            "AND i.decision='act' AND h.timestamp>=?", ((driver.clock() - timedelta(days=7)).isoformat(),))
        result['dismissals_per_week'] = len(dismissed)
        return result
    finally:
        if own:
            runtime.close()


def report_lines(detail):
    if not detail:
        return []
    def number(value):
        return 'pending' if value is None else f'{value:.4f}'
    return [f"Salience: {detail['factors']} factors, mean {detail['factor_mean']:.3f}; "
            f"below 1 {detail['factor_below']:.1%}, above 1 {detail['factor_above']:.1%}; "
            f"{detail['served']} served, {detail['outcomes']} outcomes; teachers {detail['teachers']}",
        f"Brier {number(detail['served_brier'])} / p0 {number(detail['p0_brier'])}; "
        f"log loss {number(detail['served_log_loss'])} / p0 {number(detail['p0_log_loss'])}; "
        f"usefulness per acted concern {number(detail['usefulness_per_acted_concern'])}; "
        f"dismissals/week {detail['dismissals_per_week']}",
        f"Exploration {detail['exploration_share']:.1%}; {detail['propensity_passes']} propensity passes; "
        + BIAS_NOTICE]


register_factory('salience', lambda **kwargs: Driver(**kwargs))
register(FactorSpec('salience', .5, 1.5, factor, wants_context=True, observe=observe))
