"""Owner recall's registered rebuild driver and reference-only live evidence.

Qualification captures are local gate-suite scores, never sealed release sets or
query vectors. They bind the complete state, previous closure and document identity.
Only the query log persists queries. Passage vectors are read from the current index.
"""
from __future__ import annotations

import asyncio
from contextlib import closing, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
import hashlib
import hmac
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from ...memory import recall_exit
from .. import _io, builds
from ..admission import Admission, recall_refusal
from ..budgets import Budget, BudgetCheck
from ..gates import Check, Evidence, GateReport, Item, MIN_POOLED, SetLedger, Suite, evaluate
from ..locking import on_busy
from ..nightly import register_factory
from ..objectives import RECALL_USE
from ..query_adapter import QueryAdapterLearner, load_revision, save_revision, weights_digest
from ..query_log import QueryLog
from ..revisions import Pinned

_CURRENT: ContextVar[Any] = ContextVar('recall_adapter_request', default=None)


def transform_identity(state):
    """Both factor scopes invalidate calibration; the zero map is the identity."""
    global_digest = weights_digest(state['U'], state['V'])
    shown_digest = weights_digest(state['shown_U'], state['shown_V'])
    if shown_digest == 'identity':
        return 'identity' if global_digest == 'identity' else 'query-adapter-1:' + global_digest
    return 'query-adapter-1:' + hashlib.sha256(_io.canonical([global_digest, shown_digest])).hexdigest()


def current_request():
    return _CURRENT.get()


@dataclass
class RecallRequest:
    runtime: Any
    name: str
    turn_id: str
    pinned: Any = None
    cutoff: dict | None = None
    transform: str = 'identity'
    memo: dict = field(default_factory=dict)
    metadata: Any = None

    @classmethod
    @on_busy()
    def create(cls, runtime, *, owner, turn_id):
        if owner is not True or not turn_id or runtime is None or runtime.enabled is not True:
            recall_exit.code('no_turn_id' if not turn_id else 'not_owner' if owner is not True else 'runtime_unavailable')
            return None
        name = runtime.find('query_adapter')
        if name is None:
            recall_exit.code('runtime_unavailable')
            return None
        runtime.loaded[name].driver.attach(runtime)
        return cls(runtime, name, str(turn_id))

    @contextmanager
    def use(self):
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)

    @property
    def driver(self):
        return self.runtime.loaded[self.name].driver

    def pin(self, pinned):
        self.cutoff = self.driver.cutoff(pinned)
        self.pinned = pinned
        self.transform = transform_identity(pinned.state) if pinned.state else 'identity'
        if self.cutoff is not None and self.metadata is not None:
            from ...memory.recall import calibration_fingerprint
            with self.use():
                actual = calibration_fingerprint(self.metadata())
            if actual != self.cutoff['fingerprint']:
                self.reset()
                raise ValueError('query adapter calibration configuration changed')

    def reset(self):
        self.pinned, self.cutoff, self.transform = None, None, 'identity'


class RecallRows:
    def __init__(self, driver):
        self.driver = driver

    def source_of(self, row):
        return None if row['teacher'] in {'owner_label', 'verified_outcome'} else row['source_kind']

    def rows(self, objective, *, since=None):
        driver = self.driver
        for record in driver.records():
            if since is not None and _io.parse(record['at']) < _io.parse(since):
                continue
            query = driver.queries.get(record['turn_id'])
            if query is None:
                continue
            passages, provenance = {}, {}
            for passage in record['passages']:
                value = driver.read_passage(passage)
                if value is not None:
                    passages[passage['id']] = list(map(float, value))
                    provenance[passage['id']] = passage['source_kind']
            if not passages:
                continue
            state = json.dumps({'query': query.tolist(), 'passages': passages}, sort_keys=True)
            for signal in record.get('signals', []):
                preferred, disfavoured = signal['preferred'], signal['disfavoured']
                if preferred not in passages or disfavoured not in passages:
                    continue
                label = {'preferred': preferred, 'disfavoured': disfavoured,
                         'shown': {key: rank for key, rank in record.get('shown', {}).items() if key in passages},
                         'passage_sources': provenance}
                if signal['teacher'] == 'stronger_model':
                    label.update(model_role='reranker', reranked=record.get('reranked', {}))
                    # The mathematical learner also confines model updates to the captured wider set.
                    label['shown'] = {**label['shown'], **label['reranked']}
                row = {'ref': record['turn_id'], 'point': 'recall_use', 'version': 'query-adapter-1',
                       'state': state, 'state_hash': driver.keyed(['state', state]), 'label': label,
                       'teacher': signal['teacher'], 'push_away': signal['push_away'],
                       'source_kind': signal['source_kind'], 'audience': 'owner', 'at': record['at']}
                if not recall_refusal(row, ()):
                    yield row

    def refs_resting_on(self, ids):
        """Capture dependencies before deletion, including expired-query manifests."""
        ids = set(ids)
        driver = self.driver
        refs = {record['turn_id'] for record in driver.records()
                if ids.intersection(record['dependencies'])}
        refs.update(driver.queries.refs_resting_on(ids))
        pending = driver.pending_erasure()
        for source in ids:
            refs.update(pending.get(driver.keyed(['erasure-source', source]), ()))
        if refs:
            for source in ids:
                pending[driver.keyed(['erasure-source', source])] = sorted(refs)
            _io.write_json(driver.pending_path, pending)
        # Tombstone erased turn ids even when their first recall has not completed.
        driver.erase(refs | ids)
        return tuple(sorted(refs))

    def erase(self, refs):
        self.driver.erase(refs)
        pending = self.driver.pending_erasure()
        pending = {key: sorted(set(values) - set(refs)) for key, values in pending.items()}
        _io.write_json(self.driver.pending_path,
                       {key: values for key, values in pending.items() if values})


class Driver:
    # Qualification, rather than training, declares a row minimum (gates.evaluate).
    minimum_labels = {'gate': MIN_POOLED}

    def qualification_capture_present(self, build):
        return self._capture(build) is not None

    def label_counts(self, build):
        capture = self._capture(build)
        if capture is None:
            return {'gate': 0}
        gate = capture['gate']
        paired = zip(gate['ids'], *(gate[arm] for arm in ('reference', 'base', 'candidate', 'control')))
        return {'gate': len({ref for ref, *labels in paired if all(type(value) is bool for value in labels)})}

    def __init__(self, *, store, controller, source, declaration, clock):
        if (store.mode != 'rebuild' or declaration['audience'] != 'owner'
                or declaration['personal_data'] is not True or set(declaration['dependents']) != {'recall_cutoff'}
                or declaration['budget']['type'] != 'operator_norm'):
            raise ValueError('query_adapter requires owner personal-data rebuild and recall_cutoff')
        self.store, self.controller, self.source = store, controller, source
        self.declaration, self.clock = declaration, clock
        self.document_identity = declaration.get('document_identity')
        if not isinstance(self.document_identity, dict) or not self.document_identity:
            raise ValueError('document_identity: declare the frozen index identity')
        self.learner = QueryAdapterLearner(declaration.get('dims'), rank=declaration.get('rank', 32),
                                          bound=declaration['budget']['bound'])
        teachers = set(declaration['teachers'])
        self.objective = replace(RECALL_USE,
            losses={key: value for key, value in RECALL_USE.losses.items() if key in teachers},
            blend={key: value for key, value in RECALL_USE.blend.items() if key in teachers})
        self.runtime = None
        self.queries = QueryLog(store.directory / 'query-vectors.db', clock=clock)
        self.directory = store.directory / 'recalls'
        self.sets = SetLedger(store.directory / 'sets')
        source.source = RecallRows(self)
        self.admission = Admission(self.objective, learner=store.name, source_of=source.source_of,
                                   protected_refs=self.sets.protected_refs, clock=clock)
        self.store.budgets.declare(Budget(store.name, 'operator_norm', self.learner.bound))
        self.last_period = None
        self.passage_reader = None
        self._qualification = {}
        if not self.controller.events():
            self.controller._event('recall', store.revisions.base.id, reason=(
                "Query adapter starts at identity with the environment cutoff. No training rows yet; "
                "unlocated teachers supply no rows. Each qualified adapter and cutoff serves immediately "
                "with the previous closure as live comparator and automatic joint rollback."))

    def attach(self, runtime):
        self.runtime = runtime
        self.queries.runtime = runtime
        with runtime.lock:
            pending = {ref for refs in self.pending_erasure().values() for ref in refs}
            if pending:
                runtime.erase(sorted(pending))

    @property
    def pending_path(self):
        # A recovery intent lives beside the runtime's intents, outside the
        # learner scan until Runtime.erase returns clean. Completion removes it.
        return self.store.root / 'query-erasure-intents' / (self.store.name + '.json')

    def pending_erasure(self):
        path = self.pending_path
        return _io.read_json(path) if path.exists() else {}

    def keyed(self, value):
        return hmac.new((self.store.root / 'comparator.key').read_bytes(), _io.canonical(value),
                        hashlib.sha256).hexdigest()

    def _path(self, turn_id):
        return self.directory / (self.keyed(['recall', turn_id]) + '.json')

    def records(self):
        return [record for path in sorted(self.directory.glob('*.json'))
                if self.queries.sources_current((record := _io.read_json(path))['dependencies'])]

    def evidence(self, turn_id):
        path = self._path(turn_id)
        record = _io.read_json(path) if path.exists() else None
        return record if record is not None and self.queries.sources_current(record['dependencies']) else None

    def erased(self, turn_id):
        path = self.store.directory / 'erased-recalls.json'
        return path.exists() and self.keyed(['recall', turn_id]) in _io.read_json(path)

    def capture(self, turn_id, vector, passages, served, reference):
        with self.controller.lock:
            if self.erased(turn_id) or self.evidence(turn_id) is not None:
                return False
            dependencies = sorted({turn_id, *served, *reference,
                                   *(p['source_id'] for p in passages)})
            if not self.queries.record(turn_id, vector, audience='owner', source_ids=dependencies):
                return False
            # Whitelist the durable fields: source text and vectors never reach this record.
            passages = [{key: p[key] for key in ('id', 'source_id', 'source_kind', 'message_hash') if key in p}
                        for p in passages]
            _io.write_json(self._path(turn_id), {'turn_id': turn_id, 'at': _io.stamp(self.clock()),
                'dependencies': dependencies, 'passages': passages, 'shown': {}, 'signals': [],
                'served': [self.keyed(['source', key]) for key in served],
                'reference': [self.keyed(['source', key]) for key in reference]})
            return True

    def erase(self, refs):
        refs = set(refs)
        if not refs:
            return
        with self.controller.lock:
            path = self.store.directory / 'erased-recalls.json'
            tombstones = set(_io.read_json(path)) if path.exists() else set()
            tombstones.update(self.keyed(['recall', ref]) for ref in refs)
            _io.write_json(path, sorted(tombstones))
            self.queries.erase_sources(refs)
            for ref in refs:
                _io.remove(self._path(ref))
            # Portable exports are derived weight copies too. Recreate only clean
            # exports on the next build; the revision ledger owns the serving state.
            exports = self.store.directory / 'adapters'
            if exports.exists():
                shutil.rmtree(exports)

    def read_passage(self, passage):
        """Read an existing projection by id, without embedding or endpoint access."""
        if self.passage_reader is not None:
            return self.passage_reader(passage)
        from ...vector import get_store
        from ...vector.collections import Collection
        from ...turns import get_turn_idempotency_ledger
        from ...turns.source_vectors import hydrate
        vector_store = get_store()
        if vector_store is None or vector_store.catalog is None:
            return None
        if asdict(vector_store.identity.document_identity) != self.document_identity:
            return None
        generation = vector_store.catalog.read_generation(vector_store.identity)
        path = Path(vector_store._data_dir)
        if generation['id'] != 'legacy':
            path = path / 'generations' / generation['id']
        import lancedb
        with vector_store.reads.hold():
            table = lancedb.connect(str(path)).open_table(Collection.CONVERSATIONS.value)
            rows = table.search().where('id = ' + vector_store._quoted(passage['id'])).limit(1).to_list()
        if not rows:
            return None
        row = rows[0]
        metadata = json.loads(row['metadata'])
        if (metadata.get('source_turn_id') != passage['source_id']
                or not vector_store._eligible(Collection.CONVERSATIONS, row['id'], metadata)):
            return None
        ledger = get_turn_idempotency_ledger(self.runtime.data_dir if self.runtime else self.store.root.parent)
        if hydrate(ledger, metadata) is None:
            return None
        return row['vector']

    def record_selection(self, turn_id, rows):
        with self.controller.lock:
            record = self.evidence(turn_id)
            if record is None:
                return
            shown = {}
            for rank, row in enumerate(rows, 1):
                sources = set(row.get('source_turn_ids') or ())
                sources.update([row.get('source_turn_id')])
                sources.update(ref['source_id'] for ref in row.get('_annotation_source_refs', ()))
                for passage in record['passages']:
                    hashes = row.get('_annotation_message_hashes', {}).get(passage['source_id'], ())
                    if (passage['source_id'] in sources and (not passage.get('message_hash')
                            or passage['message_hash'] == row.get('source_message_hash')
                            or passage['message_hash'] in hashes)):
                        shown.setdefault(passage['id'], rank)
            record['shown'] = shown
            _io.write_json(self._path(turn_id), record)

    def add_source(self, turn_id, source_id, *, ledger, owner_id):
        """Locate a corrected-to owner passage even when neither top-k list found it."""
        from ...turns.source_vectors import chunks, hydrate
        with self.controller.lock:
            record = self.evidence(turn_id)
            if record is None or self.erased(source_id):
                return
            with closing(ledger._connect()) as conn:
                source = conn.execute("SELECT * FROM turn_sources WHERE turn_id=? AND contact_id=? "
                                      "AND scope='person'", (source_id, owner_id)).fetchone()
                if source is None:
                    return
                projections = list(chunks(conn, source))
            known = {p['id'] for p in record['passages']}
            for _, metadata in projections:
                current = hydrate(ledger, metadata, contact_id=owner_id, session_id=source['session_id'])
                if current is None or current.get('role') != 'user' or metadata['kind'] != 'source_quote':
                    continue
                passage = {'id': metadata['projection_id'], 'source_id': source_id,
                           'source_kind': 'owner', 'message_hash': metadata['source_message_hash']}
                if passage['id'] not in known and self.read_passage(passage) is not None:
                    record['passages'].append(passage)
                    known.add(passage['id'])
                    record['dependencies'] = sorted(set(record['dependencies']) | {source_id})
            _io.write_json(self._path(turn_id), record)

    def record_trace(self, turn_id, trace):
        """Keep bounded reference-only scores; unavailable reranker evidence teaches nothing."""
        with self.controller.lock:
            record = self.evidence(turn_id)
            if record is None:
                return
            scored = [row for row in trace.get('rows', ())
                      if row.get('submitted') and row.get('rerank_score') is not None
                      and row.get('turn_id') and row.get('exit') != 'erased']
            scored.sort(key=lambda row: row['rerank_score'], reverse=True)
            ranks, scores = {}, {}
            for rank, row in enumerate(scored[:20], 1):
                matched = [p for p in record['passages'] if p['source_id'] == row['turn_id']
                           and p.get('message_hash') == row.get('message_hash')]
                # A trace cannot locate an individual chunk of a multi-chunk message.
                if len(matched) == 1:
                    ranks.setdefault(matched[0]['id'], rank)
                    scores.setdefault(matched[0]['id'], row['rerank_score'])
            record['reranked'], record['rerank_scores'] = ranks, scores
            _io.write_json(self._path(turn_id), record)
            if len(scores) >= 2:
                preferred, disfavoured = max(scores, key=scores.get), min(scores, key=scores.get)
                if scores[preferred] > scores[disfavoured]:
                    self.signal(turn_id, preferred=preferred, disfavoured=disfavoured,
                                teacher='stronger_model', evidence_refs=record['dependencies'])

    def signal(self, turn_id, *, preferred, disfavoured, teacher, evidence_refs,
               attributed=False, push_away=False):
        """A trusted resolver's reference-backed signal; an unlocated teacher yields nothing."""
        if teacher not in self.objective.losses or teacher == 'synthetic' or not evidence_refs:
            return False
        if teacher == 'verified_outcome' and attributed is not True:
            return False
        if teacher == 'behavioural_proxy' and push_away is not True:
            return False
        with self.controller.lock:
            record = self.evidence(turn_id)
            if record is None:
                return False
            passages = {p['id']: p for p in record['passages']}
            if preferred not in passages or disfavoured not in passages or preferred == disfavoured:
                return False
            signal = {'preferred': preferred, 'disfavoured': disfavoured, 'teacher': teacher,
                      'push_away': push_away, 'source_kind': {'owner_label': 'explicit',
                      'verified_outcome': 'resolver'}.get(teacher, teacher)}
            row = {'teacher': teacher, 'label': {'preferred': preferred, 'disfavoured': disfavoured,
                   'shown': record['shown'], 'passage_sources': {key: p['source_kind'] for key, p in passages.items()},
                   'reranked': record.get('reranked', {}), 'model_role': 'reranker'}}
            if recall_refusal(row, ()):
                return False
            # Each ref/teacher is one pair, as the core's manifest contract requires.
            record['signals'] = [s for s in record['signals'] if s['teacher'] != teacher] + [signal]
            record['dependencies'] = sorted(set(record['dependencies']) | set(evidence_refs))
            _io.write_json(self._path(turn_id), record)
            return True

    def outcome(self, turn_id, *, source_id, teacher, evidence_refs, attributed=False):
        if teacher not in {'owner_label', 'verified_outcome'} or not evidence_refs:
            raise ValueError('live proof needs an owner label or attributed verified outcome')
        if teacher == 'verified_outcome' and attributed is not True:
            raise ValueError('verified recall outcomes require attribution')
        with self.controller.lock:
            record = self.evidence(turn_id)
            if record is None or self.runtime is None:
                raise ValueError('outcome has no live recall evidence')
            record['dependencies'] = sorted(set(record['dependencies']) | {source_id} | set(evidence_refs))
            _io.write_json(self._path(turn_id), record)
            key = self.keyed(['source', source_id])
            return self.runtime.outcome(self.store.name, turn_id, teacher=teacher,
                                        served_correct=key in record['served'],
                                        reference_correct=key in record['reference'])

    async def live(self, runtime):
        from .service_health import off_loop
        await off_loop(self.attach, runtime)
        # Plan assumption: refresh reference-backed outcome joins once per minute.
        while True:
            await asyncio.to_thread(self.resolve_outcomes)
            await asyncio.sleep(60)

    def resolve_outcomes(self):
        """Consume only exact attribution already in the owner's lesson ledger.

        A retrieval correction is associated with the immediately preceding owner
        turn in the same session. Generic useful-claim counters do not locate the
        judging message, so they supply no proof here.
        """
        from ...api.routers.mind import get_mind
        mind = get_mind()
        if mind is None or getattr(mind, 'lessons', None) is None:
            return
        lessons = mind.lessons
        for lesson in lessons._fold().values():
            if (lesson.status not in {'active', 'superseded'} or lesson.verified != 'owner'
                    or lesson.correction != 'retrieval' or not (lesson.retrieval_source or '').startswith('turn:')):
                continue
            target = lesson.retrieval_source[5:]
            owner_evidence = [ref for ref in lesson.evidence if ref.startswith('turn:')]
            # The lesson log does not identify which of several cited turns was
            # the correcting message. Such a lesson cannot grade a particular recall.
            if len(owner_evidence) != 1:
                continue
            for evidence in owner_evidence:
                if not evidence.startswith('turn:'):
                    continue
                judging = evidence[5:]
                with closing(lessons.ledger._connect()) as conn:
                    source = conn.execute("SELECT * FROM turn_sources WHERE turn_id=? AND contact_id=? "
                                          "AND scope='person'", (judging, lessons.owner_id)).fetchone()
                    if source is None:
                        continue
                    previous = conn.execute("SELECT turn_id FROM turn_sources WHERE contact_id=? AND "
                        "session_id=? AND scope='person' AND coalesce(occurred_at,ingested_at) < ? "
                        "ORDER BY coalesce(occurred_at,ingested_at) DESC LIMIT 1",
                        (lessons.owner_id, source['session_id'], source['occurred_at'] or source['ingested_at'])).fetchone()
                if previous is None:
                    continue
                turn_id = previous['turn_id']
                self.add_source(turn_id, target, ledger=lessons.ledger, owner_id=lessons.owner_id)
                record = self.evidence(turn_id)
                if record is None:
                    continue
                # Only indexed, already attributed passages can become training targets.
                preferred = next((p['id'] for p in record['passages'] if p['source_id'] == target), None)
                disfavoured = next((key for key in record['shown'] if key != preferred), None)
                if preferred and disfavoured:
                    self.signal(turn_id, preferred=preferred, disfavoured=disfavoured, teacher='owner_label',
                                evidence_refs=[judging, target, 'mind:lesson:' + lesson.id + ':admitted'])
                self.outcome(turn_id, source_id=target, teacher='owner_label',
                             evidence_refs=[judging, target, 'mind:lesson:' + lesson.id + ':admitted'])

    def period(self):
        self.queries.prune()
        if self.runtime is not None:
            self.resolve_outcomes()
        period = self.store.periods.open()
        admitted = self.admission.admit(self.source.rows(self.objective), period=period.id,
                                        already=self.store.decided())
        self.store.periods.add_manifest(period.id, self.store.manifests.put(admitted.manifest))
        self.store.periods.close(period.id)
        self.last_period = period.id
        return admitted.manifest

    def build(self):
        return builds.rebuild(self.learner, self.objective, self.store.current_manifests(), self.source)

    def control(self):
        """Fit shuffled pair directions on exactly the admitted rows.

        Builder assumption: seed zero and equiprobable pair directions. Passage
        ids remain within each original pair, preserving its shown-set boundary.
        """
        rows = builds.training_rows(self.objective, self.store.current_manifests(), self.source)
        random = np.random.default_rng(0)
        shuffled = []
        for row in rows:
            label = dict(row.label)
            if random.random() >= .5:
                label['preferred'], label['disfavoured'] = label['disfavoured'], label['preferred']
            shuffled.append(replace(row, label=label))
        return self.learner.fit(self.learner.base(), shuffled, objective=self.objective, seed=0)

    def _capture(self, build):
        capture_path = self.store.directory / 'qualification' / (build.digest + '.json')
        if not capture_path.exists():
            return None
        return _io.read_json(capture_path)

    def _refit(self, build):
        from ...recall_cli import refit_query_adapter_cutoff
        capture = self._capture(build)
        if capture is None:
            return None
        transform = transform_identity(build.state)
        previous = self.store.revisions.serving()
        if previous.state == build.digest:
            previous = self.controller.previous(previous.id)
        if (capture.get('state') != build.digest or capture.get('query_transform') != transform
                or capture.get('document_identity') != self.document_identity
                or capture.get('reference') != previous.id or capture.get('role') != 'gate_suite'):
            raise ValueError('qualification does not match the complete revision and previous closure')
        if capture.get('control_state') != _io.state_digest(self.control()):
            raise ValueError('qualification does not match the fitted shuffled control')
        gate = capture['gate']
        protected = [*gate['ids'], *(row['query_id'] for split in ('development', 'held_out')
                                   for row in capture[split])]
        dependencies = set(build.refs)
        for record in self.records():
            if record['turn_id'] in build.refs:
                dependencies.update(record['dependencies'])
        if dependencies.intersection(protected):
            raise ValueError('qualification overlaps training references')
        self.sets.register(capture['set_id'], 'gate_suite', protected)
        self._qualification[build.digest] = capture
        consumed = self.store.directory / 'cutoff-checks' / (build.digest + '.json')
        if consumed.exists():
            saved = _io.read_json(consumed)
            if saved['capture'] != _io.digest(capture):
                raise ValueError('a revision already consumed its held-out check')
            return saved.get('record')
        # A crash or failed held-out check is consumed too, before evaluating it.
        _io.write_json(consumed, {'capture': _io.digest(capture), 'record': None})
        record = refit_query_adapter_cutoff(capture['metadata'], transform, capture['development'],
                                            capture['held_out'], reference_metrics=capture.get('reference_metrics'))
        _io.write_json(consumed, {'capture': _io.digest(capture), 'record': record})
        if record:
            _io.write_json(self.store.directory / 'cutoffs' / (record['id'].split(':')[1] + '.json'), record)
        return record

    def propose(self, build):
        previous = self.store.revisions.serving()
        cutoff = self._refit(build)
        revisions = self.store.directory / 'adapters' / build.digest
        for scope, prefix in (('global', ''), ('shown', 'shown_')):
            path = revisions / scope
            if not path.exists():
                save_revision(path, build.state[prefix + 'U'], build.state[prefix + 'V'],
                              bound=self.learner.bound, document_identity=self.document_identity, scope=scope)
        closure = self.store.revisions.compose(state=build.digest, objective=self.objective.identity,
            calibration=(cutoff['id'],) if cutoff else (),
            dependents={'recall_cutoff': cutoff['id'] if cutoff else 'unqualified'}, parent=previous.id,
            manifests=build.manifests, periods=tuple(p.id for p in self.store.periods.retained()),
            audience=build.audience, mode='rebuild')
        self.controller.propose(closure, build.state)
        self.store.budgets.record(BudgetCheck(self.store.name, build.digest, 'operator_norm',
                                              self.learner.bound, self.learner.budget_used(build.state)))
        return self.store.revisions.get(closure.id)

    def cutoff(self, pinned):
        if pinned.closure.mode == 'base':
            return None
        ident = pinned.closure.dependents['recall_cutoff']
        if ident not in pinned.closure.calibration or not ident.startswith('recall-cutoff-1:'):
            raise ValueError('closure lacks its qualified cutoff')
        record = _io.read_json(self.store.directory / 'cutoffs' / (ident.split(':')[1] + '.json'))
        payload = {key: value for key, value in record.items() if key != 'id'}
        # The refit uses compact sorted JSON for its content identity.
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                          allow_nan=False).encode()).hexdigest()
        if ident != 'recall-cutoff-1:' + digest or record['query_transform'] != transform_identity(pinned.state):
            raise ValueError('cutoff and adapter do not match')
        return record

    def gate(self, closure, build):
        capture = self._qualification.get(build.digest)
        if not capture or not closure.calibration:
            return GateReport(closure.id, 'recall', (Check('calibration', False,
                'matching development, held-out and gate-suite capture required'),))
        gate = capture['gate']
        for key in ('reference', 'base', 'candidate', 'control'):
            if any(type(value) is not bool for value in gate[key]):
                raise ValueError('gate correctness must be boolean')
        suite = Suite(capture['set_id'], tuple(Item(ref, ref, None) for ref in gate['ids']), ())
        restored = load_revision(self.store.directory / 'adapters' / build.digest / 'global',
                                 document_identity=self.document_identity)
        # Builder assumption: eight deterministic probes check serialized serving parity.
        probes = np.random.default_rng(0).normal(size=(8, self.learner.dims))
        evidence = Evidence(**{key: gate[key] for key in ('reference', 'base', 'candidate', 'control',
                           'canaries_base', 'canaries_candidate')},
            served=restored.apply(probes), offline=self.learner.apply(build.state, probes), metrics={},
            budget=self.store.budgets.result(self.store.name, build.digest), calibration=closure.calibration,
            parent_calibration=self.store.revisions.get(closure.parent).calibration, state=build.state)
        report = evaluate(closure.id, suite, evidence, sets=self.sets)
        cutoff = self.cutoff(Pinned(closure, build.state))
        missing = ', '.join(key for key, count in build.teacher_rows.items() if not count)
        self.controller._event('recall', closure.id, reason=(
            f"Query adapter {transform_identity(build.state)}; cutoff {cutoff['cutoff']:g}; "
            f"rows by teacher {dict(build.teacher_rows)}; unavailable teachers: {missing or 'none'}. "
            "Live comparison uses the identity, then the previous joint closure; proof requires owner labels "
            "or attributed verified outcomes; measured regression automatically rolls back adapter and cutoff."))
        return report


register_factory('query_adapter', Driver)
