"""Live owner recall: joint revisions, reference evidence and erasure."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from colonymind.plasticity import _io, builds
from colonymind.plasticity.learners.query_adapter import RecallRequest, transform_identity
from colonymind.plasticity.nightly import Runtime


class Config:
    def __init__(self, home, *, enabled=True, declared=True):
        self.home = home
        self.values = {"mind.faculties.plasticity": enabled,
                       "plasticity.learners": {"recall-test": declaration()} if declared else {}}

    def get(self, key, default=None):
        return self.values.get(key, default)


def declaration():
    return {"implementation": "query_adapter", "kind": "query-adapter", "base": "fixture-base",
            "mode": "rebuild", "personal_data": True, "audience": "owner", "dims": 4, "rank": 2,
            "document_identity": {"model": "fixture", "dimensions": 4},
            "teachers": {"owner_label": "pairwise", "verified_outcome": "pairwise",
                         "stronger_model": "pairwise"},
            "model": {"roles": ["query_adapter"], "change": "client_adapter"},
            "budget": {"type": "operator_norm", "bound": .05},
            "gates": ["retention", "specificity", "canaries", "calibration"],
            "arms": ["identity", "shuffled", "normalised_squared_error"],
            "erasure_path": "source erasure and clean rebuild", "dependents": ["recall_cutoff"],
            "effect": "rank", "chain": "recall"}


@pytest.fixture
def runtime(tmp_path):
    now = datetime(2026, 1, 10, tzinfo=timezone.utc)
    active = Runtime(Config(tmp_path), clock=lambda: now)
    assert active.rejected == {}
    yield active
    active.close()


def test_registered_owner_request_and_identity_fallback(runtime, tmp_path):
    assert runtime.find("query_adapter") == "recall-test"
    assert RecallRequest.create(runtime, owner=False, turn_id="contact") is None
    assert RecallRequest.create(runtime, owner=True, turn_id="") is None
    for config in (Config(tmp_path / "off", enabled=False), Config(tmp_path / "absent", declared=False)):
        inactive = Runtime(config)
        assert RecallRequest.create(inactive, owner=True, turn_id="turn") is None
    request = RecallRequest.create(runtime, owner=True, turn_id="turn")
    assert request is not None
    assert request.cutoff is None
    assert transform_identity(runtime.loaded['recall-test'].driver.learner.base()) == 'identity'


def test_both_factor_scopes_bind_the_transform(runtime):
    state = runtime.loaded['recall-test'].driver.learner.base()
    global_identity = transform_identity(state)
    state['shown_U'][0, 0] = .01
    assert transform_identity(state) != global_identity


def test_query_evidence_keeps_only_references_and_erasure_survives_retention(runtime):
    driver = runtime.loaded['recall-test'].driver
    driver.attach(runtime)
    vector = np.array([.1357911, .2468022, .3579133, .4680244], dtype=np.float32)
    driver.capture('turn', vector, [{'id': 'chunk-a', 'source_id': 'source-a', 'source_kind': 'owner'}],
                   ['source-a'], ['source-b'])
    assert driver.queries.get('turn') is not None
    record = driver.evidence('turn')
    assert 'query' not in record and 'vector' not in record
    assert record['passages'][0]['source_id'] == 'source-a'
    for path in runtime.root.rglob('*'):
        if path.is_file() and path != driver.queries.path:
            assert vector.tobytes() not in path.read_bytes()
    driver.queries._clock = lambda: datetime(2026, 3, 1, tzinfo=timezone.utc)
    driver.queries.prune()
    assert driver.queries.get('turn') is None
    assert 'turn' in driver.source.source.refs_resting_on(['source-a'])
    assert driver.evidence('turn') is None
    assert not driver.capture('turn', vector, [], [], [])


def training(driver, *, turn='training', source='source-a', query=(.6, .8, 0., 0.)):
    driver.capture(turn, query,
        [{'id': source, 'source_id': source, 'source_kind': 'owner'},
         {'id': source + '-negative', 'source_id': source + '-negative', 'source_kind': 'owner'}],
        [source], [source + '-negative'])
    record = driver.evidence(turn)
    record['at'] = (driver.clock() - timedelta(days=5)).isoformat()
    record['shown'] = {source: 1, source + '-negative': 2}
    _io.write_json(driver._path(turn), record)
    assert driver.signal(turn, preferred=source, disfavoured=source + '-negative', teacher='owner_label',
                         evidence_refs=['label-' + turn])
    driver.passage_reader = lambda passage: [0., 1., 0., 0.] if not passage['id'].endswith('negative') else [1., 0., 0., 0.]


def qualification(driver, build, *, bad=False):
    from colonymind.memory.recall import calibration_fingerprint
    transform = transform_identity(build.state)
    metadata = {'provider': 'fixture', 'model': 'fixture', 'index_generation': 'fixed'}
    fingerprint = calibration_fingerprint({**metadata, 'query_transform': transform})
    def split(prefix):
        return [{'query_id': prefix + '-answer', 'scores': [.9], 'relevant': [True], 'no_answer': False,
                 'calibration_fingerprint': fingerprint}] + [
            {'query_id': f'{prefix}-{index}', 'scores': [.2 if not bad or prefix == 'dev' else .95],
             'relevant': [False], 'no_answer': True, 'calibration_fingerprint': fingerprint}
            for index in range(150)]
    capture = {'state': build.digest, 'control_state': _io.state_digest(driver.control()),
        'query_transform': transform, 'document_identity': driver.document_identity,
        'reference': driver.store.revisions.serving().id, 'role': 'gate_suite', 'set_id': 'public-gates',
        'metadata': metadata, 'development': split('dev'), 'held_out': split('held'),
        'gate': {'ids': [f'gate-{index}' for index in range(1000)],
                 'base': [True] * 1000, 'reference': [True] * 1000, 'candidate': [True] * 1000,
                 'control': [False] * 1000, 'canaries_base': {'secret': 0.}, 'canaries_candidate': {'secret': 0.}}}
    _io.write_json(driver.store.directory / 'qualification' / (build.digest + '.json'), capture)
    return capture


def promote(runtime):
    driver = runtime.loaded['recall-test'].driver
    driver.attach(runtime)
    training(driver)
    driver.period()
    build = driver.build()
    assert build.rows == 1
    qualification(driver, build)
    results = runtime.run()
    assert results[0]['result'] == 'promoted', results
    return runtime.pin('recall-test', 'owner')


def test_nightly_promotes_adapter_and_cutoff_together_and_failed_refit_keeps_them(runtime):
    pinned = promote(runtime)
    driver = runtime.loaded['recall-test'].driver
    assert driver.cutoff(pinned)['query_transform'] == transform_identity(pinned.state)
    assert runtime.pin_chain('recall', 'owner')['recall-test'].closure.id == pinned.closure.id
    training(driver, turn='another', source='source-b', query=(.8, .6, 0., 0.))
    driver.period()
    changed = driver.build()
    assert changed.digest != pinned.closure.state
    qualification(driver, changed, bad=True)
    result = runtime.run()
    assert result[0]['result'] == 'dropped'
    assert runtime.pin('recall-test', 'owner').closure.id == pinned.closure.id
    assert driver.cutoff(runtime.pin('recall-test', 'owner')) == driver.cutoff(pinned)


def test_held_out_consumption_is_persistent_even_when_it_fails(runtime):
    driver = runtime.loaded['recall-test'].driver
    driver.attach(runtime)
    training(driver)
    driver.period()
    build = driver.build()
    qualification(driver, build, bad=True)
    assert driver._refit(build) is None
    qualification(driver, build)
    with pytest.raises(ValueError, match='consumed'):
        driver._refit(build)


def test_source_erasure_unserves_rebuilds_and_forgets_exported_private_factors(runtime):
    from colonymind.turns.erasure_hooks import notify_source_erasure
    pinned = promote(runtime)
    driver = runtime.loaded['recall-test'].driver
    notify_source_erasure(['source-a'])
    assert runtime.pin('recall-test', 'owner').closure.id != pinned.closure.id
    assert driver.queries.get('training') is None
    assert driver.evidence('training') is None
    clean = builds.rebuild(driver.learner, driver.objective, [], driver.source)
    rebuilt = runtime.loaded['recall-test'].last_build
    for key in clean.state:
        np.testing.assert_allclose(rebuilt.state[key], clean.state[key], rtol=1e-5)
    assert not (driver.store.directory / 'adapters' / pinned.closure.state).exists()


def test_only_attributed_proof_can_roll_back_joint_closure_and_report(runtime):
    from colonymind.reports.plasticity import lines
    pinned = promote(runtime)
    driver = runtime.loaded['recall-test'].driver
    for index in range(12):
        turn = f'live-{index}'
        runtime.serve('recall-test', turn, [.6, .8, 0., 0.],
            predict=lambda pin, inputs: ['bad'] if pin.closure.mode != 'base' else ['good'])
        driver.capture(turn, [.6, .8, 0., 0.], [], ['bad'], ['good'])
        with pytest.raises(ValueError):
            driver.outcome(turn, source_id='good', teacher='deliberation', evidence_refs=['label'])
        with pytest.raises(ValueError):
            driver.outcome(turn, source_id='good', teacher='verified_outcome', evidence_refs=['label'])
        result = driver.outcome(turn, source_id='good', teacher='owner_label', evidence_refs=['label'])
        if result['rolled_back']:
            break
    assert result['rolled_back']
    base = runtime.pin('recall-test', 'owner')
    assert base.closure.mode == 'base' and driver.cutoff(base) is None
    assert runtime.pin_chain('recall', 'owner')['recall-test'].closure.id == base.closure.id
    assert 'cutoff' in ' '.join(lines(runtime.report()))
    assert any(row['event'] == 'rolled_back' for row in driver.controller.events())
    assert pinned.closure.id != base.closure.id


def test_erasing_a_turn_before_capture_prevents_it_being_logged_later(runtime):
    from colonymind.turns.erasure_hooks import notify_source_erasure
    driver = runtime.loaded['recall-test'].driver
    driver.attach(runtime)
    notify_source_erasure(['not-captured-yet'])
    assert not driver.capture('not-captured-yet', [.6, .8, 0., 0.], [], [], [])
    assert driver.queries.get('not-captured-yet') is None


def test_qualification_refuses_training_overlap_in_passage_or_label_dependency(runtime):
    driver = runtime.loaded['recall-test'].driver
    driver.attach(runtime)
    training(driver)
    driver.period()
    build = driver.build()
    capture = qualification(driver, build)
    capture['gate']['ids'][0] = 'label-training'
    _io.write_json(driver.store.directory / 'qualification' / (build.digest + '.json'), capture)
    with pytest.raises(ValueError, match='overlap'):
        driver._refit(build)


def test_shown_only_transform_uses_the_revision_identity_contract(runtime):
    from colonymind.vector.indexes import query_transform_identity
    state = runtime.loaded['recall-test'].driver.learner.base()
    state['shown_U'][0, 0] = .01
    assert query_transform_identity(transform_identity(state)) == transform_identity(state)


def test_unchanged_nightly_build_reuses_qualification_without_rollback(runtime):
    pinned = promote(runtime)
    result = runtime.run()
    assert result[0]['result'] == 'promoted', result
    assert runtime.pin('recall-test', 'owner').closure.id == pinned.closure.id


def test_reranker_does_not_label_other_chunks_of_a_source(runtime):
    driver = runtime.loaded['recall-test'].driver
    driver.attach(runtime)
    driver.capture('chunked', [.6, .8, 0., 0.], [
        {'id': 'shown-chunk', 'source_id': 'source', 'source_kind': 'owner', 'message_hash': 'first'},
        {'id': 'unshown-chunk', 'source_id': 'source', 'source_kind': 'owner', 'message_hash': 'second'},
        {'id': 'negative', 'source_id': 'other', 'source_kind': 'owner', 'message_hash': 'other'}], [], [])
    driver.record_trace('chunked', {'rows': [
        {'turn_id': 'source', 'message_hash': 'first', 'submitted': True, 'rerank_score': .9},
        {'turn_id': 'other', 'message_hash': 'other', 'submitted': True, 'rerank_score': .1}]})
    assert set(driver.evidence('chunked')['reranked']) == {'shown-chunk', 'negative'}


def test_retrieval_correction_can_read_a_missed_owner_passage_from_existing_index(runtime):
    from colonymind.turns import TurnIdempotencyLedger
    driver = runtime.loaded['recall-test'].driver
    driver.attach(runtime)
    driver.capture('miss', [.6, .8, 0., 0.], [], [], [])
    ledger = TurnIdempotencyLedger(runtime.data_dir / 'turn-idempotency.db')
    ledger.record_source('earlier', contact_id='owner', session_id='session',
                         messages=[{'role': 'user', 'content': 'The code is in the green folder.'}])
    ledger.record_source('contact-source', contact_id='contact', session_id='session',
                         messages=[{'role': 'user', 'content': 'A different source.'}])
    driver.passage_reader = lambda passage: [0., 1., 0., 0.]
    driver.add_source('miss', 'earlier', ledger=ledger, owner_id='owner')
    driver.add_source('miss', 'contact-source', ledger=ledger, owner_id='owner')
    record = driver.evidence('miss')
    assert {p['source_id'] for p in record['passages']} == {'earlier'}
    assert 'earlier' in record['dependencies']
    assert 'contact-source' not in record['dependencies']


def test_source_erasure_retries_after_query_deletion_before_runtime_unserve(runtime, monkeypatch):
    from colonymind.turns.erasure_hooks import notify_source_erasure
    pinned = promote(runtime)
    driver = runtime.loaded['recall-test'].driver
    original = runtime.erase
    def interrupted(refs):
        raise RuntimeError('interrupted before runtime erasure')
    monkeypatch.setattr(runtime, 'erase', interrupted)
    with pytest.raises(RuntimeError):
        notify_source_erasure(['source-a'])
    assert driver.queries.get('training') is None
    monkeypatch.setattr(runtime, 'erase', original)
    notify_source_erasure(['source-a'])
    assert runtime.pin('recall-test', 'owner').closure.id != pinned.closure.id
