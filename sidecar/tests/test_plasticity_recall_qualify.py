"""Qualification captures are measured, bounded and reference-only."""
import argparse
import json

import pytest

from colonymind.plasticity import cli
from test_plasticity_query_adapter_driver import runtime, training  # noqa: F401


def queries():
    def row(key, answer=True):
        return {"query_id": key, "query": "Private question " + key,
                "relevant": ["answer"] if answer else [], "no_answer": not answer}
    return {"role": "gate_suite", "set_id": "local-gates",
            "development": [row("dev-answer"), row("dev-empty", False)],
            "held_out": [row("held-answer")] + [row(f"held-{i}", False) for i in range(150)],
            "gate": [row(f"gate-{i}") for i in range(1000)],
            "canaries": [dict(row("privacy", False), forbidden=["private-source"])]}


def test_qualify_parser():
    parser = argparse.ArgumentParser()
    cli.add_parser(parser.add_subparsers())
    args = parser.parse_args(["plasticity", "qualify", "recall-test", "--queries", "queries.json"])
    assert args.plasticity_command == "qualify"


@pytest.mark.asyncio
async def test_run_command_imports_before_loading_config(monkeypatch):
    from colonymind import config
    from colonymind.plasticity.recall_qualify import run_command

    def unavailable_config():
        raise FileNotFoundError("qualification config unavailable")

    monkeypatch.setattr(config, "load_config", unavailable_config)
    with pytest.raises(FileNotFoundError, match="qualification config unavailable"):
        await run_command(argparse.Namespace())


def test_qualify_command_emits_capture_and_budget_error(monkeypatch, capsys):
    from colonymind.plasticity import recall_qualify
    parser = argparse.ArgumentParser()
    cli.add_parser(parser.add_subparsers())
    async def run(args):
        assert args.learner == 'recall-test'
        return {'capture': 'qualification/state.json'}
    monkeypatch.setattr(recall_qualify, 'run_command', run)
    args = parser.parse_args(['plasticity', 'qualify', 'recall-test', '--queries', 'input.json', '--json'])
    assert cli.run(args) == 0
    assert json.loads(capsys.readouterr().out) == {'capture': 'qualification/state.json'}
    async def exhausted(args):
        raise TimeoutError('qualification query budget exhausted')
    monkeypatch.setattr(recall_qualify, 'run_command', exhausted)
    assert cli.run(args) == 2
    assert 'budget exhausted' in capsys.readouterr().err


@pytest.mark.asyncio
async def test_capture_contract_and_no_text(runtime):
    from colonymind.plasticity.recall_qualify import qualify
    driver = runtime.loaded["recall-test"].driver
    driver.attach(runtime)
    training(driver)
    driver.period()
    scorer = FixtureScorer(driver)
    path = await qualify(driver, queries(), scorer=scorer)
    capture = json.loads(path.read_text())
    build = driver.build()
    assert capture["state"] == build.digest
    assert driver._refit(build) is not None
    closure = driver.propose(build)
    assert driver.gate(closure, build).passed
    assert scorer.calls == 2 + 151 + 4 * 1001
    for file in driver.store.directory.rglob("*"):
        if file.is_file():
            assert b"Private question" not in file.read_bytes()


class FixtureScorer:
    metadata = {"provider": "fixture", "model": "fixture", "index_generation": "fixed"}

    def __init__(self, driver):
        self.document_identity = driver.document_identity
        self.calls = 0

    async def score(self, row, state, *, cutoff=None, arm="candidate"):
        self.calls += 1
        answer = not row["no_answer"]
        selected = ["answer"] if answer and arm != "control" else []
        return {"scores": [.9 if answer else .2], "relevant": [answer], "selected": selected}


@pytest.mark.asyncio
@pytest.mark.parametrize("change,reason", [
    ("sealed", "gate_suite"), ("overlap", "overlap"), ("duplicate", "distinct"),
    ("held", "150"), ("gate", "1000"), ("identity", "identity"),
    ("canaries", "canaries"), ("missing_answers", "each calibration"),
    ("sealed_name", "role"), ("changed_set", "frozen"),
])
async def test_refusals_before_scoring(runtime, change, reason):
    from colonymind.plasticity.recall_qualify import qualify
    driver = runtime.loaded["recall-test"].driver
    driver.attach(runtime)
    training(driver)
    driver.period()
    payload, scorer = queries(), FixtureScorer(driver)
    if change == "sealed":
        payload["role"] = "sealed"
    elif change == "overlap":
        payload["gate"][0]["query_id"] = "label-training"
    elif change == "duplicate":
        payload["held_out"][0]["query_id"] = payload["development"][0]["query_id"]
    elif change == "held":
        payload["held_out"].pop()
    elif change == "gate":
        payload["gate"].pop()
    elif change == "identity":
        scorer.document_identity = {"other": True}
    elif change == "canaries":
        payload['canaries'] = []
    elif change == "missing_answers":
        payload['development'].pop(0)
    elif change == "sealed_name":
        driver.sets.register(payload['set_id'], 'sealed', ['unrelated'])
    else:
        driver.sets.register(payload['set_id'], 'gate_suite', ['unrelated'])
    with pytest.raises(ValueError, match=reason):
        await qualify(driver, payload, scorer=scorer)
    assert scorer.calls == 0
    assert not list(driver.store.directory.glob("qualification/*"))


@pytest.mark.asyncio
async def test_budget_stops_without_writing(runtime):
    from colonymind.plasticity.recall_qualify import qualify
    driver = runtime.loaded["recall-test"].driver
    scorer = FixtureScorer(driver)
    with pytest.raises(TimeoutError, match="budget"):
        await qualify(driver, queries(), scorer=scorer, max_queries=1)
    assert not list(driver.store.directory.glob("qualification/*"))


@pytest.mark.asyncio
async def test_measured_gain_promotes_and_loss_preserves_joint_closure(runtime):
    from colonymind.plasticity.recall_qualify import qualify
    driver = runtime.loaded['recall-test'].driver
    driver.attach(runtime)
    training(driver)
    driver.period()
    await qualify(driver, queries(), scorer=FixtureScorer(driver))
    assert runtime.run()[0]['result'] == 'promoted'
    first = runtime.pin('recall-test', 'owner')
    first_cutoff = driver.cutoff(first)
    training(driver, turn='next-training', source='next-source', query=(.8, .6, 0., 0.))
    driver.period()
    class Loss(FixtureScorer):
        async def score(self, row, state, *, cutoff=None, arm='candidate'):
            result = await super().score(row, state, cutoff=cutoff, arm=arm)
            if arm == 'candidate' and row['query_id'].startswith('gate-'):
                result['selected'] = []
            return result
    await qualify(driver, queries(), scorer=Loss(driver))
    assert runtime.run()[0]['result'] == 'dropped'
    after = runtime.pin('recall-test', 'owner')
    assert after.closure.id == first.closure.id
    assert driver.cutoff(after) == first_cutoff


@pytest.mark.asyncio
async def test_capture_is_immutable_and_sealed_members_cannot_be_reused(runtime):
    from colonymind.plasticity.recall_qualify import qualify
    driver = runtime.loaded['recall-test'].driver
    scorer = FixtureScorer(driver)
    driver.sets.register('sealed-release', 'sealed', ['gate-7'])
    with pytest.raises(ValueError, match='sealed'):
        await qualify(driver, queries(), scorer=scorer)
    assert scorer.calls == 0
    payload = queries()
    payload['gate'][7]['query_id'] = 'replacement-gate'
    path = await qualify(driver, payload, scorer=scorer)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='already'):
        await qualify(driver, payload, scorer=scorer)
    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_time_budget_cancels_score_without_capture(runtime):
    import asyncio
    from colonymind.plasticity.recall_qualify import qualify
    driver = runtime.loaded['recall-test'].driver
    class Slow(FixtureScorer):
        async def score(self, *args, **kwargs):
            await asyncio.sleep(1)
    with pytest.raises(TimeoutError, match='budget'):
        await qualify(driver, queries(), scorer=Slow(driver), max_seconds=.01)
    assert not list(driver.store.directory.glob('qualification/*'))


@pytest.mark.asyncio
async def test_real_index_and_selector_with_fake_providers(tmp_path, monkeypatch):
    import asyncio
    from dataclasses import asdict
    from recall_route import recall_app, OWNER, ANSWER
    from colonymind.plasticity.nightly import Runtime
    from colonymind.plasticity.recall_qualify import RecallScorer
    from test_plasticity_query_adapter_driver import Config
    app = await asyncio.wait_for(recall_app(monkeypatch, tmp_path / 'instance', initiatives=0), 15)
    config = Config(tmp_path / 'learner')
    config.values['plasticity.learners']['recall-test'].update(dims=3,
        document_identity=asdict(app.embedder.index_identity.document_identity))
    active = Runtime(config)
    try:
        driver = active.loaded['recall-test'].driver
        scorer = RecallScorer(driver, ledger=app.ledger, store=app.store, pipeline=app.embedder,
                              reranker=app.reranker, owner=OWNER)
        result = await scorer.score({'query_id': 'probe', 'query': 'hydrofoil departure',
            'relevant': ['answer'], 'no_answer': False}, driver.learner.base())
        assert 'answer' in result['selected']
        assert any(relevant and score == .99 for relevant, score in zip(result['relevant'], result['scores']))
        assert len(app.embedder.queries) == 1
        assert app.reranker.calls
        assert not driver.records()
        for path in driver.store.directory.rglob('*'):
            if path.is_file():
                assert ANSWER.encode() not in path.read_bytes()
                assert b'hydrofoil departure' not in path.read_bytes()
    finally:
        active.close()
        await app.store.close()


class FixturePipeline:
    from colonymind.vector.indexes import EmbeddingIdentity
    index_identity = EmbeddingIdentity('fixture', 'fixture', '1', 4)

    async def embed_query(self, query):
        return [.6, .8, 0., 0.]


class FixtureIndex:
    """An exact fixture index, with canonical projection metadata and query vectors."""
    identity = FixturePipeline.index_identity

    def __init__(self, ledger):
        from types import SimpleNamespace
        from colonymind.turns.source_vectors import chunks
        self.calls = []
        self.catalog = SimpleNamespace(read_generation=lambda identity: {'id': 'fixture-generation'})
        self.rows = []
        with ledger._connect() as conn:
            for source in conn.execute('SELECT * FROM turn_sources').fetchall():
                self.rows.extend(chunks(conn, source))

    async def _table(self, *args, **kwargs):
        from types import SimpleNamespace
        async def schema():
            return SimpleNamespace(names=['scope_key'])
        return SimpleNamespace(schema=schema)

    @staticmethod
    def _quoted(value):
        return repr(value)

    async def search(self, collection, vector, **kwargs):
        from colonymind.vector.query import VectorResult
        self.calls.append(vector)
        return [VectorResult(meta['projection_id'], .9, text, meta) for text, meta in self.rows]


@pytest.mark.asyncio
async def test_fixture_index_producer_runs_selection_for_every_arm(runtime, monkeypatch, tmp_path):
    import asyncio
    from dataclasses import asdict
    from colonymind.memory.recall import calibration_fingerprint, provider_calibration_metadata
    from colonymind.plasticity.learners.query_adapter import current_request, transform_identity
    from colonymind.plasticity.recall_qualify import qualify, RecallScorer
    from colonymind.turns import TurnIdempotencyLedger
    driver = runtime.loaded['recall-test'].driver
    driver.attach(runtime)
    driver.document_identity = asdict(FixturePipeline.index_identity.document_identity)
    training(driver)
    driver.period()
    control = transform_identity(driver.control())
    candidate = transform_identity(driver.build().state)
    assert control != candidate
    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    ledger.record_source('answer', contact_id='owner', session_id='past',
                         messages=[{'role': 'user', 'content': 'The answer is in the green folder.'}])
    index = FixtureIndex(ledger)
    payload = queries()
    no_answer = {row['query'] for split in ('development', 'held_out', 'canaries')
                 for row in payload[split] if row['no_answer']}
    class Reranker:
        def calibration_metadata(self):
            return {'provider': 'fixture', 'model': 'fixture-reranker'}
        async def rerank(self, query, documents, top_k):
            score = .2 if query in no_answer or current_request().transform == control else .9
            return [{'index': i, 'score': score} for i in range(len(documents))]
    reranker = Reranker()
    monkeypatch.setenv('COLONYMIND_RECALL_RERANK_MIN_SCORE', '.5')
    monkeypatch.setenv('COLONYMIND_RECALL_RERANK_CALIBRATION',
                       calibration_fingerprint(provider_calibration_metadata(reranker)))
    scorer = RecallScorer(driver, ledger=ledger, store=index, pipeline=FixturePipeline(),
                          reranker=reranker, owner='owner')
    # This verifies 4,157 complete selections, not per-query latency. Give the
    # bounded batch room on CI; the recall latency gates keep their own limits.
    path = await asyncio.wait_for(qualify(driver, payload, scorer=scorer), 180)
    capture = json.loads(path.read_text())
    assert all(capture['gate']['candidate'])
    assert not any(capture['gate']['control'][:1000])
    assert len(index.calls) == 4157
    assert runtime.run()[0]['result'] == 'promoted'
    assert b'Private question' not in path.read_bytes()
