"""The paired claim worker offers prior records found by meaning, through the served host's vector index.

In a paired plugin arm the claim worker runs on its own thread and loop (its model requests are told apart by that
thread), while the vector store and the embedding pipeline belong to the served host's loop. The worker therefore
leaves vector jobs to the host, and until now its projection had no vector search at all: the priors' vector lane
(claims v18) never ran in a paired arm. The worker's projection now searches the host's index on the host's loop in
every plugin arm, and the sidecar's own switch decides whether it searches at all: the arm that ablates exactly that
lane (``minus_priors_by_meaning``) exports ``COLONYMIND_PRIORS_BY_MEANING=0``, as an install does.
"""
import asyncio
import threading

import pytest

from colonymind.beliefs import source_projection
from colonymind.beliefs.source_projection import SourceClaimProjection
from colonymind.qualification import paired_worker
from test_qualification_paired_runner import fixture  # noqa: F401  (pytest fixture)
from test_source_claim_priors import FILLERS, MOVED, OFFICE, ChoosingModel, Pipeline, claim_rows, outputs, store


class LoopThread:
    """An event loop on its own named thread, as the served host (uvicorn) and the claim worker each have."""

    def __init__(self, name):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name=name, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def run(self, coroutine, timeout=60):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout)

    def __exit__(self, *exc):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(10)
        self.loop.close()


def form(tmp_path, monkeypatch, *, lane, open_host=True):
    """OFFICE and twenty fillers, then the paraphrased change, each extracted on the worker's thread while the
    vectors are embedded and searched on the host's; returns the ledger, the scripted model and Berlin's claim.
    The worker's lane is given in every arm, as ``source_worker`` gives it; ``lane`` is the flag the arm exports."""
    monkeypatch.setenv(source_projection.PRIORS_BY_MEANING_FLAG, '1' if lane else '0')
    host, worker = LoopThread('host-loop'), LoopThread(paired_worker.SOURCE_WORKER_THREAD)
    with host, worker:
        ledger, vectors = host.run(store(tmp_path, Pipeline()))
        projection = SourceClaimProjection(ledger)        # as run_source_claim_worker builds it with vectors=False
        model = ChoosingModel(outputs())
        searched = paired_worker.HostVectors()
        if open_host:
            searched.open(host.loop, vectors)

        async def drain():
            while await vectors.process_one():
                pass

        def say(texts, start):
            for index, text in enumerate(texts, start):
                ledger.record_source(f'contact-a-t{index}', contact_id='contact-a', session_id=f's{index}',
                                     occurred_at=f'2026-03-01T{index // 60:02d}:{index % 60:02d}:00+00:00',
                                     messages=[{'role': 'user', 'content': text}])
                assert worker.run(projection.process_one(model))
            host.run(drain())

        with paired_worker.prior_vector_lane(worker.thread, searched):
            say([OFFICE, *FILLERS], 0)
            berlin, = [row for row in claim_rows(ledger).values() if row['value'] == 'Berlin']
            model.calls.clear()
            say([MOVED], 100)
        return ledger, model, berlin, searched


def test_the_worker_on_its_own_thread_is_offered_a_prior_found_by_meaning_on_the_hosts_loop(tmp_path, monkeypatch):
    ledger, model, berlin, searched = form(tmp_path, monkeypatch, lane=True)
    offered = [row['id'] for row in model.calls[0][0]['prior_assertions']]
    assert berlin['id'] in offered and len(offered) <= 16
    rows = claim_rows(ledger)
    munich, = [row for row in rows.values() if row['value'] == 'Munich' and row['subject_key'] == 'speaker']
    assert munich['operation'] == 'change' and munich['prior_claim_id'] == berlin['id']
    assert rows[berlin['id']]['superseded_by'] == munich['id']
    assert searched.counts['searches'] >= 1 and searched.counts['found'] >= 1 and searched.counts['failed'] == 0


def test_the_ablation_leaves_the_worker_its_keyword_lane_alone(tmp_path, monkeypatch):
    """``minus_priors_by_meaning``: the sidecar's switch is off, so the worker's priors are exactly what they were
    before the vector lane, and the host's index is never searched, though the lane is given."""
    ledger, model, berlin, searched = form(tmp_path, monkeypatch, lane=False)
    offered = [row['id'] for row in model.calls[0][0]['prior_assertions']]
    assert berlin['id'] not in offered
    assert searched.counts['searches'] == 0


def test_without_the_hosts_embedder_the_lane_finds_nothing_and_the_keyword_lane_stands(tmp_path, monkeypatch):
    """An arm whose host opened no embedder (no endpoint in the plan, or semantic recall off): the lane is asked
    and finds nothing, as on an install without vectors."""
    ledger, model, berlin, searched = form(tmp_path, monkeypatch, lane=True, open_host=False)
    assert berlin['id'] not in [row['id'] for row in model.calls[0][0]['prior_assertions']]
    assert searched.counts['found'] == 0 and searched.counts['unavailable'] >= 1


def test_only_the_workers_thread_gets_the_lane(tmp_path):
    """Every other projection keeps its own vectors, or none: the host's dry run passes its own, and a projection
    on any other thread is left alone."""
    async def probe(projection):
        return await projection.similar_turns({'turn_id': 'x', 'contact_id': 'contact-a', 'session_id': 's'},
                                              {'role': 'user', 'content': MOVED})
    host, worker = LoopThread('host-loop'), LoopThread(paired_worker.SOURCE_WORKER_THREAD)
    with host, worker:
        ledger, vectors = host.run(store(tmp_path, Pipeline()))
        ledger.record_source('contact-a-t0', contact_id='contact-a', session_id='s0',
                             messages=[{'role': 'user', 'content': OFFICE}])

        async def drain():
            while await vectors.process_one():
                pass
        host.run(drain())
        searched = paired_worker.HostVectors()
        searched.open(host.loop, vectors)
        with paired_worker.prior_vector_lane(worker.thread, searched):
            assert host.run(probe(SourceClaimProjection(ledger))) == []
            assert worker.run(probe(SourceClaimProjection(ledger))) == ['contact-a-t0']
        # Outside the episode nothing is patched.
        assert worker.run(probe(SourceClaimProjection(ledger))) == []


def test_a_search_the_host_never_answers_is_bounded_and_the_keyword_lane_stands(tmp_path, monkeypatch):
    monkeypatch.setattr(paired_worker, 'PRIOR_SEARCH_WAIT_S', 0.2)

    class Hung:
        async def search(self, *args, **kwargs):
            await asyncio.sleep(3600)
    host, worker = LoopThread('host-loop'), LoopThread(paired_worker.SOURCE_WORKER_THREAD)
    with host, worker:
        searched = paired_worker.HostVectors()
        searched.open(host.loop, Hung())
        ledger = host.run(store(tmp_path, Pipeline()))[0]
        with paired_worker.prior_vector_lane(worker.thread, searched):
            found = worker.run(SourceClaimProjection(ledger).similar_turns(
                {'turn_id': 'x', 'contact_id': 'contact-a', 'session_id': 's'}, {'role': 'user', 'content': MOVED}))
        assert found == [] and searched.counts['failed'] == 1


def test_the_served_host_publishes_its_index_for_the_worker_while_it_runs(tmp_path, monkeypatch):
    """The host's lifespan opens the search the worker's lane uses on the host's own loop, and withdraws it before
    the store closes; an arm without an embedder opens nothing."""
    import colonymind.vector as vector
    from fastapi import FastAPI
    from test_qualification_mind_arm import EmbeddingEndpoint
    from colonymind.qualification import native_memory_worker as worker
    from colonymind.qualification import paired
    monkeypatch.setattr(vector, '_store', None)
    monkeypatch.setattr(vector, '_pipeline', None)
    monkeypatch.setattr(paired_worker, 'HOST_VECTORS', paired_worker.HostVectors())
    seen = {}
    with EmbeddingEndpoint() as endpoint:
        embedding = {'base_url': endpoint.url, 'model': 'e', 'dimensions': EmbeddingEndpoint.DIMENSIONS}
        for name in ('all-on', 'all-on-semantic_recall'):
            section = worker.mind_section(paired_worker.mind_switches(paired.PROFILES[name]))
            for key, value in worker.embedding_environment({'embedding': embedding}, section).items():
                monkeypatch.setenv(key, value)

            async def run():
                async with paired_worker.provider_read_lifespan(tmp_path / name)(FastAPI()):
                    target = paired_worker.HOST_VECTORS.target
                    return target is not None and target[0] is asyncio.get_running_loop()
            seen[name] = asyncio.run(run())
            assert paired_worker.HOST_VECTORS.target is None
    assert seen == {'all-on': True, 'all-on-semantic_recall': False}


def test_a_plan_with_an_embedding_endpoint_refuses_an_image_whose_worker_leaves_the_priors_keyword_only(
        fixture, monkeypatch):
    """An image built before the lane runs every embedding arm with keyword priors only: with an endpoint in the
    plan it is refused, so no series mixes the two."""
    from colonymind.qualification import paired, paired_container
    from test_qualification_mind_arm import EMBEDDING
    original = paired_container.configuration

    def older(*args, **kwargs):
        supplied, recipe = original(*args, **kwargs)
        recipe['container_payload'] = {**recipe['container_payload'], 'embedding': 'paired-embedding-1'}
        return supplied, recipe
    monkeypatch.setattr(paired_container, 'configuration', older)
    with pytest.raises(ValueError, match='embedding endpoint'):
        paired.plan(fixture.output, native_binding='candidate', evidence_mode='controlled',
                    arms=['base_hermes', 'all-on'], embedding=dict(EMBEDDING), **fixture.resources)
    monkeypatch.setattr(paired_container, 'configuration', original)
    manifest = paired.plan(fixture.output, native_binding='candidate', evidence_mode='controlled',
                           arms=['base_hermes', 'all-on'], embedding=dict(EMBEDDING), **fixture.resources)
    assert manifest['comparison']['embedding'] == EMBEDDING


@pytest.mark.parametrize('flag, lane', [(None, 'on'), ('1', 'on'), ('0', 'off')])
def test_the_lanes_counts_reach_the_episode_evidence(monkeypatch, flag, lane):
    """The evidence says what the running projection reads (the exported flag, on where none is set), not what the
    arm's name says."""
    host = paired_worker.HostVectors()
    host.counts.update(searches=3, found=2)
    monkeypatch.setattr(paired_worker, 'HOST_VECTORS', host)
    if flag is None:
        monkeypatch.delenv(source_projection.PRIORS_BY_MEANING_FLAG, raising=False)
    else:
        monkeypatch.setenv(source_projection.PRIORS_BY_MEANING_FLAG, flag)
    assert paired_worker.priors_by_meaning_evidence() == {
        'lane': lane, 'searches': 3, 'found': 2, 'failed': 0, 'unavailable': 0}
