"""The paired plan's reranker: one endpoint for every plugin arm, with the fingerprint its cutoff was calibrated under.

Recall's calibrated cutoff applies only while the reranker's configuration matches the fingerprint it was
calibrated under; on a mismatch it is switched off, silently. A plan now takes ``--rerank-config`` (endpoint base,
model, prompt style, revision, cutoff), computes the fingerprint the served host will compute for it, refuses a
declared fingerprint that differs, and records it in the comparison. The worker gives every plugin arm the reranker
through the environment an install exports, and its served host opens it and reports whether the cutoff applies.
"""
import asyncio
import math
import os

import pytest
from fastapi import FastAPI

from colonymind.api.routers import host
from colonymind.memory.recall import calibration_fingerprint, provider_calibration_metadata
from colonymind.qualification import native_memory_worker as worker
from colonymind.qualification import paired, paired_worker
from test_qualification_mind_arm import EMBEDDING
from test_qualification_paired_runner import fixture  # noqa: F401  (pytest fixture)

CONFIGURED = {'base_url': 'http://127.0.0.1:9102', 'model': 'rerank-model', 'prompt_style': 'qwen3',
              'revision': 'rev-1', 'cutoff': 0.62}


def fingerprint(rerank=CONFIGURED, embedding=EMBEDDING):
    """What the calibration recorded for ``rerank`` under ``embedding`` (``COLONYMIND_RECALL_RERANK_CALIBRATION``)."""
    return paired.rerank_fingerprint({key: value for key, value in rerank.items() if key != 'fingerprint'}, embedding)


RERANK = {**CONFIGURED, 'fingerprint': fingerprint()}


@pytest.mark.parametrize('bad, message', [
    ({'base_url': 'http://user:REDACTED_CREDENTIAL@127.0.0.1:9102'}, 'credential-free'),
    ({'base_url': 'ftp://127.0.0.1:9102'}, 'credential-free'),
    ({'base_url': 'http://127.0.0.1:9102/v1'}, 'without /v1'),
    ({'model': ' '}, 'model'),
    ({'cutoff': math.inf}, 'cutoff'),
    ({'cutoff': True}, 'cutoff'),
    ({'prompt_style': 'other'}, 'prompt style'),
    ({'timeout_ms': 0}, 'timeout'),
    ({'api_key_env': 'NOT_DECLARED'}, 'api_key_env'),
    ({'extra': 1}, 'declares'),
])
def test_a_rerank_config_is_checked_before_anything_runs(bad, message):
    with pytest.raises(ValueError, match=message):
        paired.validate_rerank({**RERANK, **bad}, {'DECLARED_KEY'}, embedding=EMBEDDING)
    assert paired.validate_rerank({**RERANK, 'api_key_env': 'DECLARED_KEY'}, {'DECLARED_KEY'},
                                  embedding=EMBEDDING)['api_key_env'] == 'DECLARED_KEY'


def test_a_declared_fingerprint_that_the_configuration_does_not_give_is_refused():
    """The cutoff was calibrated under another configuration: served, it would be switched off without a word."""
    with pytest.raises(ValueError, match='fingerprint'):
        paired.validate_rerank({**RERANK, 'fingerprint': 'f' * 64}, set(), embedding=EMBEDDING)
    good = fingerprint()
    assert paired.validate_rerank({**RERANK, 'fingerprint': good}, set(), embedding=EMBEDDING)['fingerprint'] == good


def test_a_rerank_config_without_the_fingerprint_its_calibration_recorded_is_refused():
    """Only the calibration knows what its cutoff was calibrated under: a plan that computed the fingerprint from the
    configuration it is given would label any cutoff verified, whatever changed since it was calibrated."""
    with pytest.raises(ValueError, match='fingerprint'):
        paired.validate_rerank(dict(CONFIGURED), set(), embedding=EMBEDDING)


def test_a_cutoff_reused_after_the_candidate_format_changed_is_refused(monkeypatch):
    """The fingerprint covers the candidate format: a cutoff calibrated on the old one needs a recalibration."""
    from colonymind.memory import recall
    monkeypatch.setattr(recall, 'CANDIDATE_FORMAT', recall.CANDIDATE_FORMAT + '-changed')
    with pytest.raises(ValueError, match='fingerprint'):
        paired.validate_rerank(dict(RERANK), set(), embedding=EMBEDDING)


def test_the_plan_records_the_reranker_and_its_fingerprint_for_every_arm(fixture):
    arms = ['base_hermes', 'all-on', 'all-on-semantic_recall']
    fixture.output.mkdir(mode=0o700)
    without = paired.plan(fixture.output / 'without', native_binding='candidate', evidence_mode='controlled',
                          arms=arms, reference_arm='all-on', embedding=dict(EMBEDDING), **fixture.resources)
    manifest = paired.plan(fixture.output / 'with', native_binding='candidate', evidence_mode='controlled',
                           arms=arms, reference_arm='all-on', embedding=dict(EMBEDDING), rerank=dict(RERANK),
                           **fixture.resources)
    recorded = manifest['comparison']['reranker']
    assert recorded == {**RERANK, 'timeout_ms': 1200}
    assert manifest['options']['rerank'] == recorded and 'reranker' not in without['comparison']
    assert manifest['comparison_key'] != without['comparison_key']
    for pair in manifest['pairs']:
        for arm in arms:
            assert pair['arms'][arm]['case']['inputs']['reranker'] == recorded


def test_a_rerank_config_needs_an_image_whose_worker_serves_it(fixture, monkeypatch):
    from colonymind.qualification import paired_container
    original = paired_container.configuration

    def older(*args, **kwargs):
        supplied, recipe = original(*args, **kwargs)
        recipe['container_payload'] = {k: v for k, v in recipe['container_payload'].items() if k != 'rerank'}
        return supplied, recipe
    monkeypatch.setattr(paired_container, 'configuration', older)
    with pytest.raises(ValueError, match='reranker'):
        paired.plan(fixture.output, native_binding='candidate', evidence_mode='controlled',
                    arms=['base_hermes', 'all-on'], rerank=dict(RERANK), **fixture.resources)
    assert paired_worker.inspect_payload.__code__.co_names.count('RERANK_PROTOCOL') == 1


@pytest.mark.parametrize('arm', ['all-on', 'all-on-semantic_recall', 'colonymind'])
def test_every_plugin_arm_serves_the_reranker_with_the_cutoff_verified(tmp_path, monkeypatch, arm):
    """The served host computes the plan's fingerprint in every plugin arm, the one without semantic recall
    included (an install exports the embedding model whether or not the faculty is on), so the cutoff applies."""
    for name in list(os.environ):
        if name.startswith(('COLONYMIND_RERANK', 'COLONYMIND_RECALL_RERANK', 'COLONYMIND_EMBED')):
            monkeypatch.delenv(name)
    rerank = {**RERANK, 'timeout_ms': 1200}
    section = worker.mind_section(paired_worker.mind_switches(paired.PROFILES[arm]))
    inputs = {'embedding': EMBEDDING, 'reranker': rerank}
    for key, value in {**worker.embedding_environment(inputs, section), **worker.rerank_environment(inputs)}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(host, '_reranker', None)

    async def run():
        async with paired_worker.served_reranker() as opened:
            provider = host._reranker
            return opened, provider is not None and calibration_fingerprint(provider_calibration_metadata(provider))
    opened, served = asyncio.run(run())
    assert served == rerank['fingerprint'] and opened == {'opened': True, 'calibration': 'verified'}
    assert host._reranker is None                                         # withdrawn when the host stops
    assert os.environ['COLONYMIND_RECALL_RERANK_MIN_SCORE'] == '0.62'


def test_without_a_reranker_in_the_plan_nothing_opens(monkeypatch):
    for name in list(os.environ):
        if name.startswith(('COLONYMIND_RERANK', 'COLONYMIND_RECALL_RERANK')):
            monkeypatch.delenv(name)
    assert worker.rerank_environment({}) == {}
    monkeypatch.setattr(host, '_reranker', None)

    async def run():
        async with paired_worker.served_reranker() as opened:
            return opened, host._reranker
    assert asyncio.run(run()) == ({'opened': False}, None)


def test_the_limitations_say_whether_the_arm_reranks():
    assert paired_worker.recall_limitation({}) == 'no embedding/reranking'
    assert paired_worker.recall_limitation({'embedding': EMBEDDING}) == 'no reranking'
    assert paired_worker.recall_limitation({'embedding': EMBEDDING, 'reranker': RERANK}) is None
    assert paired_worker.recall_limitation({'reranker': RERANK}) == 'no embedding'
