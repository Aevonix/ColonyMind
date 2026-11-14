"""The cutoff survives identity, but no learned adapter. Fixture widths and endpoint port are test assumptions."""
import pytest

from colonymind.memory.recall import calibration_fingerprint

METADATA = {'provider': 'fixture', 'model': 'ranker', 'endpoint': 'http://127.0.0.1:9000',
            'embedding_model': 'embedder', 'embedding_dimensions': 2, 'index_generation': 'prior'}
GOLDEN = 'f87f32482acd8133035667ee2ffc71c432415be7407d72a4b04eab61bc1cdef6'


def test_identity_transform_preserves_the_previous_cutoff_golden():
    assert calibration_fingerprint(METADATA) == GOLDEN
    assert calibration_fingerprint({**METADATA, 'query_transform': 'identity'}) == GOLDEN


def test_each_adapter_revision_invalidates_the_cutoff_and_does_not_mutate_metadata():
    first = {**METADATA, 'query_transform': 'query-adapter-1:' + 'a' * 64}
    second = {**METADATA, 'query_transform': 'query-adapter-1:' + 'b' * 64}
    assert len({calibration_fingerprint(first), calibration_fingerprint(second), GOLDEN}) == 3
    assert first['query_transform'].endswith('a' * 64)


@pytest.mark.parametrize('transform', ['', None, 'unknown', 'query-adapter-1:identity'])
def test_invalid_transform_identity_is_not_silently_accepted(transform):
    with pytest.raises(ValueError, match='query_transform'):
        calibration_fingerprint({**METADATA, 'query_transform': transform})
