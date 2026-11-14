"""YAML margin diagnostics and aggregate certificate retention after erasure."""
import copy
from types import SimpleNamespace

import pytest
import yaml

from colonymind.config import load_config
from colonymind.instinct.outcomes import Sources
from colonymind.plasticity.certificates import GateRecords
from colonymind.plasticity.learners.reply_timing import DECLARATION
from colonymind.plasticity.nightly import Runtime
from test_plasticity_reply_timing import NOW
from test_plasticity_reply_timing_certificates import gate_case


@pytest.mark.parametrize('scalar,accepted', [('5e-3', False), ('0.005', True), ('5.0e-3', True)])
def test_yaml_margin_reports_string_value_and_accepts_numeric_spellings(tmp_path, scalar, accepted):
    spec = copy.deepcopy(DECLARATION)
    del spec['retention_margin']
    text = yaml.safe_dump({'plasticity': {'learners': {'timing': spec}}})
    # Insert the exact unquoted spelling users write, bypassing safe_dump's quoting.
    text = text.replace('      implementation: reply_timing\n',
                        f'      implementation: reply_timing\n      retention_margin: {scalar}\n')
    (tmp_path / 'colonymind.yaml').write_text(text)
    config = load_config(tmp_path, environ={})
    value = config.data['plasticity']['learners']['timing']['retention_margin']
    assert type(value) is (float if accepted else str)
    service = Runtime(config, clock=lambda: NOW,
                      sources={'timing': SimpleNamespace(sources=Sources(tmp_path, owner_id='Owner'))})
    try:
        if accepted:
            assert service.loaded['timing'].driver.retention_margin == .005
        else:
            assert 'timing' not in service.loaded
            assert 'retention_margin must be non-negative and finite' in service.rejected['timing']
            assert "received '5e-3' (type str)" in service.rejected['timing']
    finally:
        service.close()


def test_unresolved_source_erasure_keeps_all_certificate_statistics_without_digests(gate_case, tmp_path):
    driver, closure, build, _ = gate_case(.04, .1)
    driver.gate(closure, build)
    driver.gate(closure, build)
    original = driver.store.certificates.records()
    assert len(original) == 2 and all(record['gate_set'] for record in original)
    assert all(record['comparisons']['serving_candidate']['lower'] is not None for record in original)

    config = load_config(tmp_path, environ={})
    config.data['plasticity']['learners'] = {}
    service = Runtime(config, clock=lambda: NOW)
    try:
        assert not service.loaded
        # No saved forward provenance or loaded resolver can map this source.
        service.erase_sources(['unresolved-source'])
        assert driver.store.rebuild_required.exists()
        expected = [{**record, 'gate_set': None} for record in original]
        assert driver.store.certificates.records() == expected
        # Retrying the conservative reset preserves the same complete evidence.
        service.erase_sources(['unresolved-source'])
        assert driver.store.certificates.records() == expected
    finally:
        service.close()
    assert GateRecords(driver.store.certificates.directory).records() == expected
