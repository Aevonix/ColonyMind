"""The host controls notice language independently of transport bounds."""
import pytest

from test_hands_surfaces import modules, runtime, notice


def test_composer_defaults_preserve_host_punctuation(modules):
    line = "Progress — source–notes"
    assert modules.notices.compose([{"line": line}]).encode() == line.encode()
    with pytest.raises(ValueError):
        modules.notices.compose([{"line": line}], forbidden_patterns=[r"[\u2013\u2014]"])
    assert modules.notices.compose([{"line": "Plain update"}], forbidden_patterns=[r"[\u2013\u2014]"]) == "Plain update"


@pytest.mark.parametrize("line", ["", "x" * 201, "one\ntwo", "one\rtwo"])
def test_transport_validation_always_applies(modules, line):
    with pytest.raises(ValueError):
        modules.notices.compose([{"line": line}])


def test_emitter_reads_host_notice_settings(modules, runtime, tmp_path):
    delivered = []
    policy = {"forbidden_patterns": [r"[\u2013\u2014]"]}
    emitter = modules.notices.NoticeEmitter(
        client=runtime.client, emit=runtime.emit, sink=lambda **kw: delivered.append(kw) or {"platform_message_id": "receipt"},
        ledger_path=tmp_path / "notices.db", now=lambda: 71.0, settings=lambda: policy,
    )
    line = "Progress — source–notes"
    runtime.client.pending = [{**notice(), "line": line}]
    emitter.pump()
    assert delivered == []
    policy["forbidden_patterns"] = []
    emitter.pump()
    assert delivered[0]["notices"][0]["line"].encode() == line.encode()
