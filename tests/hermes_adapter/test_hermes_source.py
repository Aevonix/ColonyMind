"""Source equality must identify the whole installed Hermes distribution."""
import base64
import hashlib
import json
from types import SimpleNamespace

import pytest

import hermes_source
from hermes_source import hermes_record


@pytest.fixture
def sources(tmp_path, monkeypatch):
    checkout = tmp_path / "reviewed source"
    installed = tmp_path / "installed"
    for root in (checkout, installed):
        (root / "agent").mkdir(parents=True)
        (root / "agent/turn.py").write_text("REVIEWED = True\n")
        (root / "agent/other.py").write_text("OTHER = True\n")
        (root / "agent/prompt.txt").write_text("Reviewed prompt\n")
    monkeypatch.setattr(hermes_source, "HERMES_ROOT", checkout)
    module = SimpleNamespace(__name__="agent.turn", __file__=str(installed / "agent/turn.py"))
    return checkout, installed, module


def distribution(monkeypatch, installed, *, origin=None, record=None):
    texts = {"direct_url.json": json.dumps(origin) if origin is not None else None, "RECORD": record}
    dist = SimpleNamespace(read_text=texts.get, locate_file=lambda name: installed / name)
    monkeypatch.setattr(hermes_source.metadata, "distributions", lambda **kwargs: [dist])
    return dist


def wheel_record(root):
    rows = []
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).rstrip(b"=").decode()
        rows.append(f"{path.relative_to(root).as_posix()},sha256={digest},{path.stat().st_size}")
    return "\n".join(rows)


def test_editable_distribution_must_point_at_reviewed_checkout(sources, monkeypatch):
    checkout, installed, module = sources
    distribution(monkeypatch, installed, origin={"url": checkout.as_uri(), "dir_info": {"editable": True}})
    hermes_source.assert_source_module(module)
    # The imported module's bytes still match, but the distribution comes from another tree.
    distribution(monkeypatch, installed, origin={"url": installed.as_uri(), "dir_info": {"editable": True}})
    with pytest.raises(AssertionError, match="does not belong"):
        hermes_source.assert_source_module(module)


def test_editable_identity_keeps_the_module_byte_check(sources, monkeypatch):
    checkout, installed, module = sources
    distribution(monkeypatch, installed, origin={"url": checkout.as_uri(), "dir_info": {"editable": True}})
    (installed / "agent/turn.py").write_text("REVIEWED = False\n")
    with pytest.raises(AssertionError):
        hermes_source.assert_source_module(module)


def test_wheel_record_identifies_source_beyond_the_imported_module(sources, monkeypatch):
    checkout, installed, module = sources
    dist = distribution(monkeypatch, installed, record=wheel_record(installed))
    hermes_source.assert_source_module(module)
    (checkout / "agent/other.py").write_text("OTHER = False\n")
    assert hermes_record.distribution_source_problems(dist, checkout) == ["modified: agent/other.py"]
    with pytest.raises(AssertionError, match="modified: agent/other.py"):
        hermes_source.assert_source_module(module)


@pytest.mark.parametrize("tree", ["checkout", "installed"])
@pytest.mark.parametrize("change", ["missing", "modified"])
@pytest.mark.parametrize("relative", ["agent/other.py", "agent/prompt.txt"])
def test_wheel_rejects_missing_or_modified_files_in_either_tree(sources, monkeypatch, tree, change, relative):
    checkout, installed, module = sources
    distribution(monkeypatch, installed, record=wheel_record(installed))
    path = (checkout if tree == "checkout" else installed) / relative
    if change == "missing":
        path.unlink()
    else:
        path.write_text("OTHER = False\n")
    with pytest.raises(AssertionError, match=f"{change}: {relative}"):
        hermes_source.assert_source_module(module)


@pytest.mark.parametrize("record", [None, "", "agent/turn.py,,", "agent/turn.py,md5=abc,10",
                                  "hermes_agent.dist-info/METADATA,sha256=abc,10"])
def test_distribution_needs_nonempty_source_record_evidence(sources, monkeypatch, record):
    _, installed, module = sources
    distribution(monkeypatch, installed, record=record)
    with pytest.raises(AssertionError, match="RECORD"):
        hermes_source.assert_source_module(module)


def test_record_checker_still_checks_metadata_and_scripts_for_installed_integrity(sources):
    _, installed, _ = sources
    record = wheel_record(installed) + "\nhermes_agent.dist-info/METADATA,sha256=abc,10\n../bin/hermes,sha256=abc,10"
    checked, changed = hermes_record.record_mismatches(record, installed)
    assert checked == 3
    assert changed == ["missing: hermes_agent.dist-info/METADATA", "missing: ../bin/hermes"]


@pytest.mark.parametrize("pin,family", [("S", "S"), ("S217", "S217"), ("S22", "S")])
def test_reviewed_source_keeps_its_native_contract_family(monkeypatch, pin, family):
    monkeypatch.setattr(hermes_source, "source_identity", lambda _: hermes_source.PINS[pin])
    assert hermes_source.assert_supported_source() == family


def test_a_matching_commit_with_another_tree_is_rejected(monkeypatch):
    commit, _ = hermes_source.PINS["S22"]
    monkeypatch.setattr(hermes_source, "source_identity", lambda _: (commit, "0" * 40))
    with pytest.raises(AssertionError, match="compatibility review"):
        hermes_source.assert_supported_source()


@pytest.mark.parametrize("origin_matches", [True, False])
def test_checkout_egg_info_cannot_shadow_installed_provenance(sources, monkeypatch, origin_matches):
    checkout, installed, module = sources
    egg = checkout / "hermes_agent.egg-info"
    egg.mkdir()
    (egg / "PKG-INFO").write_text("Name: hermes-agent\nVersion: 0.21.5\n")
    info = installed / "hermes_agent-0.21.5.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: hermes-agent\nVersion: 0.21.5\n")
    origin = checkout if origin_matches else installed
    (info / "direct_url.json").write_text(json.dumps({"url": origin.as_uri(), "dir_info": {"editable": True}}))
    monkeypatch.setattr(hermes_source.sys, "path", [str(checkout), str(installed)])
    assert hermes_source.metadata.distribution("hermes-agent").read_text("RECORD") is None
    if origin_matches:
        hermes_source.assert_source_module(module)
    else:
        with pytest.raises(AssertionError, match="does not belong"):
            hermes_source.assert_source_module(module)


def test_checkout_egg_info_without_an_installed_distribution_is_refused(sources, monkeypatch):
    checkout, _, module = sources
    egg = checkout / "hermes_agent.egg-info"
    egg.mkdir()
    (egg / "PKG-INFO").write_text("Name: hermes-agent\nVersion: 0.21.5\n")
    monkeypatch.setattr(hermes_source.sys, "path", [str(checkout)])
    with pytest.raises(AssertionError, match="found 0"):
        hermes_source.assert_source_module(module)


def test_ambiguous_installed_distributions_are_refused(sources, monkeypatch):
    checkout, installed, module = sources
    dist = distribution(monkeypatch, installed, origin={"url": checkout.as_uri(), "dir_info": {"editable": True}})
    monkeypatch.setattr(hermes_source.metadata, "distributions", lambda **kwargs: [dist, dist])
    with pytest.raises(AssertionError, match="found 2"):
        hermes_source.assert_source_module(module)
