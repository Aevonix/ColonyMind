"""Render Release A and configured host surfaces, then compare their UTF-8 bytes."""
import importlib
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

import pytest

from host_policy_fixtures import HOST_SETTINGS, host_config
from colonymind import config
from colonymind.hands import brief, notices as hand_notices
from colonymind.memory import notice, recall
from colonymind.mind import drift, status
from test_hands_controller import c, running
from test_mind_drift import world
from test_recall_notice_details import record


ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).parent / "fixtures/compatibility_sources"


@pytest.fixture
def release_a(monkeypatch):
    """Use frozen renderer source, without a checkout or writes."""
    loaded = {}

    def load(path, package):
        name = package + "._release_a_" + Path(path).stem
        source = (BASELINE / path).read_text()
        module = ModuleType(name)
        module.__file__ = str(ROOT / path)
        module.__package__ = package
        monkeypatch.setitem(sys.modules, name, module)
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        return module

    for name, path, package in (
        ("config", "sidecar/colonymind/config.py", "colonymind"),
        ("drift", "sidecar/colonymind/mind/drift.py", "colonymind.mind"),
        ("tick", "sidecar/colonymind/mind/tick.py", "colonymind.mind"),
        ("brief", "sidecar/colonymind/hands/brief.py", "colonymind.hands"),
        ("status", "sidecar/colonymind/mind/status.py", "colonymind.mind"),
        ("notice", "sidecar/colonymind/memory/notice.py", "colonymind.memory"),
        ("hand_notices", "sidecar/colonymind/hands/notices.py", "colonymind.hands"),
    ):
        loaded[name] = load(path, package)
    # Adapter modules use relative imports; isolate them from live registration.
    packages = []
    for name, directory, module, key in (
        ("host_identity_parity_adapter", "plugins/colonymind-hands", "notices", "adapter"),
        ("host_identity_parity_hermes", "plugins/hermes-plugin", "client", "hermes"),
    ):
        package = ModuleType(name)
        package.__path__ = [str(ROOT / directory)]
        monkeypatch.setitem(sys.modules, name, package)
        loaded[key] = importlib.import_module(name + "." + module)
        loaded["old_" + key] = load(directory + "/" + module + ".py", name)
        packages.append(name)
    yield SimpleNamespace(**loaded)
    for key in list(sys.modules):
        if any(key.startswith(name + ".") for name in packages):
            del sys.modules[key]


def same_bytes(current, previous):
    assert current.encode("utf-8") == previous.encode("utf-8")


def test_fixture_holds_exact_moved_defaults(host_config, release_a):
    assert set(HOST_SETTINGS) == {"mind", "context", "hands"}
    assert set(HOST_SETTINGS["mind"]) == {"drift", "status_replacements"}
    assert set(HOST_SETTINGS["mind"]["drift"]) == {"validators"}
    assert set(HOST_SETTINGS["context"]) == {"recall_notice_replacements"}
    assert set(HOST_SETTINGS["hands"]) == {"notices"}
    assert set(HOST_SETTINGS["hands"]["notices"]) == {"forbidden_patterns"}
    old = release_a.drift
    assert host_config.get("mind.drift.validators") == {
        key: [getattr(old, attr).pattern] for key, attr in (
            ("hedge", "_HEDGE"), ("agency", "_AGENCY"), ("causal", "_CAUSAL"),
            ("apology", "_APOLOGY"), ("adjustment", "_ADJUST"), ("tone", "_CLINICAL"))
    }
    assert host_config.get("mind.status_replacements") == [
        {"pattern": r"\s+[\u2010-\u2015\u2e3a\u2e3b\ufe58\ufe63\uff0d]+\s+", "replacement": ", "},
        {"pattern": r"[\u2010-\u2015\u2e3a\u2e3b\ufe58\ufe63\uff0d]", "replacement": "-"},
    ]
    # Release A wrote literal glyphs; the host fixture uses equivalent Unicode escapes.
    assert [re.sub(r"\\u([0-9a-f]{4})", lambda m: chr(int(m[1], 16)), item["pattern"])
            for item in host_config.get("mind.status_replacements")] == [
                release_a.status._SPACED_DASH.pattern, release_a.status._ANY_DASH.pattern]
    assert host_config.get("context.recall_notice_replacements") == [{"pattern": r"[\u2013\u2014]", "replacement": "-"}]
    assert host_config.get("hands.notices.forbidden_patterns") == [r"[\u2013\u2014]"]


def test_identity_and_task_prompt_bytes(host_config, release_a, world):
    world.mind.config.update(host_config.get("mind"))
    world.identity = {"agent": {"name": "Fixture agent", "values": ["Check — sources", "Keep spacing"],
                                "boundaries": ["Ask before publishing"]}, "owner": {"name": "Fixture owner"}}
    for limit in (None, 40, config.CONSTITUTION_CHARS):
        same_bytes(config.render_constitution(world.identity, limit=limit),
                   release_a.config.render_constitution(world.identity, limit=limit))
        same_bytes(release_a.hermes.render_constitution(world.identity, limit=limit),
                   release_a.old_hermes.render_constitution(world.identity, limit=limit))
    same_bytes(world.mind._self_context(), release_a.tick.Mind._self_context(world.mind))
    same_bytes(world.mind.section(), release_a.tick.Mind.section(world.mind))
    for promises, working in ((None, None), ([], []), (["Check — record"], ["Compare–results"])):
        kwargs = {"promises": promises, "working": working, "stranger": "Fictional Stranger"}
        current, previous = drift.Record(**kwargs), release_a.drift.Record(**kwargs)
        same_bytes(drift.record_block(current), release_a.drift.record_block(previous))
        same_bytes(drift.battery_prompt(current), release_a.drift.battery_prompt(previous))
        same_bytes("\n\n".join((world.mind._self_context(), drift.record_block(current))),
                   "\n\n".join((release_a.tick.Mind._self_context(world.mind), release_a.drift.record_block(previous))))
    # A5's host identity/style are unchanged by B8, including supplied punctuation.
    for parent_is_owner in (False, True):
        for persona, style in (("", ""), ("You are {name}.  Inspect — evidence.", "Keep–style.\n")):
            kwargs = dict(goal="Check — artifact", context="Compare–source", intent_version=2,
                          acceptance={"evidence": "receipt"}, name="Fixture hand", parent_is_owner=parent_is_owner,
                          persona_addendum=persona, style=style, budgets={"calls": 3},
                          recall=lambda **_: [{"reference": "source:fixture", "text": "Original — evidence"}],
                          previous=[{"branch": "branch:fixture", "summary": "Earlier–result", "level": "E1"}])
            same_bytes(brief.build_brief(**kwargs).text, release_a.brief.build_brief(**kwargs).text)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("level", list(status.LEVEL_MEANINGS))
async def test_status_block_bytes(host_config, release_a, world, enabled, level):
    world.mind.config.update(host_config.get("mind"))
    world.mind.authority.set_enabled(enabled)
    world.mind.policy.level = level
    for reason in ("", "pause — pending–source", "\u2010\u2011\u2012\u2015\u2e3a\u2e3b\ufe58\ufe63\uff0d", "long — " * 20):
        world.mind.off_reason = reason
        for limit in (200, status.MAX_CHARS):
            problems = {name: lambda: True for name in status.PROBLEMS}
            same_bytes(status.build(world.mind, problems=problems, limit=limit),
                       release_a.status.build(world.mind, problems=problems, limit=limit))
        same_bytes(await status.collect(world.mind), await release_a.status.collect(world.mind))


@pytest.mark.parametrize("text", ["Plain record", "café — suite–notes", "MEDIA:fixture [[as_document]] — source",
                                 "long–record " * 25])
def test_recall_notice_bytes(host_config, release_a, text):
    rows = [record(text), record("Other person's — record", id="other", contact_id="contact-b")]
    content = recall.render_memory_context(rows)
    # Read the staged host configuration through the renderer's normal loading path.
    current = notice.recall_notice_details(rows, content=content, contact_id="contact-a")
    previous = release_a.notice.recall_notice_details(rows, content=content, contact_id="contact-a")
    assert [line.encode() for line in current] == [line.encode() for line in previous]


def test_adapter_hand_notice_bytes_and_rejections(host_config, release_a):
    rows = [{"line": line} for line in ("Progress: café source", "Completed the fixture check.")]
    same_bytes(release_a.adapter.compose(rows, forbidden_patterns=host_config.get("hands.notices.forbidden_patterns")),
               release_a.old_adapter.compose(rows))
    for line in ("Progress — record", "Progress–record", "x" * 201, "", "one\ntwo"):
        with pytest.raises(ValueError):
            release_a.adapter.compose([{"line": line}], forbidden_patterns=host_config.get("hands.notices.forbidden_patterns"))
        with pytest.raises(ValueError):
            release_a.old_adapter.compose([{"line": line}])


async def test_hand_notice_bytes_and_rejections(host_config, release_a, c):
    c.config["notices"].update(host_config.get("hands.notices"))
    plan = await running(c)
    current, previous = [], []
    for index, line in enumerate(("Progress: café source", "Completed the fixture check.")):
        current.append(await hand_notices.NoticeOperations._notice(c, f"ntc:new-{index}", plan["hand_id"], "routine", line))
        previous.append(await release_a.hand_notices.NoticeOperations._notice(c, f"ntc:old-{index}", plan["hand_id"], "routine", line))
    same_bytes(release_a.adapter.compose(current, forbidden_patterns=host_config.get("hands.notices.forbidden_patterns")),
               release_a.old_adapter.compose(previous))
    # The private renderers above insert directly. Finish their transaction
    # before exercising the controller's public transactional notice method.
    await c.store.db.commit()
    # Release A's renderers reject dashes and overlong lines; empty and multiline notices are new rejections only.
    for line in ("Progress — record", "Progress–record", "x" * 201, "", "one\ntwo"):
        with pytest.raises(ValueError):
            await c.notice("ntc:invalid-new", plan["hand_id"], "routine", line)
        with pytest.raises(ValueError):
            release_a.adapter.compose([{"line": line}], forbidden_patterns=host_config.get("hands.notices.forbidden_patterns"))
        if line and "\n" not in line:
            with pytest.raises(ValueError):
                await release_a.hand_notices.NoticeOperations._notice(c, "ntc:invalid-old", plan["hand_id"], "routine", line)
            with pytest.raises(ValueError):
                release_a.old_adapter.compose([{"line": line}])
