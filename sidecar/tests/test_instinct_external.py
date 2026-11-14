"""External decision points (``colonymind.instinct.external``): the questions the Hermes gates ask.

Two points ship (``inbound_triage`` and ``tool_screen``, with the wording the bake-off measures), and an operator
adds their own under ``instinct.external`` in ``colonymind.yaml``: a yes/no or a choice of at most 8 labels, a
template over named fields, effect ``route`` or ``filter`` only, and the fast and reader tiers only, never the
deliberate tier, so an external point never spends the mind's model. A ``kind: cron`` point is the cron wake gate's
question for one job. Every tier ships in shadow; ``inbound_triage`` never acts in this release.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from colonymind.instinct import REGISTRY, external
from colonymind.instinct.registry import register
from colonymind.instinct.tiers import Instinct

BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "decisions"


def _questions():
    sys.path.insert(0, str(BENCH))
    try:
        spec = importlib.util.spec_from_file_location("decisions_questions_external", BENCH / "questions.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(BENCH))


CRON = {"kind": "cron", "question": "Does this text contain a reply from the supplier about the open order?",
        "watch": "order 4471 from the supplier"}


# -- the shipped points ---------------------------------------------------------------------------------------------

def test_the_shipped_points_carry_the_wording_the_bake_off_measures():
    """A calibration record is keyed by the point's version (a hash of what a model reads): the shipped wording is
    the bake-off's, so a record fitted on its sets matches the point that ships."""
    measured = _questions()
    table = external.build({})
    assert set(table) == {"inbound_triage", "tool_screen"}
    assert table["inbound_triage"].point.version == measured.INBOUND_TRIAGE.version
    assert table["tool_screen"].point.version == measured.TOOL_SCREEN.version
    cron = external.build({"cron_supplier_reply": CRON})["cron_supplier_reply"].point
    assert cron.version == measured.cron_point("supplier_reply").version


def test_every_external_tier_ships_in_shadow_and_none_is_deliberate():
    table = external.build({"cron_supplier_reply": CRON})
    for name, entry in table.items():
        assert entry.point.effect in ("route", "filter"), name
        assert {tier.name for tier in entry.point.tiers} <= {"fast", "reader"}, name
        assert all(tier.state == "shadow" for tier in entry.point.tiers), name
        register(entry.point, registry={})          # the registry's own checks pass


def test_the_gates_deadlines_are_the_plans_choices():
    table = external.build({})
    assert {tier.deadline_ms for tier in table["inbound_triage"].point.tiers} == {150}
    assert {tier.deadline_ms for tier in table["tool_screen"].point.tiers} == {400}


def test_no_external_point_is_a_mind_point():
    table = external.build({"cron_supplier_reply": CRON})
    assert not set(table) & set(REGISTRY)
    assert "inbound_triage" not in REGISTRY and "tool_screen" not in REGISTRY


# -- operator points ------------------------------------------------------------------------------------------------

def test_an_operator_point_is_a_question_over_named_fields():
    table = external.build({"voice_addressed": {
        "question": "Is the speaker talking to the assistant?",
        "template": 'Partial transcript: "{text}"', "effect": "route"}})
    point = table["voice_addressed"].point
    assert point.type == "yes_no" and point.effect == "route" and point.fields == ("text",)
    # Not claimed literal: nothing measured says a fast decision model reads it as well as a fact the text states.
    assert not point.literal
    assert [tier.name for tier in point.tiers] == ["fast", "reader"]


@pytest.mark.parametrize("entry, words", [
    ({"question": "Is it?", "effect": "ask"}, "effect"),
    ({"question": "Is it?", "effect": "read"}, "effect"),
    ({"question": "Is it?", "effect": "rank"}, "effect"),
    ({"question": "Is it?", "tiers": ["fast", "deliberate"]}, "deliberate"),
    ({"question": "Is it?", "tiers": ["rules"]}, "tiers"),
    ({"question": "Is it?", "colour": "blue"}, "unknown key"),
    ({"question": ""}, "question"),
    ({"question": "Which?", "type": "choice",
      "labels": {f"l{index}": "a kind" for index in range(8)} | {"none": "neither"}}, "labels"),
    ({"question": "Which?", "type": "choice", "labels": {"a": "the first", "b": "the second"}}, "none"),
    ({"question": "Which?", "type": "choice", "labels": {"allow": "let it in", "none": "neither"}}, "allow"),
    ({"question": "Is it?", "template": 'Words: "{_unchanged}"'}, "template"),
    ({"question": "Is it?", "template": 'Words: "{text'}, "template"),
    ({"question": "Is it?", "bars": {"maybe": 0.9}}, "bars"),
    ({"kind": "cron", "question": "Is it?", "template": "{text}"}, "cron"),
    ({"kind": "chat", "question": "Is it?"}, "kind"),
], ids=["effect-ask", "effect-read", "effect-rank", "deliberate", "rules-tier", "unknown-key", "no-question",
        "nine-labels", "choice-without-none", "forbidden-label", "reserved-field", "bad-template", "bad-bar",
        "cron-with-template", "unknown-kind"])
def test_an_operator_point_outside_the_rules_is_refused_at_load(entry, words):
    with pytest.raises(external.ExternalError) as refused:
        external.validate({"my_point": entry})
    assert words in str(refused.value)
    assert "instinct.external.my_point" in str(refused.value)
    # The sidecar's lenient read leaves it out, so the route refuses it and the gate fails open.
    assert "my_point" not in external.build({"my_point": entry})


@pytest.mark.parametrize("name", ["owner_send", "inbound_triage", "tool_screen", "Bad-Name", "x" * 60])
def test_an_operator_point_may_not_take_a_mind_or_shipped_name(name):
    with pytest.raises(external.ExternalError):
        external.validate({name: {"question": "Is it?"}})


def test_a_cron_point_is_named_for_its_job_and_reads_its_watch_from_the_configuration():
    with pytest.raises(external.ExternalError, match="cron_"):
        external.validate({"supplier_reply": CRON})
    entry = external.build({"cron_supplier_reply": CRON})["cron_supplier_reply"]
    assert entry.kind == "cron" and entry.fixed == {"watch": "order 4471 from the supplier"}
    assert entry.point.effect == "route" and entry.point.literal


def test_colonymind_yaml_refuses_a_malformed_external_point_by_name(tmp_path):
    from colonymind.config import ConfigError, load_config
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump({"instinct": {"external": {
        "cron_supplier_reply": CRON, "my_point": {"question": "Is it?", "tiers": ["deliberate"]}}}}))
    with pytest.raises(ConfigError, match="instinct.external.my_point"):
        load_config(tmp_path, environ={})
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump({"instinct": {"external": {
        "cron_supplier_reply": CRON}}}))
    loaded = load_config(tmp_path, environ={})
    assert loaded.get("instinct")["external"]["cron_supplier_reply"]["kind"] == "cron"


# -- the engine the route asks --------------------------------------------------------------------------------------

def test_the_engine_is_built_from_the_instinct_section_and_logs_to_the_instances_database(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("COLONYMIND_INSTINCT", json.dumps({
        "providers": {"local": {"kind": "local-decision.v1", "url": "http://127.0.0.1:9", "timeout_ms": 250}},
        "fast": "local", "external": {"cron_supplier_reply": {**CRON, "bars": {"no": 0.95}}}}))
    monkeypatch.setattr(external, "_state_dir", lambda: tmp_path)
    engine, table = external.engine()
    assert isinstance(engine, Instinct)
    assert set(engine.registry) == set(table) == {"inbound_triage", "tool_screen", "cron_supplier_reply"}
    assert engine.fast == "local" and "local" in engine.providers
    assert engine.points_config["cron_supplier_reply"]["bars"] == {"no": 0.95}
    assert engine.store is not None and engine.store.path == tmp_path / "instinct.db"
    assert external.engine()[0] is engine            # one engine per configuration and state directory


def test_triage_never_goes_live_whatever_its_state_says(tmp_path):
    engine, table = external.build_engine({}, state_dir=tmp_path)
    assert external.gate_state(engine, "inbound_triage") == "shadow"
    engine.set_state("inbound_triage", "fast", "live", source="test")
    assert engine.state("inbound_triage", "fast") == "live"
    assert external.gate_state(engine, "inbound_triage") == "shadow"
    engine.set_state("tool_screen", "reader", "live", source="test")
    assert external.gate_state(engine, "tool_screen") == "live"
