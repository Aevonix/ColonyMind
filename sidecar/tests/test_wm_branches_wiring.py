"""Working memory's defaults, benchmark settings, faculty report and operator reset."""
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import json
import sys
from types import SimpleNamespace

import pytest

from colonymind.config import DEFAULTS, load_config
from colonymind.qualification import native_memory_worker, paired, paired_worker
from colonymind.reports import faculties
from colonymind.reports.registry import BY_KEY, FACULTIES


NOW = datetime(2026, 9, 20, 12, tzinfo=timezone.utc)


def _report():
    return {
        "enabled": True, "branches": 3, "spans": 8, "consults": 5,
        "refusals": {"hourly_budget": 2, "token_cap": 1},
        "withdrawn": True, "withdrawal_reason": "The live comparison crossed the decline bound",
        "comparator": {"labelled": 12, "losses": 9, "gains": 1},
        "cached_token_share": 0.75,
    }


def test_working_memory_ships_on_with_bounded_defaults(tmp_path):
    config = load_config(tmp_path, environ={})
    assert config.get("mind.faculties.working_memory") is True
    assert config.get("working_memory.retention_days") == 14
    assert config.get("working_memory.token_cap") == 48000
    assert config.get("working_memory.hourly_budget") == 20
    assert config.get("working_memory.summary_only") is False
    keys = [faculty.key for faculty in FACULTIES]
    assert keys[keys.index("plasticity") + 1] == "working_memory"
    assert BY_KEY["working_memory"].arm == "all-on-working_memory"


@pytest.mark.parametrize("reference", ["full", "all-on"])
def test_off_arm_writes_only_the_working_memory_faculty(reference):
    baseline = native_memory_worker.mind_section(paired_worker.mind_switches(paired.PROFILES[reference]))
    profile = paired.PROFILES[reference + "-working_memory"]
    section = native_memory_worker.mind_section(paired_worker.mind_switches(profile))
    wanted = deepcopy(baseline)
    wanted["faculties"]["working_memory"] = False
    assert section == wanted
    config = native_memory_worker.instance_config("owner", section, profile=profile)
    assert config["mind"]["faculties"]["working_memory"] is False


def test_summary_arm_keeps_capture_on_and_writes_summary_only():
    profile = paired.validate_profiles(None)["wm.summary_only"]
    section = native_memory_worker.mind_section(paired_worker.mind_switches(profile))
    assert section["faculties"]["working_memory"] is True
    config = native_memory_worker.instance_config("owner", section, profile=profile)
    assert config["working_memory"]["summary_only"] is True
    assert DEFAULTS["working_memory"]["summary_only"] is False


@pytest.mark.parametrize("arm, available", [
    ("base_hermes", False), ("colonymind", True), ("colonymind-initiative", False),
    ("all-on", True), ("all-on-working_memory", False), ("wm.summary_only", False),
])
def test_worker_toolset_follows_the_effective_working_memory_arm(arm, available):
    selected = paired_worker.plugin_tools("memory", profile=paired.PROFILES[arm])
    assert ("colonymind_consult" in selected) is available
    assert selected[:2] == paired_worker.PLUGIN_TOOL_SETS["memory"]


def test_report_includes_counts_refusals_comparison_and_cache_share():
    report = importlib.import_module("colonymind.reports.working_memory")
    collected = report.collect(None, runtime=SimpleNamespace(report=_report))
    text = report.document(collected).text()
    for phrase in ("3 branches", "8 spans", "5 consults", "hourly budget: 2", "token cap: 1",
                   "12 labelled", "9 losses", "1 gain", "75.0%", "withdrawn", "decline bound",
                   "colonymind wm reset"):
        assert phrase in text, phrase
    assert json.loads(report.to_json(collected)) == _report()


def test_report_dispatcher_reads_working_memory_with_the_mind_off(tmp_path, monkeypatch):
    (tmp_path / "colonymind.yaml").write_text("mind: {enabled: false}\n")
    report = importlib.import_module("colonymind.reports.working_memory")
    monkeypatch.setattr(report, "collect", lambda config: {**_report(), "withdrawn": False})
    collected = faculties.collect(tmp_path, now=NOW)
    section = next(section for section in collected.sections if section.faculty.key == "working_memory")
    assert section.on is True
    assert any("3 branches" in line and "8 spans" in line for line in section.lines)
    assert not section.wrong


def test_fresh_report_is_read_only(tmp_path):
    report = importlib.import_module("colonymind.reports.working_memory")
    config = load_config(tmp_path, environ={})
    before = set(tmp_path.rglob("*"))
    collected = report.collect(config)
    assert collected["branches"] == collected["spans"] == collected["consults"] == 0
    assert set(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("enabled, summary_only", [(True, False), (True, True), (False, False)])
@pytest.mark.parametrize("existing_ledger", [False, True])
def test_fresh_report_preserves_the_configured_state(tmp_path, enabled, summary_only, existing_ledger):
    from colonymind.reports import working_memory
    if existing_ledger:
        from colonymind.turns.idempotency import TurnIdempotencyLedger
        TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    (tmp_path / "colonymind.yaml").write_text(json.dumps({
        "mind": {"faculties": {"working_memory": enabled}},
        "working_memory": {"summary_only": summary_only},
    }))
    config = load_config(tmp_path, environ={})
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    state = working_memory.collect(config)
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert state["enabled"] is enabled
    assert state["summary_only"] is summary_only
    assert state["tool_available"] is (enabled and not summary_only)
    assert state["withdrawn"] is False and state["withdrawal_reason"] == ""
    assert state["cached_token_share"] is None
    assert state["comparator"]["labelled"] == 0
    assert state["comparator"]["lower_bound"] is None
    assert state["comparator"]["upper_bound"] is None
    assert state["comparator"]["reason"] == "awaiting paired owner labels"
    text = working_memory.document(state).text()
    assert "cached-token share is unavailable" in text
    if not enabled:
        assert "Working memory is switched off" in text
    elif summary_only:
        assert "Only summaries are available" in text


@pytest.mark.parametrize("arm, enabled, summary_only", [
    ("all-on", True, False), ("all-on-working_memory", False, False), ("wm.summary_only", True, True),
])
def test_workers_serialize_the_arm_for_the_real_branch_store(tmp_path, monkeypatch, arm, enabled, summary_only):
    from colonymind.memory.branches import BranchRefused, get_store
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", "owner")
    profile = paired.PROFILES[arm]
    section = native_memory_worker.mind_section(paired_worker.mind_switches(profile))
    values = native_memory_worker.instance_config("owner", section, profile=profile)
    (tmp_path / "colonymind.yaml").write_text(json.dumps(values))
    store = get_store(load_config(tmp_path, environ={}))
    assert store.enabled is enabled
    assert store.summary_only is summary_only
    assert store.tool_available is ("colonymind_consult" in paired_worker.plugin_tools("memory", profile=profile))
    capture = {"parent_session": "parent", "child_session": "child", "requester": "owner",
               "brief": "Read the sample", "messages": [{"role": "assistant", "content": "The sample is blue."}]}
    if not enabled:
        with pytest.raises(BranchRefused, match="disabled"):
            store.capture(**capture)
        assert store.report()["branches"] == 0
    else:
        branch = store.capture(**capture)
        assert len(branch["spans"]) == 2
        if summary_only:
            with pytest.raises(BranchRefused, match="summary_only"):
                asyncio.run(store.consult([branch["id"]], "What color?", requester="owner",
                                          session_id="parent", router=None))


def test_persisted_regression_report_and_operator_reset_use_the_same_store(tmp_path, monkeypatch, capsys):
    from colonymind import cli
    from colonymind.memory.branches import BranchRefused, get_store
    from colonymind.reports import working_memory
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", "owner")
    for key in ("COLONYMIND_HOME", "COLONYMIND_STATE_DIR", "COLONYMIND_INSTANCE_SELECTED"):
        monkeypatch.setenv(key, "")
    (tmp_path / "colonymind.yaml").write_text("owner: {contact_id: owner}\n")
    config = load_config(tmp_path, environ={})
    store = get_store(config)
    branch = store.capture(parent_session="parent", child_session="child", requester="owner",
                           brief="Read the sample", messages=[{"role": "tool", "content": "The sample is blue."}],
                           summary="The sample has a color.")
    for index in range(20):
        ref = f"pair-{index}"
        store.record_pair(ref, session_id="parent", question="What color?", answer="Red", reference_answer="Blue")
        store.outcome(ref, served_correct=False, reference_correct=True)
    with pytest.raises(BranchRefused, match="withdrawn"):
        asyncio.run(store.consult([branch["id"]], "What color?", requester="owner", session_id="parent", router=None))
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = working_memory.collect(config)
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert result["branches"] == 1 and result["spans"] == 3 and result["consults"] == 20
    assert result["withdrawn"] is True and result["tool_available"] is False
    assert result["refusals"] == {"withdrawn": 1}
    assert result["comparator"]["labelled"] == result["comparator"]["losses"] == 20
    assert result["comparator"]["gains"] == 0
    assert result["comparator"]["lower_bound"] > result["comparator"]["bar"]
    section = next(item for item in faculties.collect(tmp_path, now=NOW).sections
                   if item.faculty.key == "working_memory")
    assert any(result["withdrawal_reason"] in line for line in section.lines)
    monkeypatch.setattr(sys, "argv", ["colonymind", "--instance", str(tmp_path), "wm", "reset"])
    with pytest.raises(SystemExit) as done:
        cli.main()
    assert done.value.code == 0
    assert "withdrawal cleared" in capsys.readouterr().out
    cleared = working_memory.collect(config)
    assert cleared["withdrawn"] is False and cleared["withdrawal_reason"] == ""
    assert cleared["tool_available"] is True
    assert cleared["comparator"]["labelled"] == cleared["comparator"]["losses"] == 0
    assert cleared["branches"] == 1 and cleared["spans"] == 3 and cleared["consults"] == 20


def test_reset_command_clears_the_selected_store(tmp_path, monkeypatch, capsys):
    from colonymind import cli
    resets, selected = [], []
    store = SimpleNamespace(reset=lambda: resets.append(True))

    def get_store(config):
        selected.append(config.home)
        return store

    monkeypatch.setitem(sys.modules, "colonymind.memory.branches", SimpleNamespace(get_store=get_store))
    monkeypatch.setattr(sys, "argv", ["colonymind", "--instance", str(tmp_path), "wm", "reset"])
    for key in ("COLONYMIND_HOME", "COLONYMIND_STATE_DIR", "COLONYMIND_INSTANCE_SELECTED"):
        monkeypatch.setenv(key, "")
    with pytest.raises(SystemExit) as done:
        cli.main()
    assert done.value.code == 0
    assert selected == [tmp_path]
    assert resets == [True]
    assert "withdrawal cleared" in capsys.readouterr().out


@pytest.mark.parametrize("reverse", [False, True])
def test_faculty_benchmark_reports_summary_arm_as_a_mechanism_comparison(tmp_path, reverse):
    from colonymind.reports.bench import Loaded
    names = ["all-on", "wm.summary_only"]
    if reverse:
        names.reverse()
    report = {"profiles": {name: deepcopy(paired.PROFILES[name]) for name in names},
              "dataset": {"version": "working-memory-1", "split": "development"},
              "statistics": {"contrasts": [{
                  "treatment": names[0], "comparator": names[1], "verdict": "demonstrated",
                  "treatment_pass_rate": 0.2 if reverse else 0.7,
                  "comparator_pass_rate": 0.7 if reverse else 0.2,
                  "delta_pp": -50 if reverse else 50,
                  "units": 20, "wins": 0 if reverse else 10, "losses": 10 if reverse else 0,
                  "ci_pp": {"lower": -70 if reverse else 30, "upper": -30 if reverse else 70},
                  "sign_test": {"p_two_sided": 0.001},
              }]}}
    before = deepcopy(report)
    loaded = Loaded(tmp_path / "working-memory.report.json", NOW.timestamp(), report)
    collected = faculties.collect(tmp_path, now=NOW, bench=[loaded])
    section = next(item for item in collected.sections if item.faculty.key == "working_memory")
    evidence = " ".join(section.evidence)
    assert "mechanism comparison" in evidence
    assert "70.0%" in evidence and "20.0%" in evidence
    assert "faculty-wide verdict" in evidence
    assert section.reading == "not measured"
    assert not collected.evidence
    assert "noise floor" not in faculties.document(collected).text()
    assert report == before
