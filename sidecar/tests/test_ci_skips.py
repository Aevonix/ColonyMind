"""The CI skip guard rejects missing coverage and never hides a failed test."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_sidecar_skips", ROOT / "scripts/check_sidecar_skips.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
CASE_ID = ("tests.hermes_adapter.test_hands_example", "test_case")


def report(tmp_path, *, marker=None, expected="", skipped=None, outcome=None,
           nodeid="tests/hermes_adapter/test_hands_example.py::test_case"):
    suite = ET.Element("testsuite")
    module, *classes, name = nodeid.split("::")
    classname = ".".join([module.removesuffix(".py").replace("/", "."), *classes])
    case = ET.SubElement(suite, "testcase", classname=classname, name=name)
    properties = ET.SubElement(case, "properties")
    ET.SubElement(properties, "property", name="guard_nodeid", value=nodeid)
    if marker:
        ET.SubElement(properties, "property", name="guard_marker", value=marker)
        ET.SubElement(properties, "property", name="guard_expected_skip", value=expected)
    if skipped is not None:
        ET.SubElement(case, "skipped", message=skipped)
    if outcome:
        ET.SubElement(case, outcome)
    path = tmp_path / "results.xml"
    ET.ElementTree(suite).write(path)
    return path


@pytest.mark.parametrize("marker,reason", guard.SKIP_REASONS.items())
def test_guard_accepts_exact_skip_set_and_runs_with_available_prerequisite(tmp_path, marker, reason):
    marked = [CASE_ID] if marker == "requires_sidecar" else []
    assert guard.check_report(report(tmp_path, marker=marker, expected=reason, skipped=reason), marked) == []
    assert guard.check_report(report(tmp_path, marker=marker)) == []


@pytest.mark.parametrize("marker,reason", guard.SKIP_REASONS.items())
def test_guard_rejects_a_marked_test_that_did_not_skip_or_skipped_for_another_reason(tmp_path, marker, reason):
    assert guard.check_report(report(tmp_path, marker=marker, expected=reason))
    assert guard.check_report(report(tmp_path, marker=marker, expected=reason, skipped="another reason"))


@pytest.mark.parametrize("reason", [*guard.SKIP_REASONS.values(), "unexpected hands skip"])
def test_guard_rejects_an_unmarked_skip(tmp_path, reason):
    assert guard.check_report(report(tmp_path, skipped=reason))


@pytest.mark.parametrize("outcome", ["failure", "error"])
def test_guard_rejects_failures_and_errors_even_when_skip_sets_match(tmp_path, outcome):
    assert guard.check_report(report(tmp_path, outcome=outcome))


def test_guard_rejects_empty_reports_and_missing_test_identities(tmp_path):
    path = tmp_path / "results.xml"
    path.write_text("<testsuite />")
    assert guard.check_report(path)
    path.write_text("<testsuite><testcase><error /></testcase></testsuite>")
    assert guard.check_report(path)


def test_guard_rejects_a_skip_when_the_prerequisite_is_available(tmp_path):
    marker, reason = next(iter(guard.SKIP_REASONS.items()))
    assert guard.check_report(report(tmp_path, marker=marker, skipped=reason))


@pytest.mark.parametrize("marker", guard.SKIP_REASONS)
@pytest.mark.parametrize("reason", [*guard.SKIP_REASONS.values(), "set HERMES_SOURCE_ROOT"])
def test_guard_reports_any_skip_when_a_marked_test_must_run(tmp_path, marker, reason):
    problems = guard.check_report(report(tmp_path, marker=marker, skipped=reason))
    assert any("expected to run but skipped" in problem for problem in problems)


@pytest.mark.parametrize("reason", [*guard.SKIP_REASONS.values(), "set HERMES_SOURCE_ROOT"])
def test_guard_rejects_unguarded_skips_outside_hands(tmp_path, reason):
    path = report(tmp_path, skipped=reason, nodeid="tests/hermes_adapter/test_step0_load.py::test_case")
    assert guard.check_report(path)


def test_sidecar_absent_requires_exact_collected_identities_not_just_equal_counts(tmp_path):
    path = report(tmp_path, skipped=guard.SIDECAR_ABSENT)
    guard.check_skips(path, [CASE_ID])
    with pytest.raises(ValueError, match="expected"):
        guard.check_skips(path, [(CASE_ID[0], "test_another_case")])
    with pytest.raises(ValueError, match="expected"):
        guard.check_skips(report(tmp_path), [CASE_ID])


@pytest.mark.parametrize("module", ["colonymind", "colonymind.hands.wire"])
def test_sidecar_importorskip_is_refused_even_outside_hands(tmp_path, module):
    path = report(tmp_path, skipped=f"could not import '{module}': No module named '{module}'")
    tree = ET.parse(path)
    case = next(tree.iter("testcase"))
    case.set("classname", "tests.hermes_adapter.test_example")
    case.find("properties/property").set("value", "tests/hermes_adapter/test_example.py::test_case")
    tree.write(path)
    assert any("sidecar import skip" in problem for problem in guard.check_report(path))


def test_sidecar_marker_collection_preserves_class_and_parameter_identity(monkeypatch):
    def collect(args, plugins):
        assert args == ["tests/hermes_adapter", "--collect-only", "-m", "requires_sidecar", "-q"]
        plugins[0].pytest_collection_finish(SimpleNamespace(items=[SimpleNamespace(
            nodeid="tests/hermes_adapter/test_example.py::TestClass::test_case[value]")]))
        return pytest.ExitCode.OK

    monkeypatch.setattr(pytest, "main", collect)
    assert guard.collect_requires_sidecar() == [("tests.hermes_adapter.test_example.TestClass", "test_case[value]")]


@pytest.mark.parametrize("outcome", ["failed", "skipped", "interrupted"])
def test_sidecar_marker_collection_must_complete_without_import_skips(monkeypatch, outcome):
    def collect(args, plugins):
        plugins[0].pytest_collectreport(SimpleNamespace(failed=outcome == "failed", skipped=outcome == "skipped"))
        return pytest.ExitCode.INTERRUPTED if outcome == "interrupted" else pytest.ExitCode.OK

    monkeypatch.setattr(pytest, "main", collect)
    with pytest.raises(ValueError, match="marker collection did not complete"):
        guard.collect_requires_sidecar()


@pytest.mark.parametrize("sidecar_present", [False, True])
def test_guard_cli_keeps_the_single_report_argument_and_detects_sidecar_absence(tmp_path, monkeypatch, sidecar_present):
    path = report(tmp_path, skipped=None if sidecar_present else guard.SIDECAR_ABSENT)
    monkeypatch.setattr(guard.sys, "argv", ["check_sidecar_skips.py", str(path)])
    monkeypatch.setattr(guard.importlib.util, "find_spec", lambda name: object() if sidecar_present else None)
    collected = []

    def collect():
        collected.append(True)
        return [CASE_ID]

    monkeypatch.setattr(guard, "collect_requires_sidecar", collect)
    assert guard.main() == 0
    assert collected == ([] if sidecar_present else [True])


CASE = ("tests.hermes_adapter.test_contract", "test_real_sidecar")


def sidecar_report(tmp_path, skipped, *, reason=None, failure=False):
    suite = ET.Element("testsuite")
    for classname, name in skipped:
        case = ET.SubElement(suite, "testcase", classname=classname, name=name)
        properties = ET.SubElement(case, "properties")
        ET.SubElement(properties, "property", name="guard_nodeid", value=classname + "::" + name)
        ET.SubElement(case, "skipped", message=reason or guard.SIDECAR_ABSENT)
    if failure:
        case = ET.SubElement(suite, "testcase", classname="adapter", name="test_import")
        ET.SubElement(case, "error", message="unmarked dependency")
    path = tmp_path / "results.xml"
    ET.ElementTree(suite).write(path)
    return path


def test_every_marked_test_must_have_the_distinctive_skip(tmp_path):
    guard.check_skips(sidecar_report(tmp_path, [CASE]), [CASE])


@pytest.mark.parametrize("skipped,marked,reason,failure", [
    ([], [CASE], None, False),
    ([CASE], [], None, False),
    ([(CASE[0], "test_other")], [CASE], None, False),
    ([CASE], [CASE], "some unrelated skip", False),
    ([CASE, CASE], [CASE], None, False),
    ([CASE], [CASE], None, True),
])
def test_missing_extra_wrong_or_failed_tests_reject_the_sidecar_report(tmp_path, skipped, marked, reason, failure):
    with pytest.raises(ValueError):
        guard.check_skips(sidecar_report(tmp_path, skipped, reason=reason, failure=failure), marked)


def patch_hook_xfail_reason():
    """The reason the real test raises, read from its source rather than from the guard."""
    module = ast.parse((ROOT / "tests/hermes_adapter/test_work_capture.py").read_text())
    [value] = [node.value for node in module.body if isinstance(node, ast.Assign)
               and [ast.unparse(target) for target in node.targets] == ["PATCH_HOOK_GC_XFAIL"]]
    return ast.literal_eval(value)


def xfail_report(tmp_path, *, nodeid=guard.PATCH_HOOK_XFAIL_NODEID, message=None,
                 skip_type="pytest.xfail", marker=None):
    message = patch_hook_xfail_reason() if message is None else message
    path = report(tmp_path, nodeid=nodeid, skipped=message, marker=marker)
    tree = ET.parse(path)
    tree.find("testcase/skipped").set("type", skip_type)
    tree.write(path)
    return path


def test_guard_accepts_the_tracked_patch_hook_latency_xfail(tmp_path):
    assert guard.check_report(xfail_report(tmp_path)) == []


@pytest.mark.parametrize("changes", [
    {"nodeid": "tests/hermes_adapter/test_work_capture.py::test_other"},
    {"message": "unexpected xfail"},
    {"message": "3.14 patch hook p99 over 1 ms; tracked bug"},
    {"skip_type": "pytest.skip"},
    {"marker": "unknown_prerequisite"},
    {"marker": "requires_live_qualification"},
    {"marker": "requires_sidecar"},
])
def test_guard_does_not_allow_other_xfails_or_xfail_a_must_run_test(tmp_path, changes):
    assert guard.check_report(xfail_report(tmp_path, **changes))


def test_hardware_latency_hooks_keep_the_marker_and_enforce_outside_github_actions():
    module = ast.parse((ROOT / "tests/hermes_adapter/test_work_capture.py").read_text())
    for name in (
        "test_new_metadata_hooks_are_sync_extensible_enqueue_only_under_one_ms",
        "test_maximum_resource_patch_hook_stays_under_one_ms",
    ):
        [test] = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name]
        # Neither a skip nor an xfail may absorb a functional failure on any Python.
        assert [ast.unparse(node) for node in test.decorator_list] == ["pytest.mark.hardware_latency"]
        assert "hardware_latency" in [arg.arg for arg in test.args.args]
        # The measurement is unconditional, with the same strict local bar on every Python.
        body = test.body
        if name == "test_maximum_resource_patch_hook_stays_under_one_ms":
            # Only this bar's failure on 3.13 and 3.14 becomes the bounded GC xfail.
            [guarded] = [node for node in body if isinstance(node, ast.Try)]
            [handler] = guarded.handlers
            assert ast.unparse(handler.type) == "pytest.fail.Exception"
            assert not guarded.orelse and not guarded.finalbody
            assert [ast.unparse(node) for node in handler.body] == [
                "if sys.version_info[:2] not in {(3, 13), (3, 14)} or statistics.quantiles(cpu_samples, n=100)[98] >= 2.5:"
                "\n    raise",
                "pytest.xfail(PATCH_HOOK_GC_XFAIL)"]
            # The guard allows this xfail by the same reason prefix the test raises.
            # The guard matches that prefix only: this pinned handler is what limits
            # which Pythons may xfail and how far the bar may be missed.
            assert patch_hook_xfail_reason().startswith(guard.PATCH_HOOK_XFAIL_PREFIX)
            body = guarded.body
        [call] = [node.value for node in body if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                  and node.value.func.id == "hardware_latency"]
        assert [ast.unparse(arg) for arg in call.args] == ["samples", "cpu_samples"]
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert set(keywords) == {"threshold_ms", "enforced"}
        assert ast.literal_eval(keywords["threshold_ms"]) == 1
        assert ast.unparse(keywords["enforced"]) == "os.environ.get('GITHUB_ACTIONS') != 'true'"


@pytest.mark.parametrize("name", ["test_pass_runs_off_the_event_loop_while_a_large_sample_file_is_processed",
                                  "test_loop_side_lock_users_skip_busy_steps_without_waiting"])
def test_service_health_stall_uses_the_same_policy_with_its_maximum_gap_bar(name):
    module = ast.parse((ROOT / "sidecar/tests/test_plasticity_service_health_stall.py").read_text())
    [test] = [node for node in module.body if isinstance(node, ast.AsyncFunctionDef) and node.name == name]
    assert [ast.unparse(node) for node in test.decorator_list] == ["pytest.mark.hardware_latency"]
    assert "hardware_latency" in [arg.arg for arg in test.args.args]
    [call] = [node for node in ast.walk(test) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Name) and node.func.id == "hardware_latency"]
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    assert ast.literal_eval(keywords["threshold_ms"]) == 250
    assert ast.literal_eval(keywords["statistic"]) == "max"
    assert ast.unparse(keywords["enforced"]) == "os.environ.get('GITHUB_ACTIONS') != 'true'"
