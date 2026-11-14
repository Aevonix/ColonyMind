"""Exercise the shell runner with isolated tiny venvs and a stand-in test suite."""
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/suite_env.sh"


def venv(path):
    subprocess.run([sys._base_executable, "-m", "venv", "--without-pip", str(path)], check=True)
    py = path / "bin/python"
    site = Path(subprocess.check_output([str(py), "-I", "-c",
        "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
    return py, site


@pytest.fixture
def runner(tmp_path):
    tree = tmp_path / "tree"
    (tree / "scripts").mkdir(parents=True)
    (tree / "scripts/suite_env.sh").write_text(SCRIPT.read_text())
    (tree / "scripts/copy_suite_hermes.py").write_text(SCRIPT.with_name("copy_suite_hermes.py").read_text())
    (tree / "scripts/suite_results.py").write_text(SCRIPT.with_name("suite_results.py").read_text())
    for directory in ("sidecar/tests", "tests", "docs/internal", "colonymind", "colonymind_hermes"):
        (tree / directory).mkdir(parents=True)
    for name in ("colonymind", "colonymind_hermes"):
        (tree / name / "__init__.py").write_text("")
    (tree / "docs/internal/suite-known-failures.json").write_text('{"failures": {}}')
    subprocess.run(["git", "init", "--quiet", str(tree)], check=True)
    # No commit needed: the script's identity command can fail in this synthetic fixture.
    py, site = venv(tmp_path / "suite")
    (site / "fixture.pth").write_text(str(tree) + "\n")
    (site / "colonymind_hermes-1.0.dist-info").mkdir()
    (site / "colonymind_hermes-1.0.dist-info/METADATA").write_text(
        "Name: colonymind-hermes\nVersion: 1.0\n")
    (py.parent.parent / "SUITE-TREE").write_text(str(tree) + "\n")
    hermes, hermes_site = venv(tmp_path / "stock")
    metadata = hermes_site / "hermes_agent-0.21.7.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: hermes-agent\nVersion: 0.21.7\n")
    hook = hermes.parent / "hermes-fixture"
    hook.write_text(f"#!{hermes}\nimport sys; print(sys.prefix)\n")
    hook.chmod(0o700)
    launcher = hermes.parent / "hermes-long-fixture"
    launcher.write_text(f"#!/bin/sh\n'''exec' '{hermes}' \"$0\" \"$@\"\n' '''\nimport sys; print(sys.prefix)\n")
    launcher.chmod(0o700)
    (site / "pytest").mkdir()
    (site / "pytest/__init__.py").write_text("")
    (site / "pytest/__main__.py").write_text('''
import json, os, pathlib, subprocess, sys
py = pathlib.Path(os.environ['COLONYMIND_TEST_HERMES_PYTHON'])
prefix = subprocess.check_output([str(py), '-I', '-c', 'import sys; print(sys.prefix)'], text=True).strip()
assert pathlib.Path(prefix) == py.parent.parent
assert pathlib.Path(prefix).is_relative_to(pathlib.Path(os.environ['TMPDIR']))
for command in ('hermes-fixture', 'hermes-long-fixture'):
    assert subprocess.check_output([str(py.parent / command)], text=True).strip() == prefix
leaf = 'sidecar' if pathlib.Path.cwd().name == 'sidecar' else 'adapter'
if leaf == 'adapter':
    assert 'scripts.suite_poll_loop' in sys.argv
(pathlib.Path(os.environ['TMPDIR']) / (leaf + '-proof.json')).write_text(json.dumps({'python': str(py)}))
status = int(os.environ.get('FIXTURE_PYTEST_EXIT', '0'))
report = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--junitxml='))
pathlib.Path(report).write_text(
    f'<testsuites><testsuite tests="1" failures="{int(status == 1)}" errors="0" skipped="0">'
    '<testcase classname="tests.test_fixture" name="test_result">'
    + ('<failure message="synthetic failure"/>' if status == 1 else '')
    + '</testcase></testsuite></testsuites>')
results = next((arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--suite-results=')), None)
failed = {'tests/test_fixture.py::test_result': ['tests.test_fixture', 'test_result']} if status == 1 else {}
if results:
    pathlib.Path(results).write_text(json.dumps({'failed': failed, 'uncollected': []}))
if status == 1:
    print('FAILED tests/test_fixture.py::test_result - synthetic failure')
    print('1 failed in 0.01s')
else:
    print('1 passed in 0.01s')
sys.exit(status)
''')
    # The fixture's stand-in pytest must not be shadowed by the outer suite's libraries.
    env = {**os.environ, "PYTHONPATH": "", "PYTHON312": str(py), "SUITE_VENV": str(py.parent.parent),
           "HERMES_SRC": str(tree), "HERMES_PYTHON": str(hermes), "SUITE_PRIVATE_ROOT": str(tmp_path)}

    def run(mode="run", **overrides):
        return subprocess.run(["bash", str(tree / "scripts/suite_env.sh"), mode],
                              env={**env, **overrides}, text=True, capture_output=True, timeout=30)

    return tree, py, hermes, site, run


def test_run_makes_a_new_private_hermes_copy_each_time(runner):
    tree, py, hermes, site, run = runner
    original = (hermes.parent / "hermes-fixture").read_bytes()
    copies = []
    for _ in range(2):
        result = run()
        assert result.returncode == 0, result.stdout + result.stderr
        assert "== import-proof:" in result.stdout
        out = Path(result.stdout.split("== full output: ", 1)[1].splitlines()[0])
        proof = json.loads((out / "adapter-proof.json").read_text())
        assert proof == json.loads((out / "sidecar-proof.json").read_text())
        copies.append(proof["python"])
        assert Path(proof["python"]).parent.parent.stat().st_mode & 0o777 == 0o700
        assert Path(proof["python"]) != hermes
    assert copies[0] != copies[1]
    assert (hermes.parent / "hermes-fixture").read_bytes() == original


def test_template_is_refused_by_prefix_even_through_an_interpreter_alias(runner, tmp_path):
    tree, py, hermes, site, run = runner
    (hermes.parent.parent / "TEMPLATE-DO-NOT-USE-DIRECTLY").write_text("template\n")
    alias = hermes.parent / "python-alias"
    alias.symlink_to(hermes)
    result = run(HERMES_PYTHON=str(alias))
    assert result.returncode == 2
    assert "HERMES_PYTHON must be a per-run copy; this one is marked as a template" in result.stderr
    assert "== sidecar:" not in result.stdout


@pytest.mark.parametrize("mode", ["run", "both", "build"])
def test_a_venv_recorded_for_another_tree_is_refused(runner, mode):
    tree, py, hermes, site, run = runner
    (py.parent.parent / "SUITE-TREE").write_text(str(tree.parent / "other-tree") + "\n")
    result = run(mode)
    assert result.returncode == 2 and "not built for this tree" in result.stderr


def test_imports_from_another_tree_are_refused_before_counting(runner, tmp_path):
    tree, py, hermes, site, run = runner
    other = tmp_path / "other"
    (other / "colonymind").mkdir(parents=True)
    (other / "colonymind/__init__.py").write_text("")
    (tree / "pyproject.toml").write_text(
        '[tool.setuptools.package-dir]\ncolonymind = ' + json.dumps(str(other / 'colonymind')) + '\n')
    result = run()
    assert result.returncode != 0 and "Import provenance mismatch" in result.stderr
    assert "== import-proof:" in result.stdout and "== sidecar:" not in result.stdout


def test_unversioned_hermes_is_refused(runner):
    tree, py, hermes, site, run = runner
    metadata, = hermes.parent.parent.glob("lib/python*/site-packages/hermes_agent-*.dist-info/METADATA")
    metadata.write_text("Name: hermes-agent\nVersion: 0.0.0\n")
    result = run()
    assert result.returncode == 2 and "hermes-agent 0.0.0" in result.stderr


def test_aborted_pytest_run_cannot_produce_suite_counts(runner):
    *_, run = runner
    result = run(FIXTURE_PYTEST_EXIT="2")
    assert result.returncode == 2 and "refusing suite counts" in result.stderr
    assert "== sidecar:" not in result.stdout


def test_timeout_exit_one_cannot_produce_suite_counts(runner):
    tree, py, hermes, site, run = runner
    # Same abrupt exit as pytest-timeout's thread method: no summary, no JUnit.
    (site / "pytest/__main__.py").write_text(
        "import os, sys\n"
        "print('Timeout (>300.0s) from pytest-timeout.', file=sys.stderr, flush=True)\n"
        "os._exit(1)\n")
    result = run()
    assert result.returncode == 2 and "refusing suite counts" in result.stderr
    assert "== sidecar:" not in result.stdout
    assert "== known failures:" not in result.stdout


@pytest.mark.parametrize("phase", ["sidecar", "adapter"])
@pytest.mark.parametrize("report", [None, '<testsuites><testsuite', '<unrelated/>'])
def test_each_phase_requires_a_complete_junit_report(runner, phase, report):
    tree, py, hermes, site, run = runner
    standin = site / "pytest/__main__.py"
    standin.write_text(standin.read_text().replace(
        'sys.exit(status)',
        f"if leaf == {phase!r}:\n"
        + ("    pathlib.Path(report).unlink()\n" if report is None else
           f"    pathlib.Path(report).write_text({report!r})\n")
        + 'sys.exit(status)'))
    result = run()
    assert result.returncode == 2 and f"refusing suite counts ({phase}:" in result.stderr
    assert "== sidecar:" not in result.stdout
    assert "== known failures:" not in result.stdout


def test_completed_exit_one_is_compared_with_known_failures(runner):
    tree, py, hermes, site, run = runner
    failures = {prefix + 'tests/test_fixture.py::test_result': 'synthetic known failure'
                for prefix in ('sidecar/', '')}
    (tree / "docs/internal/suite-known-failures.json").write_text(json.dumps({'failures': failures}))
    result = run(FIXTURE_PYTEST_EXIT="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 of 2 still fail; 0 new" in result.stdout
    assert result.stdout.splitlines()[-1] == "== known failures: 2 of 2 still fail; 0 new; 0 now pass"


def test_shared_dependency_venv_uses_checkout_packages(runner):
    tree, py, hermes, site, run = runner
    (py.parent.parent / "SUITE-TREE").unlink()
    (site / "fixture.pth").write_text("")
    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"colonymind.__file__={tree}/colonymind/__init__.py" in result.stdout
    assert not (py.parent.parent / "SUITE-TREE").exists()


def test_new_failures_have_a_nonzero_final_verdict(runner):
    *_, run = runner
    result = run(FIXTURE_PYTEST_EXIT="1")
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.splitlines()[-1] == "== known failures: 0 of 0 still fail; 2 new; 0 now pass"


def test_logged_errors_are_not_failed_test_ids(runner):
    tree, py, hermes, site, run = runner
    standin = site / 'pytest/__main__.py'
    standin.write_text("print('ERROR    fixture.logger: a handled error')\n" + standin.read_text())
    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines()[-1] == "== known failures: 0 of 0 still fail; 0 new; 0 now pass"


def test_existing_dependency_path_follows_checkout_packages(runner, tmp_path):
    tree, py, hermes, site, run = runner
    extra = tmp_path / 'existing-libraries'
    extra.mkdir()
    (extra / 'fixture_dependency.py').write_text('VALUE = 42\n')
    standin = site / 'pytest/__main__.py'
    standin.write_text('import fixture_dependency\nassert fixture_dependency.VALUE == 42\n' + standin.read_text())
    result = run(PYTHONPATH=str(extra))
    assert result.returncode == 0, result.stdout + result.stderr
    assert f'colonymind.__file__={tree}/colonymind/__init__.py' in result.stdout


PASSING = "def test_ok():\n    pass\n"
FAILING = "def test_bad():\n    assert False\n"
OUTCOMES = '''import pytest


@pytest.fixture
def broken_setup():
    raise RuntimeError("synthetic setup error")


@pytest.fixture
def broken_teardown():
    yield
    raise RuntimeError("synthetic teardown error")


def test_fails():
    assert False


def test_setup_error(broken_setup):
    pass


def test_teardown_error(broken_teardown):
    pass


def test_passes():
    pass


@pytest.mark.skip(reason="synthetic skip")
def test_skipped():
    pass


@pytest.mark.xfail(reason="synthetic expected failure")
def test_expected_failure():
    assert False


@pytest.mark.xfail(reason="synthetic expected failure")
def test_unexpected_pass():
    pass
'''


@pytest.fixture
def real_pytest(runner, tmp_path):
    """Both phases under this interpreter's own pytest instead of the stand-in."""
    tree, py, hermes, site, run = runner
    real, real_site = venv(tmp_path / "real")
    (real_site / "fixture.pth").write_text(f"{tree}\n{Path(pytest.__file__).resolve().parents[1]}\n")
    (real_site / "colonymind_hermes-1.0.dist-info").mkdir()
    (real_site / "colonymind_hermes-1.0.dist-info/METADATA").write_text("Name: colonymind-hermes\nVersion: 1.0\n")
    (real.parent.parent / "SUITE-TREE").write_text(str(tree) + "\n")
    (tree / "scripts/suite_poll_loop.py").write_text(SCRIPT.with_name("suite_poll_loop.py").read_text())
    for root in (tree, tree / "sidecar"):
        (root / "pytest.ini").write_text("[pytest]\n")

    def run_suites(sidecar, adapter=PASSING, failures=None, **overrides):
        (tree / "sidecar/tests/test_case.py").write_text(sidecar)
        (tree / "tests/test_case.py").write_text(adapter)
        if failures is not None:
            (tree / "docs/internal/suite-known-failures.json").write_text(json.dumps({"failures": failures}))
        overrides = {"PYTEST_ADDOPTS": "", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", **overrides}
        return run(PYTHON312=str(real), SUITE_VENV=str(real.parent.parent), **overrides)

    return tree, run_suites


def output_directory(result):
    return Path(result.stdout.split("== full output: ", 1)[1].splitlines()[0])


def failing_ids(result):
    text = (output_directory(result) / "failing.txt").read_text(encoding="utf-8")
    assert text == "" or text.endswith("\n")
    return sorted(text.split("\n")[:-1])


def stand_in_writes(site, junit=None, results=None):
    """Make the stand-in pytest write this JUnit report, or these recorded results, instead of its own."""
    standin = site / "pytest/__main__.py"
    code = ""
    if junit is not None:
        code += f"pathlib.Path(report).write_text({junit!r})\n"
    if results is not None:
        code += f"if results:\n    pathlib.Path(results).write_text({json.dumps(results)!r})\n"
    standin.write_text(standin.read_text().replace("sys.exit(status)", code + "sys.exit(status)"))


def recorded(failed):
    return {"failed": failed, "uncollected": []}


@pytest.mark.parametrize("options", ["--color=yes", "--no-summary"])
def test_output_options_do_not_change_which_tests_failed(real_pytest, options):
    tree, run_suites = real_pytest
    result = run_suites(OUTCOMES, FAILING, PYTEST_ADDOPTS=options)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.splitlines()[-1] == "== known failures: 0 of 0 still fail; 4 new; 0 now pass"
    # Setup and teardown errors are named like failed calls; skips and expected failures are not failures.
    assert failing_ids(result) == ["sidecar/tests/test_case.py::test_fails",
                                   "sidecar/tests/test_case.py::test_setup_error",
                                   "sidecar/tests/test_case.py::test_teardown_error",
                                   "tests/test_case.py::test_bad"]


def test_a_failed_call_with_a_failed_teardown_is_one_id_counted_once(real_pytest):
    tree, run_suites = real_pytest
    both = ("import pytest\n\n\n"
            "@pytest.fixture\ndef broken_teardown():\n    yield\n"
            "    raise RuntimeError('synthetic teardown error')\n\n\n"
            "def test_both(broken_teardown):\n    assert False\n")
    result = run_suites(both, PYTEST_ADDOPTS="--no-summary")
    assert result.returncode == 1, result.stdout + result.stderr
    # The JUnit report holds the failed call and the failed teardown as two entries of the one test.
    cases = ET.parse(output_directory(result) / "sidecar.xml").getroot().iter("testcase")
    assert [[child.tag for child in case] for case in cases] == [["failure"], ["error"]]
    assert result.stdout.splitlines()[-1] == "== known failures: 0 of 0 still fail; 1 new; 0 now pass"
    assert failing_ids(result) == ["sidecar/tests/test_case.py::test_both"]
    result = run_suites(both, failures={"sidecar/tests/test_case.py::test_both": "Known"}, PYTEST_ADDOPTS="--no-summary")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines()[-1] == "== known failures: 1 of 1 still fail; 0 new; 0 now pass"


@pytest.mark.parametrize("status,failed", [
    ("1", {}),
    ("0", {"tests/test_fixture.py::test_result": ["tests.test_fixture", "test_result"]}),
    ("1", {"tests/test_fixture.py::test_other": ["tests.test_fixture", "test_other"]}),
], ids=["failure-without-id", "id-of-a-passed-test", "id-of-another-test"])
def test_recorded_tests_that_disagree_with_the_junit_report_refuse_the_run(runner, status, failed):
    tree, py, hermes, site, run = runner
    stand_in_writes(site, results=recorded(failed))
    result = run(FIXTURE_PYTEST_EXIT=status)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "refusing suite counts (sidecar: recorded failed tests disagree with the report)" in result.stderr
    assert "== known failures:" not in result.stdout


def parametrized(*ids):
    return ("import pytest\n\n\n"
            f"@pytest.mark.parametrize('value', {list(range(len(ids)))!r}, ids={list(ids)!r})\n"
            "def test_bad(value):\n    assert False\n")


def test_a_recorded_cut_form_does_not_cover_the_complete_id(real_pytest):
    tree, run_suites = real_pytest
    result = run_suites(parametrized("alpha - one"),
                        failures={"sidecar/tests/test_case.py::test_bad[alpha": "A cut form of the failing id"})
    assert result.returncode == 1, result.stdout + result.stderr
    assert "NEW    sidecar/tests/test_case.py::test_bad[alpha - one]" in result.stdout.splitlines()
    assert result.stdout.splitlines()[-1] == "== known failures: 0 of 1 still fail; 1 new; 1 now pass"


def test_ids_that_share_a_cut_form_stay_distinct(real_pytest):
    tree, run_suites = real_pytest
    result = run_suites(parametrized("alpha - one", "alpha - two"),
                        failures={"sidecar/tests/test_case.py::test_bad[alpha - one]": "Known first parameter"})
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.splitlines()[-1] == "== known failures: 1 of 1 still fail; 1 new; 0 now pass"
    assert failing_ids(result) == ["sidecar/tests/test_case.py::test_bad[alpha - one]",
                                   "sidecar/tests/test_case.py::test_bad[alpha - two]"]


def test_nothing_trims_an_id_on_its_way_to_the_comparison(runner):
    tree, py, hermes, site, run = runner
    stand_in_writes(site, junit='<testsuites><testsuite tests="1" failures="1" errors="0" skipped="0">'
                    '<testcase classname="tests.test_fixture" name="test_result "><failure message="synthetic"/>'
                    '</testcase></testsuite></testsuites>',
                    results=recorded({"tests/test_fixture.py::test_result ": ["tests.test_fixture", "test_result "]}))
    (tree / "docs/internal/suite-known-failures.json").write_text(json.dumps(
        {"failures": {"sidecar/tests/test_fixture.py::test_result": "A trimmed form of the failing id"}}))
    result = run(FIXTURE_PYTEST_EXIT="1")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "NEW    sidecar/tests/test_fixture.py::test_result " in result.stdout.splitlines()
    assert result.stdout.splitlines()[-1] == "== known failures: 0 of 1 still fail; 2 new; 1 now pass"


def test_a_phase_that_ran_no_test_is_refused(runner):
    tree, py, hermes, site, run = runner
    stand_in_writes(site, junit='<testsuites><testsuite tests="0" failures="0" errors="0" skipped="0">'
                    '</testsuite></testsuites>')
    result = run()
    assert result.returncode == 2, result.stdout + result.stderr
    assert "refusing suite counts (sidecar: no test ran)" in result.stderr
    assert "== known failures:" not in result.stdout


PASSED_CASE = '<testcase classname="tests.test_fixture" name="test_result"/>'
SKIPPED_CASE = '<testcase classname="tests.test_fixture" name="test_result"><skipped message="synthetic"/></testcase>'


@pytest.mark.parametrize("status,suite", [
    ("0", '<testsuite tests="1" failures="0" errors="0" skipped="0"></testsuite>'),
    ("1", f'<testsuite tests="1" failures="1" errors="0" skipped="0">{PASSED_CASE}</testsuite>'),
    ("0", f'<testsuite tests="1" failures="0" errors="0" skipped="0">{SKIPPED_CASE}</testsuite>'),
], ids=["declared-test-without-entry", "declared-failure-without-entry", "undeclared-skip"])
def test_a_report_whose_declared_counts_do_not_fit_its_entries_is_refused(runner, status, suite):
    tree, py, hermes, site, run = runner
    stand_in_writes(site, junit=f"<testsuites>{suite}</testsuites>")
    result = run(FIXTURE_PYTEST_EXIT=status)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "refusing suite counts (sidecar: declared counts" in result.stderr
    assert "do not fit the test entries" in result.stderr
    assert "== known failures:" not in result.stdout


def test_a_declared_test_count_above_the_number_of_entries_is_accepted(runner):
    tree, py, hermes, site, run = runner
    # The shape a passing test with passing subtests gives: pytest counts the subtests, the report holds one entry.
    stand_in_writes(site, junit='<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0">'
                    f'{PASSED_CASE}</testsuite></testsuites>')
    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines()[-1] == "== known failures: 0 of 0 still fail; 0 new; 0 now pass"


def test_a_teardown_error_reported_as_an_older_pytest_does_is_a_new_failure(runner):
    tree, py, hermes, site, run = runner
    # pytest 8 declares a passed call and the error of its teardown as two tests and writes one entry.
    stand_in_writes(site, junit='<testsuites><testsuite tests="2" failures="0" errors="1" skipped="0">'
                    '<testcase classname="tests.test_fixture" name="test_result"><error message="synthetic"/>'
                    '</testcase></testsuite></testsuites>')
    result = run(FIXTURE_PYTEST_EXIT="1")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "NEW    sidecar/tests/test_fixture.py::test_result" in result.stdout.splitlines()
    assert result.stdout.splitlines()[-1] == "== known failures: 0 of 0 still fail; 2 new; 0 now pass"


SUBTESTS = '''import unittest


class Values(unittest.TestCase):
    def test_values(self):
        for value in range(2):
            with self.subTest(value=value):
                self.assertLess(value, LIMIT)
'''
EXPECTED_FAILURE_WITH_A_FAILED_TEARDOWN = '''import pytest


@pytest.fixture
def broken_teardown():
    yield
    raise RuntimeError("synthetic teardown error")


@pytest.mark.xfail(reason="synthetic expected failure")
def test_expected_failure(broken_teardown):
    assert False
'''


def test_a_passing_test_with_passing_subtests_is_accepted(real_pytest):
    tree, run_suites = real_pytest
    result = run_suites(SUBTESTS.replace("LIMIT", "2"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines()[-1] == "== known failures: 0 of 0 still fail; 0 new; 0 now pass"
    assert failing_ids(result) == []


def test_a_test_whose_subtests_fail_is_a_new_failure(real_pytest):
    tree, run_suites = real_pytest
    result = run_suites(SUBTESTS.replace("LIMIT", "1"))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "NEW    sidecar/tests/test_case.py::Values::test_values" in result.stdout.splitlines()
    assert result.stdout.splitlines()[-1] == "== known failures: 0 of 0 still fail; 1 new; 0 now pass"


def test_an_expected_failure_whose_teardown_raises_is_accepted(real_pytest):
    tree, run_suites = real_pytest
    result = run_suites(EXPECTED_FAILURE_WITH_A_FAILED_TEARDOWN)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines()[-1] == "== known failures: 0 of 0 still fail; 0 new; 0 now pass"
    assert failing_ids(result) == []


A_SKIP_AND_A_BODY_THAT_WOULD_FAIL = '''import pytest


@pytest.mark.skip(reason="synthetic skip")
def test_skipped():
    pass


def test_body_would_fail():
    assert False
'''


def test_a_run_in_which_no_test_body_ran_is_refused(real_pytest):
    tree, run_suites = real_pytest
    # With this option pytest calls no test: each report holds an entry for both tests and counts the skip alone,
    # so the test whose body would fail is no failure there and pytest's own status is 0.
    result = run_suites(A_SKIP_AND_A_BODY_THAT_WOULD_FAIL, A_SKIP_AND_A_BODY_THAT_WOULD_FAIL,
                        PYTEST_ADDOPTS="--setup-plan")
    assert result.returncode == 2, result.stdout + result.stderr
    for phase in ("sidecar", "adapter"):
        suite = ET.parse(output_directory(result) / f"{phase}.xml").getroot().find("testsuite")
        assert (suite.get("tests"), suite.get("failures"), suite.get("errors"), suite.get("skipped"),
                len(suite.findall("testcase"))) == ("1", "0", "0", "1", 2)
    assert "refusing suite counts (sidecar: declared counts" in result.stderr
    assert "do not fit the test entries" in result.stderr
    assert "== known failures:" not in result.stdout


ENDED_BY_A_TEST = '''import pytest


def test_runs():
    pass


def test_ends_the_session():
    pytest.exit("synthetic early end", returncode=RETURNCODE)


def test_never_runs():
    assert False
'''
ENDED_BY_A_FIXTURE = '''import pytest


@pytest.fixture
def ends_the_session():
    pytest.exit("synthetic early end", returncode=RETURNCODE)


def test_runs():
    pass


def test_ends_in_setup(ends_the_session):
    pass


def test_never_runs():
    assert False
'''


@pytest.mark.parametrize("returncode", [0, 2])
@pytest.mark.parametrize("source", [ENDED_BY_A_TEST, ENDED_BY_A_FIXTURE], ids=["test", "fixture"])
@pytest.mark.parametrize("phase", ["sidecar", "adapter"])
def test_a_session_that_a_test_or_a_fixture_ends_early_is_refused(real_pytest, phase, source, returncode):
    tree, run_suites = real_pytest
    source = source.replace("RETURNCODE", str(returncode))
    result = run_suites(source) if phase == "sidecar" else run_suites(PASSING, source)
    assert result.returncode == 2, result.stdout + result.stderr
    # One line on stderr says why, whatever status pytest.exit gave the session, and nothing follows it.
    reasons = [line for line in result.stderr.splitlines() if "refusing suite counts" in line]
    assert reasons == ["Suite did not complete normally; refusing suite counts "
                       f"({phase}: the session was ended early by pytest.exit)"], result.stderr
    assert result.stderr.splitlines()[-1] == reasons[0]
    assert "== known failures:" not in result.stdout


@pytest.mark.parametrize("record", [
    '{"failures": {"tests/test_fixture.py::test_result": "First", "tests/test_fixture.py::test_result": "Second"}}',
    '{"failures": {}, "failures": {}}',
    '[]',
    '{"causes": {}}',
    '{"failures": ["tests/test_fixture.py::test_result"]}',
    '{"failures": {"tests/test_fixture.py::test_result": 1}}',
    '{"failures": {}}\n{"failures": {}}\n',
    '{"failures": {',
], ids=["duplicate-id", "duplicate-member", "not-an-object", "no-failures-member", "failures-not-an-object",
        "cause-not-text", "two-documents", "not-json"])
def test_a_record_that_cannot_give_a_sound_verdict_is_refused(runner, record):
    tree, py, hermes, site, run = runner
    (tree / "docs/internal/suite-known-failures.json").write_text(record)
    result = run()
    assert result.returncode == 2, result.stdout + result.stderr
    assert "== known failures:" not in result.stdout
    # One line on stderr says why, and nothing follows it.
    reasons = [line for line in result.stderr.splitlines() if "known-failure record" in line]
    assert len(reasons) == 1 and result.stderr.splitlines()[-1] == reasons[0], result.stderr
    assert reasons[0].startswith("Refusing suite counts: known-failure record ")
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("failures", [{}, {"sidecar/tests/test_import.py": "A module that cannot be imported"}],
                         ids=["unrecorded", "recorded"])
def test_a_module_that_could_not_be_collected_refuses_the_run_whatever_the_exit_status(real_pytest, failures):
    tree, run_suites = real_pytest
    (tree / "sidecar/tests/test_import.py").write_text("raise ImportError('synthetic collection error')\n")
    # With this option pytest runs the other modules and exits with 1, the status of failed tests.
    result = run_suites(PASSING, failures=failures, PYTEST_ADDOPTS="--continue-on-collection-errors")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "refusing suite counts (sidecar: could not collect tests/test_import.py)" in result.stderr
    assert "== known failures:" not in result.stdout
