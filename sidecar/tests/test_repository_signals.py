"""The repository's public promises match what the code declares and CI checks.

A classifier, a CI matrix, a compatibility table or a security policy that disagrees with the packages misleads
the person deciding whether to install ColonyMind, so each one is read back against its source here.
"""

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.specifiers import SpecifierSet

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
PACKAGES = (ROOT / "pyproject.toml", ROOT / "sidecar" / "pyproject.toml")
# A reporter's address or ours, in any form a mail client would take.
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# The jobs that run ColonyMind's own code on an interpreter; building the distributions and the image do not.
INTERPRETER_JOBS = ("python", "hermes-adapter", "fresh-install")


def _project(path):
    return tomllib.loads(path.read_text())["project"]


def _accepted_pythons(path):
    accepted = SpecifierSet(_project(path)["requires-python"])
    return [f"3.{minor}" for minor in range(8, 30) if f"3.{minor}" in accepted]


def _supported_pythons():
    """The Pythons the whole product runs on: both packages accept them. The adapter alone may accept more, since it
    runs inside Hermes' interpreter rather than the sidecar's."""
    sidecar, adapter = (_accepted_pythons(path) for path in reversed(PACKAGES))
    assert set(sidecar) <= set(adapter), "the adapter must accept every Python the sidecar does"
    return sidecar


def _workflow(name):
    return yaml.safe_load((WORKFLOWS / name).read_text())


def _setup_python(job):
    steps = [step for step in job["steps"] if str(step.get("uses", "")).startswith("actions/setup-python@")]
    assert len(steps) == 1
    return steps[0]["with"]["python-version"]


def test_both_packages_classify_exactly_the_pythons_they_accept():
    for path in PACKAGES:
        classifiers = _project(path).get("classifiers", [])
        listed = [c.rsplit(" :: ", 1)[1] for c in classifiers if c.startswith("Programming Language :: Python :: 3.")]
        assert listed == _accepted_pythons(path), path
        assert "Programming Language :: Python :: 3 :: Only" in classifiers, path


def test_ci_runs_colonymind_on_every_python_it_accepts():
    """Together the adapter jobs cover its accepted Pythons; the other jobs cover the whole product's range.

    The adapter's extra Python runs without the sidecar, whose accepted range is narrower.
    """
    jobs = _workflow("ci.yml")["jobs"]
    adapter_versions = set()
    sidecar_free = ("hermes-adapter-py314", "hermes-adapter-hermes023")
    names = (*INTERPRETER_JOBS, *sidecar_free)
    for name in names:
        job = jobs[name]
        strategy = job.get("strategy", {})
        versions = strategy.get("matrix", {}).get("python-version")
        if versions is not None:
            # One interpreter's failure must not cancel the others: each version's result is reported.
            assert strategy.get("fail-fast") is False, name
            assert _setup_python(job) == "${{ matrix.python-version }}", name
        else:
            versions = [_setup_python(job)]
        if name.startswith("hermes-adapter"):
            assert any("pytest tests/hermes_adapter" in step.get("run", "") for step in job["steps"]), name
            adapter_versions.update(versions)
            if name == "hermes-adapter":
                # The sidecar-installed matrix is exactly the sidecar's range; 3.14 runs only in the sidecar-free job.
                assert versions == _supported_pythons(), name
        else:
            assert versions == _supported_pythons(), name
    assert adapter_versions == set(_accepted_pythons(PACKAGES[0]))
    for name in sidecar_free:
        runs = " ".join(step.get("run", "") for step in jobs[name]["steps"])
        assert "find_spec('colonymind') is None" in runs and "scripts/check_sidecar_skips.py" in runs, name


def test_ci_proves_a_hermes_that_its_own_installer_set_up():
    """One job sets Hermes up with Hermes' installer, pinned to a release, with uv on PATH, and then runs the four
    proof steps of fresh-install through the same scripts. They find the installer's one Hermes home through
    HERMES_INSTALLER_HOME, which no step of the job hides."""
    jobs = _workflow("ci.yml")["jobs"]
    job, source = jobs["fresh-install-installer"], jobs["fresh-install"]
    assert job["needs"] == "artifacts" and _setup_python(job) == "3.12"
    assert job["env"] == {"DIST_DIR": "${{ github.workspace }}/dist"}
    runs = [step.get("run", "") for step in job["steps"]]
    [installer] = [run for run in runs if "hermes-agent.nousresearch.com/install.sh" in run]
    assert "--branch v0.23.1" in installer
    assert "HERMES_INSTALLER_HOME=${HOME}/.hermes" in installer and '>> "${GITHUB_ENV}"' in installer
    assert "HERMES_SRC=${HOME}/.hermes/hermes-agent" in installer and '>> "${GITHUB_PATH}"' in installer
    assert any("pip install 'uv==" in run for run in runs[:runs.index(installer)])
    proof = [(step["name"], step["run"]) for step in source["steps"] if "run" in step]
    assert len(proof) == 4
    assert [(step.get("name"), step.get("run")) for step in job["steps"][-4:]] == proof
    assert not any("env" in step for step in job["steps"][-4:])


def test_ci_installs_hermes_023_for_the_adapter_suite_with_hermes_own_overrides():
    """pip cannot resolve Hermes 0.23.1's own pins; the job resolves as Hermes' installer does, with uv and the
    override list of the checkout, and accepts no dependency conflict but the one Hermes ships with."""
    job = _workflow("ci.yml")["jobs"]["hermes-adapter-hermes023"]
    assert job["needs"] == "artifacts" and _setup_python(job) == "3.14"
    [install] = [step["run"] for step in job["steps"] if "install" in step.get("name", "").lower()]
    assert '["tool"]["uv"]["override-dependencies"]' in install
    assert "uv pip install" in install and "--overrides" in install and "-e ./.qualification/hermes" in install
    assert "python -m pip install -e" not in install
    assert '"-m", "pip", "check"' in install and 'startswith("browser-harness ")' in install


def test_matrix_jobs_upload_one_artifact_per_python():
    """Artifact names are unique per workflow run, so a matrix job that uploads under one name fails its second leg."""
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for name, job in yaml.safe_load(path.read_text())["jobs"].items():
            if "matrix" not in job.get("strategy", {}):
                continue
            for step in job.get("steps", []):
                if str(step.get("uses", "")).startswith("actions/upload-artifact@"):
                    assert "matrix.python-version" in step["with"]["name"], f"{path.name}:{name}"


def test_ci_uses_reviewed_stock_s_for_imports_and_source_checks():
    workflow = _workflow("ci.yml")
    assert workflow["env"]["HERMES_REF"] == "345cd2b057a452236de401d3534b8502a7465e8d"
    assert workflow["env"]["HERMES_REF_ADAPTER_SUITE"] == "71d34bad1045610252eda02ee79549436896e784"
    source_checks = ast.parse((ROOT / "tests/hermes_adapter/hermes_source.py").read_text())
    [pins] = [ast.literal_eval(node.value) for node in source_checks.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "PINS" for target in node.targets)]
    for name in ("HERMES_REF_ADAPTER_SUITE", "HERMES_REF_ADAPTER_SUITE_023"):
        assert workflow["env"][name] in {revision for revision, tree in pins.values()}, name
    assert "HERMES_SOURCE_ROOT" not in workflow["env"]
    assert workflow["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    for name in ("fresh-install", "get-started", "python", "hermes-adapter", "hermes-adapter-py314",
                 "hermes-adapter-hermes023"):
        job = workflow["jobs"][name]
        steps = job["steps"]
        [checkout] = [step["with"] for step in steps
                      if step.get("with", {}).get("repository") == "NousResearch/hermes-agent"]
        expected_ref = {
            "hermes-adapter": "${{ env.HERMES_REF_ADAPTER_SUITE }}",
            "hermes-adapter-py314": "${{ env.HERMES_REF_ADAPTER_SUITE }}",
            "hermes-adapter-hermes023": "${{ env.HERMES_REF_ADAPTER_SUITE_023 }}",
        }.get(name, "${{ env.HERMES_REF }}")
        assert checkout["ref"] == expected_ref, name
        assert checkout["path"] == ".qualification/hermes", name
        assert checkout["persist-credentials"] is False, name
    for name, job in workflow["jobs"].items():
        assert job.get("env", {}).get("HERMES_SOURCE_ROOT") == (
            "${{ github.workspace }}/.qualification/hermes" if name.startswith("hermes-adapter") else None), name
        if not name.startswith("hermes-adapter"):
            assert "HERMES_REF_ADAPTER_SUITE" not in json.dumps(job), name
            assert "HERMES_SOURCE_ROOT" not in json.dumps(job), name
    for name in ("hermes-adapter", "python"):
        steps = workflow["jobs"][name]["steps"]
        installs = "\n".join(step.get("run", "") for step in steps if "install" in step.get("name", "").lower())
        assert "-e ./.qualification/hermes" in installs, name
        assert "git+" not in installs, name


def test_ci_checks_every_adapter_outcome_even_when_pytest_fails():
    steps = _workflow("ci.yml")["jobs"]["hermes-adapter"]["steps"]
    [suite] = [step for step in steps if "pytest tests/hermes_adapter" in step.get("run", "")]
    assert "--junitxml=hermes-adapter-results.xml" in suite["run"]
    [guard] = [step for step in steps if "scripts/check_sidecar_skips.py" in step.get("run", "")]
    assert guard["run"] == "python scripts/check_sidecar_skips.py hermes-adapter-results.xml"
    assert guard["if"] == "always()"
    assert steps.index(suite) < steps.index(guard)


def test_live_qualification_marks_every_external_prerequisite():
    source = ast.parse((ROOT / "tests/hermes_adapter/conftest.py").read_text())
    [prerequisites] = [ast.literal_eval(node.value) for node in source.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "LIVE_PREREQUISITES"
                               for target in node.targets)]
    assert prerequisites == {
        "test_cancel_releases_live_engine_and_recorded_process_integrator": "HANDS_CANCEL_RECEIPT",
        "test_socket_adapter_integrator": "HANDS_RUN_ADAPTER_CHECK",
        "test_reviewed_profile_unload_and_discovery_reproductions": "COLONYMIND_TEST_HERMES_PYTHON",
        "test_routing_drift_compensates_without_positive_receipt": "COLONYMIND_TEST_HERMES_PYTHON",
        "test_actual_gateway_sigkill_stops_unit_and_recovers": "COLONYMIND_TEST_HERMES_PYTHON",
        "test_stalled_engine_does_not_join_at_interpreter_shutdown": "COLONYMIND_TEST_HERMES_PYTHON",
        "test_scheduler_claim_during_stalled_archive_exits_and_recovers": "COLONYMIND_TEST_HERMES_PYTHON",
        "test_separate_interpreters_sidecar_has_no_hermes_and_worker_completes_archive": "COLONYMIND_TEST_SIDECAR_PYTHON",
    }
    markers = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]["markers"]
    assert {marker.split(":", 1)[0].split("(", 1)[0] for marker in markers} == {
        "requires_live_qualification", "requires_sidecar", "hardware_latency"}


def test_python_dependency_jobs_free_disk_before_installing():
    for name in ("python", "hermes-adapter"):
        steps = _workflow("ci.yml")["jobs"][name]["steps"]
        [cleanup] = [step for step in steps if step.get("name") == "Free disk space for Python dependencies"]
        assert cleanup["run"] == "sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc"
        install = next(step for step in steps if "pip install" in step.get("run", ""))
        assert steps.index(cleanup) < steps.index(install)


def _hermes_checkouts():
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text())
        for name, job in workflow["jobs"].items():
            for step in job.get("steps", []):  # a job that calls a reusable workflow has none
                options = step.get("with", {})
                if (str(step.get("uses", "")).startswith("actions/checkout@")
                        and options.get("repository") == "NousResearch/hermes-agent"):
                    yield path.name, workflow, name, job, options.get("ref")


def test_the_check_against_upstream_main_warns_and_never_gates():
    # One job follows the moving branch; every other Hermes checkout is a pinned commit or a published release.
    upstream_main = [found for found in _hermes_checkouts() if found[-1] == "main"]
    assert len(upstream_main) == 1
    _, workflow, _, job, _ = upstream_main[0]
    assert job.get("continue-on-error") is True
    # Scheduled or by hand only: no pull request, push or release waits on a branch that moves daily.
    # (PyYAML reads the bare key `on` as True.)
    assert set(workflow[True]) <= {"schedule", "workflow_dispatch"}
    assert any("pytest tests/hermes_adapter" in str(step.get("run", "")) for step in job["steps"])


def _repository():
    urls = {_project(path)["urls"]["Repository"] for path in PACKAGES}
    assert len(urls) == 1
    return urls.pop()


def _community_files():
    templates = ROOT / ".github" / "ISSUE_TEMPLATE"
    return [ROOT / "SECURITY.md", ROOT / "CODE_OF_CONDUCT.md", ROOT / "CONTRIBUTING.md",
            *sorted(templates.glob("*")), *sorted((ROOT / ".github").glob("*.md"))]


def test_security_reports_go_to_private_reporting_and_no_address_is_published():
    policy = (ROOT / "SECURITY.md").read_text()
    assert _repository() + "/security/advisories/new" in policy
    # The contributing guide sends reporters to the policy instead of naming a channel of its own.
    assert "SECURITY.md" in (ROOT / "CONTRIBUTING.md").read_text()
    published = {path.relative_to(ROOT).as_posix(): EMAIL.findall(path.read_text())
                 for path in _community_files() if path.is_file()}
    assert {path: found for path, found in published.items() if found} == {}


def test_the_code_of_conduct_says_where_to_report():
    conduct = (ROOT / "CODE_OF_CONDUCT.md").read_text()
    assert "Contributor Covenant" in conduct and "version 2.1" in conduct
    enforcement = conduct.split("## Enforcement\n", 1)[1].split("\n## ", 1)[0]
    assert "[INSERT" not in enforcement
    # The code covers incidents with no GitHub content to point at (an event, a direct message), and GitHub can limit
    # content reports to earlier contributors. The private reporting form takes any report from any account holder.
    assert _repository() + "/security/advisories/new" in enforcement
    # Reporting a comment, issue or pull request where it is stays as a shortcut.
    assert "Report content" in enforcement


ISSUE_TEMPLATES = ROOT / ".github" / "ISSUE_TEMPLATE"
FORM_ELEMENTS = {"markdown", "textarea", "input", "dropdown", "checkboxes"}


def _forms():
    return {path.name: yaml.safe_load(path.read_text())
            for path in sorted(ISSUE_TEMPLATES.glob("*.yml")) if path.name != "config.yml"}


def _fields(form):
    return {element["id"]: element for element in form["body"] if element["type"] != "markdown"}


def test_issue_forms_have_the_shape_github_accepts():
    """GitHub silently drops a malformed form from the new-issue page; these are the rules it enforces."""
    forms = _forms()
    assert set(forms) == {"bug_report.yml", "feature_request.yml"}
    for name, form in forms.items():
        assert form["name"] and form["description"], name
        assert {element["type"] for element in form["body"]} <= FORM_ELEMENTS, name
        assert any(element["type"] != "markdown" for element in form["body"]), name
        ids = [element["id"] for element in form["body"] if element["type"] != "markdown"]
        assert len(ids) == len(set(ids)) and all(re.fullmatch(r"[a-z0-9_-]+", i) for i in ids), name
        for element in form["body"]:
            assert element["attributes"]["value" if element["type"] == "markdown" else "label"], name


def test_the_bug_form_asks_for_the_doctor_report_and_the_versions(capsys, monkeypatch):
    fields = _fields(_forms()["bug_report.yml"])
    doctor = fields["doctor"]
    assert "colonymind doctor --json" in doctor["attributes"]["description"]
    assert doctor["attributes"]["render"] == "json"
    for field in ("doctor", "colonymind-version", "hermes-version", "python-version"):
        assert fields[field]["validations"]["required"] is True, field
    # The command the form asks for exists with that flag.
    from colonymind import cli
    monkeypatch.setattr(sys, "argv", ["colonymind", "doctor", "--help"])
    try:
        cli.main()
    except SystemExit as stop:
        assert stop.code == 0
    assert "--json" in capsys.readouterr().out


REDACTED = "<redacted>"


def test_the_bug_form_makes_the_reporter_redact_secrets_before_submitting():
    """What a reporter pastes is public, and the doctor report can quote a line of a file it cannot parse, key and
    all (the canary test below). So the form never calls that output clean: it asks for every secret to be replaced
    and will not submit until the reporter confirms it."""
    form = _forms()["bug_report.yml"]
    fields = _fields(form)
    for field in ("doctor", "logs"):
        assert REDACTED in fields[field]["attributes"]["description"], field
    confirmations = [option["label"] for element in form["body"] if element["type"] == "checkboxes"
                     for option in element["attributes"]["options"] if option.get("required") is True]
    assert [label for label in confirmations if REDACTED in label]
    # A private report is still read by people: the policy asks for the same before the doctor report goes in.
    versions = next(line for line in (ROOT / "SECURITY.md").read_text().split("\n- ") if "doctor --json" in line)
    assert REDACTED in versions


CANARY = "sk-canary-" + "7f3a9e0c51d24b86"


@pytest.mark.parametrize("broken", [
    pytest.param("hermes", id="hermes-config"),
    pytest.param("colonymind", id="colonymind-config"),
])
def test_doctor_output_never_carries_a_secret_from_a_file_it_cannot_parse(tmp_path, broken):
    """The bug form asks for `colonymind doctor --json`: run it as a reporter would, on an instance whose Hermes
    config and colonymind.yaml both hold a canary key, one of them on a line that does not parse."""
    home, instance, hermes, no_programs = (tmp_path / name for name in ("home", "instance", "hermes", "bin"))
    for directory in (home, instance, hermes, no_programs):
        directory.mkdir()
    files = {
        "hermes": (hermes / "config.yaml", ["model:", "  provider: custom", "  api_key: " + CANARY]),
        "colonymind": (instance / "colonymind.yaml", ["hermes:", f"  home: {json.dumps(str(hermes))}", "environment:",
                                                    "  COLONYMIND_EXAMPLE_API_KEY: " + CANARY]),
    }
    # A key pasted onto a line that already had one: the kind of slip that leaves a file unparseable.
    files[broken][1][-1] += ": pasted twice"
    for path, lines in files.values():
        path.write_text("\n".join(lines) + "\n")
    # No Hermes on the path and a sidecar address nothing listens on: the run reads files only.
    env = {"HOME": str(home), "PATH": str(no_programs), "COLONYMIND_HOME": str(instance), "LANG": "C.UTF-8"}
    run = subprocess.run([sys.executable, "-m", "colonymind", "doctor", "--json", "--url", "http://127.0.0.1:0",
                          "--timeout", "1"], env=env, capture_output=True, text=True, timeout=120, check=False)
    assert CANARY not in run.stdout + run.stderr
    report = json.loads(run.stdout)
    # Leaving the secret out must not leave the reporter guessing: the report still fails and names the line.
    unparseable = len(files[broken][1])
    assert report["ok"] is False
    assert [r for r in report["results"] if r["status"] == "fail" and f"line {unparseable}" in r["detail"]]


def test_the_new_issue_page_sends_security_reports_to_the_private_channel():
    config = yaml.safe_load((ISSUE_TEMPLATES / "config.yml").read_text())
    urls = [link["url"] for link in config["contact_links"]]
    assert _repository() + "/security/advisories/new" in urls


def test_the_pull_request_template_points_to_rules_that_exist():
    template = (ROOT / ".github" / "pull_request_template.md").read_text()
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    headings = re.findall(r'"(The [^"]+)"', template)
    assert headings
    for heading in headings:
        assert f"## {heading}\n" in contributing, heading
    assert "tests/hermes_adapter" in template and "python -m pytest" in template


def _compatibility_rows():
    rows = []
    for line in (ROOT / "docs" / "COMPATIBILITY.md").read_text().splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("|") and re.fullmatch(r"\d+\.\d+\.x", cells[0]):
            rows.append(cells)
    return rows


def test_the_compatibility_table_matches_what_this_tree_enforces():
    """The newest row separates the installer range from the two runtime environments.

    Between releases the tree carries the next release's code under the last release's version, so the newest row
    describes the tree, and the released series only needs a row of its own.
    """
    from colonymind.init import SUPPORTED_HERMES_RANGE

    rows = _compatibility_rows()
    versions = {_project(path)["version"] for path in PACKAGES}
    assert len(versions) == 1
    series = ".".join(versions.pop().split(".")[:2]) + ".x"
    assert series in [row[0] for row in rows], f"docs/COMPATIBILITY.md needs a row for {series}"
    newest = max(rows, key=lambda row: tuple(int(part) for part in row[0].split(".")[:2]))
    assert rows[0] is newest, "the newest release goes first"
    _, hermes, python, *_ = newest
    assert hermes.replace("`", "").replace(" ", "") == SUPPORTED_HERMES_RANGE
    adapter = tomllib.loads((ROOT / "pyproject.toml").read_text())
    manifest = yaml.safe_load((ROOT / "plugins/hermes-plugin/plugin.yaml").read_text())
    assert adapter["tool"]["colonymind"]["requires-hermes"] == SUPPORTED_HERMES_RANGE
    assert manifest["requires_hermes"] == SUPPORTED_HERMES_RANGE
    sidecar_python, hermes_python = python.split("; ")
    assert re.findall(r"3\.\d+", sidecar_python) == _supported_pythons()
    assert hermes_python == "Hermes: 3.14 from 0.21.7"


@pytest.mark.skipif(not all(map(shutil.which, ("bash", "awk", "sed", "grep"))), reason="the release step is a bash script")
def test_a_changelog_entry_from_the_template_becomes_the_release_page(tmp_path):
    """The Release workflow's own step, with a stand-in for `gh`, on a changelog written from the template."""
    template = (ROOT / ".github" / "RELEASE_NOTES_TEMPLATE.md").read_text()
    entry = template.split("-->", 1)[1].strip().replace("vX.Y.Z", "v9.8.7")
    heading = entry.splitlines()[0]
    assert re.fullmatch(r"## v9\.8\.7 - \S.*", heading)
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n" + entry + "\n\n## v9.8.6 - the release before\n\nOlder notes.\n")
    steps = _workflow("release.yml")["jobs"]["github-release"]["steps"]
    [script] = [step["run"] for step in steps if "CHANGELOG" in step.get("name", "")]
    stand_in = tmp_path / "bin" / "gh"
    stand_in.parent.mkdir()
    stand_in.write_text('#!/bin/sh\n[ "$1 $2" = "release view" ] && exit 1\nprintf \'%s\\n\' "$@" > "$GH_CALL"\n')
    stand_in.chmod(0o755)
    env = {**os.environ, "PATH": f"{stand_in.parent}{os.pathsep}{os.environ['PATH']}",
           "GITHUB_REF_NAME": "v9.8.7", "GH_TOKEN": "unused", "GH_CALL": str(tmp_path / "call")}
    subprocess.run(["bash", "-e", "-c", script], cwd=tmp_path, env=env, check=True)
    call = (tmp_path / "call").read_text().splitlines()
    assert call[:3] == ["release", "create", "v9.8.7"]
    assert call[call.index("--title") + 1] == heading.removeprefix("## ")
    notes = (tmp_path / "notes.md").read_text()
    assert notes.strip() and not notes.startswith("\n")
    upgrading = notes.split("### Upgrading\n", 1)[1]
    assert "uv tool upgrade colonymind" in upgrading
    assert "colonymind upgrade" in upgrading
    assert "Older notes" not in notes
