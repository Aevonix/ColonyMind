"""The installer publishes shell forwarders and selects dependencies per home."""
import hashlib
import json
import shlex
from pathlib import Path

import pytest

from colonymind import doctor, init


@pytest.fixture
def installer(tmp_path, monkeypatch):
    home = tmp_path / "owner's home" / ".hermes"
    root = home / "hermes-agent"
    launcher = root / ".hermes/bin/hermes"
    launcher.parent.mkdir(parents=True)
    store = home / "tools/python-3.14.9/bin/python3"
    store.parent.mkdir(parents=True)
    store.touch()
    bootstrap = f"import sys\nsys.path.insert(0, {str(root)!r})\nimport hermes_bootstrap\n"
    launcher.write_text("#!/bin/sh\nexec " + shlex.join([str(store), "-I", "-c", bootstrap]) + ' "$@"\n')
    public = tmp_path / "bin/hermes"
    public.parent.mkdir()
    public.write_text("#!/bin/sh\nexec " + shlex.quote(str(launcher)) + ' "$@"\n')
    monkeypatch.setattr(init, "find_hermes_executable", lambda: public)
    monkeypatch.setenv("HOME", str(home.parent))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr(init, "hermes_python_version", lambda _: "0.22.7")
    monkeypatch.setattr(init, "_run", lambda *a, **kw: pytest.fail("discovery executed a launcher"))

    def environment(selected_home, generation="a" * 32):
        key = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]
        state = selected_home / "installs" / key
        venv = state / "environments" / generation / "venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text(f"home = {store.parent}\n")
        python = venv / "bin/python"
        python.touch()
        (state / "facts.json").write_text(json.dumps({"packages": {"venv": {"environment": str(venv)}}}))
        (state / "inputs").mkdir(exist_ok=True)
        (state / "inputs/.project-root").write_text(str(root))
        return python

    return home, public, environment


def test_default_home_record_without_launching(installer):
    home, _, environment = installer
    python = environment(home)
    assert init.resolve_hermes_python(None, None) == (python, "0.22.7")


def test_active_home_wins_over_saved_generation(installer, tmp_path, monkeypatch):
    home, _, environment = installer
    stale = environment(home)
    active = tmp_path / "other-home"
    python = environment(active)
    monkeypatch.setenv("HERMES_HOME", str(active))
    assert init.resolve_hermes_python(None, str(stale))[0] == python


def test_explicit_python_still_wins(installer, tmp_path):
    home, _, environment = installer
    environment(home)
    explicit = tmp_path / "explicit-python"
    explicit.touch()
    assert init.resolve_hermes_python(str(explicit), None)[0] == explicit


def test_missing_record_uses_only_environment_query(installer, monkeypatch):
    home, public, environment = installer
    calls = []

    def query(command, **kwargs):
        calls.append(command)
        assert command == [str(public), "--run-module", "pm.environments", "--format", "json"]
        environment(home)
        from subprocess import CompletedProcess
        return CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr(init, "_run", query)
    assert init.resolve_hermes_python(None, None)[0].name == "python"
    assert len(calls) == 1


def test_only_init_and_upgrade_have_hermes_provision_a_new_home(installer, tmp_path, monkeypatch):
    """A home with nothing recorded: every reader goes on without an environment and runs nothing (the fixture
    fails the test when discovery executes a launcher); ``resolve_hermes_python`` (init, upgrade) asks Hermes."""
    home, public, environment = installer
    saved = tmp_path / "saved-python"
    saved.touch()
    assert init.installer_python(public, home) is None
    assert init.python_of_executable(public) is None
    assert init.hermes_python_candidates(None, None, home) == []
    assert init.hermes_python_candidates(None, str(saved), home) == [saved]
    # The default home's record is no answer for another home.
    environment(home)
    other = tmp_path / "other-home"
    assert init.hermes_python_candidates(None, None, other) == []
    calls = []

    def refused(command, **kwargs):
        from subprocess import CompletedProcess
        calls.append((command, kwargs["env"]["HERMES_HOME"]))
        return CompletedProcess(command, 1, "", "no environment")

    monkeypatch.setattr(init, "_run", refused)
    with pytest.raises(init.InitError, match="no Hermes dependency environment recorded"):
        init.resolve_hermes_python(None, None, other)
    assert calls == [([str(public), "--run-module", "pm.environments", "--format", "json"], str(other))]


def _instance(tmp_path, monkeypatch, hermes_home):
    import yaml

    instance = tmp_path / "instance"
    instance.mkdir()
    (instance / "colonymind.yaml").write_text(yaml.safe_dump({"hermes": {"home": str(hermes_home)}}))
    monkeypatch.setenv("COLONYMIND_HOME", str(instance))
    return instance


def test_doctor_reads_the_record_and_goes_on_without_it(installer, tmp_path, monkeypatch):
    active = tmp_path / "active"
    active.mkdir()
    _instance(tmp_path, monkeypatch, active)
    assert doctor._hermes_python() is None
    result = doctor.check_hermes_version()
    assert result.status == doctor.FAIL and result.detail == "no Hermes interpreter found"


def test_uninstall_reads_the_record_and_goes_on_without_it(installer, tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    _, _, environment = installer
    active = tmp_path / "active"
    worker = active / "profiles" / init.WORKER_PROFILE
    worker.mkdir(parents=True)
    (active / "config.yaml").write_text("memory:\n  provider: colonymind-memory\n")
    instance = _instance(tmp_path, monkeypatch, active)
    monkeypatch.setattr(init, "_mind_off", lambda cfg: None)
    monkeypatch.setattr(init, "uninstall_service", lambda cfg: "sidecar service was not installed")
    args = SimpleNamespace(home=str(instance), hermes_home=None, hermes_python=None)
    assert init.run_uninstall(args) == 0
    output = capsys.readouterr().out
    assert f"no Hermes interpreter found for {active}" in output and "is left where it is" in output
    assert "stock again" in output and not worker.exists()
    # A record that cannot be read is named, and the uninstall still ends.
    selected = environment(active)
    (selected.parents[4] / "facts.json").write_text(json.dumps({"packages": {"venv": {"environment": "/elsewhere"}}}))
    assert init.run_uninstall(args) == 0
    output = capsys.readouterr().out
    assert "cannot read Hermes dependency environment" in output and "is left where it is" in output


def test_invalid_record_never_falls_back_to_other_home(installer, tmp_path, monkeypatch):
    home, _, environment = installer
    stale = environment(home)
    active = tmp_path / "other-home"
    selected = environment(active)
    record = selected.parents[4] / "facts.json"
    record.write_text(json.dumps({"packages": {"venv": {"environment": str(stale.parent.parent)}}}))
    monkeypatch.setenv("HERMES_HOME", str(active))
    with pytest.raises(init.InitError, match="dependency environment"):
        init.resolve_hermes_python(None, str(stale))


def test_source_console_script_keeps_its_python(tmp_path):
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    launcher = python.with_name("hermes")
    launcher.write_text(f"#!{python}\n")
    assert init.python_of_executable(launcher) == python


def test_doctor_reports_configured_but_absent_in_active_home(installer, tmp_path, monkeypatch):
    from colonymind.config import load_config
    import yaml

    home, _, environment = installer
    stale = environment(home)
    active = tmp_path / "active"
    python = environment(active)
    instance = tmp_path / "instance"
    instance.mkdir()
    (instance / "colonymind.yaml").write_text(yaml.safe_dump({"hermes": {"home": str(active), "python": str(stale)}}))
    (active / "config.yaml").write_text("memory:\n  provider: colonymind-memory\n")
    monkeypatch.setenv("COLONYMIND_HOME", str(instance))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    seen = []

    def installed(target):
        seen.append(target)
        return None if target == python else "1.0.0"

    monkeypatch.setattr(init, "installed_adapter_version", installed)
    result = doctor.check_adapter_version()
    assert seen == [python]
    assert result.status == doctor.FAIL
    assert "configured but not installed" in result.detail
    assert str(python) in result.detail and str(active) in result.detail
    assert "colonymind init --hermes-home" in result.remedy and str(active) in result.remedy
