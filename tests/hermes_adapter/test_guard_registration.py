"""The registration record binds the exact guard and hooks to this process."""
import json
import logging
import os
from pathlib import Path
import stat
import sys
import time
from types import SimpleNamespace

import pytest

from guard_offline import offline_home, probe
from colonymind_hermes import guard, registration


def evidence():
    context = SimpleNamespace(register_hook=lambda name, callback: object())
    record = registration.RegistrationEvidence(context)
    record.register_hook("pre_tool_call", lambda **kw: None)
    return record


def test_register_writes_exact_checker_contract_without_private_content(offline_home):
    result = probe('''
import stat
from pathlib import Path
import psutil
from colonymind_hermes import guard
from colonymind_hermes.client import load_settings
path = Path(os.environ["COLONYMIND_HOME"]) / "hermes-plugin-registration.json"
value = json.loads(path.read_text())
metadata = path.stat()
emit(fields=sorted(value), schema=value["schema"], mode=stat.S_IMODE(metadata.st_mode),
     single_link=metadata.st_nlink == 1, owner=metadata.st_uid == os.getuid(),
     pid=value["pid"] == os.getpid(), start=abs(value["process_start"] - psutil.Process().create_time()) <= 2,
     interpreter=value["interpreter"] == sys.executable,
     module=value["module_path"] == str(Path(guard.__file__).resolve()),
     hooks=value["hooks"] == sorted(set(loaded.hooks_registered)),
     written=abs(value["written_at"] - time.time()) < 10,
     private_absent=load_settings().api_key not in path.read_text())
''', offline_home, env={"COLONYMIND_BODY_THREAD": "0"})
    assert result.pop("fields") == sorted(("schema", "pid", "process_start", "module_path",
                                           "interpreter", "hooks", "written_at"))
    assert result.pop("schema") == 1 and result.pop("mode") == 0o600
    assert all(result.values())


@pytest.mark.parametrize("existing", ["regular", "symlink", "hardlink"])
def test_atomic_replacement_never_follows_existing_links(tmp_path, monkeypatch, existing):
    path = tmp_path / registration.REGISTRATION_FILE
    foreign = tmp_path / "foreign"
    foreign.write_text("private-canary-never-log")
    if existing == "regular":
        path.write_text("old evidence")
        path.chmod(0o644)
    elif existing == "symlink":
        path.symlink_to(foreign)
    else:
        os.link(foreign, path)
    previous = path.read_text()
    replace = registration.os.replace
    observed = []
    def atomic_replace(source, target):
        metadata = Path(source).stat()
        assert path.read_text() == previous
        assert stat.S_IMODE(metadata.st_mode) == 0o600 and metadata.st_nlink == 1
        assert json.loads(Path(source).read_text())["schema"] == 1
        observed.append(True)
        return replace(source, target)
    monkeypatch.setattr(registration.os, "replace", atomic_replace)
    evidence().write(tmp_path)
    assert observed == [True]
    assert foreign.read_text() == "private-canary-never-log"
    assert not path.is_symlink() and path.stat().st_nlink == 1
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text())["module_path"] == str(Path(guard.__file__).resolve())
    assert list(tmp_path.glob(".hermes-plugin-registration-*")) == []


def test_failed_write_warns_without_leaking_and_keeps_old_record(tmp_path, monkeypatch, caplog):
    path = tmp_path / registration.REGISTRATION_FILE
    path.write_text("old evidence")
    def unavailable(*a):
        raise OSError("private-canary-never-log")
    monkeypatch.setattr(registration.os, "replace", unavailable)
    with caplog.at_level(logging.WARNING):
        evidence().write(tmp_path)
    assert path.read_text() == "old evidence"
    assert "registration_evidence_unavailable exception_type=OSError" in caplog.text
    assert "private-canary-never-log" not in caplog.text
    assert list(tmp_path.glob(".hermes-plugin-registration-*")) == []


def test_only_successful_hook_registration_is_recorded():
    def reject(*a):
        raise ValueError("private-canary-never-log")
    record = registration.RegistrationEvidence(SimpleNamespace(register_hook=reject))
    with pytest.raises(ValueError):
        record.register_hook("pre_tool_call", lambda: None)
    assert not record.hooks


def test_ignored_registration_cannot_publish_evidence(tmp_path, monkeypatch, caplog):
    record = registration.RegistrationEvidence(SimpleNamespace(register_hook=lambda *a: None))
    record.register_hook("pre_tool_call", lambda: None)
    def unexpected():
        raise AssertionError("ignored registration attempted process lookup")
    monkeypatch.setattr(registration, "process_start", unexpected)
    with caplog.at_level(logging.WARNING):
        record.write(tmp_path)
    assert not record.hooks and not (tmp_path / registration.REGISTRATION_FILE).exists()
    assert "reason=guard_hook_unregistered" in caplog.text
