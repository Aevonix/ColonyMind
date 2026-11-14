"""Distinguish Hermes' dependency conflicts from packages the adapter installs."""
from pathlib import Path

import pytest

from colonymind import doctor, init


@pytest.mark.parametrize("conflict,status", [
    ("The package `browser-harness` requires `pillow==12.3.0`, but `13.0.0` is installed", doctor.WARN),
    ("browser-harness 1.0 has requirement pillow==12.3.0, but you have pillow 13.0.0.", doctor.WARN),
    ("browser-harness 1.0 requires pillow, which is not installed.", doctor.WARN),
    ("The package `colonymind-hermes` requires `httpx<0.29`, but `0.30` is installed", doctor.FAIL),
    ("other 1.0 has requirement httpx==0.30, but you have httpx 0.28.1.", doctor.FAIL),
    ("other 1.0 has requirement anyio==1, but you have anyio 4.0.", doctor.FAIL),
    ("The package `PyYAML` requires `missing>=1`, but it's not installed", doctor.FAIL),
    ("unrecognized package-manager error", doctor.FAIL),
])
def test_dependency_conflict_ownership(monkeypatch, conflict, status):
    python = Path("/synthetic/hermes/bin/python")
    monkeypatch.setattr(doctor, "_hermes_python", lambda: python)
    monkeypatch.setattr(init, "pip_check", lambda _: (False, "Using Python 3.14 environment at: /synthetic\n"
                                                     "Checked 104 packages in 32ms\nFound 1 incompatibility\n" + conflict))
    # The actual metadata query walks the adapter's active dependency closure.
    monkeypatch.setattr(init, "adapter_dependency_names", lambda _: {"colonymind-hermes", "httpx", "pyyaml", "anyio"},
                        raising=False)
    result = doctor.check_pip_check()
    assert result.status == status
    assert conflict in result.detail
    assert "Using Python" not in result.detail
    if status == doctor.WARN:
        assert "Hermes' own environment" in result.detail
        assert "reinstall Hermes" not in result.remedy


def test_mixed_conflicts_fail(monkeypatch):
    monkeypatch.setattr(doctor, "_hermes_python", lambda: Path("/synthetic/python"))
    conflicts = ("other 1.0 has requirement pillow==1, but you have pillow 2.\n"
                 "other 1.0 has requirement httpx==1, but you have httpx 2.")
    monkeypatch.setattr(init, "pip_check", lambda _: (False, conflicts))
    monkeypatch.setattr(init, "adapter_dependency_names", lambda _: {"colonymind-hermes", "httpx"}, raising=False)
    result = doctor.check_pip_check()
    assert result.status == doctor.FAIL
    assert all(line in result.detail for line in conflicts.splitlines())
