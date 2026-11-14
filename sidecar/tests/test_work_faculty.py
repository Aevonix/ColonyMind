"""Work faculty defaults, independent operation and quiet read-only reporting."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import yaml

from colonymind.config import DEFAULTS, load_config
from colonymind.mind.tick import DEFAULT_FACULTIES, faculties_of
from colonymind.reports import faculties, work
from colonymind.reports.instance import Instance
from colonymind.reports.registry import BY_KEY

NOW = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)


def test_registration_defaults_and_mind_independence(tmp_path):
    assert DEFAULTS["mind"]["faculties"]["work_records"] is True
    assert DEFAULT_FACULTIES["work_records"] is True
    assert faculties_of({})["work_records"] is True
    assert BY_KEY["work_records"].setting == "mind.faculties.work_records"
    assert BY_KEY["work_records"].arm == "all-on-work_records"
    assert "work_records" in faculties.WITHOUT_MIND
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump({"mind": {
        "enabled": False, "faculties": {"lessons": False}}}))
    instance = Instance(tmp_path)
    try:
        assert faculties._switch(instance, BY_KEY["work_records"]) == (True, None, None)
        assert load_config(tmp_path, environ={}).get("mind.faculties.work_records") is True
    finally:
        instance.close()


def test_disabled_factory_does_not_create_any_work_store(tmp_path, monkeypatch):
    import colonymind.work as package
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    (tmp_path / "colonymind.yaml").write_text("mind:\n  faculties:\n    work_records: false\n")
    monkeypatch.setattr(package, "_store", lambda *args: pytest.fail("disabled work initialized a store"))
    assert package.work_records_enabled() is False
    assert package.get_work_store() is None
    assert not (tmp_path / "colonymind-work.db").exists()


def read(tmp_path, monkeypatch, result, *, on=True):
    from colonymind.work import store
    seen = []

    def summary(path, **kwargs):
        seen.append((path, kwargs))
        return result

    monkeypatch.setattr(store, "read_summary", summary)
    section = faculties.Section(BY_KEY["work_records"], on=on)
    instance = SimpleNamespace(home=tmp_path)
    reading = SimpleNamespace(instance=instance, owner="owner-1", start=NOW - timedelta(days=2), end=NOW)
    work.faculty(reading, section)
    return section, seen


def test_report_counts_status_rates_and_unknown_kinds(tmp_path, monkeypatch):
    section, seen = read(tmp_path, monkeypatch, {
        "total": 4, "counts": {"proven": 1, "partial": 1, "unproven": 1, "failed": 1},
        "owner_corrected": 0,
        "unknown_effects": [{"tool": "external_call", "action": "invoke", "count": 2}],
    })
    assert section.activity == 4 and section.records is True
    assert section.lines == [
        "4 owner requests; 2.0 roots per day.",
        "Status rates: proven 25.0%, partial 25.0%, unproven 25.0%, failed 25.0%.",
        "Owner-corrected rate: 0.0%.",
        "Unknown effects: external_call invoke: 2.",
    ]
    assert seen == [(tmp_path / "colonymind-work.db", {"contact_id": "owner-1",
                     "since": (NOW - timedelta(days=2)).isoformat(), "until": NOW.isoformat()})]


def test_report_off_emits_no_metric_lines_and_performs_no_read(tmp_path, monkeypatch):
    section, seen = read(tmp_path, monkeypatch, {"total": 4}, on=False)
    assert section.lines == [] and section.activity == 0 and section.records is False
    assert seen == []


def test_empty_report_is_quiet(tmp_path, monkeypatch):
    section, _ = read(tmp_path, monkeypatch, {"total": 0})
    assert section.lines == ["No owner work records in this window."]
    assert section.wrong == [] and section.records is False


def test_real_report_reads_projection_without_changing_database_files(tmp_path):
    from colonymind.turns import TurnIdempotencyLedger
    from colonymind.work.store import WorkStore
    from test_work_api import retained_work
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    at = (NOW - timedelta(hours=2)).isoformat()
    retained_work(ledger, "root-1", at=at, operations=[{
        "operation_id": "op-1", "tool_call_id": "call-1", "tool": "write_file", "action": "write",
        "effect": "write", "resource_ref": "resource:one", "success": True, "source_refs": [],
        "evidence": {"kind": "file_hash", "method": "native_post_write_sha256", "verified": True}}])
    WorkStore(ledger).reconcile()

    def snapshot():
        return {path.name: (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes())
                for path in tmp_path.iterdir() if path.is_file()}

    before = snapshot()
    instance = SimpleNamespace(home=tmp_path)
    reading = SimpleNamespace(instance=instance, owner="owner-1", start=NOW - timedelta(days=1), end=NOW)
    section = faculties.Section(BY_KEY["work_records"], on=True)
    work.faculty(reading, section)
    assert "proven 100.0%" in section.lines[1]
    assert section.activity == 1
    assert snapshot() == before
