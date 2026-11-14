"""The senses switch, bounded configuration and quiet, read-only faculty report."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import yaml

from colonymind.config import ConfigError, load_config
from colonymind.reports.registry import BY_KEY, FACULTIES

NOW = datetime(2026, 9, 20, 12, tzinfo=timezone.utc)


def test_senses_defaults_and_one_setting_benchmark_arm(tmp_path):
    config = load_config(tmp_path, environ={})
    assert config.get("mind.faculties.senses") is True
    assert config.get("mind.observations") == {
        "expiry_hours": 2, "daily_source_cap": 6, "image_max_bytes": 512 * 1024, "source_caps": {"omi": 40},
    }
    faculty = BY_KEY["senses"]
    assert faculty.setting == "mind.faculties.senses"
    assert faculty.arm == "all-on-senses"
    assert [item.key for item in FACULTIES][-3:] == ["drift_check", "senses", "hands"]


def test_senses_configuration_can_reduce_limits_and_disable_the_faculty(tmp_path):
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump({"mind": {
        "faculties": {"senses": False},
        "observations": {"expiry_hours": 0.5, "daily_source_cap": 0, "image_max_bytes": 1024},
    }}))
    config = load_config(tmp_path, environ={})
    assert config.get("mind.faculties.senses") is False
    assert config.get("mind.observations.expiry_hours") == 0.5
    assert config.get("mind.observations.daily_source_cap") == 0
    assert config.get("mind.observations.image_max_bytes") == 1024


@pytest.mark.parametrize("value", [None, [], "off"])
def test_observation_configuration_requires_a_mapping(tmp_path, value):
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump({"mind": {"observations": value}}))
    with pytest.raises(ConfigError, match="mind.observations must be a mapping"):
        load_config(tmp_path, environ={})


@pytest.mark.parametrize("key,value", [
    ("expiry_hours", 0), ("expiry_hours", -1), ("expiry_hours", True),
    ("expiry_hours", float("nan")), ("expiry_hours", float("inf")), ("expiry_hours", "two"),
    ("daily_source_cap", -1), ("daily_source_cap", 1.5), ("daily_source_cap", True),
    ("image_max_bytes", 0), ("image_max_bytes", True), ("image_max_bytes", 1024.5),
    ("image_max_bytes", 512 * 1024 + 1),
])
def test_observation_limits_refuse_unsafe_values(tmp_path, key, value):
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump({"mind": {"observations": {key: value}}}))
    with pytest.raises(ConfigError, match=f"mind.observations.{key}"):
        load_config(tmp_path, environ={})


def test_observation_routes_use_source_identifiers_and_known_labels(tmp_path):
    routes = {"device:one": "ignore", "device/two": "note", "device-three": "look", "device.four": "tell"}
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump({"mind": {"observations": {"routes": routes}}}))
    assert load_config(tmp_path, environ={}).get("mind.observations.routes") == routes


@pytest.mark.parametrize("routes", [
    None, [], "note", {"": "note"}, {"device one": "note"}, {"/device": "note"},
    {True: "note"}, {"d" * 129: "note"}, {"device": "run"}, {"device": None}, {"device": []},
])
def test_observation_routes_refuse_invalid_sources_or_actions(tmp_path, routes):
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump({"mind": {"observations": {"routes": routes}}}))
    with pytest.raises(ConfigError, match="mind.observations.routes"):
        load_config(tmp_path, environ={})


def _read(tmp_path, monkeypatch, rows, *, on=True):
    from colonymind.mind import observations
    from colonymind.reports import senses
    from colonymind.reports.faculties import Section
    from colonymind.reports.instance import Instance

    # A real read-only store checks that reporting does not initialize or migrate it.
    import sqlite3
    with sqlite3.connect(tmp_path / "mind.db") as conn:
        conn.execute("CREATE TABLE fixture (value TEXT)")
    instance = Instance(tmp_path)
    start = NOW - timedelta(days=1)
    seen = []

    def report_rows(conn, *, start, end):
        seen.append((start, end))
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        assert [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")] == ["fixture"]
        return rows

    monkeypatch.setattr(observations, "report_rows", report_rows)
    section = Section(faculty=BY_KEY["senses"], on=on)
    try:
        senses.read(SimpleNamespace(instance=instance, start=start, end=NOW), section)
    finally:
        instance.close()
    assert seen == [(start, NOW)]
    return section


def test_report_lists_surprises_concerns_and_actual_messages_per_source(tmp_path, monkeypatch):
    _message(tmp_path, "device-a")
    section = _read(tmp_path, monkeypatch, [
        {"source": "device-b", "surprises": 2, "concerns": 1, "messages": 0, "ignored": 0},
        {"source": "device-a", "surprises": 3, "concerns": 2, "messages": 1, "ignored": 0},
    ])
    assert section.lines == [
        "device-a: 3 surprises, 2 concerns and 1 message.",
        "device-b: 2 surprises, 1 concern and 0 messages.",
    ]
    assert section.activity == 9
    assert section.records is True
    assert section.wrong == []


def _message(home, source, *, status="sent", at=NOW - timedelta(hours=1), kind="message", type="percept_tell"):
    from colonymind.initiatives.store import InitiativeStore
    store = InitiativeStore(state_dir=home)
    try:
        row, _ = store.create_intention(
            kind=kind, type=type, title="Review observation", drive="curiosity", cls="internal",
            decision="act", decision_reason="fixture", status="approved", dedup_key=None,
            context={"observation_source": source}, created_at=NOW - timedelta(days=2),
        )
        store.transition(row.id, status, action=status, at=at, completed_at=at if status == "sent" else None)
    finally:
        store.close()


def test_messages_count_delivery_in_the_window_not_attempts_or_state_counters(tmp_path, monkeypatch):
    _message(tmp_path, "device-a")
    _message(tmp_path, "device-c")  # A source whose observation has expired still has a delivery record.
    for status in ("proposed", "approved", "sending", "uncertain", "dropped", "failed"):
        _message(tmp_path, "device-a", status=status)
    _message(tmp_path, "device-a", at=NOW - timedelta(days=2))
    _message(tmp_path, "device-a", at=NOW)  # The report's end is exclusive.
    _message(tmp_path, "device-a", kind="task")
    _message(tmp_path, "device-a", type="other_message")
    _message(tmp_path, None)
    section = _read(tmp_path, monkeypatch, [
        {"source": "device-a", "surprises": 1, "concerns": 1, "messages": 99, "ignored": 0},
    ])
    assert section.lines == [
        "device-a: 1 surprise, 1 concern and 1 message.",
        "device-c: 0 surprises, 0 concerns and 1 message.",
    ]


def test_missing_sources_are_quiet_and_not_an_error(tmp_path, monkeypatch):
    section = _read(tmp_path, monkeypatch, [])
    assert section.lines == ["No outside sources posted observations in this window."]
    assert section.activity == 0
    assert section.records is False
    assert section.wrong == []


def test_off_report_says_posts_are_accepted_and_ignored(tmp_path, monkeypatch):
    section = _read(tmp_path, monkeypatch, [
        {"source": "device-a", "surprises": 0, "concerns": 0, "messages": 0, "ignored": 4},
    ], on=False)
    assert "With senses off, posts are accepted and ignored." in section.lines
    assert "device-a: 0 surprises, 0 concerns and 0 messages; 4 observations ignored." in section.lines
    assert section.activity == 0
    assert section.wrong == []
