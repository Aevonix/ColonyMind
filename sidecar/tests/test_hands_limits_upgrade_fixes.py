"""B4 review regressions: tolerant reads, strict writes and isolated maintenance."""

from copy import deepcopy
from datetime import timedelta
import re
import sqlite3

import pytest
import yaml

from colonymind.config import DEFAULTS, ConfigError, dump_config, load_config, save_config, update_config, validate
from colonymind.contacts.config import ContactsConfig
from colonymind.hands.controller import Controller
from colonymind.hands.retention import RETENTION_BATCH_SIZE
from colonymind.removed_settings import REMOVED_CONTACT_ARGUMENTS, REMOVED_SETTINGS, removed_settings, upgrade_message
from test_hands_controller import c, plan  # noqa: F401
from test_hands_cleanup_limits import settle
from test_work_hands_refresh import setup_hands
from test_work_projection import stores  # noqa: F401


def old_init_config():
    data = deepcopy(DEFAULTS)
    for path in REMOVED_SETTINGS:
        _, group, key = path.split(".")
        data["hands"].setdefault(group, {})[key] = 1
    return data


async def test_old_init_file_loads_and_controller_starts(tmp_path, caplog):
    data = old_init_config()
    path = tmp_path / "colonymind.yaml"
    path.write_text(yaml.safe_dump(data))
    before = path.read_bytes()
    cfg = load_config(tmp_path, environ={})
    assert not removed_settings(cfg.data)
    assert len(caplog.records) == len(REMOVED_SETTINGS)
    assert {r.message for r in caplog.records} == {upgrade_message([key]) for key in REMOVED_SETTINGS}
    assert path.read_bytes() == before
    caplog.clear()
    controller = Controller(tmp_path / "hands.db", config=data["hands"])
    assert len(caplog.records) == len(REMOVED_SETTINGS)
    assert not removed_settings({"hands": controller.config})
    assert removed_settings(data) == list(REMOVED_SETTINGS)
    await controller.connect()
    try:
        assert (await controller.sweep())["expired"] == []
    finally:
        await controller.close()


@pytest.mark.parametrize("path", REMOVED_SETTINGS)
def test_retired_writes_are_refused_with_read_warning_message(path, tmp_path):
    save_config(deepcopy(DEFAULTS), tmp_path)
    data = deepcopy(DEFAULTS)
    _, group, key = path.split(".")
    data["hands"].setdefault(group, {})[key] = 1
    before = (tmp_path / "colonymind.yaml").read_bytes()
    for write in (lambda: update_config({"hands": {group: {key: 1}}}, tmp_path),
                  lambda: save_config(data, tmp_path), lambda: dump_config(data)):
        with pytest.raises(ConfigError) as caught:
            write()
        assert str(caught.value) == upgrade_message([path])
        assert (tmp_path / "colonymind.yaml").read_bytes() == before


@pytest.mark.parametrize("key", ["events_days", "rows_days"])
@pytest.mark.parametrize("value", [0, 1])
def test_retention_minimum_is_one_day(key, value):
    data = deepcopy(DEFAULTS)
    data["hands"]["retention"][key] = value
    if value:
        validate(data)
        Controller(":memory:", config=data["hands"])
    else:
        with pytest.raises(ConfigError, match="at least 1 day"):
            validate(data)
        with pytest.raises(ValueError, match="at least 1 day"):
            Controller(":memory:", config=data["hands"])


@pytest.mark.parametrize("path", REMOVED_CONTACT_ARGUMENTS)
def test_contacts_rejection_is_constructor_only(path, tmp_path):
    _, field = path.split(".")
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump({"contacts": {field: None}}))
    assert load_config(tmp_path, environ={}).data["contacts"][field] is None
    with pytest.raises(ValueError, match=re.escape(path)):
        ContactsConfig(**{field: None})


@pytest.mark.parametrize("operation", ["_overdue_obligations", "_prune_retained"])
async def test_maintenance_failure_keeps_expiry_and_admission(c, monkeypatch, caplog, operation):
    reserved = await plan(c, "expires")
    c.config["max_running"] = 0
    queued = await plan(c, "queued")
    assert queued["admit"] == "queued"
    c.config["max_running"] = 2
    c.test_time[0] += timedelta(seconds=c.config["lease_seconds"] + 1)

    async def broken():
        await c.execute("INSERT INTO maintenance VALUES ('broken',?, '{}')", (c.now(),))
        raise ValueError("bad maintenance row")

    monkeypatch.setattr(c, operation, broken)
    result = await c.sweep()
    assert reserved["hand_id"] in result["expired"]
    assert queued["hand_id"] in result["reserved"]
    assert (await c.get_hand(reserved["hand_id"]))["state"] == "expired"
    assert (await c.get_hand(queued["hand_id"]))["state"] == "reserved"
    assert await c.one("SELECT id FROM maintenance WHERE id='broken'") is None
    assert "core sweep continues" in caplog.text


async def test_retention_batches_uses_index_and_persists_hourly_schedule(c):
    c.config["retention"].update(events_days=1, rows_days=10)
    p = await settle(c, "closed")
    for n in range(RETENTION_BATCH_SIZE + 2):
        await c.execute("INSERT INTO events(hand_id,seq,at,type,payload,source,key) VALUES (?,?,?,?,?,?,?)",
                        (p["hand_id"], 1000 + n, c.now(), "progress", "{}", "test", str(n)))
    await c.store.db.commit()
    c.test_time[0] += timedelta(days=1)
    assert (await c.sweep())["retention"]["events"] == 0
    at = (await c.retention_status())["at"]
    c.test_time[0] += timedelta(seconds=3599)
    assert (await c.sweep())["retention"]["events"] == 0
    assert (await c.retention_status())["at"] == at
    await c.close()
    await c.connect()
    c.test_time[0] += timedelta(seconds=1)
    assert (await c.sweep())["retention"]["events"] == RETENTION_BATCH_SIZE
    assert (await c.one("SELECT count(*) AS n FROM events"))["n"] > 0
    with sqlite3.connect(c.store.path) as connection:
        query_plan = connection.execute("EXPLAIN QUERY PLAN SELECT rowid FROM events WHERE at < ? ORDER BY at LIMIT ?",
                                        (c.now(), RETENTION_BATCH_SIZE)).fetchall()
    assert any("SEARCH" in row[3] and "idx_events_retention" in row[3] for row in query_plan)
    c.test_time[0] += timedelta(hours=1)
    assert (await c.sweep())["retention"]["events"] > 0
    assert (await c.one("SELECT count(*) AS n FROM events"))["n"] == 0


async def test_task_retention_preserves_origin_refresh_until_retry(c, stores, monkeypatch):
    # Use a real Work root and place the controller beside its ledger.
    setup_hands(stores)
    controller = Controller(stores.path / "hands.db", config=c.config, clock=c.clock)
    await controller.connect()
    try:
        p = await settle(controller, "closed")
        controller.config["retention"].update(events_days=10, rows_days=1)
        c.test_time[0] += timedelta(days=1)
        assert (await controller.sweep())["retention"]["tasks"] == 0
        c.test_time[0] += timedelta(hours=1)

        def unavailable(*args):
            raise OSError("projection unavailable")

        monkeypatch.setattr("colonymind.work.refresh_work", unavailable)
        assert (await controller.sweep())["retention"]["tasks"] == 1
        assert await controller.one("SELECT id FROM tasks WHERE id=?", (p["task_id"],)) is None
        pending = await controller.one("SELECT origin_ref FROM work_refresh_origins")
        assert pending["origin_ref"]["message_id"] == "message-1"
        refreshed = []
        monkeypatch.setattr("colonymind.work.refresh_work", lambda ledger, roots: refreshed.extend(roots))
        await controller._refresh_work()
        assert refreshed == ["root"]
        assert await controller.one("SELECT origin_ref FROM work_refresh_origins") is None
    finally:
        await controller.close()
