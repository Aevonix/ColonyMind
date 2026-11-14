"""B4 gates: exact bounds, durable refusals, safe expiry and real setting readers."""
import logging

import ast
import asyncio
from copy import deepcopy
from dataclasses import fields
from datetime import timedelta
from pathlib import Path
import re

import pytest
from fastapi import HTTPException

from colonymind.config import ConfigError, DEFAULTS, load_config, validate
from colonymind.contacts.config import ContactsConfig
from colonymind.hands import controller, sessions, wire
from colonymind.removed_settings import REMOVED_ENVIRONMENT, REMOVED_SETTINGS
from colonymind.vector.config import EmbeddingConfig, VectorStoreConfig
from test_hands_controller import c, finish, plan, running, REF


async def partial(c, key, **task):
    p = await running(c, key, **task)
    await finish(c, p, status="partial")
    return p


async def assert_refusal(c, p, reason):
    assert p["admit"] == "refuse" and p["reason"] == reason and p["lease_token"] is None
    hand = await c.get_hand(p["hand_id"])
    assert hand["state"] == "refused" and hand["outcome"] == reason
    event = await c.one("SELECT payload FROM events WHERE hand_id=? AND type='refused'", (p["hand_id"],))
    assert event["payload"]["reason"] == reason
    listed = await c.list_hands(wire.ListHandsRequest(state="refused"))
    assert any(h["id"] == p["hand_id"] and h["outcome"] == reason for h in listed["hands"])
    notice = await c.one("SELECT line FROM notices WHERE hand_id=? AND budget='actionable'", (p["hand_id"],))
    assert reason in notice["line"]


async def test_cascade_task_cap_allows_last_slot_then_records_refusal(c):
    c.config["cascade"].update(max_per_task=2, max_per_hour=100)
    root = await partial(c, "initial")
    for n in range(2):
        await partial(c, f"continue-{n}", continues=root["task_id"])
    assert (await c.get_task(root["task_id"]))["usage"]["cascades"] == 2
    refused = await plan(c, "past-cap", continues=root["hand_id"])
    await assert_refusal(c, refused, "cascade_task_limit")
    assert await plan(c, "past-cap", continues=root["hand_id"]) == refused
    await c.close()
    await c.connect()
    await c.revise(root["task_id"], wire.ReviseRequest(goal="A revised artifact"), "revision")
    await assert_refusal(c, await plan(c, "after-restart", continues=root["task_id"]), "cascade_task_limit")
    assert (await c.get_task(root["task_id"]))["usage"]["cascades"] == 2


async def test_cascade_hour_cap_is_shared_and_exactly_one_hour_expires(c):
    c.config["cascade"].update(max_per_task=100, max_per_hour=1)
    a, b = await partial(c, "root-a"), await partial(c, "root-b")
    await partial(c, "at-cap", continues=a["task_id"], name="cascade-a")
    await assert_refusal(c, await plan(c, "past-cap", continues=b["task_id"], name="cascade-b"), "cascade_hour_limit")
    c.test_time[0] += timedelta(seconds=3599)
    await assert_refusal(c, await plan(c, "still-capped", continues=b["task_id"], name="cascade-b"), "cascade_hour_limit")
    c.test_time[0] += timedelta(seconds=1)
    assert (await plan(c, "hour-boundary", continues=b["task_id"], name="cascade-b"))["admit"] == "start"


async def test_cascade_hour_cap_serializes_independent_connections(c):
    c.config["cascade"].update(max_per_task=100, max_per_hour=1)
    a, b = await partial(c, "root-a"), await partial(c, "root-b")
    peer = controller.Controller(c.store.path, config=c.config, clock=c.clock)
    await peer.connect()
    try:
        peer.health.record("family-a", running=0, waiting=0, kv=0.1)
        results = await asyncio.gather(plan(c, "concurrent-a", continues=a["task_id"], name="cascade-a"),
                                       plan(peer, "concurrent-b", continues=b["task_id"], name="cascade-b"))
        assert sorted(p["admit"] for p in results) == ["refuse", "start"]
        await assert_refusal(c, next(p for p in results if p["admit"] == "refuse"), "cascade_hour_limit")
    finally:
        await peer.close()


async def test_queued_cascade_is_charged_once_and_replay_never_recharges(c):
    c.config["cascade"].update(max_per_task=1, max_per_hour=1)
    root = await partial(c, "initial")
    c.config["max_running"] = 0
    queued = await plan(c, "queue", continues=root["task_id"])
    assert queued["admit"] == "queued"
    assert await plan(c, "queue", continues=root["task_id"]) == queued
    c.config["max_running"] = 2
    assert queued["hand_id"] in (await c.sweep())["reserved"]
    assert (await c.get_task(root["task_id"]))["usage"]["cascades"] == 1
    assert (await c.one("SELECT count(*) AS n FROM events WHERE type='cascade_admitted'"))["n"] == 1


async def test_existing_continuations_count_when_limit_first_takes_effect(c):
    root = await partial(c, "initial")
    old = await partial(c, "old", continues=root["task_id"])
    task = await c.get_task(root["task_id"])
    hand = await c.get_hand(old["hand_id"])
    # State from a predecessor that had no cascade accounting.
    await c.update("tasks", "id", task["id"], usage={k: v for k, v in task["usage"].items() if k != "cascades"})
    await c.update("hands", "id", hand["id"], decisions={k: v for k, v in hand["decisions"].items() if k != "cascade_admitted_at"})
    await c.store.db.commit()
    c.config["cascade"]["max_per_task"] = 1
    await assert_refusal(c, await plan(c, "new", continues=task["id"]), "cascade_task_limit")


async def test_overdue_at_cap_is_allowed_past_cap_is_recorded_and_visible(c):
    c.config["obligations"]["overdue_seconds"] = 60
    p = await running(c)
    oid = (await c.get_task(p["task_id"]))["obligation_id"]
    c.test_time[0] += timedelta(seconds=60)
    assert (await c.sweep())["overdue"] == []
    assert "overdue" not in (await c.digest(wire.DigestRequest(session="owner")))["text"]
    c.test_time[0] += timedelta(seconds=1)
    assert (await c.sweep())["overdue"] == [oid]
    assert "overdue" in (await c.digest(wire.DigestRequest(session="owner")))["text"]
    assert (await c.obligation(oid))["state"] == "open"
    assert (await c.one("SELECT payload FROM events WHERE type='obligation_overdue'"))["payload"]["obligation_id"] == oid
    assert "overdue" in (await c.one("SELECT line FROM notices WHERE id=?", (f"ntc:{oid}:overdue",)))["line"]
    assert (await c.sweep())["overdue"] == []
    await c.close()
    await c.connect()
    assert (await c.sweep())["overdue"] == []
    await c.revise(p["task_id"], wire.ReviseRequest(goal="Revised", stop=True), "revision")
    assert "overdue" not in (await c.digest(wire.DigestRequest(session="owner")))["text"]


async def test_results_and_refusals_do_not_reset_obligation_deadline(c):
    c.config["obligations"]["overdue_seconds"] = 60
    c.config["cascade"]["max_per_task"] = 0
    p = await running(c)
    c.test_time[0] += timedelta(seconds=61)
    await finish(c, p, status="partial")
    await plan(c, "refusal", continues=p["task_id"])
    oid = (await c.get_task(p["task_id"]))["obligation_id"]
    assert (await c.sweep())["overdue"] == [oid]
    await c.told(oid, wire.ToldRequest(turn_ref="reply"), "told")
    assert "overdue" not in (await c.digest(wire.DigestRequest(session="owner")))["text"]


async def settle(c, key, *, bound=False):
    p = await (running(c, key) if bound else plan(c, key))
    await c.cancel(p["task_id"], wire.CancelRequest(reason="Finished", owner_ref=REF), key + ":cancel")
    await c.execute("UPDATE notices SET sent_at=? WHERE hand_id=?", (c.now(), p["hand_id"]))
    await c.store.db.commit()
    return p


async def test_events_retention_keeps_exact_cap_expires_past_it_and_records(c):
    c.config["retention"].update(events_days=1, rows_days=10)
    p = await settle(c, "closed")
    count = (await c.one("SELECT count(*) AS n FROM events WHERE hand_id=?", (p["hand_id"],)))["n"]
    c.test_time[0] += timedelta(days=1)
    assert (await c.sweep())["retention"]["events"] == 0
    c.test_time[0] += timedelta(hours=1)  # retention runs at most hourly
    assert (await c.sweep())["retention"]["events"] == count
    assert (await c.one("SELECT count(*) AS n FROM events WHERE hand_id=?", (p["hand_id"],)))["n"] == 0
    assert (await c.get_hand(p["hand_id"]))["state"] == "stopped"
    assert (await c.retention_status())["payload"]["totals"]["events"] == count


async def test_rows_retention_keeps_cap_expires_past_it_and_tombstones_authority(c):
    c.config["retention"].update(events_days=10, rows_days=1)
    p = await settle(c, "closed", bound=True)
    c.test_time[0] += timedelta(days=1)
    assert (await c.sweep())["retention"]["tasks"] == 0
    c.test_time[0] += timedelta(hours=1)  # retention runs at most hourly
    result = (await c.sweep())["retention"]
    assert result["tasks"] == result["hands"] == 1
    assert await c.one("SELECT id FROM tasks WHERE id=?", (p["task_id"],)) is None
    assert await c.one("SELECT id FROM hands WHERE id=?", (p["hand_id"],)) is None
    assert (await sessions.lookup(c.store, "closed:child"))["hand_id"] == p["hand_id"]
    with pytest.raises(HTTPException, match="hands_replay_expired") as caught:
        await plan(c, "closed")
    assert caught.value.status_code == 410
    assert (await c.retention_status())["payload"]["totals"]["tasks"] == 1
    assert (await c.one("SELECT response,status_code FROM idempotency WHERE key='closed'")) == {
        "response": {"expired": True}, "status_code": 410}
    assert (await c.one("PRAGMA foreign_key_check")) is None
    await c.close()
    await c.connect()
    assert (await sessions.lookup(c.store, "closed:child"))["hand_id"] == p["hand_id"]
    assert (await c.retention_status())["payload"]["totals"]["tasks"] == 1


@pytest.mark.parametrize("pin", ["debt", "question", "permit", "effect", "provenance", "notice", "dependency"])
async def test_retention_preserves_required_history(c, pin):
    c.config["retention"].update(events_days=0, rows_days=0)
    p = await settle(c, "closed")
    if pin == "debt":
        await c.update("obligations", "id", (await c.get_task(p["task_id"]))["obligation_id"], state="result_ready")
    elif pin == "question":
        await c.insert("questions", id="q-pin", hand_id=p["hand_id"], asked_at=c.now(), text="Question")
    elif pin == "permit":
        await c.insert("permits", id="p-pin", ask_id="a-pin", hand_id=p["hand_id"], task_id=p["task_id"],
                       call={}, call_digest="a" * 64, workdir="", targets=[])
    elif pin == "effect":
        await c.insert("effects", hand_id=p["hand_id"], key="e-pin", tool="example", call_hash="a" * 64,
                       workdir="", intent_at=c.now())
    elif pin == "provenance":
        await c.insert("derivations", derived_kind="relay", derived_id="source-pin", hand_id=p["hand_id"],
                       evidence_level="E0", evidence_ref="source-pin")
    elif pin == "notice":
        await c.execute("UPDATE notices SET sent_at=NULL WHERE hand_id=?", (p["hand_id"],))
    else:
        await plan(c, "dependent", depends_on=[p["task_id"]])
    await c.store.db.commit()
    c.test_time[0] += timedelta(days=2)
    result = (await c.sweep())["retention"]
    assert result["tasks"] == 0
    assert await c.get_hand(p["hand_id"])
    # Dependency targets also need their task row; their old events need no pin.
    if pin != "dependency":
        assert (await c.one("SELECT count(*) AS n FROM events WHERE hand_id=?", (p["hand_id"],)))["n"] > 0


@pytest.mark.parametrize("path", REMOVED_SETTINGS)
def test_removed_yaml_settings_warn_and_are_ignored(path, tmp_path, caplog):
    data = {}
    node = data
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = None
    import yaml
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump(data))
    with caplog.at_level(logging.WARNING):
        config = load_config(tmp_path, environ={})
    assert path in caplog.text and "To upgrade, remove" in caplog.text
    node = config
    for part in parts[:-1]:
        node = node.get(part, {})
    assert parts[-1] not in node


@pytest.mark.parametrize("key", REMOVED_ENVIRONMENT)
def test_removed_embedding_environment_is_rejected_in_all_constructors(key, monkeypatch, tmp_path):
    monkeypatch.setenv(key, "1")
    for create in (EmbeddingConfig.from_env, VectorStoreConfig.from_env,
                   lambda: load_config(tmp_path, environ={key: "1"}),
                   lambda: validate(deepcopy(DEFAULTS) | {"environment": {key: "1"}})):
        with pytest.raises(ValueError, match=key) as caught:
            create()
        assert "To upgrade, remove" in str(caught.value)


@pytest.mark.parametrize("field", [path.split(".", 1)[1] for path in REMOVED_SETTINGS if path.startswith("contacts.")])
def test_removed_contact_constructor_arguments_have_upgrade_remedy(field):
    with pytest.raises(ValueError, match=re.escape("contacts." + field)) as caught:
        ContactsConfig(**{field: None})
    assert "To upgrade, remove" in str(caught.value)


@pytest.mark.parametrize("cls,field,required", [
    (EmbeddingConfig, "quantization", {"provider": "cpu", "model_id": "fixture", "dimensions": 2}),
    (VectorStoreConfig, "cache_size", {}), (VectorStoreConfig, "batch_window_ms", {}),
])
def test_removed_vector_constructor_arguments_have_upgrade_remedy(cls, field, required):
    with pytest.raises(ValueError, match=re.escape(cls.__name__ + "." + field)) as caught:
        cls(**required, **{field: None})
    assert "To upgrade, remove" in str(caught.value)


@pytest.mark.parametrize("path", ["cascade.max_per_task", "cascade.max_per_hour", "retention.events_days",
                                 "retention.rows_days", "obligations.overdue_seconds", "brief.max_tokens"])
@pytest.mark.parametrize("value", [-1, True, 1.5, "2", None, 3153600001])
def test_limits_reject_invalid_configuration(path, value):
    data = deepcopy(DEFAULTS)
    group, key = path.split(".")
    data["hands"][group][key] = value
    with pytest.raises(ConfigError, match=re.escape("hands." + path)):
        validate(data)


def test_every_retained_documented_cleanup_setting_has_a_product_reader():
    root = Path(__file__).resolve().parents[2]
    doc = (root / "docs/HANDS.md").read_text()
    inventory = re.findall(r"\| `((?:hands\.|ContactsConfig\.|VectorStoreConfig\.)[^`]+)` \| `([^`]+)` \|", doc)
    documented = {setting for setting, _ in inventory}
    expected = {"hands." + group + "." + key for group in ("cascade", "retention", "obligations", "brief", "health", "stalls")
                for key in DEFAULTS["hands"][group]}
    expected |= {"ContactsConfig." + field.name for field in fields(ContactsConfig)}
    expected |= {"VectorStoreConfig." + field.name for field in fields(VectorStoreConfig)}
    assert documented == expected
    assert "quantization" not in {field.name for field in fields(EmbeddingConfig)}
    for setting, reader in inventory:
        module, qualified = reader.split(":")
        path = root / module if module.endswith(".py") else root / "sidecar" / (module.replace(".", "/") + ".py")
        tree = ast.parse(path.read_text())
        for name in qualified.split("."):
            tree = next(node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name == name)
        key = setting.rsplit(".", 1)[-1]
        # Require an actual subscript, get/getattr or attribute load inside the named consumer.
        reads = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and isinstance(node.slice, ast.Constant):
                reads.append(node.slice.value)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                reads.append(node.attr)
            elif isinstance(node, ast.Call) and ((isinstance(node.func, ast.Attribute) and node.func.attr == "get")
                                               or (isinstance(node.func, ast.Name) and node.func.id == "getattr")):
                reads.extend(arg.value for arg in node.args if isinstance(arg, ast.Constant))
        assert key in reads, (setting, reader)


async def test_retained_brief_limit_changes_controller_assembly(c):
    p = await plan(c, "normal")
    c.config["brief"]["max_tokens"] = len(p["brief"].encode())
    exact = await plan(c, "exact")
    assert len(exact["brief"].encode()) == c.config["brief"]["max_tokens"]
    c.config["brief"]["max_tokens"] -= 1
    with pytest.raises(HTTPException, match="brief_too_large"):
        await plan(c, "too-small")
