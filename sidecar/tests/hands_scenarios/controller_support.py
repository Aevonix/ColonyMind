"""Real controller fixtures, isolated in pytest's temporary directory."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import pytest_asyncio

from colonymind.config import DEFAULTS
from colonymind.hands import controller, wire


EPOCH = {"owner_pid": 100, "owner_started_at": 1.0}
NEXT_EPOCH = {"owner_pid": 101, "owner_started_at": 2.0}
REF = {"platform": "sms", "chat": "5550100", "message_id": "owner-request"}


@pytest_asyncio.fixture
async def database(tmp_path):
    config = deepcopy(DEFAULTS["hands"])
    config.update(max_running=1, contracts={"text-route": {
        "version": "1", "family": "text-family", "modalities": ["text"],
        "context_tokens": 100000, "max_output_tokens": 10000,
        "tool_calls": {"qualified": True, "success_share": 1.0}, "json": True,
        "effort": {"2": {"reasoning_effort": "high"}}, "decode_tok_s": 40,
        "admission": {"family": "text-family", "priority": 10}}},
        routes={role: ["text-route"] for role in ("deep", "coding", "media", "verify")})
    config["health"].update(default_family_cap=1, running_limits={"text-family": 1})
    config["locks"]["max_waiters"] = 1
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    instance = controller.Controller(tmp_path / "hands.db", config=config, clock=lambda: now)
    await instance.connect()
    instance.health.record("text-family", running=0, waiting=0, kv=0.1)
    try:
        yield instance
    finally:
        await instance.close()


async def plan(c, key="plan-one", **task):
    body = wire.PlanRequest(parent_session="owner-session", turn=key, origin="owner_turn", origin_ref=REF,
        tasks=[{"goal": "Inspect the artifact", "kind": "deep", "risk": "low",
                "workdir": str(Path(c.store.path).parent / "workspace"), **task}])
    return (await c.plan(body, key))["plans"][0]


async def bind_plan(c, planned, key):
    await c.dispatched(planned["hand_id"], wire.DispatchedRequest(epoch=EPOCH,
        delegation_id=planned["hand_id"], lease_token=planned["lease_token"]), key + ":dispatch")
    await c.bind(planned["hand_id"], wire.BindRequest(epoch=EPOCH,
        executor_ref=planned["hand_id"], child_session=key + ":child"), key + ":bind")
    return planned


async def running(c, key="plan-one", **task):
    planned = await plan(c, key, **task)
    assert planned["admit"] == "start", planned
    return await bind_plan(c, planned, key)


async def finish(c, planned, *, version=1, summary="Artifact inspected", receipts=()):
    return await c.result(planned["hand_id"], wire.ResultRequest(epoch=EPOCH, intent_version=version,
        status="done", summary_ref=planned["hand_id"] + ":branch",
        summary_hash=hashlib.sha256(summary.encode()).hexdigest(), summary_text=summary, tool_digest=[],
        receipts=list(receipts), families_used=["text-family"]), planned["hand_id"] + ":result")
