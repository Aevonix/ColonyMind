"""Context caps hold across selection modes, host allowances and slow producers."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest

from colonymind.api.routers import host
from colonymind.api.schemas.host import ContextAssembleRequest, ContextSection
from colonymind.memory import blocks, compass, turn_budget
from recall_route import recall_app
from test_recall_packing import Passing, assemble_body, post


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    monkeypatch.setenv(turn_budget.WAIT_ENV, "7500")
    blocks.reset_settings()
    yield
    blocks.reset_settings()


@pytest.mark.parametrize("rerank", ["on", "off", "shadow", "unverified", "missed"])
@pytest.mark.parametrize("selection", ["on", "off"])
@pytest.mark.parametrize("items", [3, 7, 20])
async def test_assembled_recall_has_item_character_and_time_caps(
        monkeypatch, tmp_path, rerank, selection, items):
    (tmp_path / "colonymind.yaml").write_text(json.dumps({"context": {
        "status": False, "budget_chars": 2500, "recall_items": items,
        "limits": {"colonymind-memory": 100_000}}}))
    setup = await recall_app(monkeypatch, tmp_path, reranker=Passing(), initiatives=0)
    monkeypatch.setenv(compass.SELECTION_ENV, selection)
    if rerank == "unverified":
        monkeypatch.setenv("COLONYMIND_RECALL_RERANK_CALIBRATION", "0" * 64)
    elif rerank == "missed":
        async def failing(*args, **kwargs):
            raise RuntimeError("fixture failure")
        monkeypatch.setattr(setup.reranker, "rerank", failing)
    else:
        monkeypatch.setenv("COLONYMIND_RECALL_RERANK", rerank)
    # Neither an oversized host allowance nor disabling either optional selector widens a cap.
    started = time.monotonic()
    response = await post(setup.app, "/v1/host/context/assemble", assemble_body(
        inline_chars=100_000, wait_ms=600_000))
    elapsed = time.monotonic() - started
    text = "\n\n".join(f"## {s['title'] or s['id']}\n{s['body']}" for s in response["sections"])
    packet = response["trace"]["packet"]
    assert 0 < packet["count"] <= packet["limit"] <= items
    memory = next((s for s in response["sections"] if s["id"] == compass.RECALLED), None)
    assert memory and memory["body"].count('"kind": "source_quote"') <= items
    assert len(memory.get("citations") or ()) <= items
    assert len(text) <= response["budget"]["chars_after"] <= 2500
    assert response["budget"]["recall_items"] == items
    assert response["budget"]["wait_ms"] == 7500 and elapsed < 7.5


@pytest.mark.parametrize("allowance", [None, 1, 400, 1000, 100_000])
def test_final_fit_counts_full_sections_and_cannot_be_disabled(monkeypatch, allowance):
    # Even a stale setting passed directly to the fitter has no bypass reader.
    monkeypatch.setattr(blocks, "settings", lambda: {
        "blocks": False, "budget_chars": 1000, "recall_items": 5, "limits": {"x": 100_000}})
    sections = [ContextSection(id="x", title="Evidence", body="evidence\n" * 1000, priority=90)]
    shown = [blocks.fingerprint(s.title, s.body) for s in sections]
    held = [ContextSection(id="colonymind-memory", title="Relevant Memories", body="evidence", priority=90)]
    fitted, report = host._fit_sections(sections, budget=allowance, shown=shown, held=held, where="test")
    cap = 1000 if allowance is None else min(allowance, 1000)
    assert sum(blocks.cost(s.title, s.body) for s in fitted) <= cap
    assert report["chars_after"] <= cap and report["budget"] == cap


def request(**kwargs):
    return ContextAssembleRequest.model_validate(assemble_body(trace=False, **kwargs))


@pytest.mark.parametrize("stage", ["lanes", "selection", "telemetry"])
async def test_deadline_covers_every_assembly_stage(monkeypatch, tmp_path, stage):
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setenv(turn_budget.WAIT_ENV, "300")
    cancelled = asyncio.Event()

    async def hang(*args, **kwargs):
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()

    async def sections(*args, **kwargs):
        return [ContextSection(id="temporal-context", title="Time", body="Now", priority=100)]

    monkeypatch.setattr(host, "_assemble_sections", sections)
    monkeypatch.setattr(host, "_telemetry", None)
    if stage == "lanes":
        monkeypatch.setattr(host, "_assemble_sections", hang)
    elif stage == "selection":
        monkeypatch.setattr(host, "_turn_context_selection", hang)
    else:
        monkeypatch.setattr(host, "_telemetry", SimpleNamespace(touch=hang))
    started = time.monotonic()
    response = await host.context_assemble(request(wait_ms=600_000))
    elapsed = time.monotonic() - started
    assert response.sections == [] and response.budget["timed_out"] is True
    assert response.budget["wait_ms"] == 300 and elapsed < 0.3
    assert cancelled.is_set() and turn_budget.current() is None


async def test_expiring_assembly_cancels_its_started_recall(monkeypatch, tmp_path):
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setenv(turn_budget.WAIT_ENV, "300")
    ended = asyncio.Event()
    tasks = []

    async def recall():
        try:
            await asyncio.sleep(60)
        finally:
            ended.set()

    async def lanes(*args, started, **kwargs):
        task = asyncio.create_task(recall())
        tasks.append(task)
        started.append(task)
        await asyncio.sleep(60)

    monkeypatch.setattr(host, "_assembled_lanes", lanes)
    started = time.monotonic()
    response = await host.context_assemble(request(wait_ms=5000))
    assert time.monotonic() - started < 0.3
    assert response.sections == [] and ended.is_set() and all(t.done() for t in tasks)


async def test_recipient_packets_share_the_character_and_time_caps(monkeypatch):
    monkeypatch.setenv(turn_budget.WAIT_ENV, "300")
    monkeypatch.setattr(blocks, "settings", lambda: {"budget_chars": 1000, "recall_items": 5, "limits": {}})

    async def sections(*args, **kwargs):
        assert turn_budget.current() is not None
        return [ContextSection(id="x", title="Evidence", body="evidence\n" * 1000, priority=90)]

    monkeypatch.setattr(host, "_assemble_sections", sections)
    assert len(await host.assemble_packet("contact-a", limit_chars=100_000)) <= 1000

    async def hang(*args, **kwargs):
        await asyncio.sleep(60)

    monkeypatch.setattr(host, "_assemble_sections", hang)
    started = time.monotonic()
    assert await host.assemble_packet("contact-a") == ""
    assert time.monotonic() - started < 0.3 and turn_budget.current() is None


async def test_a_producer_that_exhausts_the_clock_without_yielding_cannot_serve_late_context(monkeypatch, tmp_path):
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    remembered = []

    async def expired(*args, **kwargs):
        turn_budget.current().started -= 60
        return [ContextSection(id="x", title="Evidence", body="late evidence", priority=90)]

    monkeypatch.setattr(host, "_assemble_sections", expired)
    monkeypatch.setattr(host, "_telemetry", None)
    monkeypatch.setattr(host, "_remember_served", lambda *args, **kwargs: remembered.append(args))
    response = await host.context_assemble(request())
    assert response.sections == [] and response.budget["timed_out"] is True
    assert remembered == []

