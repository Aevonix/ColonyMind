"""What Instinct shows: ``colonymind instinct status|points|log``, ``colonymind report instinct``, the health route,
the doctor's lines and the daily maintenance. Ids and counts only: the words a point read never appear."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from colonymind.instinct import REGISTRY
from colonymind.instinct.log import Store
from colonymind.instinct.providers import Answer
from colonymind.instinct.tiers import Instinct

SECRET_WORDS = "the spare key is under the blue flowerpot"


class Provider:
    kind, model, private, max_state_chars, batch, timeout_ms = "local-decision.v1", "", True, 1200, False, 250
    readout = "letters-r1"
    types = frozenset({"yes_no", "choice"})

    def __init__(self):
        self.id = "local"

    async def answer(self, state, questions):
        await asyncio.sleep(0.01)
        return Answer({name: {"yes": 0.99, "no": 0.01} for name in questions},
                      revision="typed-decisions@1c5edc17a7acd8701df6fc341c0d179f1c62c982")


async def _instance(home):
    home.mkdir(parents=True, exist_ok=True)
    (home / "colonymind.yaml").write_text(yaml.safe_dump({"instinct": {"providers": {"local": {
        "kind": "local-decision.v1", "url": "http://127.0.0.1:9", "timeout_ms": 250}}}}))
    store = Store(home / "instinct.db")
    engine = Instinct(providers={"local": Provider()}, fast="local", store=store)
    engine.set_state("opt_out", "fast", "live", source="test")
    for n in range(3):
        await engine.decide("owner_verdict", {"reply": "Done.", "text": f"{SECRET_WORDS} {n}"}, {"turn": f"t-{n}"},
                            event="night")
        await engine.decide("opt_out", {"text": SECRET_WORDS}, {"turn": f"c-{n}"}, event="contact_turn")
    await engine.decide("owner_instruction", {"message": SECRET_WORDS}, {"turn": "t-9"}, event="owner_turn")
    await engine.drain()
    return store


def _cli(home, *args):
    return subprocess.run([sys.executable, "-m", "colonymind", "--instance", str(home), *args], capture_output=True,
                          text=True, timeout=120)


async def test_the_instinct_commands_show_ids_and_counts_and_never_the_words(tmp_path):
    home = tmp_path / "instance"
    await _instance(home)
    points = _cli(home, "instinct", "points")
    assert points.returncode == 0, points.stderr
    for name, point in REGISTRY.items():
        assert name in points.stdout and point.version in points.stdout
    assert "owner_verdict" in points.stdout and "fast live" in points.stdout
    status = _cli(home, "instinct", "status")
    assert status.returncode == 0, status.stderr
    assert "fast tier: local (local-decision.v1)" in status.stdout and "127.0.0.1" not in status.stdout
    assert "opt_out: fast live" in status.stdout and "outreach_reply: fast shadow" in status.stdout
    log = _cli(home, "instinct", "log", "--limit", "20")
    assert log.returncode == 0, log.stderr
    assert "owner_verdict" in log.stdout and "t-1" in log.stdout and "dec-" in log.stdout
    as_json = json.loads(_cli(home, "instinct", "log", "--json", "--point", "opt_out").stdout)
    assert {row["point"] for row in as_json} == {"opt_out"} and len(as_json) == 3
    for output in (points.stdout, status.stdout, log.stdout, json.dumps(as_json)):
        assert "flowerpot" not in output and "spare key" not in output


async def test_the_report_counts_each_point_and_tier_and_names_the_record_in_force(tmp_path):
    home = tmp_path / "instance"
    await _instance(home)
    done = _cli(home, "report", "instinct", "--since", "7d")
    assert done.returncode == 0, done.stderr
    text = done.stdout
    assert text.startswith("# Instinct report")
    sections = {section.split("\n", 1)[0]: section for section in text.split("\n## ")[1:]}
    assert "fast (live): 3 answered, 0 abstained" in sections["opt_out"]
    assert "fast (live): 3 answered, 3 abstained" in sections["owner_verdict"]
    assert "latency p50" in text and "p95" in text
    assert "record in force: nightly fit: 71 labels, isotonic (isotonic)" in sections["opt_out"]
    assert "record in force: none matched the provider" in sections["owner_verdict"]
    assert "owner_instruction" in text and "flowerpot" not in text
    html = _cli(home, "report", "instinct", "--format", "html")
    assert html.returncode == 0 and html.stdout.startswith("<!doctype html>")
    bad = _cli(home, "report", "instinct", "--since", "whenever")
    assert bad.returncode == 2 and "since" in bad.stderr


async def test_the_record_in_force_is_the_one_the_latest_answer_was_read_under(tmp_path):
    """A calibrated answer, then the server swaps its checkpoint: the latest answer abstained with no record, so the
    report no longer names the shipped letter record as in force. A later failed call (no answer) changes nothing."""
    from colonymind.reports.instinct import collect, document
    from colonymind.reports.text import markdown
    home = tmp_path / "instance"
    home.mkdir()
    store = Store(home / "instinct.db")
    provider = Provider()
    engine = Instinct(providers={"local": provider}, fast="local", store=store)
    engine.set_state("opt_out", "fast", "live", source="test")
    await engine.decide("opt_out", {"text": "Lose my number."}, {"turn": "t-1"})
    assert "record in force: nightly fit: 71 labels, isotonic (isotonic)" in markdown(document(collect(home)))

    async def swapped(state, questions):
        return Answer({name: {"yes": 0.99, "no": 0.01} for name in questions}, revision="another-checkpoint@0f0f")

    provider.answer = swapped
    await engine.decide("opt_out", {"text": "Lose my number."}, {"turn": "t-2"})
    collected = collect(home)
    assert collected.points["opt_out"]["fast"].record is None
    text = markdown(document(collected))
    assert "none matched the provider" in text and "record in force: nightly fit" not in text

    async def failing(state, questions):
        raise ValueError("malformed")

    provider.answer = failing
    await engine.decide("opt_out", {"text": "Lose my number."}, {"turn": "t-3"})
    assert collect(home).points["opt_out"]["fast"].record is None


async def test_the_health_route_shows_each_providers_breaker_and_the_points_counts(monkeypatch):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from colonymind import instinct
    from colonymind.api.routers import host
    engine = Instinct(providers={"local": Provider()}, fast="local")
    engine.set_state("opt_out", "fast", "live", source="test")
    await engine.decide("opt_out", {"text": SECRET_WORDS})
    monkeypatch.setattr(instinct, "shared", lambda: engine)
    app = FastAPI()
    app.include_router(host.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        body = (await client.get("/v1/host/health")).json()
    assert body["instinct"]["providers"]["local"]["breaker"] == "closed"
    assert body["instinct"]["points"]["opt_out"]["counts"]["fast"]["settled"] == 1
    assert "flowerpot" not in json.dumps(body)


def test_the_doctor_says_whether_a_fast_provider_is_configured_and_counts_points_by_state(tmp_path, monkeypatch):
    from colonymind import doctor
    home = tmp_path / "instance"
    asyncio.run(_instance(home))
    monkeypatch.setenv("COLONYMIND_HOME", str(home))
    for name in ("COLONYMIND_DECISIONS_URL", "COLONYMIND_DECISIONS_TIMEOUT_MS", "COLONYMIND_DECISIONS_POINTS"):
        monkeypatch.delenv(name, raising=False)
    result = doctor.check_instinct()
    assert result.status == doctor.PASS
    assert "fast provider local-decision.v1 configured" in result.detail and "127.0.0.1" not in result.detail
    assert "4 live" in result.detail and "decisions in the last day" in result.detail


async def test_the_daily_maintenance_prunes_the_log_and_then_waits_a_day(tmp_path):
    from colonymind.instinct import maintenance
    store = Store(tmp_path / "instinct.db")
    store.write({"id": "dec-old", "at": (datetime.now(timezone.utc) - timedelta(days=200)).isoformat(),
                 "event": "tick", "point": "opt_out", "version": "v", "tiers": {}, "label": None, "applied": False,
                 "effect": "filter", "answered_by": "fallback", "reason": "fallback", "refs": {}, "state_hash": None})
    slept = []

    async def sleep(seconds):
        slept.append(seconds)
        if len(slept) > 1:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await maintenance(tmp_path, sleep=sleep, first_delay=0)
    assert store.rows() == [] and slept[1] == 24 * 3600


def test_the_server_runs_one_maintenance_task_in_its_lifespan():
    import inspect

    from colonymind import server
    source = inspect.getsource(server.lifespan)
    assert source.count("instinct_maintenance") >= 2          # started, and cancelled on shutdown
