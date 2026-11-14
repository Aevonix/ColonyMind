"""Expectation engine (Mind M3a): predictions, resolution, surprise, calibration."""

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from threading import Barrier

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import colonymind.api.routers.host as host_mod
from colonymind.self_model.expectations import (
    ExpectationEngine, ExpectationStore, Prediction,
)


def store(tmp_path):
    return ExpectationStore(db_path=str(tmp_path / "exp.db"))


def test_create_dedups(tmp_path):
    s = store(tmp_path)
    a = s.create(subject="commitment:c1", domain="commitment",
                 expectation="done", confidence=0.7, horizon=time.time()+10,
                 source="t", dedup_key="commitment:c1")
    b = s.create(subject="commitment:c1", domain="commitment",
                 expectation="done", confidence=0.7, horizon=time.time()+10,
                 source="t", dedup_key="commitment:c1")
    assert a is not None and b is None
    assert len(s.pending()) == 1


def test_resolved_prediction_not_recreated(tmp_path):
    """A scored prediction must never come back for the same subject+horizon,
    or every checker pass re-scores the same miss forever (calibration n
    inflates, one failure reads as dozens)."""
    s = store(tmp_path)
    h = time.time() + 10
    a = s.create(subject="commitment:c1", domain="commitment",
                 expectation="done", confidence=0.7, horizon=h,
                 source="t", dedup_key="commitment:c1")
    s.resolve(a.prediction_id, "miss")
    again = s.create(subject="commitment:c1", domain="commitment",
                     expectation="done", confidence=0.7, horizon=h,
                     source="t", dedup_key="commitment:c1")
    assert again is None
    # a moved due date IS a new prediction
    moved = s.create(subject="commitment:c1", domain="commitment",
                     expectation="done", confidence=0.7, horizon=h + 3600,
                     source="t", dedup_key="commitment:c1")
    assert moved is not None


def test_repeat_surprises_merge_into_one_concern(tmp_path):
    """Misses about the same subject strengthen ONE anomaly concern (keyed by
    subject), never one concern per scoring pass."""
    ws = FakeWS()
    e = engine(tmp_path, ws=ws)
    for i in range(3):
        e._surprise(Prediction(
            prediction_id=f"p-{i}", subject="commitment:c1",
            domain="commitment", expectation="done", confidence=0.7,
            horizon=time.time(), source="t"))
    keys = {b["dedup_key"] for b in ws.bumps}
    assert len(ws.bumps) == 3 and keys == {"surprise:commitment:c1"}


def test_due_and_resolve(tmp_path):
    s = store(tmp_path)
    past = s.create(subject="x:1", domain="d", expectation="e",
                    confidence=0.6, horizon=time.time()-5, source="t",
                    dedup_key="x1")
    s.create(subject="x:2", domain="d", expectation="e", confidence=0.6,
             horizon=time.time()+9999, source="t", dedup_key="x2")
    due = s.due()
    assert len(due) == 1 and due[0].prediction_id == past.prediction_id
    s.resolve(past.prediction_id, "hit")
    assert len(s.due()) == 0


# --- resolvers + check -----------------------------------------------------

class FakeCommitments:
    def __init__(self, statuses):
        self._st = statuses  # id -> status
    def get(self, cid):
        return {"id": cid, "status": self._st.get(cid), "description": "d"}
    def list(self, status=None, limit=50, **kw):
        return {"commitments": [], "total": 0}


class FakeWS:
    def __init__(self): self.bumps = []
    def bump(self, **kw): self.bumps.append(kw)


def engine(tmp_path, commitments=None, ws=None):
    e = ExpectationEngine(store(tmp_path), workspace=ws)
    if commitments is not None:
        e._commitments = lambda: commitments
    else:
        e._commitments = lambda: None
    return e


def test_check_hit_on_fulfilled(tmp_path):
    e = engine(tmp_path, commitments=FakeCommitments({"c1": "fulfilled"}))
    p = e.store.create(subject="commitment:c1", domain="commitment",
                       expectation="e", confidence=0.8,
                       horizon=time.time()-5, source="t", dedup_key="k",
                       detail={"commitment_id": "c1"})
    counts = e.check()
    assert counts["hit"] == 1
    resolved = e.store.resolved_since(0)
    assert resolved and resolved[0].outcome == "hit"


def test_check_miss_emits_surprise(tmp_path):
    ws = FakeWS()
    e = engine(tmp_path, commitments=FakeCommitments({"c1": "overdue"}), ws=ws)
    e.store.create(subject="commitment:c1", domain="commitment",
                   expectation="fulfilled on time", confidence=0.8,
                   horizon=time.time()-5, source="t", dedup_key="k",
                   detail={"commitment_id": "c1"})
    counts = e.check()
    assert counts["miss"] == 1
    assert ws.bumps and ws.bumps[0]["kind"] == "anomaly"
    assert "surprise" in ws.bumps[0]["summary"]


def test_pending_commitment_past_due_is_miss(tmp_path):
    e = engine(tmp_path, commitments=FakeCommitments({"c1": "pending"}))
    e.store.create(subject="commitment:c1", domain="commitment",
                   expectation="e", confidence=0.7, horizon=time.time()-5,
                   source="t", dedup_key="k", detail={"commitment_id": "c1"})
    assert e.check()["miss"] == 1


def test_unresolvable_grace_then_unresolved(tmp_path):
    # no resolver matches -> stays pending until grace elapses
    e = engine(tmp_path)
    e.store.create(subject="mystery:1", domain="d", expectation="e",
                   confidence=0.5, horizon=time.time()-5, source="t",
                   dedup_key="k")
    assert e.check() == {"hit": 0, "miss": 0, "unresolved": 0}  # within grace
    # backdate horizon beyond the grace window
    with e.store._lock:
        e.store._conn.execute("UPDATE predictions SET horizon=?",
                              (time.time() - 90000,))
        e.store._conn.commit()
    assert e.check()["unresolved"] == 1


# --- generation ------------------------------------------------------------

class GenCommitments:
    def get(self, cid): return None
    def list(self, status=None, limit=50, **kw):
        if status == ["pending"]:
            due = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
            return {"commitments": [
                {"id": "c1", "due_at": due, "description": "ship the audit"},
                {"id": "c2", "due_at": None, "description": "no due date"}],
                "total": 2}
        if status == ["fulfilled"]:
            return {"commitments": [], "total": 8}
        if status == ["overdue"]:
            return {"commitments": [], "total": 2}
        return {"commitments": [], "total": 0}


class PastDueCommitments(GenCommitments):
    def list(self, status=None, limit=50, **kw):
        if status == ["pending"]:
            due = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
            return {"commitments": [
                {"id": "c1", "due_at": due, "description": "already late"}],
                "total": 1}
        return super().list(status=status, limit=limit, **kw)


# --- calibration -----------------------------------------------------------

def test_calibration_brier(tmp_path):
    e = engine(tmp_path)
    # two confident-correct, one confident-wrong
    for i, (conf, out) in enumerate([(0.9, "hit"), (0.9, "hit"), (0.9, "miss")]):
        p = e.store.create(subject=f"x:{i}", domain="d", expectation="e",
                           confidence=conf, horizon=time.time(), source="t",
                           dedup_key=f"k{i}")
        e.store.resolve(p.prediction_id, out)
    cal = e.calibration()
    # brier = mean((0.9-1)^2, (0.9-1)^2, (0.9-0)^2) = (.01+.01+.81)/3
    assert cal["d"]["brier"] == pytest.approx((0.01 + 0.01 + 0.81) / 3, abs=1e-4)
    assert cal["d"]["hit_rate"] == pytest.approx(2 / 3, abs=1e-3)


# --- API -------------------------------------------------------------------

@asynccontextmanager
async def _client(eng):
    orig = host_mod._expectations
    host_mod._expectations = eng
    app = FastAPI()
    app.include_router(host_mod.router)
    try:
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as c:
            yield c
    finally:
        host_mod._expectations = orig


async def test_api_snapshot(tmp_path):
    e = engine(tmp_path)
    e.store.create(subject="x:1", domain="d", expectation="a thing happens",
                   confidence=0.6, horizon=time.time()+9999, source="t",
                   dedup_key="k")
    async with _client(e) as c:
        r = await c.get("/v1/host/self/expectations")
        body = r.json()
        assert body["available"] and body["pending"][0]["expectation"] == "a thing happens"


async def test_api_unavailable():
    async with _client(None) as c:
        assert (await c.get("/v1/host/self/expectations")).json() == {"available": False}


def numeric_surprise(s, **overrides):
    fields = {
        "source": "sensor-a",
        "subject": "device:sample",
        "feature": "temperature",
        "summary": "Temperature exceeded the observed range",
        "observed_at": datetime(2026, 6, 5, 23, 30, tzinfo=timezone.utc),
        "severity": 0.8,
    }
    fields.update(overrides)
    return s.record_numeric_surprise(**fields)


def test_numeric_surprise_records_private_note_at_capture_time(tmp_path):
    s = store(tmp_path)
    observed_at = datetime(2026, 6, 5, 23, 30, tzinfo=timezone.utc)
    p = numeric_surprise(s, observed_at=observed_at)

    assert p is not None
    assert p.subject == "device:sample"
    assert p.domain == "upkeep"
    assert p.outcome == "miss"
    assert p.horizon == p.resolved_at == observed_at.timestamp()
    assert p.outcome_observed_at == observed_at.timestamp()
    assert p.source == "sensor-a"
    assert p.source_kind == p.cohort == "numeric_surprise"
    assert p.subject_person_id == p.viewer_scope == "owner"
    assert p.shareability == "owner_private"
    assert p.expectation == "Temperature exceeded the observed range"
    assert p.detail == {
        "kind": "numeric_surprise",
        "provenance": "derived_unverified",
        "note_only": True,
        "observed_at": observed_at.isoformat(),
        "feature": "temperature",
        "source": "sensor-a",
        "severity": 0.8,
        "probability_available": False,
        "forecast_probability": None,
    }
    assert s.pending() == []
    assert s.resolved_since(observed_at.timestamp() + 1, include_notes=True) == []
    assert s.projected_for_viewer(viewer_scope="guest") == []
    assert s.projected_for_viewer(viewer_scope="person:contact") == []
    assert s.get(p.prediction_id) == p


def test_numeric_surprise_dedups_subject_feature_utc_capture_date(tmp_path):
    s = store(tmp_path)
    first = numeric_surprise(s)
    duplicate = numeric_surprise(
        s, source="sensor-b", summary="A repeated observation", severity=0.9,
        observed_at=datetime(2026, 6, 6, 1, 30,
                             tzinfo=timezone(timedelta(hours=2))),
    )
    assert first is not None and duplicate is None
    assert len(s.resolved_since(0, domain="upkeep", include_notes=True)) == 1
    assert numeric_surprise(s, feature="pressure") is not None
    assert numeric_surprise(s, subject="device:other") is not None
    assert numeric_surprise(
        s, observed_at=datetime(2026, 6, 6, tzinfo=timezone.utc),
    ) is not None
    assert len(s.resolved_since(0, domain="upkeep", include_notes=True)) == 4


def test_numeric_surprise_dedup_survives_reopening(tmp_path):
    first_store = store(tmp_path)
    assert numeric_surprise(first_store) is not None
    first_store._conn.close()
    assert numeric_surprise(store(tmp_path), source="sensor-b") is None


def test_numeric_surprise_daily_dedup_is_atomic_across_connections(tmp_path):
    stores = [store(tmp_path), store(tmp_path)]
    ready = Barrier(2)

    def record(index):
        ready.wait(timeout=5)
        return numeric_surprise(stores[index], source=f"sensor-{index}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(record, range(2)))
    assert sum(p is not None for p in results) == 1
    assert len(stores[0].resolved_since(0, domain="upkeep", include_notes=True)) == 1


def test_numeric_surprise_never_enters_checker_or_broad_surprise_hook(
    tmp_path, monkeypatch,
):
    ws = FakeWS()
    e = engine(tmp_path, ws=ws)

    def forbidden_surprise(_prediction):
        pytest.fail("numeric observation reached the predictive surprise hook")

    monkeypatch.setattr(e, "_surprise", forbidden_surprise)
    assert numeric_surprise(e.store) is not None
    assert e.check() == {"hit": 0, "miss": 0, "unresolved": 0}
    assert ws.bumps == []
    for table in ("expectation_outcome_observations", "forecast_revisions",
                  "forecast_outcomes"):
        assert e.store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_numeric_surprise_does_not_invent_predictive_calibration(tmp_path):
    e = engine(tmp_path)
    genuine = e.store.create(
        subject="device:other", domain="upkeep", expectation="Remains available",
        confidence=0.8, horizon=time.time() + 10, source="service-probe",
        dedup_key="availability:other",
    )
    e.store.resolve(genuine.prediction_id, "hit")
    baseline = e.calibration()
    assert numeric_surprise(e.store) is not None
    assert e.calibration() == baseline
    report = e.calibration_report()
    assert report["resolved_n"] == 1
    assert report["domains"]["upkeep"]["n"] == 1
    scoped = e.calibration_report(subject_person_id="owner", viewer_scope="owner")
    assert scoped["resolved_n"] == 1
    assert all(row["cohort"] != "numeric_surprise" for row in scoped["cohorts"])


def test_numeric_surprise_rejects_ambiguous_capture_time(tmp_path):
    s = store(tmp_path)
    with pytest.raises(ValueError, match="timezone"):
        numeric_surprise(s, observed_at=datetime(2026, 6, 5))
    assert s.resolved_since(0, include_notes=True) == []


def test_numeric_surprise_is_hidden_from_legacy_miss_consumers(tmp_path):
    s = store(tmp_path)
    note = numeric_surprise(s)
    prediction = s.create(
        subject="device:other", domain="upkeep", expectation="Remains available",
        confidence=0.8, horizon=time.time() + 10, source="service-probe",
        dedup_key="availability:other",
    )
    s.resolve(prediction.prediction_id, "miss")
    assert [row.prediction_id for row in s.resolved_since(0)] == [prediction.prediction_id]
    assert [row.prediction_id for row in s.resolved_since(0, domain="upkeep")] == [prediction.prediction_id]
    assert {row.prediction_id for row in s.resolved_since(0, include_notes=True)} == {
        note.prediction_id, prediction.prediction_id,
    }
    assert s.get(note.prediction_id) == note
