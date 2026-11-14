"""Native persistence qualification plus source pagination/merge invariants.

Hermes writes every native file in a fresh subprocess/profile. Authority fixtures
are trusted server registry records, never scheduler formats or caller evidence.
"""
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colonymind.owner_goals.reads import (
    DeliveryEvidence, Destination, HermesOwnerReader, ReadError,
    list_owner_activity, merge_activity, parse_pagination, route_version, source_id,
)


@pytest.fixture
def native(tmp_path, request):
    source = os.environ.get("HERMES_SOURCE_ROOT")
    if not source:
        pytest.skip("set HERMES_SOURCE_ROOT to a reviewed native source")
    helper = Path(__file__).resolve().parents[2] / "tests" / "hermes_adapter" / "hermes_source.py"
    spec = importlib.util.spec_from_file_location("reviewed_hermes_source", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.assert_supported_source(source) == "R", "native run limitations require the reviewed source"
    home = tmp_path / "profile"
    environment = {**os.environ, "HERMES_HOME": str(home), "PYTHON_DOTENV_DISABLED": "1",
                   "PYTHONPATH": source, "COLONYMIND_SKIP_DOTENV": "1",
                   "READER_EXTRA_JOBS": str(getattr(request, "param", 0))}
    script = '''
import json
import os
from cron import jobs, executions, delivery_queue
catalog = []
for name, deliver, schedule, repeat, paused in [
    ("zibble", "telegram:101", "every 1h", None, False),
    ("wobble", "telegram:202", "every 1h", None, False),
    ("flim", "origin", "every 1h", None, False),
    ("flam", "local", "every 1h", None, False),
    ("plonk", "telegram:101", "every 1h", None, True),
    ("snork", "telegram:101", "in 3h", 1, False),
    ("blip", "telegram:101", "every 1h", 1, False),
]:
    catalog.append(jobs.create_job("zibble wobble", schedule, name=name,
        deliver=deliver, repeat=repeat, paused=paused))
for index in range(int(os.environ["READER_EXTRA_JOBS"])):
    catalog.append(jobs.create_job("zibble wobble", "every 1h", name="blorb-" + str(index),
                                  deliver="telegram:101"))
job = catalog[0]
run = executions.create_execution(job["id"], source="direct")
executions.mark_execution_running(run["id"])
output = "flibber\\n" + "x" * (2 * 1048576)
path = jobs.save_job_output(job["id"], output)
delivery_queue.enqueue(run["id"], job, output)
delivery_queue.claim_next()
delivery_queue._finish(run["id"], error="zibble")
executions.finish_execution(run["id"], success=True, delivery_outcome="failed")
print("@@RESULT@@" + json.dumps({"jobs": catalog, "execution_id": run["id"], "output": str(path)}))
'''
    result = subprocess.run([sys.executable, "-c", script], env=environment, cwd=tmp_path,
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    data = json.loads(next(line[len("@@RESULT@@"):] for line in result.stdout.splitlines()
                           if line.startswith("@@RESULT@@")))
    return home, data


class Registry:
    """Test server authority; no claims derived from the job's delivery string."""
    def __init__(self, jobs):
        self.rows = {}
        for job in jobs:
            destination = Destination("telegram", "account-a", "chat-a", None,
                                      ("owner-a",), True, True)
            if job["name"] == "wobble":
                destination = replace(destination, recipients=("owner-b",))
            if job["name"] == "flim":
                destination = replace(destination, direct=False, recipients=("owner-a", "owner-b"))
            targets = () if job["name"] == "flam" else (destination,)
            evidence = DeliveryEvidence("profile-a", job["id"], route_version(job), "evidence-a", targets)
            self.rows[job["id"], route_version(job)] = evidence

    def snapshot(self, profile_id):
        assert profile_id == "profile-a"
        return self.rows


def reader_for(native, *, ambiguous=False):
    home, data = native
    registry = Registry(data["jobs"])
    if not ambiguous:
        # Explicit registry proof of non-owner for this profile/account. A name
        # or a bare recipient identifier is never used as proof by the reader.
        job = next(j for j in data["jobs"] if j["name"] == "flim")
        key = job["id"], route_version(job)
        registry.rows[key] = replace(registry.rows[key], destinations=(
            Destination("telegram", "account-a", "chat-b", None, ("owner-b",), True, True),))
    return HermesOwnerReader(home, "profile-a", registry)


def fingerprint(home):
    return {str(p.relative_to(home)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in home.rglob("*") if p.is_file()}


def test_real_native_catalog_proof_facts_and_no_writes(native):
    reader = reader_for(native)
    before = fingerprint(native[0])
    snapshot = reader.catalog_snapshot("owner-a")
    assert {h.name for h in snapshot.handles} == {"zibble", "plonk", "snork", "blip"}
    assert {e.reason for e in snapshot.exclusions} == {"non_owner_destination", "delivery_disabled"}
    facts = {h.name: (h.enabled, h.recurring) for h in snapshot.handles}
    assert facts == {"zibble": (True, True), "plonk": (False, True),
                     "snork": (True, False), "blip": (True, False)}
    pages, offset = [], 0
    while True:
        page = reader.list_owner_cron_jobs("owner-a", 1, offset, snapshot)
        pages.extend(page["items"])
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    assert pages == sorted(snapshot.handles, key=lambda h: (h.profile_id, h.job_id))
    assert reader.list_owner_cron_jobs("owner-a", 100, 10001)["next_offset"] is None
    assert reader.list_owner_cron_jobs("owner-a", 100, 9007199254740991)["items"] == []
    assert fingerprint(native[0]) == before


@pytest.mark.parametrize("change", ["ambiguous", "unverified", "account_missing", "stale", "missing", "shared"])
def test_unproven_destinations_excluded_and_reported_unavailable(native, change):
    reader = reader_for(native, ambiguous=True)
    key = next(k for k, e in reader.registry.rows.items() if len(e.destinations) and not e.destinations[0].direct)
    evidence = reader.registry.rows[key]
    if change == "missing":
        del reader.registry.rows[key]
    elif change == "stale":
        reader.registry.rows[key] = replace(evidence, route_version="old")
    elif change != "ambiguous":
        target = Destination("telegram", "account-a", "chat-a", None, ("owner-a",), True, True)
        if change == "unverified":
            target = replace(target, verified=False)
        elif change == "account_missing":
            target = replace(target, account_id="")
        else:
            target = replace(target, recipients=("owner-a", "owner-b"))
        reader.registry.rows[key] = replace(evidence, destinations=(target,))
    snapshot = reader.catalog_snapshot("owner-a")
    assert len(snapshot.handles) == 4
    with pytest.raises(ReadError) as error:
        reader.list_owner_cron_jobs("owner-a", snapshot=snapshot)
    assert error.value.status_code == 503
    assert any(e.unresolved for e in error.value.exclusions)
    assert error.value.response() == {"error": {"code": "owner_goals_unavailable"}}


def test_native_original_large_output_and_terminal_record_do_not_prove_history(native):
    reader = reader_for(native)
    home, data = native
    assert Path(data["output"]).stat().st_size > 2 * 1048576
    with sqlite3.connect(home / "cron" / "executions.db") as conn:
        assert conn.execute("SELECT delivery_outcome FROM executions WHERE id=?",
                            (data["execution_id"],)).fetchone() == ("failed",)
    with sqlite3.connect(home / "cron" / "deliveries.db") as conn:
        assert conn.execute("SELECT job_json,content FROM deliveries").fetchone() == ("{}", "")
    before = fingerprint(home)
    handle = next(h for h in reader.catalog_snapshot("owner-a").handles if h.name == "zibble")
    for offset in (0, 1, 10001):
        with pytest.raises(ReadError, match="native_run_history_evidence_unavailable") as error:
            reader.list_owner_job_runs("owner-a", handle, "owner-a", 1, offset)
        assert error.value.status_code == 503
    # No runs for this job is not proof that history has never been pruned.
    empty = next(h for h in reader.catalog_snapshot("owner-a").handles if h.name == "snork")
    with pytest.raises(ReadError, match="native_run_history_evidence_unavailable"):
        reader.list_owner_job_runs("owner-a", empty, "owner-a")
    assert before == fingerprint(home)


def test_handle_reassignment_audience_and_snapshot_binding(native):
    reader = reader_for(native)
    snapshot = reader.catalog_snapshot("owner-a")
    handle = snapshot.handles[0]
    with pytest.raises(ReadError, match="owner_audience_mismatch"):
        reader.list_owner_job_runs("owner-a", handle, "owner-b")
    other = reader_for(native)
    with pytest.raises(ReadError, match="catalog_snapshot_mismatch"):
        other.list_owner_cron_jobs("owner-a", snapshot=snapshot)
    key = handle.job_id, handle.evidence.route_version
    reader.registry.rows[key] = replace(handle.evidence, version="evidence-b")
    with pytest.raises(ReadError, match="cron_binding_changed"):
        reader.list_owner_job_runs("owner-a", handle, "owner-a")


@pytest.mark.parametrize("native", [501], indirect=True)
def test_native_catalog_beyond_500_is_all_reachable(native):
    reader = reader_for(native)
    snapshot = reader.catalog_snapshot("owner-a")
    assert len(snapshot.handles) == 505
    handles, offset = [], 0
    while True:
        page = reader.list_owner_cron_jobs("owner-a", 100, offset, snapshot)
        assert 0 < len(page["items"]) <= 100
        handles.extend(page["items"])
        if page["next_offset"] is None:
            break
        assert page["next_offset"] == offset + len(page["items"])
        offset = page["next_offset"]
    assert tuple(handles) == snapshot.handles


def test_native_job_configuration_reassignment_rejects_old_handle(native, tmp_path):
    reader = reader_for(native)
    handle = reader.catalog_snapshot("owner-a").handles[0]
    environment = {**os.environ, "HERMES_HOME": str(native[0]),
                   "PYTHONPATH": os.environ["HERMES_SOURCE_ROOT"], "READER_JOB_ID": handle.job_id,
                   "PYTHON_DOTENV_DISABLED": "1"}
    result = subprocess.run([sys.executable, "-c", '''
import os
from cron import jobs
jobs.update_job(os.environ["READER_JOB_ID"], {"deliver": "telegram:303"})
'''], env=environment, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    with pytest.raises(ReadError, match="cron_binding_changed"):
        reader.list_owner_job_runs("owner-a", handle, "owner-a")


def test_registry_failure_and_missing_profile_never_become_empty_catalog(native, tmp_path):
    reader = reader_for(native)
    class UnavailableRegistry:
        def snapshot(self, profile_id):
            raise OSError("zibble")
    reader.registry = UnavailableRegistry()
    with pytest.raises(ReadError, match="destination_registry_unavailable"):
        reader.list_owner_cron_jobs("owner-a")
    absent = tmp_path / "absent"
    reader = HermesOwnerReader(absent, "profile-a", Registry([]))
    with pytest.raises(ReadError, match="cron_catalog_unreadable"):
        reader.list_owner_cron_jobs("owner-a")
    assert not absent.exists()


@pytest.mark.parametrize("pairs", [ [("limit", "0")], [("limit", "101")], [("offset", "-1")],
    [("limit", "+1")], [("limit", "1.0")], [("limit", "true")], [("limit", "١")],
    [("offset", "9007199254740992")], [("limit", "1"), ("limit", "2")], [("other", "1")]])
def test_invalid_pagination(pairs):
    with pytest.raises(ReadError) as error:
        parse_pagination(pairs)
    assert error.value.status_code == 422


def test_pagination_defaults_and_safe_bounds():
    assert parse_pagination([]) == {"limit": 50, "offset": 0}
    assert parse_pagination([("limit", "00100"), ("offset", "0009007199254740991")]) == {
        "limit": 100, "offset": 9007199254740991}
    for limit, offset in [(True, 0), (1, False), (1, -1), (0, 0), (1, 1.0)]:
        with pytest.raises(ReadError):
            merge_activity([], limit, offset)


def activity(identity, at="2026-01-01T00:00:00Z"):
    return {"id": source_id("activity", identity).replace(":", "_"), "source_id": identity, "kind": "progress",
            "at": at, "title": "zibble", "summary": "wobble", "artifact_ref": None,
            "sensitivity": "Secret"}


def progress_pages(items, calls=None, page_size=None):
    def read(*, limit, offset):
        if calls is not None:
            calls.append((limit, offset))
        rows = items[offset:offset + min(limit, page_size or limit)]
        return {"items": rows, "limit": limit, "offset": offset,
                "next_offset": offset + len(rows) if offset + len(rows) < len(items) else None}
    return read


def test_merge_pages_offset_dedup_ties_and_exhaustion():
    calls = []
    a = [activity("a"), activity("c"), activity("e"), activity("g")]
    b = [activity("b"), activity("c"), activity("d"), activity("f")]
    readers = [progress_pages(a, calls, 1), progress_pages(b, calls, 1)]
    assert [v["source_id"] for v in merge_activity(readers, 2, 2)["items"]] == ["c", "d"]
    assert merge_activity(readers, 2, 2)["next_offset"] == 4
    assert [v["source_id"] for v in merge_activity(readers, 2, 6)["items"]] == ["g"]
    assert merge_activity(readers, 2, 7)["next_offset"] is None
    assert calls[0][1] == calls[1][1] == 0
    assert merge_activity([], 1, 10001)["items"] == []


def test_merge_absolute_time_fraction_and_stable_namespace():
    a = activity("a", "2026-01-01T01:00:00.000000000000000000000000000001+01:00")
    b = activity("b", "2026-01-01T00:00:00Z")
    assert merge_activity([progress_pages([b]), progress_pages([a])], 1)["items"] == [a]
    assert len(source_id("cron", "profile", "job", "x" * 500)) < 256
    assert source_id("cron", "a", "b", "c") != source_id("cron", "b", "a", "c")


def test_merge_oversized_projection_and_nonterminal_empty_refusal():
    a = activity("a")
    a["summary"] = "x" * 1048576
    with pytest.raises(ReadError, match="activity_projection_invalid"):
        merge_activity([progress_pages([a])])
    def broken(**kwargs):
        return {"items": [], **kwargs, "next_offset": 1}
    with pytest.raises(ReadError, match="activity_source_unavailable"):
        merge_activity([broken])


def test_merge_valid_scalar_limits_still_reduce_encoded_byte_budget():
    rows = [activity("p-" + str(index).zfill(3)) for index in range(100)]
    for row in rows:
        row["title"] = "🫧" * 400
        row["summary"] = "🫧" * 1000
    page = merge_activity([progress_pages(rows)], 100)
    assert 0 < len(page["items"]) < 100
    assert page["next_offset"] == len(page["items"])
    assert len(json.dumps(page, ensure_ascii=True).encode()) <= 1048576


@pytest.mark.parametrize("extra_byte", [0, 1])
def test_merge_terminal_page_exact_byte_boundary(extra_byte):
    rows = [activity(f"p-{index:03}") for index in range(100)]
    for row in rows:
        row["summary"] = ""
    envelope = {"items": rows, "limit": 100, "offset": 0, "next_offset": None}
    target = 1048576 + extra_byte
    remaining = target - len(json.dumps(envelope, ensure_ascii=True).encode())
    for row in rows:
        count = min(1000, remaining // 12)
        row["summary"] = "🫧" * count
        remaining -= count * 12
    assert 0 <= remaining < 12
    rows[0]["title"] += "x" * remaining
    assert len(json.dumps(envelope, ensure_ascii=True).encode()) == target

    page = merge_activity([progress_pages(rows)], 100)
    assert len(json.dumps(page, ensure_ascii=True).encode()) <= 1048576
    assert len(page["items"]) == 100 - extra_byte
    assert page["next_offset"] == (99 if extra_byte else None)
    if extra_byte:
        tail = merge_activity([progress_pages(rows)], 100, page["next_offset"])
        assert page["items"] + tail["items"] == rows
        assert tail["next_offset"] is None


@pytest.mark.parametrize("extra_byte", [0, 1])
def test_catalog_terminal_page_exact_byte_boundary(tmp_path, extra_byte):
    jobs = [{"id": f"job-{index:03}", "name": "zibble", "enabled": True,
             "schedule": {"kind": "interval"}, "repeat": {"times": None}}
            for index in range(100)]
    cron = tmp_path / "cron"
    cron.mkdir()
    path = cron / "jobs.json"
    path.write_text(json.dumps({"jobs": jobs}))
    reader = HermesOwnerReader(tmp_path, "profile-a", Registry(jobs))
    snapshot = reader.catalog_snapshot("owner-a")
    envelope = {"items": [asdict(h) for h in snapshot.handles], "limit": 100,
                "offset": 0, "next_offset": None}
    target = 1048576 + extra_byte
    padding = target - len(json.dumps(envelope, ensure_ascii=True).encode())
    jobs[-1]["name"] += "x" * padding
    path.write_text(json.dumps({"jobs": jobs}))
    snapshot = reader.catalog_snapshot("owner-a")
    envelope["items"] = [asdict(h) for h in snapshot.handles]
    assert len(json.dumps(envelope, ensure_ascii=True).encode()) == target

    page = reader.list_owner_cron_jobs("owner-a", 100, snapshot=snapshot)
    serialized = {**page, "items": [asdict(h) for h in page["items"]]}
    assert len(json.dumps(serialized, ensure_ascii=True).encode()) <= 1048576
    assert len(page["items"]) == 100 - extra_byte
    assert page["next_offset"] == (99 if extra_byte else None)
    if extra_byte:
        tail = reader.list_owner_cron_jobs("owner-a", 100, page["next_offset"], snapshot)
        assert page["items"] + tail["items"] == list(snapshot.handles)
        assert tail["next_offset"] is None


@pytest.mark.parametrize("at", ["2026-02-30T00:00:00Z", "2026-01-01T00:00:60Z",
    "2026-01-01T00:00:00+24:00", "2026-01-01 00:00:00Z", "2026-01-01T00:00:00"])
def test_invalid_source_timestamp_is_unavailable(at):
    with pytest.raises(ReadError) as error:
        merge_activity([progress_pages([activity("p", at)])])
    assert error.value.status_code == 503


def test_unordered_activity_source_is_unavailable():
    with pytest.raises(ReadError):
        merge_activity([progress_pages([activity("b"), activity("a")])])


@pytest.mark.parametrize("failure", [OSError("zibble"), ReadError(422, "invalid_request", "zibble")])
@pytest.mark.parametrize("fail_offset", [0, 1])
def test_activity_source_failure_is_unavailable(failure, fail_offset):
    calls = []
    def broken(*, limit, offset):
        calls.append(offset)
        if offset == fail_offset:
            raise failure
        return {"items": [activity("a")], "limit": limit, "offset": offset, "next_offset": 1}
    with pytest.raises(ReadError) as error:
        merge_activity([broken], 1)
    assert error.value.status_code == 503
    assert error.value.response() == {"error": {"code": "owner_goals_unavailable"}}
    assert calls == ([0, 1] if fail_offset else [0])


def test_invalid_pagination_rejected_before_source_call():
    calls = []
    with pytest.raises(ReadError) as error:
        merge_activity([progress_pages([], calls)], 0)
    assert error.value.status_code == 422
    assert error.value.code == "invalid_request"
    assert calls == []


def test_native_attached_activity_fails_and_detachment_removes_source(native):
    reader = reader_for(native)
    handle = reader.catalog_snapshot("owner-a").handles[0]
    progress = progress_pages([activity("progress-a")])
    with pytest.raises(ReadError, match="activity_source_unavailable"):
        list_owner_activity(reader, "owner-a", [handle], "owner-a", progress)
    page = list_owner_activity(reader, "owner-a", [], "owner-a", progress)
    assert len(page["items"]) == 1 and page["next_offset"] is None
