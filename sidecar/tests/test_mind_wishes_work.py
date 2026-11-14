"""Wishes learn from the owner's real work.

The wish stage's classes come from the mind's own tasks and, with ``faculties.work_records`` on, from the work log
(``WorkStore.failures``): a piece of the owner's work whose status is ``failed`` is one check-verified failure of
each class ``work:<tool>:<kind>`` its failed calls name; work the owner corrected (the nightly owner-correction
overlay) is one owner-verified failure of ``work:<tool>:corrected``; a mind task the owner rated ``wrong`` is a
verified failure of its own class. A guard's block is no class, and unproven or partial work is no failure. A
declined work class waits the full 30 days whatever fails after the no; a mind class keeps its rule.

Everything runs against the real ledger, work projection, store and mind; only the model is a fake.
"""

from __future__ import annotations

import re
import threading
from datetime import timedelta
from types import SimpleNamespace

import pytest

from colonymind.mind import wishes as wishes_module
from colonymind.mind.wishes import WorkFailure, state_of, wish_of, work_classes
from colonymind.work.store import WorkStore
from owner_answer import owner_answers
from test_mind_consolidate import Fixture, OWNER
from test_mind_wishes import ON, SIG, NOON, WishRouter, cites, task, wish_calls, wishes
from test_work_projection import envelope, operation, stores  # noqa: F401  (the fixture)

WORK = {**ON, "faculties": {**ON["faculties"], "work_records": True}}
REQUEST = "Please update the quarterly report."
WRITE_ERROR = "work:write_file:tool_error"


def write(ident, *, resource="resource:report", **changes):
    return operation(ident, resource_ref=resource, **changes)


def read(ident="look"):
    return operation(ident, tool="read_file", action="read", effect="read", resource_ref="resource:notes",
                     evidence={"kind": "none"})


def failed_write(ident="w", kind="tool_error", **changes):
    return write(ident, success=False, evidence={"kind": "none"},
                 **({"failure_kind": kind} if kind else {}), **changes)


def guarded(ident="g", kind="plugin_block"):
    return operation(ident, tool="terminal", action="run", effect="execute", resource_ref="resource:shell",
                     success=False, failure_kind=kind, evidence={"kind": "none"})


def unproven(ident="u"):
    return write(ident, success=None, evidence={"kind": "none"})


# ---------------------------------------------------------------------------
# The reader: WorkStore.failures, codes only, on a read-only connection
# ---------------------------------------------------------------------------

def admit(stores, ident, operations, *, contact_id="owner"):
    stores.ledger.record_source(ident, contact_id=contact_id, session_id="session",
                                messages=[{"role": "user", "content": REQUEST}], derive_claims=False,
                                work={"envelope": envelope(ident, operations=operations), "principal_id": "host"})


def test_failures_reads_each_piece_of_work_with_its_failed_acting_calls_and_their_kinds(stores):
    admit(stores, "a", [read(), failed_write("w"), guarded("g"), failed_write("p", kind=None, tool="patch",
                                                                             resource="resource:other")])
    admit(stores, "b", [write("ok")])
    admit(stores, "c", [failed_write("x")], contact_id="someone-else")
    stores.work.reconcile()
    reader = WorkStore(SimpleNamespace(db_path=stores.ledger.db_path), read_only=True)
    found = reader.failures(contact_id="owner", since=None, until=None)
    assert [item["id"] for item in found] == ["a", "b"]                       # the owner's, oldest first
    first, second = found
    assert first["status"] == "failed" and first["overlay"] is None
    assert first["tools"] == ["write_file", "terminal", "patch"]               # read effects are no attempt
    assert first["failed"] == [("write_file", "tool_error"), ("terminal", "plugin_block"), ("patch", "failed")]
    assert second == {"id": "b", "created_at": second["created_at"], "status": "proven", "overlay": None,
                      "tools": ["write_file"], "failed": []}
    assert REQUEST not in repr(found)                                          # codes only, never the words


def test_a_failure_a_later_verified_call_made_good_is_left_out(stores):
    admit(stores, "a", [failed_write("first"), write("again"), failed_write("send", tool="patch",
                                                                            resource="resource:other")])
    stores.work.reconcile()
    [record] = stores.work.failures(contact_id="owner", since=None, until=None)
    assert record["status"] == "failed" and record["failed"] == [("patch", "tool_error")]


def test_failures_honour_the_window_and_a_missing_projection(stores, tmp_path):
    admit(stores, "a", [failed_write()])
    stores.work.reconcile()
    [record] = stores.work.failures(contact_id="owner", since=None, until=None)
    at = wishes_module._utc(record["created_at"])
    assert stores.work.failures(contact_id="owner", since=at + timedelta(seconds=1), until=None) == []
    assert stores.work.failures(contact_id="owner", since=None, until=at) == []
    missing = WorkStore(SimpleNamespace(db_path=tmp_path / "elsewhere" / "turn-idempotency.db"), read_only=True)
    assert missing.failures(contact_id="owner", since=None, until=None) == []


# ---------------------------------------------------------------------------
# The classes: failed work, corrected work, guard blocks, unproven and partial work
# ---------------------------------------------------------------------------

def record(ident, status, failed=(), tools=("write_file",), overlay=None, at=NOON):
    return {"id": ident, "created_at": at.isoformat(), "status": status, "overlay": overlay, "tools": list(tools),
            "failed": list(failed)}


def test_work_classes_count_failed_and_corrected_work_and_never_a_guard_block():
    found = work_classes([
        record("a", "failed", [("write_file", "tool_error"), ("write_file", "tool_error")]),
        record("b", "failed", [("write_file", "tool_error"), ("terminal", "plugin_block")],
               tools=("write_file", "terminal")),
        record("c", "failed", [("terminal", "guardrail_block")], tools=("terminal",)),
        record("d", "proven"),
        record("e", "unproven", [("write_file", "timeout")]),                   # not failed work: no failure
        record("f", "partial", tools=("write_file", "patch"), overlay="corrected"),
        record("g", "proven", tools=("patch",), overlay="corrected"),
        record("h", "proven", tools=(), overlay="corrected"),                   # nothing acted: no class
    ])
    assert set(found) == {WRITE_ERROR, "work:patch:corrected"}
    errors = found[WRITE_ERROR]
    assert [item.id for item in errors["failures"]] == ["work:a", "work:b"]    # one per piece of work
    assert errors["attempts"] == 5                                               # every piece that used the tool
    assert all(item.verified == "check" and item.outcome == "failed" for item in errors["failures"])
    corrected = found["work:patch:corrected"]
    assert [item.id for item in corrected["failures"]] == ["work:f", "work:g"]
    assert all(item.verified == "owner" for item in corrected["failures"]) and corrected["attempts"] == 2
    assert errors["topic"] == "write_file calls that end in a tool error"
    assert corrected["topic"] == "patch work the owner corrected"
    assert all(wishes_module.verified_failure(item) for entry in found.values() for item in entry["failures"])


def test_a_declined_work_class_is_held_30_days_and_a_mind_class_keeps_its_rule():
    declined = SimpleNamespace(outcome="denied", status="denied", verified=None, context={}, ask_code=None,
                               result_metadata={"wish": {"at": NOON.isoformat()}}, cancelled_at=None,
                               completed_at=None, created_at=NOON)
    newer = [WorkFailure(id="work:a", signature=WRITE_ERROR, verified="check", failed_at=NOON + timedelta(days=1),
                         failed_reason="write_file: a tool error")]
    held = wishes_module.Wishes.held
    work = {"signature": WRITE_ERROR, "failures": newer}
    assert held(declined, work, NOON + timedelta(days=2)) == "declined less than 30 days ago"
    assert held(declined, work, NOON + timedelta(days=29, hours=23)) == "declined less than 30 days ago"
    assert held(declined, work, NOON + timedelta(days=30)) is None
    mind = {"signature": SIG, "failures": [SimpleNamespace(failed_at=NOON + timedelta(days=1), created_at=None)]}
    assert held(declined, mind, NOON + timedelta(days=2)) is None               # a newer failure reopens it
    assert held(declined, {**mind, "failures": []}, NOON + timedelta(days=2)) == (
        "declined less than 30 days ago, with no failure since")


# ---------------------------------------------------------------------------
# The night: a wish for a work class, cited by the work log
# ---------------------------------------------------------------------------

def make(tmp_path, monkeypatch, answer=None, *, config=None):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    return Fixture(tmp_path, config=config or WORK, router=WishRouter(answer), at=NOON)


def work(fx, ident, operations, *, at=None):
    """One piece of the owner's work through the real ledger and projection."""
    fx.ledger.record_source(ident, contact_id=OWNER, session_id="session",
                            messages=[{"role": "user", "content": REQUEST}],
                            occurred_at=(at or fx.now - timedelta(hours=1)).isoformat(), derive_claims=False,
                            work={"envelope": envelope(ident, operations=operations), "principal_id": "host"})
    WorkStore(fx.ledger).reconcile()


def correct(fx, ident):
    store = WorkStore(fx.ledger)
    root = store.get_root(ident, contact_id=OWNER)
    assert store.write_derived(ident, contact_id=OWNER, expected_revision=root["revision"], overlay="corrected")


def wanting_class(signature, kind="tool"):
    def answer(prompt):
        label = next(line.split()[0] for line in prompt.splitlines()
                     if re.match(r"c\d+ ", line) and f" {signature} " in line)
        return {"wish": {"class": label, "kind": kind, "cites": cites(prompt, label), "needs_code": False}}
    return answer


async def asked(fx):
    await fx.mind.consolidate()
    await fx.mind.tick(force=True)
    [row] = [row for row in wishes(fx) if state_of(row) == "asked"]
    return row


async def test_failed_work_becomes_a_work_class_the_owner_is_asked_about(tmp_path, monkeypatch):
    fx = make(tmp_path, monkeypatch, wanting_class(WRITE_ERROR))
    work(fx, "r1", [read(), failed_write("w1")], at=fx.now - timedelta(days=3))
    work(fx, "r2", [failed_write("w2")], at=fx.now - timedelta(days=2))
    work(fx, "r3", [write("ok")], at=fx.now - timedelta(days=1))
    row = await asked(fx)
    [prompt] = fx.router.wish_prompts
    assert f"{WRITE_ERROR} ('write_file calls that end in a tool error'): 2 verified failure(s) in 3 attempt(s)" in prompt
    assert "is the owner's work in the work log" in prompt
    assert 'failed: "write_file: a tool error"' in prompt
    assert REQUEST not in prompt                                         # the owner's words never reach the packet
    wish = wish_of(row)
    assert wish["signature"] == WRITE_ERROR and sorted(wish["failures"]) == ["work:r1", "work:r2"]
    assert wish["baseline"] == {**wish["baseline"], "attempts": 3, "failures": 2}
    assert {"work:r1", "work:r2"} <= set(row.context["evidence"])
    assert not any(item.startswith("intention:work:") for item in row.context["evidence"])


async def test_guard_blocks_and_unproven_or_partial_work_raise_no_wish(tmp_path, monkeypatch):
    fx = make(tmp_path, monkeypatch, wanting_class(WRITE_ERROR))
    for n in range(3):
        work(fx, f"g{n}", [guarded(f"g{n}", kind=("plugin_block", "guardrail_block", "scope_block")[n])])
        work(fx, f"u{n}", [unproven(f"u{n}")])
        work(fx, f"p{n}", [write(f"ok{n}"), unproven(f"pu{n}")])
    store = WorkStore(fx.ledger)
    assert {store.get_root(f"u{n}", contact_id=OWNER)["status"] for n in range(3)} == {"unproven"}
    assert {store.get_root(f"p{n}", contact_id=OWNER)["status"] for n in range(3)} == {"partial"}
    assert {store.get_root(f"g{n}", contact_id=OWNER)["status"] for n in range(3)} == {"failed"}
    await fx.mind.consolidate()
    assert wish_calls(fx) == [] and fx.mind.wishes.classes(fx.now) == ([], {})


async def test_corrected_work_is_an_owner_verified_failure_of_its_tool(tmp_path, monkeypatch):
    fx = make(tmp_path, monkeypatch, wanting_class("work:write_file:corrected"))
    for n in range(2):
        work(fx, f"c{n}", [write(f"ok{n}")], at=fx.now - timedelta(days=2 - n))
        correct(fx, f"c{n}")
    row = await asked(fx)
    [prompt] = fx.router.wish_prompts
    assert "work:write_file:corrected ('write_file work the owner corrected'): 2 verified failure(s)" in prompt
    assert prompt.count("corrected by the owner: \"the owner corrected work done with write_file\"") == 2
    assert sorted(wish_of(row)["failures"]) == ["work:c0", "work:c1"]


async def test_a_declined_work_class_waits_the_full_30_days_whatever_fails_after(tmp_path, monkeypatch):
    fx = make(tmp_path, monkeypatch, wanting_class(WRITE_ERROR))
    work(fx, "r1", [failed_write("w1")])
    work(fx, "r2", [failed_write("w2")])
    row = await asked(fx)
    declined = await owner_answers(fx.mind, row.ask_code, "no")
    assert state_of(declined) == "denied"
    for day in range(1, 30):
        fx.shift(days=1)
        work(fx, f"later{day}", [failed_write(f"l{day}")])                   # the work keeps failing daily
        night = await fx.mind.consolidate()
        assert not night["counts"].get("wishes"), day
    assert len(wish_calls(fx)) == 1
    _, passed = fx.mind.wishes.classes(fx.now)
    assert passed == {WRITE_ERROR: "declined less than 30 days ago"}
    fx.shift(days=1)
    work(fx, "later30", [failed_write("l30")])
    await fx.mind.consolidate()
    assert len(wish_calls(fx)) == 2 and WRITE_ERROR in fx.router.wish_prompts[-1]


async def test_a_task_the_owner_rated_wrong_is_a_verified_failure_of_its_class(tmp_path, monkeypatch):
    fx = make(tmp_path, monkeypatch, wanting_class(SIG, kind="data_source"), config=ON)
    rated = [task(fx, outcome="done") for _ in range(2)]
    task(fx, outcome="done")
    for row in rated:
        assert fx.mind.outcomes.rate(row.id, "wrong").verified == "owner"
    row = await asked(fx)
    [prompt] = fx.router.wish_prompts
    assert f"{SIG} ('tide tables'): 2 verified failure(s) in 3 attempt(s)" in prompt
    assert prompt.count("rated wrong by the owner") == 2
    assert sorted(wish_of(row)["failures"]) == sorted(item.id for item in rated)
    assert {f"intention:{item.id}" for item in rated} <= set(row.context["evidence"])


async def test_with_work_records_off_or_the_work_log_unreadable_only_the_minds_tasks_count(tmp_path, monkeypatch):
    fx = make(tmp_path, monkeypatch, config=ON)                     # the fixture's default: work_records off
    assert fx.mind.wishes.work is None
    work(fx, "r1", [failed_write("w1")])
    work(fx, "r2", [failed_write("w2")])
    assert fx.mind.wishes.classes(fx.now) == ([], {})
    on = make(tmp_path / "on", monkeypatch)
    task(on), task(on)

    def unreadable(start, end):
        raise RuntimeError("disk unavailable")
    on.mind.wishes.work = unreadable
    classes, _ = on.mind.wishes.classes(on.now)
    assert [entry["signature"] for entry in classes] == [SIG]


# ---------------------------------------------------------------------------
# Work evidence and capability-wish regression scenarios
# ---------------------------------------------------------------------------

def test_a_cancelled_call_is_no_capability_failure(stores):
    """Finding 4: two pieces of work stopped by a cancellation are two stops someone chose, not two verified
    failures of a ``work:write_file:cancellation`` class."""
    for n in range(2):
        admit(stores, f"cancel-{n}", [failed_write(f"w-{n}", kind="cancellation")])
    stores.work.reconcile()
    rows = stores.work.failures(contact_id="owner", since=None, until=None)
    assert [row["failed"] for row in rows] == [[("write_file", "cancellation")]] * 2
    assert work_classes(rows) == {}


def test_a_correction_is_charged_to_the_last_acting_tool(stores):
    """Finding 5: work done with write_file, then patch, then write_file again, corrected by the owner, is a
    correction of write_file (the last acting tool), not of patch."""
    admit(stores, "corrected", [write("first"), write("middle", tool="patch"), write("last")])
    stores.work.reconcile()
    root = stores.work.get_root("corrected", contact_id="owner")
    stores.work.write_derived("corrected", contact_id="owner", expected_revision=root["revision"], overlay="corrected")
    rows = stores.work.failures(contact_id="owner", since=None, until=None)
    assert rows[0]["tools"] == ["write_file", "patch", "write_file"]
    classes = work_classes(rows)
    assert list(classes) == ["work:write_file:corrected"]
    assert classes["work:write_file:corrected"]["attempts"] == 1              # one piece of work used write_file


async def test_the_work_log_is_read_off_the_event_loop(tmp_path, monkeypatch):
    """Finding 1: consolidation (what ``POST /v1/mind/consolidate`` runs) reads the work log on a worker thread,
    and so does the tick when it measures a declined work class 14 days on."""
    fx = make(tmp_path, monkeypatch, wanting_class(WRITE_ERROR))
    work(fx, "failure-one", [failed_write("one")])
    work(fx, "failure-two", [failed_write("two")])
    loop_thread = threading.get_ident()
    observed = []
    original = WorkStore._read_failures

    def traced(self, **kwargs):
        observed.append(threading.get_ident())
        return original(self, **kwargs)

    monkeypatch.setattr(WorkStore, "_read_failures", traced)
    await fx.mind.consolidate()
    assert len(fx.router.wish_prompts) == 1                    # the work class reached the wish stage
    assert observed and loop_thread not in observed, (loop_thread, observed)
    await fx.mind.tick(force=True)
    [row] = [row for row in wishes(fx) if state_of(row) == "asked"]
    await owner_answers(fx.mind, row.ask_code, "no")
    fx.shift(days=15)
    observed.clear()
    summary = await fx.mind.tick(force=True)
    assert summary["wishes"]["controls"] == 1
    assert observed and loop_thread not in observed, (loop_thread, observed)

