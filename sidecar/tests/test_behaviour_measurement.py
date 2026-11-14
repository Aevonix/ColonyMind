"""P0 fixture gate: canonical captures, exact UTC counts, CLI privacy and read-only stores."""
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
import shutil

import pytest

from colonymind.plasticity.learners.behaviour import (
    DashCounts, OperationRow, ReplyRow, long_dash_counts, reply_counts, tool_counts,
)
from colonymind.reports import plasticity
from colonymind.turns.idempotency import TurnIdempotencyLedger
from colonymind.work.store import WorkStore
from colonymind.work.ledger import admit_event


SENTINEL = "PRIVATE_REPLY"
PROSE = 'PRIVATE_REPLY "quoted—dash" 2–4 – ― ⸺ ⸻ ﹘ alpha - beta -- done.\n'
REPLY = (PROSE + '`hidden—` https://example.invalid/a—b\n> quoted—\n'
         '```py\nhidden—\n```\n~~~\nhidden–\n~~~\n')


def operation(ident, *, tool="edit", success=False, kind=None, model="model-a", call=None):
    value = {"operation_id": ident, "tool_call_id": call or ident, "tool": tool,
             "action": "", "effect": "read", "success": success, "source_refs": []}
    if kind is not None:
        value["failure_kind"] = kind
    if model is not None:
        value["model"] = model
    return value


@pytest.fixture
def instance(tmp_path):
    (tmp_path / "colonymind.yaml").write_text("owner:\n  contact_id: owner\n")
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    WorkStore(ledger)  # A real work projection beside the canonical envelope store, without starting a worker.

    def turn(ident, text="excluded—", *, at="2026-10-01T12:00:00Z", contact="owner", scope="person",
             session="session", channel="chat:conversation", operations=(), messages=None):
        messages = messages if messages is not None else [
            {"role": "user", "content": "PRIVATE_USER " + ident}, {"role": "assistant", "content": text}]
        work = {"envelope": {"version": 1, "root_turn_id": ident, "parent_turn_id": None,
                             "operations": list(operations), "children": [], "truncated": False},
                "principal_id": "host"} if operations else None
        ledger.record_source(ident, contact_id=contact, session_id=session, scope=scope, messages=messages,
                             occurred_at=at, channel_id=channel, work=work, derive_claims=False)

    ops = [operation("ok", success=True), operation("bad", kind="malformed_arguments"),
           operation("duplicate", call="bad", kind="malformed_arguments"),
           operation("error", kind="tool_error")]
    ops += [operation(kind, kind=kind) for kind in (
        "scope_block", "guardrail_block", "plugin_block", "timeout", "cancellation", "refusal")]
    ops += [operation("old"), operation("parse-1", tool="model_response", kind="parse_failure"),
            operation("parse-2", tool="model_response", kind="parse_failure"),
            operation("request-refusal", tool="model_response", kind="refusal"),
            operation("other-model", success=True, model="model-b")]
    turn("first", operations=ops, messages=[{"role": "user", "content": "PRIVATE_USER"},
         {"role": "assistant", "content": "earlier—ignored—"}, {"role": "assistant", "content": REPLY},
         {"role": "assistant", "content": "  "}])
    turn("clean", "clean")
    turn("fenced", "```\n—\n```", at="2026-10-02T00:00:00Z",
         operations=[operation("no-success", success=None, model=None)])
    turn("unknown", "last—", at="2026-10-02T12:00:00Z", channel=None)
    turn("no-reply", at="2026-10-02T12:01:00Z", channel=None,
         messages=[{"role": "user", "content": "PRIVATE_USER no reply"}],
         operations=[operation("unanswered", kind="malformed_arguments", model="model-c")])
    turn("offset", "next", at="2026-10-02T23:30:00-02:00", channel=None)
    turn("self-session", session="mind", operations=[operation("self-call", kind="malformed_arguments")])
    turn("mind:self")
    turn("guest", contact="guest", operations=[operation("guest-call", kind="malformed_arguments")])
    turn("session-scope", scope="session")
    turn("native", messages=[{"role": "user", "content": "PRIVATE_USER observation",
                              "_native_tool_observation": "native-tool-observation-v1"},
                             {"role": "assistant", "content": "excluded—"}])
    turn("empty", messages=[{"role": "user", "content": "PRIVATE_USER empty"},
                            {"role": "assistant", "content": " "}])
    # A revoked envelope cannot contribute tool facts, even if the reply remains captured.
    turn("revoked", messages=[{"role": "user", "content": "PRIVATE_USER revoked"}],
         operations=[operation("revoked-call", kind="malformed_arguments")])
    with closing(ledger._connect()) as conn, conn:
        def event(ident, operations, *, session="session"):
            admit_event(conn, {"event_id": ident, "root_turn_id": "first", "session_id": session,
                              "work": {"version": 1, "root_turn_id": "first", "parent_turn_id": None,
                                       "operations": operations, "children": [], "truncated": False}},
                        contact_id="owner", principal_id="host", occurred_at="2026-10-02T12:00:00Z")

        # A late root-turn event adds a new call; repeated call ids across envelopes count once.
        event("late", [operation("late-bad", kind="malformed_arguments"),
                       operation("late-duplicate", call="bad", kind="malformed_arguments")])
        event("repeat", [operation("repeat-late", call="late-bad", kind="malformed_arguments", model="model-b")])
        event("wrong-session", [operation("wrong-session", kind="malformed_arguments")], session="other")
        event("wrong-contact", [operation("wrong-contact", kind="malformed_arguments")])
        conn.execute("UPDATE work_envelopes SET contact_id='guest' WHERE event_id='event:wrong-contact'")
        event("invalidated", [operation("invalidated", kind="malformed_arguments")])
        conn.execute("UPDATE work_envelopes SET invalidated=1 WHERE event_id='event:invalidated'")
        conn.execute("UPDATE work_envelopes SET invalidated=1 WHERE root_turn_id='revoked'")
        # Older captures lack failure_kind; unexpected argument payloads must never leave the reader either.
        row = conn.execute("SELECT event_id,envelope_json FROM work_envelopes WHERE event_id='turn:first'").fetchone()
        envelope = json.loads(row["envelope_json"])
        envelope["operations"][0]["arguments"] = {"value": "PRIVATE_ARGUMENT"}
        conn.execute("UPDATE work_envelopes SET envelope_json=? WHERE event_id=?", (json.dumps(envelope), row[0]))
    WorkStore(ledger).reconcile()
    return tmp_path, ledger


def test_long_dash_exclusions_and_exact_glyph_counts():
    assert long_dash_counts(REPLY) == DashCounts(66, 7, 1, 2, 1, 1, 1, 1, 1, 1, 1)
    assert long_dash_counts('"—" `—` ``a ` –`` https://example.invalid/— > ―') == (
        DashCounts(10, 2, 1, 0, 1, 0, 0, 0, 0, 0, 0))
    assert long_dash_counts("```\n—\n````\n~~~text\n–\n~~~\n> ―\n").characters == 0
    assert long_dash_counts("```\nunclosed—").long_dashes == 0
    assert long_dash_counts("one - two - three -- 1–2–3").spaced_hyphens == 2
    assert long_dash_counts("one - two - three -- 1–2–3").en_dash_ranges == 2
    assert long_dash_counts("1 - 2").spaced_hyphens == 0


def test_behaviour_fixture_exact_daily_counts_and_cli(instance):
    home, _ = instance
    before = {str(path.relative_to(home)): path.read_bytes() for path in home.rglob("*") if path.is_file()}
    result = plasticity.collect_behaviour(home, since="2026-10-01", until="2026-10-02")
    first, fenced, unknown = result["replies"]
    assert first == {"day": "2026-10-01", "platform": "chat", "model": "unknown",
                     "sessions": 1, "median_replies_per_session": 2,
                     **asdict(DashCounts(71, 7, 1, 2, 1, 1, 1, 1, 1, 1, 1)),
                     "replies": 2, "replies_with_long_dash": 1, "replies_with_substitutes": 1,
                     "long_dash_reply_share": 0.5, "long_dashes_per_1000_characters": 7000 / 71,
                     "substitute_reply_share": 0.5}
    assert fenced == {"day": "2026-10-02", "platform": "chat", "model": "unknown",
                      "sessions": 1, "median_replies_per_session": 1,
                      **asdict(DashCounts(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
                      "replies": 1, "replies_with_long_dash": 0, "replies_with_substitutes": 0,
                      "long_dash_reply_share": 0.0, "long_dashes_per_1000_characters": 0.0,
                      "substitute_reply_share": 0.0}
    assert unknown == {"day": "2026-10-02", "platform": "unknown", "model": "unknown",
                       "sessions": 1, "median_replies_per_session": 1,
                       **asdict(DashCounts(5, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)),
                       "replies": 1, "replies_with_long_dash": 1, "replies_with_substitutes": 0,
                       "long_dash_reply_share": 1.0, "long_dashes_per_1000_characters": 200.0,
                       "substitute_reply_share": 0.0}
    tools, other, missing_success, unanswered = result["tools"]
    assert tools == {"day": "2026-10-01", "platform": "chat", "model": "model-a", "calls": 11,
                     "call_failures": 10, "request_failures": 2, "teachable_failures": 3, "parse_failures": 1,
                     "failure_rate": 10 / 11, "teachable_failure_rate": 3 / 11,
                     "tool_error": 1, "unknown_success": 0,
                     "by_kind": {"malformed_arguments": 2, "tool_error": 1, "unknown": 1,
                                 "scope_block": 1, "guardrail_block": 1, "plugin_block": 1, "timeout": 1,
                                 "cancellation": 1, "refusal": 2, "parse_failure": 1},
                     "top_tools": [{"tool": "edit", "teachable_failures": 2},
                                   {"tool": "model_response", "teachable_failures": 1}]}
    assert other == {"day": "2026-10-01", "platform": "chat", "model": "model-b", "calls": 1,
                     "call_failures": 0, "request_failures": 0, "teachable_failures": 0, "parse_failures": 0,
                     "failure_rate": 0.0, "teachable_failure_rate": 0.0,
                     "tool_error": 0, "unknown_success": 0, "by_kind": {}, "top_tools": []}
    assert missing_success == {**other, "day": "2026-10-02", "model": "unknown", "unknown_success": 1}
    assert unanswered == {**other, "day": "2026-10-02", "platform": "unknown", "model": "model-c",
                          "call_failures": 1, "teachable_failures": 1, "by_kind": {"malformed_arguments": 1},
                          "failure_rate": 1.0, "teachable_failure_rate": 1.0,
                          "top_tools": [{"tool": "edit", "teachable_failures": 1}]}
    assert result["sources"]["reply_platform_column"] == "source_channels.platform"
    assert result["sources"]["reply_model_column"] is None
    assert result["sources"]["reply_platform_available"] is True
    assert result["sources"]["reply_model_available"] is False
    assert result["sources"]["operation_model_available"] is True
    assert result["sources"]["work_envelopes_available"] is True

    for flags in (["--json"], [], ["--format", "html"]):
        run = subprocess.run([sys.executable, "-m", "colonymind", "--instance", str(home), "report", "behaviour",
                              "--since", "2026-10-01", "--until", "2026-10-02", *flags],
                             env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        for sentinel in (SENTINEL, "PRIVATE_USER", "PRIVATE_ARGUMENT", "earlier", "hidden", "conversation"):
            assert sentinel not in run.stdout + run.stderr
        if flags == ["--json"]:
            assert json.loads(run.stdout) == result
        else:
            assert "11 calls; 10 call failures; 2 request failures; 3 teachable failures" in run.stdout
            assert "2026-10-01 / chat / unknown: 2 replies; 1 with long dashes (50.00%)" in run.stdout
            assert "source_channels.platform" in run.stdout
    after = {str(path.relative_to(home)): path.read_bytes() for path in home.rglob("*") if path.is_file()}
    assert after == before


def test_no_implicit_time_window_or_row_cap_and_utc_dates(instance):
    home, ledger = instance
    with closing(ledger._connect()) as conn, conn:
        source = conn.execute("SELECT * FROM turn_sources WHERE turn_id='clean'").fetchone()
        for index in range(405):
            conn.execute("INSERT INTO turn_sources VALUES (?,?,?,?,?,?,?,?)",
                         (f"old-{index}", source["content_sha256"], "owner", "old", "person",
                          source["messages_json"], "2020-01-01T00:00:00Z", "2026-10-03T00:00:00Z"))
    all_rows = plasticity.collect_behaviour(home)
    assert all_rows["replies"][0]["day"] == "2020-01-01"
    assert all_rows["replies"][0]["replies"] == 405
    last = plasticity.collect_behaviour(home, since="2026-10-03", until="2026-10-03")
    assert len(last["replies"]) == 1
    assert last["replies"][0]["day"] == "2026-10-03"
    assert last["replies"][0]["replies"] == 1


def test_metadata_comes_only_from_columns_in_the_same_store(instance):
    home, ledger = instance
    with closing(ledger._connect()) as conn, conn:
        conn.execute("DROP TABLE source_channels")
    older = plasticity.collect_behaviour(home)
    assert all(row["platform"] == row["model"] == "unknown" for row in older["replies"])
    assert older["sources"]["reply_platform_column"] is None
    assert older["sources"]["reply_platform_available"] is False
    with closing(ledger._connect()) as conn, conn:
        conn.execute("ALTER TABLE turn_sources ADD COLUMN platform TEXT")
        conn.execute("ALTER TABLE turn_sources ADD COLUMN model TEXT")
        conn.execute("UPDATE turn_sources SET platform='console',model='model-z' WHERE turn_id='first'")
    recorded = plasticity.collect_behaviour(home)
    assert recorded["sources"]["reply_platform_column"] == "turn_sources.platform"
    assert recorded["sources"]["reply_model_column"] == "turn_sources.model"
    assert recorded["sources"]["reply_platform_known"] == recorded["sources"]["reply_model_known"] == 1
    first = next(row for row in recorded["replies"] if row["model"] == "model-z")
    assert first["platform"] == "console"
    assert first["replies"] == 1


def test_call_and_parse_deduplication_crosses_model_groups():
    at = datetime(2026, 10, 1, tzinfo=timezone.utc)
    rows = [OperationRow(at, "chat", "model-a", "turn", "same-call", "read", False, "malformed_arguments"),
            OperationRow(at, "chat", "model-b", "turn", "same-call", "read", False, "malformed_arguments"),
            OperationRow(at, "chat", "model-b", "other", "same-call", "read", False, "malformed_arguments"),
            OperationRow(at, "chat", "model-a", "turn", "parse-1", "model_response", False, "parse_failure"),
            OperationRow(at, "chat", "model-b", "turn", "parse-2", "model_response", False, "parse_failure"),
            OperationRow(at, "chat", "model-b", "other", "parse-3", "model_response", False, "parse_failure")]
    first, second = tool_counts(rows)
    assert first["calls"] == 1
    assert first["call_failures"] == first["request_failures"] == 1
    assert first["teachable_failures"] == 2
    assert first["failure_rate"] == 1.0
    assert first["teachable_failure_rate"] == 2.0
    assert first["parse_failures"] == 1
    assert second["calls"] == 0
    assert second["call_failures"] == 0
    assert second["request_failures"] == second["teachable_failures"] == second["parse_failures"] == 1
    assert second["failure_rate"] is second["teachable_failure_rate"] is None


def test_median_replies_per_session_with_even_population():
    at = datetime(2026, 10, 1, tzinfo=timezone.utc)
    rows = [ReplyRow(at, "chat", "unknown", "reply", session_id=session)
            for session in ("one", "three", "three", "three")]
    assert reply_counts(rows)[0]["median_replies_per_session"] == 2


def test_live_wal_snapshot_preserves_all_store_bytes_and_sees_wal_rows(instance, tmp_path, monkeypatch):
    home, ledger = instance
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    monkeypatch.setenv("TMPDIR", str(snapshots))
    monkeypatch.setattr(tempfile, "tempdir", str(snapshots))

    def fingerprints():
        return {str(path.relative_to(home)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in home.rglob("*") if path.is_file()}

    with closing(ledger._connect()) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        writer.execute("INSERT INTO turn_sources SELECT 'wal-only',content_sha256,contact_id,'wal-session',"
                       "scope,messages_json,'2026-10-04T00:00:00Z',ingested_at "
                       "FROM turn_sources WHERE turn_id='clean'")
        writer.commit()
        # Prove this row is absent from the main file and exists only in committed WAL frames.
        uri = (home / "turn-idempotency.db").as_uri() + "?mode=ro&immutable=1"
        with closing(sqlite3.connect(uri, uri=True)) as main_only:
            assert main_only.execute("SELECT count(*) FROM turn_sources WHERE turn_id='wal-only'").fetchone()[0] == 0
        before = fingerprints()
        assert {"turn-idempotency.db", "turn-idempotency.db-wal", "turn-idempotency.db-shm"} <= before.keys()
        result = plasticity.collect_behaviour(home, since="2026-10-04", until="2026-10-04")
        assert result["sources"]["ledger_available"] is True
        assert len(result["replies"]) == 1
        assert result["replies"][0]["replies"] == 1
        assert result["replies"][0]["day"] == "2026-10-04"
        assert fingerprints() == before
        assert not list(snapshots.iterdir())


def _wal_writer(ledger):
    writer = ledger._connect()
    assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return writer


def _add_turn(writer, turn_id):
    writer.execute("INSERT INTO turn_sources SELECT ?,content_sha256,contact_id,?,scope,messages_json,"
                   "'2026-10-04T00:00:00Z',ingested_at FROM turn_sources WHERE turn_id='clean'",
                   (turn_id, f"session-{turn_id}"))
    writer.commit()


def test_snapshot_accepts_wal_growth_during_copy(instance, tmp_path, monkeypatch):
    """A live writer appending to the WAL while the database copies is the normal case on a busy store."""
    home, ledger = instance
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    copy, calls = shutil.copyfile, []
    with closing(_wal_writer(ledger)) as writer:
        _add_turn(writer, "wal-before")

        def copying(source, target, *args, **kwargs):
            result = copy(source, target, *args, **kwargs)
            calls.append(Path(source).name)
            if Path(source).name == "turn-idempotency.db":
                _add_turn(writer, "wal-during")      # appended after the database copy, before the WAL copy
            return result

        monkeypatch.setattr(shutil, "copyfile", copying)
        result = plasticity.collect_behaviour(home, since="2026-10-04", until="2026-10-04")
    assert calls == ["turn-idempotency.db", "turn-idempotency.db-wal"]     # one attempt, WAL last
    assert sum(row["replies"] for row in result["replies"]) == 2
    assert sum(row["sessions"] for row in result["replies"]) == 2


def test_snapshot_retries_when_a_checkpoint_lands_between_copies(instance, tmp_path, monkeypatch):
    """A checkpoint rewrites the database and restarts the WAL: that copy would mix generations, so it is retried."""
    home, ledger = instance
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    copy, calls = shutil.copyfile, []
    with closing(_wal_writer(ledger)) as writer:
        _add_turn(writer, "wal-before")

        def copying(source, target, *args, **kwargs):
            result = copy(source, target, *args, **kwargs)
            calls.append(Path(source).name)
            if calls == ["turn-idempotency.db"]:
                writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                _add_turn(writer, "wal-new-generation")
            return result

        monkeypatch.setattr(shutil, "copyfile", copying)
        result = plasticity.collect_behaviour(home, since="2026-10-04", until="2026-10-04")
    assert calls.count("turn-idempotency.db") == 2                         # the mixed copy was rejected
    assert sum(row["replies"] for row in result["replies"]) == 2
    assert not list(tmp_path.glob("colonymind-behaviour-*"))


@pytest.mark.parametrize("missing", ["source_attribution_invalidations", "work_envelopes"])
def test_missing_tool_dependencies_preserve_reply_counts(instance, missing):
    home, ledger = instance
    replies = plasticity.collect_behaviour(home)["replies"]
    with closing(ledger._connect()) as conn, conn:
        conn.execute(f"DROP TABLE {missing}")
    result = plasticity.collect_behaviour(home)
    assert result["replies"] == replies
    assert result["tools"] == []
    assert result["sources"]["ledger_available"] is True
    assert result["sources"]["tool_coverage"] == "unavailable"
    assert result["sources"]["missing_tool_tables"] == [missing]
    assert f"Tool coverage: unavailable; missing tables: {missing}." in plasticity.behaviour_document(result).text()


@pytest.mark.parametrize("bounds", [{"since": "yesterday"}, {"until": "2026-02-30"},
                                    {"since": "2026-10-02", "until": "2026-10-01"}])
def test_date_errors_before_opening_stores(tmp_path, bounds):
    with pytest.raises(ValueError):
        plasticity.collect_behaviour(tmp_path, **bounds)
    assert not list(tmp_path.iterdir())


def test_missing_and_unreadable_stores(tmp_path):
    result = plasticity.collect_behaviour(tmp_path)
    assert result["replies"] == result["tools"] == []
    assert result["sources"]["ledger_available"] is False
    assert not list(tmp_path.iterdir())
    (tmp_path / "turn-idempotency.db").write_bytes(b"not a database PRIVATE_USER")
    with pytest.raises(ValueError, match="counts are unknown"):
        plasticity.collect_behaviour(tmp_path)
