"""The recorder's store: lines are kept in one file of labels and timings, and a store that is missing or locked
never reaches the caller."""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

import pytest

from colonymind import recorder
from colonymind.recorder import store as store_module
from colonymind.recorder.store import DB_FILE

DOC = Path(__file__).resolve().parents[2] / "docs" / "RECORDER.md"


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    return tmp_path


def test_lines_are_kept_and_read_back(state):
    event = recorder.begin("event", "whatsapp", ref="ingress:17")
    recorder.finish(event, result="silent", choice="ignore", chooser="model")
    check = recorder.once("check", "check.mail", result="done", choice="ignore", chooser="rule", took_ms=12)
    card = recorder.begin("card", "mind.task", ref="initiative:42")
    assert recorder.mark("initiative:42", "owner_acted") == 1
    assert recorder.mark("initiative:42", "owner_said_no_need") == 1
    assert recorder.mark("initiative:43", "owner_acted") == 0

    store_module._STORES.clear()                       # a new process reads the same file
    assert (state / DB_FILE).is_file()
    lines = recorder.recent()
    assert [line["id"] for line in lines] == [card, check, event]
    assert lines[0] == {"id": card, "kind": "card", "source": "mind.task", "ref": "initiative:42", "choice": "",
                        "chooser": "", "result": "", "detail": "", "marks": ["owner_acted", "owner_said_no_need"],
                        "began_at": lines[0]["began_at"], "took_ms": None}
    assert (lines[1]["result"], lines[1]["took_ms"]) == ("done", 12)
    assert (lines[2]["choice"], lines[2]["chooser"], lines[2]["result"]) == ("ignore", "model", "silent")
    assert lines[2]["took_ms"] >= 0
    assert [line["id"] for line in recorder.recent(5, kind="check")] == [check]
    assert [line["id"] for line in recorder.recent(5, source="whatsapp")] == [event]
    assert len(recorder.recent(1)) == 1


def test_a_line_holds_labels_only(state):
    recorder.once("event", "whatsapp", result="sent", ref="+1 555 0100", detail="Hello there, how are you?",
                  choice="shout", chooser="someone")
    line = recorder.recent(1)[0]
    assert (line["ref"], line["detail"], line["choice"], line["chooser"]) == ("", "other", "other", "other")
    recorder.once("check", "Check.Mail", result="failed", detail="TimeoutError")
    line = recorder.recent(1)[0]
    assert (line["source"], line["detail"]) == ("other", "TimeoutError")     # an error class is a label
    with sqlite3.connect(state / DB_FILE) as conn:
        tables = [name for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        dump = repr(conn.execute("SELECT * FROM lines").fetchall())
    assert tables == ["lines"]
    assert "Hello" not in dump and "555" not in dump


def test_a_missing_store_never_raises(state, monkeypatch):
    recorder.once("check", "check.mail", result="done")
    for path in state.glob(DB_FILE + "*"):             # the file goes away: the next call starts a new one
        path.unlink()
    recorder.once("check", "check.mail", result="failed")
    assert [line["result"] for line in recorder.recent()] == ["failed"]

    blocker = state / "not-a-directory"
    blocker.write_text("")
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(blocker / "state"))    # no store can exist here
    line = recorder.begin("event", "sms")
    assert len(line) == 32
    assert recorder.finish(line, result="sent") is None
    assert len(recorder.once("check", "check.mail", result="done")) == 32
    assert recorder.mark("ingress:1", "owner_acted") == 0
    assert recorder.recent() == []


def test_a_locked_store_never_raises_and_leaves_an_error_line(state):
    line = recorder.begin("event", "sms", ref="ingress:1")
    lock = sqlite3.connect(state / DB_FILE)
    lock.execute("BEGIN EXCLUSIVE")
    lock.execute("UPDATE lines SET detail = detail")
    started = time.monotonic()
    try:
        assert len(recorder.begin("event", "sms")) == 32
        assert recorder.finish(line, result="sent") is None
        assert len(recorder.once("check", "check.mail", result="done")) == 32
        assert recorder.mark("ingress:1", "owner_acted") == 0
        assert [row["id"] for row in recorder.recent()] == [line]
    finally:
        lock.rollback()
        lock.close()
    assert time.monotonic() - started < 3               # each call gave up after a short wait

    recorder.once("check", "check.mail", result="done")  # the store works again: the failures become error lines
    errors = recorder.recent(kind="error")
    assert len(errors) == 4
    assert {(row["source"], row["result"], row["detail"]) for row in errors} == {
        ("recorder.store", "failed", "OperationalError")}
    assert recorder.recent(kind="event")[0]["result"] == ""    # the lost finish left its line open


def test_the_documented_query_reads_the_lines_as_labelled_examples(state):
    recorder.once("event", "whatsapp", result="silent", choice="ignore", chooser="model", ref="ingress:1")
    recorder.once("event", "whatsapp", result="sent", choice="reply", chooser="model", ref="ingress:2")
    card = recorder.begin("card", "mind.task", ref="initiative:3")
    recorder.finish(card, result="done", choice="act", chooser="model")
    recorder.begin("event", "sms", ref="ingress:4")                      # still open: not an example yet
    recorder.mark("ingress:1", "owner_said_should_have")
    recorder.mark("ingress:2", "owner_said_no_need")
    recorder.mark("initiative:3", "owner_acted")
    query = re.findall(r"```sql\n(SELECT.*?)```", DOC.read_text(encoding="utf-8"), flags=re.DOTALL)[-1]
    with sqlite3.connect(state / DB_FILE) as conn:
        rows = conn.execute(query).fetchall()
    assert rows == [("ingress:1", "event", "whatsapp", "model", "ignore", "silent", 1, 0, 0, 0),
                    ("ingress:2", "event", "whatsapp", "model", "reply", "sent", 0, 1, 0, 0),
                    ("initiative:3", "card", "mind.task", "model", "act", "done", 0, 0, 1, 0)]
