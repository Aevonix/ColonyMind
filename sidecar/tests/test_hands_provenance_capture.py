"""OC4 capture and ordinary recall share the persisted assistant metadata."""
from importlib import import_module
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from colonymind.memory.recall import render_memory_context, source_candidates
from colonymind.mind.lessons import Lessons
from colonymind.mind.outcomes import Autobiography
from colonymind.turns import TurnIdempotencyLedger


def adapter_capture():
    name = "oc4_capture_adapter"
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[2] / "plugins/hermes-plugin")]
        sys.modules[name] = package
    return import_module(name + ".capture")


@pytest.fixture
def captured(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))  # the server opens the turn ledger here
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    module = adapter_capture()
    settings = SimpleNamespace(home=tmp_path, outbox_path=tmp_path / "outbox.db", owner_contact_id=lambda: "owner",
                               mind=lambda: {"faculties": {"work_records": False}})
    outbox = module.TurnOutbox(tmp_path / "outbox.db")
    capture = module.Capture(module.SessionMap(settings, None), outbox, settings)
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    return module, capture, outbox, settings, ledger


def digest(level="E0"):
    line = "Ledger (deep, text): done; " + level + " self_report"
    return {"text": "Your hands:\n- " + line, "items": [
        {"hand_id": "h-visible", "text": line,
         "evidence": {"level": level, "scope": {"kind": "self_report"}}},
        {"hand_id": "h-hidden", "text": "Hidden (deep, text): done; E5 owner",
         "evidence": {"level": "E5", "scope": {"kind": "owner"}}}]}


def test_only_rendered_results_attach_to_the_exact_assistant_turn(captured):
    module, capture, outbox, settings, ledger = captured
    capture.record_hand_refs(session_id="parent", turn_id="turn-1", digest=digest())
    capture.post_llm_call(session_id="parent", turn_id="turn-other",
                          assistant_response="An unrelated reply.")
    capture.post_llm_call(session_id="parent", turn_id="turn-1", user_message="Check the valve.",
                          assistant_response="I verified the valve is ready.",
                          hand_refs=[{"hand_id": "forged", "evidence_level": "E5"}])
    payloads = {row["turn_id"]: row["payload"] for row in outbox.rows()}
    assert "hand_refs" not in payloads["turn-other"]
    refs = payloads["turn-1"]["hand_refs"]
    assert [(r["hand_id"], r["evidence_level"]) for r in refs] == [("h-visible", "E0")]

    class Client:
        def post(self, path, *, json, timeout):
            assert path == "/v1/host/turns/sync"
            assert "metadata" not in json["user_message"]
            assert json["assistant_message"]["metadata"]["hand_refs"] == refs
            ledger.record_source(json["context"]["turn_id"], contact_id="owner", session_id="parent",
                                 messages=[json["user_message"], json["assistant_message"]])
            return SimpleNamespace(status_code=200, is_success=True, json=lambda: {})

    assert module.deliver(Client(), capture.sessions, settings, payloads["turn-1"])
    hits = source_candidates(ledger.search_sources("valve", contact_id="owner", session_id="parent"))
    reply = next(row for row in hits if row["role"] == "assistant")
    rendered = render_memory_context([reply])
    assert "from Ledger's result" in rendered
    assert "own analysis, not checked" in rendered
    assert reply["kind"] == "source_quote"
    with ledger._connect() as connection:
        assert connection.execute("SELECT count(*) FROM source_claims").fetchone()[0] == 0


def test_capture_retry_retains_metadata_and_does_not_leak_to_next_turn(captured):
    _, capture, outbox, _, _ = captured
    capture.record_hand_refs(session_id="parent", turn_id="turn-1", digest=digest())
    for _ in range(2):
        capture.post_llm_call(session_id="parent", turn_id="turn-1", assistant_response="The valve is ready.")
    capture.post_llm_call(session_id="parent", turn_id="turn-2", assistant_response="Next task.")
    payloads = {row["turn_id"]: row["payload"] for row in outbox.rows()}
    assert payloads["turn-1"]["hand_refs"][0]["evidence_level"] == "E0"
    assert "hand_refs" not in payloads["turn-2"]


def test_hand_self_report_cannot_enter_lessons_as_a_claimed_check(captured):
    _, _, _, _, ledger = captured
    ledger.record_source("relay", contact_id="owner", session_id="parent", messages=[{
        "role": "assistant", "content": "The valve is ready.", "metadata": {
            "hand_refs": [{"hand_id": "h-visible", "evidence_level": "E0"}]}}])
    lessons = Lessons(ledger=ledger, store=None, owner_id="owner",
                      autobiography=Autobiography(ledger, owner_id="owner"))
    fields = {"signature": "valve", "kind": "strategy", "title": "Check the valve",
              "when_to_use": "Before opening", "content": "Read the ready signal."}
    assert lessons.admit(fields, verified="check", origin="night", status="active",
                         evidence=["relay"], lineage=["relay"]) is None
    assert lessons.admit(fields, verified="check", origin="hand", status="active",
                         evidence=[], lineage=[]) is None
    assert not lessons.all()


def test_unchecked_relay_cannot_become_a_lesson_witness(captured):
    _, _, _, _, ledger = captured
    ledger.record_source("relay", contact_id="owner", session_id="parent", messages=[{
        "role": "assistant", "content": "The valve is ready.", "metadata": {
            "hand_refs": [{"hand_id": "h-visible", "evidence_level": "E0"}]}}])
    lessons = Lessons(ledger=ledger, store=None, owner_id="owner",
                      autobiography=Autobiography(ledger, owner_id="owner"))
    lesson = lessons.admit({"signature": "valve", "kind": "strategy", "title": "Check valve",
        "when_to_use": "Before opening", "content": "Read the signal."}, verified="owner",
        origin="night", status="active", evidence=[], lineage=[])
    assert lesson is not None
    assert lessons.witness(lesson.id, evidence=["relay"], lineage=["relay"],
                           verified="check", origin="night") is None
    assert lessons.get(lesson.id).witnesses == 1


def test_hand_self_report_never_reaches_the_nightly_lesson_feed(captured):
    _, _, _, _, ledger = captured
    row = SimpleNamespace(kind="task", type="work", verified="check", outcome="done", verdict=None,
        success_check={"kind": "commitment_resolved"},
        result_metadata={"origin": "hand", "hand_id": "missing", "level": "E5"})
    lessons = Lessons(ledger=ledger, store=SimpleNamespace(intentions=lambda **kwargs: [row]),
                      owner_id="owner", autobiography=Autobiography(ledger, owner_id="owner"))
    assert lessons.verified_source(row) == "none"
    assert lessons._events(lessons.clock()) == []


def test_hand_provenance_and_work_capture_share_the_same_turn(captured):
    module, capture, outbox, settings, _ = captured
    settings.mind = lambda: {"faculties": {"work_records": True}}
    capture.work.refresh()
    values = {"session_id": "parent", "turn_id": "with-work", "platform": "cli",
              "user_message": "Check the example file."}
    capture.pre_llm_call(**values)
    capture.record_hand_refs(session_id="parent", turn_id="with-work", digest=digest())
    capture.work.post_tool_call(session_id="parent", turn_id="with-work", tool_call_id="write-1",
        tool_name="write_file", args={"path": "/home/user/example.txt"},
        result={"verified": True, "bytes_written": 1}, status="ok")
    capture.post_llm_call(**values, assistant_response="The example file is ready.")
    payload = outbox.rows()[0]["payload"]
    assert payload["hand_refs"][0]["hand_id"] == "h-visible"
    assert payload["work"]["operations"][0]["evidence"]["verified"] is True
    assert "example.txt" not in json.dumps(payload["work"])
