"""Verification through the actual Hermes hooks, durable outbox and wire schema."""
from contextlib import closing
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from colonymind.api.schemas.host import TurnSyncRequest
from colonymind.mind import drift
from test_host_identity_policy import LONG
from test_mind_drift import world


@pytest.fixture
def adapter(monkeypatch, world, tmp_path):
    # Load this checkout, without registering a live host plugin.
    name = "host_identity_capture_adapter"
    package = ModuleType(name)
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "plugins/hermes-plugin")]
    monkeypatch.setitem(sys.modules, name, package)
    capture_module = importlib.import_module(name + ".capture")
    settings = importlib.import_module(name + ".client").Settings(
        "http://unreachable.invalid", tmp_path / "key", "test-key", tmp_path, tmp_path,
        tmp_path / "outbox.sqlite3")
    monkeypatch.setattr(settings, "owner_contact_id", lambda: world.mind.owner_id)
    monkeypatch.setattr(capture_module.time, "time", lambda: world.now.timestamp())

    class Client:
        bodies = []

        def post(self, path, *, json, **kwargs):
            assert path == "/v1/host/turns/sync"
            body = TurnSyncRequest.model_validate(json)
            self.bodies.append(body)
            admitted = {}
            world.ledger.record_source(body.context.turn_id, contact_id=body.context.contact_id,
                session_id=body.context.session_id,
                messages=[message.model_dump(exclude_none=True)
                          for message in (body.user_message, body.assistant_message) if message],
                occurred_at=body.context.metadata["occurred_at"], derive_claims=False,
                work={"envelope": body.work.model_dump(exclude_none=True), "principal_id": "host"},
                work_result=admitted)
            assert admitted["recorded"] is True
            return SimpleNamespace(is_success=True, status_code=200,
                                   json=lambda: {"accepted": True, "work_recorded": True})

    client = Client()
    sessions = capture_module.SessionMap(settings, client)
    outbox = capture_module.TurnOutbox(settings.outbox_path)
    capture = capture_module.Capture(sessions, outbox, settings)
    yield SimpleNamespace(capture=capture, client=client, module=capture_module)
    for key in list(sys.modules):
        if key.startswith(name + "."):
            del sys.modules[key]


def capture_turn(adapter, *, turn="turn", session="session", result="", status="ok", finish=True):
    capture = adapter.capture
    values = {"session_id": session, "turn_id": turn, "platform": "cli",
              "user_message": "Inspect the source."}
    capture.pre_llm_call(**values)
    if finish:
        capture.work.post_tool_call(session_id=session, turn_id=turn, tool_call_id="call",
                                    tool_name="read_file", args={"path": "source.txt"},
                                    result=result, status=status)
    capture.post_llm_call(**values, assistant_response=LONG)
    # Read the persisted payload back, exactly as the delivery body does after restart.
    restarted = adapter.module.TurnOutbox(capture.settings.outbox_path)
    restarted.drain(lambda payload: adapter.module.deliver(
        adapter.client, capture.sessions, capture.settings, payload))


@pytest.mark.parametrize("result,status,completed,verified", [
    ("", "ok", True, True), ("Returned bytes", "ok", True, True),
    ({}, "success", True, True), ("", "error", True, False),
    ("", "", True, False), ("", "ok", False, False),
])
def test_completed_invocations_verify_through_real_capture(adapter, world, result, status, completed, verified):
    for index in range(3):
        capture_turn(adapter, turn=f"turn-{index}", result=result, status=status, finish=completed)
    assert len(adapter.client.bodies) == 3
    # Ordinary capture contains only user/assistant messages; receipts must come from work.
    assert all(body.user_message.role == "user" and body.assistant_message.role == "assistant"
               for body in adapter.client.bodies)
    reading = drift.signals(world.mind.drift._replies(world.now), [])["missing_verification"]
    assert reading["value"] == (0.0 if verified else 1.0)
    assert bool(reading["receipts"]) is verified
    if verified:
        assert all(ref.startswith("work:turn:turn-") for refs in reading["receipts"].values() for ref in refs)


@pytest.mark.parametrize("content,call,new_instruction", [
    ("Returned bytes", "call", False), ("", "call", False),
    ("Returned bytes", "orphan", False), ("Returned bytes", "call", True),
])
def test_transcript_completions_match_a_recorded_invocation(world, content, call, new_instruction):
    rows = [{"role": "user", "content": "Inspect"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call", "function": {"name": "read_file", "arguments": "{}"}}]}]
    if new_instruction:
        rows.append({"role": "user", "content": "A new instruction"})
    rows += [{"role": "tool", "content": content, "tool_call_id": call},
             {"role": "assistant", "content": LONG}]
    world.ledger.record_source("turn", contact_id=world.mind.owner_id, session_id="session", messages=rows,
                               occurred_at=world.now.isoformat(), derive_claims=False)
    assert world.mind.drift._replies(world.now)[0].tool_receipts == (
        ["tool:call"] if call == "call" and not new_instruction else [])


@pytest.mark.parametrize("change", ['session', 'instruction', 'invalidated', 'late'])
def test_work_receipts_cannot_escape_their_turn_and_session(adapter, world, change):
    capture_turn(adapter)
    assert world.mind.drift._replies(world.now)[0].tool_receipts
    world.turn("unrelated", "Another instruction", LONG, session="session")
    assert {item.turn_id: item.tool_receipts for item in world.mind.drift._replies(world.now)}["unrelated"] == []
    with closing(world.ledger._connect()) as db:
        if change == 'session':
            db.execute("UPDATE turn_sources SET session_id='different' WHERE turn_id='turn'")
        elif change == 'instruction':
            messages = json.loads(db.execute("SELECT messages_json FROM turn_sources WHERE turn_id='turn'").fetchone()[0])
            messages[0]['content'] = 'Different instruction'
            db.execute("UPDATE turn_sources SET messages_json=? WHERE turn_id='turn'", (json.dumps(messages),))
        elif change == 'invalidated':
            db.execute('UPDATE work_envelopes SET invalidated=1')
        else:
            db.execute("UPDATE work_envelopes SET kind='event'")
        db.commit()
    assert all(item.tool_receipts == [] for item in world.mind.drift._replies(world.now))
