"""Controller-created hands keep owner provenance through the real approval hold."""

from contextlib import closing
from datetime import timedelta
import hashlib
from pathlib import Path

import pytest

from colonymind.hands import approvals, wire
from colonymind.mind.outbox import ASK_MESSAGE
from test_hands_controller import EPOCH, c  # noqa: F401
from test_mind_loop import OWNER, Fixture, pinned_clock  # noqa: F401


@pytest.fixture
def mind_case(c, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    case = Fixture(Path(c.store.path).parent, config={"faculties": {"owner_gate": True}, "quiet_hours": ""})
    c.test_time[0] = case.now
    c.health.record("family-a", running=0, waiting=0, kv=0.1)
    yield case
    case.store.close()


def record_turn(case, turn, *, sender, platform, started, replied):
    case.ledger.record_source(turn, contact_id=sender, session_id="parent-session",
        channel_id=f"{platform}:{sender}", occurred_at=started.isoformat(), messages=[
            {"role": "user", "content": "Remove the temporary file after I approve."},
            {"role": "assistant", "content": "The hand will ask before removing it."}])
    with closing(case.ledger._connect()) as conn:
        conn.execute("UPDATE turn_sources SET ingested_at=? WHERE turn_id=?", (replied.isoformat(), turn))
        conn.commit()


@pytest.mark.parametrize("platform", ["whatsapp", "sms", "rcs", "api"])
@pytest.mark.parametrize("origin,sender,expected", [
    ("owner_turn", OWNER, "send"),
    ("owner_turn", "guest-contact", "wait"),
    ("mind", OWNER, "wait"),
    ("harness", OWNER, "wait"),
])
async def test_controller_hand_approval_uses_its_recorded_owner_turn(c, mind_case, platform, origin, sender, expected):
    case = mind_case
    started = case.now
    replied = started + timedelta(seconds=24.9)
    turn = "captured-origin-turn"
    chat = "parent-session" if platform == "api" else "origin-chat"
    message = turn if platform == "api" else "platform-origin-message"
    reference = {"platform": platform, "chat": chat, "message_id": message}
    read = await case.mind.read_answer({
        "platform": platform, "contact_id": sender, "chat_id": chat, "chat_type": "dm",
        "turn_id": turn, "message_id": "" if platform == "api" else message,
        "arrived_at": started.isoformat(), "text": "Remove the temporary file after I approve.", "ledger": None})
    assert read["read"] is (sender == OWNER)
    record_turn(case, turn, sender=sender, platform=platform, started=started, replied=replied)
    if sender != OWNER:
        # A guest cannot remove the pause created by an unrelated owner conversation.
        record_turn(case, "other-owner-turn", sender=OWNER, platform=platform, started=started, replied=replied)

    planned = await c.plan(wire.PlanRequest(parent_session="parent-session", turn=turn, origin=origin,
        origin_ref=reference, tasks=[{"goal": "Remove the temporary file with approval", "kind": "coding",
                                      "workdir": str(case.state)}]),
        "origin-plan")
    plan, = planned["plans"]
    assert plan["admit"] == "start", plan
    await c.dispatched(plan["hand_id"], wire.DispatchedRequest(
        epoch=EPOCH, delegation_id="delegation", lease_token=plan["lease_token"]), "origin-dispatch")
    await c.bind(plan["hand_id"], wire.BindRequest(
        epoch=EPOCH, executor_ref="worker", child_session="worker-session"), "origin-bind")

    target = case.state / "temporary-file.txt"
    target.write_text("Temporary test content")
    case.now = started + timedelta(seconds=25.4)
    body = wire.HoldRequest(epoch=EPOCH, call={"tool": "terminal", "arguments": {"command": f"rm {target}"}},
        workdir=str(target.parent), targets=[{"kind": "file", "target": str(target),
                                             "value": hashlib.sha256(target.read_bytes()).hexdigest()}])
    held = await approvals.Approvals(c.store, case.mind).hold(plan["hand_id"], body, "origin-hold")
    note = case.store.get(held.ask_id)
    assert (note.kind, note.type, note.cls) == ("note", "floor_effect", "floor")
    assert note.context["held_from"] == plan["hand_id"]
    if platform == "rcs" and origin == "owner_turn":
        assert note.context["delivery_ref"] == reference
    else:
        assert "delivery_ref" not in note.context

    case.now = started + timedelta(seconds=27)
    await case.mind._deliver_asks([note], case.now)
    notice, = case.store.intentions(kind=["message"], types=[ASK_MESSAGE])
    case.now = started + timedelta(seconds=30)
    assert case.now - replied > timedelta(seconds=5)
    ready = await case.mind.outbox_ready()
    assert (notice.id in [item["id"] for item in ready]) is (expected == "send")
    verdict = case.mind.owner_gate.ledger.last_for(notice.id)
    assert (verdict["decision"], verdict["stage"]) == (
        expected, "prompted" if expected == "send" else "pause")
    if expected == "wait":
        assert verdict["until"] == (replied + timedelta(minutes=10)).isoformat()
    case.restart()
    assert (notice.id in [item["id"] for item in await case.mind.outbox_ready()]) is (expected == "send")
    assert case.mind.owner_gate.ledger.last_for(notice.id)["id"] == verdict["id"]
    assert target.read_text() == "Temporary test content"
