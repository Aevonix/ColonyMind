"""The front learns why the controller refused its plan, and shell-only coding runs as ops."""

import json

from test_hands_auto_resume_effect import world  # noqa: F401
from test_hands_permits import event_loop, pinned_clock  # noqa: F401


def test_shell_only_coding_without_a_workdir_starts_as_an_ops_hand(world):
    """Gate w18-20261007T182631Z: WhatsApp and RCS fronts chose coding for a plain command."""
    answer, = world.runtime.dispatch_tasks([{"goal": "Run the fixture command and report its output",
                                             "kind": "coding", "needs": ["shell"]}], parent=world.parent)["hands"]
    assert answer["status"] == "dispatched" and answer["kind"] == "ops"
    assert answer["kind_adjusted"] == {"from": "coding", "to": "ops"}
    hand = world.runtime.registry.get(answer["hand_id"])
    assert hand.kind == "ops" and len(world.native.calls) == 1
    row = world.run(world.current.get_hand(answer["hand_id"]))
    assert row["kind"] == "ops" and row["decisions"]["kind_adjusted"]["reason"] == "coding_without_workdir_or_resources"


def test_a_refused_plan_carries_the_reason_and_hint_and_the_corrected_plan_runs(world):
    task = {"goal": "Patch the fixture script", "kind": "coding", "needs": ["code", "shell"],
            "context": "The script is described in /home/owner/private/notes.txt"}
    answer, = world.runtime.dispatch_tasks([task], parent=world.parent, tool_call_id="first")["hands"]
    assert answer["status"] == "refused" and answer["detail"] == "coding_requires_workdir_or_resources"
    assert "use kind ops" in answer["hint"]
    assert "/home/owner" not in json.dumps(answer) and not world.native.calls
    assert not world.runtime.degraded.is_degraded
    corrected, = world.runtime.dispatch_tasks([{**task, "kind": "ops"}], parent=world.parent,
                                              tool_call_id="second")["hands"]
    assert corrected["status"] == "dispatched" and "kind_adjusted" not in corrected
    assert len(world.native.calls) == 1
