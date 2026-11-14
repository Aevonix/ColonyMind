"""Cross-plugin provenance checks with the memory hooks running before hands."""

from types import SimpleNamespace

import pytest

from test_hands_front import rig  # noqa: F401
from test_hands_front_capture import adapter  # noqa: F401


def connect(rig, adapter):
    rig.runtime.sessions = adapter.sessions
    rig.runtime.authorized_owner = lambda sid: adapter.sessions.is_owner(sid) is True
    rig.guard.install_bridge()
    rig.agent.session_id = "owner"


def test_internal_same_text_does_not_take_an_unused_owner_ingress_row(rig, adapter):
    connect(rig, adapter)
    adapter.sessions.observe(session_id="owner", platform="telegram", sender_id="1001", turn_id="consult")
    adapter.sessions.get("owner").branch_consult_turn = "consult"
    rig.guard.pre_gateway_dispatch(event=SimpleNamespace(text="Correct", message_id="queued-owner",
        source=SimpleNamespace(platform="telegram", user_id="1001", chat_id="1001", chat_type="dm")))
    args = {"session_id": "owner", "turn_id": "completion", "platform": "telegram", "sender_id": "1001",
            "user_message": "Correct", "conversation_history": [
                {"role": "user", "content": "Correct", "display_kind": "internal_notification"}]}
    adapter.capture.pre_llm_call(**args)
    adapter.answers.pre_llm_call(**args)
    rig.guard.pre_llm_call(**args)
    adapter.capture.post_llm_call(**args, assistant_response="Noted")
    assert adapter.outbox.rows() == []
    assert adapter.calls == []
    assert adapter.sessions.get("owner").branch_consult_turn == "consult"
    assert rig.guard.owner_started_turn("owner", "completion") is False
    assert next(iter(rig.guard.inbound.values())).consumed is None


@pytest.mark.parametrize("platform", ["cli", "api", "voice"])
def test_owner_capture_stays_outside_messaging_front_scope(rig, adapter, platform):
    connect(rig, adapter)
    rig.runtime.authorized_owner = lambda sid: sid == "owner"
    args = {"session_id": "owner", "turn_id": "other-lane-turn", "platform": platform, "sender_id": "",
            "user_message": "Owner's terminal message", "conversation_history": [
                {"role": "user", "content": "Owner's terminal message"}]}
    adapter.capture.pre_llm_call(**args)
    rig.guard.pre_llm_call(**args)
    adapter.capture.post_llm_call(**args, assistant_response="Noted")
    row, = adapter.outbox.rows()
    assert row["payload"]["user_message"] == "Owner's terminal message"
    assert rig.guard.request_caps("owner", "other-lane-turn") is None


def test_unresolved_messaging_sender_cannot_bypass_ingress_provenance(rig, adapter):
    connect(rig, adapter)
    adapter.sessions.client.resolve_contact = lambda *args, **kwargs: None
    args = {"session_id": "unresolved", "turn_id": "completion", "platform": "telegram", "sender_id": "9999",
            "user_message": "yes", "conversation_history": [
                {"role": "user", "content": "yes", "display_kind": "internal_notification"}]}
    adapter.capture.pre_llm_call(**args)
    adapter.capture.post_llm_call(**args, assistant_response="Noted")
    assert adapter.sessions.get("unresolved").owner is None
    assert adapter.outbox.rows() == []


def test_confirmed_guest_preserves_existing_conversation_capture(rig, adapter):
    connect(rig, adapter)
    adapter.sessions.client.resolve_contact = lambda *args, **kwargs: {"contact_id": "p-03"}
    args = {"session_id": "guest", "turn_id": "guest-turn", "platform": "telegram", "sender_id": "9999",
            "user_message": "Guest's own words", "conversation_history": [
                {"role": "user", "content": "Guest's own words"}]}
    adapter.capture.pre_llm_call(**args)
    adapter.capture.post_llm_call(**args, assistant_response="Noted")
    row, = adapter.outbox.rows()
    assert adapter.sessions.get("guest").owner is False
    assert row["payload"]["user_message"] == "Guest's own words"


def test_failed_structural_observer_cannot_enable_lazy_same_text_matching(rig, adapter):
    connect(rig, adapter)
    rig.guard.pre_gateway_dispatch(event=SimpleNamespace(text="yes", message_id="queued-owner",
        source=SimpleNamespace(platform="telegram", user_id="1001", chat_id="1001", chat_type="dm")))

    def failed_observer(**kwargs):
        raise RuntimeError("structural provenance unavailable")

    rig.guard.bridge.observe_turn = failed_observer
    args = {"session_id": "owner", "turn_id": "completion", "platform": "telegram", "sender_id": "1001",
            "user_message": "yes", "conversation_history": [
                {"role": "user", "content": "yes", "display_kind": "internal_notification"}]}
    # Hermes catches a failed hook and continues with later hooks.
    with pytest.raises(RuntimeError, match="structural provenance unavailable"):
        adapter.capture.pre_llm_call(**args)
    adapter.answers.pre_llm_call(**args)
    assert adapter.calls == []
    assert next(iter(rig.guard.inbound.values())).consumed is None
    rig.guard.pre_llm_call(**args)
    adapter.capture.post_llm_call(**args, assistant_response="Noted")
    assert adapter.outbox.rows() == []


def test_memory_observer_arms_front_when_front_hook_ran_before_owner_resolution(rig, adapter):
    connect(rig, adapter)
    rig.client.rows = [{'hand_id': 'h-1'}]
    args = {'session_id': 'owner', 'turn_id': 'first-owner', 'platform': 'telegram', 'sender_id': '1001',
            'user_message': 'Hello', 'conversation_history': [{'role': 'user', 'content': 'Hello'}]}
    rig.guard.pre_gateway_dispatch(event=SimpleNamespace(text='Hello', message_id='first-message',
        source=SimpleNamespace(platform='telegram', user_id='1001', chat_id='1001', chat_type='dm')))
    assert rig.guard.pre_llm_call(**args) is None
    observed = adapter.capture.pre_llm_call(**args)
    state = rig.guard._state('owner', 'first-owner')
    assert state is not None and state.armed
    assert observed == {'context': 'Ready results'}
    assert rig.guard.pre_llm_call(**args) is None, 'context must be emitted only once across plugin hooks'
