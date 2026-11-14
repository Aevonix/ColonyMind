"""A classified completion review performs no recall or contact warm-up."""

import sys
from types import SimpleNamespace

import pytest

import colonymind_memory
from colonymind_memory.provider import ColonyMindMemoryProvider


@pytest.fixture
def recall(monkeypatch, tmp_path):
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path / "instance"))
    monkeypatch.setenv("COLONYMIND_PERSONA_NAME", "Nova")
    monkeypatch.setenv("COLONYMIND_AGENT_TIMEZONE", "Asia/Tokyo")
    provider = ColonyMindMemoryProvider(config={
        "url": "http://sidecar.test", "api_key": "test-key", "contact_id": "owner-contact"})
    provider._session_id = "owner-session"
    provider._turn_ids["owner-session"] = "review-turn"
    calls = []

    def contact(*args, **kwargs):
        calls.append("contact")
        return "owner-contact"

    def assemble(*args, **kwargs):
        calls.append("assemble")
        return [], {}

    def temporal(*args, **kwargs):
        calls.append("temporal")
        return "ordinary recalled context"

    monkeypatch.setattr(provider, "_prefetch_contact", contact)
    monkeypatch.setattr(provider, "_assemble", assemble)
    monkeypatch.setattr(provider, "_with_fresh_temporal_sync", temporal)
    monkeypatch.setattr(provider, "_offer", lambda *a, **kw: calls.append("offer"))
    bridge_calls = []

    def is_review_turn(session_id, turn_id):
        bridge_calls.append((session_id, turn_id))
        return (session_id, turn_id) == ("owner-session", "review-turn")

    bridge = SimpleNamespace(is_review_turn=is_review_turn)
    monkeypatch.setitem(sys.modules, "colonymind_hands_front", bridge)
    return SimpleNamespace(provider=provider, calls=calls, bridge=bridge, bridge_calls=bridge_calls)


@pytest.mark.parametrize("session_id", ["owner-session", ""])
def test_review_prefetch_skips_all_memory_work_and_resets_recall_status(recall, session_id):
    recall.provider._last_recall = 5
    assert recall.provider.prefetch("completed result", session_id=session_id) == ""
    assert recall.calls == []
    assert recall.bridge_calls == [("owner-session", "review-turn")]
    assert recall.provider._last_recall is None


@pytest.mark.parametrize("turn_id", ["owner-turn", ""])
def test_owner_or_unbound_turn_keeps_recall_even_with_completion_words(recall, turn_id):
    recall.provider._turn_ids["owner-session"] = turn_id
    result = recall.provider.prefetch("[IMPORTANT: background delegation completed]", session_id="owner-session")
    assert result == "ordinary recalled context"
    assert recall.calls == ["contact", "assemble", "temporal", "offer"]
    assert recall.bridge_calls == ([("owner-session", turn_id)] if turn_id else [])


def test_review_status_never_leaks_to_another_session(recall):
    recall.provider._turn_ids["guest-session"] = "review-turn"
    assert recall.provider.prefetch("same text", session_id="guest-session") == "ordinary recalled context"
    assert recall.bridge_calls == [("guest-session", "review-turn")]


@pytest.mark.parametrize("bridge_kind", ["missing", "old", "broken", "invalid"])
def test_unavailable_review_bridge_preserves_normal_recall(recall, monkeypatch, bridge_kind):
    if bridge_kind == "missing":
        monkeypatch.delitem(sys.modules, "colonymind_hands_front")
    elif bridge_kind == "old":
        del recall.bridge.is_review_turn
    elif bridge_kind == "broken":
        def broken(*args):
            raise RuntimeError("bridge unavailable")
        recall.bridge.is_review_turn = broken
    else:
        recall.bridge.is_review_turn = lambda *args: {"mode": "review_result"}
    assert recall.provider.prefetch("owner words", session_id="owner-session") == "ordinary recalled context"
    assert recall.calls == ["contact", "assemble", "temporal", "offer"]


def test_review_contact_hook_keeps_turn_identity_without_resolving_sender(recall, monkeypatch):
    resolved = []
    monkeypatch.setattr(recall.provider, "_resolve_handle", lambda *args: resolved.append(args))
    recall.provider.resolve_contact("telegram", "owner", "owner-session", turn_id="review-turn")
    assert resolved == []
    assert recall.provider._turn_ids["owner-session"] == "review-turn"
    assert recall.provider._turn_senders["owner-session"] == ("telegram", "owner")
    recall.provider.resolve_contact("telegram", "owner", "owner-session", turn_id="owner-turn")
    assert resolved == [("telegram", "owner")]
    assert recall.provider.prefetch("next owner message", session_id="owner-session") == "ordinary recalled context"


def test_review_clock_hook_never_resolves_a_contact(recall):
    assert recall.provider.turn_clock(session_id="owner-session", message="completed result") == ""
    assert recall.calls == []


def test_trivial_owner_message_still_gets_its_clock(recall, monkeypatch):
    from colonymind_memory import provider as provider_module
    monkeypatch.setattr(provider_module, "_request_within", lambda *a, **kw: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {}))
    recall.provider._turn_ids["owner-session"] = "owner-turn"
    clock = recall.provider.turn_clock(session_id="owner-session", message="hi")
    assert clock.startswith("Nova's time: ") and " JST (Asia/Tokyo).\n" in clock
    assert clock.splitlines()[1] == "the owner's location: no recent report; assume JST (Asia/Tokyo)."
    assert recall.calls == ["contact"]


def memory_hook(recall, monkeypatch):
    from agent.subagent_lifecycle import bind_subagent_parent
    hooks = []
    monkeypatch.setattr(recall.provider, "_resolve_handle", lambda *a, **kw: recall.calls.append("warm_contact"))
    colonymind_memory.register(SimpleNamespace(register_memory_provider=lambda provider: None,
                                            register_hook=lambda name, fn: hooks.append(fn)))
    agent = SimpleNamespace(session_id="owner-session",
                            _memory_manager=SimpleNamespace(providers=[recall.provider]))
    def invoke(**kwargs):
        with bind_subagent_parent(agent):
            return hooks[0](**kwargs)
    return invoke


@pytest.mark.parametrize("review", [True, False])
def test_memory_hook_defers_internal_contact_work_until_front_classifies(recall, monkeypatch, review):
    hook = memory_hook(recall, monkeypatch)
    recall.bridge.is_review_turn = lambda *args: False
    turn = {"session_id": "owner-session", "turn_id": "internal-turn", "platform": "telegram",
            "sender_id": "owner", "user_message": "background result", "conversation_history": [
                {"role": "user", "content": "background result", "display_kind": "internal_notification"}]}
    assert hook(**turn) is None
    assert recall.calls == []
    assert recall.provider._turn_ids["owner-session"] == "internal-turn"
    # The capture/front hook runs later in this registration order.
    recall.bridge.is_review_turn = lambda sid, tid: review and (sid, tid) == ("owner-session", "internal-turn")
    result = recall.provider.prefetch("background result", session_id="owner-session")
    assert result == ("" if review else "ordinary recalled context")
    assert recall.calls == ([] if review else ["contact", "assemble", "temporal", "offer"])


def test_memory_hook_owner_words_cannot_defer_contact_work(recall, monkeypatch):
    hook = memory_hook(recall, monkeypatch)
    marker = "[IMPORTANT: background delegation completed]"
    hook(session_id="owner-session", turn_id="owner-turn", platform="telegram", sender_id="owner",
         user_message=marker, conversation_history=[{"role": "user", "content": marker}])
    assert recall.calls == ["warm_contact", "contact"]


def test_trivial_internal_notification_keeps_its_local_clock(recall, monkeypatch):
    hook = memory_hook(recall, monkeypatch)
    context = hook(session_id="owner-session", turn_id="system-turn", platform="telegram", sender_id="owner",
                   user_message="hi", conversation_history=[
                       {"role": "user", "content": "hi", "display_kind": "internal_notification"}])
    assert context["context"].startswith("Nova's time:")
    assert context["context"].endswith("JST (Asia/Tokyo).")
    assert recall.calls == []
