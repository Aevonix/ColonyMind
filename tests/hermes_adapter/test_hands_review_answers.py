"""Review turns bypass owner-answer lookups without affecting the next owner turn."""

import sys
from types import SimpleNamespace

import pytest

from colonymind_hermes.answers import Answers, ChatLedger, TURN


@pytest.fixture
def answer_reader(monkeypatch):
    calls = []
    ledger = SimpleNamespace(record=lambda *args: calls.append(("ledger", args)))
    info = SimpleNamespace(session_id="owner-session", turn_id="review-turn", owner=True,
                           parent_session_id="", sender_id="owner", platform="telegram",
                           chat_id="owner-chat", chat_type="dm")
    sessions = SimpleNamespace(get=lambda sid: info if sid == "owner-session" else None)
    answers = Answers(None, sessions, None, ledger)

    def owner_chat(*args, **kwargs):
        calls.append(("owner_chat", args))
        return info, "owner-chat"

    def to_read(*args, **kwargs):
        calls.append(("to_read", args))
        return {"text": "yes"}

    def post(*args, **kwargs):
        calls.append(("post", args))
        return {"context": "The owner's answer was read."}

    monkeypatch.setattr(answers, "owner_chat", owner_chat)
    monkeypatch.setattr(answers, "_to_read", to_read)
    monkeypatch.setattr(answers, "_post", post)
    bridge = SimpleNamespace(is_review_turn=lambda session_id, turn_id:
                             (session_id, turn_id) == ("owner-session", "review-turn"))
    monkeypatch.setitem(sys.modules, "colonymind_hands_front", bridge)
    return SimpleNamespace(answers=answers, calls=calls, bridge=bridge, info=info)


def test_review_skips_owner_resolution_and_answer_route_but_preserves_turn_ledger(answer_reader):
    answer_reader.answers._reads["owner-session"] = {"context": "previous answer"}
    assert answer_reader.answers.pre_llm_call(session_id="owner-session", turn_id="review-turn") is None
    assert answer_reader.calls == [("ledger", ("telegram:owner-chat", TURN))]
    assert "owner-session" not in answer_reader.answers._reads


@pytest.mark.parametrize("field, value", [
    ("owner", None), ("owner", False), ("turn_id", "previous-turn"), ("chat_id", ""),
    ("chat_type", "group"), ("parent_session_id", "parent"), ("sender_id", ""), ("platform", "cron")])
def test_review_never_invents_an_owner_chat_scope(answer_reader, field, value):
    setattr(answer_reader.info, field, value)
    assert answer_reader.answers.pre_llm_call(session_id="owner-session", turn_id="review-turn") is None
    assert answer_reader.calls == []


def test_review_preserves_evidence_of_text_that_may_have_streamed(answer_reader, tmp_path, monkeypatch):
    ledger = ChatLedger(tmp_path / "chats.sqlite3")
    answer_reader.answers.ledger = ledger
    monkeypatch.setattr("colonymind_hermes.answers.time.time", lambda: 1000.0)
    answer_reader.answers.pre_llm_call(session_id="owner-session", turn_id="review-turn")
    assert ledger.under_way("telegram:owner-chat", 1001.0)
    assert any(row["kind"] == TURN for row in ledger.seen("telegram:owner-chat", until=1001.0))


def test_review_ledger_failure_never_falls_through_to_network(answer_reader):
    def failed(*args):
        raise OSError("ledger unavailable")
    answer_reader.answers.ledger.record = failed
    assert answer_reader.answers.pre_llm_call(session_id="owner-session", turn_id="review-turn") is None
    assert answer_reader.calls == []


def test_next_owner_turn_keeps_its_answer_processing(answer_reader):
    answer_reader.answers.pre_llm_call(session_id="owner-session", turn_id="review-turn")
    answer_reader.calls.clear()
    assert answer_reader.answers.pre_llm_call(session_id="owner-session", turn_id="owner-turn") == {
        "context": "The owner's answer was read."}
    assert [name for name, _ in answer_reader.calls] == ["owner_chat", "to_read", "ledger", "post"]
    assert "owner-session" in answer_reader.answers._reads


@pytest.mark.parametrize("session_id, turn_id", [("guest-session", "review-turn"), ("owner-session", "")])
def test_unknown_or_other_session_does_not_inherit_review(answer_reader, session_id, turn_id):
    assert answer_reader.answers.pre_llm_call(session_id=session_id, turn_id=turn_id) is not None
    assert answer_reader.calls[0][0] == "owner_chat"


@pytest.mark.parametrize("bridge_kind", ["missing", "old", "broken", "invalid"])
def test_missing_review_verdict_keeps_answer_processing(answer_reader, monkeypatch, bridge_kind):
    if bridge_kind == "missing":
        monkeypatch.delitem(sys.modules, "colonymind_hands_front")
    elif bridge_kind == "old":
        del answer_reader.bridge.is_review_turn
    elif bridge_kind == "broken":
        def broken(*args):
            raise RuntimeError("bridge unavailable")
        answer_reader.bridge.is_review_turn = broken
    else:
        answer_reader.bridge.is_review_turn = lambda *args: {"mode": "review_result"}
    assert answer_reader.answers.pre_llm_call(session_id="owner-session", turn_id="review-turn") is not None
    assert answer_reader.calls[0][0] == "owner_chat"
