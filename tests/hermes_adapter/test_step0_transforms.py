"""Qualify the narrow output transforms against native persistence and replay."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_source import HERMES_ROOT, assert_source_module, assert_supported_source
from test_hands_front import owner_turn, result_rows, review_turn, rig  # noqa: F401


@pytest.fixture
def native(monkeypatch, tmp_path):
    if not os.environ.get("HERMES_SOURCE_ROOT"):
        pytest.skip("set HERMES_SOURCE_ROOT to a pinned Step 0 source")
    selected = assert_supported_source()
    monkeypatch.syspath_prepend(str(HERMES_ROOT))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    from agent import turn_finalizer
    from hermes_state import SessionDB
    assert_source_module(turn_finalizer)
    # The pinned host's own fixture implements the finalizer's agent protocol.
    path = HERMES_ROOT / "tests/agent/test_turn_finalizer_final_response_persistence.py"
    spec = importlib.util.spec_from_file_location("step0_native_finalizer_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = module.FakeAgent()
    agent.session_id, agent.platform = "owner-session", "sms"
    database = SessionDB(db_path=tmp_path / "state.db")
    database.create_session(agent.session_id, source="sms")

    def persist(messages, history):
        database.replace_messages(agent.session_id, messages)
        agent.persisted_messages = database.get_messages_as_conversation(agent.session_id)

    agent._persist_session = persist
    yield SimpleNamespace(selected=selected, agent=agent, database=database, finalizer=turn_finalizer)
    database.close()


def _answers():
    from colonymind_hermes.answers import Answers
    written = []
    sessions = SimpleNamespace(
        get=lambda session: SimpleNamespace(parent_session_id="", sender_id="owner", platform="sms"),
        chat=lambda session: ("owner", "dm"), is_owner=lambda session: True)
    answers = Answers(None, sessions, None, SimpleNamespace(record=lambda *args: written.append(args)))
    return answers.transform_llm_output, written


def _hooks(monkeypatch, callbacks):
    seen = []

    def invoke(name, **kwargs):
        if name != "transform_llm_output":
            return []
        seen.append(dict(kwargs))
        return [callback(**kwargs) for callback in callbacks]

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke)
    return seen


def _finish(native, text, *, turn="turn-2", history=None, runtime=False):
    history = deepcopy(history or [])
    messages = [*deepcopy(history), {"role": "user", "content": "Current request."}]
    if runtime:
        messages.append({"role": "assistant", "content": text, "display_kind": "runtime_handoff",
                         "display_metadata": {"response_origin": "runtime", "type": "tool_handoff",
                                              "tool_call_id": "accepted-tool"}})
    result = native.finalizer.finalize_turn(
        native.agent, final_response=text, api_call_count=1, interrupted=False, failed=False,
        messages=messages, conversation_history=history, effective_task_id="task", turn_id=turn,
        user_message="Current request.", original_user_message="Current request.", _should_review_memory=False,
        _turn_exit_reason="tool_handoff" if runtime else "fallback_prior_turn_content")
    assert not result.get("cleanup_errors"), result.get("cleanup_errors")
    return result


@pytest.mark.parametrize("kind", ["invented_code", "silent_review"])
def test_transform_persistence_and_replay_selected_source(native, rig, monkeypatch, kind):
    before = [{"role": "user", "content": "Earlier request."},
              {"role": "assistant", "content": "Earlier text. Reply 'yes PXW' to approve it."}]
    if kind == "invented_code":
        callback, written = _answers()
        original = "The payment waits for you.\nReply 'yes PXW' to approve it.\nAnything else?"
        expected = "The payment waits for you.\nAnything else?"
    else:
        # Reuse the receipt fixture: a completed review makes this owned result a duplicate.
        rig.client.rows = result_rows(1)
        review_turn(rig, "turn-1")
        rig.guard.post_llm_call(session_id="owner-session", turn_id="turn-1", assistant_response="Reviewed result.")
        rig.guard.on_session_end(session_id="owner-session", turn_id="turn-1", completed=True, failed=False,
                                 interrupted=False)
        review_turn(rig, "turn-2")
        assert rig.guard._state("owner-session", "turn-2").duplicate
        callback, written = rig.guard.finalize_response, []
        original, expected = "Repeated result prose.", "[SILENT]"
    seen = _hooks(monkeypatch, [callback])
    result = _finish(native, original, history=before)
    assert result["final_response"] == expected and result["response_transformed"] is True
    assert len(seen) == 1
    replay = native.database.get_messages_as_conversation("owner-session")
    durable = expected if native.selected in {"S", "S217"} else original
    assert replay[-1]["content"] == durable
    assert [message["content"] for message in replay[:2]] == [message["content"] for message in before]
    assert len(replay) == 4
    if kind == "invented_code":
        assert len(written) == 1
    # Replay the persisted turn into another native finalization. Its prefix stays byte-for-byte.
    owner_turn(rig, turn="turn-3", ident="owner-next")
    following = _finish(native, "Next answer.", turn="turn-3", history=replay)
    assert following["final_response"] == "Next answer."
    replayed = native.database.get_messages_as_conversation("owner-session")
    assert [message["content"] for message in replayed[:4]] == [message["content"] for message in replay]
    assert len(replayed) == 6 and len(seen) == 2
    if kind == "invented_code":
        assert len(written) == 1


def test_transform_hook_has_no_runtime_origin_on_s(native, rig, monkeypatch):
    # This is a runtime receipt in a finalizer with origin metadata. A host without a discriminator
    # at its transform seam, even when the transcript carries runtime display metadata.
    owner_turn(rig, turn="turn-2")
    answers, _ = _answers()
    seen = _hooks(monkeypatch, [answers, rig.guard.finalize_response])
    receipt = "Accepted background task background-accepted."
    result = _finish(native, receipt, runtime=True)
    assert result["final_response"] == receipt
    assert native.database.get_messages_as_conversation("owner-session")[-1]["content"] == receipt
    if native.selected not in {"S", "S217"}:
        assert not seen and result["response_origin"] == "runtime"
    else:
        assert len(seen) == 1
        assert set(seen[0]) == {"response_text", "session_id", "model", "platform", "turn_id"}
        assert "response_origin" not in result


def test_unrelated_runtime_text_keeps_the_narrow_transforms(native, rig, monkeypatch):
    # A receipt-looking message without an exact owned native receipt is not a review.
    unrelated = "Runtime status: Native completion belongs to another task."
    rig.guard.pre_llm_call(
        session_id="owner-session", turn_id="turn-2", user_message=unrelated, platform="sms",
        sender_id="5550100", chat_id="5550100", session_key="route",
        conversation_history=[{"role": "user", "content": unrelated, "display_kind": "internal_notification"}])
    assert rig.guard._state("owner-session", "turn-2").mode == "system"
    answers, written = _answers()
    seen = _hooks(monkeypatch, [answers, rig.guard.finalize_response])
    result = _finish(native, unrelated)
    assert result["final_response"] == unrelated and result["response_transformed"] is False
    assert native.database.get_messages_as_conversation("owner-session")[-1]["content"] == unrelated
    assert len(seen) == 1 and written == []
