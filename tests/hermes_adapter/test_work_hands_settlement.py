"""The hands release and work capture keep their guarantees on the same hooks."""

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
import sys
from types import SimpleNamespace

import pytest

from colonymind_hermes import busy_ingress, capture as capture_module
from colonymind_hermes.capture import Capture, SessionMap, TurnOutbox, deliver
from colonymind_hermes.client import Settings, SidecarUnavailable


class Client:
    def __init__(self, release_status=200):
        self.release_status, self.calls = release_status, []

    def post(self, path, **kwargs):
        self.calls.append((path, deepcopy(kwargs["json"])))
        if path.endswith('native-settled') and (self.release_status == 'exhausted'
                or (self.release_status == 'timeout_once' and len(self.calls) == 1)):
            raise SidecarUnavailable('ReadTimeout', delivered=True)
        status = self.release_status if path.endswith("native-settled") else 200
        if status == 'timeout_once':
            status = 200
        return SimpleNamespace(status_code=status, is_success=status == 200,
                               json=lambda: {"accepted": True, "work_recorded": True})


def settings_at(path):
    return Settings("http://unreachable.invalid", path / "api.key", "test-key", path, path,
                    path / "outbox.sqlite3")




def test_busy_owner_followup_keeps_its_own_work_origin(tmp_path, monkeypatch):
    settings, client = settings_at(tmp_path), Client()
    capture = Capture(SessionMap(settings, client), TurnOutbox(settings.outbox_path), settings)
    monkeypatch.delitem(sys.modules, "colonymind_hands_front", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(capture_module, "session_env", lambda: ("telegram", "owner", "chat", "dm"))
    message_ids = {"first": "message-1", "followup": "message-2", "internal": ""}
    monkeypatch.setattr(capture_module, "turn_platform_id", lambda session, turn: message_ids[turn])
    monkeypatch.setattr(busy_ingress, "_closed", False)
    monkeypatch.setattr(busy_ingress, "_installed", [])

    adapter = SimpleNamespace(_busy_session_handler=None)
    adapter.set_busy_session_handler = lambda callback: setattr(adapter, "_busy_session_handler", callback)

    def hear(event, source):
        capture.sessions.hear(event)
        return event

    runner = SimpleNamespace(adapters={"telegram": adapter}, _hm_pre_gateway_dispatch_hook=hear)
    assert busy_ingress.install(runner) == 1
    try:
        for message_id in ("message-1", "message-2", "message-3"):
            event = SimpleNamespace(source=SimpleNamespace(platform="telegram", user_id="owner",
                chat_id="chat", chat_type="dm"), text="Save the same change", message_id=message_id,
                timestamp=datetime.now(timezone.utc), internal=False)
            asyncio.run(adapter._busy_session_handler(event, "route"))
        for turn in ("first", "followup", "internal"):
            values = dict(session_id="session", task_id="task", turn_id=turn, platform="telegram",
                          sender_id="owner", user_message="Save the same change")
            capture.pre_llm_call(**values)
            capture.post_llm_call(**values, assistant_response="Done")
        payloads = {row["turn_id"]: row["payload"] for row in capture.outbox.rows()}
        for turn in ("first", "followup"):
            assert json.loads(payloads[turn]["work"]["origin_ref"])["message_id"] == message_ids[turn]
            assert payloads[turn]["answer"]["message_id"] == message_ids[turn]
        assert "work" not in payloads["internal"]
        assert "answer" not in payloads["internal"]
    finally:
        busy_ingress.uninstall()
