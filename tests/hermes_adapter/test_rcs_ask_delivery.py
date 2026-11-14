"""RCS asks retain the controller's thread and report actual native delivery."""
import asyncio
import sys
from types import SimpleNamespace as NS

import httpx
import pytest

from colonymind_hermes import body


@pytest.fixture
def native(monkeypatch):
    calls = []

    async def send(**values):
        calls.append(values)
        return NS(success=True, message_id="native-ask-receipt", error=None)

    async def dispatch(runner, make_coro, label):
        assert runner is runner_instance
        return await make_coro()

    runner_instance = NS()
    adapter = NS(send_owner_notice=send)
    tools = NS(_live_adapter=lambda platform: (runner_instance, adapter),
               _dispatch_on_gateway_loop=dispatch,
               send_message_tool=lambda *a, **k: pytest.fail("unanchored send must not run"))
    monkeypatch.setitem(sys.modules, "gateway.config", NS(Platform=lambda value: value))
    monkeypatch.setitem(sys.modules, "model_tools", NS(_run_async=asyncio.run))
    monkeypatch.setitem(sys.modules, "tools.send_message_tool", tools)
    return calls, adapter, tools


def message():
    return {"id": "outbox-ask", "type": "ask_notice", "recipient_is_owner": True,
            "target": "rcs:original-chat", "text": "Allow this exact call once?",
            "delivery_ref": {"platform": "rcs", "chat": "original-chat", "message_id": "signed-request"}}


def test_body_delivers_once_and_records_native_receipt_in_original_chat(native, tmp_path):
    calls, _, _ = native
    posts, observed = [], []
    item = message()

    def post(path, **kwargs):
        posts.append((path, kwargs["json"]))
        return httpx.Response(200, json={"ok": True})

    client = NS(get=lambda *a, **k: httpx.Response(200, json={"messages": [item]}), post=post)
    sends = NS(hold=lambda *a: False, sent=lambda *a: observed.append(a))
    instance = body.Body(client, None, None, NS(), ledger=body.BodyLedger(tmp_path / "body.db"), sends=sends)
    assert instance.send_outbox() == 1
    assert instance.send_outbox() == 0
    assert calls == [{"chat_id": "original-chat", "content": item["text"],
                      "delivery_id": "outbox-ask", "reply_to": "signed-request"}]
    assert observed == [(item, "rcs:original-chat", "native-ask-receipt")]
    assert posts[-1][1]["result"] == "sent"
    assert posts[-1][1]["hermes_ref"] == "native-ask-receipt"


@pytest.mark.parametrize("change", [
    {"recipient_is_owner": False}, {"delivery_ref": {}}, {"id": ""},
    {"delivery_ref": {"platform": "sms", "chat": "original-chat", "message_id": "signed-request"}},
])
def test_invalid_notice_route_cannot_send(native, change):
    calls, _, _ = native
    assert body.send_owner_notice({**message(), **change}, "rcs:original-chat", "ask")["result"] == "failed"
    assert not calls


def test_changed_target_and_missing_native_sender_cannot_fall_back(native):
    calls, adapter, _ = native
    assert body.send_owner_notice(message(), "rcs:other-chat", "ask")["result"] == "failed"
    adapter.send_owner_notice = None
    assert body.send_owner_notice(message(), "rcs:original-chat", "ask")["result"] == "failed"
    assert not calls


def test_native_exception_remains_uncertain_without_fallback(native):
    calls, adapter, _ = native

    async def lost_receipt(**kwargs):
        calls.append(kwargs)
        raise TimeoutError("receipt lost")

    adapter.send_owner_notice = lost_receipt
    assert body.send_owner_notice(message(), "rcs:original-chat", "ask")["result"] == "uncertain"
    assert len(calls) == 1


def test_queued_notice_without_device_receipt_is_uncertain_and_not_shown(native, tmp_path):
    _, adapter, _ = native

    async def pending(**kwargs):
        return NS(success=False, message_id="queued-reply", error="device delivery unconfirmed")

    adapter.send_owner_notice = pending
    posts, shown = [], []
    client = NS(get=lambda *a, **k: httpx.Response(200, json={"messages": [message()]}),
                post=lambda path, **kwargs: (posts.append((path, kwargs["json"]))
                                             or httpx.Response(200, json={"ok": True})))
    instance = body.Body(client, None, None, NS(), ledger=body.BodyLedger(tmp_path / "body.db"),
                         sends=NS(hold=lambda *a: False, sent=lambda *a: shown.append(a)))
    assert instance.send_outbox() == 0
    assert instance.send_outbox() == 0
    assert not shown
    assert posts[-1][1]["result"] == "uncertain"
