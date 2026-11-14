"""The capture of an owner turn hands the mind the message's own evidence for the ingest read (docs/MIND.md, Asks).

The host plugin reads an owner message as a possible answer before its model runs (``POST /v1/mind/answers``).
A message it could not read then (its 1.5 s ran out, or a body without the plugin) is read when the turn is
captured, with explicit references only: what the message replies or reacts to, never the follow-up, because only
the plugin's ledger knows what the chat showed. The capture names the message in the turn's metadata (``answer``);
the turn route passes it to ``Mind.owner_turn`` as ``evidence``, on the platform the sidecar itself resolved, and
drops anything the ingest read must not take (a ledger, words, an owner claim). The turn is dated when its message
arrived (never later than its capture), so a message read at ingest is judged as of then.
"""

from __future__ import annotations

from datetime import datetime, timezone
import time

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from test_mind_loop import OWNER, Fixture, pinned_clock  # noqa: F401  (the pinned wall clock)


def turn(contact, text, ident, *, metadata=None, platform="whatsapp", sender=None):
    body = {"identity": {"host_id": "hermes"},
            "context": {"session_id": "s-1", "contact_id": contact, "turn_id": ident,
                        "channel_id": f"{platform}:{sender or contact}",
                        "metadata": {"occurred_at": "2026-01-06T12:00:00Z", **(metadata or {})}},
            "user_message": {"role": "user", "content": text},
            "assistant_message": {"role": "assistant", "content": "ok"}}
    if sender:
        body["sender"] = {"platform": platform, "user_id": sender}
    return body


async def test_the_turn_route_hands_the_messages_evidence_to_the_ingest_read(monkeypatch, tmp_path):
    from colonymind.api.middleware import ApiKeyMiddleware
    from colonymind.api.routers import host
    from colonymind.api.routers import mind as mind_router
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    monkeypatch.delenv("COLONYMIND_API_KEY", raising=False)
    for name in ("_contacts_store", "_telemetry", "_reranker", "_context_recall_selector", "_comms_log"):
        monkeypatch.setattr(host, name, None)
    calls = []

    class Mind:
        async def owner_turn(self, text, **kwargs):
            calls.append((text, kwargs))
            return {}

    key = "hook-key-" + "x" * 32
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware, api_key=key)
    app.include_router(host.router)
    mind_router.set_mind(Mind())
    answer = {"chat_id": "owner-chat", "chat_type": "dm", "message_id": "in-1", "reply_to_message_id": "m-1",
              "reaction": "", "reaction_removed": False, "forwarded": False, "voice": True, "own_account": False,
              "arrived_at": "2026-01-06T11:59:00Z",
              # Never the ingest read's: what the chat showed, words, a platform or an owner of the plugin's saying.
              "ledger": [{"at": "2026-01-06T11:58:00Z", "outbox_id": "o-1"}], "text": "yes",
              "platform": "sms", "contact_id": "someone-else"}
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://x") as client:
            auth = {"Authorization": f"Bearer {key}"}
            late = {**answer, "message_id": "in-4", "arrived_at": "2026-01-06T12:30:00Z"}
            for body in (turn(OWNER, "yes", "t-1", metadata={"answer": answer}),
                         turn(OWNER, "thanks", "t-2"),
                         turn(OWNER, "ok", "t-3", metadata={"answer": "not a mapping"}),
                         turn(OWNER, "no", "t-4", metadata={"answer": late})):
                response = await client.post("/v1/host/turns/sync", json=body, headers=auth)
                assert response.status_code == 200, response.text
    finally:
        mind_router.set_mind(None)
    (first, with_evidence), (_, plain), (_, malformed), (_, skewed) = calls
    assert first == "yes"
    evidence = with_evidence["evidence"]
    assert evidence == {"platform": "whatsapp", "chat_id": "owner-chat", "chat_type": "dm", "message_id": "in-1",
                        "reply_to_message_id": "m-1", "reaction": "", "reaction_removed": False, "forwarded": False,
                        "voice": True, "own_account": False, "arrived_at": "2026-01-06T11:59:00+00:00"}
    # The owner's turn is dated when the message arrived, not when its turn was captured.
    assert with_evidence["occurred_at"] == datetime(2026, 1, 6, 11, 59, tzinfo=timezone.utc)
    assert plain.get("evidence") is None and malformed.get("evidence") is None
    assert plain["occurred_at"] == datetime(2026, 1, 6, 12, 0, tzinfo=timezone.utc)
    # An arrival after the capture (a skewed host clock) is never taken: the capture's time stands.
    assert skewed["occurred_at"] == datetime(2026, 1, 6, 12, 0, tzinfo=timezone.utc)
    assert skewed["evidence"]["arrived_at"] == "2026-01-06T12:00:00+00:00"


PLATFORM, CHAT = "whatsapp", "owner-chat"     # a platform the instance key is granted the owner on


class Ingest:
    """A real mind behind the host route: asks, their own messages delivered to the owner's chat, and owner turns
    captured with the message's evidence (``metadata.answer``), read at ingest. The wall clock the route reads
    follows the mind's (``fx.shift`` moves both), so a capture is dated where the test puts it."""

    def __init__(self, monkeypatch, tmp_path):
        from colonymind.api.routers import host
        monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path / "host"))
        monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
        (tmp_path / "host").mkdir()
        (tmp_path / "mind").mkdir()
        monkeypatch.delenv("COLONYMIND_API_KEY", raising=False)
        for name in ("_contacts_store", "_telemetry", "_reranker", "_context_recall_selector", "_comms_log"):
            monkeypatch.setattr(host, name, None)
        self.fx = Fixture(tmp_path / "mind")
        monkeypatch.setattr(time, "time", lambda: self.fx.now.timestamp())

    def deliver(self, row, message_id):
        outbox = self.fx.mind.outbox
        notice = outbox.notice(type="ask_show", title="ask", text="the ask in words", dedup_key=f"test:{message_id}",
                               extra={"asks": [row.id], "list": [row.id]})
        outbox.sending(notice.id, target=f"{PLATFORM}:{CHAT}")
        return outbox.sent(notice.id, result="sent", hermes_ref=message_id)

    async def turns(self, *captured):
        """Each ``(turn id, text, answer evidence, occurred_at)`` posted as a captured owner turn."""
        from colonymind.api.middleware import ApiKeyMiddleware
        from colonymind.api.routers import host
        from colonymind.api.routers import mind as mind_router
        key = "hook-key-" + "y" * 32
        app = FastAPI()
        app.add_middleware(ApiKeyMiddleware, api_key=key)
        app.include_router(host.router)
        mind_router.set_mind(self.fx.mind)
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://x") as client:
                for ident, text, answer, occurred in captured:
                    body = turn(OWNER, text, ident, metadata={"answer": {"chat_id": CHAT, "chat_type": "dm", **answer}},
                                platform=PLATFORM)
                    body["context"]["metadata"]["occurred_at"] = occurred.isoformat()
                    response = await client.post("/v1/host/turns/sync", json=body,
                                                 headers={"Authorization": f"Bearer {key}"})
                    assert response.status_code == 200, response.text
        finally:
            mind_router.set_mind(None)


async def test_the_ingest_read_settles_an_ask_the_message_quotes(monkeypatch, tmp_path):
    """End to end on a real mind: an owner turn captured with a quoted reply to the ask's own message approves it at
    ingest; the same words with no reference approve nothing."""
    from test_ask_answers_binding import asked
    ingest = Ingest(monkeypatch, tmp_path)
    fx = ingest.fx
    quoted, plain = asked(fx, "K7F"), asked(fx, "Q2Z")
    ingest.deliver(quoted, "m-1")
    ingest.deliver(plain, "m-2")
    fx.shift(minutes=1)
    try:
        await ingest.turns(("t-1", "yes", {"message_id": "in-1", "reply_to_message_id": "m-1"}, fx.now),
                           ("t-2", "yes", {"message_id": "in-2"}, fx.now))
        assert fx.store.get(quoted.id).status == "approved"
        assert fx.store.get(plain.id).status == "asked"
    finally:
        fx.store.close()


async def test_the_ingest_read_takes_the_whole_message_never_a_part(monkeypatch, tmp_path):
    """A message too long for the plugin's read is read at ingest, whole: a refusal past the plugin route's 8,000
    characters ("yes", a long gap, "but do not send it") still voids the yes."""
    from test_ask_answers_binding import asked
    ingest = Ingest(monkeypatch, tmp_path)
    fx = ingest.fx
    row = asked(fx, "K7F")
    ingest.deliver(row, "m-1")
    fx.shift(minutes=1)
    try:
        await ingest.turns(("t-1", "yes" + " " * 8000 + "but do not send it",
                            {"message_id": "in-1", "reply_to_message_id": "m-1"}, fx.now))
        assert fx.store.get(row.id).status == "asked"
    finally:
        fx.store.close()


async def test_the_ingest_read_judges_a_message_as_of_its_arrival(monkeypatch, tmp_path):
    """An ask too long to show whole counts a yes only once its full text had reached the owner. The owner quotes the
    ask's short message; its full text arrives after they wrote, while the model runs; the turn is captured after
    both. Read at ingest, the message is judged as of its arrival (``metadata.answer.arrived_at``), never the
    capture's time, so it approves nothing. The same yes written after the full text arrived approves it."""
    from test_ask_answers_binding import asked
    ingest = Ingest(monkeypatch, tmp_path)
    fx = ingest.fx
    row = asked(fx, context={"full_view": {"seen": False, "notice": None, "attempts": 0}})
    ingest.deliver(row, "m-1")
    fx.shift(minutes=1)
    written = fx.now
    fx.shift(seconds=30)
    fx.mind.outbox.send_full_view(row, "Re-run the archive export.", "What a yes allows, in full. " * 40)
    full = fx.store.get(fx.store.get(row.id).context["full_view"]["notice"])
    fx.mind.outbox.sending(full.id, target=f"{PLATFORM}:{CHAT}")
    fx.mind.outbox.sent(full.id, result="sent", hermes_ref="m-2")
    fx.shift(seconds=30)
    try:
        await ingest.turns(("t-1", "yes", {"message_id": "in-1", "reply_to_message_id": "m-1",
                                           "arrived_at": written.isoformat()}, fx.now))
        assert fx.store.get(row.id).status == "asked"
        fx.shift(seconds=30)
        await ingest.turns(("t-2", "yes", {"message_id": "in-2", "reply_to_message_id": "m-1",
                                           "arrived_at": fx.now.isoformat()}, fx.now))
        assert fx.store.get(row.id).status == "approved"
    finally:
        fx.store.close()
