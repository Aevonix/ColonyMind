"""Offline imported-voice dedup gate; original captures stay canonical."""
from copy import deepcopy
import socket
from types import SimpleNamespace

import pytest

from colonymind_hermes.capture import TurnOutbox, checkpoint
from colonymind_hermes.voice_writeback import is_bridged_voice_row
from colonymind_memory.provider import ColonyMindMemoryProvider


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("offline dedup gate attempted a network connection")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)


def bridged(role="user"):
    return {"role": role, "content": "[said by voice | 2026-10-14T14:03:01Z]\nNotebook",
            "display_metadata": {"bridge": "voice.writeback.v1", "origin": "voice",
                                 "voice_exchange_id": "voice-one", "payload_digest": "d" * 64,
                                 "generation_complete": True, "playback_complete": True,
                                 "source_receipt": {"source_id": "original-" + role,
                                                    "turn_id": "voice-source-" + role, "source_recorded": True}}}


def test_checkpoint_excludes_only_imported_voice_and_keeps_original_capture(tmp_path):
    outbox = TurnOutbox(tmp_path / "outbox.db")
    original = {"session_id": "voice-session", "platform": "voice", "user_message": "Notebook",
                "assistant_message": "Remembered", "turn_id": "original-voice"}
    outbox.enqueue("original-voice", original)
    before = deepcopy(outbox.rows())
    messages = [{"role": "user", "content": "regular text"}, {"role": "assistant", "content": "text answer"},
                bridged(), bridged("assistant")]
    snapshot = deepcopy(messages)
    result = checkpoint(messages, session_id="owner-main", contact_id="cid-owner", outbox=outbox,
                        platform="whatsapp", sender_id="15551234567")
    assert result["messages"] == 2
    rows = outbox.rows()
    assert next(r for r in rows if r["turn_id"] == "original-voice") == before[0]
    cp = next(r for r in rows if r["turn_id"] == result["turn_id"])
    assert cp["payload"]["checkpoint_messages"] == messages[:2]
    assert messages == snapshot  # histories/source captures are never erased or rewritten


def test_voice_only_checkpoint_writes_nothing_new(tmp_path):
    outbox = TurnOutbox(tmp_path / "outbox.db")
    assert checkpoint([bridged(), bridged("assistant")], session_id="owner-main", contact_id="cid-owner",
                      outbox=outbox) == {"state": "empty", "messages": 0}
    assert outbox.rows() == []


@pytest.mark.parametrize("change", [
    {"bridge": "another-plugin"}, {"origin": "whatsapp"}, {"voice_exchange_id": ""},
    {"payload_digest": "bad"}, {"generation_complete": False}, {"playback_complete": False},
    {"source_receipt": {"source_id": "x", "turn_id": "y", "source_recorded": False}},
    {"source_receipt": {"source_recorded": True}},
])
def test_unconfirmed_or_lookalike_marker_is_still_captured(tmp_path, change):
    row = bridged()
    row["display_metadata"].update(change)
    assert not is_bridged_voice_row(row)
    outbox = TurnOutbox(tmp_path / "outbox.db")
    assert checkpoint([row], session_id="owner-main", contact_id="cid-owner", outbox=outbox)["messages"] == 1


def test_plain_text_marker_is_never_deduplicated(tmp_path):
    row = {"role": "user", "content": bridged()["content"]}
    assert not is_bridged_voice_row(row)
    outbox = TurnOutbox(tmp_path / "outbox.db")
    assert checkpoint([row], session_id="owner-main", contact_id="cid-owner", outbox=outbox)["messages"] == 1


def test_provider_pre_compress_uses_receipt_filter_without_network(tmp_path, monkeypatch):
    from colonymind_hermes import client
    monkeypatch.setattr(client, "load_settings", lambda: SimpleNamespace(outbox_path=tmp_path / "outbox.db"))
    provider = object.__new__(ColonyMindMemoryProvider)
    provider._session_id = "owner-main"
    provider._reset_shown = lambda: None
    provider._checkpoint_contact = lambda: ("cid-owner", False)
    provider._turn_sender_context = lambda: ("whatsapp", "15551234567", "15551234567")
    provider._checkpoint_branch = lambda contact: None
    provider.on_pre_compress([bridged(), bridged("assistant"), {"role": "user", "content": "next text"}],
                             require_checkpoint=True)
    assert provider._last_checkpoint["messages"] == 1
    row, = TurnOutbox(tmp_path / "outbox.db").rows()
    assert row["payload"]["checkpoint_messages"] == [{"role": "user", "content": "next text"}]


def test_provider_session_end_cannot_recapture_bridged_exchange():
    provider = object.__new__(ColonyMindMemoryProvider)
    calls = []
    provider.sync_turn = lambda *args, **kwargs: calls.append((args, kwargs))
    provider._session_id, provider._turn_number = "owner-main", 3
    provider.on_session_end([bridged(), bridged("assistant")])
    assert calls == []
    provider.on_session_end([bridged(), bridged("assistant"), {"role": "user", "content": "new text"},
                             {"role": "assistant", "content": "new answer"}])
    assert calls == []
