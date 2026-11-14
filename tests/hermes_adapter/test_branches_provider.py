"""The provider carries bounded branch handles only for the owner's private turn."""

import httpx
import pytest

from colonymind_memory.provider import ColonyMindMemoryProvider
from conftest import build_home


@pytest.fixture
def home(tmp_path):
    return build_home(tmp_path, 1, "http://127.0.0.1:1")


@pytest.fixture
def provider(home, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(home.hermes))
    value = ColonyMindMemoryProvider({"url": "http://127.0.0.1:1", "key_file": str(home.key_file),
                                    "contact_id": "p-01"})
    value.initialize("parent", hermes_home=str(home.hermes), platform="cli")
    return value


def test_branch_section_carries_handle_and_first_line_with_bound_sender(provider, monkeypatch):
    calls = []

    def reply(timeout, method, url, **kwargs):
        calls.append((method, url, kwargs))
        return httpx.Response(200, json={"branches": [{"handle": "br-0123456789ab",
                                                      "brief": "Read the report\nExtra details"}],
                                         "tool_available": True})

    monkeypatch.setattr("colonymind_memory.provider._request_within", reply)
    section = provider._branches_section(session_id="parent", contact_id="p-01")
    assert section["title"] == "Branches"
    assert "br-0123456789ab: Read the report" in section["body"]
    assert "Extra details" not in section["body"]
    assert calls[0][2]["params"] == {"session_id": "parent", "requester": "p-01",
                                     "platform": "cli", "user_id": ""}


def test_guest_and_disabled_faculty_do_not_request_branches(provider, home, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("a withheld section made a request")

    monkeypatch.setattr("colonymind_memory.provider._request_within", fail)
    assert provider._branches_section(session_id="parent", contact_id="p-03") is None
    home.write_mind(faculties={"working_memory": False})
    assert provider._branches_section(session_id="parent", contact_id="p-01") is None


def test_owner_compression_queues_record_manifest_after_checkpoint(provider):
    from colonymind_hermes.capture import TurnOutbox
    from colonymind_hermes.client import load_settings
    provider.on_pre_compress([{"role": "user", "content": "The count is seven"}], require_checkpoint=True)
    rows = TurnOutbox(load_settings().outbox_path).rows()
    assert len(rows) == 2
    checkpoint, branch = rows
    assert branch["payload"]["kind"] == "branch_session"
    assert branch["payload"]["checkpoint_turn_id"] == checkpoint["turn_id"]
    assert branch["payload"]["sender"] == {"platform": "cli", "user_id": ""}
    assert "messages" not in branch["payload"]


def test_compression_inherits_branch_handles_but_unrelated_switch_does_not(provider, monkeypatch):
    requested = []

    def reply(timeout, method, url, **kwargs):
        session = kwargs["params"]["session_id"]
        requested.append(session)
        branches = [{"handle": "br-0123456789ab", "brief": "Original session"}] if session == "parent" else []
        return httpx.Response(200, json={"branches": branches, "tool_available": True})

    monkeypatch.setattr("colonymind_memory.provider._request_within", reply)
    provider.on_session_switch("compressed", parent_session_id="parent", reason="compression")
    assert "br-0123456789ab" in provider._branches_section(session_id="compressed", contact_id="p-01")["body"]
    assert requested == ["compressed", "parent"]
    requested.clear()
    provider.on_session_switch("unrelated")
    assert provider._branches_section(session_id="unrelated", contact_id="p-01") is None
    assert requested == ["unrelated"]


def test_compression_hook_only_enqueues_and_faculty_off_keeps_the_existing_checkpoint(provider, home, monkeypatch):
    from colonymind_hermes.capture import TurnOutbox
    from colonymind_hermes.client import load_settings

    def fail(*args, **kwargs):
        pytest.fail("compression made a branch network request")

    monkeypatch.setattr("colonymind_memory.provider._request_within", fail)
    home.write_mind(faculties={"working_memory": False})
    provider.on_pre_compress([{"role": "user", "content": "The count is seven"}], require_checkpoint=True)
    row, = TurnOutbox(load_settings().outbox_path).rows()
    assert row["payload"]["checkpoint_messages"] == [{"role": "user", "content": "The count is seven"}]
    assert "kind" not in row["payload"]


def test_withdrawn_consults_keep_short_handles_without_a_tool_instruction(provider, monkeypatch):
    monkeypatch.setattr("colonymind_memory.provider._request_within", lambda *args, **kwargs: httpx.Response(
        200, json={"branches": [{"handle": "br-0123456789ab", "brief": "x" * 180 + "\nsecond line"},
                                {"handle": "invalid", "brief": "hidden"}], "tool_available": False}))
    section = provider._branches_section(session_id="parent", contact_id="p-01")
    assert section["body"] == "br-0123456789ab: " + "x" * 120


def test_reset_discards_the_compressed_sessions_previous_branch_lineage(provider, monkeypatch):
    requested = []

    def reply(timeout, method, url, **kwargs):
        requested.append(kwargs["params"]["session_id"])
        return httpx.Response(200, json={"branches": [], "tool_available": True})

    monkeypatch.setattr("colonymind_memory.provider._request_within", reply)
    provider.on_session_switch("compressed", parent_session_id="parent", reason="compression")
    provider.on_session_switch("compressed", reset=True)
    assert provider._branches_section(session_id="compressed", contact_id="p-01") is None
    assert requested == ["compressed"]
