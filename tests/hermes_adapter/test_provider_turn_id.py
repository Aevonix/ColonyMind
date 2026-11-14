"""Only an owner's private recall carries the host's captured turn identity."""
import httpx
import pytest

from colonymind_memory.provider import ColonyMindMemoryProvider


@pytest.fixture
def provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    key = tmp_path / "api.key"
    key.write_text("fixture-key")
    value = ColonyMindMemoryProvider({"url": "http://127.0.0.1:1", "key_file": str(key), "contact_id": "owner"})
    value.initialize("session", platform="cli")
    monkeypatch.setattr(value, "_resolve_handle", lambda *a, **kw: "contact")
    return value


def body(provider, monkeypatch, *, contact="owner", session="session"):
    calls = []
    def reply(timeout, method, url, **kwargs):
        calls.append(kwargs["json"])
        return httpx.Response(200, json={"sections": []}, request=httpx.Request(method, url))
    monkeypatch.setattr("colonymind_memory.provider._request_within", reply)
    provider._prefetch_sections("Where is the folder?", session_id=session, contact_id=contact)
    return calls[-1]


def test_private_owner_id_and_missing_id_clear_previous(provider, monkeypatch):
    provider.resolve_contact("cli", "", "session", turn_id="host-turn")
    assert body(provider, monkeypatch)["context"]["turn_id"] == "host-turn"
    provider.resolve_contact("cli", "", "session")
    assert "turn_id" not in body(provider, monkeypatch)["context"]


@pytest.mark.parametrize("contact", ["contact", "guest"])
def test_other_audiences_carry_no_id(provider, monkeypatch, contact):
    provider.resolve_contact("cli", "", "session", turn_id="host-turn")
    assert "turn_id" not in body(provider, monkeypatch, contact=contact)["context"]


def test_session_ids_do_not_cross(provider, monkeypatch):
    provider.resolve_contact("cli", "", "session", turn_id="first")
    provider.resolve_contact("cli", "", "second", turn_id="second")
    assert body(provider, monkeypatch)["context"]["turn_id"] == "first"
    assert body(provider, monkeypatch, session="second")["context"]["turn_id"] == "second"


def test_registration_forwards_the_id_seen_by_capture(provider, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from agent.subagent_lifecycle import bind_subagent_parent
    import colonymind_memory
    from colonymind_hermes.capture import Capture, SessionMap
    hooks = []
    context = SimpleNamespace(register_memory_provider=lambda value: None,
                              register_hook=lambda name, function: hooks.append(function))
    colonymind_memory.register(context)
    capture = Capture(SessionMap(SimpleNamespace(outbox_path=tmp_path / 'outbox.sqlite3'), None), None, None)
    event = dict(session_id='session', turn_id='captured-host-turn', platform='cli',
                 sender_id='', user_message='Where is the folder?')
    capture.pre_llm_call(**event)
    agent = SimpleNamespace(session_id='session', _memory_manager=SimpleNamespace(providers=[provider]))
    with bind_subagent_parent(agent):
        for hook in hooks:
            hook(**event)
    assert body(provider, monkeypatch)['context']['turn_id'] == capture.sessions.get('session').turn_id


def test_compression_keeps_turn_id_and_reset_clears_it(provider, monkeypatch):
    provider.resolve_contact('cli', '', 'session', turn_id='continuing-turn')
    provider.on_session_switch('compressed', parent_session_id='session', reason='compression')
    assert body(provider, monkeypatch, session='compressed')['context']['turn_id'] == 'continuing-turn'
    provider.on_session_switch('compressed', reset=True)
    assert 'turn_id' not in body(provider, monkeypatch, session='compressed')['context']
