"""Exercise Hermes' real collector replacement and execution-context dispatch."""
import gc
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import pytest

import colonymind_memory


@pytest.fixture
def lifecycle(monkeypatch, tmp_path):
    from agent.memory_manager import MemoryManager
    from agent.subagent_lifecycle import bind_subagent_parent
    from hermes_cli import plugins
    from plugins import memory

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path / "instance"))
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    entry = SimpleNamespace(name="colonymind-memory", load=lambda: colonymind_memory)

    def load(*args, **kwargs):
        # The entry-point loader calls _ProviderCollector.collect, including
        # _drop_fallback_hooks. No general plugin discovery owns this source.
        return memory._load_provider_from_entry_point(entry, register_skills=False)

    monkeypatch.setattr(memory, "load_memory_provider", load)

    def build(session):
        provider = load()
        provider.initialize(session, platform="telegram")
        provider._resolve_handle = Mock(return_value="contact-" + session)
        provider.resolve_contact = Mock(wraps=provider.resolve_contact)
        provider._prefetch_contact = Mock(return_value=None)
        provider._clock_line = Mock(return_value="clock-" + session)
        provider.turn_clock = Mock(wraps=provider.turn_clock)
        owner = MemoryManager()
        owner.add_provider(provider)
        return SimpleNamespace(session_id=session, _memory_manager=owner), provider

    def turn(agent, turn_id="turn-1", **overrides):
        event = dict(session_id=agent.session_id, turn_id=turn_id,
                     platform="telegram", sender_id="sender-" + agent.session_id,
                     user_message="Where is the appointment?")
        event.update(overrides)
        with bind_subagent_parent(agent):
            return manager.invoke_hook("pre_llm_call", **event)

    return SimpleNamespace(manager=manager, build=build, turn=turn, load=load)


@pytest.mark.parametrize("reload_kind", ["second-agent", "memoised-identity"])
def test_first_provider_receives_turn_after_recollection(lifecycle, reload_kind):
    first, provider = lifecycle.build("first")
    assert lifecycle.turn(first) == [{"context": "clock-first"}]
    previous_hook, = lifecycle.manager._hooks["pre_llm_call"]
    if reload_kind == "second-agent":
        other, other_provider = lifecycle.build("second")
    else:
        from gateway.run_agent_cache import GatewayAgentCacheMixin

        class Cache(GatewayAgentCacheMixin):
            _MEMORY_IDENTITY_PROVIDER_MEMO = {}

        Cache._memory_provider_identity_signature("colonymind-memory")
        other_provider = Cache._MEMORY_IDENTITY_PROVIDER_MEMO["colonymind-memory"]
        hook, = lifecycle.manager._hooks["pre_llm_call"]
        Cache._memory_provider_identity_signature("colonymind-memory")
        assert lifecycle.manager._hooks["pre_llm_call"] == [hook]
    hook, = lifecycle.manager._hooks["pre_llm_call"]
    assert hook is not previous_hook
    assert provider is not other_provider
    assert lifecycle.turn(first, "after-reload") == [{"context": "clock-first"}]
    assert provider._turn_ids["first"] == "after-reload"
    assert provider._turn_senders["first"] == ("telegram", "sender-first")
    assert provider.resolve_contact.call_count == 2
    assert provider._resolve_handle.call_args.args == ("telegram", "sender-first")
    assert provider.turn_clock.call_count == 2
    assert "first" not in other_provider._turn_ids


@pytest.mark.parametrize("same_session", [False, True])
def test_concurrent_agents_only_update_their_own_provider(lifecycle, same_session):
    first, p1 = lifecycle.build("first")
    second, p2 = lifecycle.build("first" if same_session else "second")
    barrier = Barrier(2)

    def run(agent, turn_id):
        barrier.wait(timeout=10)
        return lifecycle.turn(agent, turn_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(run, first, "one")
        b = pool.submit(run, second, "two")
        assert a.result() == [{"context": "clock-first"}]
        assert b.result() == [{"context": "clock-" + second.session_id}]
    assert p1._turn_ids == {first.session_id: "one"}
    assert p2._turn_ids == {second.session_id: "two"}
    assert p1.resolve_contact.call_count == p2.resolve_contact.call_count == 1
    assert p1.turn_clock.call_count == p2.turn_clock.call_count == 1


def test_retained_hook_does_not_keep_provider_alive(lifecycle):
    agent, provider = lifecycle.build("discarded")
    reference = weakref.ref(provider)
    assert len(lifecycle.manager._hooks["pre_llm_call"]) == 1
    del agent, provider
    gc.collect()
    assert reference() is None


def test_unbound_or_different_session_does_not_update_provider(lifecycle):
    agent, provider = lifecycle.build("first")
    assert lifecycle.manager.invoke_hook("pre_llm_call", session_id="first") == []
    assert lifecycle.turn(agent, session_id="unrelated") == []
    assert provider._turn_ids == {}
    provider.turn_clock.assert_not_called()


def test_agent_without_colonymind_does_not_use_registered_provider(lifecycle):
    agent, provider = lifecycle.build("first")
    agent._memory_manager = None
    assert lifecycle.turn(agent) == []
    assert provider._turn_ids == {}
    provider.turn_clock.assert_not_called()


def test_session_rotation_and_reused_memory_manager(lifecycle):
    agent, provider = lifecycle.build("first")
    lifecycle.turn(agent)
    provider.on_session_switch("rotated", parent_session_id="first", reason="compression")
    agent.session_id = "rotated"
    reused = SimpleNamespace(session_id="rotated", _memory_manager=agent._memory_manager)
    lifecycle.load()  # an identity probe must not take ownership of the hook
    assert lifecycle.turn(reused, "rotated-turn") == [{"context": "clock-first"}]
    assert provider._turn_ids["rotated"] == "rotated-turn"
    assert provider._turn_senders["rotated"] == ("telegram", "sender-rotated")
