"""Memory and reminder contracts against the selected Step 0 Hermes source."""
from __future__ import annotations

import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_source import HERMES_ROOT, assert_source_module, assert_supported_source

ROOT = Path(__file__).resolve().parents[2]
QUERY = "Where is the office for our appointment?"
SECTION = {"id": "office", "title": "Relevant Memories", "body": "The office is room 4."}


@pytest.fixture
def selected(monkeypatch, tmp_path):
    if not os.environ.get("HERMES_SOURCE_ROOT"):
        pytest.skip("set HERMES_SOURCE_ROOT to a pinned Step 0 source")
    revision = assert_supported_source()
    monkeypatch.syspath_prepend(str(HERMES_ROOT))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path / "instance"))
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    from agent import turn_context
    assert_source_module(turn_context)
    return revision


def _provider(monkeypatch, platform, cls=None):
    from colonymind_memory.provider import ColonyMindMemoryProvider
    provider = (cls or ColonyMindMemoryProvider)(config={
        "url": "http://sidecar.test", "api_key": "synthetic-test-key", "contact_id": "owner",
        "turn_writer": "disabled"})
    provider.initialize("session", platform=platform)
    monkeypatch.setattr(provider, "_turn_sender_context", lambda session_id="": (platform, "owner", "owner"))
    monkeypatch.setattr(provider, "_prefetch_contact", lambda *args, **kwargs: "owner")
    monkeypatch.setattr(provider, "_assemble", lambda *args, **kwargs: ([dict(SECTION)], {}))
    monkeypatch.setattr(provider, "_with_fresh_temporal_sync", lambda text, **kwargs: text + "\n## Current Time\nNow.")
    return provider


def _turn(provider):
    text = provider.prefetch(QUERY, session_id="session")
    provider.sync_turn(QUERY, "Thanks.", session_id="session")
    return text


def test_api_client_history_receives_full_memory_on_second_turn(selected, monkeypatch):
    provider = _provider(monkeypatch, "api_server")
    assert provider._reuses_api_memory is (selected in {"S", "S217"})
    first = _turn(provider)
    if selected in {"S", "S217"}:
        from gateway.platforms.api_server_memory_sessions import ApiServerMemorySessions
        registry = ApiServerMemorySessions(max_size=2, idle_ttl_secs=60)
        manager = SimpleNamespace(provider=provider)
        registry.checkin(SimpleNamespace(_memory_manager=manager, session_id="session"))
        reused = registry.checkout("session")
        assert reused is manager
        second = _turn(reused.provider)
        assert provider._shown == {} and provider._offered == {} and provider._shown_field("session") == {}
    else:
        assert provider._shown and "session" in provider._shown
        # A host without API memory reuse creates a new manager and provider for each API request.
        second = _turn(_provider(monkeypatch, "api_server"))
    assert SECTION["body"] in first and SECTION["body"] in second
    assert "unchanged since" not in first + second


def test_non_api_memory_deduplication_unchanged(selected, monkeypatch):
    provider = _provider(monkeypatch, "telegram")
    assert SECTION["body"] in _turn(provider)
    second = _turn(provider)
    assert SECTION["body"] not in second
    assert "unchanged since your last turn: Relevant Memories" in second
    assert "## Current Time" in second
    assert provider._shown_field("session")["shown"]


@pytest.mark.parametrize("message", [
    QUERY,
    [{"type": "text", "text": QUERY}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,PRIVATE"}}],
    [{"type": "image_url", "image_url": {"url": "data:image/png;base64,PRIVATE"}}],
])
def test_multimodal_turn_has_one_clock(selected, monkeypatch, message):
    import colonymind_memory
    from agent import turn_context
    from agent.subagent_lifecycle import bind_subagent_parent
    provider = _provider(monkeypatch, "telegram")
    hooks = {}
    monkeypatch.setattr(provider, "resolve_contact", lambda **kwargs: None)
    colonymind_memory.register(SimpleNamespace(
        register_memory_provider=lambda value: None, register_hook=lambda name, callback: hooks.update({name: callback})))
    current = SimpleNamespace(session_id="session", _memory_manager=SimpleNamespace(providers=[provider]))
    with bind_subagent_parent(current):
        hook = hooks["pre_llm_call"](user_message=message, session_id="session", platform="telegram")
    queries = []
    def prefetch(query, **kwargs):
        queries.append(query)
        return provider.prefetch(query, **kwargs)
    agent = SimpleNamespace(session_id="session", _user_turn_count=1, _memory_manager=SimpleNamespace(
        on_turn_start=lambda *args, **kwargs: None, prefetch_all=prefetch, describe_recall=lambda: ""))
    # What the host's signature takes beside the message: a turn with no author and no earlier message.
    taken = inspect.signature(turn_context._memory_turn_start_and_prefetch).parameters
    context = {name: value for name, value in {"turn_author": None, "previous_message": ""}.items() if name in taken}
    recall = turn_context._memory_turn_start_and_prefetch(agent, message, **context)
    combined = (hook or {}).get("context", "") + recall
    # One clock a turn: the recalled context's "Current Time" block, or, where the host recalls nothing, the hook's
    # own line ("the assistant's time: ...", with the name the instance configured).
    assert combined.count("Current Time") + combined.count("'s time: ") == 1
    assert "PRIVATE" not in combined
    should_recall = isinstance(message, str) or (selected in {"S", "S217"} and message[0]["type"] == "text")
    assert queries == ([QUERY] if should_recall else [])
    assert (SECTION["body"] in combined) is should_recall


def _reminder(monkeypatch, tmp_path, platform, home_target, cls=None):
    from cron import jobs, scheduler, scheduler_delivery, scheduler_provider
    from gateway import session_context
    from colonymind_hermes.reminders import Reminders
    monkeypatch.setattr(jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", tmp_path / "cron" / "output")
    if hasattr(jobs, "_compute_provider_model_snapshots"):
        # Some hosts snapshot runtime credentials when creating a job. Origin-routing
        # fixtures must not probe cloud metadata or resolve real providers.
        monkeypatch.setattr(jobs, "_compute_provider_model_snapshots", lambda **kwargs: (None, None))
    monkeypatch.setattr(scheduler, "load_config", lambda: {})
    monkeypatch.setattr(scheduler_provider, "resolve_cron_scheduler", lambda: SimpleNamespace(register_job=lambda job: None))
    monkeypatch.setattr(scheduler_delivery, "_iter_home_target_platforms", lambda: ["telegram"])
    monkeypatch.setattr(scheduler_delivery, "_get_home_target_chat_id", lambda platform: home_target)
    monkeypatch.setattr(scheduler_delivery, "_get_home_target_thread_id", lambda platform: None)
    monkeypatch.setattr(session_context, "get_session_env", lambda name, default="": {
        "HERMES_SESSION_PLATFORM": platform, "HERMES_SESSION_CHAT_ID": "chat" if platform else "",
    }.get(name, default))
    if hasattr(session_context, "async_delivery_supported"):
        monkeypatch.setattr(session_context, "async_delivery_supported", lambda: platform != "api_server")
    current = {"status": "current", "deadline_at": "2099-01-01T12:00:00+00:00", "subject": "the report",
               "predicate": "is due", "value": "tomorrow"}
    client = SimpleNamespace(post=lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: current))
    sessions = SimpleNamespace(is_owner=lambda session: True, contact_id=lambda session: "owner")
    reminder = (cls or Reminders)(client, sessions)
    result = reminder._handle({"operation": "schedule", "source_id": "source", "source_version": "version",
                               "claim_id": "claim"}, "session")
    job = jobs.get_job(result["job_id"])
    target = scheduler_delivery._resolve_delivery_target(job)
    return job, target


@pytest.mark.parametrize("platform,home_target", [("telegram", "home"), ("", "home"), ("api_server", "home"),
                                                 ("api_server", "")])
def test_reminder_explicit_origin_selected_source(selected, monkeypatch, tmp_path, platform, home_target):
    from cron import scheduler_delivery
    assert hasattr(scheduler_delivery, "_NON_PUSH_ORIGIN_PLATFORMS") is (selected in {"S", "S217"})
    job, target = _reminder(monkeypatch, tmp_path, platform, home_target)
    assert job["attach_to_session"] is True
    if selected not in {"S", "S217"}:
        assert job["deliver"] == ("origin" if platform else "local")
        assert (target or {}).get("platform") == (platform or None)
        assert (target or {}).get("chat_id") == ("chat" if platform else None)
    else:
        assert job["deliver"] == "origin"
        if platform == "telegram":
            assert job["origin"]["platform"] == "telegram" and job["origin"]["chat_id"] == "chat"
            assert target["platform"] == "telegram" and target["chat_id"] == "chat"
        elif home_target:
            assert job["origin"] is None
            assert target["platform"] == "telegram" and target["chat_id"] == home_target
        else:
            assert job["origin"] is None
            assert target is None
