"""Unreadable lineage cannot establish automation authority; diagnostics are redacted."""
import logging
import sys
from types import SimpleNamespace

import pytest

from colonymind_hermes import capture, guard, owner_only
from colonymind_hermes.authority_diagnostics import authority_counters
from test_unknown_standing import lane  # shared round-1 authority fixture
from guard_offline import offline_home


PRIVATE = "private-canary-never-log"


def unavailable(*args, **kwargs):
    raise RuntimeError(PRIVATE)


def failing_registry(monkeypatch, sessions):
    monkeypatch.setattr(sessions, "parent", capture.SessionMap.parent.__get__(sessions))
    monkeypatch.setitem(sys.modules, "hermes_state_registry", SimpleNamespace(
        acquire=unavailable, release_or_close=lambda db: None))


@pytest.mark.parametrize("standing", ["guest", "owner", "unknown"])
@pytest.mark.parametrize("observed", [False, True])
def test_unreadable_registry_outside_cron_requires_observed_sender(lane, monkeypatch, standing, observed, caplog):
    """A session the plugin saw a turn of keeps its sender's standing while Hermes' session store cannot be read.
    One it has not seen has no standing of its own: an unreadable store is not "no parent", so the session is
    unknown and gets no tools, whoever the gateway bound for the turn, until the plugin sees its sender's turn."""
    gate, sessions = lane
    failing_registry(monkeypatch, sessions)
    sender = "owner-1" if standing == "owner" else "guest-1" if standing == "guest" else ""
    if observed and sender:
        sessions.observe(session_id=PRIVATE, platform="telegram", sender_id=sender)
    monkeypatch.setattr(capture, "session_env", lambda: ("telegram" if sender else "", sender, PRIVATE, "group"))
    own = standing if observed else "unknown"
    with caplog.at_level(logging.INFO):
        assert gate._cron_origin(PRIVATE) == ""
        assert gate.standing(PRIVATE)[0] == own
        result = gate.pre_tool_call("terminal", {"opaque": PRIVATE}, PRIVATE)
    assert (result or {}).get("action") == (None if own == "owner" else "block")
    assert "mind_run_decision" not in caplog.text
    if not observed or not sender:
        assert "ancestry_unreadable" in caplog.text
    assert PRIVATE not in caplog.text
    if sender and not observed:
        # The store still unread, the sender's next turn is one the plugin sees: the session's own standing is back.
        sessions.observe(session_id=PRIVATE, platform="telegram", sender_id=sender)
        assert gate.standing(PRIVATE)[0] == standing


@pytest.mark.parametrize("marker", ["context", "task"])
def test_unreadable_registry_in_positive_cron_context_stays_closed(lane, monkeypatch, marker, caplog):
    gate, sessions = lane
    failing_registry(monkeypatch, sessions)
    monkeypatch.setattr(guard, "cron_context", lambda: marker == "context")
    gate.client.has_mind_routes = lambda: False
    task = "cron:private-job:run" if marker == "task" else ""
    with caplog.at_level(logging.INFO):
        assert gate._cron_origin(PRIVATE, task) == "unknown"
        assert not gate._mind_cron(PRIVATE, task)
        result = gate.pre_tool_call("terminal", {"opaque": PRIVATE}, PRIVATE, task)
    assert result["action"] == "block"
    assert "source=cron origin=unknown reason=standing_unknown" in caplog.text
    assert PRIVATE not in caplog.text and "private-job" not in caplog.text


@pytest.mark.parametrize("failure", ["acquire", "query", "release", "guard"])
def test_both_ancestry_read_paths_warn_count_and_redact(lane, monkeypatch, caplog, failure):
    gate, sessions = lane
    monkeypatch.setattr(sessions, "parent", capture.SessionMap.parent.__get__(sessions))
    db = SimpleNamespace(get_session=unavailable if failure == "query" else lambda _: None)
    monkeypatch.setitem(sys.modules, "hermes_state_registry", SimpleNamespace(
        acquire=unavailable if failure == "acquire" else lambda: db,
        release_or_close=unavailable if failure == "release" else lambda _: None))
    if failure == "guard":
        monkeypatch.setattr(sessions, "parent", unavailable)
    before = authority_counters()["ancestry_unreadable"]
    with caplog.at_level(logging.WARNING):
        gate._parent_session(PRIVATE)
    assert authority_counters()["ancestry_unreadable"] == before + 1
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert caplog.records[0].getMessage() == "ancestry_unreadable exception_type=RuntimeError"
    assert PRIVATE not in caplog.text


@pytest.mark.parametrize("origin", ["mind", "owner", "unknown"])
@pytest.mark.parametrize("tool", ["web_search", "terminal"])
def test_every_cron_authority_decision_logs_source_origin_reason(lane, monkeypatch, caplog, origin, tool):
    gate, _ = lane
    monkeypatch.setattr(gate, "_cron_origin", lambda *a: origin)
    monkeypatch.setattr(gate, "_rules", lambda *a, **kw: None)
    with caplog.at_level(logging.INFO):
        result = gate.pre_tool_call(tool, {"opaque": PRIVATE}, PRIVATE)
    event, = [record for record in caplog.records if "mind_run_decision" in record.message]
    reason = "mind_run" if origin == "mind" else "standing_unknown" if origin == "unknown" else "session_standing"
    assert event.levelno == logging.INFO
    assert event.message == f"mind_run_decision tool={tool} source=cron origin={origin} reason={reason}"
    assert (result or {}).get("action") == ("block" if origin == "unknown" else None)
    assert PRIVATE not in caplog.text


def test_kanban_mind_decision_logs_source(lane, monkeypatch, caplog):
    gate, sessions = lane
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    sessions.standings.put("kanban", "task-1", "mind", "worker")
    monkeypatch.setattr(guard, "mind_run", lambda _: True)
    with caplog.at_level(logging.INFO):
        assert gate.pre_tool_call("web_search", {"opaque": PRIVATE}, PRIVATE) is None
    assert "tool=web_search source=kanban_worker origin=mind reason=mind_run" in caplog.text
    assert PRIVATE not in caplog.text


@pytest.mark.parametrize("standing", ["owner", "guest", "unknown"])
def test_worker_decision_logs_saved_standing(lane, monkeypatch, caplog, standing):
    gate, sessions = lane
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    sessions.standings.put("kanban", "task-1", standing, "worker")
    with caplog.at_level(logging.INFO):
        gate.pre_tool_call("terminal", {"opaque": PRIVATE}, PRIVATE)
    assert f"source=kanban_worker origin={standing}" in caplog.text
    assert PRIVATE not in caplog.text


@pytest.mark.parametrize("owner", [False, True])
def test_standing_fallback_warns_even_when_owner_is_independently_verified(lane, monkeypatch, caplog, owner):
    gate, sessions = lane
    monkeypatch.setattr(gate, "standing", unavailable)
    monkeypatch.setattr(sessions, "known_owner", lambda _: owner)
    with caplog.at_level(logging.WARNING):
        result = gate.pre_tool_call("terminal", {"opaque": PRIVATE}, PRIVATE)
    assert (result or {}).get("action") == (None if owner else "block")
    assert "standing_fallback tool=terminal exception_type=RuntimeError standing=" in caplog.text
    assert PRIVATE not in caplog.text


@pytest.mark.parametrize("path", ["bound", "observed", "sender", "audience"])
def test_unresolved_sender_and_withheld_owner_audience_warn(lane, monkeypatch, caplog, path):
    _, sessions = lane
    sessions.client.resolve_contact = unavailable
    with caplog.at_level(logging.WARNING):
        if path == "bound":
            monkeypatch.setattr(capture, "session_env", lambda: ("telegram", PRIVATE, PRIVATE, "group"))
            assert sessions.standing(PRIVATE)[0] == "guest"
        elif path == "observed":
            sessions.observe(session_id=PRIVATE, platform="telegram", sender_id=PRIVATE)
            assert sessions.standing(PRIVATE)[0] == "guest"
        elif path == "sender":
            assert not sessions.sender_is_owner("telegram", PRIVATE)
        else:
            monkeypatch.setattr(sessions, "owner_only", unavailable)
            assert not owner_only(sessions, {"session_id": PRIVATE})
    assert caplog.records and all(record.levelno == logging.WARNING for record in caplog.records)
    assert "standing_" in caplog.text and "exception_type=RuntimeError" in caplog.text
    assert PRIVATE not in caplog.text


def test_status_exposes_process_ancestry_counter(offline_home):
    from guard_offline import probe
    result = probe('''
from colonymind_hermes.authority_diagnostics import ancestry_unreadable, authority_counters
from colonymind_hermes.commands import status_text
ancestry_unreadable(colonymind_hermes.logger, RuntimeError("private-canary-never-log"))
count = authority_counters()["ancestry_unreadable"]
text = status_text(colonymind_hermes._BODY.client, colonymind_hermes._BODY.settings,
                   colonymind_hermes._BODY.outbox)
emit(exposed=f"authority: {count} ancestry_unreadable" in text)
''', offline_home, env={"COLONYMIND_BODY_THREAD": "0"})
    assert result == {"exposed": True}
