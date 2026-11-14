"""The app's shutdown clears every host singleton its startup set.

CI runs the whole suite in one process. A test that starts the app is often followed by one that calls a host
handler directly, and that handler must not find a reranker, a comms log or a telemetry store left behind by an
app that has already closed.
"""

from __future__ import annotations

import inspect
from fastapi.testclient import TestClient

from colonymind.api.routers import host
from colonymind.config import write_api_key


def _singletons() -> list[str]:
    """Each host singleton with a setter: ``set_<name>`` writes the module global ``_<name>``."""
    return sorted(
        name[len("set_"):] for name in dir(host)
        if name.startswith("set_") and callable(getattr(host, name))
        and hasattr(host, "_" + name[len("set_"):])
        and not inspect.isfunction(getattr(host, "_" + name[len("set_"):]))
    )


def _instance(tmp_path, monkeypatch) -> None:
    """A keyed instance whose reranker is a remote one: its warm-up loads nothing and calls no endpoint."""
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    monkeypatch.delenv("COLONYMIND_API_KEY", raising=False)
    monkeypatch.setenv("COLONYMIND_EMBED_PROVIDER", "skip")
    monkeypatch.setenv("COLONYMIND_RERANKER_MODEL", "example-reranker")
    monkeypatch.setenv("COLONYMIND_RERANKER_PROVIDER", "openai_api")
    monkeypatch.setenv("COLONYMIND_RERANKER_BASE_URL", "http://127.0.0.1:9/v1")
    write_api_key("file-key", tmp_path)


def test_shutdown_clears_every_host_singleton_the_startup_set(tmp_path, monkeypatch):
    _instance(tmp_path, monkeypatch)
    names = _singletons()
    assert {"reranker", "comms_log", "telemetry", "embedder"} <= set(names)
    for name in names:
        getattr(host, "set_" + name)(None)

    from colonymind.server import create_app
    with TestClient(create_app(), base_url="http://127.0.0.1:7777") as client:
        assert client.get("/v1/host/health").status_code == 200
        assert host._session_archive_engine() is not None
        wired = sorted(name for name in names if getattr(host, "_" + name) is not None)

    # The startup wired these, so the check below is not vacuous.
    assert {"reranker", "comms_log", "telemetry"} <= set(wired)
    left = {name: type(getattr(host, "_" + name)).__name__ for name in names if getattr(host, "_" + name) is not None}
    assert left == {}
    assert host._session_archive_engine() is None


def test_a_host_handler_after_a_closed_app_finds_no_reranker(tmp_path, monkeypatch):
    _instance(tmp_path, monkeypatch)
    host.set_reranker(None)

    from colonymind.server import create_app
    with TestClient(create_app(), base_url="http://127.0.0.1:7777"):
        assert host._reranker is not None
        assert host._memory_context_selector() is not None

    assert host._reranker is None
    # The recall selector is rebuilt for no reranker rather than kept for the closed one.
    assert host._memory_context_selector() is host._memory_context_selector()
    assert host._context_recall_selector[0] is None
