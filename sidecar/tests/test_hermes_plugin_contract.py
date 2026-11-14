"""Contract test: the Hermes colonymind plugin's endpoints must exist in the host API.

The plugin (plugins/hermes-plugin) lives in this repo precisely so its
tool->endpoint mappings stay in lockstep with the API. This test auto-discovers
every `/v1/host/...` path the plugin references and asserts each one matches a
registered route. It exists because a `colonymind_task_*` tool quietly called a
removed `/v1/host/tasks/...` endpoint for a long time, failing silently in
production — exactly the drift this catches at test time.
"""

import pathlib
import re

from fastapi import FastAPI
import pytest

from colonymind.api.routers.host import router as host_router
from colonymind.api.routers.executions import router as executions_router

from colonymind.api.routers import social_state, temporal_followups, transport, work, curated_memory

# Every router the app mounts under /v1/host — the plugin may hit any of them.
_HOST_ROUTERS = (host_router, executions_router, social_state.router,
                 temporal_followups.router, transport.router, work.router, curated_memory.router)

_INTEGRATION = pathlib.Path(__file__).resolve().parents[2] / "plugins" / "hermes-plugin"
_HOST_PATH = re.compile(r"/v1/host/(?:[A-Za-z0-9/_.-]|\{[^{}\r\n]*\})*")


def _normalize(path: str) -> str:
    path = path.split("?", 1)[0]              # drop query string
    path = path.rstrip("/")                   # ignore trailing slash
    return re.sub(r"\{[^}]*\}", "{}", path)   # path params (incl f-string exprs) -> {}


def _plugin_paths() -> set[str]:
    """Every /v1/host/... path the plugin source references, normalized.

    Quoted expressions inside f-string braces remain intact, while the closing
    quote of either a single- or double-quoted path is not part of the URL.
    """
    paths: set[str] = set()
    for f in sorted(_INTEGRATION.glob("*.py")):
        src = f.read_text(encoding="utf-8")
        for raw in _HOST_PATH.findall(src):
            if "..." in raw:        # prose ellipsis in a docstring/prompt, not a real path
                continue
            paths.add(_normalize(raw))
    return paths


@pytest.mark.parametrize("source,expected", [
    ("client.post('/v1/host/memory/curated/sources/forget', timeout=3)", "/v1/host/memory/curated/sources/forget"),
    ('client.get("/v1/host/autonomy/status")', "/v1/host/autonomy/status"),
    ('client.post(f"/v1/host/initiatives/{args[\'initiative_id\']}/finish")', "/v1/host/initiatives/{}/finish"),
])
def test_path_discovery_keeps_expression_quotes_only(source, expected):
    assert [_normalize(path) for path in _HOST_PATH.findall(source)] == [expected]


def _api_paths() -> set[str]:
    # Let FastAPI resolve included routers and their mount prefixes, as the host does.
    app = FastAPI()
    for router in _HOST_ROUTERS:
        app.include_router(router)
    return {
        _normalize(path)
        for path in app.openapi()["paths"]
        if path.startswith("/v1/host")
    }


def test_integration_dir_present():
    assert _INTEGRATION.is_dir(), f"colonymind Hermes integration missing at {_INTEGRATION}"


def test_plugin_endpoints_all_exist_in_host_api():
    plugin = _plugin_paths()
    assert plugin, "found no /v1/host paths in the plugin — regex or layout changed?"
    api = _api_paths()
    missing = sorted(p for p in plugin if p not in api)
    assert not missing, (
        "Hermes colonymind plugin references endpoints that are NOT registered in the host "
        f"API (contract drift — these will 404/405 silently in production):\n  {missing}\n"
        f"Plugin paths checked: {sorted(plugin)}"
    )
