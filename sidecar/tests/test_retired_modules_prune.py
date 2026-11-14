"""The prune: what the Hermes body already does, and what nothing used, is gone from the sidecar.

The agent SDK and the agent registry, briefings, channels, connectors, the legacy goals, the MCP server
and its harness setup, observations, patterns, proposals, repo mirrors, research and search, sessions,
the event bus and its models, the situation spine and the competence reconciliation,
the intelligence components other than the preference learner, and the theory-of-mind shared facts (the
contacts' affect stays: the appraisals write it and the social drive reads it; it moves into contacts with
the comms log's source lineage). Nothing imports them, no route serves them, no client calls those routes, no capability or
environment flag names them, the lifespan builds none of their stores, and the state they left behind
moves into the upgrade backup (moved aside, never deleted).
"""

import ast
import functools
from importlib.util import find_spec
from pathlib import Path
import inspect
import re
import sqlite3
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PACKAGE = ROOT / "colonymind"

RETIRED_MODULES = (
    "colonymind.self_model.benchmark",
    "colonymind.self_model.runtime_models",
    "colonymind.router.native_policy",
    "colonymind.qualification.native_learning",
    "colonymind.qualification.native_learning_cases",
    "colonymind.qualification.native_learning_worker",
    "colonymind.qualification.native_learning_extended",

    "colonymind.agent",
    "colonymind.agents",
    "colonymind.briefings",
    "colonymind.channels",
    "colonymind.connectors",
    "colonymind.goals",
    "colonymind.harness_integration",
    "colonymind.mcp",
    "colonymind.models",
    "colonymind.observations",
    "colonymind.patterns",
    "colonymind.proposals",
    "colonymind.repos",
    "colonymind.research",
    "colonymind.sessions",
    "colonymind.api.routers.observations",
    "colonymind.self_model.reconcile",
    "colonymind.self_model.situation",
    "colonymind.self_model.settlement",
    "colonymind.intelligence.synthesis",
    "colonymind.intelligence.learning",
    "colonymind.intelligence.turn_middleware",
    "colonymind.intelligence.components.anomaly_detector",
    "colonymind.intelligence.components.research_orchestrator",
    "colonymind.intelligence.components.self_reflector",
    "colonymind.intelligence.components.session_continuity",
    "colonymind.intelligence.components.task_planner",
    "colonymind.intelligence.components.tool_learner",
    "colonymind.tom",
    "colonymind.events.bus",
    "colonymind.events.types",
    "colonymind.initiatives.assignment",
    "colonymind.util.quiet_hours",
)
# What the mind and the body still use stays where it was, or (the comms log's source lineage) moves.
KEPT_MODULES = (
    "colonymind.intelligence.components.preference_learner",
    "colonymind.contacts.source_lineage",
    "colonymind.contacts.affect",
    "colonymind.contacts.affect_writer",
    "colonymind.secrets",
    "colonymind.self_model.judgments",
    "colonymind.self_model.appraisals",
    "colonymind.self_model.expectations",
    "colonymind.self_model.perspective",
    "colonymind.self_model.reply_forecasts",
    "colonymind.events.journal",
    "colonymind.events.broadcaster",
    "colonymind.events.stream",
    "colonymind.feedback",
    "colonymind.initiatives.store",
    "colonymind.mind.goals",
    # What the routes callers outside this repository still read need (CALLER_ROUTES).
    "colonymind.self_model.store",
    "colonymind.self_model.journal",
    "colonymind.self_model.brief",
)
RETIRED_ROUTE_PREFIXES = (
    "/v1/host/agents", "/v1/host/briefings", "/v1/host/search", "/v1/host/research", "/v1/host/connectors",
    "/v1/host/synthesis", "/v1/host/insights", "/v1/host/learning/correction", "/v1/host/self/benchmark/",
    "/v1/host/self/situation", "/v1/host/affect", "/v1/host/mind/facts",
    "/v1/host/patterns", "/v1/host/proposals", "/v1/host/observations", "/v1/host/repos",
    "/v1/host/session-report", "/v1/host/context-digest", "/v1/host/goals", "/v1/host/tasks", "/v1/channels",
)
RETIRED_ROUTES = ("/v2/host/turns/task-instruction/{turn_id:path}",
                  "/v2/host/turns/source-media/transport/{turn_id:path}",
                  "/v1/host/social/appraisals/correct", "/v1/host/social/contacts",
                  "/v1/host/self", "/v1/host/self/benchmark", "/v1/host/initiatives/{initiative_id}/claim", "/v1/host/initiatives/{initiative_id}/delegate",
                  "/v1/host/memory/sources/forget")
# The journal remains public; benchmark state is retired by the install unit.
CALLER_ROUTES = ("/v1/host/self/journal",)
CALLER_STATE = ("colonymind-self-model.db", "colonymind-action-journal.db")
RETIRED_CAPABILITIES = ("goals", "briefings", "situation", "research", "synthesis", "sessions", "shared_facts",
                        "patterns")
RETIRED_ENV = ("COLONYMIND_BRIEFING", "COLONYMIND_SEARCH_PROVIDER", "TAVILY_API_KEY", "BRAVE_API_KEY", "SERPAPI_KEY",
               "COLONYMIND_CONNECTOR", "COLONYMIND_SITUATION_", "COLONYMIND_BENCHMARK_", "COLONYMIND_SELF_MODEL_ENABLED",
               "COLONYMIND_MCP_", "COLONYMIND_CODE_PEPPER")
# The stores the pruned code wrote, and the leftovers no code names, in the instance directory.
PRUNED_STATE = (
    "briefings.db", "insights.db", "observations.db", "agents.db", "agents.db.backup", "repo-mirrors",
    "goals.db", "colonymind-goals.db", "colonymind-patterns.db", "colonymind-proposals.db", "colonymind-channels.db",
    "colonymind-facts.db", "colonymind-situation.db", "colonymind-learning-feedback.db",
    "skills.db", "sandbox", "autonomy_checkin.json", "autonomy_synthesis.json",
)
PRUNED_WORDS = re.compile(
    r"briefing|research pipeline|search provider|connector|situation spine|benchmark|selfmodel|self.model|"
    r"competence|action journal|shared ?facts|pattern ?store|proposal ?store|channel ?store|"
    r"session ?store|session report|repo ?mirror|agent ?store|invite ?store|assignment|websocket manager|"
    r"observation ?store|insight ?store|goal records|mcp|forget_turn_sources|erase_sources|source erasure cleanup", re.I)


def absent(module: str) -> bool:
    try:
        return find_spec(module) is None
    except ModuleNotFoundError:
        return True


@pytest.mark.parametrize("module", RETIRED_MODULES)
def test_the_pruned_modules_do_not_exist(module):
    assert absent(module), f"{module} still exists"


@pytest.mark.parametrize("module", KEPT_MODULES)
def test_what_the_mind_and_the_body_use_is_still_there(module):
    assert not absent(module), f"{module} is gone"


def test_the_preference_learner_imports_no_pruned_component():
    """``intelligence.components`` used to import every component on import; it exports the learner only."""
    import colonymind.intelligence.components as components
    assert components.__all__ == ["Preference", "PreferenceLearner"]


def _served_paths() -> set[str]:
    from colonymind.api.routers import host
    from colonymind.server import create_app
    app = create_app()
    paths = set(app.openapi().get("paths", {}))
    # Websocket routes and mounts are not in the OpenAPI table.
    paths |= {getattr(route, "path", "") for router in (host.router,) for route in router.routes}
    paths |= {getattr(route, "path", "") for route in app.routes}
    return paths


def test_no_pruned_route_is_served():
    paths = _served_paths()
    assert "/v1/host/context/assemble" in paths and "/v1/host/self/expectations" in paths
    served = sorted(path for path in paths
                    if path.startswith(RETIRED_ROUTE_PREFIXES) or path in RETIRED_ROUTES)
    assert served == []


def test_the_routes_callers_outside_this_repository_read_still_serve_their_records(tmp_path, monkeypatch):
    """The journal survives upgrades and the expectation engine keeps writing it."""
    from fastapi.testclient import TestClient

    from colonymind import init
    from colonymind.config import write_api_key
    from colonymind.self_model.journal import ActionJournal
    from colonymind.self_model.store import CompetenceStore
    from colonymind.server import create_app

    journal = ActionJournal(db_path=str(tmp_path / "colonymind-action-journal.db"))
    journal.record("expectation", "prediction hit: the parcel arrives today", decision="noted", outcome="hit")
    journal._conn.close()
    competence = CompetenceStore(db_path=str(tmp_path / "colonymind-self-model.db"))
    competence.record("delivery", "success", latency_secs=1.5)
    competence.close()
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", "owner-contact")
    monkeypatch.delenv("COLONYMIND_API_KEY", raising=False)
    write_api_key("caller-key", tmp_path)
    headers = {"Authorization": "Bearer caller-key"}
    with TestClient(create_app(), base_url="http://127.0.0.1:7777", client=("127.0.0.1", 50000)) as client:
        served = client.get("/v1/host/self/journal", params={"limit": 5}, headers=headers).json()
        assert served["available"] is True and served["count"] == 1
        assert served["entries"][0]["description"] == "prediction hit: the parcel arrives today"
        assert client.get("/v1/host/self", headers=headers).status_code == 404
        assert client.get("/v1/host/self/benchmark", headers=headers).status_code == 404
        from colonymind.api.routers import host
        assert host._expectations._journal is host._self_model.journal
        for path in ("/v1/host/self/benchmark/samples", "/v1/host/self/benchmark/probe",
                     "/v1/host/self/benchmark/compute"):
            assert client.post(path, json={"samples": []}, headers=headers).status_code in (404, 405)
    assert set(CALLER_ROUTES) <= _served_paths()
    assert not set(CALLER_STATE) & set(init.RETIRED_STATE)
    assert init.RETIRED_TABLES["colonymind-self-model.db"] == ("trust_stage", "trust_notices")


def test_the_retired_mcp_server_stays_gone():
    """The stdio server and its ``/mcp`` route stay gone. What replaced them is the read-first endpoint an issued
    credential reaches at ``/v1/mcp`` (docs/MCP.md; tests/test_mcp_endpoint.py)."""
    paths = _served_paths()
    assert not [path for path in paths if path == "/mcp" or path.startswith("/mcp/")]
    assert "/v1/mcp" in paths


def test_health_capabilities_never_advertise_a_pruned_subsystem():
    from colonymind.api.routers.host import supported_capabilities
    assert not set(RETIRED_CAPABILITIES) & set(supported_capabilities())


def test_the_host_router_keeps_no_handle_on_a_pruned_store():
    from colonymind.api.routers import host
    handles = ("_goals_store", "_briefings_engine", "_research_pipeline", "_search_orchestrator",
               "_connection_discoverer", "_insight_store", "_session_store", "_session_report_store",
               "_facts_store", "_pattern_store", "_agent_store", "_invite_store",
               "_assignment_engine", "_websocket_manager", "_channel_store", "_proposal_store", "_repo_mirrors",
               "_learning_feedback_store", "_connector_manager", "_situation_store",
               "_situation_reducer")
    assert [name for name in handles if hasattr(host, name) or hasattr(host, "set" + name)] == []


def test_the_autonomy_posture_names_no_connector_mode():
    source = inspect.getsource(__import__("colonymind.api.routers.host", fromlist=["x"]).get_autonomy_posture)
    assert "CONNECTORS" not in source


PART = "\0"   # a part of a string the source does not spell out: a variable, a call, a formatted field


def _rendered(node: ast.AST) -> str | None:
    """A string expression as text, each part not known from the source as PART; None if not a string."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        return "".join(value.value if isinstance(value, ast.Constant) else PART for value in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _rendered(node.left), _rendered(node.right)
        if left is None and right is None:
            return None
        return (PART if left is None else left) + (PART if right is None else right)
    return None


def _segment_matches(actual: str, pattern: str, *, covers=False) -> bool:
    """A ``{param}`` matches any segment; a segment with unknown parts matches what its known text allows."""
    if pattern.startswith("{"):
        return True
    if covers and PART in actual:
        return False  # A dynamic id might equal "media"; that does not make every id a served route.
    return re.fullmatch(".*".join(re.escape(piece) for piece in actual.split(PART)), pattern) is not None


def _matches(segments: list[str], template: tuple[tuple[str, ...], bool, bool], *, covers=False) -> bool:
    """``segments`` against a route template (its segments, whether the route itself counts, whether
    the paths below it count); a ``{param}`` segment matches any one segment, ``{name:path}`` the rest."""
    pattern, itself, below = template
    for number, part in enumerate(pattern):
        if part.endswith(":path}"):
            return True
        if number >= len(segments) or not _segment_matches(segments[number], part, covers=covers):
            return False
    return itself if len(segments) == len(pattern) else below


def _tail_matches(segments: list[str], template: tuple[tuple[str, ...], bool, bool], *, covers=False) -> bool:
    """Whether a path composed onto an unknown base can be ``template``: its first segments match the
    template's last ones, at a segment boundary that is a literal segment of the route."""
    pattern, itself, below = template
    return any(_matches(segments, (pattern[-size:], itself, below), covers=covers) for size in range(1, len(pattern) + 1)
               if not pattern[-size].startswith("{"))


def _template(path: str, *, itself: bool = True, below: bool = False) -> tuple[tuple[str, ...], bool, bool]:
    return tuple(path.strip("/").split("/")), itself, below


# A prefix covers the route and every path below it; one ending in "/" only the paths below.
RETIRED_TEMPLATES = tuple(_template(prefix, itself=not prefix.endswith("/"), below=True)
                          for prefix in RETIRED_ROUTE_PREFIXES) + tuple(_template(route) for route in RETIRED_ROUTES)


@functools.lru_cache(maxsize=1)
def _served_templates() -> tuple[tuple[tuple[str, ...], bool, bool], ...]:
    return tuple(_template(path) for path in _served_paths() if path.startswith("/"))


def calls_pruned_route(text: str) -> bool:
    """Whether a rendered string addresses a pruned route and no served one: an absolute path (alone or
    in a URL) matched whole, or a path that a base or a helper completes (it starts with "/", after an
    unknown part or not) matched against the routes' tails."""
    text = text.split("?", 1)[0].split("#", 1)[0]
    at = text.find("/v1/")
    if at >= 0:
        segments, matches = text[at:].strip("/").split("/"), _matches
    elif text.startswith("/") or text.startswith(PART + "/"):
        segments, matches = text.lstrip(PART).strip("/").split("/"), _tail_matches
    else:
        return False
    # An absolute expression must name the route's fixed segments. A dynamic
    # id can fit a parameter but cannot establish a literal such as "media".
    absolute = matches is _matches
    return (any(matches(segments, template, covers=absolute) for template in RETIRED_TEMPLATES)
            and not any(matches(segments, template, covers=absolute) for template in _served_templates()))


def pruned_route_calls(source: str) -> list[tuple[int, str]]:
    """The lines of ``source`` where a string (a literal, an f-string or a concatenation, over any number
    of lines) addresses a pruned route."""
    lines = source.splitlines()
    found = sorted({node.lineno for node in ast.walk(ast.parse(source))
                    if (text := _rendered(node)) is not None and calls_pruned_route(text)})
    return [(number, lines[number - 1].strip()) for number in found]


CALLER_SHAPES = {
    "the retired benchmark read": 'client.get("/v1/host/self/benchmark", params={"weeks": 8})',
    "a parameterized route, formatted": 'client.post(f"/v1/host/initiatives/{initiative}/claim", json={})',
    "a parameterized route, literal": 'client.post("/v1/host/initiatives/example/delegate")',
    "a base URL and a relative path": 'httpx.get(f"{base_url}/self/situation", headers=headers)',
    "a base URL added to a path": 'httpx.get(base + "/agents")',
    "a full URL": 'urllib.request.urlopen("http://127.0.0.1:7777/v1/host/briefings/latest?limit=1")',
    "a path split across lines": 'url = (sidecar_url\n       + "/v1/host/goals/"\n       + goal_id)',
    "the benchmark's sample route": 'client.post(f"{self.url}/v1/host/self/benchmark/samples", json=body)',
    "a path a helper prefixes": '_get(client, "/briefings/latest?limit=1")',
}
SERVED_SHAPES = {
    "a served route under a base": 'httpx.post(f"{base_url}/memory/search", json={})',
    "a caller route kept for its callers": 'httpx.get(f"{base_url}/v1/host/self/journal")',
    "a served parameterized route": 'client.patch(f"{sidecar}/v1/host/commitments/{commitment_id}", json={})',
    "a file path, not a route": 'open(f"{tmp_path}/notes/briefing.txt")',
    "a helper that prefixes paths": 'client.post(f"/v1/host{path}", json=data)',
    "a served route a helper completes": '_get(client, "/memory/search")',
}


@pytest.mark.parametrize("source", CALLER_SHAPES.values(), ids=CALLER_SHAPES.keys())
def test_a_call_to_a_pruned_route_is_found_in_any_shape(source):
    assert pruned_route_calls(source), source


@pytest.mark.parametrize("source", SERVED_SHAPES.values(), ids=SERVED_SHAPES.keys())
def test_a_call_to_a_served_route_is_not_mistaken_for_one(source):
    assert pruned_route_calls(source) == []


def test_no_client_calls_a_pruned_route():
    """Plugins, scripts, benchmarks, the adapter tests and the live-server suites call only served routes."""
    roots = [REPO / "plugins", REPO / "scripts", REPO / "benchmarks", REPO / "tests", REPO / "examples",
             ROOT / "scripts", ROOT / "tests" / "e2e", ROOT / "tests" / "integration", PACKAGE]
    # 1.9.0's route rules, kept as they shipped for the credentials a 1.9.0 upgrade carries, name the routes
    # 1.9.0 served; they call none.
    rules = PACKAGE / "api" / "legacy_scopes.py"
    hits = []
    for base in roots:
        for path in sorted(base.rglob("*.py")) if base.is_dir() else ():
            if path in (rules, PACKAGE / "api" / "retired_routes.py"):
                continue
            hits.extend(f"{path.relative_to(REPO)}:{number}: {text}"
                        for number, text in pruned_route_calls(path.read_text()))
    assert hits == []


def test_no_source_line_imports_a_pruned_module():
    pattern = re.compile("|".join(re.escape(module) + r"\b" for module in RETIRED_MODULES))
    hits = []
    for base in (PACKAGE, ROOT / "tests", ROOT / "scripts", REPO / "plugins", REPO / "scripts", REPO / "tests",
                 REPO / "benchmarks", REPO / "examples"):
        for path in sorted(base.rglob("*.py")) if base.is_dir() else ():
            if path.name.startswith("test_retired_modules"):
                continue   # the retired-module lists name what is gone
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line):
                    hits.append(f"{path.relative_to(REPO)}:{number}: {line.strip()}")
    assert hits == []


def test_no_environment_flag_names_a_pruned_subsystem():
    """Nor does any guide or shipped skill tell an owner to set one."""
    sources = [PACKAGE / "server.py", *sorted((REPO / "docs").rglob("*.md")),
               *sorted((REPO / "plugins").glob("*/*.md"))]
    named = [f"{path.relative_to(REPO)}: {name}" for path in sources for name in RETIRED_ENV
             if name in path.read_text()]
    assert named == []


@pytest.mark.parametrize("argv", [
    ["agent", "list"],
    ["mcp", "detect"],
    ["benchmark"],
], ids=["agent", "mcp", "benchmark"])
def test_the_cli_has_no_pruned_command(argv, monkeypatch, capsys):
    from colonymind import cli
    monkeypatch.setattr(sys, "argv", ["colonymind", *argv])
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 2 and "invalid choice" in capsys.readouterr().err


def test_the_contacts_affect_stays_the_minds_projection(tmp_path):
    """The appraisals write a non-owner speaker's valence into the contacts' affect and the social drive reads
    its trend: the store is not pruned with its unused routes. The mind gets the store the host holds, and
    the upgrade leaves it in place."""
    from types import SimpleNamespace

    from colonymind import init
    from colonymind.contacts.affect import AffectStore
    from colonymind.mind.factory import mind_kwargs
    from colonymind.turns import TurnIdempotencyLedger
    affect = AffectStore(str(tmp_path / "colonymind-affect.db"))
    host = SimpleNamespace(_affect_store=affect)
    wiring = mind_kwargs(host, config={}, store=None, state_dir=tmp_path,
                         ledger=TurnIdempotencyLedger(tmp_path / "turn-idempotency.db"), owner_id="owner")
    assert wiring["contact_affect"] is affect
    assert "colonymind-affect.db" not in init.RETIRED_STATE


def test_the_upgrade_moves_every_pruned_store_aside(tmp_path):
    """Every store the pruned code wrote, and the leftovers no code names, is in the upgrade's retired list:
    an upgrade moves them into the backup with their contents, deletes nothing, and a second pass finds none."""
    from colonymind import init
    assert set(PRUNED_STATE) <= set(init.RETIRED_STATE)
    # Whole stores move; nothing is left to drop table by table from them.
    assert "colonymind-goals.db" not in init.RETIRED_TABLES
    home = tmp_path / "home"
    home.mkdir()
    for name in PRUNED_STATE:
        if name in ("repo-mirrors", "sandbox"):
            (home / name / "inner").mkdir(parents=True)
            (home / name / "inner" / "kept.txt").write_text(name)
        elif name.endswith(".json"):
            (home / name).write_text('{"kept": "%s"}' % name)
        else:
            with sqlite3.connect(home / name) as db:
                db.execute("CREATE TABLE t (x TEXT)")
                db.execute("INSERT INTO t VALUES (?)", (name,))
    (home / "colonymind-facts.db-wal").write_bytes(b"")
    (home / "turn-idempotency.db").write_bytes(b"")
    (home / "colonymind-expectations.db").write_bytes(b"")
    assert sorted(init.retired_state_present(home)) == sorted(PRUNED_STATE)

    backup = init.backup_instance(home)
    notes = init.retire_state(home, backup)

    assert len(notes) == len(PRUNED_STATE) and init.retired_state_present(home) == []
    retired = backup / "retired"
    assert sorted(path.name for path in retired.iterdir()) == sorted(PRUNED_STATE + ("colonymind-facts.db-wal",))
    for name in PRUNED_STATE:
        if name in ("repo-mirrors", "sandbox"):
            assert (retired / name / "inner" / "kept.txt").read_text() == name
        elif name.endswith(".json"):
            assert name in (retired / name).read_text()
        else:
            with sqlite3.connect(retired / name) as db:
                assert db.execute("SELECT x FROM t").fetchall() == [(name,)]
    # The mind's stores stay where they are.
    assert (home / "turn-idempotency.db").exists() and (home / "colonymind-expectations.db").exists()
    assert init.retire_state(home, tmp_path / "again") == [] and not (tmp_path / "again").exists()


def test_the_sidecar_starts_without_building_a_pruned_store(tmp_path, monkeypatch, caplog):
    """The lifespan on an empty state directory: no pruned store is created and no warning names one."""
    import logging

    from fastapi.testclient import TestClient

    from colonymind.config import write_api_key
    from colonymind.server import create_app

    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("COLONYMIND_API_KEY", raising=False)
    write_api_key("smoke-key", tmp_path)
    caplog.set_level(logging.WARNING)
    with TestClient(create_app(), base_url="http://127.0.0.1:7777", client=("127.0.0.1", 50000)) as client:
        health = client.get("/v1/host/health", headers={"Authorization": "Bearer smoke-key"})
        assert health.status_code == 200
        assert client.get("/v1/mind/state", headers={"Authorization": "Bearer smoke-key"}).status_code == 200
    noisy = [f"{record.name}: {record.getMessage()}" for record in caplog.records
             if record.levelno >= logging.WARNING and PRUNED_WORDS.search(record.getMessage())]
    assert noisy == []
    assert [name for name in PRUNED_STATE if (tmp_path / name).exists()] == []
