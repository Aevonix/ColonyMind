"""OS0 acceptance: fixed storage, wire, states and inert plugin registration."""

from __future__ import annotations

import ast
import asyncio
import importlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[2]
HAND = "h-0123456789ab"
TASK = "t-0123456789ab"
REF = {"platform": "test", "chat": "owner-chat", "message_id": "message-1"}
EPOCH = {"owner_pid": 123, "owner_started_at": 1000.0}
MODULES = (
    "wrap", "dispatch", "spool", "budgets", "allow", "effects", "reconcile",
    "degraded", "tools", "commands", "notices", "front", "checks", "authority", "middleware",
)
TABLE_COLUMNS = {
    "tasks": "id origin origin_ref priority kind risk needs intent_version intents resources depends_on state budget usage failures_in_row cooldown_until obligation_id owner_spoke_since created_at closed_at",
    "hands": "id task_id attempt intent_version name kind effort role contract families_used model origin_ref priority parent_session child_session executor executor_ref epoch state activity activity_since outcome evidence check branch continues escalated_from budgets usage decisions created_at started_at ended_at last_event_at",
    "reservations": "hand_id token family class expires_at state",
    "locks": "resource holder_hand since waiters",
    "effects": "hand_id key tool call_hash workdir intent_at outcome exit_code result_hash done_at",
    "obligations": "id task_id owner_ref state result_ready_at reviewed_at accepted_at communicated_at communicated_ref delivered_at delivery_ref delivery closed_reason",
    "permits": "id ask_id hand_id task_id call call_digest workdir targets owner_answer_ref granted_at expires_at replay state",
    "derivations": "derived_kind derived_id hand_id evidence_level evidence_ref state",
    "events": "hand_id seq at type payload source key",
    "questions": "id hand_id asked_at text assumption blocking state answer answered_by answered_at",
    "notices": "id budget line queued_at sent_at platform_message_id hand_id attempt",
}

# Each row supplies a valid example independently of the implementation's route list.
ROUTES = [
    ("POST", "/owner_message", REF, None),
    ("POST", "/obligations/obligation-1/review", {
        "turn_ref": "turn-2", "successful": True, "rendered": True}, None),
    ("POST", "/obligations/obligation-1/communicate", {"turn_ref": "turn-2", "silent": False,
        "destination": {"platform": "test", "chat": "owner-chat"}}, None),
    ("POST", "/plan", {"parent_session": "parent", "turn": "turn-1", "origin": "owner_turn",
                       "origin_ref": REF, "tasks": [{"goal": "Check the artifact"}]}, None),
    ("POST", f"/{HAND}/dispatched", {"delegation_id": "delegation-1", "epoch": EPOCH,
                                   "lease_token": "lease-1"}, None),
    ("POST", f"/{HAND}/release", {"delegation_id": "delegation-1", "epoch": EPOCH,
                                "lease_token": "lease-1"}, None),
    ("POST", f"/{HAND}/bind", {"executor_ref": "delegation-1", "child_session": "child",
                             "epoch": EPOCH}, None),
    ("POST", f"/{HAND}/events", {"epoch": EPOCH, "events": [], "effects": []}, None),
    ("POST", f"/{HAND}/question", {"epoch": EPOCH, "question": "Which output type?",
                                 "assumption": "Text", "blocking": True}, None),
    ("POST", "/questions/question-1/answer", {"answer": "Text", "answered_by": "front"}, None),
    ("POST", f"/{HAND}/hold", {"epoch": EPOCH, "call": {"tool": "terminal", "arguments": {"command": "true"}},
                             "workdir": "/workspace", "targets": []}, None),
    ("GET", "/permits/check", None, {"hand": HAND, "call_digest": "a" * 64, "targets_now": "[]"}),
    ("POST", f"/{HAND}/result", {"epoch": EPOCH, "intent_version": 1, "status": "done",
                               "summary_ref": "br-0123456789ab", "summary_hash": "a" * 64,
                               "tool_digest": [], "receipts": [], "families_used": []}, None),
    ("POST", f"/tasks/{TASK}/revise", {"goal": "Check the revised artifact"}, None),
    ("POST", f"/tasks/{TASK}/cancel", {"reason": "No longer needed", "owner_ref": REF}, None),
    ("POST", f"/tasks/{TASK}/accept", {"hand": HAND, "reason": "Reviewed", "for_version": 1}, None),
    ("POST", "/obligations/obligation-1/told", {"turn_ref": "turn-2"}, None),
    ("POST", "/obligations/obligation-1/delivered", {"platform_message_id": "sent-1"}, None),
    ("GET", "/digest", None, {"session": "parent", "cursor": ""}),
    ("GET", "/", None, None),
    ("GET", f"/{HAND}", None, None),
    ("GET", f"/tasks/{TASK}", None, None),
    ("POST", f"/{HAND}/stop", {"epoch": EPOCH, "reason": "Stopped"}, None),
    ("POST", f"/{HAND}/name", {"name": "Scout"}, None),
    ("GET", "/notices", None, {"cursor": ""}),
    ("POST", "/notices/notice-1/sent", {"platform_message_id": "sent-2"}, None),
    ("POST", "/reconcile", {"epoch": EPOCH, "live": [], "ledger": []}, None),
]


@pytest.mark.parametrize("name,expected", [
    ("TaskState", "open waiting done failed cancelled"),
    ("HandState", "planned reserved dispatched running checking integrating done below_minimum failed stopped superseded lost unknown_effect refused expired"),
    ("HandActivity", "queued generating executing awaiting_input awaiting_approval"),
    ("EvidenceLevel", "E0 E1 E2 E3 E4 E5"),
    ("Risk", "low normal high critical"),
    ("Kind", "coding deep research ops media plan synthesize verify integrate"),
    ("Priority", "owner own auto upkeep"),
    ("ObligationState", "open result_ready reviewed accepted communicated delivered cancelled superseded"),
])
def test_closed_enums_match_design(name, expected):
    from colonymind.hands import types
    assert {item.value for item in getattr(types, name)} == set(expected.split())


def test_effort_and_record_dataclasses():
    from dataclasses import is_dataclass
    from colonymind.hands import types
    assert {item.value for item in types.Effort} == {0, 1, 2, 3}
    for name in ("Task", "Hand", "Reservation", "Lock", "Effect", "Obligation",
                 "Event", "Question", "Notice", "Evidence", "EvidenceScope", "RouteContract"):
        assert is_dataclass(getattr(types, name)), name


@pytest.mark.parametrize("table,record", [
    ("tasks", "Task"), ("hands", "Hand"), ("reservations", "Reservation"),
    ("locks", "Lock"), ("effects", "Effect"), ("obligations", "Obligation"),
    ("events", "Event"),
    ("questions", "Question"), ("notices", "Notice"),
])
def test_record_fields_cover_the_persisted_contract(table, record):
    from dataclasses import fields
    from colonymind.hands import types
    columns = {field.metadata.get("column", field.name) for field in fields(getattr(types, record))}
    assert columns == set(TABLE_COLUMNS[table].split())


def test_contract_and_evidence_preserve_exact_scope():
    from dataclasses import fields
    from colonymind.hands import types, wire
    assert {field.name for field in fields(types.RouteContract)} == set(
        "version family modalities context_tokens max_output_tokens tool_calls json effort decode_tok_s admission".split()
    )
    evidence = {
        "level": "E3",
        "scope": {"kind": "code", "reference": "commit-1", "sha256": "a" * 64,
                  "dirty_tree_sha256": "b" * 64, "command": ["pytest", "tests/test_artifact.py"],
                  "criterion_ref": "criterion-1", "producer_families": ["family-1"]},
    }
    result = wire.ResultResponse.model_validate({"outcome": "done", "evidence": evidence, "next_step": "none"})
    restored = wire.ResultResponse.model_validate_json(result.model_dump_json())
    assert restored.evidence.level is types.EvidenceLevel.E3
    assert restored.evidence.scope.command == ["pytest", "tests/test_artifact.py"]
    assert restored.evidence.scope.reference == "commit-1"
    assert restored.evidence.scope.sha256 == "a" * 64
    assert restored.evidence.scope.dirty_tree_sha256 == "b" * 64
    assert restored.evidence.scope.criterion_ref == "criterion-1"
    assert restored.evidence.scope.producer_families == ["family-1"]
    with pytest.raises(ValidationError):
        wire.ResultResponse.model_validate({"outcome": "done", "evidence": {"level": "E6", "scope": {}},
                                            "next_step": "none"})


def test_plan_contract_supports_completion_context_and_admission_states():
    from colonymind.hands import wire
    request = wire.PlanRequest.model_validate({
        "parent_session": "parent", "turn": "turn-2", "origin": "completion", "origin_ref": "completion-1",
        "tasks": [{"goal": "Check the artifact", "continue": HAND, "kind": "verify", "effort": 2,
                   "accept": {"test_command": ["pytest", "tests/test_artifact.py"]}}],
    })
    assert request.origin == "completion"
    assert request.model_dump(by_alias=True)["tasks"][0]["continue"] == HAND
    plans = []
    for admission in ("start", "queued", "refuse"):
        plans.append({
            "task_id": TASK, "hand_id": HAND, "name": "Scout", "kind": "verify", "effort": 2,
            "risk": "high", "role": "verify", "contract": {"provider": "provider-1", "version": "v1"},
            "budgets": {"wall_seconds": 600, "tool_calls": 20, "web_searches": 6, "tokens": 200000},
            "brief": "Check the artifact against its criteria.", "admit": admission,
            "reason": None if admission == "start" else "capacity", "lease_token": "lease-1" if admission == "start" else None,
        })
    response = wire.PlanResponse.model_validate({"plans": plans})
    assert response.model_dump(mode="json") == {"plans": plans}


@pytest.mark.parametrize("model,payload", [
    ("ExecutorEpoch", {"owner_pid": 0, "owner_started_at": 1000.0}),
    ("MessageRef", {"platform": "test", "chat": "owner-chat", "message_id": " "}),
    ("PlanTask", {"goal": " "}),
    ("PlanTask", {"goal": "Check the artifact", "effort": 4}),
    ("PlanTask", {"goal": "Check the artifact", "needs": ["unknown"]}),
    ("PlanTask", {"goal": "Check the artifact", "accept": {"required_sources": 0}}),
    ("PlanRequest", {"parent_session": "parent", "turn": "turn-1", "origin": "owner_turn", "origin_ref": REF, "tasks": []}),
    ("BindRequest", {"executor_ref": "delegation-1", "child_session": "child"}),
    ("DispatchedRequest", {"delegation_id": "delegation-1", "epoch": EPOCH, "lease_token": " "}),
    ("QuestionRequest", {"epoch": EPOCH, "question": "x" * 501}),
    ("ExactCall", {"tool": "terminal", "arguments": "true"}),
    ("BoundTarget", {"kind": "file", "target": "artifact.txt"}),
    ("PermitCheckRequest", {"hand": HAND, "call_digest": "short", "targets_now": "[]"}),
    ("PermitCheckRequest", {"hand": HAND, "call_digest": "a" * 64, "targets_now": "{}"}),
    ("ReviseRequest", {}),
    ("AcceptRequest", {"hand": HAND, "reason": "Reviewed", "for_version": 0}),
    ("DeliveredRequest", {"platform_message_id": " "}),
    ("NameRequest", {"name": "two words"}),
])
def test_wire_rejects_incomplete_or_invalid_shared_values(model, payload):
    from colonymind.hands import wire
    with pytest.raises(ValidationError):
        getattr(wire, model).model_validate(payload)


def test_result_contract_keeps_prose_in_the_branch_and_requires_versioned_identity():
    from colonymind.hands import wire
    payload = {
        "epoch": EPOCH, "intent_version": 1, "status": "done", "summary_ref": "br-0123456789ab",
        "summary_hash": "a" * 64, "tool_digest": [], "receipts": [], "families_used": ["family-1"],
    }
    assert wire.ResultRequest.model_validate(payload).families_used == ["family-1"]
    for field in ("epoch", "intent_version", "summary_ref", "summary_hash"):
        invalid = {key: value for key, value in payload.items() if key != field}
        with pytest.raises(ValidationError):
            wire.ResultRequest.model_validate(invalid)
    for changes in ({"summary": "The full result text"}, {"summary_hash": "short"}, {"status": "unknown"}):
        with pytest.raises(ValidationError):
            wire.ResultRequest.model_validate(payload | changes)


def test_obligation_delivery_defaults_to_unknown_until_a_receipt():
    from colonymind.hands import wire
    payload = {"id": "obligation-1", "task_id": TASK, "owner_ref": REF,
               "state": "communicated", "communicated_ref": "turn-2"}
    obligation = wire.ObligationResponse.model_validate(payload).root
    assert obligation.delivery == "unknown"
    assert obligation.delivery_ref is None
    assert obligation.delivered_at is None
    with pytest.raises(ValidationError):
        wire.ObligationResponse.model_validate(payload | {"delivery": "assumed"})


def test_event_and_effect_wire_timestamps_match_utc_record_strings():
    from colonymind.hands import wire
    timestamp = "2026-01-01T00:00:00Z"
    event = {"seq": 1, "at": timestamp, "type": "progress", "payload": {},
             "source": "plugin", "key": f"ev:{HAND}:1"}
    effect = {"key": f"eff:{HAND}:call-1", "tool": "terminal", "call_hash": "a" * 64,
              "workdir": "/workspace", "intent_at": timestamp, "outcome": "ok", "exit_code": 0,
              "result_hash": "b" * 64, "done_at": timestamp}
    request = wire.EventsRequest.model_validate({"epoch": EPOCH, "events": [event], "effects": [effect]})
    assert request.events[0].at == timestamp
    assert request.effects[0].intent_at == timestamp
    assert request.effects[0].done_at == timestamp
    with pytest.raises(ValidationError):
        wire.EventInput.model_validate(event | {"at": 1000.0})
    for field in ("intent_at", "done_at"):
        with pytest.raises(ValidationError):
            wire.EffectInput.model_validate(effect | {field: 1000.0})


def test_evidence_scope_preserves_the_planned_probe_arguments():
    from colonymind.hands import wire
    probe = ["python", "probe.py", "--check"]
    criteria = wire.AcceptanceCriteria.model_validate({"probe": probe})
    result = wire.ResultResponse.model_validate({
        "outcome": "done", "evidence": {"level": "E3", "scope": {"kind": "state", "probe": criteria.probe,
                                                                   "sha256": "a" * 64}}, "next_step": "none",
    })
    assert result.evidence.scope.probe == probe


def test_refused_hand_response_needs_no_resolved_role():
    from colonymind.hands import wire
    payload = {"id": HAND, "task_id": TASK, "attempt": 1, "intent_version": 1, "name": "Scout",
               "kind": "research", "effort": 2, "parent_session": "parent", "executor": "hermes_delegation",
               "epoch": EPOCH, "state": "refused", "outcome": "no valid route"}
    hand = wire.HandResponse.model_validate(payload).root
    assert hand.role is None
    assert hand.contract is None


def test_contract_use_history_preserves_every_served_provider_and_version():
    from colonymind.hands import wire
    history = [
        {"provider": "provider-1", "version": "v1", "family": "family-1", "model": "model-1"},
        {"provider": "provider-2", "version": "v2", "family": "family-2", "model": "model-2"},
    ]
    hand = wire.HandResponse.model_validate({
        "id": HAND, "task_id": TASK, "attempt": 1, "intent_version": 1, "name": "Scout", "kind": "research",
        "effort": 2, "role": "research", "parent_session": "parent", "executor": "hermes_delegation", "epoch": EPOCH,
        "contract": {"provider": "provider-1", "version": "v1", "used": history},
    })
    assert hand.model_dump(mode="json")["contract"]["used"] == history
    payload = {"epoch": EPOCH, "intent_version": 1, "status": "done", "summary_ref": "br-0123456789ab",
               "summary_hash": "a" * 64, "tool_digest": [], "receipts": [], "families_used": ["family-1", "family-2"]}
    assert wire.ResultRequest.model_validate(payload).contracts_used == []
    result = wire.ResultRequest.model_validate(payload | {"contracts_used": history})
    assert result.model_dump(mode="json")["contracts_used"] == history
    for field in ("provider", "version", "family", "model"):
        incomplete = {key: value for key, value in history[0].items() if key != field}
        with pytest.raises(ValidationError):
            wire.ResultRequest.model_validate(payload | {"contracts_used": [incomplete]})


def test_migrate_twice_and_reopen_preserve_schema(tmp_path):
    from colonymind.hands.store import HandsStore

    async def exercise():
        path = tmp_path / "hands.db"
        store = HandsStore(path)
        await store.connect()
        try:
            async def schema():
                return [tuple(row) for row in await (await store.db.execute(
                    "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
                )).fetchall()]

            before = await schema()
            await store.migrate()
            await store.migrate()
            assert await schema() == before
            for table, names in TABLE_COLUMNS.items():
                rows = await (await store.db.execute(f'PRAGMA table_info("{table}")')).fetchall()
                assert {row[1] for row in rows} == set(names.split()), table
            assert (await (await store.db.execute("PRAGMA journal_mode")).fetchone())[0] == "wal"
            assert (await (await store.db.execute("PRAGMA foreign_keys")).fetchone())[0] == 1
            for candidate in tmp_path.glob("hands.db*"):
                assert stat.S_IMODE(candidate.stat().st_mode) == 0o600
        finally:
            await store.close()
        await store.connect()
        try:
            assert await schema() == before
        finally:
            await store.close()

    asyncio.run(exercise())


def test_committed_task_and_obligation_survive_migration_and_reopen(tmp_path):
    from colonymind.hands.store import HandsStore

    async def exercise():
        path = tmp_path / "hands.db"
        store = HandsStore(path)
        await store.connect()
        try:
            await store.db.execute(
                "INSERT INTO tasks (id, origin, origin_ref, kind, created_at) VALUES (?, ?, ?, ?, ?)",
                (TASK, "owner_turn", json.dumps(REF), "research", "2026-01-01T00:00:00Z"),
            )
            await store.db.execute(
                "INSERT INTO obligations (id, task_id, owner_ref, state, communicated_ref) VALUES (?, ?, ?, ?, ?)",
                ("obligation-1", TASK, json.dumps(REF), "communicated", json.dumps("turn-2")),
            )
            await store.db.commit()
            await store.migrate()
        finally:
            await store.close()
        reopened = HandsStore(path)
        await reopened.connect()
        try:
            task = await (await reopened.db.execute("SELECT origin_ref, kind FROM tasks WHERE id = ?", (TASK,))).fetchone()
            assert json.loads(task[0]) == REF
            assert task[1] == "research"
            obligation = await (await reopened.db.execute(
                "SELECT state, communicated_ref, delivery, delivery_ref FROM obligations WHERE task_id = ?", (TASK,)
            )).fetchone()
            assert tuple(obligation) == ("communicated", json.dumps("turn-2"), "unknown", None)
        finally:
            await reopened.close()

    asyncio.run(exercise())


def test_store_retains_refused_hand_without_a_resolved_role(tmp_path):
    from colonymind.hands.store import HandsStore

    async def exercise():
        store = HandsStore(tmp_path / "hands.db")
        await store.connect()
        try:
            await store.db.execute(
                "INSERT INTO tasks (id, origin, origin_ref, kind, created_at) VALUES (?, ?, ?, ?, ?)",
                (TASK, "owner_turn", json.dumps(REF), "research", "2026-01-01T00:00:00Z"),
            )
            await store.db.execute(
                "INSERT INTO hands (id, task_id, attempt, intent_version, name, kind, effort, parent_session, "
                "executor, epoch, state, outcome, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (HAND, TASK, 1, 1, "Scout", "research", 2, "parent", "hermes_delegation", json.dumps(EPOCH),
                 "refused", "no valid route", "2026-01-01T00:00:00Z"),
            )
            await store.db.commit()
            row = await (await store.db.execute("SELECT role, state, outcome FROM hands WHERE id = ?", (HAND,))).fetchone()
            assert tuple(row) == (None, "refused", "no valid route")
        finally:
            await store.close()

    asyncio.run(exercise())


def _client(authority=None):
    from colonymind.api.routers.hands import router
    app = FastAPI()
    if authority is not None:
        @app.middleware("http")
        async def authenticated(request, call_next):
            request.state.colonymind_authority = authority
            return await call_next(request)
    app.include_router(router)
    return TestClient(app)


def test_removed_park_route_is_unavailable():
    from colonymind.api.auth import key_authority
    with _client(key_authority()) as client:
        response = client.post(f"/v1/hands/{HAND}/park", json={},
                               headers={"Idempotency-Key": "seam-test"})
        assert response.status_code in {404, 405}


@pytest.mark.parametrize("method,path,body,params", [
    row for row in ROUTES if row[0] == "POST" or row[1] == "/permits/check"
])
def test_every_write_requires_a_nonempty_idempotency_key(method, path, body, params):
    from colonymind.api.auth import key_authority
    with _client(key_authority()) as client:
        for headers in ({}, {"Idempotency-Key": ""}, {"Idempotency-Key": "  "}):
            response = client.request(method, "/v1/hands" + path, json=body, params=params, headers=headers)
            assert response.status_code == 422


@pytest.mark.parametrize("method,path,body,params", ROUTES)
def test_routes_refuse_anonymous_and_nonowner_credentials(method, path, body, params):
    from dataclasses import replace
    from colonymind.api.auth import key_authority
    for authority in (None, replace(key_authority(), owner_platforms=frozenset()),
                      replace(key_authority(), principal_id="credential:guest")):
        with _client(authority) as client:
            response = client.request(method, "/v1/hands" + path, json=body, params=params,
                                      headers={"Idempotency-Key": "seam-test"})
            assert response.status_code == 403, response.text


def test_every_route_has_a_response_model_and_write_header():
    from colonymind.api.routers.hands import router
    assert sum(len(route.methods) for route in router.routes) == len(ROUTES)
    with _client() as client:
        spec = client.get("/openapi.json").json()
    for route in router.routes:
        assert route.response_model is not None, route.path
        for method in route.methods:
            operation = spec["paths"][route.path][method.lower()]
            assert "501" not in operation["responses"]
            if method == "POST" or route.path == "/v1/hands/permits/check":
                assert any(p["name"] == "Idempotency-Key" and p["required"]
                           for p in operation["parameters"])


def test_server_factory_includes_the_hands_router():
    # Inspect the factory without starting unrelated services or loading instance configuration.
    tree = ast.parse((ROOT / "sidecar" / "colonymind" / "server.py").read_text())
    factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "create_app")
    router_expressions = []
    for node in ast.walk(factory):
        if isinstance(node, ast.ImportFrom) and node.module == "colonymind.api.routers":
            router_expressions.extend(f"{alias.asname or alias.name}.router" for alias in node.names if alias.name == "hands")
        elif isinstance(node, ast.ImportFrom) and node.module == "colonymind.api.routers.hands":
            router_expressions.extend(alias.asname or alias.name for alias in node.names if alias.name == "router")
    assert router_expressions, "create_app must import the hands router"
    includes = [node for node in ast.walk(factory) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == "include_router"
                and node.args and ast.unparse(node.args[0]) in router_expressions]
    assert len(includes) == 1


def _load_plugin(monkeypatch, name="seam_hands_plugin"):
    path = ROOT / "plugins" / "colonymind-hands"
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py",
                                                submodule_search_locations=[str(path)])
    plugin = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, plugin)
    spec.loader.exec_module(plugin)
    return plugin


def test_plugin_does_not_offer_park(monkeypatch):
    plugin = _load_plugin(monkeypatch)
    tools = importlib.import_module(f"{plugin.__name__}.tools")
    commands = importlib.import_module(f"{plugin.__name__}.commands")
    assert "park" not in tools.ACTIONS
    assert "park" not in tools.HAND_SCHEMA["parameters"]["properties"]["action"]["enum"]
    assert "park" not in commands.USAGE
    assert commands.run("park Scout", runtime=None, session_id="owner") == commands.USAGE


def test_plugin_registers_every_module_in_its_fixed_order(monkeypatch):
    # OS0 fixed the list; OC2 made the modules do work, so each registration is observed, not run here.
    plugin = _load_plugin(monkeypatch)
    assert plugin.MODULES == MODULES
    ctx = object()
    called = []
    for name in MODULES:
        module = importlib.import_module(f"{plugin.__name__}.{name}")
        monkeypatch.setattr(module, "register", lambda context, name=name: called.append((name, context)))
    plugin.register(ctx)
    assert called == [(name, ctx) for name in MODULES]


@pytest.mark.parametrize("register_first", [False, True])
def test_resolver_registration_survives_plugin_load_order(monkeypatch, register_first):
    plugin = _load_plugin(monkeypatch, f"seam_hands_resolver_{register_first}")
    assert plugin.get_route_resolver() is None
    # Without a registered resolver the route is the host's native delegation policy.
    native = types.ModuleType("tools.delegate_tool_config")
    native._load_config = lambda: {"provider": "native-route", "model": "native-model"}
    monkeypatch.setitem(sys.modules, "tools", types.ModuleType("tools"))
    monkeypatch.setitem(sys.modules, "tools.delegate_tool_config", native)
    assert plugin.resolve("research", 2) == {"provider": "native-route", "model": "native-model"}
    for name in MODULES:
        monkeypatch.setattr(importlib.import_module(f"{plugin.__name__}.{name}"), "register", lambda context: None)
    calls = []

    def resolver(role, effort):
        calls.append((role, effort))
        return {"provider": "provider-1", "model": "model-1"}

    ctx = object()
    if register_first:
        plugin.register(ctx)
    plugin.register_route_resolver(resolver)
    plugin.register(ctx)
    assert plugin.get_route_resolver() is resolver
    assert calls == []
    assert plugin.resolve("research", 2) == {"provider": "provider-1", "model": "model-1"}
    assert calls == [("research", 2)]
    with pytest.raises(TypeError):
        plugin.register_route_resolver(None)
    assert plugin.get_route_resolver() is resolver


def test_plugin_manifest_loads_the_public_entry_point():
    import yaml
    path = ROOT / "plugins" / "colonymind-hands"
    manifest = yaml.safe_load((path / "plugin.yaml").read_text())
    assert manifest["name"] == "colonymind-hands"
    assert manifest["entry_point"] == "__init__.py"
    assert (path / manifest["entry_point"]).is_file()


def test_generic_launch_defaults_and_registration():
    from colonymind.config import DEFAULTS
    from colonymind.reports.registry import FACULTIES
    assert DEFAULTS["hands"]["max_running"] == 1
    assert DEFAULTS["hands"]["platforms"] == []
    assert DEFAULTS["mind"]["faculties"]["hands"] is True
    assert next(f for f in FACULTIES if f.key == "hands").setting == "mind.faculties.hands"


def test_budget_cap_and_deadline_defaults_preserve_the_launch_contract():
    from colonymind.config import DEFAULTS
    from colonymind.hands.types import Kind
    hands = DEFAULTS["hands"]
    # Numeric expectations are the design's launch assumptions, not measured guarantees.
    attempt = hands["budgets"]["attempt"]
    assert set(attempt) == {kind.value for kind in Kind}
    expected = {
        "coding": (1800, 100, 6, 1000000), "deep": (1200, 20, 6, 300000),
        "research": (1200, 40, 12, 400000), "ops": (900, 60, 4, 300000),
        "media": (600, 20, 4, 200000), "plan": (600, 20, 6, 200000),
        "synthesize": (600, 20, 6, 200000), "verify": (600, 20, 6, 200000),
        "integrate": (600, 20, 6, 200000),
    }
    for kind, values in expected.items():
        assert tuple(attempt[kind][key] for key in ("wall_seconds", "tool_calls", "web_searches", "tokens")) == values
    assert hands["budgets"]["task"] == {"token_multiplier": 3, "wall_seconds": 10800, "max_hands": 6}
    assert hands["budgets"]["day"]["max_hands"] == 40
    assert hands["budgets"]["day"]["tokens_per_family"] == {}
    assert hands["budgets"]["effort_3_multiplier"] == 2
    assert hands["lease_seconds"] == 90
    assert hands["queue"]["deadline_seconds"] == {"owner": 900, "own": 1800, "auto": 3600, "upkeep": 3600}
    assert hands["locks"]["max_waiters"] == 2
    assert hands["notices"]["routine_per_day"] == 30
    assert hands["notices"]["actionable_per_day"] == 20
    assert hands["health"]["family_caps"] == {}
    assert hands["contracts"] == {}
    assert hands["matrix"] == {}
    assert not {"front", "review", "triage", "direct_line"}.intersection(hands)


# These numeric defaults are the design's assumptions for later unit implementations.
@pytest.mark.parametrize("path,expected", [
    ("dispatch_start_seconds", 120),
    ("open_tasks", {"own": 6, "mind": 4}),
    ("checks.timeout_seconds", 300),
    ("retention", {"events_days": 36500, "rows_days": 36500}),
    ("brief", {"max_tokens": 6000}),
    ("digest.max_chars", 900),
    ("obligations.overdue_seconds", 3153600000),
    ("cascade", {"max_per_task": 1000000, "max_per_hour": 1000000}),
])
def test_remaining_deadlines_and_capacity_defaults_are_frozen(path, expected):
    from colonymind.config import DEFAULTS
    actual = DEFAULTS["hands"]
    for component in path.split("."):
        actual = actual[component]
    assert actual == expected


def test_cli_stub_is_registered(capsys, monkeypatch):
    from colonymind import cli
    monkeypatch.setattr(sys, "argv", ["colonymind", "hands", "--help"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 0
    assert "hands" in capsys.readouterr().out


def test_a_task_keeps_a_superseded_obligation_beside_its_current_one(tmp_path):
    from colonymind.hands.store import HandsStore

    async def exercise():
        store = HandsStore(tmp_path / "hands.db")
        await store.connect()
        try:
            await store.db.execute(
                "INSERT INTO tasks (id, origin, origin_ref, kind, created_at) VALUES (?, ?, ?, ?, ?)",
                (TASK, "owner_turn", json.dumps(REF), "research", "2026-01-01T00:00:00Z"),
            )
            for oid, state in (("obligation-1", "superseded"), ("obligation-2", "open")):
                await store.db.execute("INSERT INTO obligations (id, task_id, owner_ref, state) VALUES (?, ?, ?, ?)",
                                       (oid, TASK, json.dumps(REF), state))
            await store.db.commit()
            rows = await (await store.db.execute(
                "SELECT id, state FROM obligations WHERE task_id = ? ORDER BY id", (TASK,))).fetchall()
            assert [tuple(row) for row in rows] == [("obligation-1", "superseded"), ("obligation-2", "open")]
        finally:
            await store.close()

    asyncio.run(exercise())


def test_an_owner_stop_needs_no_executor_epoch(tmp_path, monkeypatch):
    import httpx
    from colonymind.api.auth import key_authority
    from colonymind.hands import controller, wire
    assert wire.StopRequest.model_validate({"reason": "Stopped"}).epoch is None
    assert wire.StopRequest.model_validate({"reason": "Stopped", "epoch": EPOCH}).epoch.owner_pid == 123

    async def exercise():
        runtime = controller.Controller(tmp_path / "hands.db")
        await runtime.connect()

        async def current():
            return runtime
        monkeypatch.setattr(controller, "get_controller", current)
        app = FastAPI()

        @app.middleware("http")
        async def owner(request, call_next):
            request.state.colonymind_authority = key_authority()
            return await call_next(request)
        app.include_router(__import__("colonymind.api.routers.hands", fromlist=["router"]).router)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(f"/v1/hands/{HAND}/stop", json={"reason": "Stopped"},
                                             headers={"Idempotency-Key": "seam-test"})
        finally:
            await runtime.close()
        # The body without an epoch is valid; the controller answers for the unknown hand.
        assert response.status_code == 404, response.text
        assert response.json()["detail"] == "unknown_hands"

    asyncio.run(exercise())


def test_the_faculty_report_and_the_paired_suite_know_the_hands_faculty():
    from colonymind.qualification import native_memory_worker, paired
    from colonymind.reports.faculties import READERS
    from colonymind.reports.registry import BY_KEY
    assert "hands" in READERS
    assert BY_KEY["hands"].arm in paired.PROFILES
    assert "hands" in native_memory_worker.MIND_FACULTIES
