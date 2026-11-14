"""The public API: the stable subset of the sidecar's routes, its spec, and the release asset that carries it.

A route joins the subset by the ``public`` tag in its decorator. The exported spec holds exactly the tagged routes;
the routes a body (the Hermes adapter) drives stay out; and the subset's contract (the operations, their parameters
and the fields of every schema they use) is pinned in ``fixtures/public-api-contract.json``, so a change to any of
them fails here until someone reviews it and writes the new contract.
"""

from __future__ import annotations

import base64
import copy
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

SIDECAR = Path(__file__).resolve().parents[1]
ROOT = SIDECAR.parent
CONTRACT = SIDECAR / "tests" / "fixtures" / "public-api-contract.json"
API_DOC = ROOT / "docs" / "API.md"
RELEASE = ROOT / ".github" / "workflows" / "release.yml"
REVIEW = ("review the change, then write the new contract: cd sidecar && "
          "python -m colonymind.api.public_spec --contract tests/fixtures/public-api-contract.json")

# The stable subset, by what a caller uses it for (docs/API.md).
PUBLIC_ROUTES = {
    # Capture: a turn, and the variants that make an older sidecar refuse media it would drop.
    ("POST", "/v1/host/turns/sync"),
    ("PUT", "/v2/host/turns/{turn_id}"),
    ("PUT", "/v2/host/turns/source-media/audio/{turn_id}"),
    ("PUT", "/v2/host/turns/source-media/video/{turn_id}"),
    ("PUT", "/v2/host/turns/source-media/document/{turn_id}"),
    # A conversation a wearable or room microphone heard, and what memory holds of it.
    ("PUT", "/v2/host/conversations/observed/{conversation_id}"),
    ("GET", "/v2/host/conversations/observed/{conversation_id}"),
    # Verified work, under the owner instance credential.
    ("GET", "/v1/host/work"),
    ("GET", "/v1/host/work/{id}"),
    ("GET", "/v1/host/work/summary"),
    ("POST", "/v1/host/work/events"),
    # Context and recall.
    ("POST", "/v1/host/context/assemble"),
    ("POST", "/v1/host/memory/search"),
    # Curated overrides and tombstones are independent of historical source erasure.
    ('POST', '/v1/host/memory/curated'),
    ('GET', '/v1/host/memory/curated/search'),
    ('GET', '/v1/host/memory/curated/{memory_id}'),
    ('GET', '/v1/host/memory/curated/{memory_id}/relationships'),
    ('POST', '/v1/host/memory/curated/{memory_id}/correct'),
    ('POST', '/v1/host/memory/curated/{memory_id}/forget'),
    # Forgetting and corrections.
    ("POST", "/v1/host/memory/curated/sources/forget"),
    ("POST", "/v1/host/memory/sources/annotations"),
    # What memory holds about a person, and what a message would become.
    ("GET", "/v1/host/memory/inventory"),
    ("GET", "/v1/host/memory/inventory/claims/{claim_id}/history"),
    ("POST", "/v1/host/claims/dry-run"),
    # Commitments and contacts, read.
    ("GET", "/v1/host/commitments"),
    ("GET", "/v1/host/commitments/{commitment_id}"),
    ("GET", "/v1/host/contacts/resolve"),
    # The mind, read only.
    ("GET", "/v1/mind/state"),
    ("GET", "/v1/mind/stats"),
    ("GET", "/v1/mind/concerns"),
    ("GET", "/v1/mind/goals"),
    ("GET", "/v1/mind/owner-gate"),
    ("GET", "/v1/mind/grants"),
    ("GET", "/v1/mind/allows"),
    ("GET", "/v1/mind/narrative"),
    ("GET", "/v1/mind/drift"),
    ("GET", "/v1/mind/lessons"),
    ("GET", "/v1/mind/wishes"),
    ("GET", "/v1/mind/learning"),
    ("GET", "/v1/mind/log"),
    ("GET", "/v1/mind/log/{intention_id}"),
    ("GET", "/v1/mind/why/{intention_id}"),
    ("GET", "/v1/mind/asks"),
    ("GET", "/v1/mind/asks/stats"),
    ("GET", "/v1/mind/opinions"),
    ("GET", "/v1/mind/opinions/{opinion_id}"),
    ("GET", "/v1/mind/people"),
    ("GET", "/v1/mind/people/proposals"),
    ("GET", "/v1/mind/people/{reference}"),
    # Outside observations, under the owner's credential.
    ("GET", "/v1/mind/observations"),
    ("POST", "/v1/mind/observations/sources/{source}"),
    ("GET", "/v1/mind/observations/sources/{source}/cap"),
    ("PUT", "/v1/mind/observations/sources/{source}/cap"),
    # Health.
    ("GET", "/v1/host/health"),
}

# The body's side of the API: what the adapter pulls, reports and asks before it acts. None of it is public.
BODY_ROUTES = {
    ("GET", "/v1/mind/dispatch"),
    ("POST", "/v1/mind/dispatch/{intention_id}/bound"),
    ("POST", "/v1/mind/outcome"),
    ("GET", "/v1/mind/outbox"),
    ("POST", "/v1/mind/outbox/{intention_id}/sending"),
    ("POST", "/v1/mind/outbox/{intention_id}/sent"),
    ("POST", "/v1/mind/guard"),
    ("POST", "/v1/mind/tripwire"),
    ("POST", "/v1/mind/observations"),
    ("GET", "/v1/mind/cron-origin"),
    ("POST", "/v1/mind/cron-origin"),
    ("POST", "/v1/mind/lineage"),
    ("POST", "/v1/mind/skills/used"),
    ("POST", "/v1/mind/answers"),
    ("POST", "/v1/mind/asks/{ident}/shown"),
    ("POST", "/v1/mind/asks/{ident}/show"),
    ("POST", "/v1/instinct/decide"),
    ("POST", "/v1/instinct/outcome"),
    ("GET", "/v1/mind/blocks"),
    ("POST", "/v1/host/configure"),
}
METHODS = {"get", "put", "post", "delete", "patch", "head", "options", "trace"}


def _operations(spec):
    return {(method.upper(), path): operation for path, operations in spec["paths"].items()
            for method, operation in operations.items() if method in METHODS}


def _refs(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "$ref" and isinstance(item, str):
                yield item.rsplit("/", 1)[-1]
            else:
                yield from _refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _refs(item)


@pytest.fixture(scope="module")
def full():
    from colonymind.server import create_app
    return create_app().openapi()


@pytest.fixture(scope="module")
def public():
    from colonymind.api.public_spec import public_spec
    from colonymind.server import create_app
    return public_spec(create_app())


def test_the_exported_spec_holds_exactly_the_tagged_routes(full, public):
    from colonymind.api.public_spec import PUBLIC
    tagged = {key for key, operation in _operations(full).items() if PUBLIC in operation.get("tags", ())}
    assert set(_operations(public)) == tagged == PUBLIC_ROUTES
    # A path the subset shares with an internal route carries only its public method.
    assert set(public["paths"]["/v1/host/commitments"]) == {"get"}
    assert set(public["paths"]["/v1/mind/owner-gate"]) == {"get"}
    # The marker is how a route joins; inside the subset every route has it, so it is not repeated.
    assert all(PUBLIC not in operation.get("tags", ()) for operation in _operations(public).values())


def test_the_routes_a_body_drives_stay_out(full, public):
    served = set(_operations(full))
    assert BODY_ROUTES <= served, sorted(BODY_ROUTES - served)
    assert not BODY_ROUTES & set(_operations(public))


def test_the_minds_public_writes_only_accept_outside_observations(public):
    writes = [key for key in _operations(public) if key[1].startswith("/v1/mind/") and key[0] != "GET"]
    assert sorted(writes) == [("POST", "/v1/mind/observations/sources/{source}"),
                              ("PUT", "/v1/mind/observations/sources/{source}/cap")]


def test_the_spec_carries_every_schema_its_routes_use_and_no_other(public):
    schemas = public["components"]["schemas"]
    used = set(_refs(public["paths"]))
    queue = list(used)
    while queue:
        for name in _refs(schemas.get(queue.pop(), {})):
            if name not in used:
                used.add(name)
                queue.append(name)
    assert used <= set(schemas), sorted(used - set(schemas))
    assert set(schemas) == used, sorted(set(schemas) - used)
    assert {"MemorySearchRequest", "ContextAssembleRequest", "TurnSyncRequest"} <= used


def test_the_spec_says_how_to_authenticate_and_that_health_needs_nothing(public):
    schemes = public["components"]["securitySchemes"]
    assert schemes["bearer"] == {"type": "http", "scheme": "bearer"}
    assert schemes["apiKey"] == {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    assert public["security"] == [{"bearer": []}, {"apiKey": []}]
    operations = _operations(public)
    assert operations[("GET", "/v1/host/health")]["security"] == []
    assert all("security" not in operation for key, operation in operations.items()
               if key != ("GET", "/v1/host/health"))


def test_the_spec_names_the_release_it_describes(public):
    import colonymind
    assert public["info"]["version"] == colonymind.__version__
    assert public["info"]["title"] == "ColonyMind public API"
    assert public["openapi"].startswith("3.")


def test_the_whole_spec_names_the_release_it_describes(full):
    """The application's own version (the whole spec's, ``/openapi.json`` and ``colonymind generate-types``) is the
    package's, like the public subset's, never a number of its own."""
    import colonymind
    from colonymind.server import create_app
    assert create_app().version == colonymind.__version__
    assert full["info"]["version"] == colonymind.__version__


FIELD_ERRORS = {"$ref": "#/components/schemas/HTTPValidationError"}
REFUSAL = {"$ref": "#/components/schemas/Refusal"}


def test_a_refused_request_is_declared_in_both_shapes_it_takes(full, public):
    """A request is refused with 422 in one of two shapes: the framework's validation names each invalid field
    (``detail`` is a list), and the route's own check, or the sidecar's handler for a refused value, names its
    reason (its code alone, or ``detail.code`` with ``message``). Every operation that declares 422 declares both, in the whole spec
    and the subset alike, and a route's other refusals (capture's 409 and 503) are the same ``Refusal``."""
    for spec in (full, public):
        declared = {key: operation["responses"]["422"]["content"]["application/json"]["schema"]
                    for key, operation in _operations(spec).items() if "422" in operation["responses"]}
        assert declared and all(schema == {"anyOf": [FIELD_ERRORS, REFUSAL]} for schema in declared.values()), \
            sorted(key for key, schema in declared.items() if schema != {"anyOf": [FIELD_ERRORS, REFUSAL]})
        refusal = spec["components"]["schemas"]["Refusal"]
        shapes = refusal["properties"]["detail"]["anyOf"]
        assert refusal["required"] == ["detail"] and [shape["type"] for shape in shapes] == ["string", "object"]
        assert shapes[1]["required"] == ["code"]
    for key in CAPTURE:
        responses = _operations(public)[key]["responses"]
        for code in ("409", "503"):
            assert responses[code]["content"]["application/json"]["schema"] == REFUSAL, (key, code)


@pytest.mark.asyncio
async def test_a_refusal_answers_as_the_spec_declares(capture_app, public):
    """What a refused request actually carries is what the spec declares for it: a body the framework refuses, a
    turn id the route refuses, and a value the sidecar's handler refuses."""
    from starlette.requests import Request

    from colonymind.api.errors import refused_request
    too_long = "t" * 300
    async with AsyncClient(transport=ASGITransport(app=capture_app), base_url="http://test") as client:
        missing = await client.post("/v1/host/turns/sync", json={"context": {"session_id": "s"}})
        refused = await client.put(f"/v2/host/turns/{too_long}", json=_turn(too_long))
    assert missing.status_code == 422 and isinstance(missing.json()["detail"], list), missing.text
    assert refused.status_code == 422 and refused.json()["detail"]["code"] == "invalid_turn_id", refused.text
    operations = _operations(public)
    _answers_as_declared(public, operations[("POST", "/v1/host/turns/sync")], missing)
    _answers_as_declared(public, operations[("PUT", "/v2/host/turns/{turn_id}")], refused)
    # A value a store refuses (a plain ValueError) is answered by the sidecar's own handler, on any route.
    handled = await refused_request(Request({"type": "http", "method": "GET", "path": "/v1/host/memory/inventory",
                                             "headers": []}), ValueError("page must be a listing's own"))
    assert handled.status_code == 422
    _valid(public, REFUSAL, json.loads(handled.body))


@pytest.mark.asyncio
async def test_a_refusal_whose_detail_is_its_code_answers_as_the_spec_declares(full):
    """Many routes refuse with the reason's code alone (``{"detail": "transport_direction_status_mismatch"}``): a
    signed-in transport receipt whose direction and status disagree is one. The spec declares that shape too, so a
    client generated from it decodes the answer it gets."""
    from fastapi import FastAPI

    from onekey import AUTH, KEY
    from colonymind.api.errors import install_exception_handlers
    from colonymind.api.middleware import ApiKeyMiddleware
    from colonymind.api.routers import transport
    app = FastAPI()
    install_exception_handlers(app)
    app.add_middleware(ApiKeyMiddleware, api_key=KEY)
    app.include_router(transport.router)
    receipt = {"event_id": "e-1", "contact_id": "c-1", "channel": "sms", "direction": "in", "external_ref": "x-1",
               "receipt_ref": "r-1", "occurred_at": "2026-09-01T09:00:00+00:00", "status": "delivered"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        refused = await client.post("/v1/host/transport/observe", headers=AUTH, json=receipt)
    assert refused.status_code == 422 and refused.json() == {"detail": "transport_direction_status_mismatch"}, \
        refused.text
    _answers_as_declared(full, _operations(full)[("POST", "/v1/host/transport/observe")], refused)


def _refusals_raised():
    """Every 422 a route raises with a detail written in its source: the detail as sent, a value the source computes
    standing as a string (the routes compute codes and messages, never other shapes)."""
    import ast
    found = []
    for path in sorted((SIDECAR / "colonymind" / "api" / "routers").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "HTTPException"):
                continue
            arguments = {keyword.arg: keyword.value for keyword in node.keywords}
            status = node.args[0] if node.args else arguments.get("status_code")
            detail = node.args[1] if len(node.args) > 1 else arguments.get("detail")
            if not (isinstance(status, ast.Constant) and status.value == 422) or detail is None:
                continue

            def sent(value):
                if isinstance(value, ast.Constant):
                    return value.value
                if isinstance(value, ast.Dict):
                    return {sent(key): sent(item) for key, item in zip(value.keys, value.values)}
                return "computed"
            found.append((f"{path.name}:{node.lineno}", sent(detail)))
    return found


def test_every_refusal_a_route_raises_is_one_the_spec_declares(full):
    """Whatever shape a route's own 422 takes (its code alone, or an object with a code and words), the ``Refusal``
    every 422 declares admits it: read from every router's source, so a route that refuses in a new shape fails
    here."""
    raised = _refusals_raised()
    assert any(isinstance(detail, str) for _, detail in raised) and any(isinstance(detail, dict) for _, detail in raised)
    import jsonschema
    refusal = {"$ref": "#/components/schemas/Refusal", "components": full["components"]}
    wrong = [where for where, detail in raised
             if not jsonschema.Draft202012Validator(refusal).is_valid({"detail": detail})]
    assert wrong == [], wrong


def test_the_contract_is_the_reviewed_one(public):
    """Any change to the subset (a route, a parameter, a field, its type, default or allowed values) fails here
    until the new contract is written and reviewed with it."""
    from colonymind.api.public_spec import contract
    current = contract(public)
    reviewed = json.loads(CONTRACT.read_text(encoding="utf-8"))
    changed = sorted(key for section in ("operations", "schemas")
                     for key in set(current[section]) | set(reviewed[section])
                     if current[section].get(key) != reviewed[section].get(key))
    assert current == reviewed, f"the public API changed ({', '.join(changed)}): {REVIEW}"


def test_observed_conversation_reads_require_the_device_source(public):
    operation = public['paths']['/v2/host/conversations/observed/{conversation_id}']['get']
    source, = [parameter for parameter in operation['parameters'] if parameter['name'] == 'source']
    assert source['in'] == 'query' and source['required'] is True
    assert source['schema']['type'] == 'string'
    assert source['schema']['pattern'] == '^[a-z0-9][a-z0-9_.-]{0,31}$'


def test_the_contract_ignores_the_generators_wording():
    """What a client relies on, not how the generator phrases it: titles, descriptions, the spelling of a single
    allowed value and the framework's own validation-error schemas do not change the contract."""
    from colonymind.api.public_spec import contract

    def spec(field, extra):
        return {"paths": {"/v1/x": {"get": {"operationId": "x", "tags": ["t"], "parameters": [
            {"name": "limit", "in": "query", "required": False, "schema": {"type": "integer", "default": 20,
                                                                        "title": "Limit"}}],
            "responses": {"200": {"description": "OK", "content": {"application/json": {"schema": {
                "$ref": "#/components/schemas/X"}}}}}}}},
            "components": {"schemas": {"X": {"type": "object", "title": "X", "description": "words",
                                             "properties": {"kind": field}, "required": ["kind"]}, **extra}}}

    old = contract(spec({"enum": ["only"], "type": "string", "title": "Kind"}, {}))
    new = contract(spec({"const": "only", "type": "string", "description": "The kind."},
                        {"ValidationError": {"type": "object", "properties": {"ctx": {"type": "object"}}}}))
    assert old == new
    assert old["operations"]["GET /v1/x"]["parameters"] == {"query:limit": {"type": "integer", "required": False,
                                                                            "default": 20}}
    assert old["schemas"]["X"] == {"fields": {"kind": {"type": "enum(\"only\")", "required": True}}}
    changed = contract(spec({"enum": ["only", "other"], "type": "string"}, {}))
    assert changed != old
    nullable = contract(spec({"anyOf": [{"type": "string"}, {"type": "null"}], "default": None}, {}))
    assert nullable["schemas"]["X"]["fields"]["kind"] == {"type": "null|string", "required": True, "default": None}


# The JSON Schema keywords that narrow what a value may be. A client's previously valid request can start failing
# when any of them tightens, so each is part of the contract.
LIMITS = ("minLength", "maxLength", "pattern", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
          "multipleOf", "minItems", "maxItems", "uniqueItems", "minProperties", "maxProperties")


def _tightened(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    return value + "x"


def _limits(value, framework, at=()):
    """Every place in ``value`` where a schema narrows a value: (its location, the keyword)."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in LIMITS and isinstance(item, (bool, int, float, str)):
                yield at, key
            elif not (at == ("components", "schemas") and key in framework):
                yield from _limits(item, framework, (*at, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _limits(item, framework, (*at, index))


def _at(value, location):
    for key in location:
        value = value[key]
    return value


def test_the_contract_changes_when_a_limit_changes(public):
    """Tightening a limit breaks a client whose request was valid: a shorter search query, a smaller inventory page
    and every other bound in the subset change the contract."""
    from colonymind.api.public_spec import _FRAMEWORK_SCHEMAS, contract
    reviewed = contract(public)

    def tightened(location, key, value=None):
        spec = copy.deepcopy(public)
        schema = _at(spec, location)
        schema[key] = _tightened(schema[key]) if value is None else value
        return contract(spec)

    query = ("components", "schemas", "MemorySearchRequest", "properties", "query")
    assert _at(public, query)["maxLength"] == 4096
    assert tightened(query, "maxLength", 32) != reviewed
    [limit] = [index for index, parameter in enumerate(public["paths"]["/v1/host/memory/inventory"]["get"][
        "parameters"]) if parameter["name"] == "limit"]
    inventory = ("paths", "/v1/host/memory/inventory", "get", "parameters", limit, "schema")
    assert _at(public, inventory)["maximum"] == 200
    assert tightened(inventory, "maximum", 1) != reviewed

    places = list(_limits(public, _FRAMEWORK_SCHEMAS))
    assert len(places) > 50 and {key for _, key in places} >= {"maxLength", "minLength", "maximum", "minimum",
                                                              "pattern", "maxItems", "minItems"}
    unchanged = [(location, key) for location, key in places if tightened(location, key) == reviewed]
    assert unchanged == []


def test_the_contract_keeps_limits_wherever_they_sit():
    """A limit inside an alternative, an array's items, a map's values or an inline object, the requiredness of an
    inline object's fields and a default below a field are all in the contract."""
    from colonymind.api.public_spec import contract

    def spec(field):
        return {"paths": {}, "components": {"schemas": {"X": {"type": "object", "properties": {"f": field}}}}}

    either = {"type": ["object", "null"], "properties": {"n": {"type": "integer", "maximum": 9}}, "maxProperties": 4}
    pattern = {"type": "object", "patternProperties": {"^[a-z]+$": {"type": "integer", "minimum": 0}}}
    inline = {"type": "object", "required": ["code"], "properties": {"code": {"type": "string", "maxLength": 8},
                                                                     "note": {"type": "string", "default": ""}}}
    shapes = [
        {"anyOf": [{"type": "string", "maxLength": 64}, {"type": "null"}]},
        either,
        {"type": "array", "items": {"type": "integer", "minimum": 1}, "maxItems": 10},
        pattern,
        {"type": "object", "additionalProperties": {"type": "string", "pattern": "^[0-9a-f]{64}$"}},
        inline,
    ]
    for shape in shapes:
        before = contract(spec(shape))
        for location, key in _limits(shape, frozenset()):
            changed = copy.deepcopy(shape)
            _at(changed, location)[key] = _tightened(_at(changed, location)[key])
            assert contract(spec(changed)) != before, (shape, location, key)
    assert contract(spec({**inline, "required": []})) != contract(spec(inline))
    assert contract(spec({**inline, "properties": {**inline["properties"], "note": {"type": "string"}}})) != \
        contract(spec(inline))
    assert contract(spec({**inline, "additionalProperties": False})) != contract(spec(inline))
    assert contract(spec({**pattern, "patternProperties": {"^[a-z0-9]+$": {"type": "integer", "minimum": 0}}})) != \
        contract(spec(pattern))
    assert contract(spec(either))["schemas"]["X"]["fields"]["f"]["type"] == \
        "(null|object{n?:integer[maximum=9]})[maxProperties=4]"
    # Integral bounds read the same however the generator spells them.
    assert contract(spec({"type": "integer", "minimum": 0})) == contract(spec({"type": "integer", "minimum": 0.0}))


def test_the_contract_refuses_a_schema_keyword_it_does_not_read():
    """A keyword the contract neither pins nor knows to be wording could narrow a value unseen: writing the contract
    stops and names it, so it is read before the snapshot is trusted again."""
    from colonymind.api.public_spec import contract
    unknown = {"paths": {}, "components": {"schemas": {"X": {"type": "object", "properties": {
        "f": {"anyOf": [{"type": "string", "contentMediaType": "text/plain"}, {"type": "null"}]}}}}}}
    with pytest.raises(ValueError, match="contentMediaType"):
        contract(unknown)
    top = {"paths": {}, "components": {"schemas": {"X": {"type": "object", "properties": {}, "propertyNames": {
        "maxLength": 8}}}}}
    with pytest.raises(ValueError, match="propertyNames"):
        contract(top)


def test_the_contract_holds_each_responses_headers():
    """A response's headers are part of what a client relies on (a retry reads ``Retry-After``)."""
    from colonymind.api.public_spec import contract

    def spec(headers):
        return {"paths": {"/v1/x": {"put": {"operationId": "x", "responses": {"202": {
            "description": "Pending", "headers": headers,
            "content": {"application/json": {"schema": {"type": "object"}}}}}}}}}

    retry = {"Retry-After": {"description": "Seconds.", "required": True, "schema": {"type": "integer"}}}
    pinned = contract(spec(retry))["operations"]["PUT /v1/x"]["responses"]
    assert pinned == {"202": {"body": "object", "headers": {"Retry-After": {"type": "integer", "required": True}}}}
    assert contract(spec({}))["operations"]["PUT /v1/x"]["responses"] == {"202": {"body": "object"}}
    optional = {"Retry-After": {**retry["Retry-After"], "required": False}}
    assert contract(spec(optional)) != contract(spec(retry))
    reworded = {"Retry-After": {**retry["Retry-After"], "description": "Wait this long."}}
    assert contract(spec(reworded)) == contract(spec(retry))


# -- What capture answers, against what the spec declares it answers ------------------------------------------

CAPTURE = [("POST", "/v1/host/turns/sync"), ("PUT", "/v2/host/turns/{turn_id}"),
           ("PUT", "/v2/host/turns/source-media/document/{turn_id}")]


def _pdf() -> str:
    from pypdf import PdfWriter
    writer, output = PdfWriter(), io.BytesIO()
    writer.add_blank_page(width=100, height=100)
    writer.write(output)
    return base64.b64encode(output.getvalue()).decode()


def _turn(turn_id, text="The ferry leaves the harbour at nine.", *, document=False):
    content = text if not document else [{"type": "text", "text": text}, {
        "type": "input_document", "input_document": {"mime_type": "application/pdf", "data": _pdf()}}]
    context = {"session_id": "session-a", "contact_id": "contact-a", "channel_id": "test:thread-a"}
    if turn_id is not None:
        context["turn_id"] = turn_id
    return {"identity": {"host_id": "test-host"}, "context": context,
            "user_message": {"role": "user", "content": content},
            "assistant_message": {"role": "assistant", "content": "Understood."}}


def _valid(spec, schema, value) -> None:
    """``value`` is what ``schema`` (which may name the spec's own schemas) allows."""
    import jsonschema
    jsonschema.Draft202012Validator({**schema, "components": spec["components"]}).validate(value)


def _answers_as_declared(spec, operation, response) -> None:
    declared = operation["responses"].get(str(response.status_code))
    assert declared is not None, (response.status_code, response.text)
    for name, header in declared.get("headers", {}).items():
        value = response.headers.get(name)
        assert value is not None or not header.get("required"), (response.status_code, name)
        if value is not None:
            _valid(spec, header["schema"], int(value) if header["schema"].get("type") == "integer" else value)
    schema = declared.get("content", {}).get("application/json", {}).get("schema")
    assert schema is not None, (response.status_code, "declares no body")
    _valid(spec, schema, response.json())


@pytest.fixture
def capture_app(monkeypatch, tmp_path):
    """The capture routes on a fresh state directory, with nothing behind them but the source ledger."""
    from fastapi import FastAPI

    from colonymind.api.errors import install_exception_handlers
    from colonymind.api.routers import host
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    for name in ("_contacts_store", "_telemetry", "_reranker", "_context_recall_selector", "_comms_log"):
        monkeypatch.setattr(host, name, None)
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(host.router)
    app.include_router(host.v2_router)
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("key", CAPTURE, ids=lambda key: key[1])
async def test_capture_answers_only_as_its_spec_declares(capture_app, public, key):
    """A first turn, an identical replay, a retry while the first is still being taken in, changed content under
    the same id and an interrupted first attempt: each answer is a response the spec declares, with its headers and
    body, and every success the spec declares is one capture gives."""
    from colonymind import get_state_dir
    from colonymind.api.schemas.host import TurnSyncRequest
    from colonymind.turns import canonical_turn_digest, get_turn_idempotency_ledger
    method, template = key
    document = "source-media" in template
    ledger = get_turn_idempotency_ledger(get_state_dir())

    def reserved(turn_id, *, interrupted=False):
        body = _turn(turn_id, document=document)
        digest = canonical_turn_digest(TurnSyncRequest.model_validate(body))
        ledger.reserve(turn_id, digest)
        if interrupted:
            ledger.mark_ambiguous(turn_id, digest, RuntimeError("the first attempt stopped"))
        return body

    answers = []
    async with AsyncClient(transport=ASGITransport(app=capture_app), base_url="http://test") as client:
        async def send(expected, turn_id, body):
            path = template.replace("{turn_id}", turn_id or "")
            answers.append((expected, await client.request(method, path, json=body)))

        first = _turn("turn-first", document=document)
        await send("created", "turn-first", first)
        await send("replayed", "turn-first", first)
        await send(409, "turn-first", _turn("turn-first", "The ferry leaves at ten.", document=document))
        await send("in_progress", "turn-pending", reserved("turn-pending"))
        await send(503, "turn-interrupted", reserved("turn-interrupted", interrupted=True))
        if method == "POST":
            await send("unkeyed", None, _turn(None))

    operation = _operations(public)[key]
    for expected, response in answers:
        _answers_as_declared(public, operation, response)
        if isinstance(expected, int):
            assert response.status_code == expected, (expected, response.status_code, response.text)
        else:
            assert response.headers["Idempotency-Status"] == expected, (expected, response.text)
    given = {str(response.status_code) for _, response in answers if response.is_success}
    assert given == {code for code in operation["responses"] if code.startswith("2")}


def test_the_media_capture_variants_declare_what_the_turn_route_does(public):
    operations = _operations(public)
    turn = operations[("PUT", "/v2/host/turns/{turn_id}")]["responses"]
    for kind in ("audio", "video", "document"):
        assert operations[("PUT", f"/v2/host/turns/source-media/{kind}/{{turn_id}}")]["responses"] == turn, kind


def test_the_module_writes_the_spec_and_the_contract(tmp_path, public):
    from colonymind.api.public_spec import contract
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (str(SIDECAR), os.environ.get("PYTHONPATH"))))}
    subprocess.run([sys.executable, "-m", "colonymind.api.public_spec", str(tmp_path / "openapi.json")],
                   cwd=tmp_path, env=env, check=True, capture_output=True, timeout=300)
    assert json.loads((tmp_path / "openapi.json").read_text(encoding="utf-8")) == public
    subprocess.run([sys.executable, "-m", "colonymind.api.public_spec", "--contract", str(tmp_path / "c.json")],
                   cwd=tmp_path, env=env, check=True, capture_output=True, timeout=300)
    assert json.loads((tmp_path / "c.json").read_text(encoding="utf-8")) == contract(public)


def test_one_exporter_of_the_whole_spec():
    """The CLI's ``generate-types`` writes the whole spec; the copy beside the package that did the same is gone."""
    assert not (SIDECAR / "export_openapi.py").exists()


def test_the_api_page_lists_the_subset(public):
    text = API_DOC.read_text(encoding="utf-8")
    listed = set(re.findall(r"`(GET|POST|PUT|PATCH|DELETE) (/v\d/[^`\s]*)`", text))
    assert listed == set(_operations(public)), (sorted(set(_operations(public)) - listed),
                                                sorted(listed - set(_operations(public))))
    assert "openapi.json" in text and "CREDENTIALS.md" in text


def _job(name):
    return yaml.safe_load(RELEASE.read_text(encoding="utf-8"))["jobs"][name]


def _stand_in(directory: Path, name: str, body: str) -> None:
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)


@pytest.mark.skipif(sys.platform == "win32", reason="the release steps are bash scripts")
def test_each_release_exports_the_spec_from_the_qualified_sidecar(tmp_path):
    """The Release workflow's export step, run with a stand-in for pip: it installs the sidecar wheel
    qualification built (not the adapter's) and writes the spec of that sidecar, whatever the tag is called."""
    import colonymind
    job = _job("api-spec")
    assert job["needs"] == "qualification"
    downloads = [step["with"] for step in job["steps"] if str(step.get("uses", "")).startswith(
        "actions/download-artifact")]
    assert downloads == [{"name": "release-distributions", "path": "dist/"}]
    [upload] = [step["with"] for step in job["steps"] if str(step.get("uses", "")).startswith(
        "actions/upload-artifact")]
    assert upload["name"] == "public-api-spec" and upload["path"] == "openapi.json"
    [script] = [step["run"] for step in job["steps"] if "public_spec" in str(step.get("run", ""))]

    (tmp_path / "dist").mkdir()
    for wheel in (f"colonymind-{colonymind.__version__}-py3-none-any.whl",
                  f"colonymind_hermes-{colonymind.__version__}-py3-none-any.whl"):
        (tmp_path / "dist" / wheel).write_bytes(b"")
    stand_ins = tmp_path / "bin"
    _stand_in(stand_ins, "python", '[ "$1 $2" = "-m pip" ] && { printf \'%s\\n\' "$@" > "$PIP_CALL"; exit 0; }\n'
                                   f'exec "{sys.executable}" "$@"\n')
    env = {**os.environ, "PATH": f"{stand_ins}{os.pathsep}{os.environ['PATH']}",
           "PYTHONPATH": os.pathsep.join(filter(None, (str(SIDECAR), os.environ.get("PYTHONPATH")))),
           "PIP_CALL": str(tmp_path / "pip")}

    subprocess.run(["bash", "-e", "-c", script], cwd=tmp_path, check=True, timeout=300,
                   env={**env, "GITHUB_REF_NAME": "v9.8.7-hermes"})
    pip = (tmp_path / "pip").read_text().splitlines()
    assert pip[:3] == ["-m", "pip", "install"]
    assert f"dist/colonymind-{colonymind.__version__}-py3-none-any.whl" in pip
    assert not [arg for arg in pip if "colonymind_hermes" in arg]
    assert "requirements/release-py312.txt" in pip
    spec = json.loads((tmp_path / "openapi.json").read_text(encoding="utf-8"))
    assert spec["info"]["version"] == colonymind.__version__


@pytest.mark.skipif(sys.platform == "win32", reason="the release steps are bash scripts")
def test_the_release_page_carries_the_spec(tmp_path):
    """The Release page is made only once the spec exported, and its step attaches it (a re-run replaces it)."""
    job = _job("github-release")
    assert "api-spec" in job["needs"] and {"pypi-colonymind", "pypi-colonymind-hermes", "docker"} <= set(job["needs"])
    names = [step.get("name", "") for step in job["steps"]]
    downloads = [step["with"] for step in job["steps"] if str(step.get("uses", "")).startswith(
        "actions/download-artifact")]
    assert downloads == [{"name": "public-api-spec", "path": "api-spec/"}]
    [attach] = [step for step in job["steps"] if "public API spec" in step.get("name", "")]
    assert names.index(attach["name"]) > max(i for i, name in enumerate(names) if "CHANGELOG" in name)
    _stand_in(tmp_path / "bin", "gh", 'printf \'%s\\n\' "$@" > "$GH_CALL"\n')
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
           "GITHUB_REF_NAME": "v9.8.7", "GH_TOKEN": "unused", "GH_CALL": str(tmp_path / "call")}
    assert set(attach["env"]) == {"GH_TOKEN"}
    subprocess.run(["bash", "-e", "-c", attach["run"]], cwd=tmp_path, env=env, check=True, timeout=60)
    assert (tmp_path / "call").read_text().splitlines() == [
        "release", "upload", "v9.8.7", "api-spec/openapi.json", "--clobber"]
