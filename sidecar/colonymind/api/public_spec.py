"""The public API: the stable subset of the sidecar's routes and its OpenAPI spec (docs/API.md).

A route joins the subset by carrying the ``PUBLIC`` tag in its decorator. Every other route the sidecar serves (the
body's pull protocol and reports, the guard, the Instinct routes, configuration, the adapter's own capture
variants) is internal and may change in any release.

``python -m colonymind.api.public_spec [PATH]`` writes the subset's spec (``openapi.json`` by default); each release
attaches it as an asset. ``--contract`` writes the subset's contract instead: the operations with their parameters,
bodies and responses, and the fields of every schema they use (types, which are required, defaults and allowed
values, the limits on every value wherever it sits, and the status codes and headers of every response), without
the generator's wording. The test suite pins the reviewed contract, so a change to the subset is reviewed before it
ships; a schema keyword the contract does not read stops it rather than pass unseen.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from colonymind.api.errors import REFUSAL, declare_refusals

#: The tag that puts a route in the public subset.
PUBLIC = "public"

_METHODS = ("get", "put", "post", "delete", "patch", "head", "options", "trace")
# Schemas the framework adds for its own request validation: their shape follows the framework's version, not ours.
_FRAMEWORK_SCHEMAS = frozenset({"HTTPValidationError", "ValidationError"})
_SECURITY_SCHEMES = {
    "bearer": {"type": "http", "scheme": "bearer"},
    "apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
}
_DESCRIPTION = ("The stable subset of the ColonyMind sidecar's HTTP API. A route in it keeps its method, path and "
                "the fields it takes and returns within a major version; see docs/API.md.")

# How a capture route answers (docs/API.md, Capture). Every answer that takes a turn in says what became of it in
# the ``Idempotency-Status`` header; a retry waits the ``Retry-After`` seconds. The route's own refusals carry a
# ``code`` in ``detail``: the ``Refusal`` every route's 422 declares too (``colonymind.api.errors``).
_RETRY_AFTER = {"description": "Seconds to wait before sending the same turn again.",
                "schema": {"type": "integer", "minimum": 0}}
_REFUSAL = {"application/json": {"schema": dict(REFUSAL)}}


def _idempotency(*outcomes: str) -> Dict[str, Any]:
    return {"Idempotency-Status": {
        "description": "What became of the turn: created (taken in now), replayed (the same turn was taken in "
                       "before; its first answer), erased (its source was erased), unkeyed (a turn without an id), "
                       "in_progress (the same turn is still being taken in).",
        "required": True, "schema": {"type": "string", "enum": sorted(outcomes)}}}


def capture_responses(model: Any, *, keyed: bool) -> Dict[Any, Dict[str, Any]]:
    """The answers of a capture route that returns ``model``, for its decorator's ``responses``. ``keyed``: the
    route takes the turn's id in its path, answers ``201`` the first time and never takes a turn without an id;
    otherwise every turn taken in is ``200``."""
    first = {201: {"model": model, "description": "The turn was taken in for the first time.",
                   "headers": _idempotency("created")}} if keyed else {}
    return {
        200: {"description": "The turn's result: an identical turn was taken in before (its first answer), or its "
                             "source was erased" + ("." if keyed else "; or the turn was taken in now."),
              "headers": _idempotency("replayed", "erased", *(() if keyed else ("created", "unkeyed")))},
        **first,
        202: {"model": model, "description": "The same turn is still being taken in (accepted is false): send it "
                                              "again after Retry-After seconds.",
              "headers": {**_idempotency("in_progress"), "Retry-After": {**_RETRY_AFTER, "required": True}}},
        409: {"description": "The turn's id is bound to different content, or the path's and the body's ids differ: "
                             "sending it again cannot succeed.", "content": _REFUSAL},
        503: {"description": "The turn cannot be taken in now: a first attempt was interrupted (inspect before "
                             "sending it again), or the sealed store is unavailable (send it again after "
                             "Retry-After seconds).",
              "headers": {"Retry-After": {**_RETRY_AFTER, "required": False}}, "content": _REFUSAL},
    }


def _open_paths() -> frozenset:
    """The paths the sidecar serves without a credential, as its middleware decides them."""
    from colonymind.api.middleware import _DEV_MODE_ALLOWED
    return frozenset(_DEV_MODE_ALLOWED)


def _refs(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "$ref" and isinstance(item, str):
                yield item.rsplit("/", 1)[-1]
            else:
                yield from _refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _refs(item)


def _used(paths: Dict[str, Any], schemas: Dict[str, Any]) -> set:
    """The schemas the operations reference, directly or through another schema."""
    used, queue = set(), list(_refs(paths))
    while queue:
        name = queue.pop()
        if name in used:
            continue
        used.add(name)
        queue += _refs(schemas.get(name, {}))
    return used


def public_spec(app: Optional[Any] = None) -> Dict[str, Any]:
    """The OpenAPI spec of the routes tagged ``PUBLIC``, with the schemas they use and how to authenticate."""
    if app is None:
        from colonymind.server import create_app
        app = create_app()
    from colonymind import __version__

    full = declare_refusals(copy.deepcopy(app.openapi()))
    open_paths = _open_paths()
    paths: Dict[str, Any] = {}
    for path, operations in full.get("paths", {}).items():
        kept = {}
        for method, operation in operations.items():
            if method not in _METHODS or PUBLIC not in operation.get("tags", ()):
                continue
            operation = copy.deepcopy(operation)
            operation["tags"] = [tag for tag in operation["tags"] if tag != PUBLIC]
            if path in open_paths:
                operation["security"] = []
            kept[method] = operation
        if kept:
            paths[path] = kept
    schemas = full.get("components", {}).get("schemas", {})
    used = _used(paths, schemas)
    return {
        "openapi": full["openapi"],
        "info": {"title": "ColonyMind public API", "version": __version__, "description": _DESCRIPTION},
        "paths": paths,
        "components": {"schemas": {name: copy.deepcopy(schema) for name, schema in schemas.items() if name in used},
                       "securitySchemes": copy.deepcopy(_SECURITY_SCHEMES)},
        "security": [{name: []} for name in _SECURITY_SCHEMES],
    }


# The keywords that narrow what a value may be: tightening any of them can refuse a request that was valid, so the
# contract pins each wherever it sits (a field, an alternative, an array's items, a map's values, a header).
_LIMITS = ("exclusiveMaximum", "exclusiveMinimum", "maxItems", "maxLength", "maxProperties", "maximum", "minItems",
           "minLength", "minProperties", "minimum", "multipleOf", "pattern", "uniqueItems")
# Every keyword the contract reads, and the ones that only describe a value to a reader. Any other keyword stops the
# contract (``_type``): it could narrow a value unseen, so it is read and placed in one of these before a snapshot
# is trusted again.
_READ = frozenset({"$ref", "type", "format", "const", "enum", "anyOf", "oneOf", "allOf", "items", "properties",
                   "required", "additionalProperties", "patternProperties", "default", *_LIMITS})
_WORDING = frozenset({"title", "description", "examples", "example", "$comment"})


def _json(value: Any) -> str:
    """A value as the contract writes it: a whole number the same however the generator spells it (``0`` or
    ``0.0``)."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return json.dumps(value, sort_keys=True)


def _type(schema: Any) -> str:
    """A schema as one compact type: a schema name, a JSON type (with its format, items, fields or values), the
    allowed values, or the alternatives joined by ``|``; then its limits (``string[maxLength=64]``) and a default
    below a field (``=0``). An inline object marks an optional field ``name?``, a map's keys by their pattern
    (``/^[a-z]+$/``) or ``*``, and ``closed`` when it allows no other key."""
    if not isinstance(schema, dict) or not schema:
        return "any"
    unknown = set(schema) - _READ - _WORDING
    if unknown:
        raise ValueError(f"the public contract does not read the schema keyword {', '.join(sorted(unknown))}: pin it "
                         "in colonymind.api.public_spec (or name it wording there), then review the contract")
    text, alternatives = _shape(schema)
    limits = [f"{key}={_json(schema[key])}" for key in _LIMITS if key in schema]
    suffix = (f"[{','.join(limits)}]" if limits else "") + (f"={_json(schema['default'])}" if "default" in schema
                                                            else "")
    return f"({text}){suffix}" if suffix and alternatives else text + suffix


def _shape(schema: Dict[str, Any]) -> tuple:
    """The type ``_type`` writes before the limits, and whether it is a choice among alternatives."""
    if "$ref" in schema:
        return schema["$ref"].rsplit("/", 1)[-1], False
    if "const" in schema or "enum" in schema:
        values = [schema["const"]] if "const" in schema else schema["enum"]
        return "enum(" + ",".join(sorted(_json(value) for value in values)) + ")", False
    for key, joint in (("anyOf", "|"), ("oneOf", "|"), ("allOf", "&")):
        if key in schema:
            return joint.join(sorted({_type(item) for item in schema[key]})), True
    kind = schema.get("type")
    if isinstance(kind, list):
        # Each JSON type once; the limits and default beside them apply to the whole choice.
        plain = {key: item for key, item in schema.items() if key not in _LIMITS and key != "default"}
        return "|".join(sorted(_type({**plain, "type": item}) for item in kind)), True
    if kind == "array":
        return f"array<{_type(schema.get('items'))}>", False
    if kind == "object" or (kind is None and {"properties", "patternProperties", "additionalProperties"} & set(schema)):
        return _object(schema), False
    if kind is None:
        return "any", False
    return (f"{kind}:{schema['format']}" if "format" in schema else str(kind)), False


def _object(schema: Dict[str, Any]) -> str:
    properties = schema.get("properties") or {}
    required = set(schema.get("required", ()))
    fields = [f"{name}{'' if name in required else '?'}:{_type(item)}" for name, item in sorted(properties.items())]
    fields += [f"{name}:any" for name in sorted(required - set(properties))]
    keys = [f"/{pattern}/:{_type(item)}" for pattern, item in sorted((schema.get("patternProperties") or {}).items())]
    extra = schema.get("additionalProperties")
    if extra is False:
        keys.append("closed")
    elif isinstance(extra, dict) and extra:
        if not fields and not keys:
            return f"object<{_type(extra)}>"
        keys.append(f"*:{_type(extra)}")
    if fields:
        return "object{" + ",".join(fields + keys) + "}"
    return f"object<{','.join(keys)}>" if keys else "object"


def _field(schema: Any, required: bool) -> Dict[str, Any]:
    schema = schema if isinstance(schema, dict) else {}
    value: Dict[str, Any] = {"type": _type({key: item for key, item in schema.items() if key != "default"}),
                             "required": required}
    if "default" in schema:
        value["default"] = schema["default"]
    return value


def _content(value: Any) -> Optional[str]:
    """The JSON body's type, or ``None`` when there is none."""
    content = (value or {}).get("content", {})
    if "application/json" in content:
        return _type(content["application/json"].get("schema"))
    return next((f"{media}:{_type(item.get('schema'))}" for media, item in sorted(content.items())), None)


def _response(value: Any) -> Dict[str, Any]:
    """A response's body type (``None`` when it has none) and the headers it sets."""
    pinned: Dict[str, Any] = {"body": _content(value)}
    headers = (value or {}).get("headers") or {}
    if headers:
        pinned["headers"] = {name: _field(header.get("schema"), bool(header.get("required")))
                             for name, header in sorted(headers.items())}
    return pinned


def contract(spec: Dict[str, Any]) -> Dict[str, Any]:
    """What a client of ``spec`` relies on, without the generator's wording (titles, descriptions, examples) and
    without the framework's own validation-error schemas."""
    operations = {}
    for path, methods in spec.get("paths", {}).items():
        for method, operation in methods.items():
            if method not in _METHODS:
                continue
            body = operation.get("requestBody")
            operations[f"{method.upper()} {path}"] = {
                "operation_id": operation.get("operationId"),
                "tags": sorted(operation.get("tags", ())),
                "parameters": {f"{item['in']}:{item['name']}": _field(item.get("schema"), bool(item.get("required")))
                               for item in operation.get("parameters", ())},
                "body": None if body is None else {"type": _content(body), "required": bool(body.get("required"))},
                "responses": {code: _response(value) for code, value in sorted(operation.get("responses", {}).items())},
                "open": operation.get("security") == [],
            }
    schemas = {}
    for name, schema in spec.get("components", {}).get("schemas", {}).items():
        if name in _FRAMEWORK_SCHEMAS:
            continue
        if "properties" in schema:
            required = set(schema.get("required", ()))
            fields = {field: _field(item, field in required) for field, item in sorted(schema["properties"].items())}
            fields.update({field: {"type": "any", "required": True} for field in sorted(required - set(fields))})
            value: Dict[str, Any] = {"fields": fields}
            rest = {key: item for key, item in schema.items() if key not in ("properties", "required")}
            if rest.get("additionalProperties") is False:
                value["closed"] = True
                del rest["additionalProperties"]
            # Anything else the object says of itself: its other keys, its limits (``_type`` reads every keyword).
            shape = _type({**rest, "type": "object"})
            if shape != "object":
                value["shape"] = shape
        else:
            value = {"type": _type(schema)}
        schemas[name] = value
    return {"operations": dict(sorted(operations.items())), "schemas": dict(sorted(schemas.items()))}


def _write(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m colonymind.api.public_spec",
                                     description="Write the public API's OpenAPI spec (docs/API.md).")
    parser.add_argument("path", nargs="?", default="openapi.json", help="where to write it (default: openapi.json)")
    parser.add_argument("--contract", action="store_true",
                        help="write the reviewed contract the test suite pins instead of the spec")
    args = parser.parse_args(argv)
    spec = public_spec()
    _write(Path(args.path), contract(spec) if args.contract else spec)
    count = sum(len(methods) for methods in spec["paths"].values())
    print(f"{'Contract' if args.contract else 'Spec'} of {count} public routes written to {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
