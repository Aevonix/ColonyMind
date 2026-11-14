"""One answer for a request the sidecar's own validation refuses.

Stores and routers signal "this request cannot be honoured as sent" with a
``ValueError`` whose text names the reason (``ingress_receipt_scope_mismatch``,
"source requires session_id"). Most routes translate the ones they expect;
one that escapes used to become a bare 500 with the reason only in the log.
The handler below answers 422 with that reason instead, for every route.

A ``ValueError`` subclass defined outside this package (``json.JSONDecodeError``,
``UnicodeDecodeError``, pydantic's ``ValidationError``) is a library refusing
the sidecar's own data, not the caller's request: it is re-raised and stays a
server error.

A refused request therefore answers 422 in one of two shapes: the framework's
validation names each invalid field (``detail`` is a list), and a route's own
check, or the handler below, names its reason (``Refusal``): ``detail`` is
either that reason's code alone (a string, as many routes send it) or an
object with a ``code``, and usually a ``message``. The OpenAPI spec says both
(``declare_refusals``), where the framework alone would declare only its own.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

INVALID_REQUEST = "invalid_request"


def is_request_refusal(exc: BaseException) -> bool:
    """True for a plain ``ValueError`` or one this package defines."""
    if not isinstance(exc, ValueError):
        return False
    cls = type(exc)
    return cls is ValueError or cls.__module__.split(".", 1)[0] == __name__.split(".", 1)[0]


async def refused_request(request: Request, exc: ValueError) -> JSONResponse:
    if not is_request_refusal(exc):
        raise exc
    logger.info("%s %s refused: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=422, content={"detail": {"code": INVALID_REQUEST, "message": str(exc) or INVALID_REQUEST}})


_SCHEMAS = "#/components/schemas/"
#: A route's own refusal, and the handler's: the reason's code alone, or an object with the code and the reason in
#: words (it may carry more fields). Both are sent: many routes refuse with the code alone.
REFUSAL_SCHEMA: Dict[str, Any] = {
    "title": "Refusal",
    "description": ("A request the route refused for a reason of its own: detail is the reason's code alone, or an "
                    "object with the code and, usually, the reason in words."),
    "type": "object",
    "required": ["detail"],
    "properties": {"detail": {"anyOf": [
        {"type": "string", "description": "The reason, as a stable code (a few older refusals say it in words)."},
        {"type": "object",
         "required": ["code"],
         "properties": {"code": {"type": "string", "description": "The reason, as a stable code."},
                        "message": {"type": "string", "description": "The reason in words."}}},
    ]}},
}
REFUSAL = {"$ref": _SCHEMAS + "Refusal"}
_FIELD_ERRORS = {"$ref": _SCHEMAS + "HTTPValidationError"}
_REFUSED = ("The request was refused: detail lists each field the validation refused, or names the route's own "
            "reason (the code alone, or detail.code with detail.message).")


def _mentions(value: Any, ref: Dict[str, str]) -> bool:
    if value == ref:
        return True
    if isinstance(value, dict):
        return any(_mentions(item, ref) for item in value.values())
    if isinstance(value, list):
        return any(_mentions(item, ref) for item in value)
    return False


def declare_refusals(spec: Dict[str, Any]) -> Dict[str, Any]:
    """``spec`` (an OpenAPI document) with every 422 the framework declared (its validation's shape alone) declaring
    the ``Refusal`` shape too, and the ``Refusal`` schema beside the others wherever an answer names it. Idempotent."""
    for operations in spec.get("paths", {}).values():
        for operation in operations.values():
            answer = operation.get("responses", {}).get("422") if isinstance(operation, dict) else None
            content = (answer or {}).get("content", {}).get("application/json", {})
            if content.get("schema") == _FIELD_ERRORS:
                content["schema"] = {"anyOf": [_FIELD_ERRORS, REFUSAL]}
                answer["description"] = _REFUSED
    if _mentions(spec.get("paths", {}), REFUSAL):
        spec.setdefault("components", {}).setdefault("schemas", {})["Refusal"] = dict(REFUSAL_SCHEMA)
    return spec


def install_exception_handlers(app: FastAPI) -> FastAPI:
    """The handler above on ``app``, and its answers in ``app``'s OpenAPI spec (``declare_refusals``)."""
    app.add_exception_handler(ValueError, refused_request)
    generate = app.openapi

    def openapi() -> Dict[str, Any]:
        return declare_refusals(generate())

    app.openapi = openapi  # type: ignore[method-assign]
    return app
