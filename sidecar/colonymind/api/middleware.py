"""API key authentication and request-size middleware for the ColonyMind sidecar."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from colonymind.api.auth import (
    anonymous_authority,
    answer_credential,
    answer_credential_required,
    bearer_token,
    credential_authority,
    has_owner_authority,
    key_authority,
    observe_credential,
    query_person_is_granted,
    source_not_granted,
    token_matches,
)
from colonymind.api.credentials import (ANSWER, OBSERVE, TOOL_ORIGIN, HANDS, hands_route, CredentialTable, observed_source, route_granted,
                                       route_operation)
from colonymind.api.mcp import PATH as MCP_PATH, authenticate as mcp_authenticate
from colonymind.api.owner_location import PATH as OWNER_LOCATION_PATH


# Default cap on request body size (10 MiB). Oversized uploads are rejected
# at the middleware layer before the handler buffers them. Override via
# COLONYMIND_MAX_BODY_BYTES in the environment.
_DEFAULT_MAX_BODY_BYTES = 10 * 1024 * 1024


# Endpoints reachable without a key, even in development mode (no key).
# Health + docs keep the first run smooth; everything else requires the key
# or a loopback caller.
_DEV_MODE_ALLOWED = frozenset({
    "/v1/host/health",
    "/docs",
    "/openapi.json",
    "/redoc",
})

# Routes that are never served without a key, whatever the caller: they
# accept or return credential-grade state.
_ALWAYS_AUTH_REQUIRED = frozenset({
    "/v1/host/configure",
    "/v1/host/queue/contract",
})

# Development mode's principal is granted no owner platform: like a credential that may never write
# as the owner, it reaches only the turn and context routes, and reads besides only the mind's routes
# that serve a non-owner view (they decide the owner's view from the credential: docs/CREDENTIALS.md).
_READ_METHODS = frozenset({"GET", "HEAD"})
_NON_OWNER_VIEWS = ("/v1/mind/state", "/v1/mind/status", "/v1/mind/log", "/v1/mind/why/", "/v1/mind/asks",
                    "/v1/mind/lessons", "/v1/mind/opinions", "/v1/mind/people")


def _non_owner_view(path: str) -> bool:
    return any(path == route.rstrip("/") or path.startswith(route.rstrip("/") + "/") for route in _NON_OWNER_VIEWS)


# The owner's own areas: the mind, and the sealed store of their private values (docs/SECRETS.md). A credential
# without the owner grant reaches none of them, whatever else it may reach (``manage``, a 1.9.0 scope), except
# the mind's reads that serve a non-owner view; what a request names (a lane, a viewer, ``by``) never stands
# in for the grant.
def _owner_area(path: str) -> bool:
    return path.startswith("/v1/mind/") or path == "/v1/secrets" or path.startswith("/v1/secrets/")


_DEV_MODE_MESSAGE = (
    "No API key is configured, so only loopback clients sending a loopback "
    "Host header (localhost, 127.0.0.1, ::1) are served. Write the key to "
    "api.key in the instance directory or set COLONYMIND_API_KEY to accept "
    "other clients."
)


def _is_loopback_name(value: str | None) -> bool:
    """True for ``localhost`` or a literal loopback IP (v4, v6, v4-mapped)."""
    if not value:
        return False
    if value.lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return (mapped or address).is_loopback


def is_local_request(request: Request) -> bool:
    """True when both the peer address and the Host header are loopback.

    Development mode (no key) serves this case only. Checking the Host header
    as well as the socket peer stops a browser on the same machine from being
    pointed at the API through a DNS-rebinding hostname.
    """
    client = request.client
    if client is None or not _is_loopback_name(client.host):
        return False
    # ``//host:port`` parsing strips the port and IPv6 brackets for us.
    try:
        parts = urlsplit("//" + request.headers.get("host", ""))
        hostname = parts.hostname
    except ValueError:
        return False
    if parts.username is not None or parts.password is not None:
        return False
    return _is_loopback_name(hostname)


def _dev_mode_requires_key(request: Request, path: str) -> bool:
    return (
        path in _ALWAYS_AUTH_REQUIRED
        or path == "/v1/host/queue/work"
        or path.startswith("/v1/host/queue/work/")
        or (path.startswith("/v1/host/queue/workers/") and "/controls" in path)
        or path.startswith("/v1/host/queue/inspection/")
        or path.startswith("/v1/host/queue/attestations/")
        # A sealed value leaves the sidecar only through resolve, and a release puts one back into memory.
        or path == "/v1/secrets/resolve"
        or (path.startswith("/v1/secrets/") and path.endswith("/release"))
    )


def _answer_refusal(authority) -> JSONResponse:
    error = answer_credential_required(authority)
    return JSONResponse(status_code=error.status_code, content={"detail": error.detail})


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests that carry neither the instance key nor an issued credential.

    Each credential, the instance key included, is bound to what ``credentials`` grants it
    (``colonymind.api.credentials``, read from ``colonymind.yaml`` when no table is given) and goes
    through the same checks: it reaches the routes of the operations it is granted (by default a
    credential that may never write as the owner reaches only the turn and context routes), and every
    query-string person selector is checked against its grant.

    Production wiring may supply an owner-goal audience binder, called only after
    authentication and grant checks. Without one, owner-goal audiences stay unbound.

    Without a key the API runs in development mode: only loopback requests
    (see ``is_local_request``) are served, credential-handling paths fail
    closed with 503 so an operator cannot accidentally expose them, and its
    principal, granted no owner platform, changes nothing outside the turn and
    context routes.
    """

    def __init__(self, app, api_key: str | None = None, credentials: CredentialTable | None = None,
                 owner_goal_audience_binder=None) -> None:
        super().__init__(app)
        self._api_key = (api_key or "").strip() or None
        self._credentials = credentials if credentials is not None else CredentialTable.load()
        self._owner_goal_audience_binder = owner_goal_audience_binder

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    @staticmethod
    def _route_refusal(request: Request, path: str, authority) -> str | None:
        """Why the credential may not reach this route, or ``None``: the operations in its ``access``, or, for
        a credential carried from 1.9.0, exactly the routes 1.9.0's rules gave its scopes."""
        if authority.access == frozenset({HANDS}) and authority.credential_id != "api.key":
            return None if hands_route(path) else "hands credentials reach /v1/hands/* only"
        if path == "/v1/mind/tool-origins":
            return (None if authority.authenticated and authority.credential_id == "api.key"
                    else "only the gateway adapter credential registers tool origins")
        if route_operation(request.method, path) == TOOL_ORIGIN:
            return (None if authority.authenticated and authority.credential_id != "api.key"
                    and authority.scopes is None and authority.access == frozenset({TOOL_ORIGIN})
                    and not authority.owner_platforms else "a tool-origin credential is required")
        if authority.scopes is not None:
            from colonymind.api.legacy_scopes import required_scope, scope_granted
            if scope_granted(authority.scopes, authority.unscoped_api, request.method, path):
                return None
            return (f"credential {authority.principal_id} keeps its 1.9.0 scopes, and 1.9.0 required "
                    f"{required_scope(request.method, path)} for this route")
        operation = route_operation(request.method, path)
        if operation in authority.access or (operation == OBSERVE and "manage" in authority.access):
            return None
        doing = {"write": "write turns", "read": "read context or memory",
                 "manage": "reach this route (the owner's side of the API)",
                 OBSERVE: "reach the outside observation routes"}[operation]
        return (f"credential {authority.principal_id} may not {doing}; it may "
                + (", ".join(sorted(authority.access)) or "do nothing"))

    @staticmethod
    def _query_scope_error(request: Request, authority):
        # Several list endpoints use query-string person selectors. Validate
        # the common names globally so a new handler cannot accidentally turn
        # a body-bound person back into caller authority in development mode.
        for field in ("person_id", "contact_id", "viewer_person_id"):
            for value in request.query_params.getlist(field):
                if value and not query_person_is_granted(authority, value):
                    return JSONResponse(
                        status_code=403,
                        content={
                            "detail": {
                                "code": "person_scope_not_granted",
                                "message": f"query parameter {field} exceeds caller authority",
                            }
                        },
                    )
        return None

    async def dispatch(self, request: Request, call_next):
        from colonymind.api.credentials import SECRET_REVEAL_PATH
        from colonymind.api.legacy_scopes import session_archive_scope

        reveal = request.url.path.rstrip("/") == SECRET_REVEAL_PATH
        approval_history = (request.url.path == "/v1/host/queue/approvals/history"
                            or request.url.path.startswith("/v1/host/queue/approvals/history/"))
        archive = session_archive_scope(request.method, request.url.path) is not None
        try:
            response = await self._dispatch(request, call_next)
        except Exception:
            if not reveal:
                raise
            # Keep unexpected failures private too, including in debug mode.
            response = JSONResponse(status_code=500, content={"detail": {"code": "secret_reveal_unavailable"}})
        if reveal or approval_history:
            # Includes authentication refusals, validation errors, redirects and server errors.
            response.headers["Cache-Control"] = "no-store"
        if archive:
            # These host routes use the hub's existing refusal vocabulary,
            # including refusals issued before the handler by authentication.
            if response.status_code in {401, 403}:
                response = JSONResponse(status_code=response.status_code, content={"error": "unauthorized"})
            response.headers["Cache-Control"] = "no-store"
        return response

    async def _dispatch(self, request: Request, call_next):
        path = request.url.path
        from colonymind.api.retired_routes import RETIRED_ROUTE_CODES
        if code := RETIRED_ROUTE_CODES.get(path.rstrip('/')):
            return JSONResponse(status_code=410, content={'detail': {'code': code}})
        request.state.colonymind_authority = anonymous_authority()
        request.state.colonymind_auth_configuration = {"key_configured": self.configured}
        # This finite grant also refuses public/authentication-special routes.
        supplied = self._credentials.match(bearer_token(request))
        if supplied and HANDS in supplied.access and not hands_route(path):
            return JSONResponse(status_code=403, content={"detail": "hands credentials reach /v1/hands/* only"})

        if path == MCP_PATH:
            # The MCP endpoint takes an issued credential and nothing else: never development mode, never the
            # instance key; its tools act for whom that credential binds (docs/MCP.md).
            refusal = mcp_authenticate(request, api_key=self._api_key, credentials=self._credentials)
            return refusal if refusal is not None else await call_next(request)

        if path in _DEV_MODE_ALLOWED:
            return await call_next(request)

        if not self._api_key:
            if not is_local_request(request):
                return JSONResponse(
                    status_code=403,
                    content={"detail": {"code": "dev_mode_not_local", "message": _DEV_MODE_MESSAGE}},
                )
            if _dev_mode_requires_key(request, path):
                return JSONResponse(
                    status_code=503,
                    content={
                        "detail": (
                            "Write the API key to api.key in the instance directory or set "
                            "COLONYMIND_API_KEY in the sidecar environment to enable this endpoint."
                        )
                    },
                )
            if route_operation(request.method, path) == ANSWER:
                return _answer_refusal(request.state.colonymind_authority)
            if not route_granted(request.method, path) and not (
                    request.method.upper() in _READ_METHODS and _non_owner_view(path)):
                return JSONResponse(
                    status_code=403,
                    content={"detail": {
                        "code": "route_not_granted",
                        "message": ("development mode (no key) is never the owner: it reaches the turn and "
                                    "context routes and the mind's non-owner views only; write api.key in "
                                    "the instance directory for the owner's side of the API"),
                    }},
                )
            query_error = self._query_scope_error(request, request.state.colonymind_authority)
            if query_error is not None:
                return query_error
            return await call_next(request)

        # Both header styles are in active use: the plugin sends
        # ``Authorization: Bearer``, scripts may send ``X-API-Key``.
        token = bearer_token(request)
        if token_matches(token, self._api_key):
            authority = key_authority(self._credentials.instance)
        else:
            grant = self._credentials.match(token)
            if grant is None:
                return JSONResponse(
                    status_code=(403 if request.method == "POST" and path.rstrip("/") == OWNER_LOCATION_PATH
                                 else 401),
                    content={"detail": "Invalid or missing API key"},
                )
            authority = key_authority(grant) if grant.instance else credential_authority(grant)
        request.state.colonymind_authority = authority
        if route_operation(request.method, path) == ANSWER:
            from colonymind.api.routers.mind import get_mind
            from colonymind.mind.owner_actions import may_read
            if may_read(request, get_mind()):
                query_error = self._query_scope_error(request, authority)
                return query_error if query_error is not None else await call_next(request)
            # The hands permit asks, read and answered by id: only an ``answer`` credential, the owner channel
            # relay's own (never the instance key, whatever it is granted; docs/hands-permit-answer-contract.md).
            if not answer_credential(authority):
                return _answer_refusal(authority)
            query_error = self._query_scope_error(request, authority)
            return query_error if query_error is not None else await call_next(request)
        # The same enforcement for the instance key as for an issued credential: its grant, not the
        # secret it presents, decides what it reaches (write: the turn routes; read: context and memory;
        # manage: the rest, the owner's side of the API).
        refusal = self._route_refusal(request, path, authority)
        if refusal is not None:
            return JSONResponse(status_code=403, content={"detail": {"code": "route_not_granted",
                                                                     "message": refusal}})
        observing = route_operation(request.method, path) == OBSERVE
        origin_contract = (route_operation(request.method, path) == TOOL_ORIGIN
                           or path == "/v1/mind/tool-origins")
        if ((_owner_area(path) or observing) and not has_owner_authority(authority) and not origin_contract
                and not (request.method.upper() in _READ_METHODS and _non_owner_view(path))):
            # The mind and the sealed store are the owner's: a credential without the owner grant, even one
            # that reaches the rest of the API, gets only the mind's reads that have a non-owner view.
            return JSONResponse(
                status_code=403,
                content={"detail": {
                    "code": "not_owner",
                    "message": (f"credential {authority.principal_id} is not granted the owner: of the mind it "
                                "reads only the non-owner views, and the sealed store is the owner's"),
                }},
            )
        if observing and observe_credential(authority):
            # An ``observe`` credential reaches only the sources it lists; a route that names its source in the
            # body asks ``observe_source_granted`` itself.
            source = observed_source(request.method, path)
            if source is not None and source not in authority.observe_sources:
                error = source_not_granted(authority, source)
                return JSONResponse(status_code=error.status_code, content={"detail": error.detail})
        query_error = self._query_scope_error(request, authority)
        if query_error is not None:
            return query_error
        if self._owner_goal_audience_binder is not None:
            self._owner_goal_audience_binder(request)
        return await call_next(request)


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests that declare a body larger than ``max_bytes``.

    Short-circuits before the handler reads the payload, so oversized uploads
    cannot be used to exhaust memory. Requests without a ``Content-Length``
    header (e.g. chunked transfer encoding) are allowed to pass; FastAPI's
    own buffer limits still apply downstream.
    """

    def __init__(self, app, max_bytes: int = _DEFAULT_MAX_BODY_BYTES) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                length = int(cl)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length header"},
                )
            if length > self._max_bytes:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": (
                            f"Request body exceeds limit "
                            f"({length} > {self._max_bytes} bytes)"
                        )
                    },
                )
        return await call_next(request)
