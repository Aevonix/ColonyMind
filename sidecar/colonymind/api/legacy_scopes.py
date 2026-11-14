"""The route rules of the 1.9.0 keyring, kept for the credentials an upgrade carries from it.

A 1.9.0 principal reached a route when one of its scopes was the scope 1.9.0 required for that route
(``*``: every scope; ``api:access``, the unscoped rest, only with ``allow_unscoped_api``). A credential
the 1.9.0 upgrade carries keeps exactly those scopes (``scopes`` in ``colonymind.yaml``,
docs/CREDENTIALS.md), and reaches exactly the routes of this release those rules give it: the rules
below are 1.9.0's, as it shipped with its default worker and approval authority modes (``shadow``).
"""

from __future__ import annotations

import re

_SCOPE_RE = re.compile(r"^(?:\*|[a-z][a-z0-9_.-]*:[a-z][a-z0-9_.-]*)$")

# Canonical read-only work surfaces used to assemble an operator's current
# working set.  Keep this exact: collection reads belong to ``work:read``;
# neighbouring item reads and every mutation retain their existing authority.
WORK_READ_SURFACE_V1 = frozenset({
    ("GET", "/v1/host/goals"),
    ("GET", "/v1/host/projects"),
    ("GET", "/v1/host/queue/jobs/pending"),
    ("GET", "/v1/host/queue/jobs/neutral"),
})
_API_ACCESS_COMPATIBILITY = frozenset({"api:access"})
_NO_COMPATIBILITY_SCOPES: frozenset[str] = frozenset()


def is_scope(value: str) -> bool:
    """A 1.9.0 scope name: ``*`` or ``<area>:<action>``."""
    return bool(_SCOPE_RE.match(value))


def worker_authority_mode() -> str:
    """1.9.0's default worker authority mode, the one its routes were carried under."""
    return "shadow"


def compatible_scopes(method: str, path: str) -> frozenset[str]:
    """Return deliberate legacy scope aliases for one exact route.

    Compatibility never makes ``api:access`` a global parent of focused
    scopes.  Middleware also applies ``allow_unscoped_api`` before accepting
    this route-local alias.
    """

    if (method.upper(), path) in WORK_READ_SURFACE_V1 or (method.upper(), path) in {
        ("POST", "/v1/host/learning/correction"),
        ("POST", "/v1/host/response-guard/check"),
        ("GET", "/v1/host/contacts/resolve"),
    }:
        return _API_ACCESS_COMPATIBILITY
    return _NO_COMPATIBILITY_SCOPES


def required_scope(method: str, path: str) -> str:
    """Return the exact scope required for a scoped credential."""

    key = (method.upper(), path)
    if path == '/v1/host/memory/curated' or path.startswith('/v1/host/memory/curated/'):
        if key == ('GET', '/v1/host/memory/curated/search'):
            return 'memory:search'
        return 'memory:read' if method.upper() == 'GET' else 'memory:write'
    archive = session_archive_scope(method, path)
    if archive is not None:
        return archive
    if (method.upper() == 'POST' and path in {
            '/v1/host/transport/ingress/admit', '/v1/host/transport/ingress/handoff',
            '/v1/host/transport/ingress/coverage', '/v1/host/transport/ingress/native-settled'} or
            method.upper() == 'GET' and path == '/v1/host/transport/ingress/receipts'):
        return 'transport:write'
    if method.upper() == 'GET' and re.fullmatch(r'/v1/host/initiative-work(?:/[^/]+)?', path):
        return 'context:read'
    if method.upper() == 'POST' and re.fullmatch(r'/v1/host/initiative-work/[^/]+/(?:native-task|observe)', path):
        return 'turns:write'
    if method.upper() == 'GET' and re.fullmatch(r'/v1/host/temporal-followups(?:/[^/]+)?', path):
        return 'context:read'
    if method.upper() == 'POST' and re.fullmatch(r'/v1/host/temporal-followups/[^/]+/change', path):
        return 'turns:write'
    if key in WORK_READ_SURFACE_V1:
        return "work:read"
    if method.upper() == "GET" and path.startswith("/v1/host/memory/sources/assets/"):
        return "memory:read"
    # The memory inventory and a claim's history (1.12) are memory reads, as 1.9.0's memory reads were.
    if method.upper() == "GET" and (path == "/v1/host/memory/inventory"
                                    or re.fullmatch(r"/v1/host/memory/inventory/claims/[^/]+/history", path)):
        return "memory:read"
    if method.upper() == "POST" and path in {
        "/v1/host/contact-policy/standing",
        "/v1/host/contact-policy/provision",
    }:
        return "contacts:policy-write"
    if path.startswith("/v1/host/actions/"):
        if method.upper() == "PUT":
            return "actions:execute"
        if method.upper() == "GET":
            return "actions:verify"
    worker_mode = worker_authority_mode()
    if (
        method.upper() == "POST"
        and path == "/v1/host/queue/work/reconciliations"
    ):
        return "workers:attest"
    if (
        path == "/v1/host/queue/work"
        or path.startswith("/v1/host/queue/work/")
    ):
        return "work:read" if method.upper() == "GET" else "work:control"
    if (
        path.startswith("/v1/host/queue/workers/")
        and "/controls" in path
    ):
        return "workers:lifecycle"
    if (
        method.upper() == "POST"
        and path.startswith("/v1/host/queue/attestations/jobs/")
    ):
        return "workers:attest"
    if worker_mode in {"enforce", "invalid"} and method.upper() == "POST":
        if path == "/v1/host/queue/workers/register" or (
            path.startswith("/v1/host/queue/workers/")
            and path.endswith("/deregister")
        ):
            return "workers:register"
        if path == "/v1/host/queue/jobs/claim":
            return "workers:claim"
        if (
            path.startswith("/v1/host/queue/workers/")
            and path.endswith("/heartbeat")
        ) or (
            path.startswith("/v1/host/queue/jobs/")
            and path.endswith((
                "/start", "/complete", "/fail", "/heartbeat", "/release",
            ))
        ):
            return "workers:lifecycle"
    if (
        method.upper() == "GET"
        and path.startswith("/v1/host/queue/inspection/jobs/")
    ):
        return "workers:inspect"
    # Approval routes always have stable exact scopes so restricted scoped
    # principals can canary them in shadow without gaining generic api:access
    # fallback. Shadow may still accept the legacy bearer; enforce additionally
    # requires an exact scoped principal in middleware.
    if path.startswith("/v1/host/queue/approvals/"):
        if method.upper() == "GET":
            return "approvals:read"
        if method.upper() == "DELETE":
            return "approvals:manage"
        return "approvals:decide"
    if (
        method.upper() == "GET"
        and path == "/v1/host/queue/jobs/blocked"
    ):
        return "approvals:read"
    if (
        path.startswith("/v1/host/queue/jobs/")
        and path.endswith(("/approve", "/reject"))
    ):
        return "approvals:decide"
    if (
        method.upper() == "POST"
        and path.startswith("/v1/host/self/tools/")
        and path.endswith("/graduate")
    ):
        return "toolsmith:graduate"
    if (
        method.upper() == "POST"
        and path.startswith("/v1/host/self/tools/")
        and path.endswith("/shadow-compare")
    ):
        return "toolsmith:evaluate"
    if method.upper() == "POST" and path == "/v1/host/sandbox/run":
        return "sandbox:execute"
    if (
        method.upper() == "POST"
        and path.startswith("/v1/host/cognition/charters/")
        and path.endswith(("/request-activation", "/request-revocation"))
    ):
        return "charter:request"
    if (
        method.upper() == "POST"
        and path.startswith("/v1/host/cognition/charters/")
        and path.endswith("/ratify")
    ):
        return "charter:ratify"
    if path.startswith(
        "/v1/host/cognition/charter-transition-approvals"
    ):
        if method.upper() == "GET":
            return "charter:approval-read"
        if method.upper() == "POST" and path.endswith("/decision"):
            return "charter:approval-decide"
        return "api:access"
    exact = {
        ("GET", "/v1/host/contacts/resolve"): "turns:resolve-sender",
        ("GET", "/v1/host/admin/auth/status"): "auth:admin",
        ("GET", "/v1/host/contact-policy"): "contacts:policy-read",
        ("POST", "/v1/host/response-guard/check"): "response-guard:check",
        ("GET", "/v1/host/queue/contract"): "workers:contract",
        ("POST", "/v1/host/memory/read"): "memory:read",
        ("POST", "/v1/host/memory/search"): "memory:search",
        ("POST", "/v1/host/memory/voice-recall"): "memory:search",
        ("POST", "/v1/host/memory/voice-affect"): "memory:search",
        ("POST", "/v1/host/memory/recent"): "memory:read",
        ("POST", "/v1/host/memory/sources/annotations"): "memory:write",
        ("POST", "/v1/host/memory/sources/deadline"): "memory:read",
        ("GET", "/v1/host/memory/sources/erasures"): "turns:write",
        ("POST", "/v1/host/memory/sources/erasures"): "turns:write",
        ("GET", "/v1/host/memory/sources/claims/status"): "memory:read",
        ("GET", "/v1/host/preferences"): "context:read",
        ("GET", "/v1/host/self"): "context:read",
        ("POST", "/v1/host/transport/observe"): "transport:write",
        ("GET", "/v1/host/social/appraisals"): "context:read",
        ("POST", "/v1/host/social/appraisals/correct"): "turns:write",
        ("GET", "/v1/host/social/contacts"): "context:read",
        ("POST", "/v1/host/social/contacts/correct-identity"): "turns:write",
        ("POST", "/v1/host/preferences/learn"): "memory:write",
        ("POST", "/v1/host/learning/correction"): "memory:write",
        ("POST", "/v1/host/context/assemble"): "context:read",
        ("GET", "/v1/host/context/card"): "context:read",
        ("GET", "/v1/host/executions"): "context:read",
        ("GET", "/v1/host/context/projection-readiness"): "context:read",
        ("GET", "/v1/host/context/temporal"): "context:read",
        ("POST", "/v1/host/turns/sync"): "turns:write",
        ("GET", "/v1/host/events/replay"): "events:read",
        ("GET", "/v1/host/activity"): "events:read",
        ("POST", "/v1/host/activity/snapshot"): "turns:write",
        ("GET", "/v1/host/self/workspace"): "cognition:read",
        ("GET", "/v1/host/self/benchmark"): "cognition:benchmark-read",
        ("POST", "/v1/host/self/benchmark/samples"):
            "cognition:benchmark-manage",
        ("POST", "/v1/host/self/benchmark/recall-probe"):
            "cognition:benchmark-manage",
        ("POST", "/v1/host/self/benchmark/compute"):
            "cognition:benchmark-manage",
        ("GET", "/v1/host/self/experiments"):
            "cognition:experiment-read",
        ("POST", "/v1/host/self/experiments"):
            "cognition:experiment-manage",
        ("GET", "/v1/host/self/params"): "cognition:experiment-read",
        ("GET", "/v1/host/cognition/cpi"): "cognition:benchmark-read",
        ("POST", "/v1/host/cognition/cycle"):
            "cognition:benchmark-manage",
        ("GET", "/v1/host/self/situation"): "cognition:read",
        ("GET", "/v1/host/autonomy/posture"): "cognition:read",
        ("GET", "/v1/host/autonomy/schedule"): "cognition:read",
        ("GET", "/v1/host/autonomy/status"): "cognition:read",
        ("GET", "/v1/host/cognition/evidence"): "cognition:read",
        ("GET", "/v1/host/cognition/drives"): "drives:read",
        ("GET", "/v1/host/cognition/charters"): "charter:read",
        ("GET", "/v1/host/cognition/rankings"): "drives:read",
        ("GET", "/v1/host/cognition/spine"): "cognition:read",
        ("POST", "/v1/host/cognition/events"): "cognition:events-ingest",
        ("GET", "/v1/host/proposals"): "cognition:read",
        ("GET", "/v1/host/tom/p8/status"): "tom:read",
        ("GET", "/v1/host/tom/p8/deck"): "tom:read",
        ("GET", "/v1/host/tom2/report"): "tom:read",
        ("POST", "/v1/host/cognition/drives"): "drives:propose",
        ("POST", "/v1/host/cognition/drive-signals"): "drives:signal",
        ("POST", "/v1/host/cognition/charters"): "charter:propose",
    }
    if key in exact:
        return exact[key]
    if method.upper() == "PUT" and path.startswith("/v2/host/turns/"):
        return "turns:write"
    if (
        method.upper() == "POST"
        and path.startswith("/v1/host/cognition/concerns/")
        and path.endswith("/promote")
    ):
        return "cognition:manage"
    if (
        method.upper() == "POST"
        and path.startswith("/v1/host/cognition/goals/")
        and path.endswith("/promote")
    ):
        return "cognition:manage"
    if (
        method.upper() == "GET"
        and path.startswith("/v1/host/self/workspace/")
        and path.endswith("/resolution")
    ):
        return "cognition:read"
    if (
        method.upper() == "POST"
        and path.startswith("/v1/host/self/workspace/")
        and path.endswith("/resolve")
    ):
        return "cognition:manage"
    if path.startswith("/v1/host/self/experiments/"):
        if method.upper() == "GET":
            return "cognition:experiment-read"
        return "cognition:experiment-manage"
    return "api:access"


def session_archive_scope(method: str, path: str) -> str | None:
    """The worker has its own scope; session write alone cannot claim plans."""
    if ((method.upper() == 'GET' and path == '/v1/host/sessions/archive/plans')
            or (method.upper() == 'POST' and re.fullmatch(r'/v1/host/sessions/archive/plans/.+/receipt', path))):
        return 'sessions:archive-worker'
    if method.upper() == 'GET' and path == '/v1/host/sessions/archived':
        return 'sessions:read'
    if method.upper() in {'GET', 'POST'} and re.fullmatch(r'/v1/host/sessions/[^/]+/archive', path):
        return 'sessions:read' if method.upper() == 'GET' else 'sessions:write'
    return None


def scope_granted(scopes: frozenset[str], unscoped_api: bool, method: str, path: str) -> bool:
    """Whether a 1.9.0 principal with ``scopes`` (and ``allow_unscoped_api``) reached this route: its
    required scope, or a route-local alias of it, and ``api:access`` only with the unscoped allowance."""
    def has(scope: str) -> bool:
        return "*" in scopes or scope in scopes
    scope = required_scope(method, path)
    if scope == "api:access" and not unscoped_api:
        return False
    compatible = any(has(candidate) and not (candidate == "api:access" and not unscoped_api)
                     for candidate in compatible_scopes(method, path))
    return has(scope) or compatible


__all__ = ["WORK_READ_SURFACE_V1", "compatible_scopes", "is_scope", "required_scope", "scope_granted"]
