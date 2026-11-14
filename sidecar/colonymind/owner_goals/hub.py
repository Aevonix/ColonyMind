"""Owner-goal authority for the owner's authenticated Companion phone relay."""
from __future__ import annotations

import hashlib

from colonymind.api.auth import owner_person_id, request_authority

from .authority import Audience, VerifiedAction, verify_owner_goal_action
from .validation import GoalError, dumps


def _companion_owner(request):
    # Only middleware's authenticated server-side credential record establishes this.
    # An all-platform agent grant does not identify the owner's phone relay.
    if request is None:
        return False
    authority = request_authority(request)
    return (authority.authenticated and not authority.anonymous
            and bool(authority.credential_id) and "companion" in authority.owner_platforms
            and "manage" in authority.access)


def bind_companion_owner_audience(request):
    """Bind the private phone audience from the authenticated credential, never wire fields."""
    if _companion_owner(request):
        owner = owner_person_id()
        request.state.owner_goal_audience = Audience(owner, frozenset({owner}), verified_owner=True)


def verify_companion_owner_action(context, *, resolved_action, receipt, connection):
    """Authorize requested writes/confirmations from the owner's Companion credential.

    Agent proposal_only suggestions stay denied until a verified
    owner-turn source exists. A source_id or Owner-Goal-Action header supplies no
    authority here. Exact resolved effects and receipt evidence are bound inside
    the store's transaction; replay still requires the current phone credential.
    """
    request = context.request
    action = resolved_action["action"]
    if not _companion_owner(request) or action not in {
        "create", "patch", "progress", "attach", "detach", "delete", "confirm",
    }:
        return verify_owner_goal_action(context, resolved_action=resolved_action,
                                        receipt=receipt, connection=connection)

    authority = request_authority(request)
    audience = getattr(request.state, "owner_goal_audience", None)
    request_hash = hashlib.sha256(dumps([context.method, context.path, dict(context.body)]).encode()).hexdigest()
    if (not connection.in_transaction or context.owner_id != owner_person_id()
            or context.principal_id != authority.principal_id
            or not isinstance(audience, Audience) or not audience.owner_only
            or audience.owner_id != context.owner_id or context.audience != audience
            or context.method != request.method or context.path != request.url.path
            or request_hash != context.request_hash
            or request.headers.getlist("idempotency-key") != [context.operation_id]):
        raise GoalError(403, "not_owner")

    purpose = "confirmation" if action == "confirm" else "requested_write"
    # Include the full resolved action (proposal/version, inherited tier, target
    # revision and artifact binding included), not just the caller's request.
    binding = ["companion-owner-action-v1", context.owner_id, context.principal_id,
               context.method, context.path, context.operation_id, request_hash,
               purpose, resolved_action]
    evidence = "companion-owner-action-v1:" + hashlib.sha256(dumps(binding).encode()).hexdigest()
    if receipt is not None:
        expected = {"owner_id": context.owner_id, "principal_id": context.principal_id,
                    "operation_id": context.operation_id, "request_hash": request_hash,
                    "evidence_id": evidence}
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise GoalError(403, "not_owner")
    return VerifiedAction(context.owner_id, context.principal_id, evidence, purpose, explicit_tier=True)


async def get_companion_action_verifier():
    return verify_companion_owner_action
