"""The single transaction-bound action verification interface, implemented by the owner gate.

An API grant permits a route; it is never proof of an owner's instruction. The default
implementation deliberately authorizes no execution or receipt retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol
import sqlite3

from .validation import GoalError


@dataclass(frozen=True)
class Audience:
    owner_id: str
    person_ids: frozenset[str]
    verified_owner: bool = False
    maximum_tier: str = "Public"

    @property
    def owner_only(self) -> bool:
        return self.verified_owner and self.person_ids == frozenset({self.owner_id})

    def allows(self, sensitivity: str) -> bool:
        from .validation import TIERS
        # Owner credentials in a mixed conversation do not authorize Secret disclosure.
        if self.owner_only:
            return True
        return sensitivity != "Secret" and TIERS.index(sensitivity) <= TIERS.index(self.maximum_tier)


@dataclass(frozen=True)
class ActionContext:
    owner_id: str
    principal_id: str
    audience: Audience
    method: str
    path: str
    operation_id: str
    body: Mapping[str, Any]
    request_hash: str
    request: Any = None


@dataclass(frozen=True)
class VerifiedAction:
    owner_id: str
    principal_id: str
    evidence_id: str
    purpose: str
    explicit_tier: bool = False


class ActionVerifier(Protocol):
    def __call__(self, context: ActionContext, *, resolved_action: dict,
                 receipt: dict | None, connection: sqlite3.Connection) -> VerifiedAction:
        """Authorize before lookup, then verify exact effects inside the same transaction.

        The first call has phase='authorize', the request action and no receipt or
        resolved target data. Verify the trusted source, invocation and audience
        before permitting any lookup. The second call pins the resolved effects
        (or authenticates retained receipt evidence). Neither call alone executes.

        For replay, validate the retained evidence binding and current authenticated chat,
        audience, principal and owner, without requiring an unexpired execution capability.
        Never trust chat, sender, platform or evidence supplied as JSON.
        """
        ...


def verify_owner_goal_action(context: ActionContext, *, resolved_action: dict,
                             receipt: dict | None, connection: sqlite3.Connection) -> VerifiedAction:
    """Fail closed until the owner gate supplies its trusted verifier dependency."""
    from .validation import validate
    validate(resolved_action["action"], dict(context.body))
    if not context.body.get("source_id") or not getattr(context.request, "headers", {}).get("Owner-Goal-Action"):
        raise GoalError(422, "owner_turn_required")
    raise GoalError(403, "not_owner")


async def get_action_verifier() -> Callable:
    """FastAPI dependency: replace only from trusted server wiring, or a test double."""
    return verify_owner_goal_action
