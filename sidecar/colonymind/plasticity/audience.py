"""Audience: who a learned stack may serve.

A row's audience is who may see what it was learned from: ``owner`` (the owner alone), ``contact`` (the owner and
the contacts it concerns) or ``guest`` (anyone). A stack's audience is the narrowest audience of its training data,
so a stack that learned from one owner row is an owner stack. A requester may be served a stack only when it may
see everything the stack learned from:

- an owner stack serves owner turns only;
- a contact stack serves owner turns only too: the core cannot tell which contact a stack's rows concern, so no
  contact is served what another contact's rows taught;
- a guest stack serves anyone.

Everyone else gets the base (for a query adapter, the base is the identity map). An unknown requester, or a stack
whose audience is not one of the three, gets the base.

A named exception (``AudienceException``) is an explicit per-learner declaration that its stack may also serve the
audiences it names, with the reason it is safe (a model that outputs only a label from a closed set, for example).
It is bound to the objective it was reasoned about: a stack trained toward another objective (a changed blend, other
points) is not covered until the learner declares it again. A learner that may train a safety point never takes
one: a learner whose objective names a safety point, or names no points at all (and so may train any), is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Optional

from .objectives import SAFETY_POINTS, Objective

AUDIENCES = ("owner", "contact", "guest")
#: How widely data of each audience may be seen: the owner's alone is the narrowest.
BREADTH = {"owner": 0, "contact": 1, "guest": 2}


class AudienceError(ValueError):
    """An audience declaration that may not stand, and why."""


def narrowest(audiences: Iterable[str]) -> str:
    """The narrowest of ``audiences``: a stack's audience from its rows' audiences. ValueError when there are none
    or one is not an audience."""
    found = list(audiences)
    if not found:
        raise ValueError("a stack with no training rows has no audience of its own: the base serves")
    unknown = sorted({audience for audience in found if audience not in BREADTH}, key=str)
    if unknown:
        raise ValueError(f"not an audience: {', '.join(map(str, unknown))}")
    return min(found, key=BREADTH.__getitem__)


def may_serve(stack: str, requester: Optional[str]) -> bool:
    """Whether a stack of audience ``stack`` may serve a ``requester`` (the module docstring): the owner may see
    every stack's data, anyone may see a guest stack's, and nobody else is served a learned stack."""
    if stack not in BREADTH or requester not in BREADTH:
        return False
    return requester == "owner" or stack == "guest"


@dataclass(frozen=True)
class AudienceException:
    """A learner's named exception: its stack may also serve ``audiences``, for ``reason``."""

    learner: str
    audiences: FrozenSet[str]
    reason: str


class AudiencePolicy:
    """Which closure serves a request: the learner's serving stack where its audience allows the requester (or the
    learner's named exception does), the base otherwise."""

    def __init__(self) -> None:
        self._exceptions: Dict[str, AudienceException] = {}
        self._objectives: Dict[str, str] = {}

    def declare(self, exception: AudienceException, *, objective: Objective) -> AudienceException:
        if not isinstance(exception.reason, str) or not exception.reason.strip():
            raise AudienceError(f"learner {exception.learner}: a named exception carries its reason")
        if not exception.audiences or not set(exception.audiences) <= set(AUDIENCES):
            raise AudienceError(f"learner {exception.learner}: an exception names audiences from "
                                f"{', '.join(AUDIENCES)}")
        if not objective.points or set(objective.points) & set(SAFETY_POINTS):
            raise AudienceError(f"learner {exception.learner}: a learner that may train a safety point never takes "
                                "an audience exception")
        self._exceptions[exception.learner] = exception
        self._objectives[exception.learner] = objective.identity
        return exception

    def exception(self, learner: str) -> Optional[AudienceException]:
        return self._exceptions.get(learner)

    def select(self, learner: str, requester: Optional[str], *, serving, base):
        """The closure that serves ``requester``: ``serving`` where allowed, else ``base``."""
        if serving.id == base.id or may_serve(serving.audience, requester):
            return serving
        exception = self._exceptions.get(learner)
        if (exception is not None and requester in exception.audiences and serving.audience in BREADTH
                and serving.objective == self._objectives.get(learner)):
            return serving
        return base

    def pin(self, store, requester: Optional[str]):
        """What one request of ``requester`` reads from a learner's revision ``store``: the closure it may be served
        and that closure's state, loaded together (``revisions.Pinned``)."""
        from .revisions import Pinned

        chosen = self.select(store.learner, requester, serving=store.serving(), base=store.base)
        return Pinned(closure=chosen, state=store.state(chosen))


__all__ = ["AUDIENCES", "BREADTH", "AudienceError", "AudienceException", "AudiencePolicy", "may_serve", "narrowest"]
