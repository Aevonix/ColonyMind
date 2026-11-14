"""Revision lineage: a child closure names its parent, and the chain is walkable.

A closure's ``parent`` is the closure it was built from (the one serving when it was built); its ``upstream`` names
closures of other learners it learned from (a score head trained on a query adapter's candidates). Contamination
follows both: erasing a row that trained a parent reaches every descendant, in the learner and across learners. The
walk includes the tombstones of closures retention dropped, so a pruned closure in the middle of a chain never hides
what was built from it.

The carry hook is where a learned child that passed its gates keeps its parent's per-point decision state (its
shadow or live standing), with its own calibration refit: the caller binds the hook, and ``states.Controller`` runs
it at each promotion.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from .revisions import Closure, RevisionStore

CarryHook = Callable[[Closure, Closure], None]


class Lineage:
    """Walks closures across the learners in ``stores`` (their revision stores, by learner)."""

    def __init__(self, stores: Mapping[str, RevisionStore]):
        self.stores: Dict[str, RevisionStore] = dict(stores)
        self._hooks: List[CarryHook] = []

    def add_hook(self, hook: CarryHook) -> None:
        self._hooks.append(hook)

    def bind_instinct(self, store: Any, *, registry: Optional[Mapping[str, Any]] = None,
                      clock: Optional[Callable[[], datetime]] = None) -> CarryHook:
        """Carry the live standing of points whose child calibration the promoted closure names.
        Q1 checks every calibration link and its offline gates; the conditional write preserves demotions that
        arrive while those checks run. The child's first answer need not have reached the decision log yet."""
        from ..instinct import REGISTRY, state
        from ..util.temporal import now_utc

        table = REGISTRY if registry is None else registry
        clock = clock or now_utc

        def carried(parent: Closure, child: Closure) -> None:
            records = [record for record in store.records()
                       if record.id in child.calibration or record.revision in {child.id, child.state}]
            for record in records:
                point = table.get(record.point)
                if point is None or record.tier not in state.promotable(point):
                    continue
                before = store.transitions(point=point.name, tier=record.tier, limit=1)
                sequence = before[0]["seq"] if before else None
                if state.effective(point, record.tier, store.states()) != "live":
                    continue
                allowed = state.promoted_under(store, point, record.tier)
                if record.fingerprint in allowed:
                    continue
                kept, evidence = state.carry(store, point, record.tier, record.fingerprint)
                if kept:
                    store.set_state(point.name, record.tier, "live", source=f"{state.CARRIED} (plasticity)",
                                    evidence={**evidence, "from": list(allowed), "to": record.fingerprint,
                                              "parent_closure": parent.id, "closure": child.id},
                                    at=clock(), expected_seq=sequence)
                else:
                    store.set_state(point.name, record.tier, "demoted",
                                    source="demoted by plasticity: lineage did not carry",
                                    evidence={**evidence, "parent_closure": parent.id, "closure": child.id},
                                    at=clock(), expected_seq=sequence)

        self.add_hook(carried)
        return carried

    def carry(self, parent: Closure, child: Closure) -> None:
        """Run every carry hook for ``child`` promoted over ``parent``."""
        for hook in self._hooks:
            hook(parent, child)

    def _get(self, learner: str, closure_id: str) -> Optional[Closure]:
        store = self.stores.get(learner)
        return store.find(closure_id) if store is not None else None

    def parent(self, closure: Closure) -> Optional[Closure]:
        """The closure ``closure`` was built from, whole or as a tombstone."""
        return self._get(closure.learner, closure.parent) if closure.parent else None

    def chain(self, closure: Closure) -> List[Closure]:
        """``closure`` and its ancestors, child first, as far as the stores know them (to the base)."""
        found, seen = [closure], {closure.id}
        current = self.parent(closure)
        while current is not None and current.id not in seen:
            found.append(current)
            seen.add(current.id)
            current = self.parent(current)
        return found

    def children(self, closure: Closure) -> List[Closure]:
        """The known closures built from ``closure``: its learner's by parent, other learners' by upstream."""
        found = []
        for learner, store in sorted(self.stores.items()):
            for candidate in store.known():
                if (learner == closure.learner and candidate.parent == closure.id) or (
                        (closure.learner, closure.id) in candidate.upstream):
                    found.append(candidate)
        return found

    def descendants(self, roots: Iterable[Tuple[str, str]]) -> List[Tuple[str, Closure]]:
        """The known closures named by ``roots`` ((learner, closure id) pairs) and every descendant of them."""
        found: Dict[Tuple[str, str], Closure] = {}
        queue = [closure for closure in (self._get(learner, closure_id) for learner, closure_id in roots)
                 if closure is not None]
        while queue:
            closure = queue.pop()
            key = (closure.learner, closure.id)
            if key in found:
                continue
            found[key] = closure
            queue.extend(self.children(closure))
        return [(learner, found[(learner, closure_id)]) for learner, closure_id in sorted(found)]

    @staticmethod
    def leaves_first(closures: Iterable[Tuple[str, Closure]]) -> List[Tuple[str, Closure]]:
        """``closures`` ((learner, closure) pairs) ordered so that each comes after every one of them built from it:
        a child before its parent, a downstream closure before the upstream one it learned from. Deleting in this
        order, a crash between two deletions never removes the link that leads to a closure still on disk."""
        items: Dict[Tuple[str, str], Closure] = {(learner, closure.id): closure for learner, closure in closures}
        built_from: Dict[Tuple[str, str], List[Tuple[str, str]]] = {key: [] for key in items}
        for (learner, closure_id), closure in items.items():
            links = ([(learner, closure.parent)] if closure.parent else []) + list(closure.upstream)
            for link in links:
                if link in items and link != (learner, closure_id):
                    built_from[link].append((learner, closure_id))
        ordered: List[Tuple[str, Closure]] = []
        done: set = set()
        for start in sorted(items):
            stack: List[Tuple[Tuple[str, str], int]] = [(start, 0)]
            on_path = {start}
            while stack:
                key, index = stack[-1]
                if key in done:
                    stack.pop()
                    on_path.discard(key)
                    continue
                children = sorted(built_from[key])
                if index < len(children):
                    stack[-1] = (key, index + 1)
                    child = children[index]
                    if child not in done and child not in on_path:
                        stack.append((child, 0))
                        on_path.add(child)
                    continue
                stack.pop()
                on_path.discard(key)
                done.add(key)
                ordered.append((key[0], items[key]))
        return ordered


__all__ = ["CarryHook", "Lineage"]
