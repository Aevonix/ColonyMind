"""The sidecar's learner checks, before any implementation loads.

Model roles are configuration declarations, not guesses from model names. Engine checks
(site resolution, runtime memory guards and kind-specific gates) remain the engine's.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .audience import AUDIENCES
from .budgets import BUDGET_TYPES
from .objectives import RegistrationError, SAFETY_POINTS
from .teachers import KINDS, TEACHERS

PROTECTED_ROLES = frozenset({"identity", "owner_gate", "authority", "secrets_detection", "memory_guard",
                             "floor_patterns", "speech_voice", "live_brain", "live_embedder", "live_reranker",
                             "qualified_translator"})
MODEL_ROLES = PROTECTED_ROLES | {"decision_model", "perception", "numeric_predictor", "query_adapter",
                               "score_head", "offline_model", "drafter", "standing_lessons"}
EFFECTS = frozenset({"predict", "rank", "skip", "hold", "end_turn", "teach"})


@dataclass(frozen=True)
class Registered:
    name: str
    kind: str
    teachers: tuple[tuple[str, str], ...]
    audience: str
    dependents: tuple[str, ...]
    forced_run_share: float | None
    erasure_path: str


def _fail(check: str, detail: str) -> None:
    raise RegistrationError(f"{check}: {detail}")


def _names(value: Any, check: str, *, empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or (not value and not empty) or any(
            not isinstance(item, str) or not item.strip() for item in value) or len(set(value)) != len(value):
        _fail(check, "declare a list of distinct names")
    return tuple(value)


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def check_learner(name: str, declaration: Mapping[str, Any]) -> Registered:
    """Return an immutable checked record or name the failed check in RegistrationError.

    ``teachers`` maps teacher ids to their target types. ``model`` declares ``roles`` and
    ``change``; a decision model also declares its points and head or whole-instance coverage.
    ``dependents`` may be explicitly empty. Skip, hold and end-turn effects require a share.
    """
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
        _fail("name", "declare a learner name")
    if not isinstance(declaration, Mapping):
        _fail("declaration", "expected a mapping")
    required = ("kind", "teachers", "model", "budget", "gates", "audience", "personal_data",
                "erasure_path", "dependents", "effect")
    for field in required:
        if field not in declaration:
            _fail(field, "missing declaration")
    kind = declaration["kind"]
    if not isinstance(kind, str) or kind not in KINDS:
        _fail("kind", "unknown learner kind")
    teachers = declaration["teachers"]
    if not isinstance(teachers, Mapping) or not teachers:
        _fail("teachers", "declare teacher ids and target types")
    for teacher_id, target in teachers.items():
        teacher = TEACHERS.get(teacher_id)
        if teacher is None or kind not in teacher.kinds:
            _fail("teachers", f"{teacher_id!r} is not allowed for {kind}")
        if not isinstance(target, str) or target not in teacher.targets_for(kind):
            _fail("target_type", f"{teacher_id} cannot give {target!r} to {kind}")
    model = declaration["model"]
    if not isinstance(model, Mapping):
        _fail("model", "declare roles and change")
    roles = _names(model.get("roles"), "model.roles")
    if set(roles) - MODEL_ROLES:
        _fail("model.roles", "unknown role")
    change = model.get("change")
    if change not in ("head", "encoder", "weights", "client_adapter", "detached", "offline_adapter", "context"):
        _fail("model.change", "declare what changes")
    if kind == "behaviour":
        if change != "context" or roles != ("standing_lessons",):
            _fail("model.change", "behaviour changes only standing lesson context")
    elif change == "context":
        _fail("model.change", "context changes require behaviour")
    protected = set(roles) & PROTECTED_ROLES
    if change == "client_adapter" and kind in {"query-adapter", "supervised-refit"}:
        protected -= {"live_embedder", "live_reranker"}
    if change == "offline_adapter" and kind == "served-adapter":
        protected.discard("live_brain")
    if protected:
        _fail("protected", f"model roles never learn here: {sorted(protected)}")
    budget = declaration["budget"]
    if not isinstance(budget, Mapping) or not isinstance(budget.get("type"), str) or budget["type"] not in BUDGET_TYPES or not (
            _finite(budget.get("bound")) and budget["bound"] > 0):
        _fail("budget", "declare a type and positive finite bound")
    _names(declaration["gates"], "gates")
    if kind != "behaviour" or "arms" in declaration:
        _names(declaration.get("arms"), "arms")
    audience = declaration["audience"]
    if not isinstance(audience, str) or audience not in AUDIENCES:
        _fail("audience", "unknown audience")
    if "style" in teachers and audience != "owner":
        _fail("audience", "style targets are owner scope")
    if kind == "behaviour" and audience != "owner":
        _fail("audience", "behaviour targets are owner scope")
    if not isinstance(declaration["personal_data"], bool):
        _fail("personal_data", "declare true or false")
    erasure = declaration["erasure_path"]
    if not isinstance(erasure, str) or not erasure.strip():
        _fail("erasure_path", "every learner declares how its training is erased, including personal data")
    dependents = _names(declaration["dependents"], "dependents", empty=True)
    effect = declaration["effect"]
    if not isinstance(effect, str) or effect not in EFFECTS:
        _fail("effect", "unknown effect")
    if (kind == "behaviour") != (effect == "teach"):
        _fail("effect", "behaviour requires teach and teach requires behaviour")
    share = declaration.get("forced_run_share")
    if effect in {"skip", "hold", "end_turn"} or share is not None:
        if not _finite(share) or not 0 < share <= 1:
            _fail("forced_run_share", "declare a probability above zero and at most one")
        share = float(share)
    if "decision_model" in roles:
        points = _names(model.get("points"), "decision_model.points")
        if set(points) & SAFETY_POINTS.keys():
            _fail("decision_model", "safety points cannot share learned changes")
        if change == "head":
            if not isinstance(model.get("own_head"), str) or not model["own_head"].strip():
                _fail("decision_model.own_head", "declare this learner's own head")
        elif change == "encoder":
            instance = _names(model.get("instance_points"), "decision_model.instance_points")
            if set(instance) != set(points) or set(instance) & SAFETY_POINTS.keys():
                _fail("decision_model", "an encoder change covers every instance point and no safety point")
        else:
            _fail("decision_model.change", "declare an own head or an encoder change")
    return Registered(name, kind, tuple(sorted(teachers.items())), audience, dependents, share, erasure)
