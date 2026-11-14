"""Authority: levels x action classes, the floor, standing grants, the deny list, budgets and the breaker.

Architecture section 7. The decision is made here, in the sidecar, before
anything is queued; the worker profile's toolsets and the plugin guard enforce
it inside Hermes. Learning never writes to this module's inputs: the level,
the classes, the floor, the grants, the deny list and the budgets all come
from the owner's configuration or the owner's own confirmation.

The floor is decided by effect: at the intention level a task whose toolsets
include an executor (a shell, code, a browser, delegation) is floor-class and
asks at every level; at execution every tool call of mind-originated work is
classified by what it does (``effect_of``: the tool and its target), never by
what a task says. Nothing is parsed to set a call free: a standing grant names
a structured tool (a declared action, a declared tool, a messaging, cron,
skill, memory or vault tool) and never a shell, code, browser or delegation
call, each of which is the owner's one-tap ask when their yes to the task does
not cover it.

``decide_table`` is the pure decision function the property tests enumerate.
``Authority`` binds it to the store for the budget counters, the breaker,
the standing grants and the permits the owner's yes gives a held effect.
"""

from __future__ import annotations

from colonymind_shared.effects import (
    ACTION_KEYS, ACTION_NAME, ACTION_TIMEOUT, ACTION_TOOL, ARG_KEYS, ARG_NAME, DECLARATION_KEYS,
    EFFECT_FREE_TOOLS, EXECUTOR_TOOLS, Effect, FLOOR_PATTERNS, MESSAGE_TOOLS, MONEY_HOSTS, RESERVED_TOOLS,
    action_argv, action_problem, action_values, canonical_call, declaration_problem, effect_of,
    executor_text, finite_amount, floor_class, floor_match, money_host, read_as, table_reads,
)

import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional
from colonymind.util.temporal import now_utc

from .rank import OUTREACH_ANSWER, OUTREACH_TYPES, TASK_OUTCOME

LEVELS = ("off", "suggest", "standard", "trusted")
CLASSES = ("internal", "owner", "contact", "external", "floor")
DECISIONS = ("act", "ask", "drop", "defer")
MAY_CONTACT = ("never", "ask", "auto")
INTERNAL_SAFE_TOOLSETS = frozenset({"web", "file", "session_search", "memory", "todo"})
# Toolsets whose effects cannot be bounded per call (``mind.floor.executors``): a task that needs one is
# floor-class and asks at every level unless a standing ``executor`` grant covers its toolsets.
EXECUTOR_TOOLSETS = ("terminal", "code_execution", "coding", "debugging", "browser", "computer_use", "desktop_ui",
                     "delegation")


ASK_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # no 0/O, 1/I/L

# -- standing grants -------------------------------------------------------------------------------------------
#
# A grant names one floor class, a scope and the structured tools it covers, and has limits; only the owner sets
# one: ``mind.grants`` in the instance configuration, or their own yes to a ``grant_confirm`` ask in their chat.
# It never names a shell, code, browser or delegation tool and never reads command or code text: each call of
# those outside the workspace is the owner's one-tap ask, and what the owner wants to grant for good is a declared
# action (``mind.actions``) or a declared tool (``mind.floor.tools``). ``match`` lists the exact targets it covers.
# A malformed grant is inert and shown as invalid; it never widens anything.
GRANT_CLASSES = ("money_movement", "irreversible_deletion", "credential_change", "bulk_third_party_messaging")
GRANT_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$", re.IGNORECASE)
GRANT_KEYS = ("id", "class", "scope", "tools", "match", "max_amount", "currency", "per_day", "per_month_amount",
              "expires")
GRANT_USE = "grant_use"          # the note row one use of a grant leaves in the intention store
GRANT_CONFIRM = "grant_confirm"  # the ask a grant requested in conversation forms; the owner's yes activates it
FLOOR_EFFECT = "floor_effect"    # the ask a held effect forms; its yes is the continuation task
# mind.db keys (``mind_state``): a confirmed grant, a revoked configured grant (its fingerprint), the configured
# set's fingerprint at the last load and its last change; a cron job a mind run made (by its id) and a create or
# update let through whose result has not been reported yet; the people one task (or its lineage) reached; and
# the task that made a kanban task.
GRANT_KEY, GRANT_REVOKED = "grant:", "grant_revoked:"
GRANTS_FINGERPRINT, GRANTS_CHANGED = "grants.fingerprint", "grants.changed"
CRON_JOB, CRON_PENDING = "cron_job:", "cron_pending:"
REACH, LINEAGE = "reach:", "lineage:"
# What a grant can never name: the executor tools and toolsets (their calls are read by no grant) and
# ``colonymind_action`` itself (a grant names the declared action).
GRANTLESS = EXECUTOR_TOOLS | frozenset(EXECUTOR_TOOLSETS) | frozenset({ACTION_TOOL})


def grantable(tool: str) -> bool:
    """A tool a standing grant may name: a structured one (a declared action or tool, a messaging, cron, skill,
    memory or vault tool, the outbox), never a shell, code, browser or delegation tool or toolset. A vault tool is
    named as the older tool of its action (``browser_vault_fill``), which is how a call of Hermes' one vault tool
    is read (``read_as``); ``browser_vault`` itself, every action at once, is never a grant's name."""
    tool = str(tool or "")
    return tool not in GRANTLESS and not (tool.startswith("browser_") and not tool.startswith("browser_vault_"))


@dataclass(frozen=True)
class Grant:
    """One standing grant as the authority reads it; ``problem`` makes it inert."""

    id: str
    cls: str
    scope: str
    tools: tuple[str, ...] = ()
    match: tuple[str, ...] = ()
    max_amount: Optional[float] = None
    currency: str = ""
    per_day: Optional[int] = None
    per_month_amount: Optional[float] = None
    expires: Optional[str] = None
    source: str = "config"
    problem: str = ""

    @property
    def valid(self) -> bool:
        return not self.problem

    def expired(self, now: datetime) -> bool:
        if not self.expires:
            return False
        try:
            return now.date() > date.fromisoformat(self.expires)
        except ValueError:
            return True

    def spec(self) -> Dict[str, Any]:
        """The grant as the owner writes it (``mind.grants`` entry)."""
        return {"id": self.id, "class": self.cls, "scope": self.scope, "tools": list(self.tools),
                "match": list(self.match), "max_amount": self.max_amount, "currency": self.currency or None,
                "per_day": self.per_day, "per_month_amount": self.per_month_amount, "expires": self.expires}

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.spec(), sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def as_dict(self, now: datetime | None = None) -> Dict[str, Any]:
        expired = self.expired(now) if now is not None else False
        return {**self.spec(), "source": self.source, "valid": self.valid, "problem": self.problem or None,
                "expired": expired, "active": self.valid and not expired, "fingerprint": self.fingerprint}


def _number(value: Any, name: str, *, whole: bool = False) -> tuple[Optional[float], str]:
    """A limit as written: a finite number (NaN and infinity compare false with every amount, so they would limit
    nothing), whole where it counts uses."""
    if value is None or value == "":
        return None, ""
    if isinstance(value, bool):
        return None, f"{name} must be a number"
    try:
        number = int(value) if whole else float(value)
    except (TypeError, ValueError, OverflowError):
        return None, f"{name} must be a {'whole number' if whole else 'number'}"
    if not math.isfinite(number):
        return None, f"{name} must be a finite number"
    if number < 0 or (whole and number < 1):
        return None, f"{name} must be {'at least 1' if whole else 'zero or more'}"
    return number, ""


def parse_grant(raw: Any, *, source: str = "config") -> Grant:
    """A grant from its written form; a malformed one comes back with ``problem`` set (inert, shown invalid)."""
    if not isinstance(raw, Mapping):
        return Grant(id="?", cls="?", scope="", source=source, problem="a grant is a mapping")
    ident, cls, scope = str(raw.get("id") or "").strip(), str(raw.get("class") or "").strip(), \
        " ".join(str(raw.get("scope") or "").split())
    problems = []
    if not GRANT_ID.fullmatch(ident):
        problems.append("id is 1 to 64 letters, digits, dots, dashes or underscores")
    if cls not in GRANT_CLASSES:
        problems.append(f"class is one of {', '.join(GRANT_CLASSES)}")
    if not scope or len(scope) > 200:
        problems.append("scope is the words shown in asks and state (1 to 200 characters)")
    lists = {}
    for key in ("tools", "match"):
        value = raw.get(key) or []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) and item.strip() for item in value):
            problems.append(f"{key} is a list of strings")
            value = []
        lists[key] = tuple(item.strip() for item in value)
    if not lists["tools"] and not any(problem.startswith("tools") for problem in problems):
        problems.append("tools names the structured tools it covers: a declared action or tool, a messaging tool, "
                        "outbox, cronjob_manage, skill_manage, memory, browser_vault_fill or browser_vault_save_login")
    shells = [name for name in lists["tools"] if not grantable(name)]
    if shells:
        problems.append(f"{', '.join(shells)}: a shell, code, browser or delegation call is never a grant's (each is "
                        f"the owner's one-tap ask); declare an action for the command it should run")
    if any(mark in item for item in lists["match"] for mark in "*?["):
        problems.append("match lists the exact targets it covers (no wildcards)")
    max_amount, problem = _number(raw.get("max_amount"), "max_amount")
    problems += [problem] if problem else []
    per_month, problem = _number(raw.get("per_month_amount"), "per_month_amount")
    problems += [problem] if problem else []
    per_day, problem = _number(raw.get("per_day"), "per_day", whole=True)
    problems += [problem] if problem else []
    currency = str(raw.get("currency") or "").strip().upper()
    if currency and not re.fullmatch(r"[A-Z]{3}", currency):
        problems.append("currency is a three-letter code")
    if (max_amount is not None or per_month is not None) and cls != "money_movement":
        problems.append("amount limits apply to money_movement grants only")
    expires = raw.get("expires")
    if isinstance(expires, datetime):
        expires = expires.date().isoformat()
    elif isinstance(expires, date):
        expires = expires.isoformat()
    elif expires not in (None, ""):
        try:
            expires = date.fromisoformat(str(expires).strip()).isoformat()
        except ValueError:
            problems.append("expires is an ISO date (YYYY-MM-DD)")
            expires = None
    unknown = sorted(set(map(str, raw)) - set(GRANT_KEYS))
    if unknown:
        problems.append(f"unknown field(s): {', '.join(unknown)}")
    return Grant(id=ident or "?", cls=cls or "?", scope=scope, tools=lists["tools"], match=lists["match"],
                 max_amount=max_amount, currency=currency, per_day=int(per_day) if per_day is not None else None,
                 per_month_amount=per_month, expires=expires or None, source=source, problem="; ".join(problems))


def new_ask_code(taken: Iterable[str], *, rng: random.Random | None = None) -> str:
    """A short code the owner can type back; unique among the open asks."""
    rng = rng or random.SystemRandom()
    used = {code.upper() for code in taken}
    for _ in range(1000):
        code = "".join(rng.choice(ASK_ALPHABET) for _ in range(3))
        if code not in used:
            return code
    return "".join(rng.choice(ASK_ALPHABET) for _ in range(6))


def may_contact_of(contact: Any, *, owner_id: str | None) -> str:
    """``may_contact`` for a contact record or id (architecture 7.4).

    The owner is ``auto`` by identity; everyone else is what the
    ``contacts.may_contact`` column says, and ``ask`` when the record has no
    usable value. Nothing else (a tier, a legacy flag, familiarity) grants.
    """
    contact_id = contact if isinstance(contact, str) else getattr(contact, "contact_id", None)
    if contact_id is None and isinstance(contact, Mapping):
        contact_id = contact.get("contact_id")
    if owner_id and contact_id == owner_id:
        return "auto"
    if contact is None or isinstance(contact, str):
        return "ask"
    value = contact.get("may_contact") if isinstance(contact, Mapping) else getattr(contact, "may_contact", None)
    return str(value) if value in MAY_CONTACT else "ask"


def classify(*, kind: str, recipient: str | None, owner_id: str | None,
             toolsets: Iterable[str] = (), text: str = "", executors: Iterable[str] = EXECUTOR_TOOLSETS) -> str:
    """The class of an intention from its kind, its target and its toolsets (architecture 7.2).

    Its words never decide the floor (``text`` is accepted and not read): a task or goal whose toolsets include
    an executor is ``floor`` (it asks at every level: the owner's yes to it is what lets its commands run), and
    what each of its calls does is decided when it runs (``effect_of``). A message is classed by its
    recipient; whether it is bulk is decided per send."""
    if kind == "message":
        return "owner" if recipient and recipient == owner_id else "contact"
    if kind in {"task", "goal"}:
        names = {str(name) for name in toolsets}
        if names & {str(name) for name in executors}:
            return "floor"
        extra = names - INTERNAL_SAFE_TOOLSETS
        if extra:
            return "external"
        if recipient and recipient != owner_id:
            return "contact"
        return "owner" if recipient == owner_id and recipient else "internal"
    return "internal"


def decide_table(*, level: str, cls: str, may_contact: str = "ask", floor: bool = False,
                 deny: bool = False, budget_exhausted: bool = False,
                 breaker_tripped: bool = False, enabled: bool = True, requested: bool = False) -> str:
    """The pure decision (architecture 7.2, 7.3, 7.5, 7.6), in precedence order:

    1. the off switch or ``off`` level: ``drop``
    2. the deny list: ``drop``
    3. the floor: ``ask``, and nothing raises it
    4. the level x class table, with ``may_contact`` for the contact class; at ``suggest`` a
       ``requested`` word to the owner (``REQUESTED_TYPES``) acts like internal work: the level
       holds the mind's initiative for the digest, not what the owner asked to be told
    5. a tripped breaker demotes an ``act`` one level, to ``ask``
    6. an exhausted budget defers an ``act``; it is not an error
    """
    if not enabled or level == "off":
        return "drop"
    if deny:
        return "drop"
    if floor or cls == "floor":
        return "ask"
    if level == "suggest":
        decision = "act" if cls == "internal" or (requested and cls == "owner") else "ask"   # digest only
    elif level == "standard":
        if cls in {"internal", "owner"}:
            decision = "act"
        elif cls == "contact":
            decision = {"auto": "act", "ask": "ask", "never": "drop"}[may_contact]
        else:
            decision = "ask"                                  # external
    elif level == "trusted":
        if cls == "contact":
            decision = {"auto": "act", "ask": "ask", "never": "drop"}[may_contact]
        else:
            decision = "act"
    else:
        raise ValueError(f"unknown autonomy level {level!r}")
    if decision == "act" and breaker_tripped:
        decision = "ask"
    if decision == "act" and budget_exhausted:
        decision = "defer"
    return decision


# -- standing allows --------------------------------------------------------------------------------------------
#
# The owner's "always" on an ask (``allows.py``) stops the asking for exactly one class of work, named by its key:
# ``contact:<id>`` (that contact's permission, which the allow sets to ``auto``), ``toolset:<toolsets>`` (a task or
# goal whose toolsets beyond the internal ones are exactly these) and ``grant:<id>`` (a standing grant for a held
# structured call). A task with an executor toolset has none: a standing permission for work the owner never saw must
# keep its calls inside the task's workspace, and nothing confines a shell, code, browser or delegation call there by
# structure (the plugin confines ``write_file`` and ``patch`` only), so each such task keeps its one ask and the
# owner's yes to it is what lets its commands run. An allow applies only at these levels (my choice: at ``suggest``
# the owner reviews everything), and never lifts the off switch, the deny list, the breaker or the budgets.
ALLOW_LEVELS = frozenset({"standard", "trusted"})
ALLOW_KINDS = ("contact", "toolset", "grant")


def executor_call(tool: Any) -> bool:
    """A shell, code, browser, delegation or desktop call: what it reaches is never confined by structure."""
    tool = str(tool or "")
    return tool in EXECUTOR_TOOLS or tool.startswith("browser_")


def allow_key(*, kind: str, cls: str, toolsets: Iterable[str] = (), recipient: str | None = None,
              executors: Iterable[str] = EXECUTOR_TOOLSETS) -> Optional[str]:
    """The class of work a standing allow for this intention would cover (see ``ALLOW_KINDS``), or None: a message or
    task to a contact (``contact``), or a task or goal of class ``external`` (its extra toolsets, none of them an
    executor). Anything else has no allow: an owner's question, internal work, a task with an executor toolset, a
    floor effect held at execution (its "always" is a grant, ``allows.offer_for``)."""
    if cls == "contact" and kind in {"message", "task", "goal"} and recipient:
        return f"contact:{recipient}"
    if kind not in {"task", "goal"} or cls != "external":
        return None
    extra = sorted({str(name) for name in toolsets} - INTERNAL_SAFE_TOOLSETS)
    if not extra or set(extra) & {str(name) for name in executors}:
        return None
    return f"toolset:{','.join(extra)}"


@dataclass
class Verdict:
    decision: str
    reason: str
    cls: str
    floor: Optional[str] = None
    notice: bool = True  # False for suggest-level asks: they wait for the digest
    grant: Optional[str] = None  # the standing grant that let a floor-class intention be decided by the table
    allow: Optional[str] = None  # the owner's standing allow that let this class of work act without asking

    def as_dict(self) -> Dict[str, Any]:
        return {"decision": self.decision, "reason": self.reason, "cls": self.cls,
                "floor": self.floor, "notice": self.notice, "grant": self.grant, "allow": self.allow}


# The research-shaped task types (the ones whose report is a finding, ``outcomes.FINDING_TYPES``): one
# worker run reads sources and writes them up, so it gets a longer run and a second attempt.
RESEARCH_TASK_BUDGET = {"max_runtime_s": 1800, "max_retries": 2}
DEFAULT_TASK_TYPES = {name: dict(RESEARCH_TASK_BUDGET)
                      for name in ("research", "question", "mastery_investigation", "goal_step", "outreach_followup")}


@dataclass
class Budgets:
    tasks_per_hour: int = 4
    concurrent_tasks: int = 2
    owner_messages_per_day: int = 3
    outreach_per_day: int = 3      # unprompted outreach to the owner in any 24 h (architecture 4.10)
    contact_messages_per_day: int = 5
    per_contact_cooldown_hours: float = 24
    llm_tokens_per_day: int = 200000
    learn_share: float = 0.25
    open_goals: int = 2
    goal_tasks: int = 4            # steps an agent-owned goal may spend
    goal_horizon_days: int = 7     # the longest horizon an adopted goal may have
    task_max_runtime_s: int = 600  # one worker run of a task type ``task_types`` does not name
    task_max_retries: int = 1
    # Per task type: ``{max_runtime_s, max_retries}``, either one falling back to the two above.
    task_types: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {name: dict(value) for name, value in DEFAULT_TASK_TYPES.items()})

    @classmethod
    def from_config(cls, value: Mapping[str, Any] | None) -> "Budgets":
        budgets = cls()
        for name, default in vars(budgets).items():
            raw = (value or {}).get(name, default)
            if name == "task_types":
                # Over the defaults, one type and one field at a time: overriding research's runtime
                # keeps its retries, and the other types keep theirs.
                merged = {kind: dict(entry) for kind, entry in default.items()}
                for kind, entry in (raw.items() if isinstance(raw, Mapping) else ()):
                    if not isinstance(entry, Mapping):
                        continue
                    target = merged.setdefault(str(kind), {})
                    for key in ("max_runtime_s", "max_retries"):
                        try:
                            if key in entry:
                                target[key] = int(entry[key])
                        except (TypeError, ValueError):
                            pass
                budgets.task_types = merged
                continue
            try:
                setattr(budgets, name, type(default)(raw))
            except (TypeError, ValueError):
                setattr(budgets, name, default)
        return budgets

    def for_task(self, task_type: str | None) -> tuple[int, int]:
        """``(max_runtime_s, max_retries)``: one worker run's limit and the retries after a failed run,
        for a task of this type."""
        entry = self.task_types.get(str(task_type or "")) or {}
        return (int(entry.get("max_runtime_s", self.task_max_runtime_s)),
                int(entry.get("max_retries", self.task_max_retries)))


@dataclass
class Breaker:
    failures: int = 3
    window_hours: float = 24
    demotion_hours: float = 72

    @classmethod
    def from_config(cls, value: Mapping[str, Any] | None) -> "Breaker":
        value = value or {}
        return cls(int(value.get("failures", 3)), float(value.get("window_hours", 24)),
                   float(value.get("demotion_hours", 72)))


@dataclass
class Policy:
    """The ``mind:`` section as the authority reads it."""

    level: str = "suggest"
    enabled: bool = True
    deny_tools: frozenset[str] = frozenset()
    deny_text: tuple[re.Pattern[str], ...] = ()
    deny_commands: tuple[str, ...] = ()
    worker_toolsets: tuple[str, ...] = ("web", "file", "session_search", "memory", "todo")
    budgets: Budgets = field(default_factory=Budgets)
    breaker: Breaker = field(default_factory=Breaker)
    ask_expires_hours: float = 72
    quiet_hours: str = "22:00-07:00"
    # The floor by effect (``mind.grants`` and ``mind.floor``).
    grants: tuple[Grant, ...] = ()
    executors: tuple[str, ...] = EXECUTOR_TOOLSETS
    # ``mind.floor.tools`` as written (``effect_of`` checks each entry; one with a problem stays unbounded) and the
    # problem of each entry that has one, shown in the mind's state.
    floor_tools: Dict[str, Any] = field(default_factory=dict)
    floor_tool_problems: Dict[str, str] = field(default_factory=dict)
    money_hosts: tuple[str, ...] = MONEY_HOSTS
    bulk_recipients: int = 3
    permit_hours: float = 24
    # ``mind.actions`` as written (``effect_of`` and the plugin check each entry; one with a problem runs nothing)
    # and the problem of each entry that has one, shown in the mind's state.
    actions: Dict[str, Any] = field(default_factory=dict)
    action_problems: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, mind: Mapping[str, Any] | None) -> "Policy":
        mind = mind or {}
        deny = mind.get("deny") if isinstance(mind.get("deny"), Mapping) else {}
        patterns = []
        for raw in deny.get("text") or []:
            try:
                patterns.append(re.compile(str(raw), re.IGNORECASE))
            except re.error:
                continue
        level = str(mind.get("autonomy") or "suggest").lower()
        floor = mind.get("floor") if isinstance(mind.get("floor"), Mapping) else {}
        grants: List[Grant] = []
        written = mind.get("grants") or []
        if not isinstance(written, list):
            written = [{"problem": "mind.grants is a list of grants"}]
        for raw in written:
            grant = parse_grant(raw, source="config")
            if grant.valid and any(other.id == grant.id for other in grants):
                grant = replace(grant, problem=f"a second grant with the id {grant.id}")
            grants.append(grant)
        tools = floor.get("tools") if isinstance(floor.get("tools"), Mapping) else {}
        actions = {str(name): dict(spec) if isinstance(spec, Mapping) else spec
                   for name, spec in (mind.get("actions") if isinstance(mind.get("actions"), Mapping) else {}).items()}
        action_problems = {name: problem for name, spec in actions.items() if (problem := action_problem(name, spec))}
        if mind.get("actions") not in (None, {}) and not isinstance(mind.get("actions"), Mapping):
            action_problems["mind.actions"] = "mind.actions maps each action's name to its declaration"
        try:
            bulk = max(1, int(floor.get("bulk_recipients", 3)))
        except (TypeError, ValueError):
            bulk = 3
        try:
            permit_hours = max(0.0, float(floor.get("permit_hours", 24)))
        except (TypeError, ValueError):
            permit_hours = 24.0
        return cls(
            level=level if level in LEVELS else "suggest",
            enabled=mind.get("enabled", True) is not False,
            deny_tools=frozenset(str(name) for name in deny.get("tools") or []),
            deny_text=tuple(patterns),
            deny_commands=tuple(str(item) for item in deny.get("commands") or []),
            worker_toolsets=tuple(str(item) for item in mind.get("worker_toolsets") or cls.worker_toolsets),
            budgets=Budgets.from_config(mind.get("budgets")),
            breaker=Breaker.from_config(mind.get("breaker")),
            ask_expires_hours=float(mind.get("ask_expires_hours", 72) or 72),
            quiet_hours=str(mind.get("quiet_hours") or ""),
            grants=tuple(grants),
            executors=tuple(str(item) for item in floor.get("executors") or EXECUTOR_TOOLSETS),
            floor_tools={str(name): dict(spec) if isinstance(spec, Mapping) else spec for name, spec in tools.items()},
            floor_tool_problems={str(name): problem for name, spec in tools.items() if (problem := (
                "the effect table reads this tool itself; a declaration is for a tool it does not know"
                if table_reads(name) else declaration_problem(dict(spec) if isinstance(spec, Mapping) else spec)))},
            money_hosts=tuple(str(item).strip().lower() for item in floor.get("money_hosts") or MONEY_HOSTS),
            bulk_recipients=bulk,
            permit_hours=permit_hours,
            actions=actions,
            action_problems=action_problems,
        )

    def denied(self, text: str, tools: Iterable[str] = ()) -> Optional[str]:
        """The deny-list reason for this text or tool set, if any (about 30 lines, 7.5)."""
        for tool in tools:
            if str(tool) in self.deny_tools:
                return f"tool {tool} is on the deny list"
        for pattern in self.deny_text:
            if pattern.search(text or ""):
                return f"text matches the denied pattern {pattern.pattern!r}"
        return None


def demote(level: str) -> str:
    """One level down; ``suggest`` and ``off`` stay where they are."""
    index = LEVELS.index(level) if level in LEVELS else 2
    return LEVELS[max(1, index - 1)]


class Authority:
    """Bind the decision table to the intention store: budgets, breaker, guard."""

    BUDGET_ACTION = "queued"  # the history action counted by the budgets
    DIGEST_TYPES = ("digest", "ask_notice", "breaker_notice", "health_notice")
    # Words to the owner that are not the mind's initiative: its own reports, the reminders and
    # heads-ups the owner asked for, and the question a message the owner asked for raises (who is
    # the recipient the owner named). ``suggest`` does not hold them for the digest.
    # Words owed to the owner for what they asked: the answer to a follow-up, and the report of a task done
    # for an obligation to them. Decided like every word to the owner (the off switch, the level, the floor,
    # the deny list), they are the obligation's own delivery: bounded by what was asked (a task budget, a
    # follow-up), never held by the daily owner budget and never spending a slot of it.
    OWED_OWNER_TYPES = (OUTREACH_ANSWER, TASK_OUTCOME)
    REQUESTED_TYPES = DIGEST_TYPES + ("commitment_reminder", "commitment_due_soon", "recipient_unknown",
                                      *OWED_OWNER_TYPES)
    # Unprompted outreach has its own daily budget and never takes a slot of the owner budget, so it can
    # never defer a reminder the owner asked for; the answer to a follow-up the owner asked for is neither.
    OUTREACH_OWNER_TYPES = tuple(sorted(OUTREACH_TYPES)) + (OUTREACH_ANSWER,)

    def __init__(self, policy: Policy, store: Any, *, owner_id: str | None, clock=None,
                 grants: Callable[[], Iterable[Grant]] | None = None,
                 allows: Callable[[], Iterable[Any]] | None = None) -> None:
        self.policy = policy
        self.store = store
        self.owner_id = owner_id
        self.clock = clock or (lambda: now_utc())
        self._enabled_override: Optional[bool] = None
        # The standing grants in force (the mind's: configured, confirmed, minus revoked); the policy's own
        # configured grants when the authority stands alone.
        self._grants = grants
        # The owner's standing allows in force (``allows.Allow``: ``id``, ``key``, ``words``); none when the
        # authority stands alone.
        self._allows = allows

    # -- state ----------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """The off switch and the ``off`` level both mean no further effects (7.9)."""
        if self.policy.level == "off":
            return False
        if self._enabled_override is not None:
            return self._enabled_override
        return self.policy.enabled

    def set_enabled(self, value: Optional[bool]) -> None:
        self._enabled_override = value

    @property
    def level(self) -> str:
        return self.policy.level

    # -- breaker -----------------------------------------------------------------

    def breaker_state(self, cls: str, now: datetime | None = None) -> Dict[str, Any]:
        """3 failures of a class within 24 h demote it for 72 h, or until a reset (7.6)."""
        now = now or self.clock()
        breaker = self.policy.breaker
        since = now - timedelta(hours=breaker.window_hours)
        reset_at = self.store.last_transition_at("breaker_reset", type=f"breaker_reset:{cls}") \
            if self.store is not None else None
        # A failure the outcome marked uncounted (a timeout, ``Outcomes._breaker_count``) fails its task only.
        failures = [row for row in (self.store.failures_since(cls, since) if self.store is not None else [])
                    if (reset_at is None or (row.failed_at and row.failed_at > reset_at))
                    and ((row.result_metadata or {}).get("breaker") or {}).get("counted") is not False]
        tripped = len(failures) >= breaker.failures
        until = None
        if tripped:
            latest = max(row.failed_at for row in failures if row.failed_at)
            until = latest + timedelta(hours=breaker.demotion_hours)
            tripped = now < until
        return {"cls": cls, "tripped": tripped, "failures": len(failures),
                "until": until.isoformat() if until and tripped else None}

    def reset_breaker(self, cls: str, *, by: str = "owner") -> Dict[str, Any]:
        """Record an owner reset as an audit row: failures before it no longer count."""
        if cls not in CLASSES:
            raise ValueError(f"unknown class {cls!r}")
        row, _ = self.store.create_intention(
            kind="note", type=f"breaker_reset:{cls}", title=f"breaker reset for {cls} by {by}",
            drive="upkeep", cls="internal", decision="act", decision_reason="owner reset",
            status="done", dedup_key=None, hermes_kind="none")
        self.store.transition(row.id, "done", action="breaker_reset", details={"cls": cls, "by": by},
                              outcome="done", verified="owner", completed_at=self.clock())
        return self.breaker_state(cls)

    # -- budgets -------------------------------------------------------------------

    def budget_check(self, *, kind: str, recipient: str | None, type: str = "",
                     now: datetime | None = None, cooldown_hours: float | None = None) -> Optional[str]:
        """The reason an act must wait, or None when the budgets allow it (7.6).

        ``cooldown_hours`` is a message's own per-contact cooldown (a check-in's
        reply-and-silence backoff, architecture 4.7 item 6); it replaces the
        flat ``per_contact_cooldown_hours`` for that one decision. The daily
        contact-message cap still applies.
        """
        if self.store is None:
            return None
        now = now or self.clock()
        budgets = self.policy.budgets
        if kind in {"task", "goal"}:
            hour_ago = now - timedelta(hours=1)
            if self.store.count_transitions(self.BUDGET_ACTION, hour_ago, kind="task") >= budgets.tasks_per_hour:
                return f"budget: {budgets.tasks_per_hour} tasks per hour reached"
            if kind == "goal":
                # A goal owns no Hermes object (its steps are tasks); only the goal count is budgeted.
                open_goals = len(self.store.intentions(status=["approved", "asked", "proposed"], kind=["goal"],
                                                       limit=1000))
                if open_goals >= budgets.open_goals:
                    return f"budget: {budgets.open_goals} open goals reached"
                return None
            # A blocked task waits on someone in Hermes and runs nothing: it holds no slot.
            running = len([row for row in self.store.intentions(status=["approved", "dispatched"], kind=["task"],
                                                                limit=1000) if row.outcome != "blocked"])
            if running >= budgets.concurrent_tasks:
                return f"budget: {budgets.concurrent_tasks} concurrent tasks reached"
            return None
        if kind == "message":
            if type in self.DIGEST_TYPES:
                return None
            day_ago = now - timedelta(days=1)
            if recipient and recipient == self.owner_id:
                if type in self.OWED_OWNER_TYPES:
                    return None
                if type in OUTREACH_TYPES:
                    sent = self.store.count_transitions(self.BUDGET_ACTION, day_ago, kind="message",
                                                        recipient=recipient,
                                                        include_types=tuple(sorted(OUTREACH_TYPES)))
                    if sent >= budgets.outreach_per_day:
                        return f"budget: {budgets.outreach_per_day} outreach messages per day reached"
                    return None
                sent = self.store.count_transitions(self.BUDGET_ACTION, day_ago, kind="message",
                                                    recipient=recipient,
                                                    exclude_types=(self.DIGEST_TYPES + self.OUTREACH_OWNER_TYPES
                                                                   + self.OWED_OWNER_TYPES))
                if sent >= budgets.owner_messages_per_day:
                    return f"budget: {budgets.owner_messages_per_day} owner messages per day reached"
                return None
            sent = self.store.count_transitions(self.BUDGET_ACTION, day_ago, kind="message",
                                                exclude_types=self.DIGEST_TYPES)
            owner_sent = self.store.count_transitions(self.BUDGET_ACTION, day_ago, kind="message",
                                                      recipient=self.owner_id, exclude_types=self.DIGEST_TYPES) \
                if self.owner_id else 0
            if sent - owner_sent >= budgets.contact_messages_per_day:
                return f"budget: {budgets.contact_messages_per_day} contact messages per day reached"
            if recipient:
                hours = budgets.per_contact_cooldown_hours if cooldown_hours is None else max(0.0, float(cooldown_hours))
                last = self.store.last_transition_at(self.BUDGET_ACTION, recipient=recipient)
                if last is not None and now - last < timedelta(hours=hours):
                    return f"budget: {hours:g} h cooldown for this contact"
        return None

    def tokens_allowed(self, now: datetime | None = None) -> bool:
        if self.store is None:
            return True
        now = now or self.clock()
        return self.store.tokens_since(now - timedelta(days=1)) < self.policy.budgets.llm_tokens_per_day

    # -- standing grants and permits --------------------------------------------------------

    def grants(self) -> List[Grant]:
        """The standing grants in force, valid or not (an invalid or expired one covers nothing)."""
        try:
            return list(self._grants() if self._grants is not None else self.policy.grants)
        except Exception:  # an unreadable grant store grants nothing
            return []

    def grant_usage(self, grant_id: str, now: datetime | None = None) -> Dict[str, Any]:
        """Uses of a grant in the last 24 h and the amount it moved in the last 30 days (its ``grant_use`` rows)."""
        now = now or self.clock()
        if self.store is None:
            return {"uses_24h": 0, "amount_30d": 0.0}
        rows = [row for row in self.store.intentions(kind=["note"], since=now - timedelta(days=30), limit=5000)
                if row.type == GRANT_USE and (row.context or {}).get("grant") == grant_id]
        day_ago = now - timedelta(days=1)
        return {"uses_24h": sum(1 for row in rows if row.created_at and row.created_at >= day_ago),
                "amount_30d": round(sum(float((row.context or {}).get("amount") or 0.0) for row in rows), 2)}

    def covers(self, grant: Grant, effect: Effect, now: datetime | None = None) -> bool:
        """Whether ``grant`` lets ``effect`` act: its class, one of the structured tools it names (never a shell, code,
        browser or delegation call: no grant reads what those run), its exact targets, the amount per effect (an
        unknown or non-finite amount is never covered by a grant with an amount limit), uses per day and amount
        per month, while it is valid and unexpired. A call with a problem (a declaration or an action it cannot be
        read by) is never covered."""
        now = now or self.clock()
        if not grant.valid or grant.expired(now) or effect.problem or not grantable(effect.tool):
            return False
        if grant.cls != effect.cls or effect.tool not in grant.tools:
            return False
        if grant.match and not all(target in grant.match for target in list(effect.recipients) or [effect.target]):
            return False
        amount = finite_amount(effect.amount)
        if grant.max_amount is not None or grant.per_month_amount is not None:
            if amount is None or (grant.currency and effect.currency != grant.currency):
                return False
            if grant.max_amount is not None and amount > grant.max_amount:
                return False
        usage = self.grant_usage(grant.id, now) if (grant.per_day or grant.per_month_amount is not None) else {}
        if grant.per_day and usage.get("uses_24h", 0) >= grant.per_day:
            return False
        if grant.per_month_amount is not None and usage.get("amount_30d", 0.0) + float(amount or 0.0) \
                > grant.per_month_amount:
            return False
        return True

    def covering_grant(self, effect: Effect, now: datetime | None = None) -> Optional[Grant]:
        return next((grant for grant in self.grants() if self.covers(grant, effect, now)), None)

    def record_grant_use(self, grant_id: str, effect: Effect, *, task: str | None = None) -> Any:
        """One use of a grant, as a ``grant_use`` note in the intention store (what the limits count)."""
        if self.store is None:
            return None
        now = self.clock()
        row, _ = self.store.create_intention(
            kind="note", type=GRANT_USE, title=f"standing grant {grant_id} used: {effect.describe()}"[:200],
            drive="upkeep", cls="internal", decision="act", decision_reason=f"standing grant {grant_id}",
            status="done", dedup_key=None, hermes_kind="none", created_at=now,
            context={"grant": grant_id, "amount": effect.amount, "currency": effect.currency,
                     "effect": effect.digest, "text": effect.describe(), "task": task})
        return row

    def owner_approved(self, task: Any) -> bool:
        """The task's intention was asked and the owner said yes (a continuation carries its origin's yes)."""
        row = task
        if row is not None and row.type == FLOOR_EFFECT:
            origin = (row.context or {}).get("origin_task") if isinstance(row.context, dict) else None
            row = self.store.get(str(origin)) if origin and self.store is not None else None
        return row is not None and bool(row.ask_code) and row.decision == "ask" \
            and row.status in {"approved", "dispatched"}

    def permit_for(self, task: Any, effect: Effect, now: datetime | None = None) -> Optional[str]:
        """The owner's yes to exactly this effect: the continuation task formed from its ask, within
        ``mind.floor.permit_hours`` of the yes and not yet used (one use)."""
        if task is None or self.store is None or task.type != FLOOR_EFFECT:
            return None
        context = task.context if isinstance(task.context, dict) else {}
        if context.get("effect_digest") != effect.digest or task.status not in {"approved", "dispatched"}:
            return None
        history = self.store.get_history(task.id, limit=200)
        if any(item.action == "permit_used" for item in history):
            return None
        approved = [item.timestamp for item in history if item.action == "queued" and (item.details or {}).get("code")]
        now = now or self.clock()
        if not approved or now - max(approved) > timedelta(hours=self.policy.permit_hours):
            return None
        return task.id

    def allows(self) -> List[Any]:
        """The owner's standing allows in force; an unreadable store allows nothing."""
        try:
            return list(self._allows()) if self._allows is not None else []
        except Exception:
            return []

    def allow_for(self, key: Optional[str]) -> Optional[Any]:
        """The standing allow for exactly this class of work (``allow_key``), at a level where allows apply."""
        if not key or self.level not in ALLOW_LEVELS:
            return None
        return next((allow for allow in self.allows() if getattr(allow, "key", None) == key), None)

    def allowed(self, task: Any) -> Optional[Any]:
        """The standing toolset allow a task was formed under (``context['allow']``, recorded when it acted by it),
        while that allow is still in force at a level where allows apply and the task is approved or under way: the
        unbounded calls of those toolsets run as the owner's approved work (never a shell, code, browser or
        delegation call: ``effect_verdict``). Revoking the allow stops them at once."""
        context = getattr(task, "context", None)
        ident = context.get("allow") if isinstance(context, dict) else None
        if not ident or self.level not in ALLOW_LEVELS or getattr(task, "status", None) not in {"approved",
                                                                                                "dispatched"}:
            return None
        return next((allow for allow in self.allows() if getattr(allow, "id", None) == ident
                     and getattr(allow, "kind", None) == "toolset"), None)

    # -- the decision ----------------------------------------------------------------

    def decide(self, *, kind: str, recipient: str | None, text: str, type: str = "",
               may_contact: str = "ask", toolsets: Iterable[str] = (), tools: Iterable[str] = (),
               now: datetime | None = None, cooldown_hours: float | None = None, reached: int = 0) -> Verdict:
        """act | ask | drop | defer for one intention, with the reason (7.2-7.6).

        The floor is not read from ``text``: an executor task is floor-class (the owner's yes to it is what lets its
        commands run; no grant or standing allow makes executor use the mind's own), and a message to a contact is
        bulk when its concern or source already reached ``reached`` other people in 24 h, at least
        ``mind.floor.bulk_recipients`` (unless a bulk grant covers it). The owner's standing allow for exactly this
        class of external work (``allow_key``: its extra toolsets) decides it as internal work at ``standard`` and
        ``trusted``: the off switch, the deny list, the breaker and the budgets still apply.
        """
        now = now or self.clock()
        toolsets = [str(name) for name in toolsets]
        cls = classify(kind=kind, recipient=recipient, owner_id=self.owner_id, toolsets=toolsets,
                       executors=self.policy.executors)
        matched: Optional[str] = None
        granted: Optional[Grant] = None
        if cls == "floor":
            matched = "executor"
        elif kind == "message" and cls == "contact" and recipient and reached >= self.policy.bulk_recipients:
            effect = Effect("bulk_third_party_messaging", "outbox", target=str(recipient), recipients=(str(recipient),))
            granted = self.covering_grant(effect, now)
            if granted is None:
                matched, cls = "bulk_third_party_messaging", "floor"
        allow = None
        if matched is None and cls == "external":
            allow = self.allow_for(allow_key(kind=kind, cls=cls, toolsets=toolsets, recipient=recipient,
                                             executors=self.policy.executors))
        denied = self.policy.denied(text, tools)
        breaker = self.breaker_state(cls, now) if cls != "floor" else {"tripped": False}
        budget = self.budget_check(kind=kind, recipient=recipient, type=type, now=now, cooldown_hours=cooldown_hours)
        decision = decide_table(level=self.level, cls="internal" if allow is not None else cls,
                                may_contact=may_contact, floor=matched is not None,
                                deny=denied is not None, budget_exhausted=budget is not None,
                                breaker_tripped=bool(breaker.get("tripped")), enabled=self.enabled,
                                requested=kind == "message" and type in self.REQUESTED_TYPES)
        if self.level == "off":
            reason = "autonomy level off"
        elif not self.enabled:
            reason = "the mind is off"
        elif denied:
            reason = denied
        elif matched == "executor":
            names = ", ".join(sorted(set(toolsets) & set(self.policy.executors)))
            reason = f"floor: executor tools ({names}) always ask; the owner's yes to the task lets its commands run"
        elif matched:
            reason = (f"floor: bulk messaging ({reached + 1} people from one concern in 24 h) needs the owner's yes")
        elif decision == "defer":
            reason = budget or "budget"
        elif decision == "ask" and breaker.get("tripped"):
            reason = f"breaker: {breaker['failures']} recent {cls} failures; asking until {breaker.get('until')}"
        elif allow is not None and decision == "act":
            reason = f"{self.level}: the owner's standing allow {allow.id} ({allow.words})"
        elif self.level == "suggest" and decision == "ask":
            reason = f"{self.level}: {cls} effects are suggested in the digest"
        elif cls == "contact":
            reason = f"{self.level}: contact may_contact={may_contact}"
        else:
            reason = f"{self.level}: {cls} -> {decision}"
        if granted is not None and not denied:
            reason = f"{reason}; standing grant {granted.id} ({granted.scope})"
        return Verdict(decision=decision, reason=reason, cls=cls, floor=matched,
                       notice=not (decision == "ask" and self.level == "suggest" and matched is None),
                       grant=granted.id if granted is not None else None,
                       allow=allow.id if allow is not None and decision == "act" else None)

    # -- the guard (7.5) ------------------------------------------------------------------

    def effect_verdict(self, effect: Effect, *, task: Any = None, run: str = "mind") -> Dict[str, Any]:
        """The verdict on one call's effect at execution. The owner's yes to exactly this call (a permit, bound to
        the complete call) lets it run once. Otherwise a floor effect acts only when a standing grant covers it (the
        caller records the use), and an unbounded one when the owner approved the task or, for a call that is no
        shell, code, browser or delegation call, when the owner's standing toolset allow covers it; a call with a
        problem (a declaration or an action it cannot be read by) needs their yes to exactly it. Anything else holds.
        Only mind-originated work has grants and permits: anywhere else the floor asks."""
        shown = {"floor": effect.cls if effect.floor else None, "effect": effect.as_dict()}
        permit = self.permit_for(task, effect) if run == "mind" else None
        if permit is not None:
            return {"allow": True, "action": "allow", "reason": "the owner approved exactly this call",
                    "permit": permit, **shown}
        if run == "mind" and not effect.floor and not effect.problem and self.owner_approved(task):
            return {"allow": True, "action": "allow", "reason": "the owner approved this task", **shown}
        # A standing allow covers work the owner never saw, so only calls that stay where it was allowed: never an
        # executor call, which nothing confines to the task's workspace (the owner's yes to the task, above, does).
        allowed = self.allowed(task) if run == "mind" and not effect.floor and not effect.problem \
            and not executor_call(effect.tool) else None
        if allowed is not None:
            return {"allow": True, "action": "allow", "allowed": allowed.id,
                    "reason": f"the owner's standing allow {allowed.id} covers this task ({allowed.words})", **shown}
        grant = self.covering_grant(effect) if run == "mind" else None
        if grant is not None:
            return {"allow": True, "action": "allow", "reason": f"standing grant {grant.id} ({grant.scope})",
                    "grant": grant.id, **shown}
        if effect.floor:
            reason = f"floor: {effect.cls.replace('_', ' ')} needs the owner's yes"
        elif effect.problem:
            reason = f"{effect.problem}; each call waits for the owner's yes to exactly it"
        else:
            reason = ("an unbounded effect needs the owner's yes: this task was not approved, and no standing grant "
                      "covers a shell, code, browser or delegation call")
        return {"allow": False, "action": "ask", "hold": True, "reason": reason, **shown}

    def guard(self, *, tool: str, args: Mapping[str, Any] | None, run: str = "mind",
              recipient_may_contact: str | None = None, task: Any = None, cwd: str | None = None,
              context: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        """The sidecar half of ``pre_tool_call``: allow | block | ask with a reason.

        Off means every effect from a mind-originated run is blocked. The deny list applies to the tool name (for
        Hermes' one vault tool also the older tool it is read as, ``read_as``) and its arguments. Then the call's
        effect (``effect_of``: the tool and its target, with ``cwd`` for the
        directory a command runs in and ``context`` for what only the plugin can read) is decided by
        ``effect_verdict``: with no store a floor effect asks. A call of a declared action that its declaration
        does not allow is refused as a call to correct. A message follows the level x permission table
        the intention uses (``decide_table``: act allows, ask asks, drop and defer block; an ask carries the
        call's effect, and the owner's yes to exactly that call is a permit for it); a recipient who is the owner
        is left out of ``recipient_may_contact`` and is always reachable. ``context["bulk"]`` (more people in
        24 h than ``mind.floor.bulk_recipients``) makes a message a bulk floor effect for the people this call
        adds (``context["recipients"]``). Read-only tools never reach here.
        """
        args = dict(args or {})
        context = dict(context or {})
        text = json.dumps(args, ensure_ascii=False, sort_keys=True)
        if run == "mind" and not self.enabled:
            why = "autonomy level off" if self.level == "off" else "the mind is off"
            return {"allow": False, "action": "block", "reason": f"{why}; no effects until it is turned on"}
        denied = self.policy.denied(text, dict.fromkeys((tool, read_as(tool, args))))
        if denied:
            return {"allow": False, "action": "block", "reason": denied}
        if tool == "kanban_create" and run == "mind":
            if str(args.get("assignee") or "") != "colonymind-act":
                return {"allow": False, "action": "block", "reason": "mind tasks may only create colonymind-act tasks"}
            if str(args.get("idempotency_key") or "").startswith("mind:"):
                return {"allow": False, "action": "block", "reason": "mind: idempotency keys belong to the sidecar"}
            budget = self.budget_check(kind="task", recipient=None)
            if budget:
                return {"allow": False, "action": "block", "reason": budget}
        effect = effect_of(tool, args, cwd, context=context, declared=self.policy.floor_tools,
                           actions=self.policy.actions, money_hosts=self.policy.money_hosts)
        if effect is not None and effect.problem and tool == ACTION_TOOL:
            return {"allow": False, "action": "block", "reason": effect.problem, "effect": effect.as_dict()}
        if effect is not None and context.get("bulk") and effect.cls in {"message", "unbounded"}:
            people = tuple(sorted(str(item) for item in context.get("recipients") or effect.recipients))
            effect = Effect("bulk_third_party_messaging", tool, target=effect.target, recipients=people,
                            call=effect.call)
        allowed: Dict[str, Any] = {}
        if effect is not None and effect.cls != "message":
            if effect.floor or run == "mind":
                verdict = self.effect_verdict(effect, task=task, run=run)
                if not verdict["allow"]:
                    return verdict
                allowed = verdict
        if recipient_may_contact is not None:
            if recipient_may_contact == "never":
                return {"allow": False, "action": "block", "reason": "the recipient may not be contacted"}
            if run == "mind":
                breaker = self.breaker_state("contact") if recipient_may_contact == "auto" else {"tripped": False}
                decision = decide_table(level=self.level, cls="contact", may_contact=recipient_may_contact,
                                        breaker_tripped=bool(breaker.get("tripped")), enabled=self.enabled)
            else:   # another session's send: the contact's permission, at every level
                decision = {"auto": "act", "ask": "ask"}.get(recipient_may_contact, "ask")
            if decision == "ask":
                permit = self.permit_for(task, effect) if run == "mind" and effect is not None else None
                if permit is not None:
                    return {"allow": True, "action": "allow", "reason": "the owner approved exactly this call",
                            "permit": permit, "effect": effect.as_dict(), "floor": None}
                return {"allow": False, "action": "ask",
                        "reason": "messaging this contact needs the owner's approval"
                        if recipient_may_contact == "ask" else
                        f"{self.level}: a message to this contact is suggested, not sent",
                        **({"effect": effect.as_dict()} if effect is not None else {})}
            if decision != "act":
                return {"allow": False, "action": "block", "reason": f"{self.level}: messaging this contact is {decision}"}
        return allowed or {"allow": True, "action": "allow", "reason": "allowed"}


def ask_expiry(now: datetime, policy: Policy) -> datetime:
    return now + timedelta(hours=policy.ask_expires_hours)


def parse_quiet_hours(value: str) -> Optional[tuple[int, int]]:
    """``"22:00-07:00"`` as minutes of the day; None when unset or malformed."""
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*", value or "")
    if not match:
        return None
    start = int(match.group(1)) * 60 + int(match.group(2))
    end = int(match.group(3)) * 60 + int(match.group(4))
    if not (0 <= start < 1440 and 0 <= end < 1440):
        return None
    return start, end


def in_quiet_hours(local_minute: int, window: Optional[tuple[int, int]]) -> bool:
    if window is None:
        return False
    start, end = window
    if start == end:
        return False
    if start < end:
        return start <= local_minute < end
    return local_minute >= start or local_minute < end


def last_boundary(now: datetime, tz: Any, minute: int) -> datetime:
    """The latest local time of day ``minute`` (minutes after local midnight) at or before ``now``."""
    local = now.astimezone(tz)
    at = time(int(minute) // 60 % 24, int(minute) % 60)
    boundary = datetime.combine(local.date(), at, tzinfo=tz)
    if boundary > local:
        boundary = datetime.combine(local.date() - timedelta(days=1), at, tzinfo=tz)
    return boundary


def boundary_crossed(since: datetime, now: datetime, *, tz: Any, minute: int) -> bool:
    """Whether the local time of day ``minute`` fell in ``(since, now]``.

    The rule for anything nightly: it holds once per night crossed, whatever hour
    the clock started at, never twice for the same night, and once for a machine
    that slept through the hour. ``since`` is the persisted moment of the last run.
    """
    return since < last_boundary(now, tz, minute) <= now


__all__ = [
    "ACTION_TOOL", "ALLOW_KINDS", "ALLOW_LEVELS", "ASK_ALPHABET", "Authority", "Breaker", "Budgets", "CLASSES",
    "DECISIONS", "EFFECT_FREE_TOOLS",
    "EXECUTOR_TOOLS", "EXECUTOR_TOOLSETS", "Effect", "FLOOR_EFFECT", "FLOOR_PATTERNS", "GRANT_CLASSES", "GRANT_USE",
    "Grant", "INTERNAL_SAFE_TOOLSETS", "LEVELS", "MAY_CONTACT", "MESSAGE_TOOLS", "MONEY_HOSTS", "Policy", "Verdict",
    "action_argv", "action_problem", "action_values", "allow_key", "ask_expiry", "boundary_crossed",
    "executor_call",
    "canonical_call", "classify", "decide_table", "declaration_problem", "demote", "effect_of", "finite_amount",
    "floor_class", "grantable", "in_quiet_hours", "last_boundary", "may_contact_of", "money_host", "new_ask_code",
    "parse_grant", "parse_quiet_hours",
]
