"""``colonymind.yaml``: the one configuration file of a ColonyMind instance.

An instance lives in one directory (``$COLONYMIND_HOME``, default ``~/.colonymind``)
that holds ``colonymind.yaml``, ``api.key``, ``identity.yaml`` and the SQLite
stores. ``load_config`` merges the file over the schema defaults, applies a
small set of environment overrides and validates the result. ``apply_environment``
exports the values the sidecar process reads through ``os.environ``.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from colonymind.api.credentials import default_credentials, validate_credentials

logger = logging.getLogger(__name__)
_watch_warnings: set[str] = set()

CONFIG_FILE = "colonymind.yaml"
KEY_FILE = "api.key"
IDENTITY_FILE = "identity.yaml"
LLM_CONFIG_FILE = ".colonymind-llm-config.json"

AUTONOMY_LEVELS = ("off", "suggest", "standard", "trusted")

#: The constitution (architecture 4.2): ``identity.yaml`` ``agent.{name, values, boundaries}`` rendered as
#: one paragraph of at most this many characters. ``colonymind init`` refuses a longer one; the plugin
#: renders it into the session prompt; no other ColonyMind code path writes the file.
CONSTITUTION_CHARS = 1500
CONSTITUTION_ITEMS = 12
CONSTITUTION_ITEM_CHARS = 160

DEFAULTS: dict[str, Any] = {
    "sidecar": {"host": "127.0.0.1", "port": 7777},
    "hermes": {"home": "~/.hermes", "python": ""},
    "watch": {"enabled": True, "hermes_home": "~/.hermes", "stuck_minutes": 30,
              "repeat_failures": 3, "repeat_window_hours": 24, "health_timeout_s": 10,
              "notice_command": [], "downloads": {"url": ""}, "endpoints": [], "tmux": True},
    # ``embed_provider``: ``endpoint`` embeds at ``embed_url``; ``local`` runs the pinned model ``embed_model`` at
    # ``embed_revision`` in the sidecar (``colonymind.vector.local_embedder``), for an endpoint with no embeddings.
    "router": {"base_url": "", "model": "", "embed_provider": "endpoint", "embed_url": "", "embed_model": "",
               "embed_revision": "", "embed_dims": 0, "rerank_url": "", "rerank_model": ""},
    "owner": {"contact_id": "", "sms_handle": "", "rcs_handle": ""},
    # COLONYMIND_* settings the sidecar reads that no key above covers (a reranker prompt
    # style, recall thresholds, oversampling, an endpoint credential): exported after the
    # keys above; a value already in the process environment still wins.
    "environment": {},
    # Staged voice-prefix publication is opt-in; GET itself never builds.
    "context_card": {"enabled": False, "poll_seconds": 5, "reconcile_seconds": 300},
    "mind": {
        "enabled": True,
        "autonomy": "suggest",
        "deny": {"commands": [], "tools": [], "text": []},
        "worker_toolsets": ["web", "file", "session_search", "memory", "todo"],
        "budgets": {
            "tasks_per_hour": 4,
            "concurrent_tasks": 2,
            "owner_messages_per_day": 3,
            "contact_messages_per_day": 5,
            "per_contact_cooldown_hours": 24,
            "llm_tokens_per_day": 200000,
            "learn_share": 0.25,
            "open_goals": 2,
            "goal_tasks": 4,          # steps an agent-owned goal may spend
            "goal_horizon_days": 7,   # the longest horizon an adopted goal may have
            "task_max_runtime_s": 600,   # one worker run, for a task type task_types does not name
            "task_max_retries": 1,
            # Unprompted owner outreach (findings, open loops, care) in any 24 h; requested answers and
            # the reminders the owner asked for are not counted here.
            "outreach_per_day": 3,
            # Research-shaped work reads sources and writes them up in one run: longer, with a retry.
            "task_types": {name: {"max_runtime_s": 1800, "max_retries": 2}
                           for name in ("research", "question", "mastery_investigation", "goal_step",
                                        "outreach_followup")},
        },
        # Extra fields on every model request a mind task's worker makes (the colonymind-act profile's
        # providers carry them as extra_body): an output cap, so one runaway completion cannot hold the
        # model for the run's whole budget. null leaves a field to the provider.
        "worker_request": {"max_tokens": 8192, "top_p": 0.95},
        # The model a mind task's worker runs on: the name of one providers (or custom_providers) entry of
        # the main Hermes config. null follows the main model and its fallbacks.
        "worker_model": None,
        # Language policy belongs to the host; empty validators leave style unscored.
        "drift": {"validators": {}},
        "status_replacements": [],
        "quiet_hours": "22:00-07:00",
        "ask_expires_hours": 72,
        "breaker": {"failures": 3, "window_hours": 24, "demotion_hours": 72},
        "act_threshold": 0.6,   # the ranker's effective-score floor (architecture 3.2)
        "digest_hour": 8,       # local hour after which the daily digest goes out
        "heads_up_grace_minutes": 30,   # after a heads-up went out, the overdue reminder for that row waits this long
        # Drive weights: 0 turns a drive off (architecture 4.5).
        "drives": {"duty": 1.0, "social": 0.5, "curiosity": 0.5, "mastery": 1.0, "upkeep": 1.0},
        # One binary flag per faculty; each is one benchmark arm (architecture 4, evals section 3). A faculty
        # ships on unless a paired measurement showed it harmful ("not measured yet" is never a reason for off);
        # `colonymind report faculties` says what each did and how to switch it off.
        "faculties": {
            "initiative": True,
            "drives": True,         # weights, satiation and goal adoption; off = flat priority
            "deliberation": True,   # the one tool-less call per tick; off = templates only
            "goals": True,          # agent-owned goals
            "people": True,
            "affect": True,
            "affect_rules": False,  # the affect mechanism arm: every consumer reads its stateless rule
            "opinions": True,
            "broadcast": True,
            "semantic_recall": True,
            "consolidation": True,
            "self_narrative": True,
            "lessons": True,
            # Proven lessons written out as skills: off until skills beat lessons alone (their own measured question).
            "skills": False,
            "outreach": True,       # owner outreach: the social drive turned toward the owner (M11)
            "owner_gate": True,     # one gate every unprompted owner message passes (docs/OWNER-GATE.md)
            # The learning rules (memory/learning.py): restated claims and lessons are witnesses, not rows; a
            # model-last novelty gate; preference changes need the owner; the cold tier; reinforcement by
            # useful recall.
            "learning": True,
            "plasticity": True,
            "working_memory": True,
            "work_records": True,  # owner work and deterministic postcondition checks
            "hands": True,          # durable delegated work
            # The mastery drive's nightly wish for a capability it lacks (architecture 4.8 item 5): always an
            # owner question that only the owner's yes grants; its outcomes are measured (``colonymind mind wishes``).
            "wishes": True,
            # The model's own words beside the typed capability: labelled, checked, never what a yes gives.
            "wishes_named": True,
            # The identity battery and drift detection (docs/MIND.md, Identity check).
            "drift_check": True,
            "senses": True,
            # Two more flags are read here when the file sets them, and are on otherwise: the uses of the embedding
            # model beyond recall (``MEANING_FACULTIES``). ``contradictions_by_meaning``: each night a person's
            # statements filed under two names close in meaning are compared too (docs/CONSOLIDATION.md).
            # ``priors_by_meaning``: the claim extractor is also offered earlier records found by meaning
            # (docs/SOURCE-CLAIMS.md).
        },
        # Assumptions: outside observations expire after 2 hours, each source contributes at most 6 a day,
        # and an original image clip is at most 512 KiB. A zero daily cap ignores every observation.
        # ``source_caps`` sets one source's daily cap in place of daily_source_cap; the owner's runtime cap
        # (``PUT /v1/mind/observations/sources/{source}/cap``) stands over both until cleared.
        "observations": {"expiry_hours": 2, "daily_source_cap": 6, "image_max_bytes": 512 * 1024,
                         "source_caps": {"omi": 40}},
        # The owner gate's settings (read only with faculties.owner_gate on): a second message on a topic
        # inside topic_window_hours waits for the digest, the same words inside repeat_window_hours are
        # dropped, a message waits while the owner spoke inside conversation_minutes or was sent one inside
        # spacing_minutes (never longer in all than max_wait_minutes), and under confidence (or at night) the
        # main model gives a second opinion.
        "owner_gate": {"topic_window_hours": 12, "repeat_window_hours": 24, "conversation_minutes": 10,
                       "spacing_minutes": 10, "max_wait_minutes": 60, "confidence": 0.75, "second_opinion": True,
                       "judge_timeout_s": 30},
    },
    # Instinct (``colonymind.instinct``), the one decision mechanism: decision providers by name, the provider each
    # model tier asks, and bars a point's labels may be raised to. No url: no fast tier, and every point keeps its
    # existing path. A point's state (shadow, live) is not configuration: it lives in instinct.db.
    "instinct": {"providers": {"local": {"kind": "local-decision.v1", "url": "", "timeout_ms": 250}},
                 "fast": "local", "reader": "", "points": {}, "external": {}},
    # The model-backed source projections (claim extraction, appraisal, capture, media descriptions):
    # jobs queued a day or more before this release started drain at most this many an hour; new turns
    # are projected at once. 0 leaves that backlog pending.
    "projections": {"backlog_per_hour": 12},
    # The per-turn context. ``status``: the owner's turn carries a status block generated from the running
    # mind (``colonymind.mind.status``): the autonomy level, quiet hours, used budgets, the breaker, the
    # faculties switched off and in-process health problems (on, like every faculty not shown harmful).
    # The text blocks every turn carries (colonymind/memory/blocks.py): a hard limit per block, one global
    # budget admits fresh state first, a marker for anything held back and a fossil guard. Host allowances
    # can shorten the character budget. The owner-adjustable 20-record default matches the independent
    # rerank submission bound; a turn whose rerank is missed still falls back to five.
    "context": {"status": True, "budget_chars": 16000, "recall_items": 20, "fossil_days": 7, "limits": {},
                "recall_notice_replacements": [], "recall_alarm_turns": 2},
    # Who each API credential may write as (``colonymind.api.credentials``, docs/CREDENTIALS.md): the
    # instance key (api.key, the Hermes adapter's) writes the owner's turns from these platforms only.
    "credentials": default_credentials(),
    # Relative to the instance's data directory; the reflector target waits for a lesson.
    "plasticity": {
        "directory": "plasticity", "behaviour": {"restarts": []},
        "learners": {
            "behaviour-tool-failure": {
                "implementation": "behaviour", "kind": "behaviour", "effect": "teach",
                "base": "standing-none@1", "mode": "rebuild",
                "teachers": {"verified_outcome": "binary"},
                "model": {"roles": ["standing_lessons"], "change": "context"},
                "budget": {"type": "ladder", "bound": 3}, "audience": "owner",
                "personal_data": True, "dependents": [],
                "erasure_path": "reads live owner turns; erased turns leave evaluation; certificates hold counts only",
                "gates": ["lesson_current", "standing_cap", "budget", "finite", "served_offline",
                          "baseline_target", "baseline_guards"],
                "target": {"signal": "tool_failure", "rank": 2},
                "lesson": {"source": "reflector"},
                "baseline": {"days": 10, "min_replies": 60, "min_session_days": 12, "target_cap": .03,
                             "guards": {"substitute_rate": .02, "median_length": .15,
                                        "tool_failure_rate": .01, "owner_corrections": .10}},
            },
        },
    },
    # WM3 assumptions: 14 days, 48K tokens, 20 consults/hour; summary_only keeps capture and removes the tool.
    "working_memory": {"retention_days": 14, "token_cap": 48000, "hourly_budget": 20, "summary_only": False},
    # Hands: numeric values are configurable assumptions. Cascade, retention and overdue defaults
    # deliberately leave ordinary use unconstrained until a host supplies measured limits.
    # Host configuration supplies platform admission, route contracts, roles and family-specific limits.
    "hands": {
        "max_running": 1,
        "platforms": [],            # empty allow-list denies every platform
        "contracts": {},
        "matrix": {},
        "persona_addendum": "",
        "style": "",
        "lease_seconds": 90,
        "dispatch_start_seconds": 120,
        "open_tasks": {"own": 6, "mind": 4},   # by origin; owner tasks have no count limit
        "retention": {"events_days": 36500, "rows_days": 36500},
        "brief": {"max_tokens": 6000},
        "digest": {"max_chars": 900},
        "obligations": {"overdue_seconds": 3153600000},
        "checks": {"timeout_seconds": 300},
        "cascade": {"max_per_task": 1000000, "max_per_hour": 1000000},
        "budgets": {
            "attempt": {
                "coding": {"wall_seconds": 1800, "tool_calls": 100, "web_searches": 6, "tokens": 1000000},
                "deep": {"wall_seconds": 1200, "tool_calls": 20, "web_searches": 6, "tokens": 300000},
                "research": {"wall_seconds": 1200, "tool_calls": 40, "web_searches": 12, "tokens": 400000},
                "ops": {"wall_seconds": 900, "tool_calls": 60, "web_searches": 4, "tokens": 300000},
                "media": {"wall_seconds": 600, "tool_calls": 20, "web_searches": 4, "tokens": 200000},
                "plan": {"wall_seconds": 600, "tool_calls": 20, "web_searches": 6, "tokens": 200000},
                "synthesize": {"wall_seconds": 600, "tool_calls": 20, "web_searches": 6, "tokens": 200000},
                "verify": {"wall_seconds": 600, "tool_calls": 20, "web_searches": 6, "tokens": 200000},
                "integrate": {"wall_seconds": 600, "tool_calls": 20, "web_searches": 6, "tokens": 200000},
            },
            "effort_3_multiplier": 2,   # tokens and wall time only
            "wall_warning_fraction": 0.8,
            "no_new_information_warn": 3,
            "no_new_information_stop": 5,
            "repeat_tool_calls": 3,
            # After a token budget stop, the plugin allows one final report call (assumptions).
            "final_report_tokens": 8192,
            "final_report_seconds": 300,
            "task": {"token_multiplier": 3, "wall_seconds": 10800, "max_hands": 6},
            # Daily limits apply to own/auto/upkeep work; owner work is excluded.
            "day": {"max_hands": 40, "default_tokens_per_family": 3000000, "tokens_per_family": {}},
        },
        "queue": {
            "max_queued": 6,
            "max_owner": 4,
            "max_own": 2,
            # Upkeep shares the autonomous deadline (implementation assumption).
            "deadline_seconds": {"owner": 900, "own": 1800, "auto": 3600, "upkeep": 3600},
        },
        "health": {
            "poll_seconds": 5,
            "max_waiting": 0,
            "max_kv_fraction": 0.8,
            "default_family_cap": 1,
            "family_caps": {},
            "running_limits": {},
            "unreadable_owner_cap": 1,
        },
        "stalls": {"generation_multiplier": 2, "tool_grace_seconds": 60},
        "locks": {"max_waiters": 2},
        "spool": {"replay_seconds": 10, "progress_seconds": 10},
        "cooldowns": {
            "task_failures": 2,
            "task_seconds": 3600,
            "task_park_failures": 3,
            "resource_failures": 3,
            "resource_window_seconds": 86400,
            "resource_seconds": 21600,
        },
        "notices": {
            "forbidden_patterns": [],
            "routine_per_day": 30,
            "actionable_per_day": 20,
            "merge_seconds": 60,
            "progress_seconds": 600,
            "overflow_seconds": 3600,
            "queued_actionable_seconds": 300,
            "max_chars": 200,
        },
        "degraded": {"controller_failures": 2, "failure_window_seconds": 30, "max_running": 1},
    },
}

# Faculties read from ``mind.faculties`` like the flags in DEFAULTS, on unless the file sets them to false, and not
# written into a new colonymind.yaml: the two uses of the embedding model beyond recall.
MEANING_FACULTIES = ("contradictions_by_meaning", "priors_by_meaning")

# The whole environment override set. Everything else is configured in the file.
ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "COLONYMIND_SIDECAR_HOST": ("sidecar", "host"),
    "COLONYMIND_SIDECAR_PORT": ("sidecar", "port"),
    "HERMES_HOME": ("hermes", "home"),
    "COLONYMIND_MIND_ENABLED": ("mind", "enabled"),
    "COLONYMIND_AUTONOMY": ("mind", "autonomy"),
    "COLONYMIND_OWNER_SMS_HANDLE": ("owner", "sms_handle"),
    "COLONYMIND_OWNER_RCS_HANDLE": ("owner", "rcs_handle"),
}

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})

ENVIRONMENT_NAME = re.compile(r"^COLONYMIND_[A-Z][A-Z0-9_]*$")

#: Names the other keys of colonymind.yaml (or identity.yaml) already define. The
#: ``environment`` mapping refuses them and names the key, so each setting has one place.
RESERVED_ENVIRONMENT: dict[str, str] = {
    "COLONYMIND_HOME": "the instance directory ($COLONYMIND_HOME)",
    "COLONYMIND_STATE_DIR": "the instance directory ($COLONYMIND_HOME)",
    "COLONYMIND_CONTACTS_DB": "the instance directory ($COLONYMIND_HOME)",
    "COLONYMIND_SIDECAR_HOST": "sidecar.host",
    "COLONYMIND_SIDECAR_PORT": "sidecar.port",
    "COLONYMIND_API_KEY": "api.key",
    "COLONYMIND_MIND_ENABLED": "mind.enabled",
    "COLONYMIND_AUTONOMY": "mind.autonomy",
    "COLONYMIND_OWNER_CONTACT_ID": "owner.contact_id",
    "COLONYMIND_OWNER_SMS_HANDLE": "owner.sms_handle",
    "COLONYMIND_OWNER_RCS_HANDLE": "owner.rcs_handle",
    "COLONYMIND_OWNER_NAME": "identity.yaml owner.name",
    "COLONYMIND_PERSONA_NAME": "identity.yaml agent.name",
    "COLONYMIND_AGENT_VALUES": "identity.yaml agent.values",
    "COLONYMIND_AGENT_TIMEZONE": "identity.yaml agent.timezone",
    "COLONYMIND_TIMEZONE": "identity.yaml agent.timezone",
    "COLONYMIND_AGENT_QUIET_HOURS": "identity.yaml agent.quiet_hours",
    "COLONYMIND_EMBED_PROVIDER": "router.embed_provider and router.embed_url",
    "COLONYMIND_EMBED_BASE_URL": "router.embed_url",
    "COLONYMIND_EMBED_MODEL": "router.embed_model",
    "COLONYMIND_EMBED_DIMS": "router.embed_dims",
    "COLONYMIND_RERANKER_PROVIDER": "router.rerank_url",
    "COLONYMIND_RERANKER_BASE_URL": "router.rerank_url",
    "COLONYMIND_RERANKER_MODEL": "router.rerank_model",
    "COLONYMIND_INSTINCT": "instinct",
    # Retired with the 1.11 decision layer: nothing reads them (the doctor names one still set).
    "COLONYMIND_DECISIONS_URL": "instinct.providers.local.url",
    "COLONYMIND_DECISIONS_TIMEOUT_MS": "instinct.providers.local.timeout_ms",
    "COLONYMIND_DECISIONS_POINTS": "instinct (a point's state lives in instinct.db)",
    "COLONYMIND_CONTEXT_STATUS": "context.status",
    "COLONYMIND_RECALL_EXIT_ALARM_TURNS": "context.recall_alarm_turns",
    "COLONYMIND_LEARNING": "mind.faculties.learning",
    "COLONYMIND_PRIORS_BY_MEANING": "mind.faculties.priors_by_meaning",
}
_SECRET_MARKERS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL")


def looks_secret(name: str) -> bool:
    """A variable whose name says it holds a credential; its value never reaches a log."""
    return any(marker in name.upper() for marker in _SECRET_MARKERS)


def _validate_environment(mapping: Any) -> dict[str, str]:
    if mapping is None:
        return {}
    if not isinstance(mapping, dict):
        raise ConfigError("environment must be a mapping of COLONYMIND_* names to values")
    from colonymind.removed_settings import reject_removed_environment
    try:
        reject_removed_environment(mapping)
    except ValueError as error:
        raise ConfigError(str(error)) from None
    result: dict[str, str] = {}
    for raw_name, value in mapping.items():
        name = str(raw_name)
        if name == "HERMES_HOME":
            raise ConfigError("environment.HERMES_HOME has its own key: set hermes.home instead")
        if not ENVIRONMENT_NAME.match(name):
            raise ConfigError(f"environment.{name}: names must be COLONYMIND_ followed by capitals, digits "
                              "and underscores")
        if name in RESERVED_ENVIRONMENT:
            raise ConfigError(f"environment.{name} has its own key: set {RESERVED_ENVIRONMENT[name]} instead")
        if isinstance(value, bool):
            raise ConfigError(f"environment.{name}: YAML read the value as a boolean; quote it "
                              f"(\"{'on' if value else 'off'}\") so the sidecar receives the word")
        if value is None or isinstance(value, (dict, list, tuple, set)):
            raise ConfigError(f"environment.{name} must be a string or a number")
        text = str(value)
        if not text.strip():
            raise ConfigError(f"environment.{name} is empty; remove the entry")
        if any(ord(char) < 32 for char in text):
            raise ConfigError(f"environment.{name} must not contain control characters")
        result[name] = text
    return result


class ConfigError(ValueError):
    """``colonymind.yaml`` is missing, malformed or holds an invalid value."""


def instance_home(explicit: str | os.PathLike[str] | None = None) -> Path:
    """The instance directory: an explicit path, ``$COLONYMIND_HOME``, else ``~/.colonymind``."""
    selected = explicit or os.environ.get("COLONYMIND_HOME") or os.environ.get("COLONYMIND_STATE_DIR")
    return Path(selected or Path.home() / ".colonymind").expanduser()


def config_path(home: Path | None = None) -> Path:
    return instance_home(home) / CONFIG_FILE


def key_path(home: Path | None = None) -> Path:
    return instance_home(home) / KEY_FILE


def identity_path(home: Path | None = None) -> Path:
    return instance_home(home) / IDENTITY_FILE


def _merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    raise ConfigError(f"{field_name} must be true or false, not {value!r}")


def _require_str_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ConfigError(f"{field_name} must be a list of strings")
    return list(value)


EMBED_PROVIDERS = ("endpoint", "local")
#: What only an embedding endpoint reads under ``environment``, with why the local embedder has no use for it: init
#: drops these when the local embedder replaces an endpoint, and a file that pairs the local embedder with one of the
#: last two is refused when it loads (an in-process provider refuses a requested width, and the local revision has
#: its own key).
ENDPOINT_ENVIRONMENT: dict[str, str] = {
    "COLONYMIND_EMBED_API_KEY": "it was the key for the embedding endpoint",
    "COLONYMIND_EMBED_QUERY_INSTRUCTION": "it was the endpoint model's query instruction (the local model takes "
                                         "the pipeline's own)",
    "COLONYMIND_EMBED_REVISION": "it was the endpoint's weights revision (router.embed_revision pins the local "
                                "model's)",
    "COLONYMIND_EMBED_REQUEST_DIMS": "it was the width asked of the endpoint (an in-process model has its own)",
}
_LOCAL_REFUSED = ("COLONYMIND_EMBED_REVISION", "COLONYMIND_EMBED_REQUEST_DIMS")
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_LOCAL_REMEDY = "'colonymind init --embed-url local' records a pinned one"


def _validate_embed_provider(router: dict[str, Any], environment: dict[str, str]) -> None:
    """``router.embed_provider``: an endpoint (``embed_url``, or none: keyword recall) or the local embedder, whose
    model, commit and width are pinned here and nowhere else."""
    provider = str(router.get("embed_provider") or "endpoint").strip().lower()
    if provider not in EMBED_PROVIDERS:
        raise ConfigError("router.embed_provider must be endpoint or local")
    router["embed_provider"] = provider
    if provider == "endpoint":
        if router["embed_revision"]:
            raise ConfigError("router.embed_revision pins the local embedder's weights (router.embed_provider: "
                              "local); an endpoint's revision is declared as COLONYMIND_EMBED_REVISION under "
                              "environment")
        return
    if router["embed_url"]:
        raise ConfigError("router.embed_url must be empty when router.embed_provider is local (the sidecar runs "
                          "the embedder itself); set router.embed_provider: endpoint to use the endpoint")
    if not _MODEL_ID.match(router["embed_model"]) or ".." in router["embed_model"]:
        raise ConfigError("router.embed_model must name the local embedder's repository (owner/name); "
                          + _LOCAL_REMEDY)
    if not _COMMIT.match(router["embed_revision"]):
        raise ConfigError("router.embed_revision must be the full commit of the local embedder's weights; "
                          + _LOCAL_REMEDY)
    if router["embed_dims"] <= 0:
        raise ConfigError("router.embed_dims must be the local embedder's width; " + _LOCAL_REMEDY)
    for name in _LOCAL_REFUSED:
        if name in environment:
            raise ConfigError(f"environment.{name} is an embedding endpoint's setting, and the local embedder "
                              f"cannot take it ({ENDPOINT_ENVIRONMENT[name]}): remove it from environment, or "
                              "run 'colonymind init --embed-url local', which does")


def _validate_guests(value: Any) -> dict[str, Any]:
    """``guests`` (the operator's opt-ins; docs/HERMES-ADAPTER.md, Owner and guest): ``executor_platforms`` names the
    platforms whose non-owner sessions keep every tool (compared in lower case), ``tools`` the extra tools a non-owner
    session may call anywhere. Each is a list of non-empty names; the plugin reads anything else as nothing."""
    if not isinstance(value, dict):
        raise ConfigError("guests must be a mapping")
    lists = {}
    for key in ("executor_platforms", "tools"):
        names = _require_str_list(value.get(key), field_name=f"guests.{key}")
        if any(not name.strip() for name in names):
            raise ConfigError(f"guests.{key} must be a list of non-empty names")
        lists[key] = [name.strip().lower() if key == "executor_platforms" else name.strip() for name in names]
    return {**value, **lists}


def validate(data: dict[str, Any], *, strict_watch: bool = False) -> dict[str, Any]:
    """Coerce and check every field ColonyMind reads; unknown keys are kept as they are."""
    if not isinstance(data, dict):
        raise ConfigError("colonymind.yaml must be a mapping")
    from colonymind.jobwatch.settings import validate_watch
    try:
        data["watch"] = validate_watch(copy.deepcopy(data.get("watch", DEFAULTS["watch"])))
    except Exception as error:
        from colonymind.health_counters import bump
        # Validation messages may contain values; expose only the known field.
        key = 'watch'
        if isinstance(error, ConfigError):
            field = str(error).split()[0] if str(error) else ''
            if field in {'watch.' + name for name in DEFAULTS['watch']} | {'watch.downloads.url'}:
                key = field
            elif str(error).startswith('watch endpoint'):
                key = 'watch.endpoints'
        if strict_watch:
            raise ConfigError(key + ' is invalid') from None
        bump('context_reads_failed')
        if key not in _watch_warnings:
            _watch_warnings.add(key)
            logger.warning('Ignoring invalid %s (%s) sqlite=%s', key, type(error).__name__,
                           getattr(error, 'sqlite_errorname', None))
        home = data.get('watch', {}).get('hermes_home') if isinstance(data.get('watch'), dict) else None
        data['watch'] = copy.deepcopy(DEFAULTS['watch'])
        if isinstance(home, str) and home.strip():
            data['watch']['hermes_home'] = home
    from colonymind.removed_settings import reject_removed_settings
    from colonymind.hands.settings import validate_limits
    try:
        reject_removed_settings(data)
        validate_limits(data.get("hands"))
    except ValueError as error:
        raise ConfigError(str(error)) from None
    for section in ("sidecar", "hermes", "router", "owner", "mind", "projections", "context"):
        if not isinstance(data.get(section), dict):
            raise ConfigError(f"{section} must be a mapping")
    data["environment"] = _validate_environment(data.get("environment"))
    card = data.get("context_card")
    if not isinstance(card, dict):
        raise ConfigError("context_card must be a mapping")
    card["enabled"] = _parse_bool(card.get("enabled", False), field_name="context_card.enabled")
    for field, default in (("poll_seconds", 5), ("reconcile_seconds", 300)):
        value = card.get(field, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= value <= 3600:
            raise ConfigError(f"context_card.{field} must be a number of seconds between 1 and 3600")
        card[field] = value
    from colonymind.identity.policy import patterns, replacements, validate_validators
    try:
        drift = data["mind"].get("drift", {})
        if not isinstance(drift, dict) or set(drift) - {"validators"}:
            raise ValueError("mind.drift must contain a validators mapping")
        validate_validators(drift.get("validators", {}))
        replacements(data["mind"].get("status_replacements", []))
        replacements(data["context"].get("recall_notice_replacements", []))
        patterns(data.get("hands", {}).get("notices", {}).get("forbidden_patterns", []))
    except (ValueError, TypeError, AttributeError, re.error) as error:
        raise ConfigError(f"invalid host text policy: {error}") from None
    for platform in ("sms", "rcs"):
        field = f"{platform}_handle"
        data["owner"][field] = validate_owner_phone_handle(data["owner"].get(field), platform=platform)
    sidecar = data["sidecar"]
    sidecar["host"] = str(sidecar.get("host") or "127.0.0.1").strip()
    try:
        sidecar["port"] = int(sidecar.get("port") or 7777)
    except (TypeError, ValueError):
        raise ConfigError("sidecar.port must be an integer") from None
    if not 1 <= sidecar["port"] <= 65535:
        raise ConfigError("sidecar.port must be between 1 and 65535")
    hermes = data["hermes"]
    hermes["home"] = str(hermes.get("home") or "~/.hermes")
    hermes["python"] = str(hermes.get("python") or "")
    router = data["router"]
    for key in ("base_url", "model", "embed_url", "embed_model", "embed_revision", "rerank_url", "rerank_model"):
        router[key] = str(router.get(key) or "")
    if router["rerank_url"] and not router["rerank_model"]:
        raise ConfigError("router.rerank_model is required when router.rerank_url is set")
    dims = router.get("embed_dims")
    if dims in (None, ""):
        dims = 0
    if isinstance(dims, bool) or not isinstance(dims, int):
        try:
            dims = int(str(dims).strip())
        except (TypeError, ValueError):
            raise ConfigError("router.embed_dims must be a whole number of dimensions "
                              "(0 learns it from the endpoint's first embedding)") from None
    if dims < 0:
        raise ConfigError("router.embed_dims must not be negative")
    router["embed_dims"] = dims
    _validate_embed_provider(router, data["environment"])
    data["owner"]["contact_id"] = str(data["owner"].get("contact_id") or "")
    if "guests" in data:
        data["guests"] = _validate_guests(data["guests"])
    mind = data["mind"]
    mind["enabled"] = _parse_bool(mind.get("enabled", True), field_name="mind.enabled")
    autonomy = str(mind.get("autonomy") or "suggest").strip().lower()
    if autonomy not in AUTONOMY_LEVELS:
        raise ConfigError(f"mind.autonomy must be one of {', '.join(AUTONOMY_LEVELS)}")
    mind["autonomy"] = autonomy
    deny = mind.get("deny")
    if not isinstance(deny, dict):
        raise ConfigError("mind.deny must be a mapping")
    for key in ("commands", "tools", "text"):
        deny[key] = _require_str_list(deny.get(key), field_name=f"mind.deny.{key}")
    mind["worker_toolsets"] = _require_str_list(mind.get("worker_toolsets"), field_name="mind.worker_toolsets")
    faculties = mind.get("faculties")
    if not isinstance(faculties, dict):
        raise ConfigError("mind.faculties must be a mapping")
    for name, value in list(faculties.items()):
        faculties[name] = _parse_bool(value, field_name=f"mind.faculties.{name}")
    observations = mind.get("observations")
    if not isinstance(observations, dict):
        raise ConfigError("mind.observations must be a mapping")
    expiry = _number(observations.get("expiry_hours"))
    if expiry is None or expiry <= 0:
        raise ConfigError("mind.observations.expiry_hours must be a positive finite number")
    observations["expiry_hours"] = expiry
    cap = observations.get("daily_source_cap")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
        raise ConfigError("mind.observations.daily_source_cap must be a non-negative whole number")
    source_caps = observations.get("source_caps", {})
    if not isinstance(source_caps, dict):
        raise ConfigError("mind.observations.source_caps must be a mapping of source identifiers to daily caps")
    for source, value in source_caps.items():
        if not isinstance(source, str) or not 0 < len(source) <= 128 or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.:/-]*", source):
            raise ConfigError("mind.observations.source_caps keys must be short opaque source identifiers")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(f"mind.observations.source_caps.{source} must be a non-negative whole number")
    image_max = observations.get("image_max_bytes")
    if isinstance(image_max, bool) or not isinstance(image_max, int) or not 0 < image_max <= 512 * 1024:
        raise ConfigError("mind.observations.image_max_bytes must be a whole number from 1 to 524288")
    routes = observations.get("routes", {})
    if not isinstance(routes, dict):
        raise ConfigError("mind.observations.routes must be a mapping of source identifiers to route labels")
    for source, route in routes.items():
        # The same implementation assumption as observation identifiers: at most 128 opaque characters.
        if not isinstance(source, str) or not 0 < len(source) <= 128 or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.:/-]*", source):
            raise ConfigError("mind.observations.routes keys must be short opaque source identifiers")
        if not isinstance(route, str) or route not in {"ignore", "note", "look", "tell"}:
            raise ConfigError("mind.observations.routes values must be ignore, note, look or tell")
    budgets = mind.get("budgets")
    if not isinstance(budgets, dict):
        raise ConfigError("mind.budgets must be a mapping")
    for name, value in list(budgets.items()):
        if name == "task_types":
            budgets[name] = _validate_task_types(value)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"mind.budgets.{name} must be a number")
    mind["worker_request"] = _validate_worker_request(mind.get("worker_request"))
    worker_model = mind.get("worker_model")
    if worker_model is not None and (not isinstance(worker_model, str) or not worker_model.strip()
                                     or len(worker_model) > 128):
        raise ConfigError("mind.worker_model must name one providers or custom_providers entry of the main "
                          "Hermes config, or be null to follow the main model")
    mind["worker_model"] = worker_model.strip() if isinstance(worker_model, str) else None
    mind["owner_gate"] = _validate_owner_gate(mind.get("owner_gate"))
    drives = mind.get("drives")
    if not isinstance(drives, dict):
        raise ConfigError("mind.drives must be a mapping of drive weights")
    for name, value in list(drives.items()):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ConfigError(f"mind.drives.{name} must be a non-negative number (0 turns the drive off)")
        drives[name] = float(value)
    try:
        mind["act_threshold"] = float(mind.get("act_threshold", 0.6))
    except (TypeError, ValueError):
        raise ConfigError("mind.act_threshold must be a number") from None
    if not 0 < mind["act_threshold"] <= 1:
        raise ConfigError("mind.act_threshold must be between 0 and 1")
    try:
        mind["digest_hour"] = int(mind.get("digest_hour", 8))
    except (TypeError, ValueError):
        raise ConfigError("mind.digest_hour must be an hour of the day") from None
    if not 0 <= mind["digest_hour"] <= 23:
        raise ConfigError("mind.digest_hour must be between 0 and 23")
    grace = mind.get("heads_up_grace_minutes", 30)
    if isinstance(grace, bool) or not isinstance(grace, (int, float)) or grace < 0:
        raise ConfigError("mind.heads_up_grace_minutes must be a non-negative number of minutes")
    mind["heads_up_grace_minutes"] = float(grace)
    if "decisions" in data:
        raise ConfigError("the decisions section is retired: run 'colonymind upgrade' to carry it into instinct")
    data["instinct"] = _validate_instinct(data.get("instinct"))
    data["plasticity"] = _validate_plasticity(data.get("plasticity"))
    data["context"] = _validate_context(data.get("context"))
    data["credentials"] = validate_credentials(data.get("credentials"))
    rate = data["projections"].get("backlog_per_hour", 12)
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 <= rate < float("inf"):
        raise ConfigError("projections.backlog_per_hour must be a non-negative number (0 leaves the backlog pending)")
    return data


def validate_behaviour_restarts(value):
    """Host changes are explicit instants, never ambiguous local dates."""
    from datetime import datetime
    if not isinstance(value, list):
        raise ConfigError("plasticity.behaviour.restarts must be a list of timezone-aware ISO-8601 timestamps")
    for at in value:
        try:
            parsed = datetime.fromisoformat(at) if isinstance(at, str) else None
            if parsed is None or parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError
        except (ValueError, TypeError):
            raise ConfigError("plasticity.behaviour.restarts must be a list of timezone-aware ISO-8601 timestamps") from None
    return value


def _validate_plasticity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError("plasticity must be a mapping")
    directory = value.get("directory", "plasticity")
    if (not isinstance(directory, str) or not directory.strip() or Path(directory).is_absolute()
            or ".." in Path(directory).parts or not Path(directory).parts):
        raise ConfigError("plasticity.directory must be a relative subdirectory of the data directory")
    learners = value.get("learners", {})
    if not isinstance(learners, dict):
        raise ConfigError("plasticity.learners must be a mapping")
    behaviour = value.get("behaviour", {})
    if not isinstance(behaviour, dict):
        raise ConfigError("plasticity.behaviour must be a mapping")
    validate_behaviour_restarts(behaviour.get("restarts", []))
    # Each declaration is checked independently at load, so one failure never hides the others.
    return {**value, "directory": directory, "learners": learners}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value == value and abs(value) != float("inf") else None


def _validate_instinct(value: Any) -> dict[str, Any]:
    """``instinct``: providers by name (``kind``, an http(s) ``url`` or empty, ``timeout_ms`` from 1 to 5000,
    ``model``), the provider the ``fast`` and ``reader`` tiers ask, per point ``bars`` over its labels, and the
    operators' ``external`` points (``colonymind.instinct.settings``; each external point is checked by
    ``colonymind.instinct.external``, the Hermes gates' questions)."""
    from colonymind.instinct import external, settings
    try:
        checked = settings.validate(value)
        checked["external"] = external.validate(checked.get("external"))
        return checked
    except (settings.SettingsError, external.ExternalError) as error:
        raise ConfigError(str(error)) from None


def _validate_context(value: Any) -> dict[str, Any]:
    """``context``: the status block, mandatory character and recall caps, fossil age and per-block limits."""
    value = {} if value is None else value
    if not isinstance(value, dict):
        raise ConfigError("context must be a mapping with status, budget_chars, recall_items, fossil_days and limits")
    if "blocks" in value:
        raise ConfigError("context.blocks was removed; context fitting is mandatory")
    result = dict(value)
    result["status"] = _parse_bool(value.get("status", True), field_name="context.status")
    items = value.get("recall_items", 20)
    if isinstance(items, bool) or not isinstance(items, int) or items < 1:
        raise ConfigError("context.recall_items must be a positive whole number of records")
    result["recall_items"] = items
    alarm_turns = value.get("recall_alarm_turns", 2)
    if isinstance(alarm_turns, bool) or not isinstance(alarm_turns, int) or alarm_turns < 1:
        raise ConfigError("context.recall_alarm_turns must be a positive whole number of turns")
    result["recall_alarm_turns"] = alarm_turns
    budget = value.get("budget_chars", 16000)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1000:
        raise ConfigError("context.budget_chars must be a whole number of characters, at least 1000")
    days = value.get("fossil_days", 7)
    if isinstance(days, bool) or not isinstance(days, (int, float)) or not 0 < days < float("inf"):
        raise ConfigError("context.fossil_days must be a positive number of days")
    limits = {} if value.get("limits") is None else value["limits"]
    if not isinstance(limits, dict):
        raise ConfigError("context.limits must be a mapping of section ids to characters")
    for name, item in limits.items():
        if isinstance(item, bool) or not isinstance(item, int) or item < 100:
            raise ConfigError(f"context.limits.{name} must be a whole number of characters, at least 100")
    result["limits"] = {str(name): int(item) for name, item in limits.items()}
    return result


OWNER_GATE_LIMITS = {"topic_window_hours": (0, 720), "repeat_window_hours": (0, 720), "conversation_minutes": (0, 240),
                     "spacing_minutes": (0, 240), "max_wait_minutes": (0, 1440), "confidence": (0, 1),
                     "judge_timeout_s": (1, 600)}


def _validate_owner_gate(value: Any) -> dict:
    """``mind.owner_gate``: known keys only, each number inside its range, ``second_opinion`` a boolean."""
    if not isinstance(value, dict):
        raise ConfigError("mind.owner_gate must be a mapping")
    unknown = sorted(set(value) - set(OWNER_GATE_LIMITS) - {"second_opinion"})
    if unknown:
        raise ConfigError(f"mind.owner_gate has unknown keys: {', '.join(unknown)}")
    checked = dict(value)
    for key, (low, high) in OWNER_GATE_LIMITS.items():
        if key not in checked:
            continue
        number = checked[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not low <= number <= high:
            raise ConfigError(f"mind.owner_gate.{key} must be a number from {low} to {high}")
        checked[key] = float(number)
    if "second_opinion" in checked:
        checked["second_opinion"] = _parse_bool(checked["second_opinion"], field_name="mind.owner_gate.second_opinion")
    return checked


def _validate_task_types(value: Any) -> dict[str, dict[str, int]]:
    """``mind.budgets.task_types``: per task type, a positive ``max_runtime_s`` and ``max_retries`` >= 0."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError("mind.budgets.task_types must be a mapping of task types to {max_runtime_s, max_retries}")
    result: dict[str, dict[str, int]] = {}
    for kind, entry in value.items():
        where = f"mind.budgets.task_types.{kind}"
        if not isinstance(entry, dict):
            raise ConfigError(f"{where} must be a mapping with max_runtime_s and/or max_retries")
        unknown = sorted(set(entry) - {"max_runtime_s", "max_retries"})
        if unknown:
            raise ConfigError(f"{where}: unknown key {unknown[0]} (max_runtime_s or max_retries)")
        runtime, retries = entry.get("max_runtime_s"), entry.get("max_retries")
        if runtime is not None and (isinstance(runtime, bool) or not isinstance(runtime, int) or runtime <= 0):
            raise ConfigError(f"{where}.max_runtime_s must be a positive whole number of seconds")
        if retries is not None and (isinstance(retries, bool) or not isinstance(retries, int) or retries < 0):
            raise ConfigError(f"{where}.max_retries must be a whole number, 0 or more")
        result[str(kind)] = {key: val for key, val in (("max_runtime_s", runtime), ("max_retries", retries))
                             if val is not None}
    return result


def _validate_worker_request(value: Any) -> dict[str, Any]:
    """``mind.worker_request``: extra request fields for the worker profile; null drops a field."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError("mind.worker_request must be a mapping of request fields (max_tokens, top_p, ...)")
    tokens, top_p = value.get("max_tokens"), value.get("top_p")
    if tokens is not None and (isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0):
        raise ConfigError("mind.worker_request.max_tokens must be a positive whole number of tokens, or null")
    if top_p is not None and (isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1):
        raise ConfigError("mind.worker_request.top_p must be a number above 0 and at most 1, or null")
    return dict(value)


def _apply_env_overrides(data: dict[str, Any], environ: dict[str, str]) -> dict[str, Any]:
    for name, path in ENV_OVERRIDES.items():
        raw = environ.get(name)
        if raw is None or (not str(raw).strip() and name not in {"COLONYMIND_OWNER_SMS_HANDLE", "COLONYMIND_OWNER_RCS_HANDLE"}):
            continue
        target = data
        for segment in path[:-1]:
            target = target.setdefault(segment, {})
        target[path[-1]] = raw.strip()
    return data


def validate_owner_phone_handle(value: Any, *, platform: str) -> str:
    """An explicit international phone number, or empty to disable this binding. Never echo the value."""
    if platform not in {"sms", "rcs"}:
        raise ConfigError("unsupported owner phone platform")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ConfigError(f"owner.{platform}_handle must be an E.164 string or empty")
    value = value.strip()
    if value and not re.fullmatch(r"\+[1-9][0-9]{6,14}", value):
        raise ConfigError(f"owner.{platform}_handle must be an E.164 string or empty")
    return value


def owner_phone_handle(platform: str, home: str | os.PathLike[str] | None = None, *,
                       environ: dict[str, str] | None = None) -> str:
    """Read the deployment's phone binding each time, without persisting it or inferring another channel.

    An explicitly empty environment value disables the file setting. Do not export the file value into
    the process environment: editing or removing the local setting must revoke the binding immediately.
    """
    if platform not in {"sms", "rcs"}:
        raise ConfigError("unsupported owner phone platform")
    environ = os.environ if environ is None else environ
    variable = f"COLONYMIND_OWNER_{platform.upper()}_HANDLE"
    if variable in environ:
        return validate_owner_phone_handle(environ[variable], platform=platform)
    path = config_path(instance_home(home))
    data = read_raw(path) if path.is_file() else {}
    owner = data.get("owner")
    return validate_owner_phone_handle(owner.get(f"{platform}_handle") if isinstance(owner, dict) else None,
                                       platform=platform)


@dataclass
class Config:
    """A validated configuration plus the directory it came from."""

    home: Path
    data: dict[str, Any] = field(default_factory=dict)
    exists: bool = False

    def get(self, dotted: str, default: Any = None) -> Any:
        value: Any = self.data
        for segment in dotted.split("."):
            if not isinstance(value, dict) or segment not in value:
                return default
            value = value[segment]
        return value

    @property
    def sidecar_url(self) -> str:
        host = self.get("sidecar.host")
        host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.get('sidecar.port')}"

    @property
    def hermes_home(self) -> Path:
        return Path(self.get("hermes.home")).expanduser()

    @property
    def path(self) -> Path:
        return self.home / CONFIG_FILE


def _yaml_problem(exc) -> str:
    """A YAML error's problem and position, without the source line PyYAML quotes (it can hold a key)."""
    mark = getattr(exc, "problem_mark", None)
    where = f", line {mark.line + 1}, column {mark.column + 1}" if mark is not None else ""
    return f"{getattr(exc, 'problem', None) or type(exc).__name__}{where}"


def read_raw(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc.strerror}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {_yaml_problem(exc)}") from None
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a mapping")
    return loaded


def load_config(home: str | os.PathLike[str] | None = None, *, environ: dict[str, str] | None = None,
                required: bool = False, strict_watch: bool = False) -> Config:
    """Merge defaults, the file and the environment overrides; validate the result.

    With ``required`` a missing file raises ``ConfigError``; otherwise the
    defaults are returned with ``exists=False`` so unconfigured library use
    keeps working.
    """
    environ = dict(os.environ if environ is None else environ)
    from colonymind.removed_settings import reject_removed_environment
    try:
        reject_removed_environment(environ)
    except ValueError as error:
        raise ConfigError(str(error)) from None
    directory = instance_home(home)
    path = directory / CONFIG_FILE
    raw: dict[str, Any] = {}
    exists = path.is_file()
    if exists:
        raw = read_raw(path)
    elif required:
        raise ConfigError(f"{path} does not exist; run 'colonymind init' first")
    from colonymind.removed_settings import ignore_removed_settings
    data = _merge(DEFAULTS, ignore_removed_settings(raw))
    data = _apply_env_overrides(data, environ)
    return Config(home=directory, data=validate(data, strict_watch=strict_watch), exists=exists)


def dump_config(data: dict[str, Any]) -> str:
    from colonymind.removed_settings import reject_removed_settings
    try:
        reject_removed_settings(data)
    except ValueError as error:
        raise ConfigError(str(error)) from None
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)


def _atomic_write(path: Path, content: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_config(data: dict[str, Any], home: str | os.PathLike[str] | None = None) -> Path:
    """Validate and write ``colonymind.yaml``; returns its path."""
    validated = validate(copy.deepcopy(data))
    path = config_path(instance_home(home))
    _atomic_write(path, dump_config(validated).encode("utf-8"), mode=0o600)
    return path


def update_config(changes: dict[str, Any], home: str | os.PathLike[str] | None = None) -> Path | None:
    """Merge ``changes`` into ``colonymind.yaml`` without expanding it with defaults.

    Returns the path written, or None when the instance has no configuration
    file yet (library use). The merged result is validated before writing.
    """
    directory = instance_home(home)
    path = directory / CONFIG_FILE
    from colonymind.removed_settings import reject_removed_settings
    try:
        reject_removed_settings(changes)
    except ValueError as error:
        raise ConfigError(str(error)) from None
    if not path.is_file():
        return None
    raw = _merge(read_raw(path), changes)
    validate(_merge(DEFAULTS, copy.deepcopy(raw)))
    _atomic_write(path, dump_config(raw).encode("utf-8"), mode=0o600)
    return path


def read_api_key(home: str | os.PathLike[str] | None = None, *, environ: dict[str, str] | None = None) -> str | None:
    """The one API key: ``COLONYMIND_API_KEY`` when set, else the first line of ``api.key``."""
    environ = os.environ if environ is None else environ
    explicit = (environ.get("COLONYMIND_API_KEY") or "").strip()
    if explicit:
        return explicit
    path = key_path(instance_home(home))
    try:
        first = path.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError):
        return None
    return first or None


def write_api_key(key: str, home: str | os.PathLike[str] | None = None) -> Path:
    key = key.strip()
    if not key or any(ord(char) < 33 or ord(char) > 126 for char in key):
        raise ConfigError("the API key must be one printable ASCII token")
    path = key_path(instance_home(home))
    _atomic_write(path, (key + "\n").encode("ascii"), mode=0o600)
    return path


def key_file_is_private(path: Path) -> bool:
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return False
    return not mode & 0o077


def load_identity(home: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    path = identity_path(instance_home(home))
    if not path.is_file():
        return {}
    data = read_raw(path)
    owner = data.get("owner") if isinstance(data.get("owner"), dict) else {}
    agent = data.get("agent") if isinstance(data.get("agent"), dict) else {}
    data["owner"], data["agent"] = owner, agent
    return data


def save_identity(data: dict[str, Any], home: str | os.PathLike[str] | None = None) -> Path:
    path = identity_path(instance_home(home))
    _atomic_write(path, dump_config(data).encode("utf-8"), mode=0o600)
    return path


def constitution_list(value: Any) -> list[str]:
    """``agent.values`` or ``agent.boundaries`` as the constitution reads them: at most 12 distinct
    one-line strings of 1 to 160 characters, in the owner's order."""
    if not isinstance(value, (list, tuple)):
        return []
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split())
        if 1 <= len(text) <= CONSTITUTION_ITEM_CHARS and text not in items:
            items.append(text)
        if len(items) >= CONSTITUTION_ITEMS:
            break
    return items


def render_constitution(identity: Any, *, limit: int | None = CONSTITUTION_CHARS) -> str:
    """The constitution as one paragraph: ``You are <name>. Your values: a; b. Your boundaries: x; y.``

    Empty lists are omitted. Over ``limit`` (None: no limit, which ``constitution_length`` uses so init and
    doctor can say how far over a file is; init refuses one) whole values are held back from the last, then
    boundaries, and the text ends saying how many: never a cut mid-item, and the boundaries go last.
    """
    agent = identity.get("agent") if isinstance(identity, dict) else None
    agent = agent if isinstance(agent, dict) else {}
    name = " ".join(str(agent.get("name") or "").split())
    values, boundaries = constitution_list(agent.get("values")), constitution_list(agent.get("boundaries"))
    held = 0
    while True:
        text = " ".join(part for part in (
            f"You are {name}." if name else "", "Your values: " + "; ".join(values) + "." if values else "",
            "Your boundaries: " + "; ".join(boundaries) + "." if boundaries else "",
            f"({held} more item{'' if held == 1 else 's'} held back: over the {limit}-character limit.)" if held else "")
            if part)
        if limit is None or len(text) <= limit or not (values or boundaries):
            return text if limit is None else text[:limit]
        (values or boundaries).pop()
        held += 1


def constitution_length(identity: Any) -> int:
    """The rendered length before clipping."""
    return len(render_constitution(identity, limit=None))


def _embed_provider(config: Config) -> str:
    """The sidecar's embedding provider: the endpoint's client, the local embedder's device, or ``skip``."""
    if config.get("mind.faculties.semantic_recall") is False:
        return "skip"
    if config.get("router.embed_provider") == "local":
        from colonymind.vector import local_embedder
        return local_embedder.device()
    return "openai_api" if config.get("router.embed_url") else "skip"


def apply_environment(config: Config, *, environ: dict[str, str] | None = None) -> dict[str, str]:
    """Export the configuration to the environment names the sidecar reads.

    Values already present in the environment win, so a service unit or a
    shell can still pin one setting. The returned mapping lists what was set.
    """
    target = os.environ if environ is None else environ
    home = config.home
    identity = load_identity(home)
    owner = identity.get("owner", {})
    agent = identity.get("agent", {})
    key = read_api_key(home, environ=target)
    values: dict[str, str] = {
        "COLONYMIND_HOME": str(home),
        "COLONYMIND_STATE_DIR": str(home),
        "COLONYMIND_SIDECAR_HOST": str(config.get("sidecar.host")),
        "COLONYMIND_SIDECAR_PORT": str(config.get("sidecar.port")),
        "HERMES_HOME": str(config.hermes_home),
        # 1.9.0 instances hold contacts.db; new ones use the server's default name.
        "COLONYMIND_CONTACTS_DB": str(home / ("contacts.db" if (home / "contacts.db").exists()
                                             else "colonymind-contacts.db")),
        # Semantic recall is one binary switch: an embedder (an endpoint, or the local one) and the faculty flag
        # on; the ``full-semantic_recall`` arm turns the flag off with the embedder still recorded.
        "COLONYMIND_EMBED_PROVIDER": _embed_provider(config),
        # The claim projection runs outside the mind, so the learning faculty reaches it as an environment flag,
        # and so does the vector lane of its prior records (on unless the file switches it off).
        "COLONYMIND_LEARNING": "1" if config.get("mind.faculties.learning") is True else "0",
        "COLONYMIND_PRIORS_BY_MEANING": "0" if config.get("mind.faculties.priors_by_meaning") is False else "1",
    }
    if key:
        values["COLONYMIND_API_KEY"] = key
    if config.get("owner.contact_id"):
        values["COLONYMIND_OWNER_CONTACT_ID"] = str(config.get("owner.contact_id"))
    if owner.get("name"):
        values["COLONYMIND_OWNER_NAME"] = str(owner["name"])
    if agent.get("name"):
        values["COLONYMIND_PERSONA_NAME"] = str(agent["name"])
    if agent.get("timezone"):
        values["COLONYMIND_AGENT_TIMEZONE"] = str(agent["timezone"])
        values["COLONYMIND_TIMEZONE"] = str(agent["timezone"])
    if agent.get("quiet_hours"):
        values["COLONYMIND_AGENT_QUIET_HOURS"] = str(agent["quiet_hours"])
    if config.get("router.embed_provider") == "local":
        # The pinned directory in the instance, never a name to download: a start needs no network, and the index
        # identity names the exact weights.
        from colonymind.vector import local_embedder
        values.update(local_embedder.environment(home, str(config.get("router.embed_model")),
                                                 str(config.get("router.embed_revision"))))
        values["COLONYMIND_EMBED_DIMS"] = str(int(config.get("router.embed_dims")))
    if config.get("router.embed_url"):
        values["COLONYMIND_EMBED_BASE_URL"] = str(config.get("router.embed_url"))
        if config.get("router.embed_model"):
            values["COLONYMIND_EMBED_MODEL"] = str(config.get("router.embed_model"))
        # An explicit width is validated against every vector; without one the
        # provider learns the width from the endpoint's first embedding.
        if config.get("router.embed_dims"):
            values["COLONYMIND_EMBED_DIMS"] = str(int(config.get("router.embed_dims")))
    if config.get("router.rerank_url"):
        # A reranker endpoint is configured the way the embedding endpoint is:
        # the remote provider, the model it serves, and recall told to use it.
        # The environment still wins, so "shadow" can be pinned to measure first.
        values["COLONYMIND_RERANKER_PROVIDER"] = "openai_api"
        values["COLONYMIND_RERANKER_BASE_URL"] = str(config.get("router.rerank_url"))
        values["COLONYMIND_RERANKER_MODEL"] = str(config.get("router.rerank_model"))
        values["COLONYMIND_RECALL_RERANK"] = "on"
    # Instinct, read by ``colonymind.instinct.shared``: the whole section as JSON (a service unit that pins it pins
    # the whole section).
    values["COLONYMIND_INSTINCT"] = json.dumps(config.get("instinct"), sort_keys=True)
    # The status block, read by ``colonymind.mind.status`` on every assemble.
    values["COLONYMIND_CONTEXT_STATUS"] = "on" if config.get("context.status") else "off"
    values["COLONYMIND_RECALL_EXIT_ALARM_TURNS"] = str(config.get("context.recall_alarm_turns"))
    # The mapping says explicitly what the keys above only imply (validated: COLONYMIND_
    # names, none that a key already owns), so it lands over the derived values and under
    # the process environment.
    mapping = config.get("environment") or {}
    values.update(mapping)
    applied: dict[str, str] = {}
    for name, value in values.items():
        if name in target and str(target[name]).strip():
            continue
        target[name] = value
        applied[name] = value
    exported = [name for name in mapping if name in applied]
    if exported:
        # Names only; a credential's value never reaches a log, and neither does its name.
        named = [name for name in exported if not looks_secret(name)]
        withheld = len(exported) - len(named)
        logger.info("environment from colonymind.yaml: %s%s", ", ".join(named) or "(none named)",
                    f" and {withheld} credential entr{'y' if withheld == 1 else 'ies'} (names withheld)"
                    if withheld else "")
    return applied


# ---------------------------------------------------------------------------
# Plain environment switches for the subsystems that still read one.
# ---------------------------------------------------------------------------

def env_choice(name: str, valid: Iterable[str], fallback: str) -> str:
    """One environment switch with an allowed set; unset or invalid gives the fallback."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return fallback
    value = raw.strip().lower()
    return value if value in set(valid) else fallback


def env_bool(name: str, fallback: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return fallback
    return raw.strip().lower() in _TRUTHY
