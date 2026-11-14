"""The gates' configuration: ``plugins.colonymind-gates`` in the Hermes ``config.yaml``.

```yaml
plugins:
  enabled: [colonymind, colonymind-gates]
  colonymind-gates:
    triage: {enabled: true, chats: []}      # chats: only these chat ids (empty: every chat)
    screen: {enabled: true}
    cron:
      supplier_reply:                       # its script is scripts/colonymind-gate-supplier_reply.py
        sources:                            # what the job reads: files, a loopback URL, a command's output
          - file: inbox/supplier.txt        # relative to the Hermes home
          - url: http://127.0.0.1:8080/feed
          - command: [python3, fetch.py]    # run in the Hermes scripts directory
        script: fetch_supplier.py           # the job's own pre-run script, run first
        point: cron_supplier_reply          # its question in colonymind.yaml (default cron_<job>)
        force_wake_after: 4
```

Every gate is off unless its entry enables it. An entry outside these rules leaves its gate off (triage and the
screen) or wakes its job (the cron gate): a bad configuration never holds anything back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

PLUGIN_ID = "colonymind-gates"
#: The cron gate's own budget for reading the sources and asking (a chosen value, not measured): past it, wake.
CRON_DEADLINE_S = 2.0
#: How long a non-owner's message may wait for triage (my choice), and a tool result for its screen (my choice).
TRIAGE_DEADLINE_S = 0.150
SCREEN_DEADLINE_S = 0.400
#: A job wakes after this many skips in a row whatever its answers say (my choice).
FORCE_WAKE_AFTER = 4
#: The longest state the fast tier reads at once (the local decision contract's limit). A state is never cut to fit
#: it: a longer one skips the tier. So an item or a window is cut short enough that its question's whole state fits,
#: the question's own words around it included.
STATE_CHARS = 1200
#: The words the shipped screen question puts around a window and the tool's name (its template, 'Result of
#: {tool}: "{text}"', in ``colonymind.instinct.external``).
SCREEN_WORDS = len('Result of : ""')
#: The words a cron question puts around an item (its template, 'The job watches for: {watch}\nNew since the last
#: run: "{item}"'), with the longest ``watch`` a cron question may have (400 characters): the gate cannot see it.
CRON_WORDS = len('The job watches for: \nNew since the last run: ""') + 400
#: The longest item the cron gate sends.
ITEM_CHARS = STATE_CHARS - CRON_WORDS
#: At most this many windows of one result are screened (my choice); the rest are counted as unscreened.
MAX_WINDOWS = 8
#: A fire with more new items than this wakes without asking (my choice): too much is new to call it nothing.
MAX_ITEMS = 16
MAX_SOURCES = 8

_JOB = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
_POINT = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
_LOOPBACK = re.compile(r"^https?://(127\.0\.0\.1|localhost|\[::1\])(:\d{1,5})?(/\S*)?$")


class GateConfigError(ValueError):
    """A gate's entry outside the rules; the message names the key."""


@dataclass(frozen=True)
class TriageConfig:
    enabled: bool = False
    chats: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScreenConfig:
    enabled: bool = False


@dataclass(frozen=True)
class CronJob:
    name: str
    point: str
    sources: tuple[dict, ...]
    script: str = ""
    force_wake_after: int = FORCE_WAKE_AFTER


def window_chars(tool: str) -> int:
    """The longest window of a ``tool``'s result the screen sends: what the state leaves beside the question's own
    words and the tool's name, as a model reads it (whitespace folded)."""
    return max(1, STATE_CHARS - SCREEN_WORDS - len(" ".join(str(tool).split())))


def section(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``plugins.colonymind-gates`` from the merged Hermes configuration ({} when absent)."""
    from colonymind_hermes.client import hermes_config
    config = hermes_config() if config is None else config
    plugins = config.get("plugins") if isinstance(config, Mapping) else None
    value = plugins.get(PLUGIN_ID) if isinstance(plugins, Mapping) else None
    return dict(value) if isinstance(value, Mapping) else {}


def _entry(value: Any, where: str, keys: set[str]) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise GateConfigError(f"{where} must be a mapping")
    unknown = sorted(set(value) - keys)
    if unknown:
        raise GateConfigError(f"{where}: unknown key {unknown[0]} (it takes {', '.join(sorted(keys))})")
    enabled = value.get("enabled", False)
    if not isinstance(enabled, bool):
        raise GateConfigError(f"{where}.enabled must be true or false")
    return dict(value)


def triage(value: Mapping[str, Any]) -> TriageConfig:
    """The triage gate's entry. It has no live mode in this release: it asks, and never skips a message."""
    entry = _entry(value.get("triage"), f"plugins.{PLUGIN_ID}.triage", {"enabled", "chats"})
    chats = entry.get("chats") or []
    if not isinstance(chats, list) or not all(isinstance(chat, (str, int)) for chat in chats):
        raise GateConfigError(f"plugins.{PLUGIN_ID}.triage.chats must be a list of chat ids")
    return TriageConfig(enabled=bool(entry.get("enabled", False)), chats=tuple(str(chat) for chat in chats))


def screen(value: Mapping[str, Any]) -> ScreenConfig:
    entry = _entry(value.get("screen"), f"plugins.{PLUGIN_ID}.screen", {"enabled"})
    return ScreenConfig(enabled=bool(entry.get("enabled", False)))


def _source(value: Any, where: str) -> dict:
    if not isinstance(value, Mapping) or len(value) != 1:
        raise GateConfigError(f"{where} must be one of file, url or command")
    (kind, target), = value.items()
    if kind == "file" and isinstance(target, str) and target.strip():
        return {"file": target.strip()}
    if kind == "url" and isinstance(target, str) and _LOOPBACK.match(target.strip()):
        return {"url": target.strip()}
    if kind == "command" and isinstance(target, list) and 1 <= len(target) <= 32 \
            and all(isinstance(part, str) and part for part in target):
        return {"command": list(target)}
    raise GateConfigError(f"{where}: a file path, a loopback http(s) URL, or a command as a list of arguments")


def cron_job(value: Mapping[str, Any], name: str) -> CronJob:
    """One gated job's entry; GateConfigError when it is missing or outside the rules (the job then wakes)."""
    where = f"plugins.{PLUGIN_ID}.cron.{name}"
    if not _JOB.match(name or ""):
        raise GateConfigError(f"{where}: a job's gate name is lower-case letters, digits and underscores")
    jobs = value.get("cron") or {}
    if not isinstance(jobs, Mapping) or not isinstance(jobs.get(name), Mapping):
        raise GateConfigError(f"{where} is not configured")
    entry = dict(jobs[name])
    unknown = sorted(set(entry) - {"sources", "script", "point", "force_wake_after"})
    if unknown:
        raise GateConfigError(f"{where}: unknown key {unknown[0]}")
    sources = entry.get("sources")
    if not isinstance(sources, list) or not 1 <= len(sources) <= MAX_SOURCES:
        raise GateConfigError(f"{where}.sources must list 1 to {MAX_SOURCES} sources")
    point = entry.get("point") or f"cron_{name}"
    if not isinstance(point, str) or not _POINT.match(point):
        raise GateConfigError(f"{where}.point must name an external point")
    script = entry.get("script") or ""
    if not isinstance(script, str):
        raise GateConfigError(f"{where}.script must be a script in the Hermes scripts directory")
    force = entry.get("force_wake_after", FORCE_WAKE_AFTER)
    if isinstance(force, bool) or not isinstance(force, int) or not 1 <= force <= 100:
        raise GateConfigError(f"{where}.force_wake_after must be a whole number from 1 to 100")
    return CronJob(name=name, point=point, script=script, force_wake_after=force,
                   sources=tuple(_source(item, f"{where}.sources[{index}]") for index, item in enumerate(sources)))


__all__ = ["CRON_DEADLINE_S", "CRON_WORDS", "FORCE_WAKE_AFTER", "ITEM_CHARS", "MAX_ITEMS", "MAX_WINDOWS", "PLUGIN_ID",
           "SCREEN_DEADLINE_S", "SCREEN_WORDS", "STATE_CHARS", "TRIAGE_DEADLINE_S", "CronJob", "GateConfigError",
           "ScreenConfig", "TriageConfig", "cron_job", "screen", "section", "triage", "window_chars"]
