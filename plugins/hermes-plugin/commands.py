"""``/mind``: the owner's, read-only in chat, plus ``off``.

Hermes hands a plugin command only its argument string, never the sender, so
``gate`` drops anyone else's ``/mind`` at ``pre_gateway_dispatch``, where the
sender is still known, and every mutation other than the off switch goes
through the CLI or the owner-checked tools (architecture 7.10). ``status``
works on every sidecar version; ``log``, ``why`` and ``asks`` need the
``/v1/mind`` routes. An ask is shown in words, never by its code: the owner
answers it in their own words (docs/MIND.md, Asks).

``gate`` also hears every other message first (``SessionMap.hear``) and reads
the owner's reaction on an ask's own message at once (``Answers.read_reaction``):
a tap that answered an ask takes no model turn.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from . import __version__
from .body import Body, mind_state
from .busy_ingress import seen
from .capture import SessionMap, TurnOutbox
from .client import ColonyMindClient, Settings, SidecarUnavailable
from .authority_diagnostics import authority_counters

USAGE = "usage: /mind status | log | why <id> | asks | off (other changes: the colonymind CLI)"
ROUTES_MISSING = "This sidecar does not serve the mind routes yet; only status and off work here."
ENTRY_KEYS = ("id", "kind", "drive", "cls", "decision", "status", "outcome", "verified", "title", "summary",
              "hermes_ref", "created_at", "expires_at", "at")


def render(value: Any) -> str:
    """Chat text for a mind route answer: its ``text`` when given, else one line per entry."""
    if isinstance(value, dict) and isinstance(value.get("text"), str):
        return value["text"]
    entries = value
    if isinstance(value, dict):
        entries = next((value[key] for key in ("entries", "asks", "items", "log") if isinstance(value.get(key), list)),
                       None)
    if isinstance(entries, list):
        if not entries:
            return "(nothing)"
        lines = []
        for entry in entries:
            if isinstance(entry, dict):
                lines.append("  ".join(f"{key}={entry[key]}" for key in ENTRY_KEYS if entry.get(key) not in (None, "")))
            else:
                lines.append(str(entry))
        return "\n".join(lines)
    if isinstance(value, dict):
        return "\n".join(f"{key}: {json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else item}"
                         for key, item in value.items())
    return str(value)


def status_text(client: ColonyMindClient, settings: Settings, outbox: TurnOutbox, body: Body | None = None) -> str:
    mind = settings.mind()
    health = client.health()
    lines = [
        f"ColonyMind adapter {__version__}",
        f"sidecar: {settings.sidecar_url} ({'reachable' if health else 'unreachable'})",
        f"mind: {'on' if mind.get('enabled', True) is not False else 'off'}, "
        f"autonomy {mind.get('autonomy', 'standard')}",
        f"worker profile: {settings.worker_profile}",
        f"authority: {authority_counters()['ancestry_unreadable']} ancestry_unreadable",
    ]
    try:
        refused = outbox.refused_count()
        lines.append(f"turn outbox: {outbox.pending_count()} pending"
                     + (f", {refused} not captured: refused by the sidecar's credential grants" if refused else ""))
    except Exception as error:
        lines.append(f"turn outbox: unavailable ({type(error).__name__})")
    detail = mind_state(client)
    if detail:
        for key in ("enabled", "autonomy", "level", "queued", "asks", "last_tick", "last_tick_at", "breaker"):
            if key in detail:
                value = detail[key]
                if key == "asks" and isinstance(value, list):
                    value = f"{len(value)} waiting for your OK" if value else "none"
                lines.append(f"{key}: {value}")
    if body is not None:
        beat = body.heartbeat()
        lines.append(f"body: {beat['ticks']} ticks, {beat['mind_ticks']} mind ticks, last pull {beat['last_pull_at'] or 'never'}"
                     + (" (stale)" if beat.get("stale") else ""))
    return "\n".join(lines)


def _forward(client: ColonyMindClient, path: str, params: dict[str, Any] | None = None) -> str:
    if client.has_mind_routes() is not True:
        return ROUTES_MISSING
    try:
        response = client.get(path, params=params, timeout=5)
    except SidecarUnavailable:
        return "The sidecar is unreachable."
    if response.status_code == 404:
        return ROUTES_MISSING if not response.text else "Not found."
    if not response.is_success:
        return f"The sidecar answered HTTP {response.status_code}."
    try:
        return render(response.json())
    except ValueError:
        return response.text


def asks_text(client: ColonyMindClient) -> str:
    if client.has_mind_routes() is not True:
        return ROUTES_MISSING
    state = mind_state(client)
    if state is None:
        return "The sidecar is unreachable."
    asks = [ask for ask in state.get("asks") or [] if isinstance(ask, dict)]
    if not asks:
        return "No open asks."
    lines = ["Waiting for your OK (answer in your own words where it was asked):"]
    for number, ask in enumerate(asks, 1):
        line = f"{number}. {' '.join(str(ask.get('title') or 'an ask').split())}"
        lines.append(line + (f' (the message: "{ask["message"]}")' if ask.get("message") else ""))
    return "\n".join(lines)


MIND_COMMAND = re.compile(r"^\s*/mind(?:@\S+)?(?:\s|$)", re.IGNORECASE)


def gate(sessions: SessionMap, answers: Any = None) -> Callable[..., dict[str, str] | None]:
    """``pre_gateway_dispatch``: drop a ``/mind`` whose sender is not the owner; hear any other message first
    (``SessionMap.hear``: its own evidence, before Hermes adds to it). The owner's reaction on an ask's own message
    is read now (``Answers.read_reaction``) and, when it answered an ask, dropped: a tap takes no model turn."""
    def pre_gateway_dispatch(event: Any = None, **_: Any) -> dict[str, str] | None:
        if seen(event):   # heard on the busy path already; the idle path hands a late arrival over again
            return None
        if not MIND_COMMAND.match(str(getattr(event, "text", "") or "")):
            try:
                heard = sessions.hear(event)
            except Exception:  # hearing never disturbs dispatch; the turn is then read without its evidence
                return None
            if heard is not None and heard.reacts and answers is not None:
                try:
                    if answers.read_reaction(heard):
                        return {"action": "skip", "reason": "colonymind: a reaction that answered an ask"}
                except Exception:  # a reaction not read goes on as the conversation it is
                    pass
            return None
        source = getattr(event, "source", None)
        platform = getattr(source, "platform", "")
        platform = str(getattr(platform, "value", platform) or "")
        if sessions.sender_is_owner(platform, str(getattr(source, "user_id", "") or "")):
            return None
        return {"action": "skip", "reason": "colonymind: /mind is the owner's"}
    return pre_gateway_dispatch


def handler(client: ColonyMindClient, settings: Settings, outbox: TurnOutbox,
            body: Body) -> Callable[[str], str]:
    def mind(raw_args: str = "") -> str:
        words = str(raw_args or "").split()
        verb = words[0].lower() if words else "status"
        if verb == "status":
            return status_text(client, settings, outbox, body)
        if verb == "log":
            return _forward(client, "/v1/mind/log", {"limit": 20})
        if verb == "asks":
            return asks_text(client)
        if verb == "why":
            return _forward(client, f"/v1/mind/why/{words[1]}") if len(words) > 1 else USAGE
        if verb == "off":
            notes = []
            if client.has_mind_routes() is True:
                try:
                    response = client.post("/v1/mind/off", timeout=5)
                    notes.append("sidecar: off" if response.is_success else f"sidecar: HTTP {response.status_code}")
                except SidecarUnavailable:
                    notes.append("sidecar: unreachable (turn it off again once it is back)")
            else:
                notes.append("sidecar: no mind routes in this version")
            cleanup = body.off_cleanup()
            notes.append(f"archived {len(cleanup.get('archived', []))} unstarted mind task(s)")
            if cleanup.get("error"):
                notes.append(f"kanban: {cleanup['error']}")
            return "Mind off. " + "; ".join(notes) + ". Turning it back on needs the colonymind CLI."
        return USAGE
    return mind


__all__ = ["MIND_COMMAND", "ROUTES_MISSING", "USAGE", "asks_text", "gate", "handler", "render", "status_text"]
