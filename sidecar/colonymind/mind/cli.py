"""``colonymind mind``: status, the audit log, asks, verdicts, the level, the off switch and the opinions.

The CLI talks to the running sidecar with the instance key; ``off`` and
``level`` also work when the sidecar is down, by writing the marker file and
``colonymind.yaml`` directly. It is the operator's console, and anything that
can run a command on this machine (an agent's shell included) can run it: so
it narrows (``no``, ``off``, a revoke, a lower level) and never approves. A
yes to an ask and a new standing grant are the owner's own answer in their
chat; ``show`` sends an ask there again, and ``grants add`` prints the
configuration entry an operator may add instead (docs/MIND.md, Asks).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

import yaml

from colonymind.config import load_config, read_api_key, update_config

from .audit import render_log, render_stats
from .authority import CLASSES, LEVELS, parse_grant
from .outcomes import VERDICTS
from .tick import CONSOLIDATION_WAIT_S, OFF_MARKER
from colonymind.util.temporal import now_utc

COMMANDS = ("status", "log", "why", "asks", "show", "yes", "no", "rate", "level", "reset", "off", "on", "tick", "stats",
            "concerns", "goals", "interest", "outreach", "consolidate", "narrative", "opinions", "lessons", "grants",
            "allows", "learning", "wishes", "drift", "asks-report")
GRANT_ACTIONS = ("list", "add", "revoke")
ALLOW_ACTIONS = ("list", "revoke")
OPINION_ACTIONS = ("list", "show", "withdraw", "reconsider")
LESSON_ACTIONS = ("list", "show", "retire", "relay")
LEARNING_ACTIONS = ("show", "log", "rejected", "pending", "ratify", "decline")


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("mind", help="The mind: status, audit log, asks, verdicts, level, off switch")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    commands = parser.add_subparsers(dest="mind_command")
    commands.add_parser("status", help="Enabled, level, queues, breaker, budgets")
    log = commands.add_parser("log", help="The audit log, newest first")
    log.add_argument("--limit", type=int, default=20)
    log.add_argument("--status", default=None, help="Comma-separated statuses")
    log.add_argument("--kind", default=None, help="Comma-separated kinds")
    why = commands.add_parser("why", help="One intention: drive, evidence, decision, Hermes ref, outcome")
    why.add_argument("id")
    commands.add_parser("asks", help="Open asks with their codes")
    report = commands.add_parser("asks-report", help="How asks went over a window: formed, first seen, settled by "
                                                     "whom and how, how fast, held work (counts and times only)")
    report.add_argument("--since", default="7d", help="A span back from now (30m, 24h, 7d, 2w) or a past date "
                                                      "(2026-09-01); default 7d")
    report.add_argument("--json", dest="report_json", action="store_true", help="Machine-readable output")
    show = commands.add_parser("show", help="Send an open ask to the owner's chat again, where their answer counts")
    show.add_argument("code", help="The ask's code (from 'colonymind mind asks') or its id")
    yes = commands.add_parser("yes", help="Approving is the owner's, in their own chat: says how (see show)")
    yes.add_argument("code")
    no = commands.add_parser("no", help="Decline an ask (recorded as the operator's, not the owner's)")
    no.add_argument("code")
    rate = commands.add_parser("rate", help="Give a verdict on an intention")
    rate.add_argument("id")
    rate.add_argument("verdict", choices=VERDICTS)
    level = commands.add_parser("level", help="Show or set the autonomy level")
    level.add_argument("autonomy", nargs="?", choices=LEVELS)
    reset = commands.add_parser("reset", help="Reset a tripped breaker")
    reset.add_argument("cls", choices=[cls for cls in CLASSES if cls != "floor"])
    off = commands.add_parser("off", help="No further effects (works with the model endpoint down)")
    off.add_argument("--reason", default="owner")
    commands.add_parser("on", help="Turn the mind back on")
    commands.add_parser("tick", help="Run one tick now")
    stats = commands.add_parser("stats", help="The in-vivo panel over the audit log")
    stats.add_argument("--days", type=int, default=7)
    concerns = commands.add_parser("concerns", help="What is on the mind: open concerns and the broadcast set")
    concerns.add_argument("--limit", type=int, default=24)
    commands.add_parser("goals", help="The agent-owned goals that are open")
    commands.add_parser("wishes", help="The mastery drive's wishes: evidence, the owner's word, measured outcome")
    interest = commands.add_parser("interest", help="Seed an interest for the curiosity drive")
    interest.add_argument("topic")
    interest.add_argument("--why", default="")
    outreach = commands.add_parser("outreach", help="Owner outreach: status, off (a pause until turned on) or on")
    outreach.add_argument("state", nargs="?", choices=("status", "on", "off"), default="status")
    commands.add_parser("consolidate", help="Run the nightly consolidation now: the self-narrative delta, "
                                            "lessons, a wish, contradictions, per-contact digests, episode "
                                            "summaries")
    commands.add_parser("narrative", help="The self-narrative as the prompt section renders it")
    opinions = commands.add_parser("opinions", help="The agent's opinions: list, show <id>, withdraw or reconsider <id>")
    opinions.add_argument("action", nargs="?", default="list", choices=OPINION_ACTIONS)
    opinions.add_argument("id", nargs="?", type=int)
    opinions.add_argument("--query", default="", help="list: only the opinions relevant to this text")
    opinions.add_argument("--history", action="store_true", help="list: every revision, not only the current views")
    opinions.add_argument("--reason", default="", help="withdraw/reconsider: why (required)")
    grants = commands.add_parser("grants", help="Standing grants: list, revoke <id>, add (prints the mind.grants "
                                                "entry to configure: a grant is the owner's to give in their chat)")
    grants.add_argument("action", nargs="?", default="list", choices=GRANT_ACTIONS)
    grants.add_argument("id", nargs="?")
    allows = commands.add_parser("allows", help="Standing allows (the owner's \"always\" to an ask): list, revoke <id> "
                                                "(a narrowing: the asking comes back)")
    allows.add_argument("action", nargs="?", default="list", choices=ALLOW_ACTIONS)
    allows.add_argument("id", nargs="?")
    grants.add_argument("--class", dest="grant_class", default=None, help="money_movement | irreversible_deletion | "
                        "credential_change | bulk_third_party_messaging")
    grants.add_argument("--scope", default="", help="the words shown in asks and state")
    grants.add_argument("--tools", default="", help="comma-separated structured tools it covers: a declared action "
                        "or tool, a messaging tool, outbox, cronjob_manage, skill_manage, memory, a vault tool (never a "
                        "shell, code, browser or delegation tool)")
    grants.add_argument("--match", action="append", default=[], help="an exact target it covers (repeatable)")
    grants.add_argument("--max-amount", type=float, default=None)
    grants.add_argument("--currency", default=None)
    grants.add_argument("--per-day", type=int, default=None)
    grants.add_argument("--per-month-amount", type=float, default=None)
    grants.add_argument("--expires", default=None, help="ISO date after which the grant is inert")
    drift = commands.add_parser("drift", help="The identity check: the last run, the trend, the owner agreement")
    drift.add_argument("--run", action="store_true", help="Run one check now (the battery, the probe, the signals)")
    lessons = commands.add_parser("lessons", help="What the mind learned: list, show <id>, retire <id>, relay (an "
                                                  "instruction the owner gave you through another channel)")
    lessons.add_argument("action", nargs="?", default="list", choices=LESSON_ACTIONS)
    lessons.add_argument("id", nargs="?")
    lessons.add_argument("--all", action="store_true", help="list: also superseded and retired lessons")
    lessons.add_argument("--reason", default="", help="retire: why (required)")
    lessons.add_argument("--kind", choices=("strategy", "pitfall"), help="relay: what to do, or what to avoid")
    lessons.add_argument("--title", default="", help="relay: the instruction in a few words")
    lessons.add_argument("--when", default="", help="relay: when it applies")
    lessons.add_argument("--content", default="", help="relay: the instruction itself")
    lessons.add_argument("--said-at", default="", help="relay: when the owner gave it (an ISO date or time)")
    lessons.add_argument("--note", default="", help="relay: how it reached you (the evidence, in your words)")
    lessons.add_argument("--topic", default=None, help="relay: a topic that names it (default: its title); a later "
                                                       "relay on the same topic replaces it")
    learning = commands.add_parser("learning", help="The learning rules: show, log, rejected, pending, "
                                                    "ratify <id>, decline <id>")
    learning.add_argument("action", nargs="?", default="show", choices=LEARNING_ACTIONS)
    learning.add_argument("id", nargs="?")
    learning.add_argument("--store", choices=("claim", "lesson"), help="log/rejected: one store")
    learning.add_argument("--limit", type=int, default=30)
    learning.add_argument("--reason", default="", help="ratify/decline: why")


class Sidecar:
    def __init__(self, home: Optional[str]) -> None:
        self.cfg = load_config(home)
        self.key = read_api_key(self.cfg.home) or ""
        self.url = self.cfg.sidecar_url

    def call(self, method: str, path: str, *, json_body: Any = None, params: Dict[str, Any] | None = None,
             timeout: float = 30.0) -> Any:
        import httpx
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        try:
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                response = client.request(method, self.url + path, headers=headers, json=json_body, params=params)
        except httpx.HTTPError as exc:
            raise SystemExit(f"sidecar unreachable at {self.url} ({type(exc).__name__})") from None
        if response.status_code == 404 and path.startswith("/v1/mind"):
            detail = response.json().get("detail") if response.headers.get("content-type", "").startswith("application/json") else None
            if isinstance(detail, dict):
                raise SystemExit(f"{detail.get('message') or detail.get('code')}")
            raise SystemExit("this sidecar does not serve the mind routes; upgrade and restart it")
        if not response.is_success:
            detail: Any = response.text
            try:
                detail = response.json().get("detail", detail)
            except ValueError:
                pass
            message = detail.get("message") or detail.get("code") if isinstance(detail, dict) else detail
            raise SystemExit(f"sidecar HTTP {response.status_code}: {message}")
        return response.json()


def _emit(value: Any, *, as_json: bool, text: Optional[str] = None) -> None:
    if as_json:
        print(json.dumps(value, indent=2, sort_keys=True))
    elif text is not None:
        print(text)
    else:
        print(json.dumps(value, indent=2, sort_keys=True))


def run(args: argparse.Namespace) -> int:
    command = getattr(args, "mind_command", None) or "status"
    home = getattr(args, "instance", None)
    as_json = bool(getattr(args, "json", False))
    sidecar = Sidecar(home)
    try:
        if command == "status":
            value = sidecar.call("GET", "/v1/mind/state")
            lines = [f"mind: {'on' if value.get('enabled') else 'off'}"
                     + (f" ({value.get('off_reason')})" if value.get("off_reason") else ""),
                     f"autonomy: {value.get('autonomy')}",
                     f"queued tasks: {value.get('queued')}, outbox: {value.get('outbox')}, "
                     f"dispatched: {value.get('dispatched')}, open asks: "
                     f"{len(value['asks']) if isinstance(value.get('asks'), list) else value.get('asks')}",
                     f"ticks: {value.get('ticks')}, last tick: {value.get('last_tick')}, "
                     f"last body pull: {value.get('last_pull')}"
                     + (" (body stale)" if value.get("body_stale") else ""),
                     "breaker: " + ", ".join(f"{b['cls']}={'tripped' if b['tripped'] else 'ok'}"
                                             for b in value.get("breaker", []))]
            if isinstance(value.get("affect"), dict):
                lines.append(_affect_line(value["affect"]))
            _emit(value, as_json=as_json, text="\n".join(lines))
        elif command == "log":
            value = sidecar.call("GET", "/v1/mind/log", params={k: v for k, v in {
                "limit": args.limit, "status": args.status, "kind": args.kind}.items() if v is not None})
            _emit(value["entries"], as_json=as_json, text=render_log(value["entries"]))
        elif command == "why":
            value = sidecar.call("GET", f"/v1/mind/why/{args.id}")
            _emit(value, as_json=as_json, text=value.get("sentence"))
        elif command == "asks":
            value = sidecar.call("GET", "/v1/mind/asks")
            _emit(value["asks"], as_json=as_json, text=value.get("text"))
        elif command == "asks-report":
            from colonymind.reports.faculties import window
            try:
                start, end, _ = window(args.since, now=now_utc())
            except ValueError as error:
                raise SystemExit(f"colonymind mind asks-report: {error}") from None
            hours = round((end - start).total_seconds() / 3600.0, 4)
            value = sidecar.call("GET", "/v1/mind/asks/stats", params={"since_hours": hours})
            _emit(value, as_json=as_json or bool(getattr(args, "report_json", False)), text=value.get("text"))
        elif command == "yes":
            print(yes_refusal(args.code))
            return 1
        elif command == "show":
            value = sidecar.call("POST", f"/v1/mind/asks/{quote(args.code.strip(), safe='')}/show")
            _emit(value, as_json=as_json, text=value.get("text"))
        elif command == "no":
            value = sidecar.call("POST", f"/v1/mind/asks/{quote(args.code.strip().upper(), safe='')}/no")
            _emit(value, as_json=as_json, text=value.get("text"))
        elif command == "rate":
            value = sidecar.call("POST", "/v1/mind/rate", json_body={"id": args.id, "verdict": args.verdict})
            _emit(value, as_json=as_json, text=f"{value.get('title')}: verdict {value.get('verdict')}")
        elif command == "level":
            if not args.autonomy:
                value = sidecar.call("GET", "/v1/mind/state")
                _emit({"autonomy": value.get("autonomy")}, as_json=as_json, text=f"autonomy: {value.get('autonomy')}")
            else:
                update_config({"mind": {"autonomy": args.autonomy}}, home)
                try:
                    value = sidecar.call("POST", "/v1/mind/level", json_body={"autonomy": args.autonomy})
                except SystemExit as exc:
                    value = {"autonomy": args.autonomy, "note": f"written to colonymind.yaml; sidecar not updated ({exc})"}
                _emit(value, as_json=as_json, text=f"autonomy: {value.get('autonomy')}"
                      + (f" ({value['note']})" if value.get("note") else ""))
        elif command == "reset":
            value = sidecar.call("POST", "/v1/mind/reset", json_body={"cls": args.cls})
            _emit(value, as_json=as_json, text=f"breaker {args.cls}: {'tripped' if value.get('tripped') else 'reset'}")
        elif command == "off":
            try:
                value = sidecar.call("POST", "/v1/mind/off", json_body={"reason": args.reason, "by": "cli"})
                note = f"{value.get('cancelled_messages', 0)} unsent message(s) cancelled"
            except SystemExit as exc:
                marker = _state_dir(sidecar) / OFF_MARKER
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(f"{args.reason} at {now_utc().isoformat()}\n", encoding="utf-8")
                value = {"enabled": False, "reason": args.reason, "note": f"sidecar unreachable ({exc}); marker written"}
                note = value["note"]
            _emit(value, as_json=as_json, text=f"mind off ({note}). Unstarted mind tasks are archived at the next "
                                               "plugin tick; running workers end at their runtime limit.")
        elif command == "on":
            update_config({"mind": {"enabled": True}}, home)
            try:
                value = sidecar.call("POST", "/v1/mind/on", json_body={"by": "cli"})
            except SystemExit as exc:
                (_state_dir(sidecar) / OFF_MARKER).unlink(missing_ok=True)
                value = {"enabled": True, "note": f"sidecar unreachable ({exc}); marker removed"}
            _emit(value, as_json=as_json, text="mind on" + (f" ({value['note']})" if value.get("note") else ""))
        elif command == "tick":
            value = sidecar.call("POST", "/v1/mind/tick", timeout=CONSOLIDATION_WAIT_S + 120)
            formed = ", ".join(f"{item['type']}={item['decision']}" for item in value.get("formed", [])) or "nothing"
            _emit(value, as_json=as_json, text=f"tick {value.get('tick')}: formed {formed}"
                  + (f"; skipped: {value['skipped']}" if value.get("skipped") else ""))
        elif command == "stats":
            value = sidecar.call("GET", "/v1/mind/stats")
            _emit(value, as_json=as_json, text=render_stats({k: v for k, v in value.items() if k != "text"}))
        elif command == "concerns":
            value = sidecar.call("GET", "/v1/mind/concerns", params={"limit": args.limit})
            drives = ", ".join(f"{name}={item['effective']:g} (level {item['level']:g})"
                               for name, item in (value.get("drives") or {}).items())
            _emit(value, as_json=as_json, text=f"drives: {drives}\n{value.get('text')}")
        elif command == "goals":
            value = sidecar.call("GET", "/v1/mind/goals")
            _emit(value, as_json=as_json, text=value.get("text"))
        elif command == "wishes":
            value = sidecar.call("GET", "/v1/mind/wishes")
            _emit(value, as_json=as_json, text=value.get("text"))
        elif command == "interest":
            value = sidecar.call("POST", "/v1/mind/interests", json_body={"topic": args.topic, "why": args.why, "by": "cli"})
            _emit(value, as_json=as_json, text=f"interest: {value.get('topic')} (weight {value.get('weight')})")
        elif command == "outreach":
            if args.state == "status":
                value = sidecar.call("GET", "/v1/mind/state").get("outreach") or {"enabled": False}
            else:
                value = sidecar.call("POST", "/v1/mind/outreach", json_body={"state": args.state})
            _emit(value, as_json=as_json, text=_outreach_line(value))
        elif command == "consolidate":
            value = sidecar.call("POST", "/v1/mind/consolidate", timeout=960)
            counts = ", ".join(f"{k}={v}" for k, v in sorted((value.get("counts") or {}).items())) or "nothing to consolidate"
            text = (f"consolidation {value.get('local_date')}: {value.get('calls', 0)} call(s), "
                    f"{value.get('tokens', 0)} tokens; {counts}")
            if value.get("skipped"):
                text = f"consolidation {value.get('local_date')}: skipped ({value['skipped']})"
            elif value.get("errors"):
                text += "; errors: " + ", ".join(str(item) for item in value["errors"])
            _emit(value, as_json=as_json, text=text)
        elif command == "narrative":
            value = sidecar.call("GET", "/v1/mind/narrative")
            text = value.get("text") or ("(self-narrative off)" if not value.get("enabled") else "(nothing recorded yet)")
            _emit(value, as_json=as_json, text=text)
        elif command == "drift":
            if getattr(args, "run", False):
                value = sidecar.call("POST", "/v1/mind/drift/run", timeout=180)
                last = {"verdict": value.get("verdict"), "trigger": value.get("trigger"), "score": value.get("score"),
                        "moved": value.get("moved")}
            else:
                value = sidecar.call("GET", "/v1/mind/drift")
                last = value.get("last") or {}
            _emit(value, as_json=as_json, text=_drift_line(last))
        elif command == "opinions":
            return _opinions(sidecar, args, as_json=as_json)
        elif command == "lessons":
            return _lessons(sidecar, args, as_json=as_json)
        elif command == "grants":
            return _grants(sidecar, args, as_json=as_json)
        elif command == "allows":
            return _allows(sidecar, args, as_json=as_json)
        elif command == "learning":
            return _learning(sidecar, args, as_json=as_json)
        else:
            print(f"unknown mind command {command}", file=sys.stderr)
            return 2
    except SystemExit as exc:
        if exc.code and not isinstance(exc.code, int):
            print(exc.code, file=sys.stderr)
            return 1
        raise
    return 0


def _drift_line(last: Dict[str, Any]) -> str:
    if not last or not last.get("verdict"):
        return "identity check: no run yet"
    score = last.get("score")
    moved = "; ".join(str(item) for item in last.get("moved") or []) or "nothing moved"
    return (f"identity check ({last.get('trigger')}): {last.get('verdict')}, battery "
            f"{'unscored' if score is None else format(score, '.2f')}; {moved}")


def _outreach_line(value: Dict[str, Any]) -> str:
    """``outreach: on; 1 of 3 today[; paused until <when>][; muted: <topics>]``, or off."""
    if not value.get("enabled"):
        return "outreach: off (mind.faculties.outreach)"
    parts = [f"outreach: on; {value.get('sent_24h', 0)} of {value.get('per_day', 0)} today"]
    paused = value.get("paused_until")
    if paused:
        parts.append("paused until you turn it on" if paused == "indefinite" else f"paused until {paused}")
    if value.get("muted"):
        parts.append("muted: " + ", ".join(value["muted"]))
    return "; ".join(parts)


def _affect_line(affect: Dict[str, Any]) -> str:
    """``affect: <tone or calm>; load <x>[, overloaded][, satiated]; switch: <topics or none> [<source>]``."""
    if not affect.get("enabled"):
        return "affect: off"
    load = affect.get("load") if isinstance(affect.get("load"), dict) else {}
    flags = "".join([", overloaded" if load.get("overloaded") else "", ", satiated" if affect.get("satiated") else ""])
    switch = ", ".join(str(topic) for topic in affect.get("switch") or []) or "none"
    return (f"affect: {affect.get('line') or 'calm'}; load {float(load.get('level') or 0.0):g}{flags}; "
            f"switch: {switch} [{affect.get('source')}]")


def _opinion_line(row: Dict[str, Any]) -> str:
    about = {"person": f" about {row.get('subject')}", "approach": f" (approach to {row.get('subject')})"}.get(
        row.get("subject_kind") or "topic", "")
    status = "" if row.get("status") in {None, "current"} else f" [{row.get('status')}]"
    return (f"[{row.get('id')}] {row.get('topic') or '(erased)'}{about}{status}: {row.get('stance') or ''}"
            + (f" (audience {row.get('audience')})" if row.get("audience") else ""))


def _opinions(sidecar: Sidecar, args: argparse.Namespace, *, as_json: bool) -> int:
    """``colonymind mind opinions``: the owner's reads and controls, ``by=cli``."""
    action = args.action or "list"
    if action != "list" and args.id is None:
        print(f"usage: colonymind mind opinions {action} <id>", file=sys.stderr)
        return 2
    if action == "list":
        value = sidecar.call("GET", "/v1/mind/opinions", params={
            "q": args.query, "history": "true" if args.history else "false", "by": "cli", "limit": 50})
        rows = value.get("opinions") or []
        text = ("opinions: off" if not value.get("enabled") else
                "\n".join(_opinion_line(row) for row in rows) or "no opinions recorded")
        _emit(value, as_json=as_json, text=text)
    elif action == "show":
        value = sidecar.call("GET", f"/v1/mind/opinions/{args.id}", params={"by": "cli"})
        row = value.get("opinion") or {}
        premises = [f"  rests on: {p.get('text')} ({p.get('ref')})" for p in row.get("premises") or []]
        decision = row.get("owner_decision") if isinstance(row.get("owner_decision"), dict) else {}
        decided = [f"  you decided: {decision.get('text')} (turn:{decision.get('turn_id')})"] if decision else []
        lines = [_opinion_line(row), f"  because: {row.get('reason') or ''}",
                 f"  would change if: {row.get('revise_if') or '(not stated)'}", *premises, *decided,
                 "  history: " + " <- ".join(str(item.get("id")) for item in value.get("history") or [])]
        _emit(value, as_json=as_json, text="\n".join(lines))
    else:
        if not args.reason.strip():
            print(f"colonymind mind opinions {action} needs --reason", file=sys.stderr)
            return 2
        value = sidecar.call("POST", f"/v1/mind/opinions/{args.id}/{action}",
                             json_body={"reason": args.reason, "by": "cli"})
        _emit(value, as_json=as_json, text=f"opinion {args.id}: {value.get('status')} "
                                           f"(revision {value.get('revision_id')})")
    return 0


def _lesson_line(row: Dict[str, Any]) -> str:
    tally = row.get("tally") or {}
    source = "relayed by the operator" if row.get("relayed") else row.get("verified")
    return (f"{row.get('id')} [{row.get('status')}, {row.get('kind')}, {source}] {row.get('title')}: "
            f"when {row.get('when_to_use')} ({tally.get('wins', 0)} wins in {tally.get('uses', 0)} verified uses)")


RELAY_FIELDS = (("kind", "--kind"), ("title", "--title"), ("when", "--when"), ("content", "--content"),
                ("said_at", "--said-at"), ("note", "--note"))


def _relay(sidecar: Sidecar, args: argparse.Namespace, *, as_json: bool) -> int:
    """``colonymind mind lessons relay``: an instruction the owner gave through another channel, recorded as a lesson
    that says so (relayed by the operator, with when the owner said it and your note), never as their message."""
    missing = [flag for name, flag in RELAY_FIELDS if not str(getattr(args, name, None) or "").strip()]
    if missing:
        print(f"colonymind mind lessons relay needs {', '.join(missing)}", file=sys.stderr)
        return 2
    body = {"kind": args.kind, "title": args.title, "when_to_use": args.when, "content": args.content,
            "said_at": args.said_at, "note": args.note}
    if args.topic:
        body["topic"] = args.topic
    value = sidecar.call("POST", "/v1/mind/lessons/relayed", json_body=body)
    said = (value.get("relayed") or {}).get("said_at")
    text = (f"lesson {value.get('id')}: the owner's instruction of {said}, relayed by the operator"
            + ("" if value.get("created") else " (already recorded)"))
    _emit(value, as_json=as_json, text=text)
    return 0


def _learning(sidecar: Sidecar, args: argparse.Namespace, *, as_json: bool) -> int:
    """``colonymind mind learning``: the learning rules' counts, log and the preference changes held for the
    owner, who ratifies or declines them here (``by=cli``)."""
    action = args.action or "show"
    if action in {"ratify", "decline"}:
        if not args.id:
            print(f"usage: colonymind mind learning {action} <id>", file=sys.stderr)
            return 2
        value = sidecar.call("POST", f"/v1/mind/learning/pending/{args.id}/{action}",
                             json_body={"reason": args.reason, "by": "cli"})
        _emit(value, as_json=as_json, text=f"{args.id}: {value.get('status')}")
        return 0
    params: Dict[str, Any] = {"limit": args.limit}
    if args.store:
        params["store"] = args.store
    if action == "rejected":
        params["action"] = "rejected"
    value = sidecar.call("GET", "/v1/mind/learning", params=params)
    if action == "pending":
        lines = [f"{row['id']} {row['kind']}: {row['operation']} of {row['target_id']} by {row['by_id']}"
                 for row in value.get("pending") or []]
        _emit(value, as_json=as_json, text="\n".join(lines) or "nothing waits for you")
        return 0
    if action in {"log", "rejected"}:
        lines = [f"{row['id']} {row['store']} {row['action']} [{row['stage']}] {row['reason']}"
                 + (f" ({row['item_id']})" if row.get("item_id") else "") for row in value.get("log") or []]
        _emit(value, as_json=as_json, text="\n".join(lines) or "(empty)")
        return 0
    summary = value.get("summary") or {}
    lines = [f"learning rules: {'on' if value.get('enabled') else 'off'}; {summary.get('pending', 0)} pending, "
             f"{summary.get('claim_witnesses', 0)} claim witnesses"]
    for store, counts in sorted((summary.get("log") or {}).items()):
        lines.append(f"{store}: " + ", ".join(f"{count} {name}" for name, count in sorted(counts.items())))
    if summary.get("tiers"):
        lines.append("tiers: " + ", ".join(f"{count} {name}" for name, count in sorted(summary["tiers"].items())))
    _emit(value, as_json=as_json, text="\n".join(lines))
    return 0


def _lessons(sidecar: Sidecar, args: argparse.Namespace, *, as_json: bool) -> int:
    """``colonymind mind lessons``: the owner's view of what the mind learned, and retirement, ``by=cli``."""
    action = args.action or "list"
    if action == "relay":
        return _relay(sidecar, args, as_json=as_json)
    if action != "list" and not args.id:
        print(f"usage: colonymind mind lessons {action} <id>", file=sys.stderr)
        return 2
    if action == "retire":
        if not args.reason.strip():
            print("colonymind mind lessons retire needs --reason", file=sys.stderr)
            return 2
        value = sidecar.call("POST", f"/v1/mind/lessons/{args.id}/retire",
                             json_body={"reason": args.reason, "by": "cli"})
        _emit(value, as_json=as_json, text=f"lesson {args.id}: {value.get('status')} ({value.get('closed_reason')})")
        return 0
    params = {} if getattr(args, "all", False) or action == "show" else {"status": "active,candidate"}
    value = sidecar.call("GET", "/v1/mind/lessons", params=params)
    rows = value.get("lessons") or []
    if action == "list":
        skills = value.get("skills") if isinstance(value.get("skills"), dict) else {}
        loads = skills.get("loads") or {}
        lines = [_lesson_line(row) for row in rows]
        lines += [f"skill {name}: {int(loads.get(name, 0))} loads" for name in skills.get("owned") or []]
        text = "\n".join(lines) or (
            "no lessons" if value.get("enabled") else "lessons: off (the stored lessons are kept)")
        _emit(value, as_json=as_json, text=text)
        return 0
    row = next((item for item in rows if item.get("id") == args.id), None)
    if row is None:
        print(f"no lesson {args.id}", file=sys.stderr)
        return 1
    lines = [_lesson_line(row), f"  {row.get('content')}", f"  signature: {row.get('signature')}, origin "
             f"{row.get('origin')}", "  evidence: " + ", ".join(str(item) for item in row.get("evidence") or [])]
    relayed = row.get("relayed") if isinstance(row.get("relayed"), dict) else None
    if relayed:
        lines.append(f"  relayed by the operator ({relayed.get('recorded_by')}, {relayed.get('recorded_at')}); the "
                     f"owner said it {relayed.get('said_at')}: {relayed.get('note')}")
    if row.get("correction"):
        lines.append(f"  correction: {row['correction']}" + (f" ({row['retrieval_source']})"
                                                             if row.get("retrieval_source") else ""))
    if row.get("supersedes"):
        lines.append(f"  supersedes: {row['supersedes']}")
    if row.get("closed_reason"):
        lines.append(f"  {row.get('status')}: {row['closed_reason']}")
    _emit(row, as_json=as_json, text="\n".join(lines))
    return 0


def yes_refusal(code: str) -> str:
    """What ``colonymind mind yes`` says instead of approving: where a yes comes from, and the two things the console
    can do about the ask."""
    code = str(code or "").strip().upper()
    return (f"Not approved: an ask is approved only by the owner's own answer in their chat, never from this "
            f"console (anything that can run a command here could run this one).\n"
            f"  colonymind mind show {code}   sends the ask to the owner's chat again\n"
            f"  colonymind mind no {code}     declines it here")


def _grant_entry(args: argparse.Namespace) -> Dict[str, Any]:
    """The ``mind.grants`` entry the options describe, with only what was given."""
    entry: Dict[str, Any] = {"id": args.id, "class": args.grant_class, "scope": args.scope,
                             "tools": [item.strip() for item in (args.tools or "").split(",") if item.strip()],
                             "match": list(args.match or []), "max_amount": args.max_amount,
                             "currency": args.currency, "per_day": args.per_day,
                             "per_month_amount": args.per_month_amount, "expires": args.expires}
    return {key: value for key, value in entry.items() if value not in (None, "", [])}


def _grants(sidecar: Sidecar, args: argparse.Namespace, *, as_json: bool) -> int:
    """``colonymind mind grants``: list or revoke (a narrowing) a standing grant. ``add`` adds nothing: a grant is the
    owner's to give in their own chat, or the operator's to configure, and it prints the entry for that."""
    action = getattr(args, "action", "list") or "list"
    if action == "list":
        value = sidecar.call("GET", "/v1/mind/grants")
        _emit(value, as_json=as_json, text=value.get("text"))
        return 0
    if not args.id:
        print(f"grants {action} needs the grant's id", file=sys.stderr)
        return 2
    if action == "revoke":
        value = sidecar.call("POST", f"/v1/mind/grants/{args.id}/revoke", json_body={"by": "cli"})
        _emit(value, as_json=as_json, text=value.get("text"))
        return 0
    entry = _grant_entry(args)
    grant = parse_grant(entry, source="config")
    if not grant.valid:
        print(f"grants add: {grant.problem}", file=sys.stderr)
        return 2
    print("Not added: a standing grant is the owner's to give. Ask for it in your own chat (the agent files it and "
          "you answer there), or add this entry to mind.grants in colonymind.yaml and restart the sidecar (the owner "
          "is told when the configured grants change):\n")
    print(yaml.safe_dump({"mind": {"grants": [entry]}}, sort_keys=False, allow_unicode=True).rstrip())
    return 1


def _allows(sidecar: Sidecar, args: argparse.Namespace, *, as_json: bool) -> int:
    """``colonymind mind allows``: list the owner's standing allows, or revoke one (a narrowing). Nothing here adds
    one: only the owner's own "always" to an ask in their chat does."""
    if (getattr(args, "action", "list") or "list") == "list":
        value = sidecar.call("GET", "/v1/mind/allows")
        _emit(value, as_json=as_json, text=value.get("text"))
        return 0
    if not args.id:
        print("allows revoke needs the allow's id (colonymind mind allows lists them)", file=sys.stderr)
        return 2
    value = sidecar.call("POST", f"/v1/mind/allows/{quote(args.id.strip(), safe='')}/revoke", json_body={"by": "cli"})
    _emit(value, as_json=as_json, text=value.get("text"))
    return 0


def _state_dir(sidecar: Sidecar) -> Path:
    from colonymind import get_state_dir
    return get_state_dir()


__all__ = ["COMMANDS", "add_parser", "run"]
