"""``colonymind memory reextract``: re-extract a person's history under the current extraction version.

The command works on the instance's memory ledger (``turn-idempotency.db`` in the state directory) directly: it
queues the run for the sidecar's background lane (``beliefs.reextract``), shows progress and cancels what still
waits. It needs no running sidecar to plan or queue; the sidecar's lane takes the queue up whenever it runs. It is
the owner's to start: it spends model calls and forms again what memory holds about a person, so nothing starts it
but this command. Before it queues anything it prints the plan: the count, and the estimate at the per-message rate
this instance measured.

``--dry-run`` opens the ledger read-only: it prints the same plan and writes nothing.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, List

LEDGER = "turn-idempotency.db"
STATES = (("pending", "waiting"), ("running", "working"), ("complete", "done"),
          ("kept", "done but for a message that kept its earlier claims (the next run reads it again)"),
          ("failed", "failed"), ("skipped", "skipped"), ("cancelled", "cancelled"))


def add_parser(sub: argparse._SubParsersAction) -> None:
    """``memory reextract``. The ``memory`` group is shared with the other memory commands: it is made here only
    when no other command made it first, and ``reextract`` joins its subcommands either way."""
    memory = sub.choices.get("memory") or sub.add_parser("memory", help="What memory holds about a person")
    commands = next((action for action in memory._actions if isinstance(action, argparse._SubParsersAction)), None)
    if commands is None:
        commands = memory.add_subparsers(dest="memory_command")
    parser = commands.add_parser(
        "reextract", help="Form a person's claims again under the current extraction version",
        description="Queue the re-extraction of the attributed sources whose claims an older extraction version "
                    "formed. The sidecar works the queue in the background, one person at a time in the order "
                    "things were said; imported history and checkpoints are never queued. The plan and its "
                    "estimate are printed first.")
    who = parser.add_mutually_exclusive_group()
    who.add_argument("--person", default=None, help="Only this person (a contact id)")
    who.add_argument("--all", action="store_true", help="Everyone")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and the estimate; write nothing")
    parser.add_argument("--limit", type=int, default=None,
                        help="At most this many sources (the first, in the order they were said); a small run "
                             "measures the per-message rate for the estimate")
    parser.add_argument("--status", action="store_true", help="The runs and their progress")
    parser.add_argument("--cancel", action="store_true",
                        help="Cancel every source that still waits, for everyone; its claims stay as they are. "
                             "Takes no other option but --all and --json")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")


def state_dir() -> Path:
    """The state directory the sidecar uses (``colonymind.get_state_dir``), found without creating anything."""
    explicit = os.environ.get("COLONYMIND_STATE_DIR") or os.environ.get("COLONYMIND_HOME")
    if explicit:
        return Path(explicit).expanduser()
    home = Path.home() / ".colonymind"
    return home if (home / "colonymind.yaml").is_file() else home / "data"


def _read_only(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def _writable(path: Path) -> sqlite3.Connection:
    """The ledger as the sidecar opens it (``TurnIdempotencyLedger._connect``), without its start-up steps: the
    command changes only the re-extraction tables."""
    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA secure_delete=ON")
    return conn


def _fail(message: str, code: int) -> int:
    print(f"colonymind memory reextract: {message}", file=sys.stderr)
    return code


def run(args: argparse.Namespace) -> int:
    from . import reextract
    path = state_dir() / LEDGER
    if args.cancel:
        # Cancelling writes, and it cancels everything: an option that would narrow it or make it a preview is
        # refused rather than ignored.
        others = [flag for flag, given in (("--dry-run", args.dry_run), ("--status", args.status),
                                           ("--person", args.person), ("--limit", args.limit is not None)) if given]
        if others:
            return _fail(f"--cancel cancels every waiting source and takes no {' or '.join(others)}", 2)
    if not (args.status or args.cancel):
        if not args.person and not args.all:
            return _fail("name --person <id> or --all (or ask for --status or --cancel)", 2)
        if args.limit is not None and args.limit < 1:
            return _fail("--limit must be at least 1", 2)
    if not path.is_file():
        return _fail(f"no memory ledger at {path}", 1)
    try:
        if args.status:
            with closing(_read_only(path)) as conn:
                state = reextract.status(conn)
            _print(state, render_status(state), args.json)
            return 0
        if args.cancel:
            with closing(_writable(path)) as conn:
                result = reextract.cancel(conn)
            _print(result, [f"Cancelled {_count(result['cancelled'], 'source')} that were waiting; their claims "
                            "stay as they are."], args.json)
            return 0
        person = args.person or None
        if args.dry_run:
            with closing(_read_only(path)) as conn:
                plan = reextract.plan(conn, person=person, limit=args.limit)
            _print(plan, render_plan(plan) + ["Dry run: nothing was written."], args.json)
            return 0
        with closing(_writable(path)) as conn:
            plan = reextract.plan(conn, person=person, limit=args.limit)
            if not args.json:
                print("\n".join(render_plan(plan)), flush=True)
            started = reextract.start(conn, person=person, limit=args.limit)
    except sqlite3.Error as exc:
        return _fail(f"the memory ledger could not be read ({type(exc).__name__}: {exc})", 1)
    if args.json:
        print(json.dumps({**plan, **started}, indent=2, sort_keys=True))
    elif started["queued"]:
        print(f"Queued {_count(started['queued'], 'source')} (run {started['run_id'][:12]}). The sidecar works "
              "through the queue in the background, one person at a time in the order things were said. "
              "Progress: colonymind memory reextract --status")
    elif plan["sources"]:
        print("Nothing was queued: another run took those sources first.")
    return 0


def _print(value: Dict[str, Any], lines: List[str], as_json: bool) -> None:
    print(json.dumps(value, indent=2, sort_keys=True) if as_json else "\n".join(lines))


def _count(number: int, noun: str, plural: str | None = None) -> str:
    return f"{number:,} {noun if number == 1 else (plural or noun + 's')}"


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} s"
    if seconds < 90 * 60:
        return f"{round(seconds / 60)} min"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def _rate(rate: Dict[str, Any]) -> str:
    parts = [f"{rate['calls_per_message']:.1f} calls"]
    if rate.get("tokens_per_message") is not None:
        parts.append(f"{rate['tokens_per_message']:,} tokens")
    parts.append(f"{rate['seconds_per_message']:.1f} s of model time")
    return ", ".join(parts[:-1]) + " and " + parts[-1] + f" a message, measured on {_count(rate['messages'], 'message')}"


def render_plan(plan: Dict[str, Any]) -> List[str]:
    who = f"person {plan['person']}" if plan.get("person") else "everyone"
    lines = [f"Re-extraction under {plan['version']} ({who}"
             + (f", at most {_count(plan['limit'], 'source')}" if plan.get("limit") else "") + ")"]
    queued = (f"  {_count(plan['queued'], 'source')} already queued, not counted here (progress: colonymind memory "
              "reextract --status)") if plan.get("queued") else None
    if not plan["sources"]:
        if queued:
            return lines + ["  Nothing new to queue.", queued]
        return lines + [f"  Nothing to re-extract: every attributed source's claims were formed under "
                        f"{plan['version']}."]
    versions = ", ".join(f"{version}: {_count(count, 'source')}"
                         for version, count in sorted(plan["from_versions"].items()))
    lines.append(f"  {_count(plan['people'], 'person', 'people')}, {_count(plan['sources'], 'source')}, "
                 f"{_count(plan['messages'], 'message')} ({plan['model_messages']:,} read by the model); "
                 f"{_count(plan['claims'], 'claim')} formed by older versions ({versions})")
    if queued:
        lines.append(queued)
    estimate, rate = plan["estimate"], plan.get("rate")
    if rate is not None:
        cost = [f"about {estimate['calls']:,} model calls"]
        if estimate.get("tokens") is not None:
            cost.append(f"{round(estimate['tokens']):,} tokens")
        cost.append(f"{_duration(estimate['seconds'])} of model time")
        lines.append("  estimate: " + ", ".join(cost[:-1]) + " and " + cost[-1] + f" (at {_rate(rate)}; the lane "
                     "yields to live work, so it takes longer than that)")
    else:
        share = plan.get("review_share")
        basis = (f"{share['share']:.2f} reviews an extraction, measured on {_count(share['jobs'], 'claim job')}"
                 if share else "a review for each extraction, not measured yet")
        lines.append(f"  estimate: about {estimate['calls']:,} model calls (one extraction a message and {basis})")
        lines.append("  time and tokens: not measured yet; a small run measures them (--limit 20, on a clone of the "
                     "instance)")
    return lines


def render_status(state: Dict[str, Any]) -> List[str]:
    lines = [f"Re-extraction under {state['version']}"]
    if not state["runs"]:
        return lines + ["  No run yet."]
    for run in state["runs"]:
        who = f"person {run['person']}" if run.get("person") else "everyone"
        when = datetime.fromtimestamp(run["created_at"], timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines.append(f"  run {run['id'][:12]} ({who}), started {when}: {run['status']}")
        counts = [f"{run['jobs'][status]:,} {word}" for status, word in STATES if run["jobs"].get(status)]
        lines.append("    " + (", ".join(counts) or "no sources"))
        if run["reports"]:
            lines.append("    reported: " + ", ".join(f"{count:,} {kind.replace('_', ' ')}"
                                                   for kind, count in sorted(run["reports"].items())))
    if state.get("rate") is not None:
        left = (f"; about {_duration(state['remaining_seconds'])} of model time left for "
                f"{_count(state['remaining_messages'], 'message')}" if state["remaining_messages"] else "")
        lines.append(f"  measured: {_rate(state['rate'])}{left}")
    elif state["remaining_messages"]:
        lines.append(f"  {_count(state['remaining_messages'], 'message')} left; no rate measured yet")
    return lines
