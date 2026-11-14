"""Inspect plasticity and request rollback or erasure in the selected instance."""
from __future__ import annotations

import argparse
import json
import sys


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("plasticity", help="Learning in weights: status, revisions, comparison and erasure")
    commands = parser.add_subparsers(dest="plasticity_command")
    for name, help_text in (
        ("status", "Faculty status, loaded learners and registration failures"),
        ("learners", "Configured learners and what each serves"),
        ("revisions", "A learner's retained revisions and their states"),
        ("comparator", "The live revision against its previous revision on the same traffic"),
        ("rollback", "Withdraw a learner's serving revision and restore its previous revision"),
        ("erase", "Unserve, remove and rebuild learning that used the given references"),
        ("qualify", "Measure a query adapter on a frozen local gate suite"),
    ):
        command = commands.add_parser(name, help=help_text)
        if name in {"revisions", "comparator", "rollback", "qualify"}:
            command.add_argument("learner", help="The configured learner name")
        elif name == "erase":
            command.add_argument("--refs", nargs="+", required=True, metavar="REF", help="Record references")
        command.add_argument("--json", action="store_true", help="Print the same result as JSON")
        if name == "qualify":
            command.add_argument("--queries", required=True, help="Frozen local query suite JSON")
            command.add_argument("--embed-url", help="Override the configured embedding endpoint")
            command.add_argument("--rerank-url", help="Override the configured reranking endpoint")
            command.add_argument("--max-queries", type=int, default=10000, help="Maximum selections across arms")
            command.add_argument("--max-seconds", type=float, default=3600, help="Total scoring time budget")


def _runtime():
    from colonymind.config import load_config
    from .nightly import Runtime
    return Runtime.from_config(load_config())


def _text(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            _text(item, f"{prefix}{str(key).replace('_', ' ')}: ")
    elif isinstance(value, list):
        if not value:
            print(prefix + "none")
        for item in value:
            _text(item, prefix)
    else:
        print(prefix + ("yes" if value is True else "no" if value is False else str(value)))


def run(args: argparse.Namespace) -> int:
    command = getattr(args, "plasticity_command", None)
    if command not in {"status", "learners", "revisions", "comparator", "rollback", "erase", "qualify"}:
        print("usage: colonymind plasticity {status,learners,revisions,comparator,rollback,erase,qualify} [--json]",
              file=sys.stderr)
        return 2
    try:
        if command == "qualify":
            import asyncio
            from .recall_qualify import run_command
            result = asyncio.run(run_command(args))
        elif command in {"status", "learners"}:
            runtime = _runtime()
            result = getattr(runtime, command)()
        elif command == "erase":
            runtime = _runtime()
            result = runtime.erase(args.refs)
        else:
            runtime = _runtime()
            result = getattr(runtime, command)(args.learner)
        if args.json:
            print(json.dumps(result, sort_keys=True, allow_nan=False))
        else:
            _text(result)
    except (KeyError, ValueError, OSError) as error:
        print(f"colonymind plasticity {command}: {error}", file=sys.stderr)
        return 2
    return 0


__all__ = ["add_parser", "run"]
