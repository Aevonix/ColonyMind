"""``colonymind report``: reports a person can read (docs/MIND.md, Reports).

``faculties`` reads the selected instance's stores read-only and reports what each faculty did over a window,
with the latest paired benchmark evidence beside it when ``--bench`` names a results directory; ``bench DIR``
renders the paired benchmark's report files under DIR in plain words; ``instinct`` reports what each decision point
did per tier (``reports/instinct.py``); ``usage`` counts the model calls by task, role and endpoint, and what memory
costs per captured message (``reports/usage.py``); ``claims`` counts the claims formed from the messages captured in a
window, and what became of those whose subject an earlier message named (``reports/claims.py``). Nothing here writes
anywhere but ``--out``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .text import FORMATS, render


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("report", help="Reports a person can read: what each faculty did, benchmark results")
    commands = parser.add_subparsers(dest="report_command")
    behaviour = commands.add_parser("behaviour", help="Daily reply and tool behaviour counts (read-only)")
    behaviour.add_argument("--since", default=None, metavar="YYYY-MM-DD", help="First UTC day (inclusive)")
    behaviour.add_argument("--until", default=None, metavar="YYYY-MM-DD", help="Last UTC day (inclusive)")
    behaviour.add_argument("--json", action="store_true", help="Print the same counts as JSON")
    behaviour.add_argument("--format", choices=FORMATS, default="md", help="md (default) or html")
    behaviour.add_argument("--out", default=None, help="Write the report to this file instead of printing it")
    plasticity = commands.add_parser("plasticity", help="Learners, teacher evidence and live comparisons (read-only)")
    plasticity.add_argument("--json", action="store_true", help="Print the same result as JSON")
    plasticity.add_argument("--format", choices=FORMATS, default="md", help="md (default) or html")
    plasticity.add_argument("--out", default=None, help="Write the report to this file instead of printing it")
    faculties = commands.add_parser("faculties", help="What each faculty of the mind did over a window (read-only)")
    faculties.add_argument("--since", default="7d", help="A span back from now (30m, 24h, 7d, 2w) or a date "
                                                         "(2026-09-01); default 7d")
    faculties.add_argument("--format", choices=FORMATS, default="md", help="md (default) or html")
    faculties.add_argument("--out", default=None, help="Write the report to this file instead of printing it")
    faculties.add_argument("--bench", default=None, help="A paired benchmark results directory: the latest "
                                                         "evidence for each faculty goes beside it")
    instinct = commands.add_parser("instinct", help="What each decision point did per tier over a window (read-only)")
    instinct.add_argument("--since", default="7d", help="A span back from now (30m, 24h, 7d, 2w) or a date; default 7d")
    instinct.add_argument("--format", choices=FORMATS, default="md", help="md (default) or html")
    instinct.add_argument("--out", default=None, help="Write the report to this file instead of printing it")
    usage = commands.add_parser("usage", help="Model calls by task, role and endpoint, in counts only (read-only)")
    usage.add_argument("--since", default="7d", help="A span back from now (30m, 24h, 7d, 2w) or a date; default 7d")
    usage.add_argument("--json", action="store_true", help="The same numbers as JSON (sorted keys, a fixed shape)")
    usage.add_argument("--format", choices=FORMATS, default="md", help="md (default) or html")
    usage.add_argument("--out", default=None, help="Write the report to this file instead of printing it")
    usage.add_argument("--alias", default=None, metavar="NAME", help="Say what this instance's rows record for a "
                       "role or endpoint name (its alias when it is not a plain name); prints no report")
    claims = commands.add_parser("claims", help="Claims formed from the messages captured in a window, and those "
                                                "whose subject an earlier message named, in counts only (read-only)")
    claims.add_argument("--since", default="7d", help="A span back from now (30m, 24h, 7d, 2w) or a date; default 7d")
    claims.add_argument("--json", action="store_true", help="The same numbers as JSON (sorted keys, a fixed shape)")
    claims.add_argument("--format", choices=FORMATS, default="md", help="md (default) or html")
    claims.add_argument("--out", default=None, help="Write the report to this file instead of printing it")
    bench = commands.add_parser("bench", help="The paired benchmark's report files under DIR, in plain words")
    bench.add_argument("directory", help="A results directory: plan directories and their report files")
    bench.add_argument("--format", choices=FORMATS, default="md", help="md (default) or html")
    bench.add_argument("--out", default=None, help="Write the report to this file instead of printing it")


def _emit(text: str, out: str | None) -> None:
    if out:
        path = Path(out).expanduser()
        path.write_text(text, encoding="utf-8")
        print(f"Wrote {path}")
    else:
        sys.stdout.write(text)


def run(args: argparse.Namespace) -> int:
    command = getattr(args, "report_command", None)
    if command == "behaviour":
        from colonymind.config import ConfigError, instance_home
        from . import plasticity
        try:
            collected = plasticity.collect_behaviour(instance_home(), since=args.since, until=args.until)
        except (ConfigError, ValueError, OSError) as error:
            print(f"colonymind report behaviour: {error}", file=sys.stderr)
            return 2
        _emit(plasticity.to_json(collected) if args.json else
              render(plasticity.behaviour_document(collected), args.format), args.out)
        return 0
    if command == "plasticity":
        from colonymind.config import ConfigError, load_config
        from . import plasticity
        try:
            collected = plasticity.collect(load_config())
        except (ConfigError, ValueError, OSError) as error:
            print(f"colonymind report plasticity: {error}", file=sys.stderr)
            return 2
        _emit(plasticity.to_json(collected) if args.json else render(plasticity.document(collected), args.format),
              args.out)
        return 0
    if command == "faculties":
        from colonymind.config import ConfigError, instance_home
        from . import bench, faculties
        try:
            loaded = bench.load(args.bench) if args.bench else None
            collected = faculties.collect(instance_home(), since=args.since, bench=loaded)
        except (ValueError, FileNotFoundError, ConfigError) as error:
            print(f"colonymind report faculties: {error}", file=sys.stderr)
            return 2
        _emit(render(faculties.document(collected), args.format), args.out)
        return 0
    if command == "instinct":
        from colonymind.config import instance_home
        from . import instinct
        try:
            collected = instinct.collect(instance_home(), since=args.since)
        except ValueError as error:
            print(f"colonymind report instinct: {error}", file=sys.stderr)
            return 2
        _emit(render(instinct.document(collected), args.format), args.out)
        return 0
    if command == "usage":
        from colonymind import usage as ledger
        from colonymind.config import instance_home
        from . import usage
        if getattr(args, "alias", None) is not None:
            # An alias is keyed with this instance's secret: only the instance itself can say which name it is.
            try:
                recorded = ledger.alias_of(instance_home(), args.alias)
            except (OSError, ValueError) as error:
                print(f"colonymind report usage: the instance's secret cannot be read ({type(error).__name__})",
                      file=sys.stderr)
                return 1
            # A ledger an earlier build wrote holds the name's plain digest prefix until this build's first write
            # keys it in place: those rows are the name's too.
            try:
                earlier = ledger.earlier_alias_of(instance_home(), args.alias)
            except ledger.Unreadable as error:
                print(f"colonymind report usage: {error}", file=sys.stderr)
                return 1
            if earlier is not None:
                print(f"{args.alias} is recorded as {earlier} in the rows an earlier build wrote"
                      + (f", and as {recorded} from the next write, which keys those rows the same way"
                         if recorded is not None else "; the next write keys them under this instance's own secret"))
            else:
                print(f"{args.alias} is recorded as {recorded}" if recorded is not None
                      else f"{args.alias}: no alias recorded in this instance yet")
            return 0
        try:
            collected = usage.collect(instance_home(), since=args.since)
        except ValueError as error:
            print(f"colonymind report usage: {error}", file=sys.stderr)
            return 2
        except ledger.Unreadable as error:             # locked or damaged: its usage is unknown, never zero
            print(f"colonymind report usage: {error}", file=sys.stderr)
            return 1
        _emit(usage.to_json(collected) if args.json else render(usage.document(collected), args.format), args.out)
        return 0
    if command == "claims":
        from colonymind.config import instance_home
        from . import claims
        try:
            collected = claims.collect(instance_home(), since=args.since)
        except ValueError as error:
            print(f"colonymind report claims: {error}", file=sys.stderr)
            return 2
        except claims.Unreadable as error:            # damaged or not ours: what it formed is unknown, never zero
            print(f"colonymind report claims: {error}", file=sys.stderr)
            return 1
        _emit(claims.to_json(collected) if args.json else render(claims.document(collected), args.format), args.out)
        return 0
    if command == "bench":
        from . import bench
        try:
            loaded = bench.load(args.directory)
        except FileNotFoundError as error:
            print(f"colonymind report bench: {error}", file=sys.stderr)
            return 2
        _emit(render(bench.document(loaded), args.format), args.out)
        return 0
    print("usage: colonymind report faculties [--since 7d] [--format md|html] [--out PATH] [--bench DIR]\n"
          "       colonymind report instinct [--since 7d] [--format md|html] [--out PATH]\n"
          "       colonymind report usage [--since 7d] [--json | --format md|html] [--out PATH] [--alias NAME]\n"
          "       colonymind report claims [--since 7d] [--json | --format md|html] [--out PATH]\n"
          "       colonymind report plasticity [--json | --format md|html] [--out PATH]\n"
          "       colonymind report behaviour [--since YYYY-MM-DD] [--until YYYY-MM-DD] "
          "[--json | --format md|html] [--out PATH]\n"
          "       colonymind report bench DIR [--format md|html] [--out PATH]", file=sys.stderr)
    return 2


__all__ = ["add_parser", "run"]
