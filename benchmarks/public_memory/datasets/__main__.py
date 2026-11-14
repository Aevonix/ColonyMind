"""Fetch, verify and subset the public memory datasets. Run from the repository root:

    python -m benchmarks.public_memory.datasets fetch [NAME ...]
    python -m benchmarks.public_memory.datasets verify NAME
    python -m benchmarks.public_memory.datasets subset NAME --seed S (--size N | --quota TYPE=N ...) --out FILE
    python -m benchmarks.public_memory.datasets check NAME RECORD

Files go to ``--data-dir``, else ``COLONYMIND_BENCH_DATA_DIR``, else the user's cache directory. ``--offline`` never
downloads: a missing file is an error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import NAMES, beam, longmemeval_s, pinned, precisionmembench, subsets
from .records import DatasetError

ADAPTERS = {"longmemeval_s": longmemeval_s, "beam": beam, "precisionmembench": precisionmembench}
QUESTION_SETS = ("longmemeval_s", "beam")


def _load(args):
    return ADAPTERS[args.name].load(args.data_dir, download=not args.offline)


def _fetch(args) -> int:
    unknown = sorted(set(args.names) - set(NAMES))
    if unknown:
        raise DatasetError(f"no dataset named {', '.join(unknown)}; choose from {', '.join(NAMES)}")
    for name in args.names or NAMES:
        for pin in ADAPTERS[name].PINNED:
            path = pinned.fetch(pin, args.data_dir, download=not args.offline)
            print(f"{name}\t{pin.sha256}\t{path}")
    return 0


def _verify(args) -> int:
    data = _load(args)
    if isinstance(data, precisionmembench.Corpus):
        report = {"dataset": data.name, "sha256": dict(data.sha256), "counts": data.counts(),
                  "beliefs": len(data.beliefs), "cases": len(data.cases), "sessions": len(data.sessions),
                  "scored_turns": data.scored_turns()}
    else:
        report = {"dataset": data.name, "sha256": data.sha256, "counts": data.counts(),
                  "questions": len(data.questions), "abstention": len(data.abstention_ids()),
                  "conversations": len(data.conversations),
                  "sessions": sum(len(item.sessions) for item in data.conversations.values()),
                  "questions_with_evidence": sum(1 for item in data.questions if item.evidence)}
    report["findings"] = {name: len(ids) for name, ids in data.findings.items()}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _quotas(values) -> dict:
    quotas = {}
    for value in values or ():
        name, _, count = value.rpartition("=")
        if not name or not count.isdigit():
            raise DatasetError(f"--quota {value!r} is not TYPE=N")
        quotas[name] = int(count)
    return quotas


def _subset(args) -> int:
    dataset = _load(args)
    exclude = [question_id for path in args.exclude or ()
               for question_id in json.loads(Path(path).read_text(encoding="utf-8"))["ids"]]
    drawn = subsets.draw(dataset, seed=args.seed, size=args.size, quotas=_quotas(args.quota) or None, by=args.by,
                         exclude=exclude)
    Path(args.out).write_text(json.dumps(drawn.record(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "size": len(drawn.ids), "strata": dict(drawn.strata)}, sort_keys=True))
    return 0


def _check(args) -> int:
    record = json.loads(Path(args.record).read_text(encoding="utf-8"))
    checked = subsets.check(record, _load(args))
    print(json.dumps({"record": str(args.record), "size": len(checked.ids), "matches": True}))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.public_memory.datasets", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=Path, default=None)
    common.add_argument("--offline", action="store_true", help="never download; a missing file is an error")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", parents=[common], help="download and hash-check the pinned files")
    fetch.add_argument("names", nargs="*", metavar="NAME", help=f"any of {', '.join(NAMES)}; all by default")
    verify = commands.add_parser("verify", parents=[common], help="parse a dataset and check its counts")
    verify.add_argument("name", choices=NAMES)
    subset = commands.add_parser("subset", parents=[common], help="draw a seeded, stratified subset")
    subset.add_argument("name", choices=QUESTION_SETS)
    subset.add_argument("--seed", required=True)
    subset.add_argument("--size", type=int)
    subset.add_argument("--quota", action="append", metavar="TYPE=N")
    subset.add_argument("--by", choices=sorted(subsets.BY), default="type")
    subset.add_argument("--exclude", action="append", metavar="RECORD", help="leave out the ids of this record")
    subset.add_argument("--out", required=True, type=Path)
    check = commands.add_parser("check", parents=[common], help="check a subset record against the data")
    check.add_argument("name", choices=QUESTION_SETS)
    check.add_argument("record", type=Path)
    args = parser.parse_args(argv)
    handlers = {"fetch": _fetch, "verify": _verify, "subset": _subset, "check": _check}
    try:
        return handlers[args.command](args)
    except (DatasetError, pinned.PinError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
