"""``colonymind instinct``: ``status``, ``points`` and ``log`` for the selected instance (docs/DECISIONS.md), and the
owner's hand on learning: ``label`` (the right label for a decision), ``fit`` (import labelled sets, run the nightly
pass now), ``promote`` and ``demote`` (a tier's state, with its evidence), ``records`` and ``restore``.

From the instance's own files (``colonymind.yaml`` and ``instinct.db``), so only whoever can read the instance, its
owner, sees or changes them. Ids, labels and counts: never the words a point read, and never a provider's address.
A running sidecar reads a changed state or record within half a minute.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from colonymind.util.temporal import now_utc


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("instinct", help="Instinct, the mind's quick decisions: status, points and the log")
    commands = parser.add_subparsers(dest="instinct_command")
    status = commands.add_parser("status", help="Providers, each point's state and the last day's decisions")
    status.add_argument("--json", action="store_true", help="Print JSON")
    points = commands.add_parser("points", help="Every decision point: version, effect, tiers and states")
    points.add_argument("--json", action="store_true", help="Print JSON")
    log = commands.add_parser("log", help="Recent decisions: ids, tiers, labels and references (no text)")
    log.add_argument("--point", default=None, help="Only this point")
    log.add_argument("--limit", type=int, default=20, help="The most rows (default 20)")
    log.add_argument("--since", default=None, help="A span back from now (30m, 24h, 7d) or a date")
    log.add_argument("--json", action="store_true", help="Print JSON")
    label = commands.add_parser("label", help="The right label for a logged decision (your explicit outcome)")
    label.add_argument("decision", help="The decision id (dec-...)")
    label.add_argument("label", help="The label that was right there")
    fit = commands.add_parser("fit", help="Import labelled sets and run the nightly pass now (outcomes, refit, "
                                          "demotions)")
    fit.add_argument("--sets", nargs="*", default=[], metavar="FILE",
                     help="Labelled answer files (colonymind-instinct-sets/1) to import first")
    fit.add_argument("--json", action="store_true", help="Print JSON")
    for verb, words in (("promote", "Put a model tier live on its evidence (refused without it)"),
                        ("demote", "Take a live model tier back to shadow (demoted)")):
        command = commands.add_parser(verb, help=words)
        command.add_argument("point", help="The decision point")
        command.add_argument("--tier", required=True, choices=("fast", "reader"), help="The model tier")
        if verb == "demote":
            command.add_argument("--reason", default="the owner's call", help="Why (kept with the transition)")
        else:
            command.add_argument("--revision", default=None,
                                 help="The revision the tier's provider serves (needed only before its first answer)")
    records = commands.add_parser("records", help="Calibration records put in force, newest first")
    records.add_argument("--point", default=None, help="Only this point")
    records.add_argument("--json", action="store_true", help="Print JSON")
    restore = commands.add_parser("restore", help="Put an earlier calibration record back in force")
    restore.add_argument("record", help="The record id (from records)")


def _instance() -> Path:
    from colonymind.config import instance_home
    return instance_home()


def _store(home: Path) -> Any:
    from .log import DB_FILE, Store
    path = home / DB_FILE
    return Store(path, create=False) if path.is_file() else None


def _section(home: Path) -> Dict[str, Any]:
    from colonymind.config import ConfigError, load_config

    from . import settings
    try:
        return settings.parse(load_config(home).get("instinct") or {})
    except ConfigError:
        return settings.parse({})


def _tiers(point: Any, overrides: Dict[Any, str]) -> List[str]:
    from . import state as point_state
    out = []
    for tier in point.tiers:
        if tier.on_path:
            out.append(f"{tier.name} (existing path)")
        else:
            out.append(f"{tier.name} {point_state.effective(point, tier.name, overrides)}")
    return out or ["rules and fallback only"]


def points_listing(home: Path) -> List[Dict[str, Any]]:
    from . import REGISTRY
    store = _store(home)
    overrides = store.states() if store is not None else {}
    return [{"point": name, "version": point.version, "type": point.type, "effect": point.effect,
             "labels": list(point.labels), "tiers": _tiers(point, overrides), "turn_path": point.turn_path}
            for name, point in REGISTRY.items()]


def status(home: Path) -> Dict[str, Any]:
    from .state import CARRIED
    section = _section(home)
    store = _store(home)
    tiers = {}
    for tier in ("fast", "reader"):
        name = section[tier]
        entry = section["providers"].get(name) if name else None
        tiers[tier] = ({"provider": name, "kind": entry["kind"], "configured": bool(entry.get("url"))}
                       if entry else None)
    rows = store.rows(since=now_utc() - timedelta(days=1)) if store is not None else []
    answered: Dict[str, int] = {}
    for row in rows:
        answered[row["answered_by"]] = answered.get(row["answered_by"], 0) + 1
    return {"instance": str(home), "tiers": tiers, "points": points_listing(home),
            "last_day": {"decisions": len(rows), "answered_by": answered,
                         "carried": sum(change["source"].startswith(CARRIED) for change in
                                        store.transitions(since=now_utc() - timedelta(days=1), limit=None))
                         if store is not None else 0}}


def _since(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    from colonymind.reports.faculties import window
    start, _, _ = window(value, now=now_utc())
    return start


def log_rows(home: Path, *, point: Optional[str] = None, limit: int = 20,
             since: Optional[str] = None) -> List[Dict[str, Any]]:
    store = _store(home)
    if store is None:
        return []
    rows = store.rows(since=_since(since), point=point, limit=max(1, int(limit)))
    return [{key: row[key] for key in ("id", "at", "event", "point", "version", "answered_by", "label", "applied",
                                       "reason", "refs")}
            | {"tiers": {tier: {key: entry[key] for key in ("state", "outcome", "label", "probability", "latency_ms")
                                if key in entry} for tier, entry in row["tiers"].items()}}
            for row in rows]


def _writable(home: Path) -> Any:
    from .log import DB_FILE, Store
    path = home / DB_FILE
    if not path.is_file():
        raise ValueError(f"no decision log in {home} yet")
    return Store(path)


def _learning(command: str, args: argparse.Namespace, home: Path) -> int:
    from . import fit, outcomes
    from . import state as point_state
    now = now_utc()
    if command == "label":
        try:
            outcome = outcomes.label(_writable(home), args.decision, args.label, now=now)
        except KeyError:
            raise ValueError(f"no decision {args.decision} in the log") from None
        print(f"{args.decision}: {outcome['label']} recorded as your explicit outcome "
              f"({'it agrees with' if outcome['verdict'] == 'right' else 'it corrects'} what was decided)")
        return 0
    if command == "fit":
        from .log import DB_FILE, Store
        store = Store(home / DB_FILE)
        imported = []
        for name in args.sets:
            try:
                document = json.loads(Path(name).read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise ValueError(f"{name}: not readable JSON ({type(error).__name__})") from None
            imported.append(fit.import_sets(store, document, now=now))
        summary = {"imported": imported, **fit.nightly(home, now=now)}
        if args.json:
            print(json.dumps(summary, indent=2, default=str))
            return 0
        for entry in imported:
            print(f"imported {entry['set']}: {entry['kept']} answers kept"
                  + (f", left out {entry['skipped']}" if entry["skipped"] else ""))
        print(f"forgotten (erased sources): {summary.get('forgotten', 0)}")
        for point, counts in sorted((summary.get("resolved") or {}).items()):
            print(f"outcomes {point}: " + ", ".join(f"{count} {kind}" for kind, count in sorted(counts.items())))
        for entry in summary.get("refits") or []:
            print(f"refit {entry['point']} {entry['tier']}: {entry['action']} ({entry['reason']})")
        for entry in summary.get("carried") or []:
            print(f"carried by lineage {entry['point']} {entry['tier']}")
        for entry in summary.get("demoted") or []:
            print(f"demoted {entry['point']} {entry['tier']}: {entry['evidence'].get('reason')}")
        for entry in summary.get("unmeasured") or []:
            unlabelled = entry.get("unlabelled", 0)
            for count, reason in ((entry["unmeasured"] - unlabelled, "nothing to compare them with"),
                                  (unlabelled, "no outcome to say who was right")):
                if count:
                    print(f"drift not measured on {count} settled answers of {entry['point']} {entry['tier']}: "
                          f"{reason} ({entry['compared']} compared)")
        return 0
    if command == "promote":
        named = _fingerprint(home, args.point, args.tier, args.revision) if args.revision is not None else None
        try:
            done = point_state.promote(_writable(home), args.point, args.tier, now=now, by="owner", fingerprint=named)
        except point_state.Refused as error:
            print(f"colonymind instinct promote: refused: {error}", file=sys.stderr)
            return 2
        evidence = done["evidence"]
        answered = evidence["answered"]
        compared = (f"{answered['n']} earlier answers compared, agreement {answered['agreement']:.3f}"
                    if answered["n"] else "no earlier answer to compare")
        print(f"{args.point} {args.tier} is live: record {evidence['record']}, offline pass on "
              f"{evidence['frozen']['frozen']} frozen items, {compared}")
        return 0
    if command == "demote":
        done = point_state.demote(_writable(home), args.point, args.tier, reason=args.reason, now=now, by="owner")
        print(f"{args.point} {args.tier} demoted" if done else f"{args.point} {args.tier} was not live: unchanged")
        return 0
    if command == "records":
        store = _store(home)
        history = store.record_history(point=args.point) if store is not None else []
        if args.json:
            print(json.dumps([{**entry, "record": entry["record"].as_dict(), "id": entry["record"].id}
                              for entry in history], indent=2))
            return 0
        for entry in history:
            record = entry["record"]
            bars = ", ".join(f"{label} {bar:.2f}" for label, bar in sorted(record.bars.items()))
            print(f"{entry['at']}  {record.id}  {record.point} {record.tier} {record.method} n={record.n}  "
                  f"bars: {bars}  ({entry['reason']}"
                  + (f"; replaced {entry['replaced']})" if entry["replaced"] else ")"))
        if not history:
            print("no calibration record put in force here: the shipped ones apply")
        return 0
    if command == "restore":
        try:
            record = _writable(home).restore_record(args.record)
        except KeyError:
            raise ValueError(f"no record {args.record} in the history or among the shipped ones") from None
        print(f"record {record.id} is in force again for {record.point} {record.tier}")
        return 0
    raise ValueError(f"unknown command {command}")


def _fingerprint(home: Path, point_name: str, tier: str, revision: str) -> str:
    """The fingerprint ``tier`` of ``point_name`` answers under when its configured provider serves ``revision``."""
    from . import REGISTRY, calibration
    point = REGISTRY.get(point_name)
    if point is None:
        raise ValueError(f"no point {point_name!r}")
    from .providers import logits
    section = _section(home)
    name = section[tier]
    entry = section["providers"].get(name) if name else None
    if not entry or not entry.get("url"):
        raise ValueError(f"no {tier} provider configured (a provider with no url leaves its tier off)")
    model = entry.get("model", "") or ""
    if entry["kind"] == logits.KIND:            # as the engine builds it (``logits.from_entry``)
        model = model.strip()
        if not model:
            raise ValueError(f"the {tier} reader names no model: its tier is off")
    return calibration.fingerprint(point.version, tier, entry["kind"], model, revision)


def _describe_tier(tier: str, value: Optional[Dict[str, Any]]) -> str:
    if value is None:
        return f"{tier} tier: none configured"
    return (f"{tier} tier: {value['provider']} ({value['kind']})" if value["configured"]
            else f"{tier} tier: {value['provider']} ({value['kind']}) has no url: the tier is off")


def run(args: argparse.Namespace) -> int:
    command = getattr(args, "instinct_command", None)
    home = _instance()
    try:
        if command == "points":
            listing = points_listing(home)
            if args.json:
                print(json.dumps(listing, indent=2))
            else:
                for item in listing:
                    print(f"{item['point']}  version {item['version']}  {item['type']}, effect {item['effect']}  "
                          f"tiers: {', '.join(item['tiers'])}  labels: {', '.join(item['labels'])}")
            return 0
        if command == "status":
            value = status(home)
            if args.json:
                print(json.dumps(value, indent=2))
                return 0
            print(f"Instinct in {value['instance']}")
            for tier in ("fast", "reader"):
                print(_describe_tier(tier, value["tiers"][tier]))
            print("points:")
            for item in value["points"]:
                print(f"  {item['point']}: {', '.join(item['tiers'])}")
            day = value["last_day"]
            by = ", ".join(f"{count} by {tier}" for tier, count in sorted(day["answered_by"].items()))
            print(f"last day: {day['decisions']} decisions" + (f" ({by})" if by else ""))
            print(f"last day: {day['carried']} carried by lineage")
            return 0
        if command == "log":
            rows = log_rows(home, point=args.point, limit=args.limit, since=args.since)
            if args.json:
                print(json.dumps(rows, indent=2))
                return 0
            for row in rows:
                tiers = " ".join(f"{tier}={entry.get('outcome')}({entry.get('state')})"
                                 for tier, entry in row["tiers"].items())
                refs = " ".join(f"{key}={value}" for key, value in row["refs"].items())
                print(f"{row['at']}  {row['id']}  {row['event'] or '-'}  {row['point']}  {row['answered_by']}  "
                      f"{row['label'] or '-'}  {'applied' if row['applied'] else 'not applied'}  {tiers}  {refs}")
            if not rows:
                print("no decisions logged")
            return 0
        if command in {"label", "fit", "promote", "demote", "records", "restore"}:
            return _learning(command, args, home)
    except (KeyError, ValueError) as error:
        print(f"colonymind instinct {command}: {error}", file=sys.stderr)
        return 2
    print("usage: colonymind instinct status|points|log|label|fit|promote|demote|records|restore", file=sys.stderr)
    return 2


__all__ = ["add_parser", "log_rows", "points_listing", "run", "status"]
