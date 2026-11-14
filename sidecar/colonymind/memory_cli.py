"""``colonymind memory``: what memory holds about a person, from the terminal (docs/SOURCE-ANNOTATIONS.md, The
memory inventory).

``colonymind memory inventory <who>`` lists the person's claims, newest first: each with its state, its quotation, its
claim id and the source revision the annotate and forget routes act on. ``colonymind memory history <claim>`` prints
one claim's own chain, first statement first. Both ask the running sidecar as the instance key, the owner's
credential, and change nothing: corrections still go through annotate, removals through forget.

Both read each claim as the owner console does (``colonymind.secrets.claims``): a value the capture detector would
seal in a claim's words, or in its predicate beside its value (``wifi password: ...``), is withheld wherever the
listing or the history prints it, ``--json`` included.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List
from urllib.parse import quote

from colonymind.secrets.claims import withheld_claims

INVENTORY = "/v1/host/memory/inventory"


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("memory", help="What memory holds about a person: the inventory and a claim's history")
    commands = parser.add_subparsers(dest="memory_command")
    listing = commands.add_parser("inventory", help="A person's claims, newest first, with state, quote and source")
    listing.add_argument("person", help="Whose memory: a contact id")
    listing.add_argument("--limit", type=int, default=50, help="Claims per page (default 50, at most 200)")
    listing.add_argument("--page", default=None, help="Continue a listing from the next page it printed")
    listing.add_argument("--all", action="store_true", help="Every page of the listing")
    listing.add_argument("--json", action="store_true", help="Print the sidecar's answer as JSON")
    history = commands.add_parser("history", help="One claim's own chain: what it replaced, what replaced it")
    history.add_argument("claim", help="A claim id, as the inventory prints it")
    history.add_argument("--as", dest="person", default=None,
                         help="Whose memory holds it (default: the claim's own person)")
    history.add_argument("--json", action="store_true", help="Print the sidecar's answer as JSON")


def run(args: argparse.Namespace) -> int:
    from colonymind.mind.cli import Sidecar
    command = getattr(args, "memory_command", None)
    if command not in ("inventory", "history"):
        print("usage: colonymind memory {inventory,history,reextract} ...", file=sys.stderr)
        return 2
    sidecar = Sidecar(getattr(args, "instance", None))
    try:
        if command == "inventory":
            return _inventory(sidecar, args)
        return _history(sidecar, args)
    except SystemExit as exc:
        if exc.code and not isinstance(exc.code, int):
            print(f"colonymind memory: {exc.code}", file=sys.stderr)
            return 1
        raise


def _inventory(sidecar, args: argparse.Namespace) -> int:
    params: Dict[str, Any] = {"person_id": args.person, "limit": args.limit}
    if not args.all:
        value = sidecar.call("GET", INVENTORY, params={**params, **({"page": args.page} if args.page else {})})
        value = {**value, "claims": withheld_claims(value.get("claims") or [])}
        print(json.dumps(value, indent=2, sort_keys=True) if args.json else "\n".join(render_page(value, args)))
        return 0
    pages: List[Dict[str, Any]] = []
    token = args.page
    while True:
        # A listing the person's memory lost a claim under is refused (409) and reported: run it again.
        value = sidecar.call("GET", INVENTORY, params={**params, **({"page": token} if token else {})})
        pages.append(value)
        token = value.get("next_page")
        if not token:
            break
    whole = {**pages[-1], "claims": withheld_claims([card for value in pages for card in value.get("claims", [])])}
    print(json.dumps(whole, indent=2, sort_keys=True) if args.json else "\n".join(render_page(whole, args)))
    return 0


def _history(sidecar, args: argparse.Namespace) -> int:
    value = sidecar.call("GET", f"{INVENTORY}/claims/{quote(args.claim, safe=':')}/history",
                         params={"person_id": args.person} if args.person else None)
    if isinstance(value, dict):
        value = {**value, "steps": withheld_claims(value.get("steps") or [])}
    print(json.dumps(value, indent=2, sort_keys=True) if args.json else "\n".join(render_history(value)))
    return 0


def _card_lines(card: Dict[str, Any], indent: str = "    ") -> List[str]:
    source = card.get("source") or {}
    label = "owner confirmed" if card.get("verification") == "owner_confirmed" else "unverified"
    lines = [f"[{card.get('state')}] {card.get('predicate')}: {card.get('value')}  ({label})",
             f"{indent}\"{card.get('quote')}\""]
    where = (f"source {source.get('source_id')} version {str(source.get('source_version') or '')[:12]}"
             f" (session {source.get('session_id')})")
    extra = f", {card['corrections']} correction{'' if card['corrections'] == 1 else 's'}" \
        if card.get("corrections") else ""
    lines.append(f"{indent}claim {card.get('claim_id')}, {where}, said {card.get('said_at')}{extra}")
    return lines


def render_page(value: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    claims = value.get("claims") or []
    lines = [f"Memory about {value.get('person_id')}: {len(claims)} claim{'' if len(claims) == 1 else 's'}"
             + ("" if getattr(args, "all", False) else " on this page") + ", newest first."]
    for card in claims:
        lines += ["  " + line for line in _card_lines(card)]
    if value.get("next_page"):
        lines.append(f"More: colonymind memory inventory {value.get('person_id')} --limit {value.get('limit')} "
                     f"--page {value['next_page']}")
    else:
        lines.append("(end of the listing)")
    return lines


def _by(step: Dict[str, Any]) -> str:
    by = step.get("by") or {}
    if by.get("erased"):
        return "(erased)"
    if "ref" in by:
        return f"the owner's answer {by['ref']}" if by.get("owner_answer") else str(by["ref"])
    if "correction" in by:
        return f"{by.get('author')}: \"{by.get('correction')}\" (source {(by.get('source') or {}).get('source_id')})"
    source = (by.get("source") or {}).get("source_id")
    return f"{by.get('predicate')}: {by.get('value')} [{by.get('state')}] \"{by.get('quote')}\" (source {source})"


def render_history(value: Dict[str, Any]) -> List[str]:
    lines = [f"History of {value.get('claim_id')} ({value.get('person_id')}), first statement first:"]
    for number, step in enumerate(value.get("steps") or [], 1):
        event = step.get("event")
        if event == "stated":
            card = step.get("claim")
            text = "(erased)" if card is None else (
                f"{card.get('predicate')}: {card.get('value')} [{card.get('state')}] \"{card.get('quote')}\" "
                f"(source {(card.get('source') or {}).get('source_id')})")
            lines.append(f"{number:>3}. stated: {step.get('claim_id')} {text}")
        elif event == "corrected":
            lines.append(f"{number:>3}. corrected: {step.get('claim_id')} by {_by(step)}, recorded against source "
                         f"{step.get('corrects')}")
        elif event == "preference_change":
            lines.append(f"{number:>3}. preference change ({step.get('status')}) of {step.get('claim_id')} by "
                         f"{_by(step)}")
        else:
            lines.append(f"{number:>3}. {event.replace('_', ' ')}: {step.get('claim_id')} by {_by(step)}")
    for gap in value.get("gaps") or []:
        lines.append(f"(what came {gap.get('side')} {gap.get('claim_id')} was erased with it: the chain may go on)")
    if value.get("limit_reached"):
        lines.append("(the walk stopped at its bound: the chain goes on)")
    return lines


__all__ = ["add_parser", "render_history", "render_page", "run"]
