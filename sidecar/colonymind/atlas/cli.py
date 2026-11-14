"""``colonymind atlas``: build and check the owner's atlas from the instance's ledger, offline.

It reads the instance's databases directly (read-only), so it works with the sidecar stopped, and
writes only the output directory (default ``<instance>/atlas``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from .site import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, DEFAULT_PAGE_LINES, build_atlas, check_atlas
from .snapshot import AtlasError


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("atlas", help="The atlas: a static, owner-only, offline site of what was said, cited")
    commands = parser.add_subparsers(dest="atlas_command")
    build = commands.add_parser("build", help="Build or refresh it (only changed pages are written)")
    build.add_argument("--out", default=None, help="Output directory (default <instance>/atlas)")
    build.add_argument("--timezone", default=None, help="Months and dates in this IANA zone (default: the agent's)")
    build.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help="Size bound for the whole site")
    build.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES, help="At most this many lines, newest first")
    build.add_argument("--page-lines", type=int, default=DEFAULT_PAGE_LINES, help="Lines per page before the next page")
    build.add_argument("--json", action="store_true", help="Machine-readable report")
    check = commands.add_parser("check", help="Audit it: links, nothing external, owner-only, no erasure since")
    check.add_argument("--out", default=None, help="Output directory (default <instance>/atlas)")
    check.add_argument("--json", action="store_true", help="Machine-readable report")


def _paths() -> Dict[str, Any]:
    from colonymind import get_state_dir
    from colonymind.identity import get_owner_contact_id
    state = get_state_dir()
    contacts = os.environ.get("COLONYMIND_CONTACTS_DB")
    if not contacts:
        contacts = str(state / "contacts.db") if (state / "contacts.db").is_file() else str(state / "colonymind-contacts.db")
    return {"state": state, "ledger": state / "turn-idempotency.db", "contacts": Path(contacts),
            "owner_id": get_owner_contact_id(), "owner_name": os.environ.get("COLONYMIND_OWNER_NAME") or None,
            "timezone": os.environ.get("COLONYMIND_AGENT_TIMEZONE") or os.environ.get("COLONYMIND_TIMEZONE") or "UTC"}


def run(args: argparse.Namespace) -> int:
    command = getattr(args, "atlas_command", None) or "build"
    as_json = bool(getattr(args, "json", False))
    paths = _paths()
    out = Path(getattr(args, "out", None) or paths["state"] / "atlas").expanduser()
    try:
        if command == "check":
            result = check_atlas(out, ledger_path=paths["ledger"] if paths["ledger"].is_file() else None)
            if as_json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print(f"atlas {out}: {result['pages']} pages, {result['links']} links checked, "
                      f"{len(result['problems'])} problems")
                for problem in result["problems"]:
                    print(f"  {problem}")
            return 1 if result["problems"] else 0
        report = build_atlas(
            ledger_path=paths["ledger"], out_dir=out, contacts_path=paths["contacts"], owner_id=paths["owner_id"],
            owner_name=paths["owner_name"], timezone_name=getattr(args, "timezone", None) or paths["timezone"],
            max_bytes=getattr(args, "max_bytes", DEFAULT_MAX_BYTES), max_lines=getattr(args, "max_lines", DEFAULT_MAX_LINES),
            page_lines=getattr(args, "page_lines", DEFAULT_PAGE_LINES))
    except AtlasError as error:
        print(f"atlas: {error}", file=sys.stderr)
        return 2
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        dropped = f", {report['dropped_lines']} older lines left out by the size bound" if report.get("dropped_lines") else ""
        print(f"atlas: {report['lines']} lines on {report['pages']} pages ({report['written']} written, "
              f"{report['unchanged']} unchanged, {report['deleted']} deleted), {report['name_links']} name links, "
              f"{report['bytes']} bytes{dropped}")
        print(f"  open {out / 'index.html'}")
    return 0
