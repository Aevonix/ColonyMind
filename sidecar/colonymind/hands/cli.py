"""Local read-only inspection and authenticated mutations through the sidecar."""

import json
import os
from pathlib import Path
import secrets
import sys

import httpx

from colonymind.config import load_config, read_api_key
from colonymind.reports.instance import open_readonly
from .store import DB_FILE


def add_parser(subparsers):
    parser = subparsers.add_parser("hands", help="Inspect and control hands")
    commands = parser.add_subparsers(dest="hands_command", required=True)
    for command in ("list", "tasks", "obligations", "report"):
        commands.add_parser(command)
    for command in ("show", "log", "stop", "revise"):
        child = commands.add_parser(command)
        child.add_argument("id")
        if command == "stop":
            child.add_argument("--reason", default="Owner stopped this hand")
        if command == "revise":
            child.add_argument("--goal", required=True)
            child.add_argument("--stop", action="store_true")
    parser.set_defaults(hands_handler=run)
    return parser


def read_rows(path, command, ident=None):
    db = open_readonly(Path(path))
    if db is None:
        return []
    try:
        if command == "log":
            rows = db.execute("SELECT * FROM events WHERE hand_id=? ORDER BY seq", (ident,))
        elif command == "show":
            table = "tasks" if ident.startswith("t-") else "hands"
            rows = db.execute(f"SELECT * FROM {table} WHERE id=?", (ident,))
        else:
            table = {"list": "hands", "tasks": "tasks", "obligations": "obligations"}[command]
            rows = db.execute(f"SELECT * FROM {table}")
        from .controller import Controller
        return [Controller.decode(row) for row in rows]
    finally:
        db.close()


def run(args):
    from colonymind.reports.hands import collect, lines
    config = load_config()
    path = config.home / DB_FILE
    command = args.hands_command
    try:
        if command == "report":
            print("\n".join(lines(collect(path=path))))
        elif command in {"stop", "revise"}:
            credential = os.environ.get("COLONYMIND_CLIENT_API_KEY") or read_api_key(config.home)
            if not credential:
                print("An owner instance credential is required.", file=sys.stderr)
                return 1
            sidecar = config.data["sidecar"]
            base = os.environ.get("COLONYMIND_URL") or f"http://{sidecar['host']}:{sidecar['port']}"
            route = f"/v1/hands/{args.id}/stop" if command == "stop" else f"/v1/hands/tasks/{args.id}/revise"
            payload = {"reason": args.reason} if command == "stop" else {"goal": args.goal, "stop": args.stop}
            # Assumption: control requests have a ten-second client deadline.
            response = httpx.post(base + route, json=payload, timeout=10,
                                  headers={"Authorization": "Bearer " + credential,
                                           "Idempotency-Key": "cli:" + secrets.token_hex(12)})
            response.raise_for_status()
            print(json.dumps(response.json(), indent=2))
        else:
            print(json.dumps(read_rows(path, command, getattr(args, "id", None)), indent=2))
        return 0
    except (OSError, ValueError, httpx.HTTPError) as error:
        print(f"Hands command failed: {type(error).__name__}", file=sys.stderr)
        return 1
