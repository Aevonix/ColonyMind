"""``colonymind secrets``: the local owner's view of the sealed store (docs/SECRETS.md).

  list [--contact ID]               references: label, scope, rule, dates, uses (never a value)
  show <ref>                        one reference and its uses
  use <ref> to_file <path>          replace the reference in that file with its value (the file stays 0600 if it was)
  use <ref> to_env <NAME>           store the value as NAME in Hermes' own credentials (``save_env_value``,
                                    run in Hermes' interpreter, the value on its standard input)
  release <ref>                     a false positive: the value goes back into memory where the reference stood
  forget <ref>                      the value is deleted; the reference stays
  migrate [--dry-run]               seal what was stored before capture sealed (counts only)

The terminal is the owner's own, so these need no ask; they never print a value.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def _vault():
    from colonymind.secrets.vault import get_vault
    return get_vault()


def _owner() -> str | None:
    try:
        from colonymind.identity import get_owner_contact_id
        return get_owner_contact_id() or None
    except Exception:
        return None


def _entry(ref: str) -> dict:
    entry = _vault().entry(ref)
    if entry is None:
        print(f"no sealed value has the reference {ref}", file=sys.stderr)
        sys.exit(1)
    return entry


def add_parser(sub) -> None:
    secrets_p = sub.add_parser("secrets", help="The sealed store: secrets that arrived in conversation, by reference")
    verbs = secrets_p.add_subparsers(dest="secrets_cmd", required=True)
    listing = verbs.add_parser("list", help="References: label, scope, dates, uses (never a value)")
    listing.add_argument("--contact", default=None, help="Only this person's references")
    listing.add_argument("--json", action="store_true")
    show = verbs.add_parser("show", help="One reference and its uses")
    show.add_argument("ref")
    use = verbs.add_parser("use", help="Put your own value where it is needed, without printing it")
    use.add_argument("ref")
    use.add_argument("how", choices=["to_file", "to_env"])
    use.add_argument("target", help="the file (to_file) or the variable name (to_env)")
    release = verbs.add_parser("release", help="A false positive: the value goes back into memory")
    release.add_argument("ref")
    forget = verbs.add_parser("forget", help="Delete the value; the reference stays")
    forget.add_argument("ref")
    migrate = verbs.add_parser("migrate", help="Seal what was stored before capture sealed")
    migrate.add_argument("--dry-run", action="store_true", help="Count what would be sealed; write nothing")


def cmd_secrets_list(args: "argparse.Namespace") -> None:
    entries = _vault().entries(contact_id=args.contact)
    if getattr(args, "json", False):
        print(json.dumps(entries, indent=2))
        return
    if not entries:
        print("No sealed values.")
        return
    owner = _owner()
    for item in entries:
        whose = "yours" if owner and item["scope_contact"] == owner else item["scope_contact"]
        print(f"{item['ref']}  {item['state']:<11} {item['kind']:<12} {item['label']}  ({whose}, "
              f"{item['created_at'][:16]}, used {item['uses']}x)")


def cmd_secrets_show(args: "argparse.Namespace") -> None:
    entry = _entry(args.ref)
    print(json.dumps({**entry, "use_log": _vault().uses(args.ref)}, indent=2))


def _own_value(ref: str) -> str:
    from colonymind.secrets.vault import SecretForgotten, VaultUnavailable
    entry = _entry(ref)
    if entry["scope_contact"] != _owner():
        print("that value is another person's: it stays sealed", file=sys.stderr)
        sys.exit(1)
    try:
        return _vault().resolve(ref)
    except SecretForgotten:
        print("that value was deleted", file=sys.stderr)
    except VaultUnavailable as error:
        print(f"the value is unavailable: {error}", file=sys.stderr)
    sys.exit(1)


def _write_into_file(path: Path, ref: str, value: str) -> int:
    from colonymind.secrets.refs import references_in
    text = path.read_text(encoding="utf-8")
    spans = [item for item in references_in(text) if item.ref_id == ref]
    if not spans:
        return 0
    for item in reversed(spans):
        text = text[:item.start] + value + text[item.end:]
    mode = path.stat().st_mode & 0o777
    temporary = path.with_name(f".{path.name}.colonymind-tmp")
    fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)
    return len(spans)


def _hermes_python() -> Path | None:
    try:
        from colonymind.doctor import _hermes_python as locate
        return locate()
    except Exception:
        return None


SAVE_ENV = ("import sys\nfrom hermes_cli.config import save_env_value\n"
            "save_env_value(sys.argv[1], sys.stdin.read())\n")


def cmd_secrets_use(args: "argparse.Namespace") -> None:
    value = _own_value(args.ref)
    if args.how == "to_file":
        count = _write_into_file(Path(args.target).expanduser(), args.ref, value)
        if not count:
            print(f"{args.target} does not contain {args.ref}", file=sys.stderr)
            sys.exit(1)
        _vault().note_use(args.ref, "use", "to_file")
        print(f"Filled {count} place(s) in {args.target}.")
        return
    python = _hermes_python()
    if python is None:
        print("Hermes' interpreter was not found; set hermes.python in colonymind.yaml", file=sys.stderr)
        sys.exit(1)
    done = subprocess.run([str(python), "-c", SAVE_ENV, args.target], input=value, text=True,
                          capture_output=True, timeout=60)
    if done.returncode != 0:
        print("Hermes did not store it (its error names no value): "
              + (done.stderr.strip().splitlines() or ["unknown"])[-1][:200], file=sys.stderr)
        sys.exit(1)
    _vault().note_use(args.ref, "use", "to_env")
    print(f"Stored as {args.target} in Hermes' credentials.")


def cmd_secrets_release(args: "argparse.Namespace") -> None:
    from colonymind.secrets.migrate import release
    entry = _entry(args.ref)
    if entry["scope_contact"] != _owner():
        print("that value is another person's: it stays sealed", file=sys.stderr)
        sys.exit(1)
    result = release(args.ref)
    print(f"Released {args.ref}: back in {result['turns']} turn(s) and {result['other_rows']} other row(s).")


def cmd_secrets_forget(args: "argparse.Namespace") -> None:
    _entry(args.ref)
    _vault().forget(args.ref)
    print(f"Deleted the value of {args.ref}; the reference now reads as a value the owner deleted.")


def cmd_secrets_migrate(args: "argparse.Namespace") -> None:
    from colonymind.config import load_config
    from colonymind.secrets.migrate import migrate
    try:
        hermes_home = load_config().hermes_home
    except Exception:
        hermes_home = None
    result = migrate(dry_run=bool(args.dry_run), hermes_home=hermes_home)
    print(json.dumps(result, indent=2, sort_keys=True))


__all__ = ["add_parser", "cmd_secrets_forget", "cmd_secrets_list", "cmd_secrets_migrate", "cmd_secrets_release",
           "cmd_secrets_show", "cmd_secrets_use"]
