"""``colonymind credentials``: issue, bind and retire the API credentials of an instance.

The instance key (``api.key``) is the Hermes adapter's. Every other client (an integration, a voice
gateway, a script) gets its own credential, bound in ``colonymind.yaml`` ``credentials`` to what it may
write as (``colonymind.api.credentials``, docs/CREDENTIALS.md). Only a secret's SHA-256 digest is kept;
the secret itself is written once, to a private file or the terminal, and never to a log. No credential
approves an ask or a grant: that is the owner's own message in their chat (docs/MIND.md, Asks), so there is
nothing here to mint one with. The one exception is an owner channel relay's ``answer`` credential, which records
the owner's explicit answer to a hands permit ask (docs/hands-permit-answer-contract.md); issuing one is the
owner's decision, and the instance key never gets it. A hub's ``observe`` credential posts the owner's outside
observations for the sources it lists (``--observe-source``) and reaches nothing else.
"""

from __future__ import annotations

import argparse
import copy
import os
import secrets
from pathlib import Path
from typing import Any

from colonymind import config as configuration
from colonymind.api.credentials import (
    ANSWER,
    INSTANCE_CREDENTIAL,
    OBSERVE,
    OPERATIONS,
    CredentialTable,
    describe,
    secret_digest,
    terminal_owner,
    validate_credentials,
)
from colonymind.config import ConfigError

RESTART = "restart the sidecar to apply: colonymind service restart"
ACCESS_HELP = ("What it may do (repeat per operation): read (context and memory), write (turns), manage (the rest "
               "of the API), note (notes through the MCP endpoint), answer (alone: an owner channel relay's reads and "
               "answers of the hands permit asks, with --owner-platform for that channel), observe (alone: a hub's posts "
               "of outside observations, with --owner-platform and --observe-source); by default read and write, "
               "and manage too with --owner-platform; tool-origin (alone, without an owner platform: "
               "claim tool origins and request external-write owner actions); hands (alone: manage /v1/hands/*, "
               "optional --owner-platform companion enables owner-started tasks and Hand question answers; "
               "cannot answer permit asks)")
OBSERVE_SOURCE_HELP = "A source an observe credential posts observations for (repeat per source)"
TERMINAL_CHECK = ("  'colonymind doctor' says whether the agent's tools run on this machine; if they do, leave the "
                  "terminal out (--owner-platform for each other platform)")


def _say(message: str = "") -> None:
    print(message, flush=True)


def _home(args) -> Path:
    return configuration.instance_home(getattr(args, "home", None))


def _raw(home: Path) -> dict[str, Any]:
    path = home / configuration.CONFIG_FILE
    if not path.is_file():
        raise ConfigError(f"{path} does not exist; run 'colonymind init' first")
    return configuration.read_raw(path)


def _with_credentials(home: Path, credentials: dict[str, Any]) -> dict[str, Any]:
    """``colonymind.yaml`` as written with ``credentials`` replaced, validated with the rest of the file."""
    raw = _raw(home)
    raw["credentials"] = validate_credentials(copy.deepcopy(credentials))
    configuration.validate(configuration._merge(configuration.DEFAULTS, copy.deepcopy(raw)))
    return raw


def write_credentials(home: Path, credentials: dict[str, Any]) -> None:
    """Replace ``credentials`` in ``colonymind.yaml``, keeping every other key as written."""
    raw = _with_credentials(home, credentials)
    configuration._atomic_write(home / configuration.CONFIG_FILE,
                                configuration.dump_config(raw).encode("utf-8"), mode=0o600)


def _current(home: Path) -> dict[str, Any]:
    return validate_credentials(_raw(home).get("credentials"))


def _line(table: CredentialTable, name: str) -> str:
    grant = table.get(name)
    return f"{name} {describe(grant)}"


def _platforms(values: Any) -> list[str] | None:
    if values is None:
        return None
    return [str(value) for value in values]


def _access(args) -> list[str] | None:
    """``--access``, as given (None: keep the credential's, or the default for a new one)."""
    values = getattr(args, "access", None)
    return None if values is None else list(dict.fromkeys(str(value) for value in values))


def _observe_sources(args) -> list[str] | None:
    """``--observe-source``, as given (None: keep the credential's)."""
    values = getattr(args, "observe_source", None)
    return None if values is None else list(dict.fromkeys(str(value) for value in values))


def _write_secret(path: Path, secret: str) -> None:
    path = path.expanduser()
    if path.exists():
        raise ConfigError(f"{path} already exists; choose another --key-file")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="ascii") as stream:
        stream.write(secret + "\n")
    os.chmod(path, 0o600)


def _granted(table: CredentialTable, name: str) -> None:
    """What a new or changed grant means, said once after it: the terminal among its owner platforms."""
    grant = table.get(name)
    if grant is not None and terminal_owner(grant) and grant.contacts != "own":
        _say(TERMINAL_CHECK)


def _list(home: Path) -> int:
    table = CredentialTable.from_config(_current(home))
    for grant in table.grants():
        _say(f"{grant.name} {describe(grant)}")
    return 0


def _add(home: Path, args) -> int:
    name = str(args.name or "").strip()
    credentials = _current(home)
    if name == INSTANCE_CREDENTIAL:
        raise ConfigError(f"{INSTANCE_CREDENTIAL} is the instance key (api.key); give the new credential "
                          "another name")
    if name in credentials:
        raise ConfigError(f"a credential named {name} exists; 'colonymind credentials set' changes its grants")
    owner = _platforms(getattr(args, "owner_platform", None)) or []
    access = _access(args)
    # A credential that answers asks (an owner channel's relay) or posts observations (a hub) writes as no one: its
    # contacts are its own.
    alone = access is not None and (ANSWER in access or OBSERVE in access)
    contacts = getattr(args, "contacts", None) or ("all" if owner and not alone else "own")
    secret = secrets.token_urlsafe(32)
    credentials[name] = {"sha256": secret_digest(secret), "owner": owner, "contacts": contacts}
    if access is not None:
        credentials[name]["access"] = access
    sources = _observe_sources(args)
    if sources is not None:
        credentials[name]["observe_sources"] = sources
    _with_credentials(home, credentials)       # refuse before the secret is written anywhere
    key_file = getattr(args, "key_file", None)
    if key_file:
        _write_secret(Path(key_file), secret)
    write_credentials(home, credentials)
    table = CredentialTable.from_config(_current(home))
    _say(f"credential {_line(table, name)}")
    _granted(table, name)
    if key_file:
        _say(f"  its secret is in {Path(key_file).expanduser()} (mode 600); it is shown nowhere else")
    else:
        _say("  its secret, shown this once (only its digest is kept):")
        _say(f"  secret: {secret}")
    _say("  " + RESTART)
    return 0


def _set(home: Path, args) -> int:
    name = str(args.name or "").strip()
    credentials = _current(home)
    if name not in credentials:
        raise ConfigError(f"no credential named {name}")
    entry = dict(credentials[name])
    owner = _platforms(getattr(args, "owner_platform", None))
    if getattr(args, "no_owner", False):
        owner = []
    if owner is not None:
        entry["owner"] = owner
    if getattr(args, "contacts", None):
        entry["contacts"] = args.contacts
    access = _access(args)
    if access is not None:
        entry["access"] = access
        if OBSERVE not in access:
            entry.pop("observe_sources", None)
    sources = _observe_sources(args)
    if sources is not None:
        entry["observe_sources"] = sources
    credentials[name] = entry
    write_credentials(home, credentials)
    table = CredentialTable.from_config(_current(home))
    _say(f"credential {_line(table, name)}")
    _granted(table, name)
    _say("  " + RESTART)
    return 0


def _remove(home: Path, args) -> int:
    name = str(args.name or "").strip()
    credentials = _current(home)
    if name == INSTANCE_CREDENTIAL:
        raise ConfigError("the instance key stays; replace api.key to rotate it")
    if name not in credentials:
        raise ConfigError(f"no credential named {name}")
    del credentials[name]
    write_credentials(home, credentials)
    _say(f"credential {name} removed; its secret is no longer accepted once the sidecar restarts")
    _say("  " + RESTART)
    return 0


def run(args) -> int:
    """Returns the exit code."""
    try:
        home = _home(args)
        command = getattr(args, "credentials_command", None) or getattr(args, "command", None)
        if command == "add":
            return _add(home, args)
        if command == "set":
            return _set(home, args)
        if command == "remove":
            return _remove(home, args)
        return _list(home)
    except (ConfigError, OSError) as exc:
        _say(f"colonymind credentials failed: {exc}")
        return 1


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("credentials", help="API credentials and who each may write as")
    parser.add_argument("--home", default=None, help="Instance directory (default: $COLONYMIND_HOME or ~/.colonymind)")
    commands = parser.add_subparsers(dest="credentials_command")
    commands.add_parser("list", help="Every credential and what it may write as")
    add = commands.add_parser("add", help="Issue a credential; by default it writes only as its own identity")
    add.add_argument("name", help="Lowercase letters, digits, '.', '_' or '-'")
    add.add_argument("--owner-platform", action="append", metavar="PLATFORM",
                     help="A platform it may write the owner's turns from (repeat per platform)")
    add.add_argument("--contacts", choices=("all", "own"),
                     help="all: anyone the sidecar resolves; own: only credential:<name> (the default "
                          "without --owner-platform)")
    add.add_argument("--key-file", help="Write the secret to this new file (mode 600) instead of the terminal")
    add.add_argument("--access", action="append", choices=OPERATIONS, help=ACCESS_HELP)
    add.add_argument("--observe-source", action="append", metavar="SOURCE", help=OBSERVE_SOURCE_HELP)
    change = commands.add_parser("set", help="Change what a credential may write as")
    change.add_argument("name")
    change.add_argument("--owner-platform", action="append", metavar="PLATFORM",
                        help="The platforms it may write the owner's turns from (replaces the list)")
    change.add_argument("--no-owner", action="store_true", help="It never writes as the owner")
    change.add_argument("--contacts", choices=("all", "own"))
    change.add_argument("--access", action="append", choices=OPERATIONS, help=ACCESS_HELP + " (replaces the list)")
    change.add_argument("--observe-source", action="append", metavar="SOURCE",
                        help=OBSERVE_SOURCE_HELP + " (replaces the list)")
    remove = commands.add_parser("remove", help="Retire an issued credential")
    remove.add_argument("name")
