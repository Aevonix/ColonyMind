"""Scan the tracked files for details of a particular deployment (CONTRIBUTING.md, The genericity rule).

Usage: ``python scripts/genericity_scan.py [--strict-publication] [--root DIR] [--denylist FILE] [PATH ...]``. With no paths it scans
``git ls-files``. A finding prints ``path:line:column: rule`` and never the matched text; any finding exits 1, and a
denylist that is named but cannot be read or compiled exits 2.

ColonyMind is a generic engine: nothing in it may name a particular deployment, its people, its network, its model
endpoints or its private measurements. The repository's own rules find the generic shapes of such details:

- ``home-path``: a home directory with a user's name in it (``/home/NAME/``, ``/Users/NAME/``), other than the
  placeholder names (user, me, someone, ...);
- ``private-address``: an IPv4 address on a private or shared network (10/8, 172.16/12, 192.168/16, the 100.64/10
  space tailnets use). By default, the conventional example subnets (the first /24 of each private range and
  the second /24 of the 192.168 range) and network-prefix notation are exempt;
- ``tailnet-host``: a tailnet host name (``*.ts.net``);
- ``email``: an email address other than the project's own and those on the reserved example domains (RFC 2606:
  ``example.com``, ``.test``, ``.example``, ``.invalid``, ``.localhost``);
- ``phone``: a North American number written as one (``+1 ...``, ``(NPA) NXX-XXXX``, ``NPA-NXX-XXXX``, or a
  messaging handle's digits) outside the fictional 555 exchange and area code.

A deployment keeps its own words (its people, hosts, addresses, model endpoints and serving aliases, and anything
else it never publishes) in a private denylist outside the repository: one regular expression per line, matched
without regard to case, with ``#`` comments and blank lines ignored. The scan reads the file ``--denylist`` names,
else the one ``COLONYMIND_GENERICITY_DENYLIST`` names, else ``~/.config/colonymind/genericity-denylist`` when it
exists. A private rule is reported by its line number in that file, so neither the scan's output nor a CI log
repeats the words the denylist exists to keep out.

A line carrying ``genericity: example`` is skipped: a deliberate example that only looks like a detail. A real
detail is never marked; it is replaced with a neutral example and kept in the deployment's own files.

``--strict-publication`` reports every private/shared-network IPv4 occurrence, including network-prefix notation
and conventional example subnets, and never skips marked lines, including private denylist matches. Findings
from marked lines or conventional example subnets are labelled ``(example)`` and still fail the scan. Reserved
email domains, fictional phone numbers and placeholder home directories remain safe synthetic fixtures.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
LINE_MARKER = "genericity: example"
DENYLIST_ENV = "COLONYMIND_GENERICITY_DENYLIST"
DEFAULT_DENYLIST = Path("~/.config/colonymind/genericity-denylist")
PROJECT_ADDRESSES = frozenset()
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".mp3", ".mp4", ".wav", ".ogg",
                   ".zip", ".gz", ".tar", ".whl", ".db", ".sqlite", ".sqlite3", ".woff", ".woff2", ".ttf", ".parquet"}

PLACEHOLDER_USERS = frozenset({"user", "username", "me", "you", "someone", "somebody", "private", "example",
                               "runner", "name", "alice", "bob", "shared", "owner", "operator", "admin", "ubuntu"})
HOME_PATH = re.compile(r"(?<![\w$}.\-])/(?:home|Users)/([A-Za-z0-9_][A-Za-z0-9_.-]*)")

SCANNED_NETWORKS = tuple(ipaddress.ip_network(net) for net in
                         ("10." + "0.0.0/8", "172.16." + "0.0/12", "192.168." + "0.0/16", "100.64." + "0.0/10"))
EXAMPLE_NETWORKS = tuple(ipaddress.ip_network(net) for net in
                         ("10." + "0.0.0/24", "172.16." + "0.0/24", "192.168." + "0.0/24", "192.168." + "1.0/24"))
IPV4 = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d{1,3}){3})(?:/(\d{1,2}))?(?![\w]|\.\d)")

TAILNET_HOST = re.compile(r"\b[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9-]+)*\.ts\.net\b", re.IGNORECASE)

# Not an address at a numbered host; a URL's credentials (``https://user:REDACTED_CREDENTIAL@host``) are passed over below.
EMAIL = re.compile(r"(?<![\w.%+-])([A-Za-z0-9._%+-]+)@([A-Za-z0-9-]*[A-Za-z-][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)+)")
EXAMPLE_DOMAINS = frozenset({"example.com", "example.org", "example.net"})
RESERVED_TLDS = frozenset({"test", "example", "invalid", "localhost"})
TOKEN_START = re.compile(r"[\s\"'`(<\[{,;]")
# Messaging handles whose local part is a phone number: the phone rule reads them.
PHONE_DOMAINS = frozenset({"s.whatsapp.net", "c.us", "g.us"})

_NPA = r"[2-9]\d{2}"
_NXX = r"[2-9]\d{2}"
PHONES = (
    # +1, then the area code (bracketed or not) and the number, with or without separators.
    re.compile(rf"(?<![\w+])\+1[ .-]?\(?({_NPA})\)?[ .-]?({_NXX})[ .-]?(\d{{4}})(?![\d.]\d|\d)"),
    # A bracketed area code: (NPA) NXX-XXXX.
    re.compile(rf"(?<![\w(])\(({_NPA})\)[ .-]?({_NXX})[ .-]?(\d{{4}})(?![\d.]\d|\d)"),
    # Dashes only: NPA-NXX-XXXX (spaces or dots alone read as numbers, as in drawing coordinates).
    re.compile(rf"(?<![\w-])({_NPA})-({_NXX})-(\d{{4}})(?![\w-])"),
    # A messaging handle's digits: 1NPANXXXXXX@s.whatsapp.net.
    re.compile(rf"(?<![\w+])1?({_NPA})({_NXX})(\d{{4}})@(?:s\.whatsapp\.net|c\.us)\b"),
)


class DenylistError(Exception):
    """A named denylist cannot be read, or one of its patterns does not compile (the message names its line)."""


Finding = Tuple[int, int, str]


def _home_paths(text: str) -> Iterator[Tuple[int, str]]:
    for match in HOME_PATH.finditer(text):
        if match.group(1).lower() not in PLACEHOLDER_USERS:
            yield match.start(), "home-path"


def _private_addresses(text: str, *, strict_publication: bool = False) -> Iterator[Tuple[int, str]]:
    for match in IPV4.finditer(text):
        try:
            address = ipaddress.ip_address(match.group(1))
        except ValueError:
            continue
        if not any(address in network for network in SCANNED_NETWORKS):
            continue
        if not strict_publication and match.group(2) is not None:
            try:
                ipaddress.ip_network(f"{match.group(1)}/{match.group(2)}", strict=True)
                continue                                            # a network with its prefix, not an address
            except ValueError:
                pass
        example = any(address in network for network in EXAMPLE_NETWORKS)
        if example and not strict_publication:
            continue
        yield match.start(), "private-address (example)" if example else "private-address"


def _tailnet_hosts(text: str) -> Iterator[Tuple[int, str]]:
    for match in TAILNET_HOST.finditer(text):
        yield match.start(), "tailnet-host"


def _emails(text: str) -> Iterator[Tuple[int, str]]:
    for match in EMAIL.finditer(text):
        address = match.group(0).lower()
        domain = match.group(2).lower().rstrip(".")
        if address in PROJECT_ADDRESSES or domain in PHONE_DOMAINS:
            continue
        if domain in EXAMPLE_DOMAINS or any(domain.endswith("." + known) for known in EXAMPLE_DOMAINS):
            continue
        if domain.rsplit(".", 1)[-1] in RESERVED_TLDS:
            continue
        if match.group(1).lower() == "git":                         # an SSH remote (git@host:owner/repo)
            continue
        before = text[max(0, match.start() - 512):match.start()]
        token = TOKEN_START.split(before)[-1]
        if "://" in token:                                          # a URL's credentials, not an address
            continue
        yield match.start(), "email"


def _phones(text: str) -> Iterator[Tuple[int, str]]:
    seen = set()
    for pattern in PHONES:
        for match in pattern.finditer(text):
            if "555" in (match.group(1), match.group(2)) or match.start() in seen:
                continue
            seen.add(match.start())
            yield match.start(), "phone"


GENERIC_RULES = (_home_paths, _private_addresses, _tailnet_hosts, _emails, _phones)


def load_denylist(path: Path) -> List[Tuple[int, "re.Pattern[str]"]]:
    """``(line, pattern)`` of every rule in the private denylist at ``path``."""
    try:
        lines = path.expanduser().read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise DenylistError(f"the denylist {path} cannot be read ({type(error).__name__})") from None
    rules = []
    for number, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rules.append((number, re.compile(line, re.IGNORECASE)))
        except re.error:
            raise DenylistError(f"the denylist's pattern on line {number} does not compile") from None
    return rules


def resolve_denylist(named: Optional[str]) -> Optional[Path]:
    """The denylist to read: the one named on the command line, else in the environment, else the default when it
    exists; None when there is none."""
    if named:
        return Path(named)
    if os.environ.get(DENYLIST_ENV):
        return Path(os.environ[DENYLIST_ENV])
    default = DEFAULT_DENYLIST.expanduser()
    return default if default.is_file() else None


def scan_text(text: str, private=(), *, strict_publication: bool = False) -> List[Finding]:
    """``(line, column, rule)`` in order; strict publication includes examples."""
    starts, offset = [], 0
    lines = text.split("\n")
    for line in lines:
        starts.append(offset)
        offset += len(line) + 1
    hits: List[Tuple[int, str]] = []
    for rule in GENERIC_RULES:
        hits.extend(_private_addresses(text, strict_publication=strict_publication)
                    if rule is _private_addresses else rule(text))
    for number, pattern in private:
        hits.extend((match.start(), f"private rule {number}") for match in pattern.finditer(text))
    findings = []
    for position, rule in hits:
        index = _line_of(starts, position)
        if LINE_MARKER in lines[index]:
            if not strict_publication:
                continue
            if not rule.endswith(" (example)"):
                rule += " (example)"
        findings.append((index + 1, position - starts[index] + 1, rule))
    return sorted(set(findings))


def _line_of(starts: List[int], position: int) -> int:
    low, high = 0, len(starts) - 1
    while low < high:
        middle = (low + high + 1) // 2
        if starts[middle] <= position:
            low = middle
        else:
            high = middle - 1
    return low


def tracked(root: Path) -> List[str]:
    done = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True)
    return [name for name in done.stdout.decode("utf-8").split("\0") if name]


def scan(root: Path, paths: List[str], private=(), *, strict_publication: bool = False) -> List[Tuple[str, int, int, str]]:
    results = []
    for name in paths:
        path = root / name
        if path.suffix.lower() in BINARY_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        results += [(name, line, column, rule) for line, column, rule in
                    scan_text(text, private, strict_publication=strict_publication)]
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--strict-publication", action="store_true",
                        help="report private network literals and marked examples without exemptions")
    parser.add_argument("--denylist", default=None,
                        help=f"the private denylist (default: ${DENYLIST_ENV}, else {DEFAULT_DENYLIST} if present)")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if any((root / name).is_dir() for name in args.paths):
        parser.error("PATH arguments must be files; use --root DIR to scan a directory's tracked files")
    denylist = resolve_denylist(args.denylist)
    try:
        private = load_denylist(denylist) if denylist is not None else []
    except DenylistError as error:
        print(f"genericity scan: {error}", file=sys.stderr)
        return 2
    findings = scan(root, args.paths or tracked(root), private, strict_publication=args.strict_publication)
    for name, line, column, rule in findings:
        print(f"{name}:{line}:{column}: {rule}")
    source = f"{len(private)} private rule(s)" if denylist is not None else "no private denylist"
    print(f"genericity scan: {len(findings)} finding(s) ({source})")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
