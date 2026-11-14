"""Check or set the version, which stands in eight tracked files.

Usage: ``python scripts/set_version.py [--root DIR] --check`` or
``python scripts/set_version.py [--root DIR] NEW_VERSION``. ``PLACES`` names the eight files.

``--check`` prints the version and exits 0 when every place holds the same one and it has the form given below.
Otherwise it exits 1 and says why on standard error: it names each place with what it holds when the places differ,
and the one string they share when that is no version. It changes nothing.

``NEW_VERSION`` is written into every place, then the check runs. In each file only the version inside its one
version line changes; every other byte stays as it was. A version is ``X.Y.Z`` or ``X.Y.ZrcN``, in decimal numbers
without leading zeros: the two forms these places have held so far. Both read back from every place as written. The
plugin manifests hold the version unquoted, where two numeric parts would read as a number, and a build rewrites
another spelling of the same version, so a wheel would no longer say what its manifest says. Any other string is
refused, and so is a place that cannot be found or written, or whose version line does not look as expected. A
refusal says why on standard error, exits 2 and changes no file: every place is read and judged before the first
one is written.

A place is written whole or not at all: its new bytes go into a file beside it, which gets the mode of the place and
then takes its name. When a write fails all the same, the places written before it get their old bytes back the same
way, no file is left beside any place, one line on standard error names the place that failed, and the exit status
is 2. When a place cannot be put back either, the lines name every place with what it now holds; the exit status is 2.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:rc(?:0|[1-9][0-9]*))?")

_HELD = rb"([0-9A-Za-z.+!_-]+)"
#: For each kind of file: what its version line starts with, and the whole line as expected (group 1: the version).
KINDS = {
    "toml": (re.compile(rb"version[ \t]*="), re.compile(rb'version = "' + _HELD + rb'"')),
    "python": (re.compile(rb"__version__[ \t]*="), re.compile(rb'__version__ = "' + _HELD + rb'"')),
    "yaml": (re.compile(rb"version[ \t]*:"), re.compile(rb"version: " + _HELD)),
}

#: Every tracked place that must hold the version, relative to the repository root.
PLACES = (
    ("pyproject.toml", "toml"),
    ("sidecar/pyproject.toml", "toml"),
    ("plugins/hermes-plugin/__init__.py", "python"),
    ("plugins/instinct-gates/__init__.py", "python"),
    ("plugins/hermes-plugin/plugin.yaml", "yaml"),
    ("plugins/instinct-gates/plugin.yaml", "yaml"),
    ("plugins/colonymind-hands/plugin.yaml", "yaml"),
    ("plugins/colonymind-memory/plugin.yaml", "yaml"),
)


class Unusable(Exception):
    """A place that cannot be found, or whose version line does not look as expected."""


def held(root: Path, name: str, kind: str) -> tuple[bytes, int, int]:
    """The bytes of one place, and where the version starts and ends in them."""
    starts, expected = KINDS[kind]
    try:
        data = (root / name).read_bytes()
    except OSError as exc:
        raise Unusable(f"cannot be read ({type(exc).__name__})") from None
    found, offset = [], 0
    for number, line in enumerate(data.splitlines(keepends=True), 1):
        if starts.match(line):
            found.append((number, offset, line.rstrip(b"\r\n")))
        offset += len(line)
    if not found:
        raise Unusable("has no version line")
    if len(found) > 1:
        raise Unusable("has more than one version line: lines " + ", ".join(str(number) for number, _, _ in found))
    number, offset, line = found[0]
    match = expected.fullmatch(line)
    if not match:
        shown = line.decode("utf-8", "replace")
        raise Unusable(f"line {number} does not look as expected: {shown!r}")
    return data, offset + match.start(1), offset + match.end(1)


def holding(root: Path) -> tuple[dict[str, str], bool]:
    """What each place holds, or why nothing can be read from it; and whether every place holds something."""
    holds, usable = {}, True
    for name, kind in PLACES:
        try:
            data, start, end = held(root, name, kind)
            holds[name] = data[start:end].decode("ascii")
        except Unusable as exc:
            holds[name], usable = str(exc), False
    return holds, usable


def check(root: Path) -> int:
    """Print the version when every place holds the same one; otherwise say what the places hold instead."""
    holds, usable = holding(root)
    if usable and len(set(holds.values())) == 1:
        shared = holds[PLACES[0][0]]
        if VERSION.fullmatch(shared):
            print(shared)
            return 0
        print(f"every place holds {shared!r}, which is not a version of the form X.Y.Z or X.Y.ZrcN", file=sys.stderr)
        return 1
    print("the places do not hold one version:", file=sys.stderr)
    for name, value in holds.items():
        print(f"  {name}: {value}", file=sys.stderr)
    return 1


def put(path: Path, data: bytes) -> None:
    """Give the file at ``path`` these bytes, whole or not at all, and keep its mode.

    The bytes go into a new file beside it, which then takes its name. A place that is a link is written where it
    points, and stays a link. When any step fails the file is as it was and the new file is gone.
    """
    path = Path(os.path.realpath(path))
    mode = stat.S_IMODE(os.stat(path).st_mode)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    handle = open(temporary, "xb")   # before the cleanup below: a file that already has this name is not ours
    try:
        with handle:
            handle.write(data)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def set_version(root: Path, version: str) -> int:
    """Write the version into every place and run the check, or refuse and change nothing."""
    if not VERSION.fullmatch(version):
        print(f"refused: {version!r} is not a version of the form X.Y.Z or X.Y.ZrcN", file=sys.stderr)
        return 2
    writes, refused = [], []
    for name, kind in PLACES:
        try:
            data, start, end = held(root, name, kind)
        except Unusable as exc:
            refused.append(f"{name}: {exc}")
            continue
        if os.access(root / name, os.W_OK):
            writes.append((name, data, data[:start] + version.encode("ascii") + data[end:]))
        else:
            refused.append(f"{name}: cannot be written")
    if refused:
        print("refused, no file changed:", file=sys.stderr)
        for reason in refused:
            print(f"  {reason}", file=sys.stderr)
        return 2
    written = []
    for name, old, new in writes:
        if new == old:
            continue
        try:
            put(root / name, new)
        except OSError as exc:
            return put_back(root, written, f"{name} could not be written ({exc.strerror or type(exc).__name__})")
        written.append((name, old))
    return check(root)


def put_back(root: Path, written: list[tuple[str, bytes]], failure: str) -> int:
    """Give the places written before a failed write their old bytes again, and say how that went."""
    stuck = False
    for name, old in reversed(written):
        try:
            put(root / name, old)
        except OSError:
            stuck = True
    if not stuck:
        print(f"failed: {failure}; every place holds what it held before", file=sys.stderr)
        return 2
    print(f"failed: {failure}, and not every place could be put back; the places now hold:", file=sys.stderr)
    for name, value in holding(root)[0].items():
        print(f"  {name}: {value}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(ROOT), metavar="DIR",
                        help="the tree to work on (default: this repository)")
    parser.add_argument("--check", action="store_true", help="print the version, or name what each place holds")
    parser.add_argument("version", nargs="?", metavar="NEW_VERSION", help="the version to write into every place")
    args = parser.parse_args(argv)
    if args.check == (args.version is not None):
        parser.error("give either --check or NEW_VERSION")
    root = Path(args.root)
    return check(root) if args.check else set_version(root, args.version)


if __name__ == "__main__":
    sys.exit(main())
