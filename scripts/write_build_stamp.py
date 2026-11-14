"""Write the build stamp: the commit, the branch and whether the tree had uncommitted changes.

Usage: ``python scripts/write_build_stamp.py [--root DIR]`` in a git checkout, or
``python scripts/write_build_stamp.py [--root DIR] --commit ID --branch NAME (--clean | --dirty)`` for a tree that
is not one, such as a copy made by hand.

The stamp is one JSON file, ``sidecar/colonymind/build_stamp.json`` in the tree, with exactly three facts: the full
commit id, the branch name (empty when the tree is detached) and whether the tree had uncommitted changes, where a
file git does not know counts as a change. ``colonymind doctor`` shows them, from the source tree or from a wheel
built from it. The file is listed in ``.gitignore`` and is never tracked.

In a git checkout the facts come from git and from nowhere else: git is asked about the tree's own repository,
whatever repository the environment names (see :func:`own_environment`). A tree that is not the top of a checkout
takes them from the arguments, all three of them: ``--branch ""`` says the tree was detached. Nothing is guessed. A
branch name, from git or from ``--branch``, is a fact only when it has at most 255 characters and every one of them
prints. Without git and without arguments, with arguments in a checkout, with a fact missing, with a commit that is
not a full id, or with a branch name that is no such name or that git gives in bytes that are not UTF-8, the script
says why in one line on standard error, exits 2 and writes nothing. The file is put in place in one step, so a
reader sees the stamp that was there before or the new one, never a part of one, and no other file is touched: the
file the stamp is first written into, ``build_stamp.json.PID.tmp`` beside it, takes its name or is removed, and git
ignores that name too.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Where the stamp stands in a tree. ``sidecar/colonymind/build_stamp.py`` reads the file of this name beside itself.
STAMP = "sidecar/colonymind/build_stamp.json"

COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")

#: The longest branch name a stamp holds. ``sidecar/colonymind/build_stamp.py`` reads no longer one.
BRANCH_LENGTH = 255


class Refused(Exception):
    """The facts cannot be had without a guess, or the stamp cannot be written."""


def is_branch(name: str) -> bool:
    """Whether the name can stand in a stamp: every character prints, and it is not too long. Empty: detached."""
    return name.isprintable() and len(name) <= BRANCH_LENGTH


def git(root: Path, env: dict | None, *arguments: str) -> subprocess.CompletedProcess | None:
    """One git command that only reads, run in the tree; ``None`` when there is no git to run.

    What git prints comes back in bytes: a path or a branch name need not be text in any one encoding.
    """
    try:
        return subprocess.run(["git", "--no-optional-locks", "-C", str(root), *arguments], env=env,
                              capture_output=True)
    except OSError:
        return None


def own_environment(root: Path) -> dict | None:
    """The environment git gets when it is asked for the facts; ``None`` when git names no variable to leave out.

    It is the caller's environment without the variables that belong to one repository, such as ``GIT_DIR``,
    ``GIT_WORK_TREE`` and ``GIT_INDEX_FILE``: with one of them set, git would answer for the repository it names and
    not for the tree. Their names are asked from git, ``git rev-parse --local-env-vars``, each time, and are not
    listed in this file: the list belongs to the git that then answers, and a copy of it here would be right only
    for the git version it was copied from. That one question is put in the caller's environment; it reads no
    repository.
    """
    listed = git(root, None, "rev-parse", "--local-env-vars")
    if listed is None or listed.returncode != 0:
        return None
    names = set(listed.stdout.decode("ascii", "replace").split())
    if "GIT_DIR" not in names:
        return None  # not the list: a git too old to know the question prints the question back
    return {name: value for name, value in os.environ.items() if name not in names}


def from_git(root: Path) -> dict | None:
    """The three facts as git gives them, or ``None`` when the tree is not the top of a git checkout."""
    env = own_environment(root)
    if env is None:
        return None
    top = git(root, env, "rev-parse", "--show-toplevel")
    if top is None or top.returncode != 0:
        return None
    try:
        if not os.path.samefile(os.fsdecode(top.stdout).rstrip("\n"), root):
            return None  # the tree lies inside another checkout, whose commit is not this tree's
    except OSError:
        return None
    commit = git(root, env, "rev-parse", "--verify", "HEAD")
    branch = git(root, env, "symbolic-ref", "--short", "-q", "HEAD")
    status = git(root, env, "status", "--porcelain", "--untracked-files=normal")
    named = "" if commit is None else commit.stdout.decode("ascii", "replace").strip()
    if commit is None or commit.returncode != 0 or not COMMIT.fullmatch(named):
        raise Refused("git names no commit for this tree")
    if branch is None or branch.returncode not in (0, 1):  # 1: no branch, the tree is detached
        raise Refused("git does not say which branch this tree is on")
    if status is None or status.returncode != 0:
        raise Refused("git does not say whether this tree has uncommitted changes")
    try:
        name = branch.stdout.decode("utf-8").rstrip("\n") if branch.returncode == 0 else ""
    except UnicodeDecodeError:
        raise Refused("the branch name git gives is not UTF-8") from None
    if not is_branch(name):
        raise Refused(f"the branch name git gives has more than {BRANCH_LENGTH} characters or one that does not print")
    return {"commit": named, "branch": name, "dirty": bool(status.stdout.strip())}


def facts(root: Path, commit: str | None, branch: str | None, dirty: bool | None) -> dict:
    """The three facts, from git in a checkout and from the arguments in any other tree."""
    given = {"--commit": commit, "--branch": branch, "--clean or --dirty": dirty}
    missing = [name for name, value in given.items() if value is None]
    stamp = from_git(root)
    if stamp is not None:
        if len(missing) < len(given):
            raise Refused("this tree is a git checkout, which gives the facts itself: give none as arguments")
        return stamp
    if len(missing) == len(given):
        raise Refused("this tree is not the top of a git checkout that git can read: "
                      "give --commit ID, --branch NAME and --clean or --dirty")
    if missing:
        raise Refused("a tree that is not a git checkout needs every fact: no " + ", no ".join(missing))
    if not COMMIT.fullmatch(commit):
        raise Refused(f"--commit takes the full commit id, not {commit!r}")
    if not is_branch(branch):
        raise Refused(f"--branch takes a name of at most {BRANCH_LENGTH} characters that all print")
    return {"commit": commit, "branch": branch, "dirty": dirty}


def write_whole(path: Path, text: str) -> None:
    """Put the file in place in one step: a reader sees the file that was there or the new one, never a part."""
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        handle = open(temporary, "x", encoding="utf-8")
    except OSError as exc:
        raise Refused(f"{path} cannot be written ({type(exc).__name__})") from None
    try:
        with handle:
            handle.write(text)
        os.replace(temporary, path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise Refused(f"{path} cannot be written ({type(exc).__name__})") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(ROOT), metavar="DIR", help="the tree to stamp (default: this repository)")
    parser.add_argument("--commit", metavar="ID", help="the full commit id, for a tree that is not a git checkout")
    parser.add_argument("--branch", metavar="NAME", help='its branch name, "" when the tree was detached')
    state = parser.add_mutually_exclusive_group()
    state.add_argument("--clean", dest="dirty", action="store_const", const=False,
                       help="that tree had no uncommitted changes")
    state.add_argument("--dirty", dest="dirty", action="store_const", const=True,
                       help="that tree had uncommitted changes")
    args = parser.parse_args(argv)
    root = Path(args.root)
    path = root / STAMP
    try:
        if not path.parent.is_dir():
            raise Refused(f"{root} has no {Path(STAMP).parent} directory")
        stamp = facts(root, args.commit, args.branch, args.dirty)
        write_whole(path, json.dumps(stamp, indent=2) + "\n")
    except Refused as exc:
        print(f"refused, nothing written: {exc}", file=sys.stderr)
        return 2
    said = (f"{path}: commit {stamp['commit']}, branch {stamp['branch'] or '(none: detached)'}, "
            + ("changed" if stamp["dirty"] else "clean"))
    # The stamp is written. Standard output may not have every character of the path or of the branch name: such a
    # character is printed as an escape, so that this line cannot fail on its own text.
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(said.encode(encoding, "backslashreplace").decode(encoding))
    return 0


if __name__ == "__main__":
    sys.exit(main())
