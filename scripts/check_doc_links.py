"""Check the links in the repository's Markdown pages: every relative link reaches a file inside the repository,
and every fragment reaches a heading (or an explicit anchor) of the page it names.

Usage: ``python scripts/check_doc_links.py [--root DIR] [--repository-url URL] [PATH ...]``. With no paths it checks
every tracked Markdown file (``git ls-files``: stage a new page to check it), or, outside a checkout, every one under
the root. A broken link prints ``path:line: target (reason)``; any broken link exits 1. In a checkout a link must
also reach a tracked file: one that is untracked or ignored is missing from the published repository.

What counts as a link: inline links and images, reference definitions, and the ``href``, ``src`` and ``srcset``
attributes of HTML in a page. A destination may sit on a later line of its paragraph (a line may break before it and
before its title), in a quote too; a broken one is reported on the line it is on. Code blocks and code spans are not
links. Indentation counts from the content of the list item a line is in, so an item's indented paragraphs are prose
and its code is indented four columns past its content. A link with a scheme is external and never fetched, with one
exception: a link to this repository's own files on its main branch (the package pages use them, since a package
index renders no relative link) is checked against the tree. The repository URL is read from
``[project.urls] Repository`` in the root ``pyproject.toml`` unless given.

A changelog (``CHANGELOG.md``) is checked in its newest release section only: an older section describes the tree
of its own release, where a doc a later release moved or retired still existed.

Heading ids follow the rendered page: lower case, punctuation removed, spaces as hyphens, and a repeated heading
numbered ``-1``, ``-2`` in order.
"""

from __future__ import annotations

import argparse
import bisect
import os
import re
import subprocess
import sys
import tomllib
import unicodedata
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = (".md", ".markdown")
SKIP_DIRS = {"node_modules", "__pycache__", "build", "dist", "venv"}
CURRENT_REFS = {"main", "master", "HEAD"}
HISTORY = {"changelog.md"}
RELEASE_SECTION = re.compile(r"^ {0,3}## ")

FENCE = re.compile(r"(`{3,}|~{3,})")
LIST_ITEM = re.compile(r"([-+*]|(\d{1,9})[.)])(?=[ \t]|$)")
THEMATIC_BREAK = re.compile(r"([-*_])(?:[ \t]*\1){2,}[ \t]*$")
BLOCK_START = re.compile(r"#{1,6}(?:[ \t]|$)|>|`{3,}|~{3,}")
ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
CODE_SPAN = re.compile(r"(`+)(.+?)\1")
REFERENCE = re.compile(r"^ {0,3}\[(?!\^)[^\]]+\]:[ \t]*\n?[ \t]*<?([^\s>]+)>?(?=[ \t]*(?:$|[\"'(]))", re.MULTILINE)
QUOTE = re.compile(r"^(?: {0,3}>[ \t]?)+")
HTML_ATTRIBUTE = re.compile(r"""\b(href|src|srcset)\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.IGNORECASE)
HTML_ANCHOR = re.compile(r"""<[A-Za-z][^>]*?\s(?:name|id)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
INLINE_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
HTML_TAG = re.compile(r"<[^>]+>")


def repository_files(root: Path) -> list[str] | None:
    """Every tracked file of a checkout, relative to the root; None outside a checkout."""
    if not (root / ".git").exists():
        return None
    try:
        done = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return sorted({name for name in done.stdout.decode("utf-8").split("\0") if name and (root / name).is_file()})


def markdown_files(root: Path, files: list[str] | None) -> list[str]:
    """The pages to check, relative to the root: the checkout's Markdown files, or every one under the root."""
    if files is not None:
        return [name for name in files if name.lower().endswith(MARKDOWN)]
    found = []
    for path in root.rglob("*"):
        parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS or part.endswith(".egg-info") or (part.startswith(".") and part != ".github")
               for part in parts[:-1]):
            continue
        if path.is_file() and path.name.lower().endswith(MARKDOWN):
            found.append(path.relative_to(root).as_posix())
    return sorted(found)


def prose_lines(text: str, blank_code: bool = True) -> list[tuple[int, str]]:
    """``(line number, text)`` of every line outside fenced and indented code, with code spans blanked unless
    ``blank_code`` is false (a heading's id keeps its code text).

    Indentation counts from the content of the list item a line is in, as the page renders it: an item's later
    paragraphs are indented to its content (four columns or more under a number or a nested bullet), its indented
    code four columns past that, and a fence in it is code at the item's own indent."""
    lines: list[tuple[int, str]] = []
    fence: tuple[str, int] | None = None   # the open fence's marker, and the column its indent counts from
    items: list[int] = []                  # the content column of each open list item, outermost first
    paragraph = quoted = empty_item = False   # the line before: in a paragraph; in a quote; an empty item's marker
    for number, line in enumerate(text.splitlines(), 1):
        expanded = line.expandtabs(4)
        stripped = expanded.lstrip(" ")
        width = len(expanded) - len(stripped)
        if fence:
            marker, base = fence
            if not stripped.strip() or width >= base:
                closing = FENCE.match(stripped)
                if closing and width - base < 4 and closing.group(1)[0] == marker[0] \
                        and len(closing.group(1)) >= len(marker) and not stripped[len(closing.group(1)):].strip():
                    fence = None
                paragraph = quoted = False
                continue
            fence = None                   # a line left of the item's content ends the item and its fence
            _close_items(items, width)
        if not stripped.strip():
            if empty_item:
                items.pop()                # an item whose marker is followed by a blank line is empty
            lines.append((number, line))
            paragraph = quoted = empty_item = False
            continue
        base = items[-1] if items else 0
        # A block opens on a line indented under four columns past the item it would land in. Right after a line
        # of a paragraph, inside that paragraph's own block, a list marker opens an item only as a bullet or a 1
        # with words after it; any other line there continues the paragraph, lazily when it is left of the item.
        landing = next((column for column in reversed(items) if column <= width), 0)
        opens = width - landing < 4
        opening = FENCE.match(stripped) if opens else None
        item = _list_item(stripped, interrupting=paragraph and width >= base and not quoted) \
            if opens and not THEMATIC_BREAK.match(stripped) else None
        block = opens and bool(BLOCK_START.match(stripped) or THEMATIC_BREAK.match(stripped) or item)
        if not paragraph or block:
            _close_items(items, width)
            base = items[-1] if items else 0
        if not paragraph and width - base >= 4:
            empty_item = False
            continue   # an indented code block: it interrupts no paragraph
        quoted = stripped.startswith(">") or (quoted and paragraph and not block)
        paragraph, empty_item = True, False
        if opening:
            fence, paragraph = (opening.group(1), base), False
            continue
        if item:
            after = stripped[item.end():]
            gap = len(after) - len(after.lstrip(" "))
            content = width + len(item.group(1)) + (gap if after.strip() and gap <= 4 else 1)
            items.append(content)
            paragraph = bool(after.strip()) and gap <= 4   # an empty item, or one opening on code, has none
            empty_item = not after.strip()
            inner = FENCE.match(after.lstrip(" "))
            if inner:
                fence, paragraph = (inner.group(1), content), False
                continue
        elif opens and (ATX.match(stripped) or THEMATIC_BREAK.match(stripped)):
            paragraph = False
        lines.append((number, CODE_SPAN.sub(lambda match: " " * len(match.group(0)), line) if blank_code else line))
    return lines


def _list_item(stripped: str, interrupting: bool) -> re.Match | None:
    """The list marker a line (its indent removed) starts with, or None. Right after a paragraph line only a bullet
    or a 1 with words after it starts an item."""
    item = LIST_ITEM.match(stripped)
    if item and interrupting and (not stripped[item.end():].strip() or item.group(2) not in (None, "1")):
        return None
    return item


def _close_items(items: list[int], width: int) -> None:
    """Close the list items a line indented ``width`` columns is not inside."""
    while items and width < items[-1]:
        items.pop()


def link_targets(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """``(line number, target)`` of every link in a page's prose lines, in page order. A link lies within one
    paragraph, so the lines of each are read together (a quote's markers blanked) and a destination on a later line
    than its link's text is found and numbered by the line it is on."""
    found = []
    for block in _paragraphs(lines):
        text = "\n".join(_unquoted(line) for _, line in block)
        starts, offset = [], 0
        for _, line in block:
            starts.append(offset)
            offset += len(line) + 1
        found += [(block[bisect.bisect_right(starts, position) - 1][0], target)
                  for position, target in _targets(text)]
    return found


def _unquoted(line: str) -> str:
    """The line with its blockquote markers blanked, so positions stay put."""
    return QUOTE.sub(lambda match: " " * len(match.group(0)), line)


def _paragraphs(lines: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    """Runs of adjacent non-blank prose lines. A blank line, code between two lines and the end of a heading all
    end a run: no link continues past them."""
    runs: list[list[tuple[int, str]]] = []
    run: list[tuple[int, str]] = []
    for number, line in lines:
        bare = _unquoted(line)
        if run and (not bare.strip() or number != run[-1][0] + 1 or ATX.match(_unquoted(run[-1][1]))
                    or ATX.match(bare)):
            runs.append(run)
            run = []
        if bare.strip():
            run.append((number, line))
    if run:
        runs.append(run)
    return runs


def _targets(text: str) -> list[tuple[int, str]]:
    """``(position, target)`` of every link destination in one paragraph's text, in order."""
    found = []
    start = text.find("](")
    while start != -1:
        target = _inline_target(text, start + 2)
        if target:
            found.append(target)
        start = text.find("](", start + 2)
    found += [(match.start(1), match.group(1)) for match in REFERENCE.finditer(text)]
    for match in HTML_ATTRIBUTE.finditer(text):
        group = 2 if match.group(2) is not None else 3
        value, position = match.group(group), match.start(group)
        if match.group(1).lower() == "srcset":
            found += [(position, candidate.strip().split()[0]) for candidate in value.split(",") if candidate.strip()]
        elif value:
            found.append((position, value))
    return sorted(found, key=lambda item: item[0])


def _inline_target(text: str, index: int) -> tuple[int, str] | None:
    """``(position, destination)`` of an inline link whose ``(`` ends at ``index``, or None when it is not one.
    One line break may come before the destination and one before its title."""
    while index < len(text) and text[index] in " \t\n":
        index += 1
    if index < len(text) and text[index] == "<":
        end = text.find(">", index)
        if end == -1 or "\n" in text[index:end]:
            return None
        return index + 1, text[index + 1:end]
    depth, begin = 0, index
    while index < len(text):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                break
            depth -= 1
        elif char in " \t\n":
            break
        index += 1
    target = text[begin:index]
    rest = text[index:].lstrip()
    if not target or not rest:
        return None
    if rest[0] == ")" or rest[0] in "\"'(":
        return begin, target
    return None


def slug(text: str) -> str:
    """The id a rendered heading gets (before numbering repeats)."""
    text = INLINE_LINK.sub(lambda match: "" if match.group(0).startswith("!") else match.group(1), text)
    pieces, last = [], 0
    for match in CODE_SPAN.finditer(text):
        pieces.append(_plain(text[last:match.start()]))
        pieces.append(match.group(2))
        last = match.end()
    pieces.append(_plain(text[last:]))
    kept = []
    for char in "".join(pieces).strip().lower():
        if char.isalnum() or char in "-_ " or unicodedata.category(char).startswith("M"):
            kept.append(char)
    return "".join(kept).replace(" ", "-")


def _plain(text: str) -> str:
    """Heading text outside code with its HTML tags and emphasis markers removed."""
    text = HTML_TAG.sub("", text)
    text = re.sub(r"(\*+|~~)", "", text)
    return re.sub(r"(?<![\w])_+(?=\S)|(?<=\S)_+(?![\w])", "", text)


def anchors(text: str) -> set[str]:
    """Every fragment a page answers to: its numbered heading ids and its explicit HTML anchors."""
    seen: dict[str, int] = {}
    found = set()

    def add(heading: str) -> None:
        base = slug(heading)
        name = base
        while name in seen:
            seen[base] += 1
            name = f"{base}-{seen[base]}"
        seen[name] = 0
        found.add(name)

    previous = None
    for number, line in prose_lines(text, blank_code=False):
        heading = ATX.match(line)
        setext = SETEXT.match(line)
        if heading:
            add(heading.group(2) or "")
            previous = None
            continue
        if setext and previous is not None and previous[0] == number - 1:
            add(previous[1])
            previous = None
            continue
        found.update(match.group(1) for match in HTML_ANCHOR.finditer(line))
        stripped = line.strip()
        previous = (number, stripped) if stripped and not stripped.startswith(("|", ">", "-", "*", "+", "<")) \
            and not re.match(r"\d+[.)]\s", stripped) else None
    return found


def repository_urls(root: Path) -> list[str]:
    """The repository's own web address, from the root ``pyproject.toml``."""
    try:
        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []
    url = ((data.get("project") or {}).get("urls") or {}).get("Repository")
    return [url.rstrip("/")] if url else []


def local_path_of(target: str, repositories: list[str]) -> str | None:
    """The repository-relative path a link to this repository's main branch names, or None."""
    for base in repositories:
        prefix = base + "/"
        if not target.startswith(prefix):
            continue
        parts = target[len(prefix):].split("/", 2)
        if len(parts) >= 2 and parts[0] in ("blob", "tree", "raw") and parts[1] in CURRENT_REFS:
            return "/" + (parts[2] if len(parts) == 3 else "")
    return None


class Checker:
    def __init__(self, root: Path, repositories: list[str], files: list[str] | None = None):
        self.root = root.resolve()
        self.repositories = repositories
        self._anchors: dict[Path, set[str]] = {}
        self.published: set[str] | None = None
        if files is not None:
            self.published = set(files)
            for name in files:
                parent = Path(name).parent
                while parent != Path("."):
                    self.published.add(parent.as_posix())
                    parent = parent.parent

    def anchors_of(self, path: Path) -> set[str]:
        if path not in self._anchors:
            self._anchors[path] = anchors(path.read_text(encoding="utf-8"))
        return self._anchors[path]

    def problem(self, page: Path, target: str) -> str | None:
        """Why ``target``, linked from ``page``, is broken; None when it resolves or is not checked."""
        link = target
        if SCHEME.match(target) or target.startswith("//"):
            link = local_path_of(target, self.repositories)
            if link is None:
                return None
        path_part, _, fragment = link.partition("#")
        path_part = unquote(path_part.split("?", 1)[0])
        if not path_part:
            destination = page
        elif path_part.startswith("/"):
            destination = self.root / path_part.lstrip("/")
        else:
            destination = page.parent / path_part
        destination = Path(os.path.normpath(destination))
        if destination != self.root and self.root not in destination.parents:
            return "outside the repository"
        if not destination.exists():
            return "no such file"
        relative = destination.relative_to(self.root).as_posix()
        if self.published is not None and destination != self.root and relative not in self.published:
            return "not tracked"
        if fragment and destination.is_file() and destination.name.lower().endswith(MARKDOWN):
            wanted = unquote(fragment)
            names = self.anchors_of(destination)
            if wanted not in names and wanted.lower() not in names:
                return "no such heading"
        return None

    def check(self, name: str) -> tuple[int, list[tuple[str, int, str, str]]]:
        page = self.root / name
        lines = prose_lines(page.read_text(encoding="utf-8"))
        if page.name.lower() in HISTORY:
            lines = _newest_release(lines)
        broken, count = [], 0
        for number, target in link_targets(lines):
            count += 1
            reason = self.problem(page, target)
            if reason:
                broken.append((name, number, target, reason))
        return count, broken


def _newest_release(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """A changelog's lines before its second release section."""
    sections = 0
    for index, (_, line) in enumerate(lines):
        if RELEASE_SECTION.match(line):
            sections += 1
            if sections > 1:
                return lines[:index]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--repository-url", action="append", default=None,
                        help="this repository's web address (default: [project.urls] Repository)")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    repositories = [url.rstrip("/") for url in args.repository_url] if args.repository_url else repository_urls(root)
    files = repository_files(root)
    checker = Checker(root, repositories, files)
    pages = args.paths or markdown_files(root, files)
    total, broken = 0, []
    for name in pages:
        count, found = checker.check(name)
        total += count
        broken += found
    for name, number, target, reason in broken:
        print(f"{name}:{number}: {target} ({reason})")
    print(f"doc links: {len(broken)} broken of {total} checked in {len(pages)} page(s)")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
