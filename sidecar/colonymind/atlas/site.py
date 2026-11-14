"""Build and check the atlas directory: incremental, size-bounded, owner-only, erasure-honouring.

``build_atlas`` renders the whole site in memory from one ledger snapshot and then writes only the
pages whose bytes changed and deletes every page the new snapshot no longer produces (an erased
source's page and every quote of it). One build runs at a time per directory (a second waits), and
the manifest it starts from is read under that exclusion. The manifest lists every page the build
may leave, written before any page; beyond it, every atlas-named page on disk (the four top pages
and their numbered continuations, and the pages in the atlas's own folders) that the new snapshot
does not produce is deleted, so no stale or hand-edited manifest can shelter a quote an erasure
should remove. Nothing else is ever deleted, and a directory that holds files but no manifest is
refused.

``check_atlas`` audits a built atlas from its files alone: every link resolves to a page and an
anchor inside it, nothing reaches outside it (no scheme, no absolute path, no script, image,
stylesheet link or frame), every page matches the hash it was written with and is readable only
by its owner, and, given the ledger, no source was erased or reattributed after the build.
"""
from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import stat
import tempfile
import time
from contextlib import closing, contextmanager, suppress
from dataclasses import astuple
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple
from urllib.parse import unquote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .render import Rendered, Site
from .snapshot import AtlasError, Snapshot, erasure_mark, open_read_only, read_snapshot

RENDER_VERSION = "atlas-v1"
MANIFEST = "atlas-manifest.json"
LOCK = ".atlas.lock"
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_LINES = 50_000
DEFAULT_PAGE_LINES = 400
SNAPSHOT_ATTEMPTS = 3
_PAGE = re.compile(r"^(?:(?:people|topics|sources|months)/)?[A-Za-z0-9-]+\.html$")
_TOP_PAGE = re.compile(r"^(?:index|people|months|topics)(?:-[0-9]+)?\.html$")
_FOLDER_PAGE = re.compile(r"^[A-Za-z0-9-]+\.html$")
_FORBIDDEN_TAGS = {"script", "img", "link", "iframe", "frame", "frameset", "object", "embed", "form", "base",
                   "audio", "video", "source", "track", "picture", "svg", "math", "applet", "portal"}
_REFERENCE_ATTRS = {"src", "srcset", "action", "formaction", "poster", "data", "background", "cite", "ping",
                    "xlink:href", "manifest", "longdesc", "codebase", "archive"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _zone(name: Optional[str]):
    try:
        return ZoneInfo(name or "UTC")
    except (ZoneInfoNotFoundError, ValueError):
        raise AtlasError(f"unknown timezone {name!r}") from None


def _read_manifest(out: Path) -> Optional[Dict[str, Any]]:
    path = out / MANIFEST
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text())
        if not isinstance(manifest, dict) or not isinstance(manifest.get("pages"), dict):
            raise ValueError("no page list")
        return manifest
    except (OSError, ValueError) as error:
        raise AtlasError(f"the atlas manifest in {out} is unreadable ({type(error).__name__}); "
                         "delete the directory to rebuild it") from None


FOLDERS = ("people", "topics", "sources", "months")


def _own_folders(out: Path, pages: Dict[str, bytes]) -> None:
    """Each atlas folder is a plain directory inside ``out``: a link planted in its place (to anywhere
    else) is refused before anything is written or removed, and the folders a build needs are made."""
    needed = {path.split("/", 1)[0] for path in pages if "/" in path}
    for name in FOLDERS:
        folder = out / name
        if name in needed:
            try:
                os.mkdir(folder, 0o700)
            except FileExistsError:
                pass
        try:
            mode = os.lstat(folder).st_mode
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(mode):
            raise AtlasError(f"{folder} is not a plain directory; the atlas writes only into folders it made")
        os.chmod(folder, 0o700)


def _private_file(path: Path, data: bytes) -> None:
    """Replace ``path`` atomically with a 0600 file holding ``data``. The temporary file has a fresh,
    unpredictable name and is created exclusively, so no link planted in the directory can redirect
    the write; the replace swaps a link at ``path`` itself, never what it points to."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _file_sha(path: Path) -> Optional[str]:
    """The hash of a plain file at ``path``; a link or anything else is never taken as a page."""
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return None
        return _sha(path.read_bytes())
    except OSError:
        return None


def _pages_on_disk(out: Path) -> Set[str]:
    """Every atlas-named page in ``out``, tracked or not."""
    found = {entry.name for entry in os.scandir(out) if _TOP_PAGE.match(entry.name)}
    for name in FOLDERS:
        folder = out / name
        with suppress(FileNotFoundError):
            if stat.S_ISDIR(os.lstat(folder).st_mode):
                found |= {f"{name}/{entry.name}" for entry in os.scandir(folder) if _FOLDER_PAGE.match(entry.name)}
    return found


@contextmanager
def _exclusive(out: Path) -> Iterator[None]:
    """One build at a time in ``out``: a second build waits until the first is done. Without POSIX
    file locks there is no exclusion, so no build (imported here: the CLI's help must not need them)."""
    try:
        import fcntl
    except ImportError:
        raise AtlasError("the atlas needs POSIX file locks to build") from None
    try:
        descriptor = os.open(out / LOCK, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except OSError as error:
        raise AtlasError(f"cannot lock {out} ({type(error).__name__})") from None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _claim_directory(out: Path) -> Optional[Dict[str, Any]]:
    """The output directory is new, empty or an atlas (it has a manifest); never anything else."""
    if out.exists():
        if not out.is_dir():
            raise AtlasError(f"{out} is not a directory")
        manifest = _read_manifest(out)
        if manifest is None and any(entry.name != LOCK and not entry.name.startswith(f".{MANIFEST}.")
                                    for entry in os.scandir(out)):
            raise AtlasError(f"{out} holds files and no atlas manifest; choose a new or empty directory")
    else:
        out.mkdir(mode=0o700, parents=True)
        manifest = None
    os.chmod(out, 0o700)
    return manifest


def _manifest_json(meta: Dict[str, Any], pages: Dict[str, str], state: str) -> bytes:
    return json.dumps({**meta, "state": state, "pages": dict(sorted(pages.items()))}, indent=1, sort_keys=True).encode()


def _publish(out: Path, pages: Dict[str, bytes], old: Optional[Dict[str, Any]], meta: Dict[str, Any]) -> Tuple[int, int, int]:
    old_pages = {path: digest for path, digest in (old or {}).get("pages", {}).items() if _PAGE.match(str(path))}
    hashes = {path: _sha(data) for path, data in pages.items()}
    _own_folders(out, pages)
    # Every page this run may leave on disk is tracked before the first one is written.
    _private_file(out / MANIFEST, _manifest_json(meta, {**old_pages, **hashes}, "building"))
    for folder in [out, *(out / name for name in FOLDERS)]:
        for stray in folder.glob(".*.tmp") if folder.is_dir() else ():
            stray.unlink()
    written = unchanged = deleted = 0
    for path in sorted(pages):
        target = out / path
        if _file_sha(target) == hashes[path]:
            if stat.S_IMODE(target.stat().st_mode) & 0o077:
                os.chmod(target, 0o600)
            unchanged += 1
            continue
        _private_file(target, pages[path])
        written += 1
    for path in sorted((set(old_pages) | _pages_on_disk(out)) - set(pages)):
        with suppress(FileNotFoundError):
            (out / path).unlink()
            deleted += 1
    _private_file(out / MANIFEST, _manifest_json(meta, hashes, "complete"))
    return written, unchanged, deleted


def _fingerprint(snapshot: Snapshot, options: Dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps({"version": RENDER_VERSION, "options": options}, sort_keys=True).encode())
    for line in snapshot.lines:
        digest.update(json.dumps(astuple(line), ensure_ascii=False).encode())
        digest.update(snapshot.messages[(line.turn_id, line.message_hash)].encode())
    speakers = sorted({line.speaker for line in snapshot.lines})
    digest.update(json.dumps([[speaker, snapshot.names.get(speaker)] for speaker in speakers]).encode())
    digest.update(json.dumps(sorted([list(key), value] for key, value in snapshot.notes.items())).encode())
    digest.update(json.dumps({"omitted": snapshot.omitted}).encode())
    return digest.hexdigest()


def _unchanged(out: Path, manifest: Optional[Dict[str, Any]], fingerprint: str, mark: List[int]) -> bool:
    """Nothing to do: the same content, the same erasure mark (an erasure that removes no line still
    moves the mark the check compares against), and the pages on disk exactly as the manifest says."""
    if not manifest or manifest.get("state") != "complete" or manifest.get("fingerprint") != fingerprint:
        return False
    if manifest.get("erasure_mark") != mark:
        return False
    if _pages_on_disk(out) != set(manifest["pages"]):
        return False
    return all(_PAGE.match(path) and _file_sha(out / path) == digest for path, digest in manifest["pages"].items())


def _fit(snapshot: Snapshot, site_options: Dict[str, Any], *, max_lines: int, max_bytes: int,
         overhead: Callable[[int, Rendered], int]) -> Tuple[int, Rendered, int]:
    """The most lines, newest first, whose whole site (its pages and ``overhead``, the manifest that
    lists them) fits ``max_bytes``: the first render when it fits, else a search seeded by the byte
    ratio, to within 0.5% of the lines."""
    lines = snapshot.lines

    def render(keep: int) -> Tuple[Rendered, int]:
        rendered = Site(lines[:keep], names=snapshot.names, messages=snapshot.messages, notes=snapshot.notes,
                        dropped=len(lines) - keep + snapshot.omitted, **site_options).render()
        return rendered, sum(len(data) for data in rendered.pages.values()) + overhead(keep, rendered)

    keep = min(len(lines), max(0, max_lines))
    rendered, size = render(keep)
    if size <= max_bytes:
        return keep, rendered, size
    best: Optional[Tuple[int, Rendered, int]] = None
    low, high = 0, keep                                   # high never fits; low is assumed to
    guess = int(keep * max_bytes / size)
    while high - low > max(1, keep // 200):
        middle = guess if low < guess < high else (low + high) // 2
        guess = -1
        rendered, size = render(middle)
        if size <= max_bytes:
            low, best = middle, (middle, rendered, size)
        else:
            high = middle
    if best is None or best[0] != low:
        rendered, size = render(low)
        if size > max_bytes:
            raise AtlasError(f"even an empty atlas is {size} bytes, over the {max_bytes}-byte bound")
        best = (low, rendered, size)
    return best


def build_atlas(*, ledger_path: Path | str, out_dir: Path | str, contacts_path: Path | str | None = None,
                owner_id: Optional[str] = None, owner_name: Optional[str] = None,
                timezone_name: Optional[str] = "UTC", now: Optional[datetime] = None,
                max_bytes: int = DEFAULT_MAX_BYTES, max_lines: int = DEFAULT_MAX_LINES,
                page_lines: int = DEFAULT_PAGE_LINES) -> Dict[str, Any]:
    """Build or refresh the atlas in ``out_dir`` and return the build report (counts only, no content)."""
    from colonymind.util.temporal import now_utc

    started = time.monotonic()
    tz = _zone(timezone_name)
    now = now or now_utc()
    out = Path(out_dir).expanduser()
    ledger = Path(ledger_path)
    contacts = Path(contacts_path) if contacts_path else None
    if max_bytes <= 0 or page_lines <= 0 or max_lines < 0:
        raise AtlasError("max_bytes and page_lines must be positive, max_lines not negative")
    options = {"timezone": getattr(tz, "key", str(tz)), "max_bytes": max_bytes, "max_lines": max_lines,
               "page_lines": page_lines, "owner_id": owner_id, "owner_name": owner_name}
    site_options = {"owner_id": owner_id, "owner_name": owner_name, "tz": tz, "page_lines": page_lines}
    if not ledger.is_file():
        raise AtlasError(f"no ledger at {ledger}")
    _claim_directory(out)
    with _exclusive(out):
        return _build(out, ledger, contacts, now=now, options=options, site_options=site_options,
                      max_lines=max_lines, max_bytes=max_bytes, started=started)


def _build(out: Path, ledger: Path, contacts: Optional[Path], *, now: datetime, options: Dict[str, Any],
           site_options: Dict[str, Any], max_lines: int, max_bytes: int, started: float) -> Dict[str, Any]:
    manifest = _claim_directory(out)          # again, under the exclusion: another build may have finished
    report: Dict[str, Any] = {}
    for _ in range(SNAPSHOT_ATTEMPTS):
        snapshot = read_snapshot(ledger, contacts, now=now, max_lines=max_lines)
        fingerprint = _fingerprint(snapshot, options)
        if _unchanged(out, manifest, fingerprint, snapshot.erasure_mark):
            size = sum(os.lstat(out / path).st_size for path in [*manifest["pages"], MANIFEST])
            report = {**manifest["report"], "bytes": size, "written": 0, "deleted": 0,
                      "unchanged": len(manifest["pages"])}
        else:
            base = {"version": RENDER_VERSION, "audience": "owner", "fingerprint": fingerprint,
                    "erasure_mark": snapshot.erasure_mark}

            def summary(kept: int, rendered: Rendered) -> Dict[str, Any]:
                return {"pages": len(rendered.pages), "lines": kept, "dropped_lines": len(snapshot.lines) - kept + snapshot.omitted,
                        **rendered.counts, "name_links": rendered.stats.name_links,
                        "cite_links": rendered.stats.cite_links}

            def manifest_size(kept: int, rendered: Rendered) -> int:
                # Exact: every page hash has the same length, so placeholders size it without hashing.
                placeholders = dict.fromkeys(rendered.pages, "0" * 64)
                return len(_manifest_json({**base, "report": summary(kept, rendered)}, placeholders, "complete"))

            kept, rendered, size = _fit(snapshot, site_options, max_lines=max_lines, max_bytes=max_bytes,
                                        overhead=manifest_size)
            report = summary(kept, rendered)
            written, unchanged, deleted = _publish(out, rendered.pages, manifest, {**base, "report": report})
            report = {**report, "bytes": size, "written": written, "unchanged": unchanged, "deleted": deleted}
            manifest = _read_manifest(out)
        # An erasure that landed while the pages were written is honoured before returning.
        with closing(open_read_only(ledger)) as conn:
            if erasure_mark(conn) == snapshot.erasure_mark:
                break
    else:
        # Erasures kept landing through every attempt: the pages may still quote an erased source.
        # Never present that as a finished build: the manifest says so, the check fails, the build raises.
        if manifest is not None:
            _private_file(out / MANIFEST, json.dumps({**manifest, "state": "stale"}, indent=1, sort_keys=True).encode())
        raise AtlasError(f"sources kept being erased while the atlas was built ({SNAPSHOT_ATTEMPTS} attempts); "
                         "its pages may still quote an erased source: run colonymind atlas build again")
    report.update(out=str(out), fingerprint=fingerprint, erasure_mark=snapshot.erasure_mark,
                  elapsed_ms=int((time.monotonic() - started) * 1000))
    return report


class _Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.hrefs: List[str] = []
        self.problems: List[str] = []
        self.policy = False
        self._style = False

    def handle_starttag(self, tag, attrs):
        if tag in _FORBIDDEN_TAGS:
            self.problems.append(f"<{tag}> loads or runs something")
        if tag == "meta":
            equiv = next(((value or "").lower() for key, value in attrs if key == "http-equiv"), "")
            content = next((value or "" for key, value in attrs if key == "content"), "")
            if equiv == "refresh":
                self.problems.append("a meta refresh")
            elif equiv == "content-security-policy" and "default-src 'none'" in content:
                self.policy = True
        self._style = tag == "style"
        for key, value in attrs:
            value = value or ""
            if key == "id":
                self.ids.add(value)
            elif key == "href":
                self.hrefs.append(value)
            elif key in _REFERENCE_ATTRS or key.startswith("on"):
                self.problems.append(f"a {key} attribute")
            elif key == "style" and ("url(" in value.lower() or "@import" in value.lower()):
                self.problems.append("a style that fetches")

    def handle_endtag(self, tag):
        if tag == "style":
            self._style = False

    def handle_data(self, data):
        if self._style and ("url(" in data.lower() or "@import" in data.lower()):
            self.problems.append("a stylesheet that fetches")


def check_atlas(out_dir: Path | str, *, ledger_path: Path | str | None = None) -> Dict[str, Any]:
    """Audit a built atlas; ``problems`` is empty when every check passes."""
    out = Path(out_dir).expanduser()
    try:
        manifest = _read_manifest(out) if out.is_dir() else None
    except AtlasError as error:
        return {"problems": [str(error)], "pages": 0, "links": 0}
    if manifest is None:
        return {"problems": [f"no atlas in {out}"], "pages": 0, "links": 0}
    problems: List[str] = []
    if manifest.get("state") == "stale":
        problems.append("sources were erased while the last build ran and it could not catch up; "
                        "run colonymind atlas build")
    elif manifest.get("state") != "complete":
        problems.append("the last build did not finish; run colonymind atlas build")
    if stat.S_IMODE(out.stat().st_mode) & 0o077:
        problems.append("the atlas directory is open to other users")
    pages = {path: digest for path, digest in manifest["pages"].items()}
    parsed: Dict[str, _Page] = {}
    for path, digest in sorted(pages.items()):
        target = out / path
        if not _PAGE.match(path) or not target.is_file():
            problems.append(f"{path}: missing")
            continue
        # Only what lies inside the atlas must be real: the configured root itself may be reached through a link.
        parts = Path(path).parts
        if any((out.joinpath(*parts[:depth])).is_symlink() for depth in range(1, len(parts) + 1)):
            problems.append(f"{path}: a link, not a page the atlas wrote")
            continue
        data = target.read_bytes()
        if _sha(data) != digest:
            problems.append(f"{path}: changed since the build")
        if stat.S_IMODE(target.stat().st_mode) & 0o077:
            problems.append(f"{path}: readable by other users")
        page = _Page()
        page.feed(data.decode("utf-8", errors="replace"))
        parsed[path] = page
        problems.extend(f"{path}: {problem}" for problem in page.problems)
        if not page.policy:
            problems.append(f"{path}: no content security policy that blocks loading")
    links = 0
    for path, page in sorted(parsed.items()):
        for href in page.hrefs:
            links += 1
            if re.match(r"^[a-z][a-z0-9+.-]*:", href, re.I) or href.startswith(("/", "\\")):
                problems.append(f"{path}: a link leaves the atlas ({href[:60]})")
                continue
            target, _, fragment = href.partition("#")
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(path), unquote(target))) if target else path
            if resolved not in parsed:
                problems.append(f"{path}: a link to a page that is not there ({href[:60]})")
            elif fragment and unquote(fragment) not in parsed[resolved].ids:
                problems.append(f"{path}: a link to an anchor that is not there ({href[:60]})")
    for extra in sorted(out.rglob("*.html")):
        relative = extra.relative_to(out).as_posix()
        if relative not in pages:
            problems.append(f"{relative}: a page the atlas does not track")
    if ledger_path is not None:
        with closing(open_read_only(Path(ledger_path))) as conn:
            mark = erasure_mark(conn)
        built = manifest.get("erasure_mark") or [0, 0]
        if mark[0] != built[0]:
            problems.append("sources were erased after this build; run colonymind atlas build")
        if mark[1] != built[1]:
            problems.append("sources were reattributed after this build; run colonymind atlas build")
    return {"problems": problems, "pages": len(pages), "links": links}
