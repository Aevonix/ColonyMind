"""Periodic consistent snapshots of the file-based state: the vector store and the other files.

A snapshot is a manifest ``{relative path: [sha256, size, mode]}`` plus content-addressed
blobs, so a file already on the standby is never sent again and a standby holds each content
once. The primary builds blobs as hard links (no copy, and a link keeps a file alive if the
application deletes it mid-snapshot); small ordinary files are copied first so an in-place
rewrite cannot change a blob after it was hashed.

``vectors``: each Lance table is taken at one version V, its latest committed manifest: that
manifest and the table's data, deletion and index files no newer than it (Lance writes a
version's files before committing its manifest and never rewrites a file). Older manifests,
transaction records and the version hint are left out, so the copy has exactly one version.
The standby opens every table, checks the version and scans a column of every row before the
snapshot becomes current; a snapshot that fails is reported back and replaced.

``files``: everything else in the instance directory (identity, configuration, source media,
upgrade backups), except live SQLite stores (shipped as pages), the vector store, the twin's
own directory and transient files.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

from colonymind.twin.config import TWIN_DIR
from colonymind.twin.segment import file_sha256, fsync_dir, publish
from colonymind.twin.source import is_sqlite

KINDS = ('vectors', 'files')
# Upgrade backups hold finished copies of stores: they travel as files, never as live stores.
STATIC_DIRS = ('backups',)
# kind, run (the shipper process that made it: a number above every earlier run and every run
# the standby reported, see ``primary.Names``), seq within that run, digest of the manifest's bytes.
MANIFEST = re.compile(r'^(vectors|files)-(\d{20})-(\d{12})-([0-9a-f]{24})\.json$')


def manifest_version(name: str) -> str | None:
    """``<run>-<seq>``: fixed width, so a later snapshot sorts after an earlier one, also across
    a restart of the shipper (whose snapshot numbers start again)."""
    match = MANIFEST.match(name)
    return f'{match.group(2)}-{match.group(3)}' if match else None


def new_run() -> str:
    return f'{time.time_ns():020d}'
_U64 = (1 << 64) - 1
_SMALL = 1 << 20
TRANSIENT = ('*.pid', '*.sock', '*.lock', '*.log', '*.tmp', '*.part', '*-wal', '*-shm', '*-journal',
             'logs/*', 'tmp/*', '*/__pycache__/*', '__pycache__/*', 'last_user_message_at.json')


def _manifest_version(name: str) -> int | None:
    match = re.fullmatch(r'(\d+)\.manifest', name)
    if not match:
        return None
    number = int(match.group(1))
    return _U64 - number if number > (1 << 63) else number


def lance_tables(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob('*.lance') if p.is_dir() and (p / '_versions').is_dir())


def _vector_files(home: Path) -> tuple[dict[str, Path], dict[str, int]]:
    root = home / 'lancedb'
    chosen: dict[str, Path] = {}
    tables: dict[str, int] = {}
    if not root.is_dir():
        return chosen, tables
    table_dirs = lance_tables(root)
    for table in table_dirs:
        versions = [(v, p) for p in (table / '_versions').iterdir() if (v := _manifest_version(p.name)) is not None]
        if not versions:
            continue
        version, manifest = max(versions)
        horizon = manifest.stat().st_mtime_ns
        chosen[manifest.relative_to(home).as_posix()] = manifest
        for sub in ('data', '_deletions', '_indices'):
            for path in (table / sub).rglob('*') if (table / sub).is_dir() else ():
                if path.is_file() and path.stat().st_mtime_ns <= horizon:
                    chosen[path.relative_to(home).as_posix()] = path
        tables[table.relative_to(home).as_posix()] = version
    inside = tuple(t.relative_to(home).as_posix() + '/' for t in table_dirs)
    for path in root.rglob('*'):
        relative = path.relative_to(home).as_posix()
        if path.is_file() and not relative.startswith(inside) and not path.name.startswith('.'):
            chosen[relative] = path
    return chosen, tables


def _other_files(home: Path, exclude: set[str]) -> dict[str, Path]:
    chosen = {}
    for directory, dirs, files in os.walk(home):
        base = Path(directory)
        relative_dir = base.relative_to(home).as_posix()
        if relative_dir == '.':
            dirs[:] = [d for d in dirs if d not in (TWIN_DIR, 'lancedb')]
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for name in files:
            path = base / name
            relative = path.relative_to(home).as_posix()
            if relative in exclude or any(fnmatch.fnmatch(relative, pattern) for pattern in TRANSIENT):
                continue
            if not path.is_file() or path.is_symlink():
                continue
            if is_sqlite(path) and not relative.startswith(tuple(d + '/' for d in STATIC_DIRS)):
                continue
            chosen[relative] = path
    return chosen


class TreeSnapshotter:
    """Builds snapshots of one kind into ``<outbox>/trees``."""

    def __init__(self, home: Path, outbox: Path, kind: str, *, exclude: set[str] | None = None, clock=time.time,
                 run: str | None = None):
        assert kind in KINDS
        self.home, self.kind, self.clock = Path(home), kind, clock
        self.run = run or new_run()
        self.trees = Path(outbox) / 'trees'
        self.blobs = self.trees / 'blobs'
        self.exclude = exclude or set()
        self._hashes: dict[tuple, str] = {}
        self.last_fingerprint: str | None = None

    def _select(self) -> tuple[dict[str, Path], dict[str, int]]:
        if self.kind == 'vectors':
            return _vector_files(self.home)
        return _other_files(self.home, self.exclude), {}

    def fingerprint(self) -> str:
        files, tables = self._select()
        digest = hashlib.sha256(json.dumps(tables, sort_keys=True).encode())
        for relative in sorted(files):
            try:
                stat = files[relative].stat()
            except FileNotFoundError:
                continue
            digest.update(f'{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\0{stat.st_ino}\n'.encode())
        return digest.hexdigest()

    def _blob(self, path: Path) -> tuple[str, int, int]:
        stat = path.stat()
        mode = stat.st_mode & 0o777
        self.blobs.mkdir(parents=True, exist_ok=True)
        if self.kind == 'files' and stat.st_size <= _SMALL:
            data = path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            blob = self.blobs / sha
            if not blob.exists():
                temporary = self.blobs / f'.{sha}.{os.getpid()}.part'
                temporary.write_bytes(data)
                os.replace(temporary, blob)
            return sha, len(data), mode
        key = (path.as_posix(), stat.st_ino, stat.st_size, stat.st_mtime_ns)
        sha = self._hashes.get(key) or file_sha256(path)
        self._hashes[key] = sha
        blob = self.blobs / sha
        if not blob.exists():
            try:
                os.link(path, blob)
            except OSError:
                shutil.copyfile(path, blob)
        return sha, stat.st_size, mode

    def snapshot(self, seq: int) -> Path:
        """Write one manifest (and its blobs); raises FileNotFoundError if a file vanished."""
        fingerprint = self.fingerprint()
        files, tables = self._select()
        entries = {relative: list(self._blob(path)) for relative, path in sorted(files.items())}
        manifest = {'kind': self.kind, 'run': self.run, 'seq': seq, 'created_at': self.clock(), 'files': entries,
                    'tables': tables}
        self.trees.mkdir(parents=True, exist_ok=True)
        temporary = self.trees / f'.{self.kind}-{seq}.{os.getpid()}.part'
        temporary.write_text(json.dumps(manifest, sort_keys=True))
        final = publish(temporary, self.trees, lambda sha: f'{self.kind}-{self.run}-{seq:012d}-{sha[:24]}.json')
        self.last_fingerprint = fingerprint
        return final


def referenced_blobs(manifests: list[Path]) -> set[str]:
    wanted = set()
    for path in manifests:
        try:
            wanted.update(entry[0] for entry in json.loads(path.read_text())['files'].values())
        except (OSError, ValueError, KeyError):
            continue
    return wanted


# -- standby side ------------------------------------------------------------------------

class TreeInstallError(RuntimeError):
    def __init__(self, message: str, *, blob: Path | None = None):
        super().__init__(message)
        self.blob = blob          # a blob whose bytes do not match its name


def validate_vectors(root: Path, tables: dict[str, int]) -> dict[str, int]:
    """Open every table at the snapshot and scan one column of every row; returns row counts."""
    import lancedb
    counts = {}
    for relative, version in tables.items():
        table_dir = root / relative
        db = lancedb.connect(str(table_dir.parent))
        table = db.open_table(table_dir.name[:-len('.lance')])
        if table.version != version:
            raise TreeInstallError(f'{relative}: version {table.version}, expected {version}')
        expected = table.count_rows()
        scanned = table.search().select([table.schema.names[0]]).limit(None).to_arrow().num_rows
        if scanned != expected:
            raise TreeInstallError(f'{relative}: scanned {scanned} rows of {expected}')
        counts[relative] = expected
    return counts


def install(manifest_path: Path, blobs: Path, destination: Path, verified: dict) -> dict:
    """Materialise a manifest under ``destination`` (hard links to verified blobs)."""
    manifest = json.loads(manifest_path.read_text())
    building = destination.with_name(destination.name + '.building')
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    for relative, (sha, size, mode) in manifest['files'].items():
        if relative.startswith('/') or '..' in Path(relative).parts:
            raise TreeInstallError('unsafe path in manifest')
        blob = blobs / sha
        try:
            stat = blob.stat()
        except FileNotFoundError:
            raise TreeInstallError('blob not yet transferred') from None
        key = (sha, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if key not in verified:
            if stat.st_size != size or file_sha256(blob) != sha:
                raise TreeInstallError(f'blob for {relative} does not match its digest', blob=blob)
            verified[key] = True
        target = building / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(blob, target)
        except OSError:
            shutil.copyfile(blob, target)
        os.chmod(target, mode & 0o777)
    detail = {'files': len(manifest['files'])}
    if manifest['kind'] == 'vectors':
        detail['rows'] = validate_vectors(building, manifest['tables'])
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(building, destination)
    fsync_dir(destination.parent)
    return detail
