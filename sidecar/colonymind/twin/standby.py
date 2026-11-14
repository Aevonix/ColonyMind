"""The standby side: apply segments to read-only replicas, install trees, ack, promote.

A replica is replaced, never edited in place: a batch of segments is applied to a clone of the
published file, which is then renamed over it. A reader holding the old file keeps a
consistent snapshot; a new reader sees the new one; a crash mid-batch leaves the published
file untouched and the batch is applied again (page images are idempotent). Published
replicas are mode 0400 and meant to be opened ``immutable``: the standby cannot fork the
record because nothing on it can write a replica except this applier.
"""

from __future__ import annotations

import copy
import os
import shutil
import sqlite3
import struct
import sys
import time
import uuid
from pathlib import Path

from colonymind.twin import segment
from colonymind.twin.config import PROMOTED, Journal, TwinConfig, read_json, save, twin_dir, write_json
from colonymind.twin.primary import EXTERNAL
from colonymind.twin.source import file_digest
from colonymind.twin.trees import KINDS, MANIFEST, TreeInstallError, install, manifest_version

MAX_BATCH_SEGMENTS = 64
MAX_BATCH_BYTES = 256 << 20
_FICLONE = 0x40049409


class PromoteError(RuntimeError):
    pass


def safe_key(key: str) -> str:
    parts = Path(key).parts
    if not key or key.startswith('/') or '..' in parts or '\\' in key:
        raise ValueError(f'unsafe store name: {key!r}')
    return key


def clone(source: Path, destination: Path) -> None:
    """Copy-on-write when the filesystem can (FICLONE, clonefile), else an ordinary copy."""
    destination.unlink(missing_ok=True)
    try:
        if sys.platform.startswith('linux'):
            import fcntl
            with open(source, 'rb') as src, open(destination, 'wb') as dst:
                fcntl.ioctl(dst.fileno(), _FICLONE, src.fileno())
            return
        if sys.platform == 'darwin':
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.clonefile(os.fsencode(source), os.fsencode(destination), 0) == 0:
                return
    except (OSError, AttributeError):
        pass
    destination.unlink(missing_ok=True)
    shutil.copyfile(source, destination)


def connect(home: Path, key: str) -> sqlite3.Connection:
    """A read-only connection to one replica (``@external/<name>`` for outside stores)."""
    path = twin_dir(Path(home)) / 'replica' / 'db' / safe_key(key)
    if not path.is_file():
        raise FileNotFoundError(f'no replica of {key}')
    return sqlite3.connect(path.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)


def tree_path(home: Path, kind: str) -> Path | None:
    current = twin_dir(Path(home)) / 'replica' / 'trees' / kind / 'current'
    return current.resolve() if current.exists() else None


class Standby:
    def __init__(self, home: Path, config: TwinConfig, *, clock=time.time):
        self.home, self.config, self.clock = Path(home), config, clock
        self.dir = twin_dir(self.home)
        self.inbox = self.dir / 'inbox'
        self.replicas = self.dir / 'replica' / 'db'
        self.tree_root = self.dir / 'replica' / 'trees'
        self.journal = Journal(self.home, config, clock=clock)
        self.state = read_json(self.dir / 'state.json') or {}
        for name, default in (('generation', ''), ('seq', 0), ('created_at', None), ('stores', {}), ('listed', []),
                              ('trees', {}), ('tree_failed', {}), ('reseed', None), ('promoted', None),
                              ('primary_stale', {})):
            self.state.setdefault(name, default)
        self.verified_blobs: dict = {}
        self.lag: float | None = None

    def replica_path(self, key: str) -> Path:
        return self.replicas / safe_key(key)

    # -- the cycle -----------------------------------------------------------------------
    def cycle(self) -> dict:
        now = self.clock()
        if self.state['promoted']:
            if not self.state['promoted'].get('complete'):
                self._publish_fence(now)      # an interrupted promotion keeps its fence up
            return self.status(now)
        self._maybe_switch_generation()
        applied = 0
        while True:
            batch = self._ready_batch()
            if not batch:
                break
            try:
                self._apply(batch)
            except FileNotFoundError:
                break                        # a superseded generation was removed mid-batch
            except ValueError as error:      # a seed or segment that does not verify
                self.request_reseed(f'apply failed: {error}')
                break
            applied += len(batch)
        for kind in KINDS:
            self._install_tree(kind)
        rejected = [kind for kind in KINDS if (failed := self.state['tree_failed'].get(kind))
                    and str(failed) > str((self.state['trees'].get(kind) or {}).get('version') or '')]
        self.journal.alarm('tree', bool(rejected), ', '.join(
            f"{kind}: snapshot {self.state['tree_failed'][kind]} rejected" for kind in rejected))
        self._check_spool()
        created = self.state['created_at']
        self.lag = None if created is None else max(0.0, now - created)
        self.journal.alarm('lag', self.lag is None or self.lag > self.config.max_lag_seconds,
                           'never synchronised' if self.lag is None else f'{self.lag:.1f}s behind')
        self._write_ack(now)
        return self.status(now)

    def _generations(self) -> list[Path]:
        return sorted(p for p in self.inbox.glob('gen-*') if p.is_dir()) if self.inbox.is_dir() else []

    def _maybe_switch_generation(self) -> None:
        generations = self._generations()
        if not generations or generations[-1].name <= self.state['generation']:
            return
        newest = generations[-1]
        if self._segment(newest, 1) is None:
            return
        self.journal.event('generation-switch', old=self.state['generation'], new=newest.name)
        self.state['generation'], self.state['seq'] = newest.name, 0
        request = self.state['reseed']
        if request and request.get('generation', '') < newest.name:
            self.state['reseed'] = None
        self._save()

    def _segment(self, directory: Path, seq: int) -> Path | None:
        for path in directory.glob(f'{seq:012d}-*.seg'):
            if not segment.NAME.match(path.name):
                continue
            if segment.verified(path):
                return path
            self._discard(path, 'segment does not match its digest')
        return None

    def _discard(self, path: Path, why: str) -> None:
        """Remove a damaged inbox file so the mirror sends it again.

        The inbox is the primary's to write, but a file there is complete once it has its
        final name (both transports rename into place) and its name carries its digest: a file
        that does not match is damaged, and a mirror that compares sizes (or sizes and times)
        would never replace it. This is the one thing the standby removes from the inbox.
        """
        path.unlink(missing_ok=True)
        self.journal.event('inbox-file-discarded', file=path.name, why=why)

    def _ready_batch(self) -> list[tuple[Path, dict, int]]:
        if not self.state['generation']:
            return []
        directory = self.inbox / self.state['generation']
        batch, size, seq = [], 0, self.state['seq']
        while len(batch) < MAX_BATCH_SEGMENTS and size < MAX_BATCH_BYTES:
            path = self._segment(directory, seq + 1)
            if path is None:
                break
            try:
                header, base = segment.read_header(path)
                seeds = [e['seed'] for e in header['entries'] if e.get('seed')]
                if any((directory / s['name']).stat().st_size != s['size'] for s in seeds):
                    break
            except FileNotFoundError:       # not yet transferred, or a superseded generation
                break
            batch.append((path, header, base))
            size += path.stat().st_size + sum(s['size'] for s in seeds)
            seq += 1
        return batch

    def _apply(self, batch: list[tuple[Path, dict, int]]) -> None:
        """Apply a batch to work files and publish them; the state moves only once they are.

        The batch's effect on the state is built on a copy and saved after every replica is
        renamed into place: a failure anywhere leaves the saved and the in-memory state (and so
        the ack) at the previous batch, and the batch is applied again (page images are
        idempotent). A divergence found on the way is recorded at once.
        """
        directory = self.inbox / self.state['generation']
        self.replicas.mkdir(parents=True, exist_ok=True)
        works: dict[str, tuple[Path, int]] = {}
        broken: set[str] = set()
        staged = {name: copy.deepcopy(self.state[name])
                  for name in ('seq', 'created_at', 'stores', 'listed', 'primary_stale')}
        infos = staged['stores']
        for path, header, base in batch:
            for entry in header['entries']:
                key = safe_key(entry['key'])
                if key in broken and not entry.get('seed'):
                    continue
                broken.discard(key)
                target = self.replica_path(key)
                work = works.get(key, (None, 0))[0]
                if entry.get('seed'):
                    seed = directory / entry['seed']['name']
                    if segment.file_sha256(seed) != entry['seed']['sha256']:
                        raise ValueError(f'seed for {key} does not match its digest')
                    work = target.with_name(f'.{target.name}.apply')
                    work.parent.mkdir(parents=True, exist_ok=True)
                    work.unlink(missing_ok=True)
                    shutil.copyfile(seed, work)
                elif work is None:
                    if not target.is_file():
                        infos.setdefault(key, {})['diverged'] = True
                        self._diverged(key, 'a change arrived for a store never seeded')
                        broken.add(key)
                        continue
                    work = target.with_name(f'.{target.name}.apply')
                    clone(target, work)
                page_size = int(entry['page_size'])
                fd = os.open(work, os.O_RDWR)
                try:
                    for pgno, page in segment.iter_pages(path, base, entry):
                        os.pwrite(fd, page, (pgno - 1) * page_size)
                    os.ftruncate(fd, int(entry['page_count']) * page_size)
                    if entry.get('digest'):
                        os.fsync(fd)
                finally:
                    os.close(fd)
                works[key] = (work, page_size)
                info = infos.setdefault(key, {})
                info.update(generation=header['generation'], seq=header['seq'], page_size=page_size,
                            page_count=int(entry['page_count']), reason=entry['reason'], diverged=False)
                if entry.get('digest'):
                    if file_digest(work, page_size) != entry['digest']:
                        work.unlink(missing_ok=True)
                        works.pop(key, None)
                        broken.add(key)
                        infos[key]['diverged'] = True
                        self._diverged(key, f'digest mismatch after {entry["reason"]}')
                        continue
                    info['verified_at'] = self.clock()
                if entry.get('repaired'):
                    self.journal.event('anti-entropy-repair', store=key, pages=entry['repaired'])
            # The primary lists every store it has in every segment: a replica it does not list
            # is of a store deleted there (also while the shipper was stopped) and is dropped.
            listed = header.get('stores')
            if listed is not None:
                for key in sorted(set(infos) - set(listed)):
                    self.replica_path(key).unlink(missing_ok=True)
                    infos.pop(key, None)
                    work = works.pop(key, (None, 0))[0]
                    if work is not None:
                        work.unlink(missing_ok=True)
                    self.journal.event('store-dropped', store=key)
                staged['listed'] = sorted(listed)
            staged['seq'], staged['created_at'] = header['seq'], header['created_at']
            # Stores the primary could not read, and since when (its clock): their replicas are
            # only as recent as that.
            staged['primary_stale'] = dict(header.get('stale') or {})
        try:
            for key, (work, _) in works.items():
                with open(work, 'rb+') as handle:
                    os.fsync(handle.fileno())
                os.chmod(work, 0o400)
                os.replace(work, self.replica_path(key))
                segment.fsync_dir(self.replica_path(key).parent)
        except BaseException:
            for work, _ in works.values():
                work.unlink(missing_ok=True)
            raise
        state = {**self.state, **staged}
        write_json(self.dir / 'state.json', state)
        self.state = state
        if not broken and not any(i.get('diverged') for i in infos.values()):
            self.journal.alarm('diverged', False)

    def _diverged(self, key: str, why: str) -> None:
        self.state['stores'].setdefault(key, {})['diverged'] = True
        self.journal.alarm('diverged', True, f'{key}: {why}')
        self.request_reseed(f'{key}: {why}')

    def request_reseed(self, reason: str) -> None:
        if self.state['reseed']:
            return
        self.state['reseed'] = {'nonce': uuid.uuid4().hex, 'reason': reason[:200],
                                'generation': self.state['generation'], 'at': self.clock()}
        self.journal.event('reseed-requested', reason=reason[:200])
        self._save()
        self._write_ack(self.clock())

    def _check_spool(self) -> None:
        """The inbox cap counts what waits to be applied beyond the generation's base seeds.

        As on the primary, the first seed of each store in a generation is its unavoidable base:
        counting it would make a store larger than the cap request a re-seed on every seed.
        Applied files are not counted either; the primary deletes them once it reads the ack.
        """
        generation, applied = self.state['generation'], int(self.state['seq'])
        seeded = {key for key, info in self.state['stores'].items() if info.get('generation') == generation}
        total = 0
        directory = self.inbox / generation if generation else None
        if directory is not None and directory.is_dir():
            waiting = sorted((int(m.group(1)), p) for p in directory.glob('*.seg')
                             if (m := segment.NAME.match(p.name)) and int(m.group(1)) > applied)
            for _, path in waiting:
                try:          # the primary's mirror may delete files meanwhile
                    header, _ = segment.read_header(path)
                    total += path.stat().st_size
                except (OSError, ValueError, struct.error):   # damaged: discarded when its turn comes
                    continue
                for entry in header.get('entries') or []:
                    seed = entry.get('seed') or {}
                    if not seed:
                        continue
                    if entry.get('key') in seeded:
                        total += int(seed.get('size') or 0)
                    else:
                        seeded.add(entry.get('key'))
        over = total > self.config.max_spool_bytes
        self.journal.alarm('spool_cap', over, f'{total} bytes waiting in the inbox')
        if over:
            self.request_reseed('inbox over its size cap')

    def _install_tree(self, kind: str) -> None:
        trees = self.inbox / 'trees'
        if not trees.is_dir():
            return
        installed = str((self.state['trees'].get(kind) or {}).get('version') or '')
        failed = self.state['tree_failed'].get(kind)
        candidates = sorted((version, p) for p in trees.glob(f'{kind}-*.json')
                            if (version := manifest_version(p.name)))
        candidates = [(v, p) for v, p in candidates if v > installed and v != failed]
        if not candidates:
            return
        version, manifest = candidates[-1]
        seq = int(version.split('-')[1])
        if segment.file_sha256(manifest)[:24] != MANIFEST.match(manifest.name).group(4):
            return
        destination = self.tree_root / kind / version       # unique: never the one `current` names
        try:
            detail = install(manifest, trees / 'blobs', destination, self.verified_blobs)
        except TreeInstallError as error:
            if 'not yet transferred' in str(error):
                return
            if error.blob is not None:
                self._discard(error.blob, f'{kind} blob does not match its digest')
            self.state['tree_failed'][kind] = version
            self.journal.event('tree-rejected', kind=kind, seq=seq, detail=str(error)[:200])
            self._save()
            return
        except Exception as error:      # a Lance table that will not open is a failed snapshot
            self.state['tree_failed'][kind] = version
            self.journal.event('tree-rejected', kind=kind, seq=seq, detail=f'{type(error).__name__}: {error}'[:200])
            self._save()
            return
        link = self.tree_root / kind / 'current'
        temporary = self.tree_root / kind / f'.current.{os.getpid()}'
        temporary.unlink(missing_ok=True)
        os.symlink(destination.name, temporary)
        os.replace(temporary, link)
        for old in (self.tree_root / kind).iterdir():
            if old.is_dir() and not old.is_symlink() and old.name != destination.name:
                shutil.rmtree(old, ignore_errors=True)
        self.state['trees'][kind] = {'seq': seq, 'version': version, 'installed_at': self.clock(), **detail}
        self._save()
        self.journal.event('tree-installed', kind=kind, seq=seq, files=detail['files'])

    def _save(self) -> None:
        write_json(self.dir / 'state.json', self.state)

    def _publish_fence(self, now: float) -> None:
        """The promotion's fence: the ack (the old primary stops shipping when it reads it) and
        the marker (its transport refuses to mirror here). Each is written unless it already
        names this promotion."""
        promoted = self.state['promoted']
        if ((read_json(self.dir / 'ack.json') or {}).get('promoted') or {}).get('epoch') != promoted['epoch']:
            self._write_ack(now)
        if (read_json(self.dir / PROMOTED) or {}).get('epoch') != promoted['epoch']:
            write_json(self.dir / PROMOTED, promoted)

    def _write_ack(self, now: float) -> None:
        write_json(self.dir / 'ack.json', {
            'generation': self.state['generation'], 'seq': self.state['seq'], 'at': now,
            'trees': {k: v.get('version') for k, v in self.state['trees'].items()},
            'tree_failed': self.state['tree_failed'], 'reseed': self.state['reseed'],
            'promoted': self.state['promoted']})

    def status(self, now: float | None = None) -> dict:
        now = self.clock() if now is None else now
        generation = self.state['generation']
        stale = [k for k in self.state['listed'] if (self.state['stores'].get(k) or {}).get('generation') != generation]
        status = {
            'role': 'standby', 'at': now, 'generation': generation, 'seq': self.state['seq'],
            'lag_seconds': None if self.lag is None else round(self.lag, 3), 'alarms': dict(self.journal.alarms),
            'stores': self.state['stores'], 'stale': stale, 'trees': self.state['trees'],
            'reseed': self.state['reseed'], 'promoted': self.state['promoted'],
            'primary_unreadable': self.state['primary_stale'],
        }
        write_json(self.dir / 'status.json', status, durable=False)
        return status

    # -- promotion -----------------------------------------------------------------------
    def promote(self, *, force: bool = False) -> dict:
        """Apply what has arrived, check it, fence the old primary, and move it into place.

        The fence is recorded before anything moves, so a promotion that fails part way (a
        file that will not move, a crash) is not a standby any more: calling this again
        resumes the placement, which is idempotent (what moved is no longer in the replica),
        without applying anything that arrived after the fence.
        """
        promoted = self.state['promoted']
        resumed = bool(promoted)
        if promoted and promoted.get('complete'):
            raise PromoteError('this standby was already promoted')
        if promoted:
            problems = list(promoted.get('problems') or [])
            self.journal.event('promotion-resumed', epoch=promoted.get('epoch'))
        else:
            self.cycle()
            problems = self._promotion_problems()
            if problems and not force:
                raise PromoteError('; '.join(problems))
            now = self.clock()
            promoted = {'at': now, 'epoch': uuid.uuid4().hex, 'generation': self.state['generation'],
                        'seq': self.state['seq'], 'primary_time': self.state['created_at'],
                        'forced': bool(problems), 'problems': problems, 'complete': False}
            self.state['promoted'] = promoted
            self._save()
        # The fence, on every call and before anything moves: a retry after a failure anywhere
        # past the saved state publishes it (a resumed promotion never skips it).
        self._publish_fence(self.clock())
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(self.clock()))
        displaced = self.dir / 'displaced' / stamp
        moved, left, merged = [], [], []
        for key in sorted(set(self.state['stores']) & set(self.state['listed'])):
            source = self.replica_path(key)
            if not source.is_file():
                continue
            if key.startswith(EXTERNAL):
                if not self.config.external_dir:
                    left.append(key)
                    continue
                destination = Path(self.config.external_dir).expanduser() / key[len(EXTERNAL):]
                if destination.is_file():
                    merged.append({key: _merge_rows(source, destination)})
                    continue
            else:
                destination = self.home / key
            # Writable before it moves: a promotion interrupted right after a move leaves a store
            # the application can write, and a retry (which skips what moved) has nothing to fix.
            os.chmod(source, 0o600)
            _place(source, destination, displaced, self.home)
            moved.append(key)
        trees = {}
        files_root = tree_path(self.home, 'files')
        if files_root is not None:
            kept = []
            for path in sorted(p for p in files_root.rglob('*') if p.is_file()):
                relative = path.relative_to(files_root).as_posix()
                destination = self.home / relative
                if relative in self.config.keep_local and destination.exists():
                    _copy_private(path, destination.with_name(destination.name + '.from-primary'))
                    kept.append(relative)
                    continue
                _place(path, destination, displaced, self.home)
            trees['files'] = {'seq': self.state['trees'].get('files', {}).get('seq'), 'kept_local': kept}
        vectors_root = tree_path(self.home, 'vectors')
        if vectors_root is not None and (vectors_root / 'lancedb').is_dir():
            _place(vectors_root / 'lancedb', self.home / 'lancedb', displaced, self.home)
            trees['vectors'] = {'seq': self.state['trees'].get('vectors', {}).get('seq')}
        promoted = {**promoted, 'complete': True}
        report = {'promoted': promoted, 'stores': moved, 'left_in_replica': left, 'merged': merged,
                  'trees': trees, 'problems': problems, 'resumed': resumed,
                  'displaced': str(displaced) if displaced.exists() else None}
        write_json(self.dir / f'promotion-{stamp}.json', report)
        self.config.role, self.config.target, self.config.promoted = 'primary', '', promoted
        save(self.home, self.config)
        shutil.rmtree(self.inbox, ignore_errors=True)
        self.state['promoted'] = promoted
        self._save()
        self.journal.event('promoted', **{k: v for k, v in report.items() if k != 'promoted'}, epoch=promoted['epoch'])
        return report

    def _promotion_problems(self) -> list[str]:
        generation = self.state['generation']
        problems = []
        for key in self.state['listed']:
            info = self.state['stores'].get(key) or {}
            if info.get('generation') != generation:
                problems.append(f'{key}: not yet seeded in the current generation')
            elif info.get('diverged'):
                problems.append(f'{key}: diverged')
        for key in self.state['stores']:
            path = self.replica_path(key)
            if path.is_file():
                with sqlite3.connect(path.resolve().as_uri() + '?mode=ro&immutable=1', uri=True) as conn:
                    result = conn.execute('PRAGMA quick_check').fetchone()[0]
                conn.close()
                if result != 'ok':
                    problems.append(f'{key}: quick_check {result}')
        for key, since in sorted(self.state['primary_stale'].items()):
            behind = float(self.state['created_at'] or 0) - float(since)
            if behind > self.config.max_lag_seconds:
                problems.append(f'{key}: the primary could not read it for {behind:.0f}s before its last segment')
        # The instance is its stores and its files: identity, key, configuration and vectors
        # travel as tree snapshots, and a promotion without them is not the same instance.
        for kind in KINDS:
            installed = str((self.state['trees'].get(kind) or {}).get('version') or '')
            failed = self.state['tree_failed'].get(kind)
            if not installed:
                problems.append(f'{kind}: no snapshot installed' + (' (the newest was rejected)' if failed else ''))
            elif failed and str(failed) > installed:
                problems.append(f'{kind}: the newest snapshot was rejected; an older one is installed')
        if not generation:
            problems.append('nothing has been replicated yet')
        return problems


def _place(source: Path, destination: Path, displaced: Path, home: Path) -> None:
    """Move ``source`` to ``destination``; whatever was there goes to ``displaced``, never deleted."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('', '-wal', '-shm', '-journal'):
        existing = destination.with_name(destination.name + suffix)
        if existing.exists() or existing.is_symlink():
            try:
                relative = existing.relative_to(home)
            except ValueError:
                relative = Path('external') / existing.name
            aside = displaced / relative
            aside.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(existing), str(aside))
    shutil.move(str(source), str(destination))


def _copy_private(source: Path, destination: Path) -> None:
    """Copy with the source's permission bits, never readable by others on the way, atomically."""
    mode = os.stat(source).st_mode & 0o777
    temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.part')
    temporary.unlink(missing_ok=True)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with open(source, 'rb') as src, os.fdopen(fd, 'wb', closefd=False) as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fchmod(fd, mode)
            os.fsync(fd)
    except BaseException:
        os.close(fd)
        temporary.unlink(missing_ok=True)
        raise
    os.close(fd)
    os.replace(temporary, destination)
    segment.fsync_dir(destination.parent)


def _merge_rows(replica: Path, destination: Path) -> dict:
    """Add the replica's rows the destination lacks (tables of the same shape, INSERT OR IGNORE)."""
    added = {}
    conn = sqlite3.connect(destination.resolve().as_uri(), uri=True, timeout=30)
    try:
        conn.execute('ATTACH DATABASE ? AS twin', (replica.resolve().as_uri() + '?mode=ro&immutable=1',))
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM twin.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            ours = [r[1] for r in conn.execute(f'PRAGMA main.table_info({quoted})')]
            theirs = [r[1] for r in conn.execute(f'PRAGMA twin.table_info({quoted})')]
            if not ours or ours != theirs:
                continue
            before = conn.total_changes
            conn.execute(f'INSERT OR IGNORE INTO main.{quoted} SELECT * FROM twin.{quoted}')
            added[table] = conn.total_changes - before
        conn.commit()
        conn.execute('DETACH DATABASE twin')
    finally:
        conn.close()
    return added
