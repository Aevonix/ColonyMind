"""The primary side: poll every store, write segments, snapshot trees, mirror, watch the lag.

It runs beside the sidecar as its own process and only ever reads the instance: a standby that
is slow, full or unreachable costs the primary spooled files, never a failed or blocked write.
The spool has a size cap; past it (or when the disk runs low) the spool is dropped and a new
generation re-seeds every store instead of filling the disk.
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from colonymind.instance import read_instance_id
from colonymind.twin import segment
from colonymind.twin.config import Journal, TwinConfig, read_json, twin_dir, write_json
from colonymind.twin.source import SourceError, SpaceError, SqliteSource, is_sqlite
from colonymind.twin.transport import TransportError, for_target
from colonymind.twin.trees import KINDS, MANIFEST, STATIC_DIRS, TreeSnapshotter, manifest_version, referenced_blobs

EXTERNAL = '@external/'
_SIDE_FILES = ('-wal', '-shm', '-journal')


def default_external(home: Path) -> list[str]:
    """The plugin's turn outbox, where the plugin puts it by default."""
    try:
        from colonymind.config import load_config
        hermes = Path(load_config(home).hermes_home)
    except Exception:
        hermes = Path(os.environ.get('HERMES_HOME') or Path.home() / '.hermes')
    return [str(hermes / 'state' / 'colonymind-turn-outbox.sqlite3')]


def discover(home: Path, external: list[str]) -> dict[str, Path]:
    """Every SQLite store in the instance directory, plus the configured outside stores."""
    stores: dict[str, Path] = {}
    for directory, dirs, files in os.walk(home):
        base = Path(directory)
        if base == home:
            dirs[:] = [d for d in dirs if d not in ('.twin', 'lancedb', *STATIC_DIRS)]
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for name in files:
            if name.endswith(_SIDE_FILES) or name.startswith('.'):
                continue
            path = base / name
            if path.is_file() and not path.is_symlink() and is_sqlite(path):
                stores[path.relative_to(home).as_posix()] = path
    for raw in external:
        path = Path(raw).expanduser()
        if path.is_file() and is_sqlite(path):
            stores[EXTERNAL + path.name] = path
    return stores


_GENERATION = re.compile(r'^gen-(\d{20})-')
_TREE_VERSION = re.compile(r'^(\d{20})-\d{12}$')


class Names:
    """Generation and snapshot-run numbers: nanoseconds of the real clock, but always above the
    last one used (kept in ``ids.json``) and above any the standby reports.

    The standby orders generations and snapshots by these numbers, so a name that sorts below
    what it already has is never applied. A clock that stepped back across a restart would give
    one; the high-water mark prevents it, and if that record is lost too, the standby's ack
    names what it has and the shipper starts again above it (``Shipper._handle_ack``).
    """

    def __init__(self, path: Path):
        self.path = path
        try:
            self.high = int((read_json(path) or {}).get('high') or 0)
        except (TypeError, ValueError):
            self.high = 0

    def next(self, above: int = 0) -> int:
        value = max(time.time_ns(), self.high + 1, above + 1)
        self.high = value
        try:
            write_json(self.path, {'high': value})
        except OSError:
            pass        # still increasing in this process; a lost record is caught by the ack
        return value


def withheld_files(home: Path) -> set[str]:
    """Files the standby never receives. The sealed store's key follows the backup rule: it travels only when
    ``secrets.vault.key_in_backups`` is on, so by default the standby holds the store's ciphertext and not its
    key. A configuration that cannot be read withholds the key."""
    import yaml

    from colonymind.secrets.config import parse_settings
    from colonymind.secrets.vault import KEY_NAME
    try:
        data = yaml.safe_load((home / 'colonymind.yaml').read_text()) if (home / 'colonymind.yaml').is_file() else {}
        settings = parse_settings((data or {}).get('secrets') if isinstance(data or {}, dict) else None)
    except (OSError, ValueError, yaml.YAMLError):
        return {KEY_NAME}
    return set() if settings.key_in_backups else {KEY_NAME}


def _generation_id(number: int) -> str:
    return f'gen-{number:020d}-{uuid.uuid4().hex[:8]}'


def _generation_number(name: str) -> int:
    match = _GENERATION.match(name or '')
    return int(match.group(1)) if match else 0


class Transfers:
    """One transfer at a time on its own thread: read the standby's ack, then mirror the spool.

    A transfer can block for as long as the transport's timeout (a partition, a black-holed
    ssh). The tick never waits for one longer than ``transfer_wait``, so polling goes on: the
    pins are still released at ``release_bytes`` and the free-space floor is still checked. The
    result is taken by the tick that finds the transfer finished, so everything that follows
    from an ack (pruning, a new generation, the fence) happens on the tick's thread.
    """

    def __init__(self, transport, outbox: Path):
        self.transport, self.outbox = transport, outbox
        self.thread: threading.Thread | None = None
        self.result: dict | None = None

    def busy(self) -> bool:
        """Running, or finished with a result nobody has taken yet: either way nothing starts."""
        return self.thread is not None

    def start(self) -> bool:
        """Start a transfer unless one is running or its result is still to be taken, so a
        result (an ack, a re-seed request, the fence) is never overwritten before it is handled."""
        if self.busy():
            return False
        self.result = None
        self.thread = threading.Thread(target=self._run, name='twin-transfer', daemon=True)
        self.thread.start()
        return True

    def _run(self) -> None:
        result: dict = {}
        try:
            result['ack'] = self.transport.read_ack()
        except Exception as error:          # reported as a transport failure, never lost
            result['error'] = error
        if not (result.get('ack') or {}).get('promoted'):
            try:
                result['mirrored'] = self.transport.mirror(self.outbox)
            except Exception as error:
                result['error'] = error
        self.result = result

    def wait(self, timeout: float) -> None:
        if self.thread is not None:
            self.thread.join(timeout)

    def take(self) -> dict | None:
        if self.thread is None or self.thread.is_alive():
            return None
        self.thread = None
        return self.result or {}


class Shipper:
    def __init__(self, home: Path, config: TwinConfig, *, transport=None, clock=time.time):
        self.home, self.config, self.clock = Path(home), config, clock
        self.dir = twin_dir(self.home)
        self.outbox, self.scratch = self.dir / 'outbox', self.dir / 'tmp'
        self.transport = transport or for_target(config.target, ssh_command=config.ssh_command,
                                                 control_dir=self.dir / 'ssh')
        self.journal = Journal(self.home, config, clock=clock)
        self.transfers = Transfers(self.transport, self.outbox)
        self.instance = read_instance_id(self.home)
        self.external = config.external if config.external else default_external(self.home)
        self.sources: dict[str, SqliteSource] = {}
        self.errors: dict[str, float] = {}
        # A store that cannot be read is as far behind as the time since it last could: that
        # counts in the lag, raises the `store` alarm and travels to the standby (``stale``).
        self.last_ok: dict[str, float] = {}
        self.failing: dict[str, float] = {}
        # seq -> (created, bytes, files, bytes of first seeds): a generation's first seed of a
        # store is its unavoidable base and does not count against the spool cap.
        self.unacked: dict[int, tuple[float, int, list[Path], int]] = {}
        self.seeded_in_generation: set[str] = set()
        self.generation, self.seq, self.acked = '', 0, 0
        self.names = Names(self.dir / 'ids.json')
        # Snapshots are named by this run and a number within it: the standby orders them by
        # (run, seq) and acknowledges one by that version, so an ack from before a restart
        # never acknowledges a snapshot of this run.
        self.tree_run = f'{self.names.next():020d}'
        self.trees = {kind: TreeSnapshotter(self.home, self.outbox, kind, clock=clock, run=self.tree_run,
                                            exclude=withheld_files(self.home) if kind == 'files' else None)
                      for kind in KINDS}
        self.tree_seq = {kind: 0 for kind in KINDS}
        self.tree_pending: dict[str, tuple[str, Path]] = {}
        self.tree_acked: dict[str, Path] = {}
        self.tree_due = {kind: 0.0 for kind in KINDS}
        self.handled_requests: set[str] = set()
        self.fenced = False
        self.paused = False
        self.last_written = self.last_push = self.last_discover = self.last_anti_entropy = 0.0
        # Anti-entropy walks the stores one per tick, so a pass never stalls one tick for the
        # time it takes to re-read every store.
        self.entropy_queue: list[str] = []
        self.entropy_tries = 0
        self.entropy_repaired = 0
        self.shipped = {'segments': 0, 'pages': 0, 'seed_bytes': 0, 'segment_bytes': 0, 'mirrored_bytes': 0}
        self.lag = 0.0
        self.new_generation('start')

    # -- generations ---------------------------------------------------------------------
    def new_generation(self, reason: str, *, above: int = 0) -> None:
        now = self.clock()
        self.generation, self.seq, self.acked = _generation_id(self.names.next(above)), 0, 0
        self.unacked.clear()
        self.seeded_in_generation = set()
        self.entropy_queue = []
        for source in self.sources.values():
            source.forget()
        if self.outbox.is_dir():
            for old in self.outbox.glob('gen-*'):
                shutil.rmtree(old, ignore_errors=True)
        (self.outbox / self.generation).mkdir(parents=True, exist_ok=True)
        self.last_anti_entropy = now
        self.journal.event('generation', generation=self.generation, reason=reason)

    def spool_bytes(self) -> int:
        return sum(entry[1] for entry in self.unacked.values())

    def capped_bytes(self) -> int:
        """Unacknowledged bytes beyond the generation's base seeds: what the cap limits."""
        return sum(entry[1] - entry[3] for entry in self.unacked.values())

    # -- the tick ------------------------------------------------------------------------
    def tick(self) -> dict:
        now = self.clock()
        if self.fenced:
            return self.status(now)
        self._collect()
        if self.fenced:
            return self.status(now)
        request = self.dir / 'reseed.request'
        if request.exists():
            reason = request.read_text(errors='replace').strip()[:120]
            request.unlink(missing_ok=True)
            self.new_generation(f'operator: {reason}')
        if now - self.last_discover >= self.config.rediscover_interval or not self.sources:
            self._discover(now)
        if not self._space_ok():
            return self.status(now)
        if self.capped_bytes() > self.config.max_spool_bytes:
            self.journal.alarm('spool_cap', True, f'{self.capped_bytes()} bytes unacknowledged')
            self.new_generation('spool-cap')
        if not self.entropy_queue and now - self.last_anti_entropy >= self.config.anti_entropy_interval:
            self.entropy_queue, self.entropy_tries, self.entropy_repaired = sorted(self.sources), 0, 0
            self.last_anti_entropy = now
        checking = self.entropy_queue[0] if self.entropy_queue else None
        entries, short = [], []
        # A delta is consumed when it is polled (the source moves on): until its segment is
        # published, a failure anywhere loses it, so the stores that produced one start again.
        try:
            for key, source in sorted(self.sources.items()):
                try:
                    delta = source.poll(anti_entropy=key == checking)
                except SpaceError as error:
                    short.append(str(error))
                    self._source_error(key, error, now)
                    continue
                except (SourceError, sqlite3.Error, OSError) as error:
                    self._source_error(key, error, now)
                    continue
                self.errors.pop(key, None)
                self.failing.pop(key, None)
                self.last_ok[key] = now
                if delta is not None:
                    entries.append(delta)
                    if delta.repaired:
                        self.journal.alarm('anti_entropy', True, f'{key}: {delta.repaired} pages repaired')
                        self.journal.event('anti-entropy-repair', store=key, pages=delta.repaired)
                    if key == checking and delta.reason == 'anti-entropy':
                        self.entropy_repaired += delta.repaired
                        checking = None
            self.journal.alarm('disk', bool(short), '; '.join(short)[:400])
            if self.entropy_queue:
                self.entropy_tries += 1
                # A store that could not be checked this tick (released, just re-seeded, an error)
                # is retried a few times, then left to the next pass.
                if checking is None or self.entropy_queue[0] not in self.sources or self.entropy_tries >= 10:
                    if checking is not None:
                        self.journal.event('anti-entropy-skipped', store=self.entropy_queue[0])
                    self.entropy_queue.pop(0)
                    self.entropy_tries = 0
                    if not self.entropy_queue:
                        self.journal.event('anti-entropy', stores=len(self.sources), repaired=self.entropy_repaired)
            if entries or now - self.last_written >= self.config.heartbeat_interval:
                self._write_segment(entries, now)
        except BaseException as error:
            self._lose(entries, error)
            raise
        for kind in KINDS:
            self._maybe_snapshot_tree(kind, now)
        if now - self.last_push >= self.config.push_interval:
            self._collect()                # one that finished during this tick is handled first
            if self.fenced:
                return self.status(now)
            if self.transfers.start():
                self.last_push = now
                self.transfers.wait(self.config.transfer_wait)
                self._collect()
        oldest = min((entry[0] for entry in self.unacked.values()), default=None)
        self.lag = 0.0 if oldest is None else max(0.0, now - oldest)
        behind = {key: now - since for key, since in self.failing.items()}
        self.lag = max([self.lag, *behind.values()])
        late = sorted(key for key, seconds in behind.items() if seconds > self.config.max_lag_seconds)
        self.journal.alarm('store', bool(late), ', '.join(f'{k}: unreadable for {behind[k]:.0f}s' for k in late)[:400])
        self.journal.alarm('lag', self.lag > self.config.max_lag_seconds, f'{self.lag:.1f}s behind')
        return self.status(now)

    def _source_error(self, key: str, error: Exception, now: float) -> None:
        self.failing.setdefault(key, self.last_ok.get(key, now))
        if now - self.errors.get(key, 0.0) >= 60:
            self.journal.event('store-error', store=key, error=type(error).__name__, detail=str(error)[:200])
            self.errors[key] = now

    def _discover(self, now: float) -> None:
        self.last_discover = now
        found = discover(self.home, self.external)
        for key in set(self.sources) - set(found):
            self.sources.pop(key).close()
            self.failing.pop(key, None)
            self.last_ok.pop(key, None)
            self.journal.event('store-gone', store=key)
        for key, path in found.items():
            if key not in self.sources:
                self.sources[key] = SqliteSource(key, path, self.scratch, release_bytes=self.config.release_bytes,
                                                 min_free_bytes=self.config.min_free_bytes)

    def _space_ok(self) -> bool:
        try:
            usage = shutil.disk_usage(self.dir)
        except OSError:
            return True
        low = usage.free < self.config.min_free_bytes
        if low and not self.paused:
            self.paused = True
            self.journal.alarm('disk', True, f'{usage.free} bytes free')
            for source in self.sources.values():
                source.close()
            shutil.rmtree(self.outbox / self.generation, ignore_errors=True)
            self.unacked.clear()
        elif not low and self.paused:
            self.paused = False
            self.journal.alarm('disk', False)
            self.new_generation('disk-recovered')
        return not self.paused

    def _write_segment(self, deltas, now: float) -> None:
        """Publish one segment; the sequence and the spool account move only once it is."""
        seq = self.seq + 1
        directory = self.outbox / self.generation
        paths, entries, seed_bytes, base, based = [], [], 0, 0, []
        try:
            for index, delta in enumerate(deltas):
                meta = {'key': delta.key, 'page_size': delta.page_size, 'page_count': delta.page_count,
                        'reason': delta.reason, 'digest': delta.digest, 'repaired': delta.repaired, 'seed': None}
                if delta.seed is not None:
                    sha = segment.file_sha256(delta.seed)
                    size = delta.seed.stat().st_size
                    name = f'seed-{seq:012d}-{index}-{sha[:24]}.db'
                    os.replace(delta.seed, directory / name)
                    paths.append(directory / name)
                    meta['seed'] = {'name': name, 'sha256': sha, 'size': size}
                    seed_bytes += size
                    if delta.key not in self.seeded_in_generation:
                        based.append(delta.key)
                        base += size
                entries.append((meta, delta.pages))
            # Every store the primary has, in every segment: the standby drops a replica not listed.
            header = {'generation': self.generation, 'seq': seq, 'created_at': now, 'instance': self.instance,
                      'stores': sorted(self.sources), 'stale': dict(sorted(self.failing.items()))}
            paths.append(segment.write(directory, header, entries))
            size = sum(p.stat().st_size for p in paths)
        except BaseException:
            for path in paths:
                path.unlink(missing_ok=True)
            raise
        self.seq = seq
        self.seeded_in_generation.update(based)
        if base:
            base = size              # the segment that starts a store in this generation
        self.unacked[seq] = (now, size, paths, base)
        self.last_written = now
        self.shipped['segments'] += 1
        self.shipped['pages'] += sum(len(delta.pages) for delta in deltas)
        self.shipped['segment_bytes'] += size - seed_bytes
        self.shipped['seed_bytes'] += seed_bytes
        self.journal.alarm('spool_write', False)
        if self.capped_bytes() <= self.config.max_spool_bytes:
            self.journal.alarm('spool_cap', False)

    def _lose(self, deltas, error: BaseException) -> None:
        """Deltas polled but never published: their stores are shipped again from a snapshot
        (a seed within this generation), so no change is lost and no sequence number skipped."""
        for delta in deltas:
            if delta.seed is not None:
                delta.seed.unlink(missing_ok=True)
            source = self.sources.get(delta.key)
            if source is not None:
                source.forget()
        try:
            self.journal.alarm('spool_write', True, f'{type(error).__name__}: {error}'[:200])
            self.journal.event('segment-lost', stores=sorted(d.key for d in deltas), error=type(error).__name__)
        except OSError:
            pass

    def _maybe_snapshot_tree(self, kind: str, now: float) -> None:
        interval = self.config.vector_interval if kind == 'vectors' else self.config.files_interval
        if now < self.tree_due[kind]:
            return
        self.tree_due[kind] = now + interval
        snapshotter = self.trees[kind]
        try:
            if snapshotter.fingerprint() == snapshotter.last_fingerprint:
                return
            self.tree_seq[kind] += 1
            manifest = snapshotter.snapshot(self.tree_seq[kind])
        except (OSError, ValueError) as error:
            self.journal.event('tree-snapshot-failed', kind=kind, error=type(error).__name__)
            self.tree_due[kind] = now + min(interval, 30.0)
            return
        previous = self.tree_pending.get(kind)
        self.tree_pending[kind] = (manifest_version(manifest.name), manifest)
        if previous is not None:
            previous[1].unlink(missing_ok=True)
        self._prune_blobs()
        self.journal.event('tree-snapshot', kind=kind, seq=self.tree_seq[kind], manifest=manifest.name)

    def _prune_blobs(self) -> None:
        manifests = [p for _, p in self.tree_pending.values()] + list(self.tree_acked.values())
        keep = {p.name for p in manifests}
        trees = self.outbox / 'trees'
        for path in trees.glob('*.json'):
            if MANIFEST.match(path.name) and path.name not in keep:
                path.unlink(missing_ok=True)
        wanted = referenced_blobs(manifests)
        blobs = trees / 'blobs'
        if blobs.is_dir():
            for blob in blobs.iterdir():
                if not blob.name.startswith('.') and blob.name not in wanted:
                    blob.unlink(missing_ok=True)

    def _collect(self) -> None:
        """Take the result of a finished transfer: the transport alarm, then the ack."""
        result = self.transfers.take()
        if result is None:
            return
        error = result.get('error')
        self.shipped['mirrored_bytes'] += int(result.get('mirrored') or 0)
        detail = '' if error is None else (str(error) if isinstance(error, TransportError)
                                           else f'{type(error).__name__}: {error}')
        self.journal.alarm('transport', error is not None, detail[:200])
        ack = result.get('ack')
        if ack:
            self._handle_ack(ack)

    def _handle_ack(self, ack: dict) -> None:
        if ack.get('promoted'):
            self.fenced = True
            for source in self.sources.values():
                source.close()
            self.journal.alarm('fenced', True, 'the standby was promoted; this instance no longer ships')
            self.journal.event('fenced', promoted=ack['promoted'])
            return
        standby_generation = str(ack.get('generation') or '')
        if standby_generation > self.generation:
            # The standby is on a generation that sorts after ours (our clock stepped back and
            # our record of names was lost): it would never apply ours. Start again above it.
            self.new_generation('standby-ahead', above=_generation_number(standby_generation))
            return
        if ack.get('generation') == self.generation:
            applied = int(ack.get('seq') or 0)
            for seq in [s for s in self.unacked if s <= applied]:
                for path in self.unacked.pop(seq)[2]:
                    path.unlink(missing_ok=True)
            self.acked = max(self.acked, applied)
        request = ack.get('reseed') or {}
        nonce = request.get('nonce')
        if nonce and nonce not in self.handled_requests and request.get('generation') in (self.generation, None, ''):
            self.handled_requests.add(nonce)
            self.new_generation(f'standby-request: {request.get("reason", "")}'[:120])
        runs = [int(m.group(1)) for value in [*(ack.get('trees') or {}).values(), *(ack.get('tree_failed') or {}).values()]
                if (m := _TREE_VERSION.match(str(value or '')))]
        if runs and max(runs) > int(self.tree_run):
            self._new_tree_run(max(runs))
            return
        for kind in KINDS:
            pending = self.tree_pending.get(kind)
            if pending is None:
                continue
            installed = str((ack.get('trees') or {}).get(kind) or '')
            if installed.split('-')[0] == self.tree_run and installed >= pending[0]:
                old = self.tree_acked.get(kind)
                self.tree_acked[kind] = pending[1]
                del self.tree_pending[kind]
                if old is not None and old != pending[1]:
                    old.unlink(missing_ok=True)
                self._prune_blobs()
            elif (ack.get('tree_failed') or {}).get(kind) == pending[0]:
                self.tree_due[kind] = 0.0
                self.trees[kind].last_fingerprint = None

    def _new_tree_run(self, above: int) -> None:
        """The standby holds a snapshot of a run above ours, so it would refuse all of ours:
        name every later snapshot from a run above it and take one of each kind now."""
        self.tree_run = f'{self.names.next(above):020d}'
        for kind in KINDS:
            self.trees[kind].run = self.tree_run
            self.trees[kind].last_fingerprint = None
            self.tree_seq[kind], self.tree_due[kind] = 0, 0.0
            pending = self.tree_pending.pop(kind, None)
            if pending is not None:
                pending[1].unlink(missing_ok=True)
        self._prune_blobs()
        self.journal.event('tree-run', run=self.tree_run, reason='standby-ahead')

    def status(self, now: float | None = None) -> dict:
        now = self.clock() if now is None else now
        status = {
            'role': 'primary', 'at': now, 'generation': self.generation, 'seq': self.seq, 'acked_seq': self.acked,
            'lag_seconds': round(self.lag, 3), 'unacked_segments': len(self.unacked),
            'spool_bytes': self.spool_bytes(), 'fenced': self.fenced, 'paused': self.paused,
            'transport': self.transport.describe(), 'alarms': dict(self.journal.alarms), 'shipped': dict(self.shipped),
            'trees': {kind: {'pending': p[0]} for kind, p in self.tree_pending.items()},
            'stores': {key: {'wal_pages': s.stats.wal_frames_pages, 'resyncs': s.stats.resyncs, 'seeds': s.stats.seeds,
                             'lossless_restarts': s.stats.lossless_restarts, 'releases': s.stats.releases,
                             'anti_entropy_pages': s.stats.anti_entropy_pages, 'error': key in self.errors,
                             'unreadable_since': self.failing.get(key)}
                       for key, s in sorted(self.sources.items())},
        }
        write_json(self.dir / 'status.json', status, durable=False)
        return status

    def close(self) -> None:
        self.transfers.wait(1.0)       # a blocked transfer is left to its own timeout
        for source in self.sources.values():
            source.close()
