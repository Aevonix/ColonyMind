#!/usr/bin/env python3
"""Measure the memory twin on this machine: two local instances, a timed write load.

Reports end-to-end replication lag (commit on the primary to visible on the standby), bytes
shipped against a whole-file copy per change, the recovery point and duration of a promotion
under load, the time to re-seed, and pages repaired by anti-entropy (expected 0).

    python scripts/twin_bench.py [--seconds 20] [--interval 0.5] [--push-interval 1.0] [--rate 200]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sidecar'))

from colonymind.twin.config import TwinConfig, save  # noqa: E402
from colonymind.twin.primary import Shipper  # noqa: E402
from colonymind.twin.standby import Standby  # noqa: E402


def _percentile(values, q):
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 3) if ordered else None


def run(seconds: float, interval: float, push: float, rate: float) -> dict:
    base = Path(tempfile.mkdtemp(prefix='twin-bench-'))
    primary, standby_home = base / 'primary', base / 'standby'
    primary.mkdir()
    standby_home.mkdir()
    common = dict(interval=interval, push_interval=push, heartbeat_interval=5.0, anti_entropy_interval=seconds / 2)
    cp = TwinConfig(role='primary', target=str(standby_home), external=[str(base / 'none.sqlite3')], **common)
    cs = TwinConfig(role='standby', **common)
    save(primary, cp)
    save(standby_home, cs)
    conn = sqlite3.connect(primary / 'load.db', isolation_level=None, check_same_thread=False)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('CREATE TABLE c (n INTEGER PRIMARY KEY, at REAL, blob BLOB)')
    conn.executemany('INSERT INTO c VALUES (?, ?, ?)', [(-i, 0.0, os.urandom(1000)) for i in range(1, 20001)])
    shipper, standby = Shipper(primary, cp), Standby(standby_home, cs)
    committed: dict[int, float] = {}
    stop, applying = threading.Event(), threading.Event()
    applying.set()
    counts = {'written_bytes': 0}

    def writer():
        n = 0
        while not stop.is_set():
            blob = os.urandom(random.randint(100, 2000))
            conn.execute('INSERT INTO c VALUES (?, ?, ?)', (n, time.time(), blob))
            committed[n] = time.time()
            counts['written_bytes'] += len(blob) + 16
            n += 1
            time.sleep(1.0 / rate)

    def ship():
        while not stop.is_set() and not shipper.fenced:
            shipper.tick()
            time.sleep(interval)

    def apply():
        while not stop.is_set():
            if applying.is_set():
                standby.cycle()
            time.sleep(interval)

    lags, seen = [], -1
    threads = [threading.Thread(target=f) for f in (writer, ship, apply)]
    for thread in threads:
        thread.start()
    replica = standby_home / '.twin' / 'replica' / 'db' / 'load.db'
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if replica.exists():
            probe = sqlite3.connect(replica.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
            top = probe.execute('SELECT max(n) FROM c').fetchone()[0]
            probe.close()
            now = time.time()
            if top is not None and top > seen:
                lags.extend(now - committed[n] for n in range(seen + 1, top + 1) if n in committed)
                seen = top
        time.sleep(0.02)
    shipped = dict(shipper.shipped)
    whole_file = (primary / 'load.db').stat().st_size
    ticks_with_changes = shipper.sources['load.db'].stats.reasons.get('wal', 0)
    # Re-seed: time for the standby to hold a verified copy from a new generation.
    started = time.monotonic()
    (primary / '.twin' / 'reseed.request').write_text('bench\n')
    generation = shipper.generation
    while shipper.generation == generation or standby.state['generation'] != shipper.generation or \
            (standby.state['stores'].get('load.db') or {}).get('generation') != shipper.generation:
        time.sleep(0.02)
    reseed_seconds = time.monotonic() - started
    time.sleep(2 * push + 2 * interval)
    # Promotion under load.
    applying.clear()
    time.sleep(interval)
    at_start = conn.execute('SELECT max(n) FROM c').fetchone()[0]
    started = time.monotonic()
    report = standby.promote()
    promote_seconds = time.monotonic() - started
    promoted = sqlite3.connect(standby_home / 'load.db').execute('SELECT max(n) FROM c').fetchone()[0]
    fence_started = time.monotonic()
    while not shipper.fenced and time.monotonic() - fence_started < 10:
        time.sleep(0.02)
    stop.set()
    for thread in threads:
        thread.join()
    stats = shipper.sources['load.db'].stats
    return {
        'settings': {'seconds': seconds, 'interval': interval, 'push_interval': push, 'rate_per_s': rate},
        'lag_seconds': {'samples': len(lags), 'p50': _percentile(lags, 0.5), 'p95': _percentile(lags, 0.95),
                        'max': round(max(lags), 3) if lags else None,
                        'mean': round(statistics.mean(lags), 3) if lags else None},
        'bytes': {'payload_written': counts['written_bytes'], 'segments': shipped['segment_bytes'],
                  'seeds': shipped['seed_bytes'], 'whole_file_copy_per_changed_tick': whole_file * ticks_with_changes},
        'wal': {'pages_shipped': stats.wal_frames_pages, 'lossless_restarts': stats.lossless_restarts,
                'releases': stats.releases, 'resyncs': stats.resyncs, 'seeds': stats.seeds,
                'reasons': stats.reasons},
        'anti_entropy': {'runs': stats.anti_entropy_runs, 'pages_repaired': stats.anti_entropy_pages},
        'reseed_seconds': round(reseed_seconds, 3),
        'promote': {'seconds': round(promote_seconds, 3), 'rows_behind_at_start': at_start - promoted,
                    'fenced_after_seconds': round(time.monotonic() - fence_started, 3),
                    'forced': report['promoted']['forced']},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--interval', type=float, default=0.5)
    parser.add_argument('--push-interval', type=float, default=1.0)
    parser.add_argument('--rate', type=float, default=200, help='commits per second')
    args = parser.parse_args()
    print(json.dumps(run(args.seconds, args.interval, args.push_interval, args.rate), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
