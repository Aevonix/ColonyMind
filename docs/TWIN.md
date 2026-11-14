# The memory twin

A live, read-only replica of one instance on a second machine: every SQLite store the sidecar
owns, the plugin's turn outbox, consistent snapshots of the vector store and the instance's
other files. The standby serves reads, and one command promotes it to primary. The twin is a
separate process beside the sidecar; the sidecar, the plugin and every store are unchanged,
and an instance that never runs `colonymind twin init` runs exactly as before.

Code: `sidecar/colonymind/twin/`. Tests: `sidecar/tests/test_twin_*.py`. Measurement:
`scripts/twin_bench.py`.

## At a glance

| | |
|---|---|
| What travels | every SQLite file in the instance directory (found by its header, so new stores are picked up), configured outside stores (by default the plugin's turn outbox), the vector store, all other files |
| How | committed WAL frames, read under a held read transaction; a consistent snapshot diffed page by page when that is not possible |
| Lag | about 0.75 s median and 1.5 s p95 at the defaults, 0.2 s and 0.37 s at a 0.1 s poll (measured, below) |
| Direction | one way. The standby writes only its replicas and its `ack.json`; the primary writes only the standby's `inbox` |
| Promotion | `colonymind twin promote`: apply what has arrived, check, fence the old primary, move the replicas into place |
| Disk safety | a spool cap and a free-space floor. Past either, the spool is dropped and a new generation re-seeds; the disk does not fill. A snapshot (seed or resync) is taken only when the free space left after a full copy of the store stays above the floor, measured inside the read transaction the copy is taken from (its page count, plus the log), so a checkpoint cannot make it undercount; otherwise the store waits and the `disk` alarm names it |
| Audit | `.twin/events.jsonl` on both sides; `colonymind twin status` (exit 1 while an alarm is raised) |
| Off | stop `twin run`, then `colonymind twin init --off`, or never run it; nothing else refers to the twin |

## Compared with the usual design

A common memory twin replicates typed stores logically, row by row, mirrors files every few
minutes, keeps writes off the copy with a front door, and caps replication slots. This twin:

| The usual design | This twin |
|---|---|
| Typed stores replicated logically, row by row, because physical replication was impossible across CPU architectures | SQLite's file format is the same on every architecture, so pages replicate physically across architectures. Nothing to map per table, and a new store needs no code |
| A write round-trips in under three seconds | Commit on the primary to visible on the standby: 0.75 s median, 1.49 s p95, 1.95 s max at the defaults; 0.20 / 0.37 / 1.30 s at `--interval 0.1` (`scripts/twin_bench.py`, 200 commits/s) |
| Files mirrored every five minutes | Vector store and files snapshotted at most every five minutes, and only when they changed. A Lance table is taken at one committed version, and the standby opens and scans every table before the snapshot becomes current |
| A front door routes writes to the primary so the copy does not fork (it can, when a write bypasses the door) | Forking is impossible by construction: replicas are 0400 files opened `immutable`, `twin query` runs under an authorizer that denies every write, and the applier is the only writer. A replica altered on the standby is caught by digest verification and re-seeded (tested) |
| Replication slots: a lag past the cap invalidates the slot and forces a re-seed | The same, plus a free-space floor. The cap counts what grew beyond a generation's base seeds, so a store larger than the cap cannot loop re-seeds (tested) |
| A conversation can move between the two bodies | Captured turns not yet delivered travel in the turn outbox and are installed (or merged with `INSERT OR IGNORE`) where the standby's plugin reads them, so they are delivered after a failover (tested) |
| (no equivalent) | Verification: every snapshot entry carries a whole-store page digest that the standby checks; a periodic anti-entropy pass re-reads every store and counts pages the fast path missed (0 in every test and benchmark run) |
| (no equivalent) | Promotion fences the old primary: its shipper stops at its next ack read and raises the `fenced` alarm, and both transports refuse to write to a promoted directory |

## How it works

### Stores: WAL frames under a pin

Each store is opened read-only (`mode=ro`), so the replicator cannot change a primary store;
a test checks that a write through that connection fails.

A WAL store's committed changes are the frames of its write-ahead log (`wal.py` reads the
documented format: salts, cumulative checksums, commit frames). Frames of a transaction still
being written, a torn frame and a stale frame from an earlier log generation are never read,
because the scan stops at the last valid commit frame.

A checkpoint can restart the log and overwrite frames not yet read. The source prevents that
by holding a read transaction between polls, the pin. SQLite does not restart the log under a
reader that uses it, and a reader on read-mark 0 blocks every backfill, so a restart seen
while the pin was held continuously cannot have lost a frame. The source then follows the new
generation. The pin is released for one poll when the log passes `release_bytes` (16 MiB), so
the application's own checkpoints can restart it and the primary's WAL stays bounded; a test
checks the bound under constant checkpoints. A restart seen after an unpinned moment may have
lost frames. The source then resynchronises: a consistent snapshot taken with SQLite's backup
API inside the pin, the current log generation replayed on top, and only pages whose final
image differs from what was shipped are sent. Replaying the whole current generation onto a
snapshot taken inside the pin yields exactly the latest commit.

A pin on read-mark 0 does not stop a restart of a fully backfilled log, and a restart can
begin while a scan is reading it: the new generation overwrites the first frames, and a read
that straddles it would return an old prefix, older than frames it never reached. Every scan
therefore reads the log header again after its frames and discards itself when the header
changed; it reads again (the new generation then holds only commits newer than the pin).

Rollback-journal stores have no log. They are resynchronised when the header's change
counter moves, which takes a shared lock only for the length of the backup.

The source never takes the write lock and never checkpoints. Writers are not blocked: in the
concurrent tests and in the end-to-end run, the writers saw zero errors.

Anti-entropy. Every `anti_entropy_interval` (15 minutes) each store is resynchronised in
full, one store per poll so that a pass never stalls a single poll. Pages it finds that the fast path had not shipped are counted
(`anti_entropy_pages`), repaired, and raise the `anti_entropy` alarm. That count was 0 in
every run. A test plants a miss and checks it is found and repaired.

Page 1 is compared without the two header counters (offsets 24 and 92) that SQLite rewrites
in any copy of a database. Apart from those counters, a replica is byte-identical to the
primary at the commit it holds.

### Segments and generations

A poll that found changes writes one segment (`segment.py`) holding, for each store, the page
images and the store's size after them. A store that is mostly new travels as a seed file (the
snapshot itself) beside the segment. Every file is written under a temporary name, fsynced and
renamed, and carries the SHA-256 of its bytes in its name, so a partial or altered file is
never applied. When nothing changes, an empty heartbeat segment is written every 5 s, so the
lag stays measurable.

A generation is one unbroken sequence of segments starting from seeds. A new generation starts
when the shipper starts, when the spool cap or the free-space floor is crossed, on
`twin reseed` on either side, and when the standby finds a store it cannot verify. Old
generations are deleted from the spool.

Generations (and snapshot runs, below) are numbered by the real clock in nanoseconds, but
never at or below the last number used, which the primary keeps in `.twin/ids.json`: a clock
that steps back across a restart cannot give a name the standby would sort as old. If that
record is lost as well, the standby's ack names a generation or run above the primary's, and
the primary starts again above it (tested with the clock stepped back an hour, with and
without the record).

### Trees: vector store and files

`trees.py` snapshots a directory as a manifest `{path: [sha256, size, mode]}` plus
content-addressed blobs. A file already on the standby is not sent again. On the primary a blob
is a hard link, so a snapshot costs no disk and a file the application deletes mid-snapshot
stays readable. Small ordinary files are copied first, so an in-place rewrite cannot change a
blob after it was hashed.

`vectors` takes each Lance table at one version V, its latest committed manifest, together
with the data, deletion and index files no newer than that manifest. Lance writes a version's
files before committing its manifest and never rewrites a file. The copy holds exactly one
version, with no history. The standby opens every table, checks it is at version V and scans a
column of every row before switching `current` to the new snapshot. A snapshot that fails is
reported in the ack and replaced (tested with a corrupted blob), and the `tree` alarm is up
while the newest snapshot of a kind is rejected. A damaged inbox file (a blob or a segment
whose bytes do not match the digest its name carries) is removed by the standby, the one thing
it removes from the inbox, so the mirror sends it again: a mirror that compares sizes (or sizes
and times) would otherwise never replace it.

A snapshot's version is `<run>-<seq>`: the shipper process that made it (numbered as
generations are, above every earlier run) and a number within that run. The standby installs
the highest version and acknowledges it by that version; the primary counts an ack only for a
version of its own run, so after a restart of the shipper (whose numbers start again) an old
ack never acknowledges a new snapshot and a new snapshot is never refused as old.

`files` covers everything else in the instance directory: identity, configuration, source
media and upgrade backups (`backups/`, which travel as files rather than as live stores). It
excludes live stores, the vector store, `.twin` and transient files (logs, pid, lock and socket
files).

### Transport

The spool (`.twin/outbox`) is mirrored to the standby's `.twin/inbox`. A local target (a path,
or a mounted share) is mirrored in Python. `ssh://[user@]host[:port]/path` uses `rsync` over
`ssh` with one persistent connection. Every spool file is immutable and uniquely named, so the
mirror copies what is missing and deletes what is gone. The primary reads the standby's
`ack.json` the same way and deletes acknowledged files.

A slow, full or unreachable standby costs the primary spooled files, never a failed or blocked
write. The shipper is a separate process, and its transport failures raise the `transport`
alarm. A test runs the primary with no standby at all.

A transfer (the ack read, then the mirror) runs on its own thread, one at a time, and a tick
waits for it at most `transfer_wait` (0.25 s). A partition that blocks rsync until its timeout
therefore never stops polling: the pins are still released at `release_bytes` and the free
space is still checked, so the primary's log stays bounded (tested with a transport that
blocks). The result of a finished transfer is taken on the tick's thread, and a new transfer
starts only once it has been: an ack, a re-seed request or the fence that arrives while the
tick is polling is handled, never overwritten (tested). rsync's exit code 24
(source files vanished) is a success: the spool is pruned while a transfer runs.

Why not an existing tool. Litestream (Apache-2.0) is a Go binary that is neither pip- nor
apt-installable, and it manages checkpoints itself, adding its own tables to each database.
`sqlite3_rsync` (public domain) arrived with SQLite 3.47, is missing from the SQLite that
common distributions ship, and hashes every page on every run. The page-level writes both
rely on (`sqlite_dbpage`, the session extension) are not compiled into CPython's `sqlite3`,
and APSW's wheels lack `sqlite_dbpage` too. What remains is the documented WAL format, about
120 lines. `rsync` is used only as an external program, never linked.

### The standby: apply, publish, serve reads

A batch of segments is applied to a copy of the published replica (copy-on-write where the
filesystem allows it: `FICLONE`, `clonefile`), which is fsynced, set to 0400 and renamed over
the replica. A reader holding the old file keeps a consistent snapshot, and a new reader sees
the new one. A crash mid-batch leaves the published file untouched, and the batch is applied
again, since page images are idempotent. Each whole-store digest is checked after its entry is
applied. A mismatch keeps the last good replica, raises `diverged` and requests a re-seed.

Reads come from the replicas. In Python, `colonymind.twin.standby.connect(home, 'colonymind-comms.db')`
gives a read-only connection (`@external/<file>` for outside stores), and
`tree_path(home, 'vectors')` gives a LanceDB directory. From a shell:

```
colonymind --instance ~/twin twin query colonymind-comms.db "SELECT count(*) FROM communications"
```

### Promotion

`colonymind twin promote`:

1. Stops a running applier. It asks through `.twin/promote.request` and takes the run lock.
2. Applies everything complete in the inbox and installs the newest valid trees.
3. Refuses (unless `--force`) if a store the primary listed has not been seeded in the current
   generation, has diverged, or fails `PRAGMA quick_check`, or if a tree (`files`, `vectors`)
   has no installed snapshot or its newest snapshot was rejected.
4. Fences. It writes `promoted` to `ack.json` and the `.twin/promoted` marker. The old primary's
   shipper stops at its next ack read, and both transports refuse to mirror into the directory.
5. Moves each replica to its place in the instance directory, the vector snapshot to
   `lancedb/` and the files into place. Anything already there is moved to
   `.twin/displaced/<stamp>/`, never deleted. Files named in `keep_local` (default
   `colonymind.yaml`) are kept if present, and the primary's copy is written beside them as
   `<name>.from-primary`. Outside stores go to `--external-dir`; an existing outbox there
   receives the missing rows with `INSERT OR IGNORE`.
6. Records `.twin/promotion-<stamp>.json` and turns the configuration into a primary with no
   standby attached.

A promotion that fails after its state is saved (a fence that cannot be written, a file that
will not move, a locked outside store, a crash) is resumed by running `twin promote` again:
the fence is published again unless it already names this promotion (the standby's runner
does the same on every cycle meanwhile), the placement is idempotent (what moved is no longer
in the replica, and a replica is made writable before it moves), and nothing that arrived
after the fence is applied.

Then start the sidecar on that machine, point the body at it, and stop the old primary if it is
still running. The alarm hook receives `COLONYMIND_TWIN_ALARM=fenced` on the old primary for
that purpose. To protect the new primary, attach a new standby with `twin init --role primary
--target ...`. Re-initialise the old primary as a standby in a fresh directory. Its old stores
are stale.

The promoted state is a prefix of the primary's history: each store holds the primary's state
at some commit, the last one that arrived. Promotion does not make a cut across stores: each
store is individually consistent, which is also what the primary guarantees across a crash.
The end-to-end tests promote while a writer is running and check the prefix property row by
row.

### Lag and alarms

- Primary lag: the age of the oldest segment not yet acknowledged, measured on the primary's
  clock alone. Standby lag: the standby's clock minus the primary time of the newest applied
  segment. It assumes synchronised clocks, and the standby's own `lag` alarm uses it, so an
  unsynchronised clock shows up as lag.
- A store the primary cannot read is as far behind as the time since it last could: that
  counts in the primary's lag, raises the `store` alarm past `max_lag_seconds`, and travels in
  each segment (`stale`), so heartbeats cannot hide it. Promotion refuses (unless `--force`)
  when the primary could not read a store for longer than `max_lag_seconds` before its last
  segment.
- A segment that cannot be written (an I/O error) takes no sequence number and loses no
  change: the stores whose changes it held are shipped again from a snapshot within the same
  generation, and the `spool_write` alarm is up until a segment is written.
- Alarms: `lag` (above `max_lag_seconds`, default 60), `store`, `transport`, `spool_cap`,
  `spool_write`, `disk`, `diverged`, `anti_entropy`, `tree`, `fenced`, and `runner_down`
  (reported by `twin status` when no runner has written status recently). Each raise and clear is logged and runs
  `alarm_command` with `COLONYMIND_TWIN_ALARM`, `COLONYMIND_TWIN_STATE` and
  `COLONYMIND_TWIN_DETAIL`.

### Privacy and retention

What leaves the primary is what the instance holds, sent to a directory you name. It includes
`api.key` and the identity files, because a promoted standby must be the same instance. The
sealed secret store's key (`secrets-vault.key`) follows the backup rule: it travels only when
`secrets.vault.key_in_backups` is on. By default the standby holds the store's ciphertext and not
its key, so after a promotion the sealed secrets stay unavailable until you install the key from
your own copy. The standby keeps no more than the primary does:

- A replica is the primary's pages, including SQLite free pages exactly as the primary has
  them.
- Spool segments are deleted once acknowledged, or when a generation ends.
- The previous tree snapshot is deleted when the next one is installed.

After an erasure, erased content on the standby lasts at most as long as the unacknowledged
spool and one tree interval (5 minutes by default) beyond the primary's own compaction. Use an
ssh target, or a share you trust, for the transport.

## Using it

On the standby machine (any empty directory becomes the future instance directory):

```
colonymind --instance ~/colonymind-twin twin init --role standby --external-dir ~/.hermes/state
colonymind --instance ~/colonymind-twin twin run          # under the service manager
```

On the primary:

```
colonymind twin init --role primary --target ssh://user@standby-host/home/user/colonymind-twin
colonymind twin run                                      # under the service manager, beside the sidecar
colonymind twin status                                   # exit 1 while an alarm is raised
```

| Command | Primary | Standby |
|---|---|---|
| `twin init` | `--target`, `--external PATH` (repeatable; default: the plugin's turn outbox), `--max-lag`, `--max-spool`, `--interval`, `--alarm-command`, `--ssh-command` | `--external-dir`, `--max-lag`, `--alarm-command` |
| `twin run` | poll, spool, snapshot, mirror | apply, install, ack |
| `twin status [--json]` | generation, seq, acked seq, lag, spool, per-store counters, alarms | generation, seq, lag, per-store verification, stale stores, trees, alarms |
| `twin reseed` | new generation on the next tick | asks the primary for one |
| `twin promote [--force]` | refused | promotes |
| `twin query STORE SQL` | | read-only query |

Configuration lives in `.twin/twin.json`. Other settings: `push_interval` (1 s), `transfer_wait` (0.25 s),
`heartbeat_interval` (5 s), `max_spool_bytes` (1 GiB), `min_free_bytes` (512 MiB),
`release_bytes` (16 MiB), `anti_entropy_interval` (900 s), `vector_interval` and
`files_interval` (300 s), `keep_local`.

## Measurements

`python scripts/twin_bench.py --seconds 20` runs two local instances with 200 commits/s of
100 to 2000 random bytes each on a 40 MB store, then forces a re-seed and promotes under
load:

| | defaults (poll 0.5 s, push 1 s) | poll 0.1 s, push 0.2 s |
|---|---|---|
| lag p50 / p95 / max | 0.75 / 1.49 / 1.95 s | 0.20 / 0.37 / 1.30 s |
| bytes shipped per byte written (incompressible payload) | 0.83 | 1.16 |
| whole-file copy per changed poll, for comparison | 393x more | 1780x more |
| anti-entropy pages repaired | 0 | 0 |
| re-seed of the 40 MB store until verified on the standby | 1.39 s | 0.68 s |
| promote duration | 0.05 s | 0.04 s |
| commits not yet on the standby when promotion started | 93 (about one push interval) | 4 |
| old primary fenced after | 0.99 s | 0.24 s |

The recovery point is about one push interval of writes. Lower `push_interval` for a smaller
one; each push over ssh costs one `rsync` call on a persistent connection.

## Limits

- Consistency is per store, as on the primary. There is no cross-store snapshot.
- Each apply batch copies the replica file, unless the filesystem supports copy-on-write. On a
  filesystem without it, a very large, busy store costs the standby one file copy per batch.
- A rollback-journal store is read under a shared lock during its snapshot, which delays that
  store's writers for the length of the backup.
- Promotion fences replication, not the old sidecar: if the old primary is still up and still
  used, stop it (the `fenced` alarm is the signal).
- Standby-side lag uses two clocks. Primary-side lag, which drives the alarm there, uses one.
- The runner holds a read transaction on every WAL store, so each store's WAL can grow to
  `release_bytes` before a checkpoint may restart it (SQLite's own default is about 4 MiB).
  Stop `twin run` before maintenance that needs exclusive access to the stores; it starts a
  new generation when it comes back.
- Tree snapshots hash new files inside the poll; after a large vector compaction one poll
  can take seconds.
