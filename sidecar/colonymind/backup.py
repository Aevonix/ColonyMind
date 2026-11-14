"""Full-state backup and restore for ColonyMind.

Creates a compressed, optionally encrypted archive of all ColonyMind state:
- All SQLite databases in COLONYMIND_STATE_DIR (auto-discovered)
- Original source images referenced by the canonical ledger snapshot
- The instance identity (instance-id, identity.yaml, colonymind.yaml; api.key and the decision log's
  secret instinct.key only in an encrypted archive, and an unencrypted one carries colonymind.yaml with its
  credential settings redacted)
- Config files (scrubbed of secrets)
- Learned weights (plasticity/)
- LanceDB vector store

Usage::

    # Backup
    colonymind backup --full --output ~/backups/

    # Stop the sidecar and other readers, restore, then start fresh readers.
    colonymind stop  # or colonymind service stop
    colonymind restore --full colonymind-backup-20260627T120000.tar.gz
"""

from __future__ import annotations

from contextlib import ExitStack, closing
import hashlib
import hmac
import json
import logging
import os
import re
import shutil
import sqlite3
import tarfile
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from colonymind.instance import INSTANCE_ID_FILE, instance_id, read_instance_id
from colonymind.util.temporal import now_utc

logger = logging.getLogger(__name__)

BACKUP_VERSION = 2

_SECRET_KEY_PATTERN = re.compile(
    r"(_KEY|_SECRET|_PASSWORD|_TOKEN|_PASSPHRASE)$", re.IGNORECASE
)


def _private_directory(path: Path) -> None:
    """Create a backup directory and any missing parents with owner-only access."""
    if not path.exists():
        if not path.parent.exists():
            _private_directory(path.parent)
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            return
        path.chmod(0o700)


def _private_file(path: Path) -> None:
    """Reserve a private destination before a writer opens it, including an existing copy."""
    _private_directory(path.parent)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        os.fchmod(fd, 0o600)
    finally:
        os.close(fd)


def _copy_private(source: Path, destination: Path) -> str:
    _private_file(Path(destination))
    copied = shutil.copyfile(source, destination)
    Path(destination).chmod(0o700 if Path(source).stat().st_mode & 0o100 else 0o600)
    return copied


def _make_tree_private(path: Path) -> None:
    """Keep moved upgrade state private, without following links outside the backup."""
    if path.is_symlink():
        return
    path.chmod(0o700 if path.is_dir() or path.stat().st_mode & 0o100 else 0o600)
    if path.is_dir():
        for child in path.iterdir():
            _make_tree_private(child)


def _copy_private_tree(source: Path, destination: Path) -> None:
    _private_directory(destination)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            _copy_private_tree(child, target)
        else:
            _copy_private(child, target)


def _copy_plasticity(source: Path, destination: Path, *, secrets: bool) -> None:
    _private_directory(destination)
    for child in source.iterdir():
        if child.name == ".lock" or child.name.endswith((".tmp", "-journal", "-wal", "-shm")):
            continue
        if not secrets and child.name in ("comparator.key", "query-vectors.db", "query-vectors.db-journal"):
            continue
        target = destination / child.name
        if child.is_dir():
            _copy_plasticity(child, target, secrets=secrets)
        elif child.suffix == ".db":
            with closing(sqlite3.connect(child.resolve().as_uri() + '?mode=ro', uri=True)) as db:
                _private_file(target)
                with closing(sqlite3.connect(target)) as copy:
                    db.backup(copy)
        else:
            _copy_private(child, target)


def _restore_plasticity(source: Path, destination: Path) -> None:
    from colonymind.plasticity.locking import StateLock

    def clear(directory: Path) -> None:
        for child in directory.iterdir():
            relative = child.relative_to(destination)
            if relative.as_posix() == ".lock" or (
                relative.as_posix() == "comparator.key" and not (source / relative).is_file()
            ):
                continue
            if child.name in ("query-vectors.db", "query-vectors.db-journal") and not (
                source / relative.with_name("query-vectors.db")
            ).is_file():
                continue
            if child.is_dir() and not child.is_symlink():
                clear(child)
                if not any(child.iterdir()):
                    child.rmdir()
            else:
                child.unlink()

    lock_existed = (destination / ".lock").exists()
    with StateLock(destination):
        clear(destination)
        _copy_plasticity(source, destination, secrets=True)
    # Restore already holds the exclusive instance lease and requires all readers
    # stopped. Do not leave a newly created lock as part of the restored weights.
    if not lock_existed:
        (destination / ".lock").unlink()


# ── Backup ───────────────────────────────────────────────────────────────


def create_full_backup(
    state_dir: str | Path,
    output_dir: str | Path,
    *,
    passphrase: Optional[bytes] = None,
    include_vectors: bool = True,
    include_host_paths: Optional[list[str]] = None,
) -> Path:
    """Create a full backup archive of ColonyMind state.

    Returns the path to the created archive.
    """
    state_dir = Path(state_dir)
    output_dir = Path(output_dir)
    _private_directory(output_dir)

    # The archive is bound to the instance it was taken from.
    instance = instance_id(state_dir)
    timestamp = now_utc().strftime("%Y%m%dT%H%M%SZ")
    base_name = f"colonymind-backup-{timestamp}"

    with tempfile.TemporaryDirectory(prefix="colonymind-backup-") as tmp:
        staging = Path(tmp) / base_name
        _private_directory(staging)

        db_manifest = _snapshot_databases(state_dir, staging / "databases")
        source_images = _snapshot_source_images(state_dir, staging)
        _snapshot_identity(state_dir, staging / "identity", secrets=passphrase is not None)
        _snapshot_config(state_dir, staging / "config")
        if (state_dir / "plasticity").is_dir():
            from colonymind.plasticity.locking import StateLock
            with StateLock(state_dir / "plasticity"):
                _copy_plasticity(state_dir / "plasticity", staging / "plasticity", secrets=passphrase is not None)

        if include_vectors:
            _snapshot_vectors(state_dir, staging / "vector")

        if include_host_paths:
            _snapshot_host_state(include_host_paths, staging / "host")

        meta = {
            "backup_version": BACKUP_VERSION,
            "colonymind_version": _get_colonymind_version(),
            "timestamp": now_utc().isoformat(),
            "instance_id": instance,
            "instance_id_hmac": _compute_identity_hmac(instance),
            "database_manifest": db_manifest,
            "source_images": source_images,
            "encrypted": passphrase is not None,
        }
        _private_file(staging / "meta.json")
        (staging / "meta.json").write_text(
            json.dumps(meta, indent=2) + "\n"
        )

        archive_path = output_dir / f"{base_name}.tar.gz"
        _create_archive(staging, archive_path)

        if passphrase is not None:
            encrypted_path = output_dir / f"{base_name}.tar.gz.enc"
            _encrypt_file(archive_path, encrypted_path, passphrase)
            archive_path.unlink()
            archive_path = encrypted_path

    logger.info("Full backup created: %s", archive_path)
    return archive_path


# ── Restore ──────────────────────────────────────────────────────────────


def restore_full_backup(
    archive_path: str | Path,
    state_dir: str | Path,
    *,
    passphrase: Optional[bytes] = None,
) -> dict[str, Any]:
    """Restore ColonyMind state from a full backup archive.

    The archive restores into a fresh state directory or into the instance it
    was taken from; another instance's directory is refused (remove its
    ``instance-id`` first if that is really what you want).

    Stop the sidecar and close embedded readers first. Restore holds an
    exclusive state lease; restart afterwards to reopen databases and caches.

    Returns a summary dict of what was restored.
    """
    archive_path = Path(archive_path)
    state_dir = Path(state_dir)
    _private_directory(state_dir)

    with tempfile.TemporaryDirectory(prefix="colonymind-restore-") as tmp, ExitStack() as admission:
        staging = Path(tmp) / "restore"
        _private_directory(staging)

        actual_archive = archive_path
        if archive_path.suffix == ".enc":
            if passphrase is None:
                raise ValueError("Archive is encrypted; passphrase required")
            actual_archive = Path(tmp) / "decrypted.tar.gz"
            _decrypt_file(archive_path, actual_archive, passphrase)

        _extract_archive(actual_archive, staging)

        root = _find_backup_root(staging)
        meta = json.loads((root / "meta.json").read_text())

        if meta.get("backup_version", 0) > BACKUP_VERSION:
            raise ValueError(
                f"Backup version {meta['backup_version']} is newer than "
                f"this ColonyMind supports (max {BACKUP_VERSION})"
            )

        # Validate every referenced original before writing any restored state.
        # A caption or an embedding cannot reconstruct the original evidence.
        source_images = _verified_source_images(root)

        # Reject invalid archives before creating admission metadata. The
        # lease then covers destination identity validation and every write.
        from colonymind.util.state_access import state_access
        admission.enter_context(state_access(state_dir, restore=True))

        existing_id = read_instance_id(state_dir)
        backup_id = _archive_instance_id(meta)

        if existing_id and existing_id != backup_id:
            raise ValueError(
                f"Backup is from instance {backup_id} but this directory is "
                f"instance {existing_id}. Restore into a fresh directory, or "
                f"remove its {INSTANCE_ID_FILE} file first."
            )

        summary: dict[str, Any] = {
            "instance_id": backup_id, "databases": [], "errors": [],
            "serving_admitted": False,
        }

        # An archive taken before a subsystem was retired still carries its state; ``colonymind upgrade``
        # moved that out of the instance, and a restore must not put it back.
        from colonymind.init import RETIRED_STATE
        retired: set[str] = set()
        summary["retired_skipped"] = []

        def keep(relative: Path) -> bool:
            if relative.parts and relative.parts[0] in RETIRED_STATE:
                if relative.parts[0] not in retired:
                    retired.add(relative.parts[0])
                    summary["retired_skipped"].append(relative.parts[0])
                return False
            return True

        identity_dir = root / "identity"
        if identity_dir.is_dir():
            _restore_directory(identity_dir, state_dir, keep=keep, errors=summary["errors"])
            summary["identity"] = True
            _make_secrets_private(identity_dir, state_dir)
        # Only an encrypted archive carries the bearer key; without it ``colonymind init`` writes a new one.
        summary["api_key"] = (identity_dir / "api.key").is_file()
        # Nor the decision log's secret: without it the log makes a new one, and its restored hashes match nothing.
        summary["instinct_key"] = (identity_dir / INSTINCT_KEY_FILE).is_file()
        if backup_id and not read_instance_id(state_dir):
            # Archives taken before the instance id existed carry the same UUID
            # under the chain's name in meta; the restored instance keeps it.
            _private_file(state_dir / INSTANCE_ID_FILE)
            (state_dir / INSTANCE_ID_FILE).write_text(backup_id + "\n")
            (state_dir / INSTANCE_ID_FILE).chmod(0o600)

        db_dir = root / "databases"
        if db_dir.is_dir():
            for db_file in sorted(db_dir.glob("*.db")):
                relative = db_file.relative_to(db_dir)
                if not keep(relative):
                    continue
                dest = state_dir / relative
                # A crashed destination can still have committed WAL.
                # SQLite replaces its logical database consistently;
                # copying the main file alone can replay the old WAL over it.
                with closing(sqlite3.connect(db_file.resolve().as_uri() + '?mode=ro', uri=True)) as source:
                    _private_file(dest)
                    with closing(sqlite3.connect(dest)) as target:
                        source.backup(target)
                relative_name = relative.as_posix()
                summary["databases"].append(relative_name)
                logger.info("Restored database: %s", relative_name)

        if source_images:
            from colonymind.vector.image_store import LocalImageStore
            images = LocalImageStore(str(state_dir), source_evidence=True)
            images._ensure_dirs()
            for record, source in source_images:
                destination = images._original_path(record['asset_hash'], record['mime_type'])
                _copy_private(source, destination)
                destination.chmod(0o600)
            summary['source_images'] = len(source_images)

        config_dir = root / "config"
        if config_dir.is_dir():
            _restore_directory(config_dir, state_dir, keep=keep, errors=summary["errors"])
            summary["config"] = True

        if (root / "plasticity").is_dir():
            _restore_plasticity(root / "plasticity", state_dir / "plasticity")
            summary["plasticity"] = True
            if not (root / "plasticity" / "comparator.key").is_file():
                summary["plasticity_key_missing"] = True

        summary["not_in_archive"] = _archive_omissions(root)
        if backup_id and INSTANCE_ID_FILE in summary["not_in_archive"]:
            summary["not_in_archive"].remove(INSTANCE_ID_FILE)
        summary["not_restored"] = ["host/ (external host and plugin state)"] if (root / "host").exists() else []

        vector_dir = root / "vector"
        if (vector_dir / "lancedb").is_dir():
            dest = state_dir / "lancedb"
            if dest.exists():
                shutil.rmtree(dest)
            _copy_private_tree(vector_dir / "lancedb", dest)
            summary["vectors"] = True
        elif "turn-idempotency.db" in summary["databases"]:
            with closing(sqlite3.connect(state_dir / "turn-idempotency.db")) as conn, conn:
                if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_vector_jobs'").fetchone():
                    summary["vector_jobs_queued"] = conn.execute(
                        "UPDATE source_vector_jobs SET status='pending', cursor=0").rowcount

    logger.info("Restore complete: %s", summary)
    return summary


def _archive_omissions(root: Path) -> list[str]:
    """Name absent state without implying that an archive inventories its source host."""
    from colonymind.secrets.vault import KEY_NAME
    missing = [name for name in (*IDENTITY_FILES, KEY_NAME)
               if not (root / "identity" / name).is_file()]
    missing.extend(name for name in (".env", "channels.json", ".colonymind-llm-config.json")
                   if not (root / "config" / name).is_file())
    if not (root / "plasticity" / "comparator.key").is_file():
        missing.append("comparator.key (plasticity/)")
    if not any((root / "plasticity").rglob("query-vectors.db")):
        missing.append("query-vectors.db (plasticity/ recall query log and journal)")
    for path, label in (("databases", "SQLite databases"), ("plasticity", "plasticity/ (learned weights)"),
                        ("vector/lancedb", "lancedb/ (derived vectors)"),
                        ("host", "host/ (external host and plugin state)")):
        if not (root / path).is_dir():
            missing.append(label)
    config = root / "identity" / "colonymind.yaml"
    if config.is_file():
        import yaml
        def redacted_paths(value, path=""):
            if value == REDACTED:
                missing.append(f"colonymind.yaml:{path} (redacted value)")
            elif isinstance(value, dict):
                for key, child in value.items():
                    redacted_paths(child, f"{path}.{key}" if path else str(key))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    redacted_paths(child, f"{path}[{index}]")
        try:
            redacted_paths(yaml.safe_load(config.read_text()))
        except Exception:
            logger.warning("Could not inspect redacted values in colonymind.yaml")
    env = root / "config" / ".env"
    if env.is_file():
        try:
            for line in env.read_text().splitlines():
                key, sep, value = line.strip().partition("=")
                if sep and value == REDACTED:
                    missing.append(f".env:{key.strip()} (redacted value)")
        except UnicodeDecodeError:
            logger.warning("Could not inspect redacted values in .env")
    return missing


def restore_source_memory(
    archive_path: str | Path,
    destination: str | Path,
    *,
    current_state: str | Path,
    passphrase: Optional[bytes] = None,
) -> dict[str, Any]:
    """Recover current canonical memory, without restoring runtime authority.

    The caller must select a surviving authoritative source ledger. Its current
    membership, revisions, scopes and erasure history replace the old archive's
    memory metadata. The archive supplies only still-owned original image bytes
    missing from that surviving state. No old grants, effects, identity, config,
    contact databases, tasks or native host state are installed.
    """
    archive_path = Path(archive_path).resolve(strict=True)
    current_state = Path(current_state).resolve()
    destination = Path(destination).absolute()
    resolved_destination = destination.resolve()
    if (destination.exists() or destination.is_symlink()
            or current_state == resolved_destination
            or current_state in resolved_destination.parents
            or resolved_destination in current_state.parents):
        raise ValueError("Memory recovery requires a fresh destination outside current state")
    current_id = read_instance_id(current_state)
    ledger = current_state / "turn-idempotency.db"
    if not current_id or not ledger.is_file():
        raise ValueError("Memory recovery requires the surviving instance id and source ledger")

    from colonymind.vector.image_store import LocalImageStore

    _private_directory(destination.parent)
    with tempfile.TemporaryDirectory(prefix="colonymind-memory-recovery-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        actual_archive = archive_path
        if archive_path.suffix == ".enc":
            if passphrase is None:
                raise ValueError("Archive is encrypted; passphrase required")
            actual_archive = staging / "decrypted.tar.gz"
            _decrypt_file(archive_path, actual_archive, passphrase)
        unpacked = staging / "archive"
        _private_directory(unpacked)
        _extract_archive(actual_archive, unpacked)
        archived = _find_backup_root(unpacked)
        meta = json.loads((archived / "meta.json").read_text())
        if meta.get("backup_version", 0) > BACKUP_VERSION:
            raise ValueError("Memory archive version is newer than this ColonyMind supports")
        if _archive_instance_id(meta) != current_id:
            raise ValueError("Memory archive and surviving state have different instance ids")

        selected = staging / "selected"
        _private_directory(selected / "databases")
        _private_file(selected / "databases" / "turn-idempotency.db")
        snapshot = selected / "databases" / "turn-idempotency.db"
        try:
            with closing(sqlite3.connect(ledger.resolve().as_uri() + "?mode=ro", uri=True)) as source:
                with closing(sqlite3.connect(snapshot)) as target:
                    source.backup(target)
                    if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise ValueError("Surviving source ledger failed its integrity check")
            snapshot.chmod(0o600)
            with closing(sqlite3.connect(snapshot.as_uri() + "?mode=ro", uri=True)) as current:
                source_count = current.execute("SELECT count(*) FROM turn_sources").fetchone()[0]
                heads = current.execute(
                    "SELECT contact_id,max(sequence) FROM source_erasures GROUP BY contact_id"
                ).fetchall()
                _check_erasure_ancestry(archived / "databases" / "turn-idempotency.db", current)
        except sqlite3.Error as error:
            raise ValueError("Surviving canonical source history is unavailable") from error
        if read_instance_id(current_state) != current_id:
            raise ValueError("Surviving instance id changed during memory recovery")

        bundle = staging / "bundle"
        bundle.mkdir(mode=0o700)
        _copy_private(snapshot, bundle / "turn-idempotency.db")
        (bundle / "turn-idempotency.db").chmod(0o600)
        output_images = LocalImageStore(str(bundle), source_evidence=True)
        live_images = LocalImageStore(str(current_state), source_evidence=True)
        old_images = LocalImageStore(str(archived), source_evidence=True)
        recovered_from_archive = 0
        records = _source_image_records(selected)
        if records:
            output_images._ensure_dirs()
        for record in records:
            chosen = None
            for source_images, from_archive in ((live_images, False), (old_images, True)):
                candidate = source_images._original_path(record["asset_hash"], record["mime_type"])
                try:
                    data = candidate.read_bytes()
                except FileNotFoundError:
                    continue
                if (len(data) != record["size_bytes"]
                        or hashlib.sha256(data).hexdigest() != record["asset_hash"]):
                    continue
                chosen = data
                recovered_from_archive += int(from_archive)
                break
            if chosen is None:
                raise ValueError("Current source image has no matching original in surviving state or archive")
            target = output_images._original_path(record["asset_hash"], record["mime_type"])
            _private_file(target)
            target.write_bytes(chosen)
            target.chmod(0o600)

        files = []
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                files.append({"path": path.relative_to(bundle).as_posix(),
                              "sha256": _file_sha256(path), "bytes": path.stat().st_size})
        summary = {
            "recovery_mode": "source_memory_only",
            "instance_id": current_id,
            "source_count": source_count,
            "source_images": len(records),
            "images_recovered_from_archive": recovered_from_archive,
            "erasure_head": max((row[1] for row in heads), default=0),
            "erasure_contacts": len(heads),
            "current_source_ledger_sha256": _file_sha256(snapshot),
            "archive_sha256": _file_sha256(archive_path),
            "files": files,
            "runtime_authority_restored": False,
            "serving_admitted": False,
        }
        receipt = bundle / "source-memory-recovery.json"
        _private_file(receipt)
        receipt.write_text(json.dumps(summary, indent=2) + "\n")
        receipt.chmod(0o600)
        # Reserve the destination only after all source and image checks pass.
        # A publication failure leaves an incomplete bundle, never a success.
        destination.mkdir(mode=0o700)
        _restore_directory(bundle, destination)
    return summary


def _file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _check_erasure_ancestry(archived_ledger: Path, current: sqlite3.Connection) -> None:
    """A selected surviving history cannot rewind or replace captured erasures."""
    if not archived_ledger.is_file():
        return
    with closing(sqlite3.connect(archived_ledger.resolve().as_uri() + "?mode=ro", uri=True)) as old:
        for table in ("source_erasures", "source_erasure_revisions"):
            if not old.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                continue
            columns = [row[1] for row in old.execute(f"PRAGMA table_info({table})")]
            selected = ",".join('"' + column.replace('"', '""') + '"' for column in columns)
            for row in old.execute(f"SELECT {selected} FROM {table}"):
                sequence = row[columns.index("sequence")]
                current_row = current.execute(
                    f"SELECT {selected} FROM {table} WHERE sequence=?", (sequence,)
                ).fetchone()
                if current_row != row:
                    raise ValueError("Surviving source erasure history does not extend the archive")
        # Source withdrawals and their materialized copy fence are append-only
        # privacy history too. A surviving ledger cannot rewind either index.
        for table, keys in (('curated_memory_events', ('sequence',)),
                            ('curated_memory_fence_hashes', ('event_sequence', 'person_id', 'message_hash'))):
            if not old.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
                continue
            columns = [row[1] for row in old.execute(f'PRAGMA table_info({table})')]
            selected = ','.join('"' + column + '"' for column in columns)
            query = f'SELECT {selected} FROM {table}'
            if table == 'curated_memory_events':
                query += " WHERE operation='forget' AND memory_id LIKE 'sources:%'"
            rows = old.execute(query).fetchall()
            if not rows:
                continue
            if not current.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
                raise ValueError('Surviving source erasure history does not extend the archive')
            predicate = ' AND '.join(key + '=?' for key in keys)
            for row in rows:
                values = [row[columns.index(key)] for key in keys]
                if current.execute(f'SELECT {selected} FROM {table} WHERE {predicate}', values).fetchone() != row:
                    raise ValueError('Surviving source erasure history does not extend the archive')


# ── Database snapshot ────────────────────────────────────────────────────


def _snapshot_databases(
    state_dir: Path, dest: Path,
) -> list[dict[str, str]]:
    """Snapshot the top-level databases."""
    _private_directory(dest)
    manifest: list[dict[str, str]] = []

    for db_path in sorted(state_dir.glob("*.db")):
        relative = db_path.relative_to(state_dir)
        relative_name = relative.as_posix()
        snap_path = dest / relative
        _private_file(snap_path)
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                escaped_snap_path = str(snap_path).replace("'", "''")
                conn.execute(f"VACUUM INTO '{escaped_snap_path}'")
            finally:
                conn.close()
            manifest.append({
                "filename": relative_name,
                "size_bytes": str(snap_path.stat().st_size),
            })
            logger.info("Snapshotted database: %s", relative_name)
        except Exception:
            logger.warning("VACUUM snapshot failed for %s; trying SQLite backup", relative_name)
            try:
                # Copying only the main file can lose committed WAL records.
                snap_path.unlink(missing_ok=True)
                _private_file(snap_path)
                with closing(sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True)) as source:
                    with closing(sqlite3.connect(snap_path)) as target:
                        source.backup(target)
                manifest.append({
                    "filename": relative_name,
                    "size_bytes": str(snap_path.stat().st_size),
                    "method": "sqlite_backup",
                })
            except Exception as exc2:
                raise RuntimeError(f"Database could not be snapshotted consistently: {relative_name}") from exc2

    return manifest


def _source_image_records(root: Path) -> list[dict]:
    """Read asset ownership from the captured database, never a later live view."""
    ledger = root / 'databases' / 'turn-idempotency.db'
    if not ledger.is_file():
        return []
    conn = sqlite3.connect(ledger.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        conn.row_factory = sqlite3.Row
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='source_media'").fetchone():
            return []
        rows = conn.execute('''SELECT DISTINCT m.asset_hash,m.mime_type,m.size_bytes
            FROM source_media m JOIN source_media_links l ON l.asset_hash=m.asset_hash
            JOIN turn_sources s ON s.turn_id=l.turn_id WHERE m.status != 'orphan' ''').fetchall()
        records = [dict(row) for row in rows]
        for row in records:
            if not re.fullmatch('[0-9a-f]{64}', row['asset_hash']):
                raise ValueError('Invalid original source image identifier')
        return records
    finally:
        conn.close()


def _verified_source_images(root: Path) -> list[tuple[dict, Path]]:
    from colonymind.vector.image_store import LocalImageStore
    images = LocalImageStore(str(root), source_evidence=True)
    result = []
    for record in _source_image_records(root):
        path = images._original_path(record['asset_hash'], record['mime_type'])
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ValueError('Backup is missing an original source image') from exc
        if len(data) != record['size_bytes'] or hashlib.sha256(data).hexdigest() != record['asset_hash']:
            raise ValueError('Backup original source image does not match its evidence hash')
        result.append((record, path))
    return result


def _snapshot_source_images(state_dir: Path, staging: Path) -> int:
    from colonymind.vector.image_store import LocalImageStore
    source = LocalImageStore(str(state_dir), source_evidence=True)
    target = LocalImageStore(str(staging), source_evidence=True)
    records = _source_image_records(staging)
    if records:
        target._ensure_dirs()
    for record in records:
        try:
            _copy_private(source._original_path(record['asset_hash'], record['mime_type']),
                            target._original_path(record['asset_hash'], record['mime_type']))
        except OSError as exc:
            # Concurrent erasure can remove an original after the DB snapshot.
            # Retry the backup later instead of publishing a broken archive.
            raise RuntimeError('Original source image changed during backup; retry') from exc
    return len(_verified_source_images(staging))


# ── Identity snapshot ────────────────────────────────────────────────────


# The instance's own secret (``colonymind.instance.SECRET_FILE``): the decision log's state hashes and hashed
# references and the model usage ledger's aliases are keyed with it, so a log restored without it keeps rows whose
# hashes match nothing (and the usage rows' aliases differ from the ones recorded after it).
INSTINCT_KEY_FILE = "instinct.key"
IDENTITY_FILES = (INSTANCE_ID_FILE, "identity.yaml", "colonymind.yaml", "api.key", INSTINCT_KEY_FILE)
SECRET_FILES = ("api.key", INSTINCT_KEY_FILE)
REDACTED = "<REDACTED>"


def _vault_key_files(secrets: bool) -> tuple[str, ...]:
    """The sealed store's key file, only in an encrypted archive and only when ``secrets.vault.key_in_backups``
    is on: by default a backup carries the store's ciphertext (``secrets-vault.db``) and never its key."""
    if not secrets:
        return ()
    try:
        from colonymind.secrets.config import current_settings
        from colonymind.secrets.vault import KEY_NAME
        return (KEY_NAME,) if current_settings().key_in_backups else ()
    except Exception:
        return ()


def _snapshot_identity(state_dir: Path, dest: Path, *, secrets: bool) -> None:
    """The instance id and the owner-authored identity and configuration.

    Secrets travel only in an encrypted archive (``secrets``): otherwise the sidecar's bearer key
    (``api.key``, which ``colonymind init`` writes anew) and the decision log's secret (``instinct.key``, which
    the log makes anew) stay behind, and ``colonymind.yaml`` is copied with its credential-named
    ``environment`` settings redacted, the rule the ``.env`` copy follows."""
    _private_directory(dest)
    for name in IDENTITY_FILES + _vault_key_files(secrets):
        src = state_dir / name
        if not src.is_file() or (name in SECRET_FILES and not secrets):
            continue
        if name == "colonymind.yaml" and not secrets:
            scrubbed = _scrub_config_file(src)
            if scrubbed is not None:
                _private_file(dest / name)
                (dest / name).write_text(scrubbed)
            continue
        _copy_private(src, dest / name)


def _make_secrets_private(identity_dir: Path, state_dir: Path) -> None:
    """Each secret a restore put back (``SECRET_FILES`` and the sealed store's key) readable by its owner alone
    (0600), whatever mode the file had when it was archived."""
    from colonymind.secrets.vault import KEY_NAME
    for name in (*SECRET_FILES, KEY_NAME):
        if (identity_dir / name).is_file() and (state_dir / name).is_file():
            os.chmod(state_dir / name, 0o600)


def _scrub_config_file(path: Path) -> Optional[str]:
    """``colonymind.yaml`` with the values of credential-named ``environment`` keys redacted; None (left
    out of the archive) when it does not parse, since then nothing can be redacted reliably."""
    import yaml
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        logger.warning("colonymind.yaml not backed up: it does not parse, so it cannot be scrubbed")
        return None
    if not isinstance(data, dict):
        return None
    environment = data.get("environment")
    if isinstance(environment, dict):
        data["environment"] = {key: REDACTED if _SECRET_KEY_PATTERN.search(str(key)) else value
                               for key, value in environment.items()}
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)


# ── Config snapshot (with secret scrubbing) ──────────────────────────────


def _snapshot_config(state_dir: Path, dest: Path) -> None:
    _private_directory(dest)
    config_files = [
        ".env", "channels.json", ".colonymind-llm-config.json",
    ]

    for name in config_files:
        src = state_dir / name
        if not src.exists():
            src = state_dir.parent / name
        if src.exists():
            if name == ".env":
                scrubbed = _scrub_env_file(src.read_text())
                _private_file(dest / name)
                (dest / name).write_text(scrubbed)
            else:
                _copy_private(src, dest / name)


def _scrub_env_file(content: str) -> str:
    lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            lines.append(line)
            continue
        if "=" in stripped:
            key, _, _ = stripped.partition("=")
            if _SECRET_KEY_PATTERN.search(key.strip()):
                lines.append(f"{key.strip()}=<REDACTED>")
                continue
        lines.append(line)
    return "\n".join(lines) + "\n"


# ── Vector store snapshot ────────────────────────────────────────────────


def _snapshot_vectors(state_dir: Path, dest: Path) -> None:
    lance_dir = state_dir / "lancedb"
    if not lance_dir.is_dir():
        return
    dest_lance = dest / "lancedb"
    _copy_private_tree(lance_dir, dest_lance)
    logger.info("Snapshotted LanceDB directory")


# ── Host state ───────────────────────────────────────────────────────────


def _snapshot_host_state(paths: list[str], dest: Path) -> None:
    _private_directory(dest)
    for p in paths:
        src = Path(p).expanduser()
        if src.is_file():
            _copy_private(src, dest / src.name)
        elif src.is_dir():
            _copy_private_tree(src, dest / src.name)


# ── Archive creation / extraction ────────────────────────────────────────


def _create_archive(source_dir: Path, archive_path: Path) -> None:
    _private_file(archive_path)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_dir, arcname=source_dir.name)


def _extract_archive(archive_path: Path, dest: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest, filter="data")


def _find_backup_root(staging: Path) -> Path:
    """Find the root directory inside the extracted archive."""
    entries = list(staging.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return staging


# ── Encryption (AES-256-GCM) ────────────────────────────────────────────


def _encrypt_file(
    src: Path, dest: Path, passphrase: bytes,
) -> None:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    salt = os.urandom(16)
    kdf = Scrypt(salt=salt, length=32, n=2**17, r=8, p=1)
    key = kdf.derive(passphrase)

    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    plaintext = src.read_bytes()
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)

    _private_file(dest)
    with dest.open("wb") as f:
        f.write(b"COLONYMIND_ENC_V1\n")
        f.write(salt)
        f.write(nonce)
        f.write(ciphertext)


def _decrypt_file(
    src: Path, dest: Path, passphrase: bytes,
) -> None:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    data = src.read_bytes()
    header = b"COLONYMIND_ENC_V1\n"
    if not data.startswith(header):
        raise ValueError("Not a ColonyMind encrypted backup (invalid header)")

    offset = len(header)
    salt = data[offset:offset + 16]
    nonce = data[offset + 16:offset + 28]
    ciphertext = data[offset + 28:]

    kdf = Scrypt(salt=salt, length=32, n=2**17, r=8, p=1)
    key = kdf.derive(passphrase)

    aesgcm = AESGCM(key)
    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    except Exception:
        raise ValueError("Decryption failed -- wrong passphrase?")

    _private_file(dest)
    dest.write_bytes(plaintext)


# ── Helpers ──────────────────────────────────────────────────────────────


def _archive_instance_id(meta: dict) -> str:
    """The instance an archive was taken from (older archives named it colonymind_id)."""
    return str(meta.get("instance_id") or meta.get("colonymind_id") or "")


def _compute_identity_hmac(instance: str) -> str:
    return hmac.new(
        instance.encode(), b"colonymind-backup-binding", hashlib.sha256
    ).hexdigest()


def _get_colonymind_version() -> str:
    try:
        from importlib.metadata import version
        return version("colonymind")
    except Exception:
        return "unknown"


def _write_private_atomic(target: Path, content: str | bytes) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content.encode("utf-8") if isinstance(content, str) else content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _restore_directory(src: Path, dest: Path, *, keep=lambda relative: True, errors=None) -> None:
    for item in src.rglob("*"):
        if item.is_file():
            rel = item.relative_to(src)
            if not keep(rel):
                continue
            target = dest / rel
            _private_directory(target.parent)
            if rel.as_posix() in ("colonymind.yaml", ".env"):
                content = _restore_config_values(item, target, errors=errors)
                if content is not None:
                    _write_private_atomic(target, content)
            else:
                _copy_private(item, target)


def _restore_config_values(archived: Path, live: Path, *, errors=None) -> str | bytes | None:
    """A redaction is not a replacement value for an installed configuration."""
    raw = archived.read_bytes()
    merge_live = REDACTED.encode() in raw and live.is_file()

    def not_applied():
        message = f"{archived.name} not applied: could not fill archived placeholders from the live file."
        logger.warning(message)
        if errors is not None:
            errors.append(message)

    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        if merge_live:
            return not_applied()
        message = f"Could not decode {archived.name} as UTF-8; archived bytes kept."
        logger.warning(message)
        if errors is not None:
            errors.append(message)
        return raw
    live_content = None
    if merge_live:
        try:
            live_content = live.read_text()
        except (OSError, UnicodeDecodeError):
            return not_applied()
    if archived.name == ".env":
        if not merge_live:
            return content
        current = {}
        for line in live_content.splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and not key.startswith("#"):
                current[key.strip()] = value
        lines = []
        for line in content.splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and value == REDACTED and key.strip() in current:
                line = f"{key.strip()}={current[key.strip()]}"
            lines.append(line)
        return "\n".join(lines) + "\n"
    import yaml
    try:
        saved = yaml.safe_load(content)
        current = yaml.safe_load(live_content) if merge_live else None
    except Exception:
        if merge_live:
            return not_applied()
        message = "Could not parse colonymind.yaml; archived bytes kept."
        logger.warning(message)
        if errors is not None:
            errors.append(message)
        return content
    if not merge_live:
        return content

    def merge(value, current):
        if value == REDACTED:
            return current if current is not None else value
        if isinstance(value, dict):
            current = current if isinstance(current, dict) else {}
            return {key: merge(child, current.get(key)) for key, child in value.items()}
        if isinstance(value, list):
            current = current if isinstance(current, list) else []
            merged = []
            for child in value:
                matches = [item for item in current if isinstance(item, dict) and isinstance(child, dict)
                           and child.get("name") is not None and item.get("name") == child["name"]]
                merged.append(merge(child, matches[0] if len(matches) == 1 else None))
            return merged
        return value
    try:
        return yaml.safe_dump(merge(saved, current),
                              sort_keys=False, allow_unicode=True, default_flow_style=False)
    except Exception:
        return not_applied()
