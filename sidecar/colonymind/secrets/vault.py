"""The sealed store: secrets that arrived in conversation, encrypted, referenced from memory.

``secrets-vault.db`` in the state directory (0600) holds one row per reference: its scope (the person whose
message carried the value, and the session), the rule that found it, its label, the turn it came from, and the
value under AES-256-GCM (a fresh nonce per row; the reference and the scope are the associated data). The key is
32 random bytes in ``secrets-vault.key`` (0600, the state directory) or in the OS keyring
(``secrets.vault.key_store: keyring``). The nightly backup copies ``*.db`` files, so a backup carries ciphertext
and never the key; a restore without the key keeps every reference and reports its value unavailable.

Nothing here logs or returns a value except ``resolve`` and ``release``, whose callers are the owner-only paths
(docs/SECRETS.md).
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets as random_source
import sqlite3
import threading
import time
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from colonymind.util.temporal import ledger_stamp

from .refs import FORGOTTEN_LABEL, reference_id

logger = logging.getLogger(__name__)

DB_NAME = "secrets-vault.db"
KEY_NAME = "secrets-vault.key"
KEYRING_SERVICE = "colonymind-secrets-vault"
KEY_STORES = ("file", "keyring")


class VaultUnavailable(RuntimeError):
    """The key is missing, unreadable or not the one a value was sealed under."""


class UnknownReference(LookupError):
    """No sealed value has this reference."""


class SecretForgotten(LookupError):
    """The owner deleted this value; the reference stays, the value is gone."""


def _private(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


@contextmanager
def _exclusive(path: Path) -> Iterator[None]:
    """Held by one process (and one thread) at a time: an advisory lock on ``path`` (made 0600 if new). Where the
    platform has no ``fcntl`` the in-process lock of the caller is the only one."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


class Vault:
    """One instance's sealed store (``get_vault`` returns the shared one for a state directory)."""

    def __init__(self, state_dir: str | os.PathLike[str], *, key_store: str = "file") -> None:
        if key_store not in KEY_STORES:
            raise ValueError(f"secrets.vault.key_store must be one of {', '.join(KEY_STORES)}")
        self.state_dir = Path(state_dir)
        self.db_path = self.state_dir / DB_NAME
        self.key_path = self.state_dir / KEY_NAME
        self.key_store = key_store
        self._key: Optional[bytes] = None
        self._lock = threading.Lock()
        self._ready = False

    # -- storage -------------------------------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            try:  # born private: never a moment readable by others (SQLite gives its journals the same mode)
                os.close(os.open(str(self.db_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
            except FileExistsError:
                pass
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        if not self._ready:
            _private(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA secure_delete=ON")
        if not self._ready:
            with self._lock:
                if not self._ready:
                    for attempt in range(100):   # the mode change takes the lock another opener may hold
                        try:
                            conn.execute("PRAGMA journal_mode=WAL")
                            break
                        except sqlite3.OperationalError as error:
                            if "locked" not in str(error) or attempt == 99:
                                raise
                            time.sleep(0.02)
                    conn.execute("""CREATE TABLE IF NOT EXISTS sealed (
                        ref TEXT PRIMARY KEY, scope_contact TEXT NOT NULL, scope_session TEXT NOT NULL DEFAULT '',
                        kind TEXT NOT NULL, label TEXT NOT NULL, source_turn TEXT NOT NULL DEFAULT '',
                        key_id TEXT NOT NULL, nonce BLOB, ciphertext BLOB,
                        created_at TEXT NOT NULL, last_used_at TEXT, uses INTEGER NOT NULL DEFAULT 0,
                        released_at TEXT, forgotten_at TEXT, last_seen_at TEXT)""")
                    if "last_seen_at" not in {row[1] for row in conn.execute("PRAGMA table_info(sealed)")}:
                        conn.execute("ALTER TABLE sealed ADD COLUMN last_seen_at TEXT")
                    conn.execute("CREATE INDEX IF NOT EXISTS sealed_scope ON sealed(scope_contact, created_at)")
                    conn.execute("""CREATE TABLE IF NOT EXISTS sealed_uses (
                        ref TEXT NOT NULL, at TEXT NOT NULL, purpose TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '')""")
                    conn.execute("CREATE TABLE IF NOT EXISTS vault_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                    conn.commit()
                    self._ready = True
        return conn

    # -- the key ------------------------------------------------------------------------------------------------

    def _load_key(self, *, create: bool) -> bytes:
        if self._key is not None:
            return self._key
        with self._lock:
            if self._key is not None:
                return self._key
            key = self._read_keyring(create=create) if self.key_store == "keyring" else self._read_file(create=create)
            if len(key) != 32:
                raise VaultUnavailable("the vault key is not 32 bytes")
            self._key = key
            return key

    def _read_file(self, *, create: bool) -> bytes:
        if not self.key_path.exists():
            if not create:
                raise VaultUnavailable("no vault key")
            self.state_dir.mkdir(parents=True, exist_ok=True)
            # Written whole under a private name, then linked into place: the key appears complete or not at
            # all, and when two writers race the first link wins and both read that one.
            staged = self.state_dir / f".{KEY_NAME}.{os.getpid()}.{random_source.token_hex(6)}"
            fd = os.open(str(staged), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(fd, "w", encoding="ascii") as handle:
                    handle.write(base64.b64encode(random_source.token_bytes(32)).decode("ascii") + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.link(staged, self.key_path)
                except FileExistsError:
                    pass  # another writer made it first: read theirs
            finally:
                staged.unlink(missing_ok=True)
        try:
            mode = self.key_path.stat().st_mode & 0o777
            if mode & 0o077:
                raise VaultUnavailable("the vault key file is readable by others; make it 0600")
            return base64.b64decode(self.key_path.read_text(encoding="ascii").strip(), validate=True)
        except VaultUnavailable:
            raise
        except (OSError, ValueError) as error:
            raise VaultUnavailable(f"the vault key cannot be read ({type(error).__name__})") from None

    def _read_keyring(self, *, create: bool) -> bytes:
        try:
            import keyring  # optional: only this key store needs it
        except ImportError:
            raise VaultUnavailable("secrets.vault.key_store is keyring but the keyring package is not "
                                   "installed") from None
        account = str(self.state_dir.resolve())
        try:
            stored = keyring.get_password(KEYRING_SERVICE, account)
            if not stored:
                if not create:
                    raise VaultUnavailable("no vault key in the keyring")
                # The first key is made by one process at a time, which reads the keyring again under the lock:
                # two first users never each keep a key of their own (the loser's values would not decrypt).
                with _exclusive(self.state_dir / f".{KEY_NAME}.lock"):
                    stored = keyring.get_password(KEYRING_SERVICE, account)
                    if not stored:
                        keyring.set_password(KEYRING_SERVICE, account,
                                             base64.b64encode(random_source.token_bytes(32)).decode("ascii"))
                        stored = keyring.get_password(KEYRING_SERVICE, account)
                if not stored:
                    raise VaultUnavailable("the keyring did not keep the vault key")
            return base64.b64decode(stored, validate=True)
        except VaultUnavailable:
            raise
        except Exception as error:
            raise VaultUnavailable(f"the keyring cannot be read ({type(error).__name__})") from None

    @staticmethod
    def _key_id(key: bytes) -> str:
        return hashlib.sha256(b"colonymind-vault-key-id\x00" + key).hexdigest()[:16]

    @property
    def available(self) -> bool:
        """Whether values can be sealed and resolved now (a key is present or can be made)."""
        try:
            self._load_key(create=True)
            return True
        except VaultUnavailable:
            return False

    def _aead(self, key: bytes):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        return AESGCM(key)

    @staticmethod
    def _aad(ref: str, contact_id: str) -> bytes:
        return f"{ref}\x00{contact_id}".encode("utf-8")

    # -- sealing ----------------------------------------------------------------------------------------------

    def _slot(self, conn: sqlite3.Connection, key: bytes, scope: str, value: str) -> Tuple[str, Optional[sqlite3.Row]]:
        """The reference ``value`` has in ``scope`` and its row (None: a free id, the value is new). A row of
        another scope or key, a forgotten one (its reference stays forgotten: the value, when it comes again,
        takes the next id) and an id collision are passed over. Raises ``VaultUnavailable``."""
        key_id, aead = self._key_id(key), self._aead(key)
        for salt in range(8):
            ref = reference_id(key, scope, value, salt=salt)
            row = conn.execute("SELECT * FROM sealed WHERE ref=?", (ref,)).fetchone()
            if row is None:
                return ref, None
            if row["scope_contact"] != scope or row["key_id"] != key_id or row["forgotten_at"] is not None:
                continue
            if self._decrypt(aead, row) == value:
                return ref, row
        raise VaultUnavailable("no free reference for this value")

    def plan(self, value: str, *, contact_id: str) -> Tuple[str, str]:
        """``(reference, state)`` for ``value`` in ``contact_id``'s scope, storing nothing: ``new`` (``seal``
        would store it under that reference), ``sealed`` or ``released``. What a source's sealing uses before the
        source is admitted (``seal.Staged``). Raises ``VaultUnavailable``."""
        if not isinstance(value, str) or not value:
            raise ValueError("nothing to seal")
        key = self._load_key(create=True)
        with closing(self._connect()) as conn:
            ref, row = self._slot(conn, key, str(contact_id or ""), value)
        return ref, "new" if row is None else "released" if row["released_at"] is not None else "sealed"

    def seal(self, value: str, *, contact_id: str, session_id: str = "", kind: str, label: str,
             source_turn: str = "", expect: Optional[str] = None) -> Optional[str]:
        """The reference for ``value`` in ``contact_id``'s scope, storing it if new, and the time it was last seen.
        None when the owner released this very value (a false positive: it stays in memory as it came). A value
        the owner forgot that arrives again takes a new reference; the forgotten one stays forgotten. ``expect``:
        the reference a plan gave it (``Staged``); when the store now gives another (the owner forgot it since),
        nothing is written. Raises ``VaultUnavailable``."""
        if not isinstance(value, str) or not value:
            raise ValueError("nothing to seal")
        scope = str(contact_id or "")
        key = self._load_key(create=True)
        key_id = self._key_id(key)
        aead = self._aead(key)
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")   # two writers sealing one new value never both insert it
            ref, row = self._slot(conn, key, scope, value)
            if expect is not None and ref != expect:
                raise VaultUnavailable("the sealed store changed while this source was sealed; send it again")
            now = ledger_stamp()
            if row is None:
                nonce = random_source.token_bytes(12)
                conn.execute(
                    "INSERT INTO sealed (ref, scope_contact, scope_session, kind, label, source_turn, key_id, "
                    "nonce, ciphertext, created_at, last_seen_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (ref, scope, str(session_id or ""), kind, label, str(source_turn or ""), key_id, nonce,
                     aead.encrypt(nonce, value.encode("utf-8"), self._aad(ref, scope)), now, now))
                return ref
            conn.execute("UPDATE sealed SET last_seen_at=? WHERE ref=?", (now, ref))
            return None if row["released_at"] is not None else ref

    def find(self, value: str, *, contact_id: str, released: bool = False) -> Optional[str]:
        """The reference ``value`` already has in ``contact_id``'s scope, or None; nothing is stored (an erasure
        by exact text looks for the sealed copy without sealing anything new). ``released`` counts a released
        value too: the form a message had before the owner released it (an erasure's identity)."""
        if not isinstance(value, str) or not value or not self.db_path.exists():
            return None
        try:
            key = self._load_key(create=False)
        except VaultUnavailable:
            return None
        scope, aead = str(contact_id or ""), self._aead(key)
        with closing(self._connect()) as conn:
            for salt in range(8):
                ref = reference_id(key, scope, value, salt=salt)
                row = conn.execute("SELECT * FROM sealed WHERE ref=?", (ref,)).fetchone()
                if row is None:
                    return None
                if (row["scope_contact"] == scope and row["key_id"] == self._key_id(key)
                        and row["forgotten_at"] is None and (released or row["released_at"] is None)):
                    try:
                        if self._decrypt(aead, row) == value:
                            return ref
                    except VaultUnavailable:
                        return None
        return None

    def identities(self, value: str, *, contact_id: str) -> List[str]:
        """Every reference ``value`` has had in ``contact_id``'s scope, in the order they were given: the live
        one, a released one, and one the owner forgot (worked out from the value itself under the key, since a
        forgotten row keeps no copy of it). What an erasure's identity check compares; nothing is stored or
        returned but references. Raises ``VaultUnavailable`` when the key cannot be read."""
        if not isinstance(value, str) or not value or not self.db_path.exists():
            return []
        key = self._load_key(create=False)
        scope, key_id, aead = str(contact_id or ""), self._key_id(key), self._aead(key)
        found: List[str] = []
        with closing(self._connect()) as conn:
            for salt in range(8):
                ref = reference_id(key, scope, value, salt=salt)
                row = conn.execute("SELECT * FROM sealed WHERE ref=?", (ref,)).fetchone()
                if row is None:
                    break
                if row["scope_contact"] != scope or row["key_id"] != key_id:
                    continue
                if row["forgotten_at"] is not None:
                    found.append(ref)
                    continue
                try:
                    if self._decrypt(aead, row) == value:
                        found.append(ref)
                except VaultUnavailable:
                    continue
        return found

    def _decrypt(self, aead, row: sqlite3.Row) -> str:
        from cryptography.exceptions import InvalidTag
        try:
            return aead.decrypt(bytes(row["nonce"]), bytes(row["ciphertext"]),
                                self._aad(row["ref"], row["scope_contact"])).decode("utf-8")
        except (InvalidTag, TypeError, ValueError):
            raise VaultUnavailable("the value cannot be decrypted with this key") from None

    # -- reading --------------------------------------------------------------------------------------------

    @staticmethod
    def _entry(row: sqlite3.Row, key_id: Optional[str]) -> Dict[str, Any]:
        state = ("forgotten" if row["forgotten_at"] else "released" if row["released_at"]
                 else "unavailable" if row["key_id"] != key_id else "sealed")
        return {"ref": row["ref"], "scope_contact": row["scope_contact"], "scope_session": row["scope_session"],
                "kind": row["kind"], "label": FORGOTTEN_LABEL if row["forgotten_at"] else row["label"],
                "source_turn": row["source_turn"], "created_at": row["created_at"],
                "last_used_at": row["last_used_at"], "last_seen_at": row["last_seen_at"],
                "uses": int(row["uses"] or 0), "state": state}

    def _current_key_id(self) -> Optional[str]:
        try:
            return self._key_id(self._load_key(create=False))
        except VaultUnavailable:
            return None

    def entry(self, ref: str) -> Optional[Dict[str, Any]]:
        """A reference's metadata (never its value), or None."""
        if not self.db_path.exists():
            return None
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM sealed WHERE ref=?", (str(ref),)).fetchone()
        return None if row is None else self._entry(row, self._current_key_id())

    def entries(self, *, contact_id: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
        if not self.db_path.exists():
            return []
        sql, args = "SELECT * FROM sealed", []
        if contact_id is not None:
            sql += " WHERE scope_contact=?"
            args.append(contact_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(max(1, int(limit)))
        key_id = self._current_key_id()
        with closing(self._connect()) as conn:
            return [self._entry(row, key_id) for row in conn.execute(sql, args).fetchall()]

    def resolve(self, ref: str) -> str:
        """The value behind ``ref``. Raises ``UnknownReference``, ``SecretForgotten`` or ``VaultUnavailable``.
        Scope and purpose are the caller's to check (``api/routers/secrets.py``)."""
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM sealed WHERE ref=?", (str(ref),)).fetchone()
        if row is None:
            raise UnknownReference(str(ref))
        if row["forgotten_at"] is not None:
            raise SecretForgotten(str(ref))
        key = self._load_key(create=False)
        if row["key_id"] != self._key_id(key):
            raise VaultUnavailable("this value was sealed under another key")
        return self._decrypt(self._aead(key), row)

    def note_use(self, ref: str, purpose: str, detail: str = "") -> None:
        with closing(self._connect()) as conn, conn:
            now = ledger_stamp()
            conn.execute("UPDATE sealed SET uses=uses+1, last_used_at=? WHERE ref=?", (now, str(ref)))
            conn.execute("INSERT INTO sealed_uses (ref, at, purpose, detail) VALUES (?,?,?,?)",
                         (str(ref), now, str(purpose)[:40], str(detail)[:200]))

    def uses(self, ref: str, *, limit: int = 50) -> List[Dict[str, Any]]:
        with closing(self._connect()) as conn:
            return [dict(row) for row in conn.execute(
                "SELECT at, purpose, detail FROM sealed_uses WHERE ref=? ORDER BY at DESC LIMIT ?",
                (str(ref), limit)).fetchall()]

    # -- the owner's choices ------------------------------------------------------------------------------------

    def release(self, ref: str) -> str:
        """A false positive: the value is returned (to go back where its reference stood) and never sealed again
        in this scope."""
        value = self.resolve(ref)
        with closing(self._connect()) as conn, conn:
            conn.execute("UPDATE sealed SET released_at=? WHERE ref=?", (ledger_stamp(), str(ref)))
        return value

    def forget(self, ref: str) -> bool:
        """The value is deleted; the reference stays and reads as a value the owner deleted."""
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute("UPDATE sealed SET nonce=NULL, ciphertext=NULL, forgotten_at=? "
                                  "WHERE ref=? AND forgotten_at IS NULL", (ledger_stamp(), str(ref)))
            return cursor.rowcount > 0

    # -- markers ------------------------------------------------------------------------------------------------

    def marker(self, name: str) -> Optional[str]:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT value FROM vault_meta WHERE key=?", (name,)).fetchone()
            return None if row is None else row["value"]

    def set_marker(self, name: str, value: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("INSERT INTO vault_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET "
                         "value=excluded.value", (name, str(value)))

    def counts(self) -> Dict[str, int]:
        if not self.db_path.exists():
            return {"sealed": 0, "released": 0, "forgotten": 0}
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT count(*) AS total, sum(released_at IS NOT NULL) AS released, "
                               "sum(forgotten_at IS NOT NULL) AS forgotten FROM sealed").fetchone()
        return {"sealed": int(row["total"] or 0), "released": int(row["released"] or 0),
                "forgotten": int(row["forgotten"] or 0)}


_VAULTS: Dict[tuple, Vault] = {}
_VAULTS_LOCK = threading.Lock()


def get_vault(state_dir: str | os.PathLike[str] | None = None, *, key_store: Optional[str] = None) -> Vault:
    """The shared vault of a state directory (the running instance's by default)."""
    if state_dir is None:
        from colonymind import get_state_dir
        state_dir = get_state_dir()
    if key_store is None:
        from .config import current_settings
        key_store = current_settings().key_store
    key = (str(Path(state_dir).resolve()), key_store)
    with _VAULTS_LOCK:
        vault = _VAULTS.get(key)
        if vault is None:
            vault = _VAULTS[key] = Vault(state_dir, key_store=key_store)
        return vault


def reset_vaults() -> None:
    """Forget the shared vaults (tests, and a restore that replaced the state directory)."""
    with _VAULTS_LOCK:
        _VAULTS.clear()


__all__ = ["DB_NAME", "KEY_NAME", "KEY_STORES", "SecretForgotten", "UnknownReference", "Vault", "VaultUnavailable",
           "get_vault", "reset_vaults"]
