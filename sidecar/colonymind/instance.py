"""The instance id: one random UUID per state directory, created on first use, and the instance's own secret.

The id names this ColonyMind in the agent registry (``agents.colonymind_id``) and binds a
backup to the instance it was taken from. It carries no key and signs nothing; the
chain that used to derive an identity from it is gone (M8).

The secret (``instance_secret``) keys what the instance keeps as a hash so that its
databases alone cannot test a guess: Instinct's decision log (state hashes, hashed
references) and the model usage ledger (the aliases of names that are not plain).
"""

from __future__ import annotations

import os
import secrets
import uuid
from pathlib import Path
from typing import Optional

INSTANCE_ID_FILE = "instance-id"
#: The instance's own secret, made first for Instinct's decision log and named after it; a backup carries it only
#: encrypted (``backup.SECRET_FILES``).
SECRET_FILE = "instinct.key"
SECRET_BYTES = 32
# The chain wrote the same UUID under this name; it is read once so existing
# registry rows keep matching, then `colonymind upgrade` moves it into the backup.
LEGACY_ID_FILE = "colonymind-id"


def instance_id(state_dir: str | Path) -> str:
    """Return the instance id stored in ``<state_dir>/instance-id``, creating it if absent."""
    home = Path(state_dir)
    path = home / INSTANCE_ID_FILE
    if path.is_file():
        value = path.read_text().strip()
        if value:
            return value
    legacy = home / LEGACY_ID_FILE
    value = legacy.read_text().strip() if legacy.is_file() else ""
    value = value or str(uuid.uuid4())
    home.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(value + "\n")
    return value


def read_instance_id(state_dir: str | Path) -> str:
    """The instance id when one exists, else ``""`` (never creates one).

    A state directory not yet upgraded still names itself in ``colonymind-id``;
    that value is the id :func:`instance_id` would adopt, so it is returned too.
    """
    home = Path(state_dir)
    for name in (INSTANCE_ID_FILE, LEGACY_ID_FILE):
        path = home / name
        value = path.read_text().strip() if path.is_file() else ""
        if value:
            return value
    return ""


def _read_secret(path: Path) -> bytes:
    """The key ``path`` holds: exactly ``SECRET_BYTES`` bytes in hex, else ``ValueError`` (an empty, cut or
    overlong file keys nothing: a key anyone could guess would let a guess be tested)."""
    try:
        value = bytes.fromhex(path.read_text(encoding="ascii").strip())
    except ValueError:
        value = b""
    if len(value) != SECRET_BYTES:
        raise ValueError(f"{SECRET_FILE} holds no {SECRET_BYTES}-byte key: restore it from a backup, or remove it "
                         "to make a new one (what the old one keyed then matches nothing)")
    return value


def instance_secret(state_dir: str | Path, *, create: bool = True) -> Optional[bytes]:
    """The instance's own secret: 32 random bytes, hex in ``<state_dir>/instinct.key`` (owner-only), made on first
    use. It is written whole under a private name and linked into place, so it appears complete or not at all; when
    two processes make it at once, the first link wins and both use it. ``create`` off: None when there is none
    yet. Raises ``ValueError`` for a file that holds no whole key (it is never replaced: what it keyed would match
    nothing) and ``OSError`` when it cannot be read or written."""
    path = Path(state_dir) / SECRET_FILE
    try:
        return _read_secret(path)
    except FileNotFoundError:
        if not create:
            return None
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.parent / f".{SECRET_FILE}.{os.getpid()}.{secrets.token_hex(6)}"
    fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as stream:
            stream.write(secrets.token_bytes(SECRET_BYTES).hex() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staged, path)
        except FileExistsError:        # another process linked its key first: use theirs
            pass
    finally:
        staged.unlink(missing_ok=True)
    return _read_secret(path)
