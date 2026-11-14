"""File helpers for the plasticity store: canonical JSON, content digests, atomic writes and array files.

Every file the store writes goes through ``write_atomic``: the bytes go to a temporary file beside the target,
are flushed and synced, and replace the target in one rename, after which the directory is synced. A reader
therefore sees the old file or the new one, never a torn one; a crash leaves at most a stray temporary, which
``sweep`` removes and which no reader opens.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

#: The rename that publishes a file, looked up at call time so a test can simulate a crash at the swap.
_replace = os.replace
TEMPORARY_SUFFIX = ".tmp"


def canonical(value: Any) -> bytes:
    """``value`` as canonical JSON bytes: sorted keys, no spaces, no NaN."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    """A content id: the first 32 hex digits of the SHA-256 of ``value``'s canonical JSON."""
    return hashlib.sha256(canonical(value)).hexdigest()[:32]


def _sync_directory(directory: Path) -> None:
    try:
        handle = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


def write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` so no reader ever sees part of it (the module docstring)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}{TEMPORARY_SUFFIX}")
    try:
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _sync_directory(path.parent)


def write_json(path: Path, value: Any) -> None:
    write_atomic(path, canonical(value))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def remove(path: Path) -> bool:
    """Delete ``path`` if it exists and sync its directory; True when something was deleted."""
    path = Path(path)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    _sync_directory(path.parent)
    return True


def sweep(directory: Path) -> int:
    """Remove the temporaries a crash left under ``directory``; the number removed."""
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    stray = [path for path in directory.rglob(f".*{TEMPORARY_SUFFIX}") if path.is_file()]
    for path in stray:
        path.unlink(missing_ok=True)
    return len(stray)


def stamp(moment: datetime) -> str:
    """An aware time as ISO 8601 in UTC."""
    if moment.tzinfo is None:
        raise ValueError("a time without a zone is not a UTC time")
    return moment.astimezone(timezone.utc).isoformat()


def parse(value: Any) -> datetime:
    """An ISO 8601 time with its zone, as an aware UTC time; ValueError otherwise (a time without a zone is
    refused, never guessed)."""
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str) and value.strip():
        moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    else:
        raise ValueError(f"not a time: {value!r}")
    if moment.tzinfo is None:
        raise ValueError(f"a time without a zone: {value!r}")
    return moment.astimezone(timezone.utc)


# -- learned states --------------------------------------------------------------------------------------------------

def state_digest(state: Mapping[str, np.ndarray]) -> str:
    """The learned state's digest: every array's name, dtype, shape and bytes, in name order."""
    hasher = hashlib.sha256()
    for name in sorted(state):
        array = np.ascontiguousarray(state[name])
        hasher.update(canonical([name, array.dtype.str, list(array.shape)]))
        hasher.update(array.tobytes())
    return hasher.hexdigest()[:32]


def save_arrays(state: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **{name: np.ascontiguousarray(array) for name, array in state.items()})
    return buffer.getvalue()


def load_arrays(data: bytes) -> Dict[str, np.ndarray]:
    with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
        return {name: loaded[name].copy() for name in loaded.files}
