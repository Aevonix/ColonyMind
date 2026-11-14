"""An append-only, private JSONL outbox with durable keyed acknowledgements."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import stat
import threading
from typing import Any, Callable


class Spool:
    """Persist before delivery; a lost acknowledgement reuses the original key.

    Keep acknowledged records for restart reconciliation. A final partial line
    can result from a killed writer and is discarded. Corruption elsewhere is
    an error: silently dropping an effect intent could permit duplicate work.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser()
        self.path = self.directory / "events.jsonl"
        self.writable = True
        self._lock = threading.RLock()
        self._replay_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._requests: dict[str, dict] = {}
        self._acks: dict[str, Any] = {}
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self._file():
            pass

    @contextmanager
    def _file(self):
        with self._lock:
            fd = None
            try:
                existed = self.path.exists()
                fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND |
                             getattr(os, "O_NOFOLLOW", 0), 0o600)
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise OSError("hands spool must be a regular file")
                fcntl.flock(fd, fcntl.LOCK_EX)
                os.fchmod(fd, 0o600)
                self._load(fd)
                if not existed:
                    os.fsync(fd)
                    parent = os.open(self.directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                    try:
                        os.fsync(parent)
                    finally:
                        os.close(parent)
                yield fd
            except OSError:
                self.writable = False
                raise
            finally:
                if fd is not None:
                    os.close(fd)

    def _load(self, fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        chunks = []
        while chunk := os.read(fd, 65536):
            chunks.append(chunk)
        data = b"".join(chunks)
        if data and not data.endswith(b"\n"):
            size = data.rfind(b"\n") + 1
            os.ftruncate(fd, size)
            os.fsync(fd)
            data = data[:size]
        requests, acks = {}, {}
        try:
            for line in data.splitlines():
                row = json.loads(line)
                key = row["key"]
                if row["type"] == "request":
                    if key in requests and requests[key] != row:
                        raise ValueError("conflicting spool key")
                    requests[key] = row
                elif row["type"] == "ack" and key in requests:
                    acks[key] = row.get("response")
                else:
                    raise ValueError("invalid spool record")
        except (ValueError, KeyError, TypeError) as exc:
            self.writable = False
            raise OSError("hands spool is corrupt") from exc
        self._requests, self._acks = requests, acks

    @staticmethod
    def _write(fd: int, row: dict) -> None:
        data = (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        while data:
            written = os.write(fd, data)
            if written <= 0:
                raise OSError("short spool write")
            data = data[written:]
        os.fsync(fd)

    def append(self, path: str, body: dict, key: str, *, metadata: dict | None = None) -> dict:
        if not path or not key:
            raise ValueError("spool requests require a path and key")
        if not self.writable:
            raise OSError("cannot record work safely")
        row = {"type": "request", "path": path, "body": deepcopy(body), "key": key}
        if metadata is not None:
            row["metadata"] = deepcopy(metadata)
        with self._file() as fd:
            previous = self._requests.get(key)
            if previous is not None:
                if any(previous[field] != row[field] for field in ("path", "body", "key")):
                    raise ValueError("idempotency key reused with different request")
                if metadata is not None and previous.get("metadata") != metadata:
                    raise ValueError("idempotency key reused with different metadata")
                return deepcopy(previous)
            self._write(fd, row)
            self._requests[key] = row
        return deepcopy(row)

    def acknowledge(self, key: str, response: Any = None) -> None:
        if not self.writable:
            raise OSError("cannot record work safely")
        with self._file() as fd:
            if key not in self._requests:
                raise KeyError(key)
            if key not in self._acks:
                self._write(fd, {"type": "ack", "key": key, "response": response})
                self._acks[key] = deepcopy(response)

    def records(self) -> list[dict]:
        with self._file():
            return deepcopy(list(self._requests.values()))

    def pending(self) -> list[dict]:
        with self._file():
            return deepcopy([row for key, row in self._requests.items() if key not in self._acks])

    def response(self, key: str) -> Any:
        with self._file():
            return deepcopy(self._acks.get(key))

    def probe(self) -> bool:
        try:
            with self._file() as fd:
                os.fsync(fd)
            self.writable = True
        except OSError:
            return False
        return True

    def replay(self, send: Callable[[str, dict, str], Any]) -> int:
        """Deliver in append order, stopping at the first failed write or ack."""
        delivered = 0
        if not self._replay_lock.acquire(blocking=False):
            return delivered
        try:
            for row in self.pending():
                try:
                    response = send(row["path"], row["body"], row["key"])
                    if response is None or response is False:
                        break
                    self.acknowledge(row["key"], response)
                except Exception:
                    break
                delivered += 1
        except OSError:
            pass
        finally:
            self._replay_lock.release()
        return delivered

    def start(self, send: Callable[[str, dict, str], Any], interval: float = 10.0) -> None:
        """Replay now and every ten seconds in one daemon per spool."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()

            def run():
                self.replay(send)
                while not self._stop.wait(interval):
                    self.replay(send)

            self._thread = threading.Thread(target=run, name="hands-spool", daemon=True)
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=0.1)


def register(ctx: Any) -> None:
    """Runtime owns spool startup so disabled plugins create no files or threads."""
