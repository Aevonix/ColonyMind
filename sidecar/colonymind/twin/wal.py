"""Read committed frames from a SQLite write-ahead log, as the file format documents it.

https://sqlite.org/fileformat2.html#the_write_ahead_log: a 32-byte header, then frames of a
24-byte header and one page image. A frame belongs to the log only while its salts match the
header's and its cumulative checksum continues the chain; the frame that ends a transaction
carries the database size in pages. Only frames up to the last such commit frame are returned,
so a transaction still being written, a torn frame or a stale frame from an earlier log
generation is never shipped.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, replace

MAGIC_LE, MAGIC_BE = 0x377F0682, 0x377F0683
HEADER, FRAME_HEADER = 32, 24
_MASK = 0xFFFFFFFF


def checksum(data: bytes, big_endian: bool, s0: int = 0, s1: int = 0) -> tuple[int, int]:
    """SQLite's WAL checksum over ``data`` (a multiple of 8 bytes), continuing ``(s0, s1)``."""
    words = struct.unpack(('>' if big_endian else '<') + f'{len(data) // 4}I', data)
    pairs = iter(words)
    for x0, x1 in zip(pairs, pairs):
        s0 = (s0 + x0 + s1) & _MASK
        s1 = (s1 + x1 + s0) & _MASK
    return s0, s1


@dataclass(frozen=True)
class Position:
    """Where a reader stands in one log generation: after ``frames`` frames, the last a commit."""
    salt1: int
    salt2: int
    checkpoint_seq: int
    page_size: int
    big_endian: bool
    frames: int
    s0: int
    s1: int

    def same_generation(self, other: 'Position') -> bool:
        return (self.salt1, self.salt2, self.checkpoint_seq) == (other.salt1, other.salt2, other.checkpoint_seq)


@dataclass
class Scan:
    """Committed changes after a position: page images (last write wins) and the final size."""
    position: Position | None     # None: no valid log (absent, empty or not yet written)
    pages: dict[int, bytes]
    db_pages: int | None           # database size after the last commit read, if any
    reset: bool                    # the log is a different generation than the one asked for


def read_header(data: bytes) -> Position | None:
    if len(data) < HEADER:
        return None
    magic, version, page_size, seq, salt1, salt2, c0, c1 = struct.unpack('>8I', data[:HEADER])
    if magic not in (MAGIC_LE, MAGIC_BE) or version != 3007000:
        return None
    big = magic == MAGIC_BE
    if page_size == 1:
        page_size = 65536
    if (c0, c1) != checksum(data[:24], big):
        return None
    return Position(salt1, salt2, seq, page_size, big, 0, c0, c1)


class LogChanged(OSError):
    """The log restarted again and again while it was being read."""


def scan(path: str | os.PathLike[str], after: Position | None = None, *, max_bytes: int | None = None,
         attempts: int = 8) -> Scan:
    """Committed frames of the log at ``path`` after ``after`` (from the start when None).

    ``reset`` is set when ``after`` names another generation: the caller cannot tell from the
    log alone whether frames of the old generation were lost and must resynchronise.

    A restart can begin while the log is read (nothing stops one under a reader on read-mark 0):
    frames at the start of the file are then overwritten by the new generation, and a read that
    straddles it would return an old prefix, older than frames it never reached. The header is
    read again after the frames; a scan whose header changed is discarded and read again.
    """
    for _ in range(attempts):
        try:
            handle = open(path, 'rb')
        except FileNotFoundError:
            return Scan(None, {}, None, after is not None)
        with handle:
            result = _scan_once(handle, after, max_bytes)
            if result is None:
                return Scan(None, {}, None, after is not None)
            first, found = result
            handle.seek(0)
            if handle.read(HEADER) == first:
                return found
    raise LogChanged(f'{path}: the log restarted during {attempts} consecutive reads')


def _scan_once(handle, after: Position | None, max_bytes: int | None) -> tuple[bytes, Scan] | None:
    raw_header = handle.read(HEADER)
    header = read_header(raw_header)
    if header is None:
        return None
    reset = after is not None and not after.same_generation(header)
    start = header if after is None or reset else after
    size = header.page_size
    handle.seek(HEADER + start.frames * (FRAME_HEADER + size))
    s0, s1, frame = start.s0, start.s1, start.frames
    pending: dict[int, bytes] = {}
    committed: dict[int, bytes] = {}
    position, db_pages, read = start, None, 0
    while max_bytes is None or read < max_bytes:
        raw = handle.read(FRAME_HEADER + size)
        if len(raw) < FRAME_HEADER + size:
            break
        pgno, commit, salt1, salt2, c0, c1 = struct.unpack('>6I', raw[:FRAME_HEADER])
        if (salt1, salt2) != (header.salt1, header.salt2) or pgno == 0:
            break
        page = raw[FRAME_HEADER:]
        s0, s1 = checksum(raw[:8], header.big_endian, s0, s1)
        s0, s1 = checksum(page, header.big_endian, s0, s1)
        if (s0, s1) != (c0, c1):
            break
        frame += 1
        read += len(raw)
        pending[pgno] = page
        if commit:
            committed.update(pending)
            pending.clear()
            db_pages = commit
            position = replace(header, frames=frame, s0=s0, s1=s1)
    return raw_header, Scan(position, committed, db_pages, reset)
