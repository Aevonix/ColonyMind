"""Segments: the unit the primary ships and the twin applies, one file per shipping tick.

A segment holds, for each store that changed, the page images to write and the store's size
after them. A store too changed to ship as pages ships as a seed file beside the segment (a
consistent SQLite snapshot) with the pages written on top. Files are written to a temporary
name and renamed, and carry the SHA-256 of their own bytes in their name, so a torn, partial
or altered file is never applied.

Layout: ``PTWSEG1\\n``, a 4-byte big-endian header length, the JSON header, then per entry a
zlib stream of ``pages`` records, each a 4-byte page number and one page image.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import zlib
from pathlib import Path
from typing import Iterator

MAGIC = b'PTWSEG1\n'
NAME = re.compile(r'^(\d{12})-([0-9a-f]{24})\.seg$')
SEED = re.compile(r'^seed-(\d{12})-(\d+)-([0-9a-f]{24})\.db$')
_CHUNK = 1 << 20


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(_CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def fsync_dir(path: str | os.PathLike[str]) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def publish(temporary: Path, directory: Path, name_for) -> Path:
    """fsync ``temporary`` and rename it into ``directory`` under ``name_for(sha256)``."""
    with open(temporary, 'rb+') as handle:
        os.fsync(handle.fileno())
    final = directory / name_for(file_sha256(temporary))
    os.replace(temporary, final)
    fsync_dir(directory)
    return final


def write(directory: Path, header: dict, entries: list[tuple[dict, dict[int, bytes]]]) -> Path:
    """Write one segment; ``entries`` pairs each entry's metadata with its page images."""
    directory.mkdir(parents=True, exist_ok=True)
    payload, described = bytearray(), []
    for meta, pages in entries:
        stream = zlib.compressobj(1)
        body = bytearray()
        for pgno in sorted(pages):
            body += stream.compress(struct.pack('>I', pgno) + pages[pgno])
        body += stream.flush()
        described.append({**meta, 'pages': len(pages), 'offset': len(payload), 'length': len(body)})
        payload += body
    head = json.dumps({**header, 'entries': described}, sort_keys=True).encode()
    temporary = directory / f'.tmp-{header["seq"]:012d}-{os.getpid()}'
    with open(temporary, 'wb') as handle:
        handle.write(MAGIC + struct.pack('>I', len(head)) + head)
        handle.write(payload)
    return publish(temporary, directory, lambda sha: f'{header["seq"]:012d}-{sha[:24]}.seg')


def verified(path: Path) -> bool:
    """True when the file's bytes still hash to the digest its name carries."""
    match = NAME.match(path.name) or SEED.match(path.name)
    if not match:
        return False
    try:
        return file_sha256(path)[:24] == match.groups()[-1]
    except OSError:
        return False


def read_header(path: Path) -> tuple[dict, int]:
    """The header and the payload's offset in the file."""
    with open(path, 'rb') as handle:
        if handle.read(len(MAGIC)) != MAGIC:
            raise ValueError('not a twin segment')
        (length,) = struct.unpack('>I', handle.read(4))
        header = json.loads(handle.read(length))
    return header, len(MAGIC) + 4 + length


def iter_pages(path: Path, base: int, entry: dict) -> Iterator[tuple[int, bytes]]:
    """Stream one entry's page images without holding them all in memory."""
    record = 4 + int(entry['page_size'])
    stream = zlib.decompressobj()
    buffer = bytearray()
    remaining, count = int(entry['length']), 0
    with open(path, 'rb') as handle:
        handle.seek(base + int(entry['offset']))
        while remaining > 0 or len(buffer) >= record:
            if len(buffer) < record and remaining > 0:
                block = handle.read(min(_CHUNK, remaining))
                if not block:
                    break
                remaining -= len(block)
                buffer += stream.decompress(block)
                if remaining == 0:
                    buffer += stream.flush()
                continue
            if len(buffer) < record:
                break
            (pgno,) = struct.unpack('>I', buffer[:4])
            yield pgno, bytes(buffer[4:record])
            del buffer[:record]
            count += 1
    if count != int(entry['pages']) or buffer:
        raise ValueError('segment entry is truncated')
