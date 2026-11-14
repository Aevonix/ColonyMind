"""Pinned upstream files: downloaded once into a local cache and used only when their bytes hash to the pin.

A pin names an immutable upstream location (a commit or dataset revision is part of the URL) and the SHA-256 of
the file there. A cached file that does not hash to its pin is refused and left where it is, so a changed or
damaged copy is never read and never silently replaced. A download is written beside the cache entry under a
temporary name and moved into place only after its hash matches.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

DATA_DIR_ENV = "COLONYMIND_BENCH_DATA_DIR"
CHUNK = 1 << 20
TIMEOUT_SECONDS = 60


class PinError(ValueError):
    """A file is missing, or its bytes are not the pinned ones."""


@dataclass(frozen=True)
class Pin:
    name: str      # the file's name in the cache
    url: str       # an immutable upstream location
    sha256: str
    size: int      # bytes, for the record


def cache_dir(directory: Optional[os.PathLike] = None) -> Path:
    """The named directory, else ``COLONYMIND_BENCH_DATA_DIR``, else the user's cache directory."""
    if directory is not None:
        return Path(directory)
    if os.environ.get(DATA_DIR_ENV):
        return Path(os.environ[DATA_DIR_ENV])
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "colonymind" / "public-datasets"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(pin: Pin, directory: Optional[os.PathLike] = None, *, download: bool = True,
          opener: Callable = urllib.request.urlopen) -> Path:
    """The cached file for ``pin``, downloaded first when it is missing and ``download`` is on."""
    folder = cache_dir(directory)
    path = folder / pin.name
    if path.exists():
        digest = sha256_file(path)
        if digest != pin.sha256:
            raise PinError(f"{path}: SHA-256 {digest} is not the pinned {pin.sha256}; the file is not used. "
                           "Remove it to download the pinned file again.")
        return path
    if not download:
        raise PinError(f"{pin.name} is not in {folder}; fetch it first "
                       "(python -m benchmarks.public_memory.datasets fetch)")
    folder.mkdir(parents=True, exist_ok=True)
    partial = folder / f".{pin.name}.{os.getpid()}.part"
    digest = hashlib.sha256()
    try:
        with opener(pin.url, timeout=TIMEOUT_SECONDS) as response, open(partial, "wb") as handle:
            for chunk in iter(lambda: response.read(CHUNK), b""):
                digest.update(chunk)
                handle.write(chunk)
        if digest.hexdigest() != pin.sha256:
            raise PinError(f"{pin.url}: SHA-256 {digest.hexdigest()} is not the pinned {pin.sha256}; "
                           "the download is discarded.")
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()
    return path
