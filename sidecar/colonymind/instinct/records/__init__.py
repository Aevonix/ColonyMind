"""Calibration records shipped as data: the records measured for a model tier on a served model, so an operator
serving the same model and revision can promote a point on them.

Each ``*.json`` file in this directory holds ``{"records": [...]}``, one entry per record:

- the key it was measured on: ``point`` and its ``version`` (the question), ``tier``, ``provider_kind``, the
  configured ``model`` and the served ``revision``;
- ``fingerprint``: that key's calibration fingerprint, as computed when the record was measured;
- ``method`` and its parameters (``temperature``, or isotonic ``maps`` and ``support``), the ``bars``, ``n`` and the
  held-out ``accuracy`` it was fitted on, its ``source`` and ``fitted_at``. A bar may be ``NEVER`` to prevent a label
  from settling.

A record applies only to the key it was measured on: the engine matches the fingerprint exactly, so another model,
another revision of it or a changed question gets no record, and its tier abstains while shadow logging goes on.
Loading checks each entry: one whose fingerprint is not its key's (a key edited after the measurement), whose tier
is not a model tier, whose provider kind is unknown or whose served revision is blank is refused; one whose point
is not registered, or is registered with another question, is not shipped. Neither is ever fatal: what fails a check is left out and logged.
"""

from __future__ import annotations

import functools
import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from ..calibration import NEVER, Record
from ..registry import MODEL_TIERS, REGISTRY
from ..settings import KINDS

logger = logging.getLogger(__name__)

DIRECTORY = Path(__file__).resolve().parent
REQUIRED = ("point", "version", "tier", "provider_kind", "model", "revision", "fingerprint", "method", "bars",
            "source", "fitted_at")


class RecordError(ValueError):
    """A shipped record that fails its checks, and why."""


def _record(entry: Any, where: str) -> Record:
    if not isinstance(entry, Mapping):
        raise RecordError(f"{where}: not a record")
    missing = [key for key in REQUIRED if key not in entry]
    if missing:
        raise RecordError(f"{where}: missing {', '.join(missing)}")
    if entry["tier"] not in MODEL_TIERS:
        raise RecordError(f"{where}: tier {entry['tier']!r} is not a model tier")
    kind = entry["provider_kind"]
    if not isinstance(kind, str) or kind.split("/", 1)[0] not in KINDS:
        raise RecordError(f"{where}: provider kind {entry['provider_kind']!r} is unknown")
    revision = entry["revision"]
    if not isinstance(revision, str) or not revision.strip():
        raise RecordError(f"{where}: a shipped record requires a served revision")
    bars = entry["bars"]
    if not isinstance(bars, Mapping) or not bars or any(
            isinstance(bar, bool) or not isinstance(bar, (int, float)) or not math.isfinite(bar)
            or not (0.0 <= float(bar) <= 1.0 or float(bar) == NEVER) for bar in bars.values()):
        raise RecordError(f"{where}: bars map labels to probabilities or NEVER")
    try:
        record = Record.from_dict(entry)
    except (TypeError, ValueError) as error:
        raise RecordError(f"{where}: {type(error).__name__}") from None
    if record.fingerprint != entry["fingerprint"]:
        raise RecordError(f"{where}: its fingerprint is not the fingerprint of its key")
    return record


def _read(directory: Path, strict: bool) -> Tuple[Record, ...]:
    out = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entries = data.get("records") if isinstance(data, dict) else None
            if not isinstance(entries, list):
                raise RecordError(f"{path.name}: no list of records")
        except (OSError, ValueError) as error:
            if strict:
                raise RecordError(f"{path.name}: {error}") from None
            logger.warning("calibration records in %s left out (%s)", path.name, type(error).__name__)
            continue
        for index, entry in enumerate(entries):
            try:
                out.append(_record(entry, f"{path.name}[{index}]"))
            except RecordError as error:
                if strict:
                    raise
                logger.warning("a shipped calibration record is left out: %s", error)
    return tuple(out)


@functools.lru_cache(maxsize=8)
def _cached(directory: str, signature: Tuple[Tuple[str, int, int], ...]) -> Tuple[Record, ...]:
    return _read(Path(directory), strict=False)


def load(directory: Optional[Path] = None, *, strict: bool = False) -> Tuple[Record, ...]:
    """Every record that passes its checks, in file order; ``strict``: RecordError for the first that does not."""
    directory = Path(directory or DIRECTORY)
    if strict:
        return _read(directory, strict=True)
    try:
        signature = tuple((path.name, path.stat().st_mtime_ns, path.stat().st_size)
                          for path in sorted(directory.glob("*.json")))
    except OSError:
        return ()
    return _cached(str(directory), signature)


def shipped(registry: Optional[Mapping[str, Any]] = None, directory: Optional[Path] = None) -> Tuple[Record, ...]:
    """The loaded records whose point is registered with the question they were measured on."""
    registry = REGISTRY if registry is None else registry
    return tuple(record for record in load(directory)
                 if record.point in registry and getattr(registry[record.point], "version", None) == record.version)


__all__ = ["DIRECTORY", "REQUIRED", "RecordError", "load", "shipped"]
