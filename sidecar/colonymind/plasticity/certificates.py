"""Durable, aggregate-only gate records, independent of learner implementation.

A run has its own file, even when the same revision and gate set run again.
No row references, inputs or labels belong in a certificate. Unmeasured or
non-finite statistics are represented by null, with the outcome explaining why.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from . import _io


@dataclass
class GateRecord:
    learner: str
    revision: str
    state: str
    serving_revision: str
    at: str
    gate_set: str | None = None
    loss: str = ""
    arms: dict[str, float | None] = field(default_factory=dict)
    comparisons: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, int | None] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    parity: dict[str, Any] = field(default_factory=lambda: {"status": "not-tested", "passed": None})
    checks: list[dict[str, Any]] = field(default_factory=list)
    outcome: str = "error"
    reason: str = ""
    error: str | None = None


class GateRecords:
    """Atomic certificates under a learner's directory; reading never loads it."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def write(self, record: GateRecord) -> str:
        run_id = uuid4().hex
        _io.write_json(self.directory / (run_id + ".json"),
                       {"schema": "gate-certificate-1", "id": run_id, **asdict(record)})
        return run_id

    def records(self, *, since: datetime | None = None, until: datetime | None = None):
        """Read in time order, optionally within the UTC interval [since, until)."""
        records = [_io.read_json(path) for path in self.directory.glob("*.json")]
        return sorted((record for record in records
                       if (since is None or _io.parse(record["at"]) >= _io.parse(since))
                       and (until is None or _io.parse(record["at"]) < _io.parse(until))),
                      key=lambda record: (record["at"], record["id"]))
