"""Training rows by reference, and the manifests that keep them.

A ``RowSource`` yields training rows as mappings with exactly the keys of ``ROW_KEYS``: the decision's id
(``ref``), its point and version, the state re-rendered from its references (``state``, in memory only, never
written), the keyed hash logged with the decision (``state_hash``, the de-duplication key), the label, the teacher
that produced it, whether it may only push away, where the label came from (``source_kind``), the narrowest
audience its references show, and its time. A source re-renders each state from its references and yields a row
only when the state still matches its logged hash, so an erased or changed source cannot re-render and leaves the
training set by construction.

A ``Manifest`` is a period's rows by reference: every key but ``state``, with the admission verdict of each row. It
is content-addressed (its id is a digest of what it lists) and holds no text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (Any, Dict, FrozenSet, Iterable, Iterator, List, Mapping, Optional, Protocol, Tuple,
                    runtime_checkable)

from . import _io
from .audience import AUDIENCES, narrowest

ROW_KEYS: Tuple[str, ...] = ("ref", "point", "version", "state", "state_hash", "label", "teacher", "push_away",
                             "source_kind", "audience", "at")
#: What a manifest keeps of a row: every key but the state, plus the admission's verdict and its reason.
ENTRY_KEYS: Tuple[str, ...] = tuple(key for key in ROW_KEYS if key != "state") + ("verdict", "reason")
#: ``admitted`` trains; ``witness`` repeats an admitted row (a witness, not a row); ``quarantine`` is not trainable
#: yet; ``refused`` never trains (its reason says why).
VERDICTS: Tuple[str, ...] = ("admitted", "witness", "quarantine", "refused")
_TEXT_KEYS = ("ref", "point", "version", "state", "state_hash", "teacher", "source_kind")


class RowError(ValueError):
    """A row that is not a training row of the contract, and why."""


@runtime_checkable
class RowSource(Protocol):
    """Where training rows come from, re-rendered on demand; nothing it yields is stored with its state."""

    def rows(self, objective: Any, *, since: Optional[Any] = None) -> Iterable[Mapping[str, Any]]:
        ...


def check_row(row: Mapping[str, Any]) -> None:
    """RowError unless ``row`` has exactly the contract's keys, each of its type."""
    if not isinstance(row, Mapping):
        raise RowError("a row is a mapping")
    keys = set(row)
    if keys != set(ROW_KEYS):
        extra, missing = sorted(keys - set(ROW_KEYS), key=str), sorted(set(ROW_KEYS) - keys)
        raise RowError(f"a row has exactly the contract's keys (extra: {', '.join(map(str, extra)) or 'none'}; "
                       f"missing: {', '.join(missing) or 'none'})")
    for key in _TEXT_KEYS:
        if not isinstance(row[key], str) or not row[key]:
            raise RowError(f"{key} is a non-empty string")
    if not isinstance(row["push_away"], bool):
        raise RowError("push_away is true or false")
    if row["audience"] not in AUDIENCES:
        raise RowError(f"audience is one of {', '.join(AUDIENCES)}")
    label = row["label"]
    if isinstance(label, float) and label != label:
        raise RowError("a label is a value, not NaN")
    try:
        _io.canonical(label)
    except (TypeError, ValueError):
        raise RowError("a label is plain data") from None
    try:
        _io.parse(row["at"])
    except (TypeError, ValueError):
        raise RowError("at is an ISO 8601 time with its zone") from None


@dataclass(frozen=True)
class Entry:
    """One row of a manifest, by reference: no state."""

    ref: str
    point: str
    version: str
    state_hash: str
    label: Any
    teacher: str
    push_away: bool
    source_kind: str
    audience: str
    at: str
    verdict: str
    reason: str = ""

    @classmethod
    def of(cls, row: Mapping[str, Any], verdict: str, reason: str = "") -> "Entry":
        if verdict not in VERDICTS:
            raise ValueError(f"verdict {verdict!r} is not one of {', '.join(VERDICTS)}")
        return cls(ref=row["ref"], point=row["point"], version=row["version"], state_hash=row["state_hash"],
                   label=row["label"], teacher=row["teacher"], push_away=row["push_away"],
                   source_kind=row["source_kind"], audience=row["audience"], at=row["at"], verdict=verdict,
                   reason=reason)

    def as_dict(self) -> Dict[str, Any]:
        return {key: getattr(self, key) for key in ENTRY_KEYS}


@dataclass(frozen=True)
class Manifest:
    """A period's training rows by reference, with the admission's verdicts and counts.

    ``witnesses``: repeats counted, not trained; ``overruled``: per model teacher, rows an outcome contradicted
    (its measured error); ``malformed``: rows that were not of the contract and so could not be listed."""

    learner: str
    objective: str
    objective_identity: str
    period: str
    admitted_at: str
    entries: Tuple[Entry, ...]
    witnesses: int = 0
    overruled: Mapping[str, int] = field(default_factory=dict)
    malformed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"learner": self.learner, "objective": self.objective, "objective_identity": self.objective_identity,
                "period": self.period, "admitted_at": self.admitted_at,
                "entries": [entry.as_dict() for entry in self.entries], "witnesses": self.witnesses,
                "overruled": dict(self.overruled), "malformed": self.malformed}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Manifest":
        return cls(learner=record["learner"], objective=record["objective"],
                   objective_identity=record["objective_identity"], period=record["period"],
                   admitted_at=record["admitted_at"],
                   entries=tuple(Entry(**entry) for entry in record["entries"]),
                   witnesses=int(record.get("witnesses", 0)), overruled=dict(record.get("overruled", {})),
                   malformed=int(record.get("malformed", 0)))

    @property
    def id(self) -> str:
        return _io.digest(self.to_dict())

    def admitted(self) -> Tuple[Entry, ...]:
        return tuple(entry for entry in self.entries if entry.verdict == "admitted")

    def refs(self) -> FrozenSet[str]:
        """Every ref the manifest names, whatever its verdict."""
        return frozenset(entry.ref for entry in self.entries)

    def names(self, refs: Iterable[str]) -> bool:
        return not self.refs().isdisjoint(refs)

    def audience(self) -> Optional[str]:
        """The narrowest audience of its admitted rows; None when it admitted none."""
        found = [entry.audience for entry in self.admitted()]
        return narrowest(found) if found else None


class ManifestStore:
    """Manifests on disk, one file per manifest, named by its content id."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def path_of(self, manifest_id: str) -> Path:
        return self.directory / f"{manifest_id}.json"

    def put(self, manifest: Manifest) -> str:
        path = self.path_of(manifest.id)
        if not path.exists():
            _io.write_json(path, manifest.to_dict())
        return manifest.id

    def get(self, manifest_id: str) -> Manifest:
        path = self.path_of(manifest_id)
        try:
            manifest = Manifest.from_dict(_io.read_json(path))
        except FileNotFoundError:
            raise KeyError(f"no manifest {manifest_id}") from None
        if manifest.id != manifest_id:
            raise ValueError(f"manifest {manifest_id}: its content does not match its id")
        return manifest

    def delete(self, manifest_id: str) -> bool:
        return _io.remove(self.path_of(manifest_id))

    def ids(self) -> List[str]:
        if not self.directory.is_dir():
            return []
        return sorted(path.stem for path in self.directory.glob("*.json"))

    def __iter__(self) -> Iterator[Manifest]:
        return (self.get(manifest_id) for manifest_id in self.ids())


__all__ = ["ENTRY_KEYS", "ROW_KEYS", "VERDICTS", "Entry", "Manifest", "ManifestStore", "RowError", "RowSource",
           "check_row"]
