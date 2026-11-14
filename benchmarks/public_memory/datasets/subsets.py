"""Seeded, stratified subsets whose ids are recorded, so a run names exactly the questions it used.

Questions are grouped into strata (by type, or by type and abstention). Each stratum gets its share of the size by
largest remainder (a tie goes to the larger stratum, then to the name that sorts first), or an explicit quota. Within
a stratum, questions are taken in the order of SHA-256 of ``"<seed>:<question id>"``: the choice depends only on the
seed and the ids, not on file order or on the Python version, and anyone can recompute it. Ids to leave out (another
subset, a development slice) are excluded before the draw and recorded.

A subset's record (``Subset.record``) carries the dataset's name and SHA-256, the seed, the strata and the ids;
``check`` draws it again from the data and refuses a record that does not match, including one whose size is not the
number of its ids.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Tuple

from .records import Dataset, DatasetError

SCHEME = "sha256(seed:id) within strata, largest remainder"
BY = {
    "type": lambda question: question.type,
    "type+abstention": lambda question: question.type + ("/abstention" if question.abstention else ""),
}


@dataclass(frozen=True)
class Subset:
    dataset: str
    sha256: str
    seed: str
    by: str
    strata: Mapping[str, int]
    ids: Tuple[str, ...]         # in the dataset's question order
    excluded: Tuple[str, ...]

    def record(self) -> dict:
        return {"scheme": SCHEME, "dataset": self.dataset, "sha256": self.sha256, "seed": self.seed, "by": self.by,
                "size": len(self.ids), "strata": dict(self.strata), "ids": list(self.ids),
                "excluded": list(self.excluded)}


def rank(seed: str, question_id: str) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode("utf-8")).hexdigest()


def _whole(count) -> bool:
    """A size or quota is a whole number, zero or more: a negative one would slice from the end of a stratum."""
    return isinstance(count, int) and not isinstance(count, bool) and count >= 0


def allocate(available: Mapping[str, int], size: int) -> dict:
    """Each stratum's share of ``size``, proportional to what it holds, by largest remainder."""
    if not _whole(size):
        raise DatasetError(f"a subset of {size!r} questions: a size is a whole number, zero or more")
    total = sum(available.values())
    if size > total:
        raise DatasetError(f"a subset of {size} from {total} questions")
    shares = {name: size * count / total for name, count in available.items()} if total else {}
    quotas = {name: int(share) for name, share in shares.items()}
    order = sorted(shares, key=lambda name: (-(shares[name] - quotas[name]), -available[name], name))
    for name in order[:size - sum(quotas.values())]:
        quotas[name] += 1
    return quotas


def draw(dataset: Dataset, *, seed, size: Optional[int] = None, quotas: Optional[Mapping[str, int]] = None,
         by: str = "type", exclude: Iterable[str] = ()) -> Subset:
    """The subset of ``dataset`` for ``seed``: ``size`` questions shared across strata, or ``quotas`` per stratum."""
    if (size is None) == (quotas is None):
        raise DatasetError("give a subset either a size or quotas")
    if by not in BY:
        raise DatasetError(f"strata by {by!r}: choose one of {', '.join(BY)}")
    seed, excluded = str(seed), tuple(sorted(set(exclude)))
    left_out = set(excluded)
    strata = {}
    for question in dataset.questions:
        if question.id not in left_out:
            strata.setdefault(BY[by](question), []).append(question)
    available = {name: len(items) for name, items in strata.items()}
    if quotas is None:
        quotas = allocate(available, size)
    for name, count in quotas.items():
        if not _whole(count):
            raise DatasetError(f"{count!r} {name} questions asked for: a quota is a whole number, zero or more")
        if name not in available:
            raise DatasetError(f"no {name} questions to draw from")
        if count > available[name]:
            raise DatasetError(f"{count} {name} questions asked for, {available[name]} available")
    chosen = set()
    for name, count in quotas.items():
        ranked = sorted(strata[name], key=lambda question: rank(seed, question.id))
        chosen.update(question.id for question in ranked[:count])
    ids = tuple(question.id for question in dataset.questions if question.id in chosen)
    counted = Counter(BY[by](question) for question in dataset.questions if question.id in chosen)
    return Subset(dataset=dataset.name, sha256=dataset.sha256, seed=seed, by=by,
                  strata={name: counted.get(name, 0) for name in quotas if quotas[name]}, ids=ids, excluded=excluded)


def check(record: Mapping, dataset: Dataset) -> Subset:
    """The subset ``record`` describes, drawn again from ``dataset``; refused if either does not match.

    The recorded size must be the number of the record's ids, and the ids must be the ones drawn again, so the size
    a record states is the size of the subset it names."""
    if record.get("scheme") != SCHEME:
        raise DatasetError(f"the record's scheme {record.get('scheme')!r} is not {SCHEME!r}")
    if (record.get("dataset"), record.get("sha256")) != (dataset.name, dataset.sha256):
        raise DatasetError(f"the record is for {record.get('dataset')} with SHA-256 {record.get('sha256')}, "
                           f"not {dataset.name} with SHA-256 {dataset.sha256}")
    size, ids = record.get("size"), record.get("ids", ())
    if not _whole(size) or size != len(ids):
        raise DatasetError(f"the record's size {size!r} is not the {len(ids)} ids it lists")
    subset = draw(dataset, seed=record["seed"], quotas=record["strata"], by=record["by"], exclude=record["excluded"])
    if sorted(subset.ids) != sorted(record["ids"]):
        raise DatasetError("the record's ids are not the ones its seed and strata draw")
    return subset
