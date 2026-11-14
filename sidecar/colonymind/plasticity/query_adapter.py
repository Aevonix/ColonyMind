"""NumPy query adapters and the portable ``query-adapter-1`` revision format.

Trusted targets train a bounded residual on queries. Model teachers train separate
factors which can only adjust scores in a supplied shown set. A global linear map
cannot itself guarantee that an unseen passage never rises, so those factors never
enter ``apply``. Passage vectors stay fixed. Training payloads are transient JSON
states supplied by a by-reference row source, never a durable vector manifest.

Numerical assumptions: projection and step caps allow 1e-6 relative float32
roundoff; revision bound validation allows 1e-9 float64 QR roundoff. A 1e-12
loss allowance avoids rejecting a neutral step due to summation roundoff, 1e-30
protects factor norm division, and squared loss uses 1e-6 target regularization.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .admission import ADVERSARIAL_SOURCE_KINDS
from .builds import TrainingRow
from .objectives import Objective

MODEL_TEACHERS = frozenset({"deliberation", "stronger_model", "behavioural_proxy"})
PROOF_TEACHERS = frozenset({"owner_label", "verified_outcome"})
SCHEMA = "query-adapter-1"


def normalize(value) -> np.ndarray:
    """Normalize finite nonzero float32 vectors, including very small vectors."""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim < 1 or not np.isfinite(array).all():
        raise ValueError("vectors must be finite")
    norm = np.linalg.norm(array.astype(np.float64), axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("zero vectors cannot be normalized")
    return (array / norm).astype(np.float32)


def spectral(u, v) -> float:
    """Operator norm of the residual, using only the two thin QR factors."""
    u, v = np.asarray(u), np.asarray(v)
    if u.ndim != 2 or u.shape != v.shape or not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError("factors must be finite matrices of equal shape")
    if not np.any(u) or not np.any(v):
        return 0.0
    ru = np.linalg.qr(u.astype(np.float64), mode="r")
    rv = np.linalg.qr(v.astype(np.float64), mode="r")
    return float(np.linalg.norm(ru @ rv.T, ord=2))


def project_factors(u, v, *, bound: float = .05) -> tuple[np.ndarray, np.ndarray]:
    """Project the factor product, retaining a margin for float32 roundoff."""
    if not math.isfinite(bound) or not 0 < bound <= .05:
        raise ValueError("operator-norm bound must be positive and at most 0.05")
    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    norm = spectral(u, v)
    if norm > bound:
        scale = math.sqrt(bound * (1 - 1e-6) / norm)
        u, v = u * scale, v * scale
    result = (np.asarray(u, dtype=np.float32), np.asarray(v, dtype=np.float32))
    # Casting an already bounded float64 state can cross the boundary too.
    norm = spectral(*result)
    if norm > bound:
        scale = math.sqrt(bound * (1 - 1e-6) / norm)
        result = tuple((factor * scale).astype(np.float32) for factor in result)
    return result


def weights_digest(u, v) -> str:
    """Interchange digest: float32 U bytes, then V bytes; zero map is identity."""
    u, v = (np.asarray(factor, dtype=np.float32, order="C") for factor in (u, v))
    if spectral(u, v) == 0:
        return "identity"
    return hashlib.sha256(u.tobytes(order="C") + v.tobytes(order="C")).hexdigest()


def apply(query, u, v) -> np.ndarray:
    """Apply a global residual and normalize; identity returns normalized input."""
    q = normalize(query)
    u, v = np.asarray(u, dtype=np.float32), np.asarray(v, dtype=np.float32)
    if u.ndim != 2 or u.shape != v.shape or q.shape[-1] != u.shape[0]:
        raise ValueError("query dimensions do not match factors")
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError("factors must be finite")
    if not np.any(u) or not np.any(v):
        return q
    return normalize(q + (q @ v) @ u.T)


@dataclass(frozen=True)
class Revision:
    metadata: dict
    u: np.ndarray
    v: np.ndarray

    def apply(self, queries) -> np.ndarray:
        if self.metadata.get("scope", "global") != "global":
            raise ValueError("a shown revision requires rank_scores, never global apply")
        return apply(queries, self.u, self.v)

    def rank_scores(self, query, passages, *, shown, adversarial=(), base_scores=None) -> dict[str, float]:
        if self.metadata.get("scope", "global") != "shown":
            raise ValueError("rank_scores requires a shown revision")
        return _rank_scores(query, passages, self.u, self.v, shown=shown, adversarial=adversarial,
                            base_scores=base_scores, bound=self.metadata["operator_norm_bound"])


def _rank_scores(query, passages, u, v, *, shown, adversarial=(), base_scores=None, bound=.05):
    ids = list(passages)
    if not set(shown) <= set(ids) or not set(adversarial) <= set(ids):
        raise ValueError("shown and adversarial ids must name supplied passages")
    if not ids:
        return {}
    q = normalize(query)
    docs = normalize([passages[key] for key in ids])
    if q.ndim != 1 or docs.ndim != 2 or docs.shape[1] != q.shape[0]:
        raise ValueError("passage and query dimensions do not match")
    baseline = docs @ q
    scores = dict(zip(ids, map(float, baseline))) if base_scores is None else dict(base_scores)
    if set(scores) != set(ids) or not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("base scores must be finite and match passage ids")
    changes = docs @ (apply(q, u, v) - q)
    for index, key in enumerate(ids):
        if key in shown:
            change = min(0., float(changes[index])) if key in adversarial else float(changes[index])
            scores[key] = float(np.clip(scores[key] + change, baseline[index] - 2 * bound,
                                         baseline[index] + 2 * bound))
    return scores


def _validate(metadata, u, v) -> None:
    if not isinstance(metadata, dict) or metadata.get("schema") != SCHEMA:
        raise ValueError("unknown revision schema")
    dims, rank = metadata.get("dims"), metadata.get("rank")
    if type(dims) is not int or type(rank) is not int or dims < 1 or rank < 1:
        raise ValueError("invalid dims or rank")
    if u.dtype != np.dtype("float32") or v.dtype != np.dtype("float32"):
        raise ValueError("revision factors must be float32")
    if u.shape != (dims, rank) or v.shape != u.shape or not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError("invalid revision factors")
    bound = metadata.get("operator_norm_bound")
    if type(bound) not in (int, float) or not math.isfinite(bound) or not 0 < bound <= .05:
        raise ValueError("invalid operator-norm bound")
    if spectral(u, v) > bound * (1 + 1e-9):
        raise ValueError("revision exceeds operator-norm bound")
    if metadata.get("scope", "global") not in ("global", "shown"):
        raise ValueError("unknown revision scope")
    if not isinstance(metadata.get("document_identity"), dict) or not metadata["document_identity"]:
        raise ValueError("missing document identity")
    if metadata.get("digest") != weights_digest(u, v):
        raise ValueError("revision digest mismatch")
    json.dumps(metadata, allow_nan=False)


def save_revision(path, u, v, *, bound, document_identity, scope="global") -> dict:
    """Write an immutable directory in the portable query-adapter-1 format."""
    u, v = (np.asarray(factor, dtype=np.float32, order="C") for factor in (u, v))
    if u.ndim != 2:
        raise ValueError("revision factors must be matrices")
    metadata = {"schema": SCHEMA, "dims": u.shape[0], "rank": u.shape[1],
                "operator_norm_bound": bound, "document_identity": document_identity,
                "digest": weights_digest(u, v), "scope": scope}
    _validate(metadata, u, v)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("revision directories are immutable")
    temporary = Path(tempfile.mkdtemp(prefix=".revision-", dir=path.parent))
    try:
        for name, factor in (("U.npy", u), ("V.npy", v)):
            with (temporary / name).open("wb") as handle:
                np.save(handle, factor, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
        with (temporary / "revision.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.rename(path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return metadata


def load_revision(path, *, document_identity=None) -> Revision:
    path = Path(path)
    metadata = json.loads((path / "revision.json").read_text(encoding="utf-8"))
    u, v = (np.load(path / name, allow_pickle=False) for name in ("U.npy", "V.npy"))
    _validate(metadata, u, v)
    if document_identity is not None and metadata["document_identity"] != document_identity:
        raise ValueError("document identity mismatch")
    u.flags.writeable = v.flags.writeable = False
    return Revision(metadata, u, v)


@dataclass(frozen=True)
class _Row:
    original: TrainingRow
    query: np.ndarray
    passages: Mapping[str, np.ndarray]
    preferred: str
    disfavoured: str
    shown: Mapping[str, int]
    adversarial: frozenset[str]


class QueryAdapterLearner:
    """Rebuild learner with contrastive training and a normalized squared arm.

    Rank 32, bound .05 and temperature .05 are the design defaults. Twenty steps,
    learning rate .1, a .01 random initial factor, a factor norm cap, and sixteen
    backtracking attempts are local training assumptions. Every fit starts from
    the supplied state and sorts rows canonically; there is no hidden optimizer
    state. ``base`` uses a fixed seed, so erasure rebuilds are reproducible.
    """

    name = "query_adapter"

    def __init__(self, dims: int, *, rank: int = 32, learning_rate: float = .1, steps: int = 20,
                 bound: float = .05, temperature: float = .05, arm: str = "contrastive"):
        if type(dims) is not int or dims < 1 or type(rank) is not int or rank < 1:
            raise ValueError("dims and rank must be positive integers")
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
               for value in (learning_rate, bound, temperature)) or bound > .05:
            raise ValueError("training settings must be positive; bound must be at most 0.05")
        if arm not in ("contrastive", "normalised_squared_error"):
            raise ValueError("unknown query adapter objective arm")
        self.dims, self.rank, self.steps = dims, rank, steps
        self.learning_rate, self.bound, self.temperature, self.arm = learning_rate, bound, temperature, arm

    def base(self) -> dict[str, np.ndarray]:
        random = np.random.default_rng(0)
        v = random.normal(0., .01, (self.dims, self.rank)).astype(np.float32)
        v *= min(1., math.sqrt(self.rank * self.bound) / max(float(np.linalg.norm(v)), 1e-30))
        return {"U": np.zeros_like(v), "V": v, "shown_U": np.zeros_like(v), "shown_V": v.copy()}

    def _state(self, state):
        if set(state) != {"U", "V", "shown_U", "shown_V"}:
            raise ValueError("query adapter state requires global and shown factors")
        copied = {key: np.array(value, dtype=np.float32, copy=True) for key, value in state.items()}
        if any(value.shape != (self.dims, self.rank) or not np.isfinite(value).all() for value in copied.values()):
            raise ValueError("state dimensions or values do not match learner")
        if self.budget_used(copied) > self.bound * (1 + 1e-9):
            raise ValueError("state exceeds operator-norm bound")
        return copied

    def apply(self, state, query) -> np.ndarray:
        return apply(query, state["U"], state["V"])

    def rank_scores(self, state, query, passages, *, shown, adversarial=()) -> dict[str, float]:
        if not passages:
            return {}
        ids = list(passages)
        scores = normalize([passages[key] for key in ids]) @ self.apply(state, query)
        return _rank_scores(query, passages, state["shown_U"], state["shown_V"], shown=shown,
                            adversarial=adversarial, base_scores=dict(zip(ids, map(float, scores))), bound=self.bound)

    def budget_used(self, state) -> float:
        return max(spectral(state["U"], state["V"]), spectral(state["shown_U"], state["shown_V"]))

    def _rows(self, rows, objective):
        result = []
        for row in sorted(rows, key=lambda item: (item.ref, item.teacher)):
            if row.teacher not in objective.losses:
                raise ValueError("training row teacher is not declared")
            if row.audience != "owner":
                continue
            payload = json.loads(row.state)
            q = normalize(payload["query"])
            passages = {key: normalize(value) for key, value in sorted(payload["passages"].items())}
            if q.shape != (self.dims,) or not passages or any(
                    not isinstance(key, str) or not key or value.shape != q.shape for key, value in passages.items()):
                raise ValueError("training query and passage dimensions must match learner")
            label = row.label
            preferred, disfavoured = label["preferred"], label["disfavoured"]
            shown, sources = label["shown"], label["passage_sources"]
            if preferred not in passages or disfavoured not in passages or preferred == disfavoured:
                raise ValueError("targets must name two supplied passages")
            if not isinstance(shown, Mapping) or not set(shown) <= set(passages) or any(
                    type(rank) is not int or rank < 1 for rank in shown.values()):
                raise ValueError("shown ranks must name supplied passages")
            allowed_sources = {"owner", "system", "synthetic", *ADVERSARIAL_SOURCE_KINDS}
            if not isinstance(sources, Mapping) or any(
                    sources.get(key) not in allowed_sources for key in (preferred, disfavoured)):
                raise ValueError("target provenance must name known source kinds")
            adversarial = frozenset(key for key in passages if sources.get(key) in ADVERSARIAL_SOURCE_KINDS)
            result.append(_Row(row, q.astype(np.float64), passages, preferred, disfavoured, shown, adversarial))
        return result

    def _term(self, item, u, v, pool):
        row = item.original
        model = row.teacher in MODEL_TEACHERS
        push = row.push_away or row.teacher == "behavioural_proxy"
        if model and (item.disfavoured not in item.shown or (not push and item.preferred not in item.shown)):
            return 0., np.zeros(self.dims)
        if not push and item.preferred in item.adversarial and row.teacher not in PROOF_TEACHERS:
            return 0., np.zeros(self.dims)
        raw = item.query + u @ (v.T @ item.query)
        norm = float(np.linalg.norm(raw))
        prediction = raw / norm
        if push:
            target = item.passages[item.disfavoured].astype(np.float64)
            loss, derivative = (float(prediction @ target) + 1) / 2, target / 2
        elif model:
            difference = item.passages[item.preferred] - item.passages[item.disfavoured]
            margin = float(prediction @ difference) / self.temperature
            loss = float(np.logaddexp(0., -margin))
            derivative = -math.exp(-float(np.logaddexp(0., margin))) * difference / self.temperature
        elif self.arm == "normalised_squared_error":
            target = item.passages[item.preferred]
            error = raw - target
            denominator = float(target @ target) + 1e-6 * self.dims
            return float(error @ error) / denominator, 2 * error / denominator
        else:
            docs = np.asarray(list(pool.values()), dtype=np.float64)
            target = item.passages[item.preferred]
            logits = docs @ prediction / self.temperature
            largest = float(np.max(logits))
            exponents = np.exp(logits - largest)
            total = float(np.sum(exponents))
            loss = largest + math.log(total) - float(prediction @ target) / self.temperature
            derivative = ((exponents / total) @ docs - target) / self.temperature
        gradient = (derivative - prediction * float(prediction @ derivative)) / norm
        return max(0., loss), gradient

    def _pool(self, rows):
        pool = {}
        for item in rows:
            for key, value in item.passages.items():
                if key in pool and not np.array_equal(pool[key], value):
                    raise ValueError("one passage id has conflicting vectors")
                pool[key] = value
        return dict(sorted(pool.items()))

    def _step(self, state, rows, teacher, blend):
        prefix = "shown_" if teacher in MODEL_TEACHERS else ""
        u, v = (state[prefix + name].astype(np.float64) for name in ("U", "V"))
        pool = self._pool(rows)
        gu, gv = np.zeros_like(u), np.zeros_like(v)
        previous_loss = 0.
        for item in rows:
            loss, gradient = self._term(item, u, v, pool)
            weight = blend * item.original.rank_weight / len(rows)
            previous_loss += weight * loss
            gu += weight * np.outer(gradient, v.T @ item.query)
            gv += weight * np.outer(item.query, u.T @ gradient)
        gradient_norm = math.sqrt(float(np.sum(gu * gu) + np.sum(gv * gv)))
        if not math.isfinite(gradient_norm):
            raise ValueError("nonfinite adapter gradient")
        if gradient_norm == 0:
            return
        rate = self.learning_rate / max(1., gradient_norm)
        cap = math.sqrt(self.rank * self.bound)
        for attempt in range(16):
            factors = [u - rate * .5 ** attempt * gu, v - rate * .5 ** attempt * gv]
            factors = [factor * min(1., cap / max(float(np.linalg.norm(factor)), 1e-30)) for factor in factors]
            new_u, new_v = project_factors(*factors, bound=self.bound)
            displacement = math.sqrt(float(np.sum((new_u - u) ** 2) + np.sum((new_v - v) ** 2)))
            if displacement > self.learning_rate * (1 + 1e-6):
                continue
            proposed = {**state, prefix + "U": new_u, prefix + "V": new_v}
            loss = sum(blend * item.original.rank_weight * self._term(item, new_u, new_v, pool)[0]
                       for item in rows) / len(rows)
            if loss > previous_loss + 1e-12:
                continue
            safe = True
            for item in rows:
                protected = set(item.adversarial) if teacher not in PROOF_TEACHERS else set()
                if item.original.push_away or teacher == "behavioural_proxy":
                    protected.add(item.disfavoured)
                if not protected:
                    continue
                before = self.rank_scores(state, item.query, item.passages, shown=item.shown,
                                          adversarial=item.adversarial)
                after = self.rank_scores(proposed, item.query, item.passages, shown=item.shown,
                                         adversarial=item.adversarial)
                if any(after[key] > before[key] for key in protected):
                    safe = False
                    break
            if safe:
                state.update({prefix + "U": new_u, prefix + "V": new_v})
                return

    def fit(self, start: Mapping[str, np.ndarray], rows: Sequence[TrainingRow], *, objective: Objective,
            seed: int) -> dict[str, np.ndarray]:
        # Initialization is fixed in base(); no stochastic updates consume seed.
        state = self._state(start)
        parsed = self._rows(rows, objective)
        groups = {teacher: [item for item in parsed if item.original.teacher == teacher]
                  for teacher in sorted(objective.losses)}
        for _ in range(self.steps):
            for teacher, items in groups.items():
                if items and objective.blend[teacher] > 0:
                    self._step(state, items, teacher, objective.blend[teacher])
        return state

    def losses(self, state: Mapping[str, np.ndarray], rows: Sequence[TrainingRow], *,
               objective: Objective) -> Mapping[str, float]:
        state = self._state(state)
        parsed = self._rows(rows, objective)
        terms = {}
        for teacher in sorted(objective.losses):
            items = [item for item in parsed if item.original.teacher == teacher]
            prefix = "shown_" if teacher in MODEL_TEACHERS else ""
            pool = self._pool(items)
            terms[teacher] = sum(item.original.rank_weight * self._term(
                item, state[prefix + "U"], state[prefix + "V"], pool)[0] for item in items) / max(1, len(items))
        return terms
