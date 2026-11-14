"""CPU numeric predictors and the shared numeric-revision-1 wire format.

Histories are newest first. Value updates use normalised squared error, a unit
gradient cap and a per-entry envelope around the period's starting weights.
The lifetime fence always refers to the original persistence or constant anchor.
Defaults are plan assumptions: rate/clip .01, value fraction .5, logistic change
2 log-odds, input range [-3, 3], time range [-1, 1], 30 warm-up errors.
No external engine or GPU package is imported.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from . import _io

SCHEMA = "numeric-revision-1"
TIME_FEATURES = ("hod_sin", "hod_cos", "how_sin", "how_cos")


def moment(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("numeric time needs a zone")
        return value.astimezone(timezone.utc)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("numeric time must be finite")
    return datetime.fromtimestamp(value, timezone.utc)


def _real(value):
    return type(value) in (int, float) and math.isfinite(value)


def weights_digest(weights):
    return hashlib.sha256(np.ascontiguousarray(weights).tobytes()).hexdigest()


class Revision:
    """Validated immutable metadata and float32 weights, independent of a learner."""

    def __init__(self, metadata, weights):
        m = copy.deepcopy(metadata)
        if m.get("schema") != SCHEMA or not isinstance(m.get("learner"), str) or not m["learner"]:
            raise ValueError("numeric schema and learner required")
        if m.get("variant") not in ("value", "logistic"):
            raise ValueError("unknown numeric variant")
        self.zone = ZoneInfo(m.get("tz", "UTC"))
        features, outputs = m.get("features"), m.get("outputs")
        if not isinstance(features, list) or not features or not isinstance(outputs, list) or not outputs:
            raise ValueError("ordered features and outputs required")
        standard = m.get("standardisation")
        if not isinstance(standard, dict):
            raise ValueError("standardisation required")
        for entry in standard.values():
            if (not isinstance(entry, dict) or set(entry) != {"mean", "scale"}
                    or not all(_real(v) for v in entry.values()) or entry["scale"] <= 0):
                raise ValueError("standardisation needs finite mean and positive scale")
        for descriptor in features:
            if not isinstance(descriptor, dict):
                raise ValueError("invalid feature descriptor")
            keys = set(descriptor)
            if keys == {"time"} and descriptor["time"] in TIME_FEATURES:
                continue
            if keys == {"const"} and type(descriptor["const"]) is int and descriptor["const"] == 1:
                continue
            if keys == {"stream", "lag"}:
                if type(descriptor["lag"]) is not int or descriptor["lag"] < 1:
                    raise ValueError("lag must be a positive integer")
                name = descriptor["stream"]
            elif keys == {"input"}:
                name = descriptor["input"]
            else:
                raise ValueError("invalid feature descriptor")
            if not isinstance(name, str) or not name or name not in standard:
                raise ValueError("numeric feature needs standardisation")
        kind = "stream" if m["variant"] == "value" else "event"
        if any(not isinstance(out, dict) or set(out) != {kind} or not isinstance(out[kind], str)
               or not out[kind] for out in outputs):
            raise ValueError("invalid output descriptor")
        if len({out[kind] for out in outputs}) != len(outputs):
            raise ValueError("duplicate numeric output")
        anchor = m.get("anchor", {})
        if m["variant"] == "value":
            if anchor != {"type": "persistence"} or any(
                    out["stream"] not in standard or {"stream": out["stream"], "lag": 1} not in features
                    for out in outputs):
                raise ValueError("persistence needs every output's lag-1 feature")
        elif (len(outputs) != 1 or set(anchor) != {"type", "p"} or anchor["type"] != "constant"
              or not _real(anchor["p"]) or not 0 < anchor["p"] < 1):
            raise ValueError("constant anchor needs a probability between zero and one")
        w = np.asarray(weights)
        if (w.dtype != np.float32 or not w.flags.c_contiguous or w.shape != (len(outputs), len(features))
                or not np.isfinite(w).all()):
            raise ValueError("weights must be finite C-order float32 (outputs, features)")
        digest = weights_digest(w)
        if m.get("weights_digest", digest) != digest:
            raise ValueError("numeric weights digest mismatch")
        m["weights_digest"] = digest
        self._metadata = m
        self.weights = w.copy()
        self.weights.flags.writeable = False
        self._persistence = False
        if m["variant"] == "value":
            identity = np.zeros_like(w)
            for i, out in enumerate(outputs):
                identity[i, features.index({"stream": out["stream"], "lag": 1})] = 1.
            self._persistence = bool(np.array_equal(w, identity))

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    def features(self, *, at, streams=None, inputs=None):
        streams, inputs = streams or {}, inputs or {}
        local = moment(at).astimezone(self.zone)
        hour = local.hour + local.minute / 60 + (local.second + local.microsecond / 1e6) / 3600
        week = local.weekday() * 24 + hour
        result = []
        for descriptor in self._metadata["features"]:
            if "time" in descriptor:
                name = descriptor["time"]
                phase = 2 * math.pi * (hour / 24 if name.startswith("hod") else week / 168)
                result.append(math.sin(phase) if name.endswith("sin") else math.cos(phase))
            elif "const" in descriptor:
                result.append(1.)
            else:
                try:
                    if "stream" in descriptor:
                        name = descriptor["stream"]
                        value = streams[name][descriptor["lag"] - 1]
                    else:
                        name = descriptor["input"]
                        value = inputs[name]
                    if isinstance(value, (str, bool)):
                        raise ValueError("numeric feature must be a number")
                    standard = self._metadata["standardisation"][name]
                    result.append((float(value) - standard["mean"]) / standard["scale"])
                except (KeyError, IndexError, TypeError) as error:
                    raise ValueError("missing numeric feature") from error
        with np.errstate(over="ignore"):
            vector = np.asarray(result, dtype=np.float32)
        if not np.isfinite(vector).all():
            raise ValueError("numeric features must be finite")
        return vector

    def forecast(self, *, at, streams=None, inputs=None):
        m = self._metadata
        x = self.features(at=at, streams=streams, inputs=inputs)
        if self._persistence:
            return {out["stream"]: float(streams[out["stream"]][0]) for out in m["outputs"]}
        with np.errstate(over="ignore"):
            y = self.weights @ x
        if not np.isfinite(y).all():
            raise ValueError("nonfinite numeric forecast")
        if m["variant"] == "logistic":
            p = m["anchor"]["p"]
            # Preserve the exact declared baseline at W=0.
            value = p if y[0] == 0 else _sigmoid(math.log(p / (1 - p)) + float(y[0]))
            return {m["outputs"][0]["event"]: value}
        result = {out["stream"]: float(y[i]) * m["standardisation"][out["stream"]]["scale"]
                  + m["standardisation"][out["stream"]]["mean"] for i, out in enumerate(m["outputs"])}
        if not all(math.isfinite(v) for v in result.values()):
            raise ValueError("nonfinite numeric forecast")
        return result


def _sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    e = math.exp(value)
    return e / (1 + e)


def load_revision(path):
    path = Path(path)
    metadata = _io.read_json(path / "revision.json")
    if "weights_digest" not in metadata:
        raise ValueError("numeric weights digest required")
    return Revision(metadata, np.load(path / "W.npy", allow_pickle=False))


def save_revision(path, metadata, weights):
    """Publish a complete immutable wire directory. Never overwrite an existing revision."""
    m = copy.deepcopy(metadata)
    m.pop("weights_digest", None)
    revision = Revision(m, np.ascontiguousarray(weights, dtype=np.float32))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("numeric revision already exists")
    pending = Path(tempfile.mkdtemp(prefix=".numeric-", dir=path.parent))
    try:
        buffer = io.BytesIO()
        np.save(buffer, revision.weights, allow_pickle=False)
        _io.write_atomic(pending / "W.npy", buffer.getvalue())
        _io.write_json(pending / "revision.json", revision.metadata)
        os.rename(pending, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if pending.exists():
            shutil.rmtree(pending)
    return revision.metadata


def bounded_step(weight, anchor, x, target, *, rate=.01, clip=.01, probability=None):
    """One float32 step, with the value rule's operation order preserved."""
    if not _real(rate) or rate < 0 or not _real(clip) or clip < 0:
        raise ValueError("rate and clip must be finite and nonnegative")
    w, base, x, y = (np.asarray(value, dtype=np.float32) for value in (weight, anchor, x, target))
    x, y = np.atleast_2d(x), np.atleast_2d(y)
    n = min(len(x), len(y))
    if not n or w.ndim != 2 or w.shape != base.shape or x.shape[1] != w.shape[1] or y.shape[1] != w.shape[0]:
        raise ValueError("incompatible numeric step shapes")
    if not all(np.isfinite(value).all() for value in (w, base, x, y)):
        raise ValueError("nonfinite numeric step")
    x, y = x[:n], y[:n]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if probability is None:
            error = x @ w.T - y
            scale = np.mean(y * y) + np.float32(1e-6)
            gradient = np.float32(2. / (n * y.shape[1])) * (error.T @ x) / scale
        else:
            if not 0 < probability < 1 or w.shape[0] != 1 or np.any((y < 0) | (y > 1)):
                raise ValueError("logistic step needs probabilities")
            logits = x @ w.T + np.float32(math.log(probability / (1 - probability)))
            predicted = np.array([_sigmoid(float(v)) for v in logits[:, 0]], dtype=np.float32)[:, None]
            gradient = ((predicted - y).T @ x) / np.float32(n)
        norm = np.sqrt(np.sum(gradient * gradient))
        step = np.float32(rate) / np.maximum(norm, np.float32(1.))
        updated = np.clip(w - step * gradient, base - np.float32(clip), base + np.float32(clip))
    if not np.isfinite(gradient).all() or not np.isfinite(norm) or not np.isfinite(updated).all():
        raise ValueError("nonfinite numeric update")
    return updated


class NumericLearner:
    """P1a learner over JSON numeric frames and scalar or ordered numeric targets.

    Frames include ``at``, ``streams``/``inputs`` and optional ``previous_at``,
    ``gap``, ``actuation``, ``alarm``, ``safety``. Safety values are headroom:
    learning waits until each declared stream is at least twice its danger line.
    The default gap of 600 seconds is a builder assumption for five-minute samples.
    """
    budget_type = "anchored_output"

    def __init__(self, metadata, *, rate=.01, clip=.01, bound=None, max_gap=600., safety=None):
        m = copy.deepcopy(metadata)
        m.pop("weights_digest", None)
        w = np.zeros((len(m["outputs"]), len(m["features"])), dtype=np.float32)
        if m["variant"] == "value":
            for i, out in enumerate(m["outputs"]):
                try:
                    w[i, m["features"].index({"stream": out["stream"], "lag": 1})] = 1.
                except ValueError:
                    raise ValueError("persistence needs an output lag-1 feature") from None
        origin = Revision(m, w)
        self.metadata = m
        self.name, self.variant = m["learner"], m["variant"]
        self.rate, self.clip = rate, clip
        self.bound = (.5 if self.variant == "value" else 2.) if bound is None else bound
        self.max_gap, self.safety = max_gap, dict(safety or {})
        if any(not _real(v) or v < 0 for v in (rate, clip, self.bound, max_gap)) or max_gap == 0:
            raise ValueError("numeric learning settings must be finite and nonnegative")
        if any(not _real(v) or v < 0 for v in self.safety.values()):
            raise ValueError("safety danger lines must be finite and nonnegative")
        self._origin = origin.weights.copy()

    @property
    def identity(self):
        return _io.digest(dict(metadata=self.metadata, rate=self.rate, clip=self.clip, bound=self.bound,
                               max_gap=self.max_gap, safety=self.safety))

    def base(self):
        return {"W": self._origin.copy()}

    def revision(self, state):
        return Revision(self.metadata, state["W"] if state else self._origin)

    def budget_used(self, state):
        delta = np.asarray(state["W"], dtype=np.float64) - self._origin
        if self.variant == "value":
            return float(np.linalg.norm(delta) / np.linalg.norm(self._origin))
        radii = [1. if "time" in f or "const" in f else 3. for f in self.metadata["features"]]
        # Constant columns contribute their joint offset, not independent signs.
        constant = sum(float(delta[0, i]) for i, f in enumerate(self.metadata["features"]) if "const" in f)
        return abs(constant) + sum(abs(float(delta[0, i])) * r for i, (f, r) in enumerate(
            zip(self.metadata["features"], radii)) if "const" not in f)

    def _project(self, w, previous):
        used = self.budget_used({"W": w})
        if used > self.bound:
            # Both fences are convex. Shorten the actual step from its feasible
            # starting point, keeping the period clip and the gradient cap too.
            before = previous["W"]
            if self.budget_used(previous) > self.bound:
                raise ValueError("numeric starting state exceeds its lifetime fence")
            low, high = 0., 1.
            feasible = before.copy()
            for _ in range(32):  # Float32 bisection resolution, builder assumption.
                middle = (low + high) / 2
                trial = before + np.float32(middle) * (w - before)
                if self.budget_used({"W": trial}) <= self.bound:
                    low, feasible = middle, trial
                else:
                    high = middle
            w = feasible
        return {"W": np.ascontiguousarray(w, dtype=np.float32)}

    def eligible(self, frame):
        if any(frame.get(flag, False) for flag in ("gap", "actuation", "alarm")):
            return False
        if "previous_at" in frame:
            gap = (moment(frame["at"]) - moment(frame["previous_at"])).total_seconds()
            if not 0 < gap <= self.max_gap:
                return False
        observed = frame.get("safety", {})
        return all(_real(observed.get(name)) and observed[name] >= 2 * line
                   for name, line in self.safety.items())

    def target(self, label):
        key = "stream" if self.variant == "value" else "event"
        values = [label[out[key]] for out in self.metadata["outputs"]] if isinstance(label, dict) else label
        y = np.asarray(values, dtype=np.float32).reshape(-1)
        if len(y) != len(self.metadata["outputs"]) or not np.isfinite(y).all():
            raise ValueError("target needs one finite value per output")
        if self.variant == "value":
            y = np.array([(v - self.metadata["standardisation"][out["stream"]]["mean"])
                          / self.metadata["standardisation"][out["stream"]]["scale"]
                          for v, out in zip(y, self.metadata["outputs"])], dtype=np.float32)
        elif np.any((y < 0) | (y > 1)):
            raise ValueError("logistic targets must be probabilities")
        return y

    def step(self, state, anchor, frame, label, *, weight=1.):
        if not self.eligible(frame) or weight == 0:
            return {"W": np.array(state["W"], copy=True)}
        x = self.revision(state).features(**{k: frame[k] for k in ("at", "streams", "inputs") if k in frame})
        w = bounded_step(state["W"], anchor["W"], x, self.target(label), rate=self.rate * weight,
                         clip=self.clip, probability=self.metadata["anchor"].get("p"))
        return self._project(w, state)

    def fit(self, start, rows, *, objective, seed):
        del seed
        anchor = {"W": self.revision(start).weights.copy()}
        state = {"W": anchor["W"].copy()}
        ordered = sorted(((json.loads(row.state), row) for row in rows),
                         key=lambda pair: (moment(pair[0]["at"]), pair[1].ref, pair[1].teacher))
        for frame, row in ordered:
            if row.push_away:
                raise ValueError("numeric targets require a pulling proof teacher")
            if objective.blend[row.teacher] > 0:
                state = self.step(state, anchor, frame, row.label, weight=row.weight)
        return state

    def predict(self, state, states):
        revision = self.revision(state)
        predictions = []
        for frame in states:
            frame = json.loads(frame) if isinstance(frame, str) else frame
            predictions.append(list(revision.forecast(**{k: frame[k] for k in ("at", "streams", "inputs")
                                                        if k in frame}).values()))
        return np.asarray(predictions).reshape(-1, len(self.metadata["outputs"]))

    def losses(self, state, rows, *, objective):
        result = {}
        for teacher in objective.losses:
            selected = [row for row in rows if row.teacher == teacher]
            if not selected:
                result[teacher] = 0.
                continue
            p = self.predict(state, [row.state for row in selected])
            y = np.array([self.target(row.label) for row in selected])
            if self.variant == "value":
                scales = np.array([self.metadata["standardisation"][o["stream"]]["scale"]
                                   for o in self.metadata["outputs"]])
                means = np.array([self.metadata["standardisation"][o["stream"]]["mean"]
                                  for o in self.metadata["outputs"]])
                error = (p - means) / scales - y
                result[teacher] = float(np.mean(error * error))
            else:
                p = np.clip(p, 1e-12, 1 - 1e-12)  # Numerical logarithm floor, builder assumption.
                result[teacher] = float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p)))
        return result

    def exposure(self, state, canaries):
        """Invalid/missing numeric inputs must remain refusals in every revision."""
        exposed = {}
        for item in canaries:
            try:
                self.predict(state, [item.state])
            except (ValueError, KeyError, TypeError):
                exposed[item.id] = 0.
            else:
                exposed[item.id] = 1.
        return exposed

    def arms(self, train, test, *, start=None, objective, seed=0):
        """Score controls on the same test inputs, with seasonal targets strictly in the past."""
        states = [row.state for row in test]
        fitted = self.fit(self.base(), train, objective=objective, seed=seed)
        order = np.random.default_rng(seed).permutation(len(train))
        shuffled = [replace(row, label=train[int(i)].label) for row, i in zip(train, order)]
        control = self.fit(self.base(), shuffled, objective=objective, seed=seed)
        result = {"learned": self.predict(start or fitted, states),
                  "persistence": self.predict(self.base(), states),
                  "time_shuffled": self.predict(control, states), "no_accumulation": self.predict(fitted, states)}
        rev = self.revision(self.base())
        x = np.asarray([rev.features(**{k: f[k] for k in ("at", "streams", "inputs") if k in f})
                        for f in (json.loads(row.state) for row in train)])
        y = np.array([self.target(row.label) for row in train])
        if self.variant == "logistic":
            y = np.log(np.clip(y, .01, .99) / np.clip(1 - y, .01, .99)) - math.log(
                self.metadata["anchor"]["p"] / (1 - self.metadata["anchor"]["p"]))
        # Ridge regularisation .01 is an offline control assumption, never a serving revision.
        ridge = np.linalg.solve(x.T @ x + .01 * np.eye(x.shape[1]), x.T @ y).T.astype(np.float32)
        result["ridge"] = self.predict({"W": ridge}, states)
        history = {moment(json.loads(row.state)["at"]): row.label for row in train}
        for name, days in (("seasonal_day", 1), ("seasonal_week", 7)):
            past = dict(history)
            forecasts = []
            for i, row in enumerate(test):
                at = moment(json.loads(row.state)["at"])
                target = past.get(at - timedelta(days=days))
                forecasts.append(result["persistence"][i] if target is None else np.asarray(target).reshape(-1))
                past[at] = row.label
            result[name] = np.asarray(forecasts)
        return result


def interval(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("metrics need finite observations")
    mean = float(values.mean())
    width = 1.959963984540054 * float(values.std(ddof=1)) / math.sqrt(len(values)) if len(values) > 1 else 0.
    return {"mean": mean, "low": mean - width, "high": mean + width, "n": len(values)}


def metrics(predicted, observed, *, variant="value", reference=None, seasonal=None):
    p, y = np.asarray(predicted, dtype=float), np.asarray(observed, dtype=float)
    if p.shape != y.shape:
        raise ValueError("metrics need paired outputs")
    squared = (p - y) ** 2
    if variant == "logistic":
        if np.any((p < 0) | (p > 1) | (y < 0) | (y > 1)):
            raise ValueError("logistic metrics need probabilities")
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return {"brier": interval(squared), "log_loss": interval(-y * np.log(p) - (1 - y) * np.log1p(-p))}
    result = {"squared_error": interval(squared)}
    for name, baseline in (("persistence", reference), ("seasonal", seasonal)):
        if baseline is not None:
            denominator = max(float(np.mean((np.asarray(baseline) - y) ** 2)), 1e-12)
            result["scaled_" + name] = interval(squared / denominator)
    return result


class Surprise:
    """Score against prior mean absolute error. Floor .01 is a builder assumption."""
    def __init__(self, *, floor=.01, count=0, total=0.):
        if not _real(floor) or floor <= 0:
            raise ValueError("surprise floor must be positive")
        self.floor, self.count, self.total = floor, count, total

    def score(self, error):
        error = abs(float(error))
        if not math.isfinite(error):
            raise ValueError("surprise error must be finite")
        result = error / max(self.floor, self.total / max(self.count, 1)) if self.count >= 30 else None
        self.count += 1
        self.total += error
        return result
