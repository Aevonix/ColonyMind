"""Paired numeric losses, daily reference-only logs and automatic rollback.

Plan assumptions: newest 200 labelled items within 168 hours, .02 relative
excess-loss bar, at least 30 pairs, one-sided 95% normal bound, 14-day retention.
Relative excess is the mean paired loss difference divided by reference mean
loss; its interval uses the same fixed denominator. A 1e-12 denominator floor
is a numerical assumption for a perfect reference.
"""
from __future__ import annotations

import json
import math
import os
from datetime import timedelta

import numpy as np

from . import _io
from .audience import may_serve
from .comparator import LiveComparator
from .revisions import Pinned

PROOF_TEACHERS = frozenset(("realized_value", "verified_outcome", "owner_label"))
DEFAULTS = {"window_items": 200, "window_hours": 168., "bar": .02}


def _settings(declaration, *, variant="value", scales=1.):
    options = declaration.get("comparator", {})
    if not isinstance(options, dict) or set(options) - set(DEFAULTS):
        raise ValueError("comparator: declare window_items, window_hours and bar")
    settings = {**DEFAULTS, **options, "variant": variant, "scales": _numbers(scales)}
    if type(settings["window_items"]) is not int or settings["window_items"] < 30:
        raise ValueError("comparator: window_items must be at least 30")
    for name in ("window_hours", "bar"):
        value = settings[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("comparator: bounds must be finite and nonnegative")
    if not settings["window_hours"] or variant not in ("value", "logistic") or np.any(
            np.asarray(settings["scales"]) <= 0):
        raise ValueError("comparator: positive window and scales and known variant required")
    return settings


def configure(controller, declaration, *, variant="value", scales=1.):
    """Factory hook, before Runtime wires the comparator. Settings persist for reloads.

    Call this for every numeric implementation. Value scales follow output order;
    callers already serving standardised values use 1. Logistic uses Brier loss.
    """
    settings = _settings(declaration, variant=variant, scales=scales)
    _io.write_json(controller.store.directory / "numeric-comparator.json", settings)
    return settings


def _numbers(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) in (int, float) and math.isfinite(value):
        return float(value)
    if isinstance(value, (tuple, list)) and value and all(type(v) in (int, float) and math.isfinite(v) for v in value):
        return [float(v) for v in value]
    raise ValueError("numeric comparator outputs must be finite numbers")


class NumericComparator(LiveComparator):
    """LiveComparator's constructor and pin/rollback contract, with numeric outcomes."""

    def __init__(self, controller, *, key, bar=.02, clock=None, lock=None, pin=None, rollback=None):
        super().__init__(controller, key=key, bar=bar, clock=clock, lock=lock, pin=pin, rollback=rollback)
        try:
            settings = _io.read_json(self.store.directory / "numeric-comparator.json")
        except FileNotFoundError:
            settings = {**DEFAULTS, "bar": bar, "variant": "value", "scales": 1.}
        settings = _settings({"comparator": {k: settings[k] for k in DEFAULTS}},
                             variant=settings["variant"], scales=settings["scales"])
        self.bar = settings["bar"]
        self.window_items, self.window_hours = settings["window_items"], settings["window_hours"]
        self.variant, self.scales = settings["variant"], settings["scales"]

    def _daily_path(self, at):
        return self.directory / ("numeric-" + _io.parse(at).date().isoformat() + ".jsonl")

    def _rewrite(self, path, rows):
        if rows:
            _io.write_atomic(path, b"".join(_io.canonical(row) + b"\n" for row in rows))
        else:
            _io.remove(path)

    def _erased(self):
        try:
            return set(_io.read_json(self.store.root / "erased-refs.json"))
        except FileNotFoundError:
            return set()

    def records(self):
        with self.lock:
            cutoff = self._clock() - timedelta(days=14)
            erased = self._erased()
            found = []
            for path in sorted(self.directory.glob("numeric-*.jsonl")):
                raw = path.read_bytes().splitlines(keepends=True)
                # An interrupted append cannot invent a served item or outcome.
                complete = [json.loads(line) for line in raw if line.endswith(b"\n")]
                removed = any(row.get("erasure_hash") in erased for row in complete)
                kept = [row for row in complete if _io.parse(row["at"]) >= cutoff
                        and row.get("erasure_hash") not in erased]
                if len(kept) != len(raw):
                    self._rewrite(path, kept)
                if removed:
                    _io.remove(self.directory / "last-rollback.json")
                found.extend(kept)
            records = sorted(found, key=lambda row: (row["at"], row["seq"]))
            current = getattr(self, 'current_records', None)
            return current(records) if current else records

    def _output_numbers(self, output):
        value = _numbers(output)
        if self.variant == "logistic" and np.any((np.asarray(value) < 0) | (np.asarray(value) > 1)):
            raise ValueError("logistic comparator outputs must be probabilities")
        return value

    def serve(self, ref, inputs, predict, *, requester="owner"):
        with self.lock:
            self.summary()
            hashed = self._path(ref).stem
            erasure_hash = self._hash(["erased", ref])
            if erasure_hash in self._erased():
                raise ValueError("an erased numeric reference cannot serve again")
            records = self.records()
            if any(row["ref_hash"] == hashed for row in records):
                raise ValueError("reference already has a numeric comparator item")
            serving = self._pin(requester)
            previous = self.controller.previous(serving.closure.id)
            if not may_serve(previous.audience, requester):
                previous = self.store.revisions.base
            reference = Pinned(previous, self.store.revisions.state(previous))
            fingerprint = self._hash(["input", inputs])
            output = predict(serving, inputs)
            if self._hash(["input", inputs]) != fingerprint:
                raise ValueError("paired predictors must leave inputs unchanged")
            before = self._output_numbers(output)
            baseline = self._output_numbers(predict(reference, inputs))
            if self._hash(["input", inputs]) != fingerprint:
                raise ValueError("paired predictors must leave inputs unchanged")
            if np.shape(before) != np.shape(baseline):
                raise ValueError("numeric comparator outputs must have the same shape")
            if np.shape(self.scales) not in ((), np.shape(before)):
                raise ValueError("numeric scales must match output order")
            at = _io.stamp(self._clock())
            record = {"ref_hash": hashed, "erasure_hash": erasure_hash,
                      "closure": serving.closure.id, "reference": previous.id,
                      "seq": max((row["seq"] for row in records), default=0) + 1,
                      "at": at, "input_hash": fingerprint, "output": before, "reference_output": baseline,
                      "variant": self.variant, "scales": self.scales, "outcome": None}
            path = self._daily_path(at)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as stream:
                stream.write(_io.canonical(record) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            return output

    def _summary(self, closure, reference):
        now = self._clock()
        cutoff = now - timedelta(hours=self.window_hours)
        rows = [row for row in self.records() if row["closure"] == closure and row["reference"] == reference
                and row["outcome"] is not None and cutoff <= _io.parse(row["at"]) <= now][-self.window_items:]
        n = len(rows)
        live = np.array([row["outcome"]["served_loss"] for row in rows])
        base = np.array([row["outcome"]["reference_loss"] for row in rows])
        difference = live - base
        denominator = max(float(base.mean()), 1e-12) if n else 1.
        excess = float(difference.mean()) / denominator if n else None
        width = 1.6448536269514722 * float(difference.std(ddof=1)) / math.sqrt(n) / denominator if n > 1 else 0.
        return {"kind": "numeric-online", "learner": self.store.name, "closure": closure, "reference": reference,
                "labelled": n, "served_loss": float(live.mean()) if n else None,
                "reference_loss": float(base.mean()) if n else None, "relative_excess": excess,
                "lower_bound": excess - width if n else None, "upper_bound": excess + width if n else None,
                "bar": self.bar, "rolled_back": False, "reason": "awaiting 30 labelled numeric outcomes"
                if n < 30 else "no measured numeric regression beyond the bar"}

    def _apply_regression(self, result):
        if result["labelled"] >= 30 and result["lower_bound"] > self.bar:
            reason = (f"numeric comparator regression: lower 95% relative excess loss {result['lower_bound']:.6f} "
                      f"exceeds bar {self.bar:.6f} over {result['labelled']} labelled items")
            changed = self._rollback(reason=reason, expected_closure=result["closure"])
            if changed == "rolled_back":
                result.update(rolled_back=True, reason=reason)
                _io.write_json(self.directory / "last-rollback.json", result)
        return result

    def summary(self):
        with self.lock:
            self.controller.recover_rollback()
            serving = self.store.revisions.serving()
            result = self._summary(serving.id, self.controller.previous(serving.id).id)
            if serving.mode != "base":
                self._apply_regression(result)
            try:
                result["last_rollback"] = _io.read_json(self.directory / "last-rollback.json")
            except FileNotFoundError:
                pass
            try:
                result["health"] = _io.read_json(self.store.directory / "numeric-health.json")
            except FileNotFoundError:
                pass
            return result

    def outcome(self, ref, observed, teacher="realized_value"):
        if teacher not in PROOF_TEACHERS:
            raise ValueError("only proof teachers can label numeric outcomes")
        observed = self._output_numbers(observed)
        with self.lock:
            self.controller.recover_rollback()
            hashed = self._path(ref).stem
            records = self.records()
            record = next((row for row in records if row["ref_hash"] == hashed), None)
            if record is None:
                raise ValueError("outcome names no served numeric comparator item")
            if np.shape(observed) != np.shape(record["output"]):
                raise ValueError("observed value must match served output shape")
            scale = 1. if record["variant"] == "logistic" else np.asarray(record["scales"])
            with np.errstate(over="ignore", invalid="ignore"):
                live = float(np.mean(((np.asarray(record["output"]) - observed) / scale) ** 2))
                base = float(np.mean(((np.asarray(record["reference_output"]) - observed) / scale) ** 2))
            if not math.isfinite(live) or not math.isfinite(base):
                raise ValueError("numeric losses must be finite")
            label = {"observed": observed, "served_loss": live, "reference_loss": base, "teacher": teacher}
            if record["outcome"] is not None and record["outcome"] != label:
                raise ValueError("numeric outcome changed under its reference")
            record["outcome"] = label
            path = self._daily_path(record["at"])
            self._rewrite(path, [row for row in records if self._daily_path(row["at"]) == path])
            result = self._summary(record["closure"], record["reference"])
            if self.store.revisions.serving().id == record["closure"] and record["closure"] != self.store.revisions.base.id:
                self._apply_regression(result)
            return result

    def erase(self, refs):
        with self.lock:
            hashes = {self._path(ref).stem for ref in refs}
            records = self.records()
            for path in self.directory.glob("numeric-*.jsonl"):
                self._rewrite(path, [row for row in records if self._daily_path(row["at"]) == path
                                     and row["ref_hash"] not in hashes])
            # Aggregate outcomes could otherwise retain evidence from the erased items.
            _io.remove(self.directory / "last-rollback.json")


from .nightly import COMPARATORS  # noqa: E402

COMPARATORS["numeric-online"] = NumericComparator
