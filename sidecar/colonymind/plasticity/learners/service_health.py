"""Service health on the sidecar's numeric learner, with warnings only.

The sampler reads only the instance directory. A shared predictor uses 12 lags
of each metric expressed relative to that series' fixed first-sample scale.
It has no endpoint text, service client, network call or actuator.

Plan assumptions: 300-second intervals, five-call floor, 12 lags, two surprises
at least 4, two-hour warning expiry, .5 error share for three fixed-arm intervals,
14-day numeric retention. Builder assumptions: newest fifth as a temporal gate
set (minimum 30); initial scale max(abs(value), 1); normal paired-loss intervals.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import json
import logging
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import replace
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np

from ... import usage
from .. import _io, builds
from ..admission import Admission
from ..budgets import Budget, BudgetCheck
from ..builds import TrainingRow
from ..gates import Check, GateReport, Item, SetLedger, Z95
from ..nightly import register_factory
from ..numeric import NumericLearner, Surprise, load_revision, moment, save_revision
from ..numeric_comparator import configure
from ..objectives import NEXT_VALUE
from ..synthetic import Cycle

logger = logging.getLogger(__name__)

GATE_MIN_ROWS = 30
TRAINING_MIN_ROWS = 1  # Existing calibration requires a nonempty training set.
INTERVAL = 300
LAGS = 12
TEMPORARY_SUFFIXES = (_io.TEMPORARY_SUFFIX, ".tmp", ".new", ".part", ".partial")
TEMPORARY_DIRECTORIES = ("numeric-gate-",)
DECLARATION = {
    "kind": "numeric-online", "implementation": "service_health", "base": "numeric-persistence@1",
    "teachers": {"realized_value": "value"}, "model": {"roles": ["numeric_predictor"], "change": "weights"},
    "budget": {"type": "anchored_output", "bound": .5}, "effect": "predict", "audience": "owner",
    "personal_data": False, "erasure_path": "rolling periods", "dependents": [], "mode": "stack",
    "gates": ["retention", "specificity", "canaries", "calibration", "budget", "finite", "served_offline"],
    "arms": ["persistence", "seasonal_day", "seasonal_week", "ridge", "time_shuffled", "no_accumulation"],
    "learning_rate": .01, "clip": .01,
    "comparator": {"window_items": 200, "window_hours": 168, "bar": .02},
}


def _keyed(key, value):
    return hmac.new(key, _io.canonical(value), hashlib.sha256).hexdigest()


def collect(directory, *, since, until, key):
    """Aggregate [since, until) usage, keeping client and endpoint alias only."""
    directory = Path(directory).resolve()
    path = directory / usage.DB_FILE
    if not path.resolve().is_relative_to(directory):
        raise ValueError("usage database must stay in the state directory")
    grouped = defaultdict(list)
    for row in usage.rows(directory, since=since):
        if not since <= moment(row["at"]) < until:
            continue
        client = row["client"] if row["client"] in usage.CLIENTS else "other"
        alias = usage.plain(row["endpoint"])
        if alias is None:
            alias = "alias-" + _keyed(key, ["endpoint", row["endpoint"]])[:12]
        grouped[(client, alias)].append(row)
    result = {}
    for (client, alias), rows in sorted(grouped.items()):
        if len(rows) < 5:
            continue
        subject = client + "." + alias
        values = {"latency_p95_ms": float(np.percentile([r["latency_ms"] for r in rows], 95)),
                  "error_share": sum(r["outcome"] != "ok" for r in rows) / len(rows), "call_count": float(len(rows))}
        for feature, value in values.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError("usage metric must be finite and nonnegative")
            result[subject + "." + feature] = {"subject": subject, "feature": feature, "value": value}
    size = 0
    for parent, directories, files in os.walk(directory, followlinks=False):
        directories[:] = [name for name in directories if not (Path(parent) / name).is_symlink()]
        # Measured without the state lock: skip in-flight atomic writes and gate exports.
        directories[:] = [name for name in directories if not name.startswith(TEMPORARY_DIRECTORIES)]
        for name in files:
            if name.endswith(TEMPORARY_SUFFIXES) or name.startswith(".capture-"):
                continue
            candidate = Path(parent) / name
            try:
                if not candidate.is_symlink():
                    size += candidate.stat().st_size
            except FileNotFoundError:
                pass  # Atomic state rewrites can remove a file during the walk.
    result["store.bytes"] = {"subject": "store", "feature": "bytes", "value": float(size)}
    return result


def _publish(runtime):
    with runtime.lock:
        runtime._publish()


async def in_place(function, *args):
    return function(*args)


async def off_loop(function, *args):
    """Run blocking store work in a worker thread so the event loop keeps serving.

    The work takes the runtime's thread lock inside the worker. As in NightlyTask, a
    cancelled caller still waits for the thread, so shutdown never closes the runtime
    under a pass that is writing state.
    """
    worker = asyncio.ensure_future(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled() and worker.exception() is not None:
            logger.warning("service health pass failed during shutdown (%s)", type(worker.exception()).__name__)
        raise


class Samples:
    """Machine-only numeric frames, keyed and retained for rolling period replay."""

    def __init__(self, store, clock):
        self.store, self.clock = store, clock
        self.path = store.directory / "numeric-samples.json"

    @property
    def key(self):
        # One disk read per call. Bulk paths read it once and pass it down, so a
        # pass over the retained rows does not reopen the key for every row.
        return (self.store.root / "comparator.key").read_bytes()

    def _read(self, key=None):
        key = self.key if key is None else key
        try:
            return [self.make_row(item["frame"], item["label"], key) for item in _io.read_json(self.path)]
        except FileNotFoundError:
            return []

    def _write(self, rows):
        # References and their state hashes are re-derived, not duplicated beside
        # manifests. Runtime deletes source rows after the core's unserve/scan.
        # Each state is already the canonical frame, so joining the items gives the
        # canonical JSON of [{"frame", "label"}, ...] without re-encoding every frame.
        _io.write_atomic(self.path, b"[" + b",".join(
            b'{"frame":' + row["state"].encode() + b',"label":' + _io.canonical(row["label"]) + b"}"
            for row in rows) + b"]")

    def _signature(self):
        try:
            found = os.stat(self.path)
        except FileNotFoundError:
            return None
        return found.st_ino, found.st_size, found.st_mtime_ns

    def snapshot(self, key=None):
        """The keyed rows read without the state lock, with the file signature that dates them.

        The signature is taken first, so a file replaced during the read fails the later check.
        """
        signature = self._signature()
        return signature, self._read(key)

    def reference(self, frame, key=None):
        return _keyed(self.key if key is None else key, ["sample", frame["at"], frame["series"]])

    def make_row(self, frame, label, key=None):
        # The frame is structured machine data, never a request or a response.
        key = self.key if key is None else key
        text = _io.canonical(frame).decode()
        digest = _keyed(key, frame)
        return {"ref": self.reference(frame, key), "point": "next_value",
                "version": "numeric-1", "state": text, "state_hash": digest, "label": float(label),
                "teacher": "realized_value", "push_away": False, "source_kind": "resolver", "audience": "owner",
                "at": frame["at"]}

    def add_many(self, pairs, key=None, snapshot=None):
        # The 14-day window is what rows(), nightly periods and the numeric
        # comparator read, so it bounds the file at 14 days of series frames.
        # A snapshot is used only while the file still has its signature.
        key = self.key if key is None else key
        if snapshot is not None and snapshot[0] == self._signature():
            saved = snapshot[1]
        else:
            saved = self._read(key)
        rows = {row["ref"]: row for row in saved}
        for frame, label in pairs:
            row = self.make_row(frame, label, key)
            if row["ref"] in rows and rows[row["ref"]] != row:
                raise ValueError("numeric sample changed under its reference")
            rows[row["ref"]] = row
        cutoff = self.clock() - timedelta(days=14)
        kept = [row for row in rows.values() if moment(row["at"]) >= cutoff]
        if kept != saved:
            self._write(kept)

    def rows(self, objective, *, since=None):
        cutoff = max(moment(since), self.clock() - timedelta(days=14)) if since else self.clock() - timedelta(days=14)
        key = self.key
        for row in self._read(key):
            if cutoff <= moment(row["at"]) <= self.clock() and (
                    not objective.points or row["point"] in objective.points):
                if _keyed(key, json.loads(row["state"])) == row["state_hash"]:
                    yield row

    def source_of(self, row):
        return None  # A system measurement, not a contact, device or model teacher.

    def erase(self, refs):
        refs = set(refs)
        self._write([row for row in self._read() if row["ref"] not in refs])


class NumericGateSets(SetLedger):
    """Keep keyed exclusion witnesses; resolve active row refs only in memory.

    An erased machine row must not survive as a second raw ref in a gate set.
    The keyed witness also prevents a held-out item from becoming training data.
    """
    def __init__(self, directory, samples):
        super().__init__(directory)
        self.samples = samples

    def register(self, set_id, role, refs):
        key = self.samples.key
        super().register(set_id, role, [_keyed(key, ["gate", ref]) for ref in refs])

    def protected_refs(self):
        protected = super().protected_refs()
        key = self.samples.key
        return frozenset(row["ref"] for row in self.samples._read(key)
                         if _keyed(key, ["gate", row["ref"]]) in protected)


class ServiceHealth(Cycle):
    """Cycle driver; Runtime owns promotion, compaction, rollback and erasure."""

    def __init__(self, *, store, controller, source, declaration, clock):
        for field in ("mode", "personal_data", "erasure_path", "effect", "audience", "dependents", "teachers"):
            if declaration.get(field) != DECLARATION[field]:
                raise ValueError("service_health: invalid " + field)
        if declaration["budget"]["type"] != "anchored_output" or not 0 < declaration["budget"]["bound"] <= .5:
            raise ValueError("service_health: anchored_output bound must be at most .5")
        self.store, self.controller, self.source, self.clock = store, controller, source, clock
        self.samples = Samples(store, clock)
        source.source = self.samples
        self.objective = NEXT_VALUE
        metadata = dict(schema="numeric-revision-1", learner=store.name, variant="value", tz="UTC",
            features=[{"stream": "value", "lag": k} for k in range(1, LAGS + 1)]
                     + [{"time": "hod_sin"}, {"time": "hod_cos"}, {"time": "how_sin"},
                        {"time": "how_cos"}, {"const": 1}],
            outputs=[{"stream": "value"}], standardisation={"value": {"mean": 0., "scale": 1.}},
            anchor={"type": "persistence"})
        self.learner = NumericLearner(metadata, rate=declaration.get("learning_rate", .01),
            clip=declaration.get("clip", .01), bound=declaration["budget"]["bound"], max_gap=INTERVAL)
        identity_path = store.directory / "numeric-layout.json"
        identity = {"identity": self.learner.identity, "metadata": metadata}
        if identity_path.exists() and _io.read_json(identity_path) != identity:
            raise ValueError("service_health: settings changed; declare a new learner identity")
        _io.write_json(identity_path, identity)
        configure(controller, declaration)
        self.sets = NumericGateSets(store.directory / "sets", self.samples)
        self.admission = Admission(self.objective, learner=store.name, source_of=source.source_of,
                                   protected_refs=self.sets.protected_refs, clock=clock)
        if store.budgets.declared(store.name) is None:
            store.budgets.declare(Budget(store.name, "anchored_output", self.learner.bound))
        self.last_period = self.last_report = None
        self._gate_rows = []
        self._suite = "numeric-empty"
        self._calibration = {}
        self.canaries = (Item("missing-numeric", '{"at":"2026-01-01T00:00:00Z"}', 0.),)

    def _training(self, row):
        return TrainingRow(row["ref"], row["point"], row["state"], row["label"], row["teacher"], False,
                           self.objective.losses[row["teacher"]], 1., row["audience"])

    def period(self):
        decided = self.store.decided()
        protected = self.sets.protected_refs()
        fresh = sorted((row for row in self.source.rows(self.objective)
                        if (row["ref"], row["teacher"]) not in decided and row["ref"] not in protected),
                       key=lambda row: (moment(row["at"]), row["ref"]))
        # Entire timestamps are held out together, so one alias cannot leak another's future.
        times = sorted({row["at"] for row in fresh})
        boundary = times[max(0, len(times) * 4 // 5)] if times else None
        self._gate_rows = [self._training(row) for row in fresh if boundary is not None and row["at"] >= boundary]
        self._suite = "numeric-" + _io.digest([row.ref for row in self._gate_rows])
        self.sets.register(self._suite, "gate_suite", [row.ref for row in self._gate_rows])
        return super().period()

    def propose(self, build):
        serving, start = self._start()
        periods = tuple(p.id for p in self.store.periods.retained()[-1:])
        rows = builds.training_rows(self.objective, self.store.manifests_of(self.store.periods.retained()[-1:]), self.source)
        if rows:
            errors = np.abs(self.learner.predict(build.state, [r.state for r in rows])[:, 0]
                            - np.array([r.label for r in rows]))
            self._calibration = {"error_scale": max(float(errors.mean()), .01), "rows": len(rows),
                                 "state": build.digest}
        else:
            self._calibration = {"error_scale": .01, "rows": 0, "state": build.digest}
        calibration = "numeric-" + _io.digest(self._calibration)
        closure = self.store.revisions.compose(state=build.digest, objective=self.objective.identity,
            calibration=(calibration,), dependents={}, parent=serving.id, manifests=build.manifests,
            periods=periods, audience=build.audience, mode="stack")
        self.controller.propose(closure, build.state)
        if periods:
            self.store.save_delta(periods[-1], builds.delta(build.state, start))
        self.store.budgets.record(BudgetCheck(self.store.name, build.digest, "anchored_output",
                                             self.learner.bound, self.learner.budget_used(build.state)))
        return closure

    def export_revision(self, closure, path):
        """Export on request; the core owns the only retained learned weight copies."""
        state = self.store.revisions.state(closure) or self.learner.base()
        return save_revision(path, self.learner.metadata, state["W"])

    def gate(self, closure, build):
        rows = self._gate_rows
        self.sets.use(self._suite, "selection", revision=closure.id)
        enough = len(rows) >= GATE_MIN_ROWS
        checks = []
        if enough:
            states = [row.state for row in rows]
            truth = np.array([row.label for row in rows])
            candidate = self.learner.predict(build.state, states)[:, 0]
            base = self.learner.predict(self.learner.base(), states)[:, 0]
            previous = self.learner.predict(self._start()[1], states)[:, 0]
            training = builds.training_rows(self.objective, self.store.manifests_of(
                self.store.periods.retained()[-1:]), self.source)
            order = np.random.default_rng(7).permutation(len(training))  # Fixed control seed, builder assumption.
            shuffled = [replace(row, label=training[int(i)].label) for row, i in zip(training, order)]
            control = self.learner.fit(self.learner.base(), shuffled, objective=self.objective, seed=0)
            control = self.learner.predict(control, states)[:, 0]
            loss = (candidate - truth) ** 2

            def gain_lower(reference):
                delta = (reference - truth) ** 2 - loss
                width = Z95 * float(delta.std(ddof=1)) / math.sqrt(len(rows))
                return float(delta.mean()) - width

            checks += [Check("retention", gain_lower(base) >= 0 and gain_lower(previous) >= 0,
                             "paired numeric loss against persistence and previous closure"),
                       Check("specificity", gain_lower(control) > 0, "paired loss against time-shuffled targets")]
            with tempfile.TemporaryDirectory(prefix="numeric-gate-", dir=self.store.directory) as temp:
                path = Path(temp) / "revision"
                self.export_revision(closure, path)
                served = load_revision(path)
                online = [served.forecast(at=(f := json.loads(row.state))["at"], streams=f["streams"])["value"]
                          for row in rows]
            agree = bool(np.allclose(online, candidate, rtol=1e-6, atol=1e-6))
        else:
            checks += [Check("retention", False, "awaiting 30 temporal gate items"),
                       Check("specificity", False, "awaiting 30 temporal gate items")]
            agree = False
        budget = self.store.budgets.result(self.store.name, build.digest)
        checks += [Check("canaries", self.learner.exposure(build.state, self.canaries)
                          == self.learner.exposure(self.learner.base(), self.canaries)),
                   Check("calibration", self._calibration.get("rows", 0) > 0
                         and closure.calibration != self.store.revisions.serving().calibration),
                   Check("budget", budget is not None and budget.ok),
                   Check("finite", all(np.isfinite(v).all() for v in build.state.values())),
                   Check("served_offline", agree)]
        self.last_report = GateReport(closure.id, self._suite, tuple(checks))
        return self.last_report

    def _live_state(self):
        try:
            return _io.read_json(self.store.directory / "numeric-live.json")
        except FileNotFoundError:
            return {"at": None, "series": {}, "warnings": [], "forecasts": 0, "surprises": 0,
                    "notes": 0, "ignored": 0, "fixed": {}}

    def _save_live(self, state):
        _io.write_json(self.store.directory / "numeric-live.json", state)
        _io.write_json(self.store.directory / "numeric-health.json", {
            "streams": len(state["series"]), **{k: state[k] for k in ("forecasts", "surprises", "notes", "ignored")},
            "lead_seconds": [w["lead_seconds"] for w in state["warnings"] if w.get("lead_seconds") is not None]})

    def warning_records(self):
        return copy.deepcopy(self._live_state()["warnings"])

    def _pass(self, runtime, at):
        """One blocking sampling pass, run in a worker by ``live``.

        Loop-side callers take the same state lock synchronously, so this pass never holds it
        for more than one step. Measuring, reading and keying the retained samples, and the
        forecast arithmetic run unlocked. Each comparator serve or outcome takes the lock for
        its own call, as a request does. The samples and live state are committed under the
        lock only if no other pass committed first, without rows erased during the pass. The
        key is read once. A pass interrupted before its commit is redone by the next one:
        served and labelled comparator items are found again by reference.
        """
        key = self.samples.key
        begun = self._live_state()
        if begun["at"] is not None and moment(begun["at"]) >= at:
            return
        values = collect(runtime.data_dir, since=at - timedelta(seconds=INTERVAL), until=at, key=key)
        snapshot = self.samples.snapshot(key)
        state, pairs = self._sample(runtime, at, values, key, copy.deepcopy(begun))
        comparator = runtime.loaded[self.store.name].comparator
        with runtime.lock:
            if self._live_state()["at"] != begun["at"]:
                return
            erased = comparator._erased()
            pairs = [(frame, value) for frame, value in pairs
                     if comparator._hash(["erased", self.samples.reference(frame, key)]) not in erased]
            self.samples.add_many(pairs, key, snapshot)
            self._save_live(state)

    def _sample(self, runtime, at, values, key, state):
        cutoff = at - timedelta(days=14)
        state["warnings"] = [w for w in state["warnings"] if moment(w["at"]) >= cutoff]
        state["series"] = {k: v for k, v in state["series"].items() if moment(v["at"]) >= cutoff}
        comparator = runtime.loaded[self.store.name].comparator
        records = {r["ref_hash"]: r for r in comparator.records()}
        pairs = []
        for series, sample in values.items():
            previous = state["series"].get(series)
            continuous = previous is not None and (at - moment(previous["at"])).total_seconds() == INTERVAL
            if previous is None:
                previous = {"mean": sample["value"], "scale": max(abs(sample["value"]), 1.), "history": [],
                            "count": 0, "total": 0., "streak": 0, "pending": None}
            value = (sample["value"] - previous["mean"]) / previous["scale"]
            pending = previous.get("pending")
            if continuous and pending is not None:
                ref = self.samples.reference(pending["frame"], key)
                if comparator._path(ref).stem in records and self._label(runtime, comparator, ref, value):
                    pairs.append((pending["frame"], value))
                    surprise = Surprise(count=previous["count"], total=previous["total"])
                    severity = surprise.score(value - pending["output"])
                    previous.update(count=surprise.count, total=surprise.total)
                    previous["streak"] = previous["streak"] + 1 if severity is not None and severity >= 4 else 0
                    if severity is not None and severity >= 4:
                        state["surprises"] += 1
                    if previous["streak"] >= 2:
                        ident = _keyed(key, ["warning", series, at.date().isoformat()])
                        if not any(w["id"] == ident for w in state["warnings"]):
                            state["warnings"].append({"id": ident, "subject": sample["subject"],
                                "feature": sample["feature"], "at": at.isoformat(), "severity": severity,
                                "delivered": False, "fixed_at": None, "lead_seconds": None})
            else:
                previous["streak"] = 0
            history = ([value] + previous["history"])[:LAGS] if continuous else [value] * LAGS
            frame = {"at": (at + timedelta(seconds=INTERVAL)).isoformat(), "previous_at": at.isoformat(),
                     "streams": {"value": history}, "series": series}
            ref = self.samples.reference(frame, key)
            hashed = comparator._path(ref).stem
            if hashed in records:
                output = records[hashed]["output"]
            else:
                output = runtime.serve(self.store.name, ref, frame,
                    predict=lambda pin, inputs: float(self.learner.predict(pin.state if pin else {}, [inputs])[0, 0]))
                state["forecasts"] += 1
            previous.update(at=at.isoformat(), history=history, pending={"frame": frame, "output": output})
            state["series"][series] = previous
            if sample["feature"] == "error_share":
                fixed = state["fixed"].get(sample["subject"], {"streak": 0, "at": None, "warning_at": None})
                consecutive = fixed["at"] is not None and (at - moment(fixed["at"])).total_seconds() == INTERVAL
                fixed["streak"] = (fixed["streak"] if consecutive else 0) + 1 if sample["value"] >= .5 else 0
                if fixed["streak"] < 3:
                    fixed["warning_at"] = None
                elif fixed["warning_at"] is None:
                    fixed["warning_at"] = at.isoformat()
                fixed["at"] = at.isoformat()
                state["fixed"][sample["subject"]] = fixed
        for warning in state["warnings"]:
            fixed = state["fixed"].get(warning["subject"], {}).get("warning_at")
            if warning["fixed_at"] is None and fixed and moment(fixed).date() == moment(warning["at"]).date():
                warning["fixed_at"] = fixed
                warning["lead_seconds"] = (moment(fixed) - moment(warning["at"])).total_seconds()
        state["fixed"] = {k: v for k, v in state["fixed"].items() if moment(v["at"]) >= cutoff}
        state["at"] = at.isoformat()
        return state, pairs

    def _label(self, runtime, comparator, ref, value):
        """Label a served forecast unless an erasure removed it after the records were read."""
        with runtime.lock:
            if comparator._hash(["erased", ref]) in comparator._erased():
                return False
            runtime.outcome(self.store.name, ref, observed=value, teacher="realized_value")
            return True

    async def sample(self, runtime, *, mind=None, at=None, offload=None):
        """Sample one interval. ``offload`` runs the blocking store work; ``live`` passes
        ``off_loop`` so the event loop keeps serving, including by default."""
        run = offload or off_loop
        if not runtime.enabled or self.store.name not in runtime.loaded:
            return
        if mind is None:
            from ...api.routers.mind import get_mind
            mind = get_mind()
            if mind is not None and getattr(mind, "plasticity", None) is not runtime:
                mind = None
        if mind is not None and not mind.faculties.get("plasticity", True):
            return
        at = moment(at or self.clock()).astimezone(timezone.utc)
        await run(self._pass, runtime, at)
        if mind is None:
            await run(_publish, runtime)
            return
        state = self._live_state()
        pending = [w for w in state["warnings"] if not w["delivered"] and at - moment(w["at"]) < timedelta(hours=2)]
        while pending:
            from ...mind.observations import MAX_BATCH, for_mind
            # A source post replaces its set. Keep its unexpired accepted notes
            # while submitting bounded batches of new warnings to the S1 cap.
            active = for_mind(mind).active(source="service-health", viewer=mind.owner_id)
            waiting = {w["id"] for w in pending}
            active = [item for item in active if item["key"] not in waiting][-(MAX_BATCH - 1):]
            batch, pending = pending[:MAX_BATCH - len(active)], pending[MAX_BATCH - len(active):]
            body = {"source": "service-health", "observations": active + [{
                "key": w["id"], "kind": "numeric_surprise", "subject": w["subject"], "feature": w["feature"],
                "severity": w["severity"], "observed_at": w["at"],
                "expires_at": (moment(w["at"]) + timedelta(hours=2)).isoformat(),
                "summary": "A service metric changed unexpectedly."} for w in batch]}
            result = await mind.observe_source("service-health", body)
            saved = mind.mind_state.get("observations.source:service-health")
            await run(self._delivered, runtime, {w["id"] for w in batch}, result.get("ignored", 0), saved)
        await run(_publish, runtime)

    def _delivered(self, runtime, delivered, ignored, saved):
        with runtime.lock:
            state = self._live_state()
            for warning in state["warnings"]:
                if warning["id"] in delivered:
                    warning["delivered"] = True
            state["ignored"] += ignored
            if saved and saved.get("text"):
                state["notes"] = sum(e.get("concerns", 0) for e in json.loads(saved["text"]).get("events", []))
            self._save_live(state)

    async def live(self, runtime):
        while runtime.enabled:
            # Samples end at a fixed UTC interval boundary, including after a process restart.
            now = self.clock()
            end = moment(math.floor(now.timestamp() / INTERVAL) * INTERVAL)
            await self.sample(runtime, at=end, offload=off_loop)
            await asyncio.sleep(max(1., INTERVAL - (self.clock() - end).total_seconds()))


register_factory("service_health", ServiceHealth)
