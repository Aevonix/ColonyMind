"""Live paired scoring, reference-only records, and measured-regression rollback.

The previous serving closure (the base for the first promotion) scores the exact same input as the live closure.
Inputs, string outputs and decision references are keyed hashes; only finite numeric outputs and boolean outcomes
are stored directly. Owner labels and verified outcomes alone provide live proof.

The rolling window is the newest 200 labelled items within seven days of service (my assumption). Rollback needs
the one-sided 95% *lower* bound of decline to exceed the declared bar: evidence of regression, using P1a's paired
statistic with signs reversed. The upper bound is reported too. Its rule-of-three floor is 3/200, above the default
one-point bar even for identical outputs, so using that upper bound to trigger rollback would reject every revision
without measured regression. A stale result never rolls back a newer revision.
"""

from __future__ import annotations

import hashlib
import hmac
import math
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import numpy as np

from . import _io
from .audience import may_serve
from .gates import DECLINE_BAR, decline_bound
from .revisions import Pinned
from .states import Controller

#: Live rolling-window limits (my assumptions).
WINDOW_ITEMS = 200
WINDOW_DAYS = 7
PROOF_TEACHERS = frozenset(("owner_label", "verified_outcome"))


def _value(value: Any) -> Any:
    """Normalize in memory for keyed hashing, never a text log."""
    if isinstance(value, np.ndarray):
        return _value(value.tolist())
    if isinstance(value, np.generic):
        return _value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_value(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    return value


class LiveComparator:
    """One learner's paired live records and outcomes, serialized with promotion and erasure.

    ``predict(Pinned, inputs)`` runs for both closures. ``pin(requester)`` optionally supplies a chain's joint pin.
    The caller supplies a persistent secret ``key``; this module never writes it.
    """

    def __init__(self, controller: Controller, *, key: bytes, bar: float = DECLINE_BAR,
                 clock: Optional[Callable[[], datetime]] = None, lock=None,
                 pin: Optional[Callable[[Optional[str]], Pinned]] = None,
                 rollback: Optional[Callable[..., str]] = None):
        if not isinstance(key, bytes) or not key:
            raise ValueError("a comparator needs a nonempty keyed-hash key")
        if not math.isfinite(bar) or bar < 0:
            raise ValueError("the decline bar must be finite and nonnegative")
        self.controller = controller
        self.store = controller.store
        self.directory = self.store.directory / "comparator"
        self.bar = float(bar)
        self._key = key
        self._clock = clock or controller._clock
        self.lock = lock if lock is not None else controller.lock
        self._pin = pin or self.store.revisions.pin
        self._rollback = rollback or controller.rollback

    def _hash(self, value: Any) -> str:
        return hmac.new(self._key, _io.canonical(_value(value)), hashlib.sha256).hexdigest()

    def _path(self, ref: str):
        if not isinstance(ref, str) or not ref:
            raise ValueError("a served item needs a decision reference")
        return self.directory / f"{self._hash(['reference', ref])}.json"

    def _output(self, output: Any) -> Dict[str, Any]:
        value = _value(output)
        if isinstance(value, (int, float, bool)):
            if not math.isfinite(value):
                raise ValueError("a comparator output must be finite")
            return {"value": value}
        return {"hash": self._hash(["output", value])}

    def serve(self, ref: str, inputs: Any, predict: Callable[[Pinned, Any], Any], *,
              requester: Optional[str] = "owner") -> Any:
        """Score the same input twice, durably record both outputs, then return the served output."""
        with self.lock:
            self.summary()
            path = self._path(ref)
            if path.exists():
                raise ValueError("a decision reference already has a served comparator item")
            serving = self._pin(requester)
            previous = self.controller.previous(serving.closure.id)
            if not may_serve(previous.audience, requester):
                previous = self.store.revisions.base
            reference = Pinned(previous, self.store.revisions.state(previous))
            input_hash = self._hash(["input", inputs])
            output = predict(serving, inputs)
            reference_output = predict(reference, inputs)
            if self._hash(["input", inputs]) != input_hash:
                raise ValueError("paired predictors must leave their shared input unchanged")
            record = {"learner": self.store.name, "ref_hash": path.stem,
                      "seq": max((item["seq"] for item in self.records()), default=0) + 1,
                      "closure": serving.closure.id, "reference": previous.id,
                      "input_hash": input_hash, "output": self._output(output),
                      "reference_output": self._output(reference_output),
                      "at": _io.stamp(self._clock()), "outcome": None}
            _io.write_json(path, record)
            return output

    def records(self) -> List[Dict[str, Any]]:
        """Reference-only paired records, ordered by service time and keyed reference."""
        with self.lock:
            return sorted((_io.read_json(path) for path in self.directory.glob("*.json")),
                          key=lambda record: (record["at"], record["seq"]))

    def _summary(self, closure: str, reference: str) -> Dict[str, Any]:
        now = self._clock()
        cutoff = now - timedelta(days=WINDOW_DAYS)
        rows = [row for row in self.records() if row["closure"] == closure and row["reference"] == reference
                and row["outcome"] is not None and cutoff <= _io.parse(row["at"]) <= now][-WINDOW_ITEMS:]
        n = len(rows)
        losses = sum(bool(row["outcome"]["reference_correct"] and not row["outcome"]["served_correct"])
                     for row in rows)
        gains = sum(bool(row["outcome"]["served_correct"] and not row["outcome"]["reference_correct"])
                    for row in rows)
        return {"learner": self.store.name, "closure": closure, "reference": reference,
                "labelled": n, "losses": losses, "gains": gains, "decline": (losses - gains) / n if n else None,
                "lower_bound": -decline_bound(n, losses=gains, gains=losses) if n else None,
                "upper_bound": decline_bound(n, losses=losses, gains=gains) if n else None,
                "bar": self.bar, "rolled_back": False,
                "reason": "awaiting labelled outcomes" if not n else "no measured regression beyond the bar"}

    def summary(self) -> Dict[str, Any]:
        """Recover interrupted rollback, including an outcome saved immediately before the intent was written."""
        with self.lock:
            self.controller.recover_rollback()
            closure = self.store.revisions.serving()
            result = self._summary(closure.id, self.controller.previous(closure.id).id)
            return self._apply_regression(result) if closure.mode != "base" else result

    def _apply_regression(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if result["labelled"] and result["lower_bound"] > self.bar:
            reason = (f"live comparator regression: lower 95% decline bound {result['lower_bound']:.6f} "
                      f"exceeds bar {self.bar:.6f} over {result['labelled']} labelled items")
            changed = self._rollback(reason=reason, expected_closure=result["closure"])
            result.update(rolled_back=changed == "rolled_back", reason=reason)
        return result

    def outcome(self, ref: str, *, served_correct: bool, reference_correct: bool,
                teacher: str = "verified_outcome") -> Dict[str, Any]:
        """Attach a real paired outcome; roll back only the revision this outcome scored."""
        if teacher not in PROOF_TEACHERS:
            raise ValueError("only owner labels and verified outcomes count as live proof")
        if type(served_correct) is not bool or type(reference_correct) is not bool:
            raise ValueError("paired correctness outcomes must be boolean")
        with self.lock:
            self.controller.recover_rollback()
            path = self._path(ref)
            try:
                record = _io.read_json(path)
            except FileNotFoundError:
                raise ValueError("the outcome names no served comparator item") from None
            outcome = {"served_correct": served_correct, "reference_correct": reference_correct,
                       "teacher": teacher, "at": _io.stamp(self._clock())}
            if record["outcome"] is None or teacher == "owner_label" or record["outcome"]["teacher"] != "owner_label":
                record["outcome"] = outcome
                _io.write_json(path, record)
            result = self._summary(record["closure"], record["reference"])
            return self._apply_regression(result)

    def erase(self, refs: Iterable[str]) -> int:
        """Delete both predictions and outcome for each erased reference, under the shared state lock."""
        with self.lock:
            return sum(_io.remove(self._path(ref)) for ref in refs)


__all__ = ["BaselineComparator", "LiveComparator", "PROOF_TEACHERS", "WINDOW_DAYS", "WINDOW_ITEMS"]


# Behaviour uses a historical baseline because replayed context contaminates concurrent arms.
BASELINE_DEFAULTS = {
    'days': 10, 'min_replies': 60, 'min_session_days': 12, 'target_cap': .03,
    'standing_days': 14, 'min_guard_units': None, 'bootstrap_samples': 1000,
    'guards': {'substitute_rate': .02, 'median_length': .15,
               'tool_failure_rate': .01, 'owner_corrections': .10},
}


def _baseline_settings(declaration):
    options = declaration.get('baseline', {})
    if not isinstance(options, dict) or set(options) - set(BASELINE_DEFAULTS):
        raise ValueError('baseline: unknown settings')
    guards = options.get('guards', {})
    if not isinstance(guards, dict) or set(guards) - set(BASELINE_DEFAULTS['guards']):
        raise ValueError('baseline: unknown guard')
    settings = {**BASELINE_DEFAULTS, **options,
                'guards': {**BASELINE_DEFAULTS['guards'], **guards}}
    if 'min_guard_units' not in options:
        settings['min_guard_units'] = settings['min_session_days']
    for name in ('days', 'min_replies', 'min_session_days', 'standing_days', 'min_guard_units', 'bootstrap_samples'):
        if type(settings[name]) is not int or settings[name] <= 0:
            raise ValueError(f'baseline: {name} must be a positive integer')
    for name, value in [('target_cap', settings['target_cap']), *settings['guards'].items()]:
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'baseline: {name} must be finite and nonnegative')
    if settings['target_cap'] > 1:
        raise ValueError('baseline: target_cap must be at most one')
    return settings


def configure(controller, declaration):
    """Persist validated behaviour settings before Runtime wires the comparator."""
    settings = _baseline_settings(declaration)
    _io.write_json(controller.store.directory / 'baseline-comparator.json', settings)
    return settings


def _effective(units):
    """Effective binomial counts under session-day ratio-estimator clustering."""
    values = list(units.values()) if isinstance(units, Mapping) else list(units)
    values = [(float(x), float(m)) for x, m in values if m > 0]
    x, n = sum(x for x, _ in values), sum(m for _, m in values)
    if not n:
        return {'x': 0., 'n': 0., 'n_eff': 0., 'x_eff': 0., 'units': 0, 'de': 1., 'rate': None}
    p, k = x / n, len(values)
    if all(m == 1 for _, m in values):
        de = 1.
    elif 0 < p < 1 and k > 1:
        variance = k / (k - 1) * sum((xi - p * mi) ** 2 for xi, mi in values) / n ** 2
        de = max(1., variance / (p * (1 - p) / n))
    else:
        de = 1.
    return {'x': x, 'n': n, 'n_eff': n / de, 'x_eff': p * n / de,
            'units': k, 'de': de, 'rate': p, 'mbar': n / k}


def _aggregate(rows):
    """Counts only: discard session/turn identities and all prose before freezing."""
    metrics = {'target': {}, **{name: {} for name in BASELINE_DEFAULTS['guards']}}
    for row in rows:
        unit = (row['session_id'], _io.parse(row['at']).date().isoformat())
        for name, groups in metrics.items():
            value = row.get(name)
            if value is None:
                continue
            if name == 'median_length':
                groups.setdefault(unit, []).append(value)
            else:
                x, n = value if isinstance(value, (tuple, list)) else (int(value), 1)
                total = groups.setdefault(unit, [0, 0])
                total[0] += x
                total[1] += n
    return {name: list(groups.values()) for name, groups in metrics.items()}


ONE_SIDED_95_Z = 1.6448536269514722


def _rate_comparison(before, after, z=ONE_SIDED_95_Z):
    from .gates import two_proportion_bounds
    b, t = _effective(before), _effective(after)
    # At p=0 or p=1 the within-arm variance cannot estimate correlation.
    # Borrow the other population's ICC, bounded to a valid correlation.
    for arm, other in ((b, t), (t, b)):
        if arm['n'] and arm['rate'] in (0, 1):
            rho = (max(0., min(1., (other['de'] - 1) / (other['mbar'] - 1)))
                   if other['rate'] not in (None, 0, 1) and other['mbar'] > 1 else 0.)
            arm['de'] = 1 + (arm['mbar'] - 1) * rho
            arm['n_eff'] = arm['n'] / arm['de']
            arm['x_eff'] = arm['rate'] * arm['n_eff']
    lower = upper = None
    if b['n_eff'] and t['n_eff']:
        lower, upper = two_proportion_bounds(b['x_eff'], b['n_eff'], t['x_eff'], t['n_eff'], z)
    return {'baseline': b, 'taught': t, 'lower': lower, 'upper': upper}


def _length_comparison(before, after, samples):
    """Session-day cluster bootstrap of the taught/baseline median length ratio."""
    if not before or not after:
        return {'lower': None, 'upper': None}
    rng = np.random.default_rng(0)
    ratios = []
    for _ in range(samples):
        b = np.median([v for i in rng.integers(len(before), size=len(before)) for v in before[i]])
        t = np.median([v for i in rng.integers(len(after), size=len(after)) for v in after[i]])
        if b > 0:
            ratios.append(float(t / b))
    if not ratios:
        return {'lower': None, 'upper': None}
    lo, hi = np.percentile(ratios, [5, 95])
    return {'lower': float(lo), 'upper': float(hi)}


class BaselineComparator(LiveComparator):
    """Frozen historical evidence and a live, session-day clustered taught window.

    Evaluations and certificates contain counts, never reply text or source identities.
    Only the runtime acts on a returned verdict; serving itself performs no writes.
    """

    def __init__(self, controller, *, key, bar=DECLINE_BAR, clock=None, lock=None, pin=None, rollback=None):
        super().__init__(controller, key=key, bar=bar, clock=clock, lock=lock, pin=pin, rollback=rollback)
        try:
            settings = _io.read_json(self.store.directory / 'baseline-comparator.json')
        except FileNotFoundError:
            settings = {}
        self.settings = _baseline_settings({'baseline': settings})

    def serve(self, ref, inputs, predict, *, requester='owner'):
        with self.lock:
            return predict(self._pin(requester), inputs)

    def records(self):
        return []

    def outcome(self, *args, **kwargs):
        raise ValueError('a baseline comparator takes no paired outcomes')

    def summary(self):
        with self.lock:
            self.controller.recover_rollback()
            try:
                return _io.read_json(self.directory / 'last-evaluation.json')
            except FileNotFoundError:
                return {}

    def evaluate(self, closure, observations):
        with self.lock:
            return self._evaluate(closure, observations)

    def _baseline_state(self):
        try:
            return _io.read_json(self.store.directory / 'behaviour-baselines.json')
        except FileNotFoundError:
            return {}

    def freeze_baseline(self, observations):
        """Freeze once per lesson at its first promotion; store aggregate counts only."""
        state = self._baseline_state()
        lesson = observations['lesson']
        if lesson not in state:
            promoted = _io.parse(observations['promoted'])
            rows = [r for r in observations['rows'] if
                    promoted - timedelta(days=self.settings['days']) <= _io.parse(r['at']) < promoted]
            state[lesson] = {'at': _io.stamp(promoted), 'passed': False,
                             'counts': {'all': _aggregate(rows), 'platforms': {
                                 platform: _aggregate([r for r in rows if r['platform'] == platform])
                                 for platform in sorted({r['platform'] for r in rows})}}}
            _io.write_json(self.store.directory / 'behaviour-baselines.json', state)
        return state

    def _evaluate(self, closure, observations):
        settings, now = self.settings, self._clock()
        promoted = _io.parse(observations['promoted'])
        state = self.freeze_baseline(observations)
        entry = state[observations['lesson']]
        baseline_at = _io.parse(entry['at'])
        restarts = sorted({_io.parse(at) for at in observations.get('restarts', ()) if
                           baseline_at < _io.parse(at) <= now})
        start = max([promoted, *restarts])
        window = _io.stamp(start)
        frozen, passed = entry['counts'], entry['passed']
        rows = [r for r in observations['rows'] if _io.parse(r['at']) <= now]
        cutoff = max(start, now - timedelta(days=settings['standing_days'])) if passed else start
        taught_rows = [r for r in rows if cutoff <= _io.parse(r['at'])]
        taught = _aggregate(taught_rows)
        target = _rate_comparison(frozen['all']['target'], taught['target'], ONE_SIDED_95_Z)
        enough = (target['taught']['n'] >= settings['min_replies']
                  and target['taught']['units'] >= settings['min_session_days']
                  and target['baseline']['n_eff'] > 0)
        target_ok = (target['lower'] is not None and target['lower'] > 0
                     and target['taught']['rate'] <= settings['target_cap'])
        reversal = target['upper'] is not None and target['upper'] < 0
        target['verdict'] = ('awaiting_evidence' if not enough else 'passed' if target_ok else
                             'failed' if reversal or passed or target['taught']['rate'] > settings['target_cap']
                             else 'awaiting_evidence')
        guards = {}
        for name, margin in settings['guards'].items():
            b, t = frozen['all'][name], taught[name]
            measured = len(b) >= settings['min_guard_units'] and len(t) >= settings['min_guard_units']
            if name == 'median_length':
                result = _length_comparison(b, t, settings['bootstrap_samples']) if measured else {
                    'lower': None, 'upper': None}
                result.update(baseline_units=len(b), taught_units=len(t))
                lo, hi = result['lower'], result['upper']
                breach = lo is not None and (lo > 1 + margin or hi < 1 - margin)
                safe = lo is not None and lo >= 1 - margin and hi <= 1 + margin
            else:
                result = _rate_comparison(t, b, ONE_SIDED_95_Z)
                # _rate_comparison names its arguments; here the difference is taught - baseline.
                result['baseline'], result['taught'] = result['taught'], result['baseline']
                lo, hi = result['lower'], result['upper']
                breach = lo is not None and lo > margin
                safe = hi is not None and hi <= margin
            result.update(margin=margin, verdict='not_measured' if not measured else 'breach' if breach else
                          'non_inferior' if safe else 'inconclusive')
            guards[name] = result
        breach = any(g['verdict'] == 'breach' for g in guards.values())
        verdict = 'failed' if breach or target['verdict'] == 'failed' else target['verdict']
        platforms = {}
        for platform in sorted(set(frozen['platforms']) | {r['platform'] for r in taught_rows}):
            before = frozen['platforms'].get(platform, {}).get('target', [])
            after = _aggregate([r for r in taught_rows if r['platform'] == platform])['target']
            platforms[platform] = _rate_comparison(before, after, ONE_SIDED_95_Z)
        result = {'kind': 'behaviour', 'closure': closure.id, 'window': window,
                  'baseline_counts': frozen, 'baseline_at': entry['at'],
                  'restarts': [{'at': _io.stamp(at), 'baseline_predates': True} for at in restarts],
                  'confounds': [f'baseline predates a head/stack change at {_io.stamp(at)}' for at in restarts],
                  'not_measured_guards': [name for name, guard in guards.items() if guard['verdict'] == 'not_measured'],
                  'target': target, 'guards': guards, 'platforms': platforms,
                  'counts': {'taught': len(taught_rows), 'baseline': int(target['baseline']['n']),
                             'session_days': target['taught']['units'],
                             'exposed': sum(bool(r.get('exposed')) for r in taught_rows)},
                  'verdict': verdict, 'certified': passed or verdict == 'passed'}
        entry['passed'] = result['certified']
        _io.write_json(self.store.directory / 'behaviour-baselines.json', state)
        _io.write_json(self.directory / 'last-evaluation.json', result)
        return result

    def erase(self, refs):
        with self.lock:
            _io.remove(self.directory / 'last-evaluation.json')
            if 'example' in refs:
                _io.remove(self.store.directory / 'example.json')
