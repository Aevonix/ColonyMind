"""The runner: ``await instinct.decide(point, fields, refs, deliberate=None)``.

The tiers, in order; the first that settles decides:

1. **rules**, in code: final where they speak, and then no model is asked, live or shadow;
2. **fast** and 3. **reader**: model tiers that answer probabilities. A live tier settles only when its calibrated
   probability clears the label's bar and the label is one it may settle, and the case is not high stakes;
   otherwise its answer was unsure. A shadow (or demoted) tier is asked beside the deciding path, logged and never
   applied, and never delays the answer. A state longer than the provider reads is never cut: the tier is skipped.
   A provider that is down, failing, slow or has no matching calibration record gives no answer, and then no
   later tier is asked beyond the existing path (live or shadow): only an unsure answer escalates;
4. **deliberate**: the caller's own typed call on the mind's model (it keeps its budget and its deadline). Where it
   is the point's existing path it is asked as it was before Instinct; any other deliberate tier is asked only on
   an unsure answer, within the point's hourly escalation budget, which every reader ask (live or shadow) also
   takes from. A failure never escalates. Where the existing path is a second opinion on a rules tier's prior (the
   owner gate's judge), a live model tier that was asked stands in for it: whatever that tier leaves unsettled
   (unsure, failed, down, unmatched, at budget) keeps the prior's reading, and the second opinion is asked only for
   a high-stakes case, so promoting a tier never adds a model call to the existing path;
5. **fallback**: the point's conservative answer.

**Compared.** Where the existing path asks a model (the point's deliberate tier is the path, as the owner gate's
second opinion is), a live model tier that settles leaves that call unasked, so drift could not be measured on its
settles. The call is asked beside the settle instead, within the point's hourly budget: its answer is logged under
the deliberate tier as ``compared`` (the existing path's answer, ``fit.existing_answer``) and never applied. The call
is the caller's own, spending the caller's budget (the night's share, the day's), so it runs inside the caller's
run: off the owner's turn it is asked before the settled answer is returned, in turn with the caller's other calls,
so the caller's budget check before a call sees every call made before it and nothing is left running after the
caller is done (cancelling the caller cancels it). On the owner's turn (``turn_path``) the owner never waits for it:
it runs in the background, one at a time per point (another settle meanwhile stays unmeasured, ``busy``), within the
shadow asks' limit, and closing the engine cancels it. At budget, busy, failed or with no usable answer, the settle
stays unmeasured.

Callers apply only an applied label. Every decision is logged by reference (``log.Store``): ids and probabilities,
never the words it read.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from colonymind.util.temporal import now_utc

from . import calibration, settings
from . import state as point_state
from .calibration import Record
from .fit import COMPARED
from .log import DB_FILE, Store, code, identifier, keyed_hash, references
from .providers import Breaker, calibration_kind
from .records import shipped
from .registry import MODEL_TIERS, REGISTRY, Point, RulesAnswer, Tier

logger = logging.getLogger(__name__)

HOUR = 3600.0
#: At most this many shadow asks in flight at once (my choice): measurement never piles up behind a slow server.
MAX_SHADOW = 32
#: How long states and local records read from ``instinct.db`` are reused before they are read again.
OVERRIDES_TTL_S = 30.0
SUM_TOLERANCE = 0.01
#: A model tier's answers that were unsure: only these escalate to the next tier.
UNSURE = frozenset({"below_bar", "excluded", "high_stakes"})

Deliberate = Callable[[], Awaitable[Any]]


@dataclass
class Decision:
    """What a point decided: the label (or None), whether it was applied, the tier and provider that answered, the
    answering tier's state, the calibrated probability, a reason code and the decision's id. ``detail`` is the
    deliberate call's own answer (a mapping it returned), for the caller; it is never logged.

    ``withheld``: a label the last live model tier asked read at or above its bar and withholds (``Tier.withholds``:
    one its point excludes, such as ``ask_answer``'s ``approve``). It is never applied; a caller may only put a
    question to the owner about it. ``fingerprint``: the calibration fingerprint that tier (or the settling one) answered under."""

    id: str
    point: str
    label: Optional[str]
    applied: bool
    tier: str
    provider: str = ""
    state: str = "live"
    probability: Optional[float] = None
    probabilities: Mapping[str, float] = field(default_factory=dict)
    reason: str = ""
    detail: Any = None
    rules: Optional[RulesAnswer] = None
    withheld: Optional[str] = None
    fingerprint: str = ""


def _utc_now() -> datetime:
    return now_utc()


def _validated(point: Point, raw: Any) -> Dict[str, float]:
    """A provider's raw answer for ``point``: its labels exactly, each a probability, together a distribution."""
    if not isinstance(raw, Mapping) or set(raw) != set(point.labels):
        raise ValueError("not an answer over the point's labels")
    values: Dict[str, float] = {}
    for label, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                or not 0.0 <= float(value) <= 1.0:
            raise ValueError("a probability is out of range")
        values[str(label)] = float(value)
    if not math.isclose(sum(values.values()), 1.0, abs_tol=SUM_TOLERANCE):
        raise ValueError("the probabilities are not a distribution")
    return values


def _rounded(values: Mapping[str, float]) -> Dict[str, float]:
    return {label: round(float(value), 6) for label, value in values.items()}


def _deliberate_answer(answer: Any) -> Tuple[Optional[str], Any]:
    if isinstance(answer, str):
        return answer, None
    if isinstance(answer, Mapping):
        label = answer.get("label", answer.get("choice"))
        return (label if isinstance(label, str) else None), dict(answer)
    return None, None


class Instinct:
    """The one decision mechanism of a process: providers by name, the tier each serves, the points' configured
    bars, the log and the records."""

    def __init__(self, *, providers: Optional[Mapping[str, Any]] = None, fast: str = "", reader: str = "",
                 points_config: Optional[Mapping[str, Any]] = None, store: Optional[Store] = None,
                 records: Optional[Sequence[Record]] = None, registry: Optional[Mapping[str, Point]] = None,
                 monotonic: Callable[[], float] = time.monotonic, clock: Optional[Callable[[], datetime]] = None,
                 max_shadow: int = MAX_SHADOW) -> None:
        self.registry = REGISTRY if registry is None else registry
        self.providers: Dict[str, Any] = dict(providers or {})
        self.fast, self.reader = str(fast or ""), str(reader or "")
        self.points_config = settings.points_section(points_config, self.registry)
        self.store = store
        self.shipped_records: Tuple[Record, ...] = tuple(
            shipped(self.registry) if records is None else records)
        self.monotonic = monotonic
        self.clock = clock or _utc_now
        self.max_shadow = max_shadow
        self._breakers: Dict[str, Breaker] = {name: Breaker(monotonic=monotonic) for name in self.providers}
        self._budget: Dict[str, Deque[float]] = defaultdict(deque)
        self._memory_states: Dict[Tuple[str, str], str] = {}
        self._cached: Optional[Tuple[float, Dict[Tuple[str, str], str], List[Record]]] = None
        self._cached_generation: Optional[bytes] = None
        self._tasks: Set[asyncio.Task] = set()
        self._writers: Set[asyncio.Task] = set()
        # The points with a comparison running in the background (on the owner's turn: at most one each).
        self._comparing: Set[str] = set()
        self._secret = os.urandom(32) if store is None else None
        self._counts: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    # -- construction ---------------------------------------------------------------------------------------------

    @classmethod
    def from_settings(cls, section: Any, *, state_dir: Optional[Path] = None, **kwargs: Any) -> "Instinct":
        """An engine from an ``instinct`` section, read leniently (``colonymind.yaml`` validates it strictly): an
        unusable provider or point entry is left out, so every point keeps its existing path. A decision server
        needs its url; a reader (``logits``) its url and the model it serves, or its tier stays off."""
        from .providers import local_decision, logits
        parsed = settings.parse(section)
        providers: Dict[str, Any] = {}
        for name, entry in parsed["providers"].items():
            if entry.get("kind") == local_decision.KIND and entry.get("url"):
                providers[name] = local_decision.LocalDecision(name, url=entry["url"],
                                                               timeout_ms=entry.get("timeout_ms"),
                                                               model=entry.get("model", ""))
            elif entry.get("kind") == logits.KIND:
                reader = logits.from_entry(name, entry)
                if reader is not None:
                    providers[name] = reader
        store = None
        if state_dir is not None:
            try:
                store = Store(Path(state_dir) / DB_FILE)
            except Exception as error:      # never a reason to stop deciding
                logger.warning("the decision log is unavailable (%s); decisions are not logged",
                               type(error).__name__)
        return cls(providers=providers, fast=parsed["fast"], reader=parsed["reader"], points_config=parsed["points"],
                   store=store, **kwargs)

    @classmethod
    def from_environment(cls, environ: Optional[Mapping[str, str]] = None, *, state_dir: Optional[Path] = None,
                         **kwargs: Any) -> "Instinct":
        """An engine from ``COLONYMIND_INSTINCT`` (the ``instinct`` section as JSON, which ``colonymind.yaml``
        exports). Unset or unreadable: no provider, every point on its existing path."""
        environ = os.environ if environ is None else environ
        raw = str(environ.get("COLONYMIND_INSTINCT") or "").strip()
        section: Any = {}
        if raw:
            try:
                section = json.loads(raw)
            except ValueError:
                logger.warning("COLONYMIND_INSTINCT is not JSON; every point keeps its existing path")
            if not isinstance(section, dict):
                logger.warning("COLONYMIND_INSTINCT is not a mapping; every point keeps its existing path")
                section = {}
        return cls.from_settings(section, state_dir=state_dir, **kwargs)

    # -- providers, states, records ---------------------------------------------------------------------------

    def _provider_name(self, tier: str) -> str:
        return self.fast if tier == "fast" else self.reader if tier == "reader" else ""

    def provider_for(self, tier: str) -> Any:
        name = self._provider_name(tier)
        return self.providers.get(name) if name else None

    def _breaker(self, provider: Any) -> Breaker:
        name = next((key for key, value in self.providers.items() if value is provider), str(provider.id))
        if name not in self._breakers:
            self._breakers[name] = Breaker(monotonic=self.monotonic)
        return self._breakers[name]

    def _stored(self) -> Tuple[Dict[Tuple[str, str], str], List[Record]]:
        if self.store is None:
            return {}, []
        now = self.monotonic()
        try:
            generation = self.store.calibration_generation()
            if (self._cached is None or generation != self._cached_generation
                    or now - self._cached[0] > OVERRIDES_TTL_S):
                self._cached = (now, self.store.states(), self.store.records())
                self._cached_generation = generation
        except Exception as error:
            logger.warning("instinct states and records unreadable (%s)", type(error).__name__)
            self._cached = (now, {}, [])
            self._cached_generation = None
        return self._cached[1], self._cached[2]

    def _overrides(self) -> Dict[Tuple[str, str], str]:
        stored, _ = self._stored()
        return {**stored, **self._memory_states}

    def state(self, point: str, tier: str) -> Optional[str]:
        return point_state.effective(self.registry[point], tier, self._overrides())

    def set_state(self, point: str, tier: str, state: str, *, source: str,
                  evidence: Optional[Mapping[str, Any]] = None) -> None:
        """Record a point's state for one tier (the upgrade, a promotion or a demotion; tests)."""
        found = self.registry[point].tier(tier)
        if found is None or found.on_path or state not in point_state.STATES:
            raise ValueError(f"{point} has no {tier} tier with a state, or {state!r} is not a state")
        if self.store is not None:
            self.store.set_state(point, tier, state, source=source, evidence=evidence)
            self._cached = None
        self._memory_states[(point, tier)] = state

    def live(self, point: str, tier: str) -> bool:
        return self.state(point, tier) == "live"

    def record_for(self, point: Point, tier: str, provider: Any, revision: str) -> Optional[Record]:
        _, local = self._stored()
        key = {"version": point.version, "tier": tier, "provider_kind": calibration_kind(provider),
               "model": str(getattr(provider, "model", "") or ""), "revision": str(revision or "")}
        return calibration.find(local, **key) or calibration.find(self.shipped_records, **key)

    def bar(self, point: Point, record: Record, label: str) -> float:
        """The higher of the code floor, the record's bar and the configured bar."""
        configured = ((self.points_config.get(point.name) or {}).get("bars") or {}).get(label, 0.0)
        return max(float(point.floors.get(label, 0.0)), float(record.bars.get(label, calibration.NEVER)),
                   float(configured))

    def _budget_take(self, point: Point) -> bool:
        window = self._budget[point.name]
        now = self.monotonic()
        while window and now - window[0] >= HOUR:
            window.popleft()
        if len(window) >= point.budget_per_hour:
            return False
        window.append(now)
        return True

    # -- the decision -------------------------------------------------------------------------------------------

    async def decide(self, point: str, fields: Optional[Mapping[str, Any]] = None,
                     refs: Optional[Mapping[str, Any]] = None, *, deliberate: Optional[Deliberate] = None,
                     event: str = "") -> Decision:
        """One decision at ``point`` (KeyError for an unknown one). Never raises for a provider, a record, the
        log or the deliberate call: whatever fails leaves the point on its existing path."""
        spec = self.registry[point]
        fields = dict(fields or {})
        row: Dict[str, Any] = {"id": f"dec-{uuid.uuid4().hex[:16]}", "at": self.clock().isoformat(),
                               "event": code(event), "point": spec.name, "version": spec.version, "tiers": {},
                               "effect": spec.effect, "refs": {}, "state_hash": None}
        shadow: List[asyncio.Task] = []
        try:
            row["refs"] = references(refs, self._hash_secret())
            decision = await self._decide(spec, fields, row, deliberate, shadow)
        except asyncio.CancelledError:
            for task in shadow:
                task.cancel()
            raise
        except Exception as error:
            logger.warning("instinct %s failed (%s); the point keeps its conservative answer", spec.name,
                           type(error).__name__)
            label = self._fallback(spec, fields, None)
            decision = Decision(row["id"], spec.name, label, label is not None, "fallback", reason="error")
        row.update(label=decision.label, applied=decision.applied, answered_by=decision.tier,
                   reason=decision.reason)
        self._finish(spec, row, shadow)
        return decision

    async def decide_many(self, event: str, requests: Sequence[Tuple[Any, ...]]) -> List[Decision]:
        """One event's questions together: ``[(point, fields, refs[, deliberate])]``, asked as one parallel burst
        (a provider that batches is asked once per burst by its own client)."""
        return list(await asyncio.gather(*(
            self.decide(item[0], item[1], item[2] if len(item) > 2 else None,
                        deliberate=item[3] if len(item) > 3 else None, event=event) for item in requests)))

    async def _decide(self, point: Point, fields: Dict[str, Any], row: Dict[str, Any],
                      deliberate: Optional[Deliberate], shadow: List[asyncio.Task]) -> Decision:
        tiers = row["tiers"]
        rules = None
        if point.rules is not None:
            try:
                rules = point.rules(fields)
            except Exception as error:
                logger.warning("instinct %s rules failed (%s)", point.name, type(error).__name__)
                rules = None
            spoke = (rules is not None and rules.speaks and rules.label is not None
                     and not self._high_stakes(point, fields, rules.label))
            tiers["rules"] = {"outcome": "spoke" if spoke else "silent",
                              **({"label": rules.label} if rules is not None and rules.label else {})}
            if spoke:
                return Decision(row["id"], point.name, rules.label, True, "rules",
                                probability=rules.probabilities.get(rules.label), probabilities=rules.probabilities,
                                reason="rules", rules=rules)
        text = point.render(fields)
        row["state_hash"] = keyed_hash(self._hash_secret(), text)
        # ``unsure``: the last live model tier asked answered below its bar (or a label it may not settle, or a high
        # stakes case), so the next tier may be asked. ``held``: a live model tier gave no answer (it failed, its
        # breaker is open, or no record matches what it serves), so nothing after it is asked, live or in shadow,
        # and the point stays on its existing path: a provider's failure never escalates.
        # ``stood_in``: a live model tier took the case (it was asked, or would have been but for its breaker or the
        # budget), so a second opinion on the prior is not asked unless the case is high stakes (``_waived``).
        unsure = held = stood_in = escalate = False
        # ``withheld``: the last live answer was a label its tier withholds, read at its bar (with the fingerprint it
        # was read under); any later live answer replaces it.
        withheld: Tuple[Optional[str], str] = (None, "")
        for name in MODEL_TIERS:
            tier = point.tier(name)
            if tier is None:
                continue
            state = self.state(point.name, name) or "shadow"
            entry: Dict[str, Any] = {"state": state}
            tiers[name] = entry
            provider = self.provider_for(name)
            if provider is None:
                entry["outcome"] = "absent"
                continue
            entry["provider"] = str(provider.id)
            if held:
                entry["outcome"] = "not_asked"
                continue
            if len(text) > int(getattr(provider, "max_state_chars", 0) or 0) \
                    or point.type not in getattr(provider, "types", ()):
                entry["outcome"] = "skipped"
                continue
            if self._breaker(provider).is_open():
                entry["outcome"] = "provider_down"
                if state == "live":
                    unsure, held, stood_in = False, True, True
                continue
            if state != "live" and len(self._tasks) >= self.max_shadow:
                entry["outcome"] = "busy"
                continue
            # A reader ask is an LLM call the existing path does not make: live or in shadow, each takes one of the
            # point's hourly escalations (my choice: the first model tier too, since its unsure answer adds a call),
            # and at budget the point keeps its existing path.
            if name == "reader":
                if not self._budget_take(point):
                    entry["outcome"] = "budget"
                    stood_in = stood_in or state == "live"
                    continue
                entry["escalation"] = True
            if state != "live":
                task = asyncio.get_running_loop().create_task(
                    self._ask(point, tier, provider, text, fields, entry))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
                shadow.append(task)
                continue
            outcome = await self._ask(point, tier, provider, text, fields, entry)
            stood_in, escalate = True, escalate or outcome == "high_stakes"
            if outcome == "settled":
                await self._compare(point, deliberate, tiers, shadow)
                return Decision(row["id"], point.name, entry["label"], True, name, provider=str(provider.id),
                                probability=entry.get("probability"),
                                probabilities=entry.get("calibrated") or {}, reason="settled", rules=rules,
                                fingerprint=str(entry.get("fingerprint") or ""))
            withheld = (entry.get("label"), str(entry.get("fingerprint") or "")) \
                if outcome == "excluded" and entry.get("label") in tier.withholds else (None, "")
            if outcome in UNSURE:
                unsure = True
            else:                                  # failed, or no record for what the provider serves
                unsure, held = False, True
        tier = point.tier("deliberate")
        if tier is not None and stood_in and not escalate and self._waived(point, tier, fields, rules):
            tiers["deliberate"] = {"state": "live", "outcome": "not_asked"}
        elif tier is not None:
            decided = await self._deliberate(point, tier, deliberate, unsure, tiers)
            if decided is not None:
                label, detail = decided
                return Decision(row["id"], point.name, label, True, "deliberate", reason="deliberate",
                                detail=detail, rules=rules, withheld=withheld[0], fingerprint=withheld[1])
        label = self._fallback(point, fields, rules)
        reason = next((entry.get("outcome") for entry in tiers.values()
                       if entry.get("outcome") in {"provider_down", "failed", "budget"}), "no_answer")
        return Decision(row["id"], point.name, label, label is not None, "fallback", reason=f"fallback_{reason}",
                        probabilities=rules.probabilities if rules is not None else {}, rules=rules,
                        withheld=withheld[0], fingerprint=withheld[1])

    async def _ask(self, point: Point, tier: Tier, provider: Any, text: str, fields: Mapping[str, Any],
                   entry: Dict[str, Any]) -> str:
        """Ask one model tier; fills ``entry`` and returns its outcome. Every answer, matched by a record or not,
        is logged with the calibration identity it was read under: the provider's kind, its configured model, the
        revision it served and the fingerprint of that key, so raw probabilities from two checkpoints behind one
        model name are never mixed."""
        breaker = self._breaker(provider)
        kind, model = calibration_kind(provider), str(getattr(provider, "model", "") or "")
        entry.update(kind=identifier(kind, self._hash_secret()), model=identifier(model, self._hash_secret()))
        limit_ms = min(int(getattr(provider, "timeout_ms", 250) or 250), tier.deadline_ms or 10 ** 9)
        started = time.perf_counter()
        try:
            # wait_for can consume an external cancellation if its child completes in
            # the same loop turn (notably on 3.11), then enter an unbounded compare.
            # Keep the provider in the caller's task so cancellation cannot be lost.
            async with asyncio.timeout(limit_ms / 1000.0):
                answer = await provider.answer(text, {point.name: point.question()})
            raw = _validated(point, answer.probabilities[point.name])
        except asyncio.CancelledError:
            raise
        except Exception as error:
            breaker.failed()
            entry.update(outcome="failed", latency_ms=round((time.perf_counter() - started) * 1000, 1),
                         error=type(error).__name__)
            return "failed"
        breaker.succeeded()
        entry.update(latency_ms=round((time.perf_counter() - started) * 1000, 1), raw=_rounded(raw))
        revision = str(getattr(answer, "revision", "") or "")
        entry.update(revision=identifier(revision, self._hash_secret()),
                     fingerprint=calibration.fingerprint(point.version, tier.name, kind, model, revision))
        record = self.record_for(point, tier.name, provider, revision)
        if record is None:
            entry["outcome"] = "no_record"
            return "no_record"
        entry["record"] = {"source": record.source, "fitted_at": record.fitted_at, "method": record.method,
                           "fingerprint": record.fingerprint}
        try:
            probabilities = calibration.calibrated(record, point.type, raw)
        except (KeyError, ValueError):
            entry["outcome"] = "no_record"
            return "no_record"
        settled, candidate, probability = calibration.reading(
            point.type, probabilities, lambda label: self.bar(point, record, label))
        entry.update(calibrated=_rounded(probabilities), label=candidate, probability=round(probability, 6))
        if settled is None:
            outcome = "below_bar"
        elif settled not in tier.may_settle:
            outcome = "excluded"
        elif self._high_stakes(point, fields, settled):
            outcome = "high_stakes"
        else:
            outcome = "settled"
        entry["outcome"] = outcome
        return outcome

    async def _deliberate(self, point: Point, tier: Tier, deliberate: Optional[Deliberate], unsure: bool,
                          tiers: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
        entry: Dict[str, Any] = {"state": "live" if tier.on_path else (self.state(point.name, "deliberate")
                                                                        or "shadow")}
        tiers["deliberate"] = entry
        if deliberate is None:
            entry["outcome"] = "absent"
            return None
        if not tier.on_path:
            if not unsure or entry["state"] != "live":
                entry["outcome"] = "not_asked"
                return None
            if not self._budget_take(point):
                entry["outcome"] = "budget"
                return None
            entry["escalation"] = True
        started = time.perf_counter()
        try:
            answer = await deliberate()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.info("instinct %s: the deliberate call failed (%s)", point.name, type(error).__name__)
            entry.update(outcome="failed", latency_ms=round((time.perf_counter() - started) * 1000, 1))
            return None
        entry["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        label, detail = _deliberate_answer(answer)
        if label is None or label not in tier.may_settle:
            entry["outcome"] = "no_answer"
            return None
        entry.update(outcome="answered", label=label)
        return label, detail

    async def _compare(self, point: Point, deliberate: Optional[Deliberate], tiers: Dict[str, Any],
                       shadow: List[asyncio.Task]) -> None:
        """A live model tier settled where the existing path asks a model (the point's deliberate tier is the path):
        ask that call beside the settle, only to compare, within the point's hourly budget; its answer is never
        applied. Off the owner's turn it is awaited here, inside the caller's run and its budget; on the owner's
        turn it runs in the background, one at a time per point, within the shadow asks' limit (module docstring,
        Compared)."""
        tier = point.tier("deliberate")
        if tier is None or not tier.on_path or deliberate is None:
            return
        entry: Dict[str, Any] = {"state": "live"}
        tiers["deliberate"] = entry
        background = point.turn_path
        if background and (point.name in self._comparing or len(self._tasks) >= self.max_shadow):
            entry["outcome"] = "busy"
            return
        if not self._budget_take(point):
            entry["outcome"] = "budget"
            return
        if not background:
            await self._compared(point, tier, deliberate, entry)
            return
        task = asyncio.get_running_loop().create_task(self._compared(point, tier, deliberate, entry))
        self._comparing.add(point.name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(lambda _: self._comparing.discard(point.name))
        shadow.append(task)

    async def _compared(self, point: Point, tier: Tier, deliberate: Deliberate, entry: Dict[str, Any]) -> None:
        entry["outcome"] = "cancelled"          # until it answers: an engine closed meanwhile leaves it so
        started = time.perf_counter()
        try:
            answer = await deliberate()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.info("instinct %s: the call compared with a settle failed (%s)", point.name, type(error).__name__)
            entry.update(outcome="failed", latency_ms=round((time.perf_counter() - started) * 1000, 1))
            return
        entry["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        label, _ = _deliberate_answer(answer)
        if label is None or label not in tier.may_settle:
            entry["outcome"] = "no_answer"
            return
        entry.update(outcome=COMPARED, label=label)

    def _waived(self, point: Point, tier: Tier, fields: Mapping[str, Any], rules: Optional[RulesAnswer]) -> bool:
        """Whether a live model tier that took the case stands in for the deliberate tier: that tier is the
        existing path's second opinion on a rules tier's prior (it read a label but did not speak), and the prior's
        label is not high stakes. The point then keeps the prior's reading (its fallback)."""
        return (tier.on_path and point.rules is not None and rules is not None and rules.label is not None
                and not self._high_stakes(point, fields, rules.label))

    @staticmethod
    def _high_stakes(point: Point, fields: Mapping[str, Any], label: Optional[str]) -> bool:
        if point.high_stakes is None:
            return False
        try:
            return bool(point.high_stakes(fields, label))
        except Exception:
            return True           # a check that cannot run never lets a model settle

    @staticmethod
    def _fallback(point: Point, fields: Mapping[str, Any], rules: Optional[RulesAnswer]) -> Optional[str]:
        if point.fallback is None:
            return None
        try:
            label = point.fallback(fields, rules)
        except Exception as error:
            logger.warning("instinct %s fallback failed (%s)", point.name, type(error).__name__)
            return None
        return label if label in point.labels else None

    # -- the log -------------------------------------------------------------------------------------------------

    def _hash_secret(self) -> bytes:
        """The instance's secret for the log's keyed hashes. One that cannot be read, or a file that holds no whole
        key (``colonymind.instance.instance_secret`` refuses it), gives a key of this process alone: the decision is
        still made and logged, its hashes matching nothing later, and never keyed with a key anyone could guess."""
        if self.store is not None:
            try:
                return self.store.secret()
            except (OSError, ValueError) as error:
                logger.warning("the instance secret is unavailable (%s)", type(error).__name__)
                if self._secret is None:
                    self._secret = os.urandom(32)
        return self._secret or b""

    def _finish(self, point: Point, row: Dict[str, Any], shadow: List[asyncio.Task]) -> None:
        if not shadow:
            self._emit(point, row)
            return

        async def written() -> None:
            await asyncio.gather(*shadow, return_exceptions=True)
            self._emit(point, row)

        writer = asyncio.get_running_loop().create_task(written())
        self._writers.add(writer)
        writer.add_done_callback(self._writers.discard)

    def _emit(self, point: Point, row: Dict[str, Any]) -> None:
        counts = self._counts[point.name]
        for tier, entry in row["tiers"].items():
            counts[tier][str(entry.get("outcome") or "unknown")] += 1
        counts["decided"][str(row.get("answered_by") or "")] += 1
        if self.store is None:
            return
        try:
            self.store.write(row)
        except Exception as error:
            logger.warning("instinct decision not logged (%s)", type(error).__name__)

    async def drain(self) -> None:
        """Wait for the shadow asks in flight and their log rows (tests, and a clean stop)."""
        while self._tasks or self._writers:
            await asyncio.gather(*list(self._tasks), *list(self._writers), return_exceptions=True)

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await self.drain()

    # -- health ------------------------------------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Each provider's breaker and the tier it serves; per point its states and counts since the start."""
        providers = {}
        for name, provider in self.providers.items():
            breaker = self._breakers.setdefault(name, Breaker(monotonic=self.monotonic))
            providers[name] = {"kind": str(getattr(provider, "kind", "")),
                               "tier": "fast" if name == self.fast else "reader" if name == self.reader else "",
                               "breaker": breaker.state, "failures": breaker.consecutive}
        overrides = self._overrides()
        points = {}
        for name, spec in self.registry.items():
            counts = self._counts.get(name) or {}
            points[name] = {"states": point_state.states_of(spec, overrides),
                            "counts": {tier: dict(counter) for tier, counter in counts.items()}}
        return {"providers": providers, "fast": self.fast, "reader": self.reader, "points": points}


__all__ = ["Decision", "Instinct", "MAX_SHADOW"]
