"""Composition F: Instinct's own tiers over the answers the arms recorded, as the sidecar would run them.

For each item, first the caller's own reader decides what it decides today (``caller``: the phrase tables, the
owner's-instruction reader, the report reader, the answer reader's tables; the owner gate's prior is the point's
own rules tier); only where the caller would ask does the item reach ``Instinct.decide``, the sidecar's runner,
with the point's tiers as F composes them:

- **fast** (arm C's recorded answers) only on a literal point;
- **reader** (arm E's recorded answers);
- **deliberate** (arm B's recorded label): the existing path where the point's model call is the typed call the
  mind makes today (asked whenever it is reached), otherwise a tier asked only on an unsure answer;
- the **fallback**: what the caller keeps when nothing settled (its reader's conservative answer, the path that
  runs, the result passed through unchanged).

So every rule of the runner applies unchanged: rules are final where they speak, a state longer than a tier reads
is skipped and never cut, a model tier settles only a label its point allows (the exclusions: an ask's approve, a
send the gate did not choose, a resume, a retraction), a high-stakes case escalates, and a provider's failure
never escalates. The bars are the higher of the point's code floor and a bar fitted on the training folds (the
temperature by least negative log-likelihood, the bar as the most accurate composition, ties to the bar that acts
least), read on the held-out fold. The fit reads each training case as the engine decides it (the engine itself,
run once with every bar at "never"): a case the rules settle counts the same whatever the bars, and a tier is
calibrated and scored only on the cases it is asked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis  # noqa: E402
import catalog  # noqa: E402
import questions  # noqa: E402

from colonymind.instinct import calibration  # noqa: E402
from colonymind.instinct.providers import Answer  # noqa: E402
from colonymind.instinct.registry import Point, Tier, register  # noqa: E402
from colonymind.instinct.tiers import Instinct  # noqa: E402

FIVE = ("outreach_reply", "opt_out", "no_reminders", "interest_settled", "owner_verdict")
#: Points whose deliberate tier is the typed call the mind makes today (asked whenever it is reached).
ON_PATH = frozenset({"owner_instruction", "report_substance", "question_answer", "claim_relation", "owner_send"})
FAST_CHARS = 1200
READER_CHARS = 4000
GATE_CONFIDENCE = 0.75
BARS = (0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99, calibration.NEVER)
NOT_LLM = frozenset({"caller", "rules", "fast", "reader"})


class Replay:
    """A provider that answers from one arm's recorded raw probabilities, keeping every state it was asked."""

    batch = False
    private = True
    timeout_ms = 1000

    def __init__(self, id: str, kind: str, answers: Mapping[str, Mapping[str, float]], *, max_state_chars: int,
                 model: str = "", revision: str = "recorded") -> None:
        self.id, self.kind, self.model, self.revision = id, kind, model, revision
        self.answers = dict(answers)
        self.max_state_chars = max_state_chars
        self.types = frozenset({"yes_no", "choice"})
        self.calls: List[str] = []

    async def answer(self, state: str, questions_: Mapping[str, Mapping[str, Any]]) -> Answer:
        self.calls.append(state)
        [(name, _)] = list(questions_.items())
        raw = self.answers.get(state)
        if raw is None:
            raise ValueError("no recorded answer for this state")
        return Answer({name: dict(raw)}, revision=self.revision)


# -- the caller's own reader, before Instinct ---------------------------------------------------------------------

def _instruction(reading: Any) -> str:
    return "stop" if reading.stop else "pause_today" if reading.pause_today else \
        "resume" if reading.resume else "none"


ASK_VERDICTS = {"yes": "approve", "always": "approve", "all_yes": "approve", "no": "decline", "all_no": "decline",
                "later": "conditional_or_question", "unclear": "conditional_or_question", "pick":
                "conditional_or_question", "none": "unrelated"}


def existing_label(name: str, item: Mapping[str, Any]) -> Dict[str, Any]:
    """The existing path's answer where it asks no model: the answer reader's tables for an ask, and for the
    Hermes-side points stock Hermes with no gate (every cron fire runs, every admitted message is answered, every
    tool result passes unchanged). ``fired``: whether the path decided the case itself."""
    fields = item["fields"]
    if name == "ask_answer":
        from colonymind.mind import answers
        reading = answers.read(fields["words"])
        return {"label": ASK_VERDICTS.get(reading.verdict, "unrelated"), "fired": reading.verdict != "unclear",
                "ms": 0.0}
    if name.startswith("cron_") or name == "inbound_triage":
        return {"label": "yes", "fired": True, "ms": 0.0}
    if name == "tool_screen":
        return {"label": "no", "fired": True, "ms": 0.0}
    raise KeyError(name)


def caller(name: str, item: Mapping[str, Any], current: Optional[Mapping[str, Any]]) -> Tuple[bool, Optional[str]]:
    """``(asked, label)``: whether the caller would ask Instinct about ``item``; if not, the label its own reader
    settled; if so, the label it keeps when nothing settles (None: no answer)."""
    fields = item["fields"]
    if name in FIVE:
        import measure
        if current is None:
            raise KeyError(f"{name} needs the existing path's recorded answer")
        row = {**current, "current": current["label"], "fields": fields}
        return measure.eligible(name, row), current["label"]
    if name == "owner_instruction":
        from colonymind.mind import reactions
        reading = reactions.read(fields["message"])
        return bool(reading.unsure), _instruction(reading)
    if name == "report_substance":
        from datetime import datetime, timezone
        from colonymind.mind import outreach
        now = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)
        finding = outreach.Finding(id="f-1", type="research", topic=fields["topic"],
                                   slug=fields["topic"].replace(" ", "-"), summary=fields["report"], completed_at=now)
        if outreach.needs_decision(finding):
            return True, None                   # undecided: the digest's, neither sent as a finding nor discarded
        settled = outreach.settle(finding, outreach.OutreachInputs(now=now, owner_id="owner"))
        return False, "empty" if settled == outreach.EMPTY else "finding"
    if name == "ask_answer":
        from colonymind.mind import answers
        reading = answers.read(fields["words"])
        if reading.verdict == "unclear" and reading.answer_word and not reading.hard_hedge:
            return True, "conditional_or_question"          # the one plain question back
        return False, ASK_VERDICTS.get(reading.verdict, "unrelated")
    if name in ("question_answer", "claim_relation", "owner_send"):
        return True, None
    if name.startswith("cron_") or name == "inbound_triage":
        return True, "yes"                                   # fail-open: the job runs, the message is answered
    if name == "tool_screen":
        return True, "no"                                    # passed through unchanged
    raise KeyError(name)


# -- the point as F composes it -------------------------------------------------------------------------------

def _kept(fields: Mapping[str, Any], rules: Any) -> Optional[str]:
    return fields.get("_kept")


def composition(spec: Point, *, fast: bool, reader: bool, deliberate: bool) -> Point:
    """``spec`` with F's tiers, all live (the counterfactual measured), through the registry's checks."""
    allowed = tuple(label for label in spec.labels if label not in spec.excluded())
    tiers = []
    if fast and spec.literal:
        tiers.append(Tier("fast", may_settle=allowed, state="live"))
    if reader:
        tiers.append(Tier("reader", may_settle=allowed, state="live"))
    if deliberate:
        on_path = spec.name in ON_PATH
        tiers.append(Tier("deliberate", may_settle=tuple(spec.labels) if on_path else allowed, state="live",
                          on_path=on_path))
    point = Point(name=spec.name, type=spec.type, instructions=spec.instructions, labels=dict(spec.labels),
                  template=spec.template, effect=spec.effect, tiers=tuple(tiers), rules=spec.rules,
                  fallback=spec.fallback if spec.name == "owner_send" else _kept, floors=dict(spec.floors),
                  literal=spec.literal, turn_path=spec.turn_path, high_stakes=spec.high_stakes)
    register(point, registry={})
    return point


def _record(point: Point, tier: str, provider: Replay, temperature: float, bar: float) -> calibration.Record:
    return calibration.Record(point=point.name, version=point.version, tier=tier, provider_kind=provider.kind,
                              model=provider.model, revision=provider.revision, temperature=temperature,
                              bars={label: bar for label in point.labels}, source="bake-off fold")


# -- what the engine does with a case, and fitting on the training folds ----------------------------------------

def _fields(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {**row["item"]["fields"], "_kept": row["kept"], "confidence": GATE_CONFIDENCE}


def _providers(tiers: Sequence[str], rows: Sequence[Mapping[str, Any]],
               states: Mapping[str, str]) -> Dict[str, Replay]:
    providers: Dict[str, Replay] = {}
    for tier, kind, limit in (("fast", "recorded-fast", FAST_CHARS), ("reader", "recorded-reader", READER_CHARS)):
        if tier in tiers:
            providers[tier] = Replay(kind, kind, {states[row["key"]]: row[tier] for row in rows if row[tier]},
                                     max_state_chars=limit)
    return providers


def _engine(point: Point, providers: Mapping[str, Replay], records: Sequence[calibration.Record]) -> Instinct:
    table: Dict[str, Point] = {}
    register(point, registry=table)
    return Instinct(providers=dict(providers), fast="fast" if "fast" in providers else "",
                    reader="reader" if "reader" in providers else "", records=list(records), registry=table,
                    monotonic=_clock())


async def _decide(engine: Instinct, point: Point, row: Mapping[str, Any], providers: Mapping[str, Replay],
                  deliberate: bool) -> Tuple[Any, List[str]]:
    """The engine's decision on ``row``, and the model tiers it asked (the deliberate tier included)."""
    asked_deliberate: List[bool] = []

    async def recorded(row: Mapping[str, Any] = row, log: List[bool] = asked_deliberate) -> Optional[str]:
        log.append(True)
        return row["deliberate"]

    before = {tier: len(provider.calls) for tier, provider in providers.items()}
    decision = await engine.decide(point.name, _fields(row), deliberate=recorded if deliberate else None)
    asked = [tier for tier, provider in providers.items() if len(provider.calls) > before[tier]]
    return decision, asked + (["deliberate"] if asked_deliberate else [])


async def _reach(point: Point, rows: Sequence[Dict[str, Any]], tiers: Sequence[str], deliberate: bool) -> None:
    """Annotate each row with what the engine does with it when no model tier settles, read by running the
    engine itself with every model tier's bar at "never": ``rules`` (the rules tier's final label, else None: then
    no bar matters), ``asked`` (the model tiers it reaches) and ``rest`` (where it ends otherwise: the deliberate
    call's label or the fallback's). A tier's bar changes a case only where the tier is asked, and a settle stops
    the chain, so the fit reads every case exactly as the engine would decide it."""
    states = {row["key"]: point.render(row["item"]["fields"]) for row in rows}
    providers = _providers(tiers, rows, states)
    engine = _engine(point, providers, [_record(point, tier, providers[tier], 1.0, calibration.NEVER)
                                        for tier in providers])
    for row in rows:
        decision, asked = await _decide(engine, point, row, providers, deliberate)
        row["reach"] = {"rules": decision.label if decision.tier == "rules" else None, "asked": frozenset(asked),
                        "rest": decision.label if decision.applied else None}
    await engine.drain()


def _settles(point: Point, raw: Optional[Mapping[str, float]], temperature: float, bar: float,
             allowed: Sequence[str]) -> Optional[str]:
    if not raw:
        return None
    probabilities = analysis.tempered(point.type, raw, temperature)
    settled, _, _ = calibration.reading(point.type, probabilities,
                                        lambda label: max(bar, float(point.floors.get(label, 0.0))))
    return settled if settled in allowed else None           # the engine's reading, bars and exclusions


def _high_stakes(point: Point, row: Mapping[str, Any], label: str) -> bool:
    if point.high_stakes is None:
        return False
    try:
        return bool(point.high_stakes(_fields(row), label))
    except Exception:
        return True                                           # as the engine: a check that fails never settles


def _fit(point: Point, rows: Sequence[Dict[str, Any]], tiers: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Per model tier, its temperature and its bar, from the training rows as the engine decides them (``_reach``):
    a case the rules settle scores the same whatever the bars, and a tier is calibrated and scored only on the
    cases it is asked. The temperature is fitted on the tier's own answers there; the bars from the last tier to
    the first (the reader's, then the fast tier's in front of it); ties go to the bar that acts least."""
    allowed = tuple(label for label in point.labels if label not in point.excluded())
    fitted: Dict[str, Dict[str, float]] = {}
    for tier in tiers:
        reached = [{"raw": row[tier], "gold": row["gold"]} for row in rows
                   if row.get(tier) and tier in row["reach"]["asked"]]
        fitted[tier] = {"temperature": analysis.fit_temperature(point.type, reached), "bar": calibration.NEVER}

    def after(row: Dict[str, Any], upto: Sequence[str]) -> Optional[str]:
        reach = row["reach"]
        if reach["rules"] is not None:
            return reach["rules"]
        for tier in upto:
            if tier not in reach["asked"]:
                continue
            label = _settles(point, row.get(tier), fitted[tier]["temperature"], fitted[tier]["bar"], allowed)
            if label is not None and not _high_stakes(point, row, label):
                return label
        return reach["rest"]

    for index in reversed(range(len(tiers))):
        tier, later = tiers[index], tiers[index + 1:]
        best, best_score = calibration.NEVER, -1
        for bar in reversed(BARS):                  # from "never" down: a tie keeps the bar that acts least
            fitted[tier]["bar"] = bar
            score = sum(after(row, (tier, *later)) == row["gold"] for row in rows)
            if score > best_score:
                best, best_score = bar, score
        fitted[tier]["bar"] = best
    return fitted


# -- the run ------------------------------------------------------------------------------------------------------

def _clock() -> Callable[[], float]:
    """A monotonic clock an hour apart per reading: each item is decided as if alone, so an hourly budget or an
    open breaker from one item never decides another."""
    state = {"now": 0.0}

    def tick() -> float:
        state["now"] += 3600.0
        return state["now"]
    return tick


async def run(name: str, items: Sequence[Dict[str, Any]], *, current: Mapping[str, Mapping[str, Any]],
              fast: Optional[Mapping[str, Mapping[str, float]]] = None,
              reader: Optional[Mapping[str, Mapping[str, float]]] = None,
              deliberate: Optional[Mapping[str, Optional[str]]] = None,
              fixed: Optional[Tuple[float, float]] = None, folds: int = analysis.FOLDS) -> Dict[str, Any]:
    """F over ``items`` of point ``name``: ``fast``, ``reader`` (keyed raw answers) and ``deliberate`` (keyed labels)
    are the recorded arms (None: that tier is absent). With ``fixed=(temperature, bar)`` one engine decides every
    item; otherwise each held-out fold is decided on records fitted on the other folds."""
    spec = questions.spec(name)
    point = composition(spec, fast=fast is not None, reader=reader is not None, deliberate=deliberate is not None)
    tiers = [tier.name for tier in point.tiers if tier.name in ("fast", "reader")]
    rows: List[Dict[str, Any]] = []
    for item in items:
        key = catalog.key(item)
        asked, kept = caller(name, item, current.get(key))
        rows.append({"key": key, "item": item, "gold": item["gold"], "subset": item["subset"], "asked": asked,
                     "kept": kept, "group": catalog.group(item),
                     "fast": (fast or {}).get(key), "reader": (reader or {}).get(key),
                     "deliberate": (deliberate or {}).get(key)})
    asked_rows = [row for row in rows if row["asked"]]
    if fixed is None:
        await _reach(point, asked_rows, tiers, deliberate is not None)
    parts = [asked_rows] if fixed is not None else analysis.folds(asked_rows, folds)
    out: Dict[str, Dict[str, Any]] = {row["key"]: {"label": row["kept"], "tier": "caller"}
                                      for row in rows if not row["asked"]}
    calls: Dict[str, List[str]] = {"fast": [], "reader": [], "deliberate": []}
    fitted_log: List[Dict[str, Any]] = []
    for index, held in enumerate(parts):
        if fixed is not None:
            fitted = {tier: {"temperature": fixed[0], "bar": fixed[1]} for tier in tiers}
        else:
            train = [row for other, part in enumerate(parts) if other != index for row in part]
            fitted = _fit(point, train, tiers)
            fitted_log.append(fitted)
        states = {row["key"]: point.render(row["item"]["fields"]) for row in held}
        providers = _providers(tiers, held, states)
        engine = _engine(point, providers, [_record(point, tier, providers[tier], fitted[tier]["temperature"],
                                                    fitted[tier]["bar"]) for tier in providers])
        for row in held:
            decision, asked = await _decide(engine, point, row, providers, deliberate is not None)
            for tier in asked:
                calls[tier].append(row["key"])
            out[row["key"]] = {"label": decision.label if decision.applied else None, "tier": decision.tier}
        await engine.drain()
    label_of = {key: entry["label"] for key, entry in out.items()}
    settled = [key for key, entry in out.items() if entry["tier"] in NOT_LLM]
    return {"items": out, "asked": calls, "fitted": fitted_log,
            "coverage": round(len(settled) / len(rows), 4) if rows else None,
            "llm_calls": len(calls["deliberate"]),
            "accuracy": analysis.by_subset([{"key": row["key"], "gold": row["gold"], "subset": row["subset"]}
                                            for row in rows], label_of)}


__all__ = ["ON_PATH", "Replay", "caller", "composition", "existing_label", "run"]
