"""``colonymind report instinct``: what each decision point did over a window, per tier, in plain words.

Read from the instance's ``instinct.db`` (read-only): per point and tier its state, how often it answered, abstained
(answered below its bar, with no matching calibration record, or with a label it may not settle), failed and was
skipped (no provider, a state too long, the provider down, the budget or too many shadow asks at once), the
escalations beyond the existing path and the budget hits, its latency (p50 and p95) and the calibration record in
force: the one its latest answer was read under, or none when that answer matched no record.

And what it learned (docs/DECISIONS.md, Calibration and learning): per model tier its shadow statistics (coverage:
the share of its answers a live tier would have settled; agreement with the deciding path; the disagreements and,
where an outcome came, who was right; for a live tier under a learned revision, drift's own reading beside them:
the disagreements counted against it and those no explicit outcome labels, which are unmeasured; the same split
between states seen in the fit sets and new ones), per point
its outcomes by source, the share of its decisions that re-render from their references (and, for a point without a
loader, its reason code), the state changes of the window (promotions, carries by lineage, demotions) and its last
one, and the open audits. Ids and counts only: the log holds no text.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from colonymind.util.temporal import now_utc

from .text import Document, plural

ANSWERED = {"settled", "below_bar", "excluded", "high_stakes", "no_record", "answered", "no_answer"}
ABSTAINED = {"below_bar", "excluded", "high_stakes", "no_record", "no_answer"}
SKIPPED = {"skipped", "absent", "provider_down", "budget", "busy", "not_asked"}


@dataclass
class TierCounts:
    state: str = ""
    outcomes: Dict[str, int] = field(default_factory=dict)
    latencies: List[float] = field(default_factory=list)
    escalations: int = 0
    record: Optional[Dict[str, Any]] = None
    shadow: Optional[Dict[str, Any]] = None


@dataclass
class Collected:
    span: str
    end: datetime
    found: bool
    decisions: int
    points: Dict[str, Dict[str, TierCounts]]
    decided: Dict[str, Dict[str, int]]
    graded: Dict[str, Dict[str, int]] = field(default_factory=dict)
    changes: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    audits: List[Dict[str, Any]] = field(default_factory=list)
    dropped: Dict[str, int] = field(default_factory=dict)
    rerendered: Dict[str, Dict[str, int]] = field(default_factory=dict)
    start: Optional[datetime] = None
    carried: int = 0


def shadow_statistics(rows: List[Dict[str, Any]], tier: str, best: Dict[str, Any], seen: set, *,
                      learned: Optional[Callable[[str], bool]] = None, proof: Optional[Mapping[str, Any]] = None,
                      registry: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """A model tier's statistics over ``rows`` (oldest first): coverage, agreement with the deciding path, the
    disagreements with their outcomes, and the same agreement for states seen in the fit sets and new ones. Beside
    them, what drift counts (``state.agreement``): ``against``, the disagreements counted against the tier, which is
    every one except under a learned revision (``learned``, by fingerprint; ``learned`` counts its disagreements),
    where only one an explicit outcome (``proof``) labels wrong counts and one none says anything of is
    ``unlabelled``, unmeasured for drift (``state.graded``). A disagreement is one whatever drift makes of it."""
    from colonymind.instinct import REGISTRY
    from colonymind.instinct.fit import existing_answer
    from colonymind.instinct.state import ANSWERED, acted, graded
    table = REGISTRY if registry is None else registry
    answered = settled = disagree = model_right = existing_right = with_outcome = unmeasured = 0
    against = learned_disagree = unlabelled = 0
    split = {"seen": [0, 0], "new": [0, 0]}                    # [answers, agreements]
    for row in rows:
        entry = (row.get("tiers") or {}).get(tier) or {}
        if entry.get("outcome") not in ANSWERED:
            continue
        answered += 1
        settled += entry.get("outcome") == "settled"
        # What it acted on: a settled label, or a withheld one read at its bar (the caller asked about or approved
        # on it), measured as drift and a promotion measure it (``state.acted``).
        spec = table.get(str(row.get("point") or ""))
        label = acted(spec, tier, entry)
        existing = existing_answer(row)
        if label is not None and existing is None:
            unmeasured += 1                   # nothing says what the existing path would have answered
            continue
        agrees = label is None or label == existing
        part = split["seen" if row.get("state_hash") in seen else "new"]
        part[0] += 1
        part[1] += agrees
        if agrees:
            continue
        disagree += 1
        if learned is not None and learned(str(entry.get("fingerprint") or "")):
            learned_disagree += 1
            labelled = (proof or {}).get(row["id"])
            said = graded(spec, label, labelled) if labelled is not None else None
            unlabelled += said is None        # nothing labels it: unmeasured for drift, never against the tier
            against += said == "wrong"
        else:
            against += 1
        outcome = best.get(row["id"])
        gold = outcome.get("label") if outcome else None
        if gold is not None:
            with_outcome += 1
            model_right += label == gold
            existing_right += existing == gold
    return {"answered": answered, "settled": settled, "disagree": disagree, "with_outcome": with_outcome,
            "model_right": model_right, "existing_right": existing_right, "split": split, "unmeasured": unmeasured,
            "against": against, "learned": learned_disagree, "unlabelled": unlabelled}


def collect(home: Path, *, since: str = "7d", registry: Optional[Mapping[str, Any]] = None) -> Collected:
    from colonymind.instinct import REGISTRY
    from colonymind.instinct import state as point_state
    from colonymind.instinct.log import DB_FILE, Store

    from .faculties import window
    now = now_utc()
    start, end, span = window(since, now=now)
    path = Path(home) / DB_FILE
    store = Store(path, create=False) if path.is_file() else None
    rows = store.rows(since=start) if store is not None else []
    overrides = store.states() if store is not None else {}
    points: Dict[str, Dict[str, TierCounts]] = {}
    decided: Dict[str, Dict[str, int]] = {}
    table = REGISTRY if registry is None else registry
    for name, point in table.items():
        points[name] = {tier.name: TierCounts(state="existing path" if tier.on_path else
                                              point_state.effective(point, tier.name, overrides) or "")
                        for tier in point.tiers}
    ordered = list(reversed(rows))        # oldest first, so the record in force is the latest one seen
    for row in ordered:
        tiers = points.setdefault(row["point"], {})
        by = decided.setdefault(row["point"], {})
        by[row["answered_by"]] = by.get(row["answered_by"], 0) + 1
        for tier, entry in row["tiers"].items():
            if tier == "rules":
                continue
            counts = tiers.setdefault(tier, TierCounts(state=str(entry.get("state") or "")))
            outcome = str(entry.get("outcome") or "unknown")
            counts.outcomes[outcome] = counts.outcomes.get(outcome, 0) + 1
            if isinstance(entry.get("latency_ms"), (int, float)):
                counts.latencies.append(float(entry["latency_ms"]))
            if entry.get("escalation"):
                counts.escalations += 1
            # An answer sets the record in force: the one it was read under, or none (a checkpoint no record
            # matches). A call that got no answer says nothing about what the provider serves.
            if "raw" in entry:
                counts.record = entry.get("record") or None
    graded: Dict[str, Dict[str, int]] = {}
    changes: Dict[str, List[Dict[str, Any]]] = {}
    audits: List[Dict[str, Any]] = []
    dropped: Dict[str, int] = {}
    rerendered: Dict[str, Dict[str, int]] = {}
    if store is not None:
        from colonymind.instinct.fit import examples_from_log, strongest
        every = store.outcomes()
        best = strongest(every)
        proof = strongest(item for item in every if item["source"] in point_state.PROOF)
        learned = point_state.learned_revisions(store)
        seen = {example.state_hash for example in [*store.examples(), *examples_from_log(store, registry=table)]
                if example.state_hash}
        by_point: Dict[str, List[Dict[str, Any]]] = {}
        for row in ordered:
            by_point.setdefault(row["point"], []).append(row)
        for name, tiers in points.items():
            for tier, counts in tiers.items():
                if tier in {"fast", "reader"}:
                    # Drift's count for a live tier; a tier in shadow is read as a promotion would read it.
                    counts.shadow = shadow_statistics(by_point.get(name, []), tier, best, seen, proof=proof,
                                                      learned=learned if counts.state == "live" else None,
                                                      registry=table)
        window_ids = {row["id"] for row in rows}
        for outcome in best.values():
            if outcome["decision"] in window_ids:
                entry = graded.setdefault(outcome["point"], {})
                entry[outcome["verdict"]] = entry.get(outcome["verdict"], 0) + 1
                entry[outcome["source"]] = entry.get(outcome["source"], 0) + 1
        for change in store.transitions(limit=None):
            changes.setdefault(change["point"], []).append(change)
        audits = store.audits()
        from colonymind.instinct.outcomes import Sources, replay_share
        rerendered = {name: dict(counts) for name, counts in
                      replay_share(store, Sources(home), since=start, registry=table).items()}
        for name, tiers in store.kept_counts().items():
            for reason, count in (tiers.get("log") or {}).items():
                dropped[reason] = dropped.get(reason, 0) + count
    return Collected(span=span, end=end, found=store is not None, decisions=len(rows), points=points,
                     decided=decided, graded=graded, changes=changes, audits=audits, dropped=dropped,
                     rerendered=rerendered, start=start,
                     carried=sum(change["source"].startswith(point_state.CARRIED) for entries in changes.values()
                                 for change in entries if start is None or change["at"] >= start.isoformat()))


def _percentile(values: List[float], share: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    return statistics.quantiles(ordered, n=100, method="inclusive")[int(share * 100) - 1]


def _line(tier: str, counts: TierCounts) -> str:
    outcomes = counts.outcomes
    total = lambda names: sum(outcomes.get(name, 0) for name in names)  # noqa: E731
    parts = [f"{total(ANSWERED)} answered", f"{total(ABSTAINED)} abstained", f"{outcomes.get('failed', 0)} failed",
             f"{total(SKIPPED)} skipped"]
    if tier in {"reader", "deliberate"}:
        parts.append(f"{plural(counts.escalations, 'escalation')} beyond the existing path")
    parts.append(f"{plural(outcomes.get('budget', 0), 'budget hit')}")
    text = f"{tier} ({counts.state}): " + ", ".join(parts)
    if counts.latencies:
        text += (f"; latency p50 {_percentile(counts.latencies, 0.5):.0f} ms, "
                 f"p95 {_percentile(counts.latencies, 0.95):.0f} ms")
    if tier in {"fast", "reader"}:
        record = counts.record
        if record:
            text += f"; record in force: {record.get('source') or 'unnamed'} ({record.get('method') or 'temperature'})"
        elif total(ANSWERED):
            text += "; record in force: none matched the provider, so the tier abstains"
        else:
            text += "; record in force: not seen in this window (no answer came)"
    return text


def _share(part: int, whole: int) -> str:
    return f"{round(100 * part / whole)}%" if whole else "n/a"


def _shadow_line(tier: str, counts: TierCounts) -> Optional[str]:
    stats = counts.shadow
    if not stats or not stats["answered"]:
        return None
    n = stats["answered"]
    compared = n - stats.get("unmeasured", 0)
    text = (f"{tier} ({counts.state}): coverage {_share(stats['settled'], n)} ({stats['settled']} of "
            f"{plural(n, 'answer')} would have settled), agreement {_share(compared - stats['disagree'], compared)} "
            f"with the deciding path; {plural(stats['disagree'], 'disagreement')}")
    if stats["disagree"]:
        text += (f", {stats['with_outcome']} with an outcome: the model right in {stats['model_right']}, the "
                 f"existing path in {stats['existing_right']}")
    seen, new = stats["split"]["seen"], stats["split"]["new"]
    text += (f"; states seen in the fit sets: {seen[0]} (agreement {_share(seen[1], seen[0])}), new: {new[0]} "
             f"(agreement {_share(new[1], new[0])})")
    if stats.get("unmeasured", 0):
        text += (f"; {stats['unmeasured']} settled with nothing to compare them with (the existing path was not "
                 "asked: not counted as agreement, drift unmeasured on them)")
    if stats.get("learned", 0):
        # Drift's own reading, beside the agreement: under a learned revision only a disagreement an explicit
        # outcome labels wrong counts against the tier, and one nothing labels is unmeasured.
        unlabelled = stats.get("unlabelled", 0)
        measured = compared - unlabelled
        text += (f"; drift under a learned revision counts a disagreement only when an explicit outcome labels the "
                 f"tier's answer wrong: {stats['against']} counted against it (drift agreement "
                 f"{_share(measured - stats['against'], measured)})")
        if unlabelled:
            text += (f", {plural(unlabelled, 'disagreement')} under a learned revision with no outcome to say who "
                     "was right (unmeasured, never against it)")
    return text


_RERENDER_WORDS = {"changed": "changed since", "missing": "their source gone", "version": "asked in another wording",
                   "no_state": "decided before any state was read"}


def _rerender_line(name: str, counts: Dict[str, int]) -> Optional[str]:
    """The share of a point's decisions that re-render from their references (what learning by reference can use);
    for a point without a loader, its reason code (``outcomes.NO_LOADER``), whether or not it decided anything."""
    from colonymind.instinct.outcomes import NO_LOADER_REASONS, no_loader_reason
    total = sum(counts.values())
    reason = no_loader_reason(name)
    if reason is not None:
        return (f"Its references do not locate its state (no loader, {reason}: {NO_LOADER_REASONS[reason]}): "
                + (f"none of {total} re-render" if total else "none of its decisions re-render")
                + " (learning can use none of them)")
    if not total:
        return None
    others = [f"{count} {_RERENDER_WORDS.get(kind, kind)}" for kind, count in sorted(counts.items())
              if kind not in {"matched", "no_loader"} and count]
    return (f"Re-rendered from its references: {counts.get('matched', 0)} of {total} match their keyed hash"
            + (f" ({', '.join(others)})" if others else ""))


def _changes_line(changes: List[Dict[str, Any]], start: Optional[datetime]) -> Optional[str]:
    """The window's state changes of one point: promotions, carries by lineage (a learned revision kept the tier's
    state) and demotions."""
    from colonymind.instinct.state import CARRIED
    since = start.isoformat() if start is not None else ""
    counted = {"promoted": 0, "carried": 0, "demoted": 0}
    for change in changes:
        if str(change.get("at") or "") < since:
            continue
        if str(change.get("source") or "").startswith(CARRIED):
            counted["carried"] += 1
        elif change.get("to_state") == "live" and str(change.get("source") or "").startswith("promoted"):
            counted["promoted"] += 1
        elif change.get("to_state") == "demoted":
            counted["demoted"] += 1
    if not any(counted.values()):
        return None
    return (f"State changes: {plural(counted['promoted'], 'promotion')}, {counted['carried']} carried by lineage (a "
            f"learned revision kept the tier's state), {plural(counted['demoted'], 'demotion')}")


def document(collected: Collected) -> Document:
    doc = Document("Instinct report")
    intro = [f"This covers {collected.span} to {collected.end.strftime('%Y-%m-%d %H:%M UTC')}:"
             f" {plural(collected.decisions, 'decision')} logged."]
    if not collected.found:
        intro.append("No decision log was found in this instance, so every count is zero.")
    intro.append("A tier in shadow is asked and logged, never applied; a live one decides where it settles; the "
                 "existing path is the typed call a point made before, moved unchanged.")
    intro.append(f"{collected.carried} carried by lineage in this window.")
    if collected.dropped:
        intro.append("Counted but no longer listed: " + ", ".join(
            f"{count} {'with an erased source' if reason == 'erased' else reason}"
            for reason, count in sorted(collected.dropped.items())) + ".")
    if collected.audits:
        intro.append(f"{plural(len(collected.audits), 'open audit')}: " + "; ".join(
            f"{audit['point']} {audit['tier']}: {audit['reason']}" for audit in collected.audits) + ".")
    doc.paragraph(" ".join(intro))
    for name, tiers in collected.points.items():
        doc.heading(name)
        by = collected.decided.get(name) or {}
        if by:
            doc.paragraph("Decided by " + ", ".join(f"{tier} {count}" for tier, count in sorted(by.items())) + ".")
        doc.bullets([_line(tier, counts) for tier, counts in tiers.items()]
                    or ["rules and the fallback only: no model tier"])
        learned = [line for line in (_shadow_line(tier, counts) for tier, counts in tiers.items()) if line]
        graded = collected.graded.get(name) or {}
        if graded:
            sources = [f"{graded[source]} {source}" for source in ("explicit", "behavioural", "synthetic",
                                                                     "interpreted") if graded.get(source)]
            learned.append(f"Outcomes: {graded.get('right', 0)} right, {graded.get('wrong', 0)} wrong"
                           + (f" ({', '.join(sources)})" if sources else ""))
        rerendered = _rerender_line(name, collected.rerendered.get(name) or {})
        if rerendered:
            learned.append(rerendered)
        window = _changes_line(collected.changes.get(name) or [], collected.start)
        if window:
            learned.append(window)
        change = (collected.changes.get(name) or [None])[0]
        if change is not None:
            learned.append(f"Last state change: {change['tier']} {change['from_state'] or 'shipped'} to "
                           f"{change['to_state']} at {change['at'][:16]} ({change['source']})")
        if learned:
            doc.bullets(learned)
    return doc


__all__ = ["Collected", "collect", "document"]
